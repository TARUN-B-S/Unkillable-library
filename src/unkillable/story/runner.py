"""StoryRunner — nightly orchestration of the Storyteller trio.

``nightly`` is idempotent for a given date: re-running replaces that day's
diary, rewrites the anomaly file, and re-merges identities instead of
duplicating anything.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from unkillable.story import anomalies, diary, identities, ingest, narrator
from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)


def parse_date(value: str) -> date:
    """Accept '' (today) or YYYY-MM-DD."""
    value = (value or "").strip()
    if not value:
        return date.today()
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid date {value!r} — expected YYYY-MM-DD") from exc


@dataclass
class NightlyResult:
    date_str: str
    n_events: int
    n_anomalies: int
    n_enriched: int
    n_identities: int
    diary_path: Path | None

    def summary(self) -> str:
        return (
            f"Story nightly {self.date_str}: {self.n_events} event(s), "
            f"{self.n_anomalies} anomaly flag(s), {self.n_enriched} enriched, "
            f"{self.n_identities} persistent identity(ies), "
            f"diary={self.diary_path}"
        )


class StoryRunner:
    """Runs the diary / anomaly / identity pipeline for one day."""

    def __init__(
        self,
        storage_root: Path | str = "storage",
        top_n: int = 5,
        ollama_url: str = "http://localhost:11434",
        describer=None,
    ) -> None:
        self.storage_root = Path(storage_root).resolve()
        self.story_dir = self.storage_root / "story"
        self.identities_file = self.story_dir / "identities.json"
        self.anomalies_file = self.story_dir / "anomalies.jsonl"
        self.diary_dir = self.story_dir / "diary"
        self.diary_index = self.story_dir / "diary_index.jsonl"
        self.top_n = top_n
        self.ollama_url = ollama_url
        self.store = identities.IdentityStore(self.identities_file)
        self.narrator = narrator.Narrator(
            describer=describer, top_n=top_n, ollama_url=ollama_url
        )

    def nightly(self, target: date | None = None, top_n: int | None = None) -> NightlyResult:
        """Orchestrate one day: identities → anomalies → narration → diary."""
        target = target or date.today()
        date_str = target.isoformat()
        top = self.top_n if top_n is None else top_n

        events = ingest.events_for_day(self.storage_root, target)
        entries_all = ingest.load_entries(self.storage_root)

        # 1) Memory Palace — drop this day's refs first, then re-assign, so
        #    re-running merges back into the same identity IDs.
        self.store.remove_events_on_date(date_str)
        for event in events:
            for obj in event.objects:
                if identities.fingerprint_of(obj) is None:
                    continue
                self.store.assign(date_str, event.camera, event.start_wall, event.event_id, obj)
        self.store.save()
        identities_count = len(self.store.identities)

        # 2) Anomaly Copilot — whole-file rewrite → idempotent.
        flags = anomalies.detect_anomalies(entries_all, target)
        anomalies.write(self.anomalies_file, flags)
        anomaly_by_event: dict[str, list[str]] = {}
        for event in events:
            labels = {obj.get("label") for obj in event.objects if obj.get("label")}
            hour = datetime.fromtimestamp(event.start_wall).hour
            msgs = anomalies.event_anomalies(flags, event.camera, hour, labels)
            if msgs:
                anomaly_by_event[event.event_id] = msgs

        # 3) Narration — templates for all, enrichment for the top-N.
        self.narrator.top_n = top
        descriptions = self.narrator.enrich(self.narrator.top_events(events, top))

        ordered_lines: list[tuple[str, str]] = []
        per_camera: Counter = Counter()
        for event in events:
            names = narrator.identity_names_for_event(self.store.identities, event, date_str)
            line = self.narrator.template_line(
                event, names, anomaly_by_event.get(event.event_id, [])
            )
            enriched = descriptions.get(event.event_id)
            if enriched:
                line += f" LLM: {enriched}"
            ordered_lines.append((event.event_id, line))
            per_camera[event.camera] += 1

        # 4) Diary text + searchable index (rewritten → idempotent).
        text = diary.build_diary_text(target, ordered_lines, dict(per_camera))
        diary_path = diary.write_diary(self.diary_dir, target, text)
        diary.rebuild_index(self.diary_dir, self.diary_index)

        return NightlyResult(
            date_str=date_str,
            n_events=len(events),
            n_anomalies=len(flags),
            n_enriched=len(descriptions),
            n_identities=identities_count,
            diary_path=diary_path,
        )