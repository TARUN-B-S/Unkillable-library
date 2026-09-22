"""Story ingest: turn indexed frames into chronological story events.

Reads ``IndexEntry`` records (``storage/index/entries.jsonl``) — the same
source the main search pipeline reads — and groups each clip's frames into a
single :class:`StoryEvent`.  Nothing here writes to the main index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from unkillable.story.identities import fingerprint_of, obj_area
from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class StoryEvent:
    """One camera clip: the frames indexed from it, aggregated detections."""

    event_id: str
    clip_name: str
    camera: str
    start_wall: float  # epoch seconds (first frame's wall-clock)
    end_wall: float
    entries: list = field(default_factory=list)  # IndexEntry (frames)
    objects: list = field(default_factory=list)  # aggregated detections

    @property
    def date_str(self) -> str:
        return datetime.fromtimestamp(self.start_wall).strftime("%Y-%m-%d")

    @property
    def time_str(self) -> str:
        return datetime.fromtimestamp(self.start_wall).strftime("%H:%M")

    @property
    def n_frames(self) -> int:
        return len(self.entries)


def load_entries(storage_root: Path | str, limit: int = 10**9) -> list:
    """All ``IndexEntry`` records (lazy import keeps the CLI start light)."""
    from unkillable.index.engine import IndexEngine

    ie = IndexEngine(storage_root=Path(storage_root))
    return ie.load_entries(limit=limit)


def entries_on_date(entries, target: date) -> list:
    """Chronological entries whose local wall-clock date matches ``target``."""
    day: list = [e for e in entries if e.wall_time and datetime.fromtimestamp(e.wall_time).date() == target]
    day.sort(key=lambda e: e.wall_time)
    return day


def group_events(day_entries) -> list[StoryEvent]:
    """Group one day's frames into one event per source clip, chronological."""
    by_clip: dict[str, list] = {}
    order: list[str] = []
    for e in day_entries:
        if e.source_clip not in by_clip:
            by_clip[e.source_clip] = []
            order.append(e.source_clip)
        by_clip[e.source_clip].append(e)

    events: list[StoryEvent] = []
    for clip in order:
        frames = by_clip[clip]
        events.append(
            StoryEvent(
                event_id=Path(clip).stem,
                clip_name=Path(clip).name,
                camera=frames[0].camera,
                start_wall=frames[0].wall_time,
                end_wall=frames[-1].wall_time,
                entries=frames,
                objects=_aggregate_objects(frames),
            )
        )
    return events


def _aggregate_objects(frames) -> list[dict]:
    """Distinct detections across a clip's frames (dedup by fingerprint).

    Keeps the largest instance of each (label, color, zone, size) so
    identity assignment and narration don't double-count repeated frames.
    """
    best: dict[tuple, dict] = {}
    for frame in frames:
        for obj in frame.objects or []:
            key = fingerprint_of(obj)
            if key is None:
                continue
            prev = best.get(key)
            if prev is None or obj_area(obj) > obj_area(prev):
                best[key] = obj
    return sorted(best.values(), key=obj_area, reverse=True)


def events_for_day(storage_root: Path | str, target: date, limit: int = 10**9) -> list[StoryEvent]:
    """Full pipeline: load → filter by date → group by clip."""
    entries = load_entries(storage_root, limit=limit)
    return group_events(entries_on_date(entries, target))