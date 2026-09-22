"""Two-tier narration: deterministic templates + Ollama enrichment top-N.

Tier 1 runs for every event (cheap, offline, deterministic).  Tier 2 sends
only the day's most important events to the local vision LLM (Ollama) and is
skipped entirely when the model is unreachable — the diary is never a
failure, only sometimes richer.
"""
from __future__ import annotations

from pathlib import Path

from unkillable.story.identities import display_name, obj_area, zone_of
from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)

# Importance weighting: heavier for classes a homeowner actually cares about.
LABEL_WEIGHT = {
    "person": 1.0,
    "car": 0.9, "truck": 0.9, "bus": 0.9, "motorcycle": 0.9, "bicycle": 0.9,
    "dog": 0.8, "cat": 0.8, "bird": 0.7,
}

_ENRICH_PROMPT = (
    "Security camera snapshot. In one short sentence, describe what is "
    "happening and anything notable: people, vehicles, colors, motion, or "
    "unusual details. Say 'nothing notable' if it is unremarkable."
)


class Narrator:
    """Turns story events into diary lines; enriches the top-N via Ollama."""

    def __init__(self, describer=None, top_n: int = 5, ollama_url: str = "http://localhost:11434") -> None:
        self.describer = describer
        self.top_n = top_n
        self.ollama_url = ollama_url

    # -- importance ranking ----------------------------------------------

    def importance(self, event) -> float:
        """Area × confidence × label-weight across the event's objects."""
        score = 0.0
        for obj in event.objects or []:
            area = obj_area(obj)
            conf = float(obj.get("confidence") or 0.0)
            weight = LABEL_WEIGHT.get(obj.get("label"), 0.6)
            score += area * conf * weight
        score *= 1.0 + 0.1 * event.n_frames
        return round(score, 4)

    def top_events(self, events, top_n: int | None = None) -> list:
        """The day's most important events with at least one detected object."""
        n = self.top_n if top_n is None else top_n
        if n <= 0:
            return []
        ranked = sorted(events, key=self.importance, reverse=True)
        return [e for e in ranked[:n] if self.importance(e) > 0.0]

    # -- tier 1: templates ----------------------------------------------

    def template_line(self, event, identity_names=None, anomaly_msgs=None) -> str:
        """Deterministic diary line for one event (same input → same text)."""
        identity_names = identity_names or []
        anomaly_msgs = anomaly_msgs or []

        counts: dict[str, int] = {}
        for obj in event.objects or []:
            label = obj.get("label") or "object"
            counts[label] = counts.get(label, 0) + 1
        det = ", ".join(f"{n} {label}{'s' if n > 1 else ''}" for label, n in sorted(counts.items()))

        parts = [f"At {event.time_str}, {event.camera}:"]
        if det:
            parts.append(det)

        top = max(event.objects or [], key=obj_area, default=None)
        if top is not None:
            bits = []
            color = top.get("color_name")
            size = top.get("size")
            zone = zone_of(top.get("bbox"))
            if color and color != "unknown":
                bits.append(color)
            if size in ("large", "huge"):
                bits.append(size)
            bits.append(top.get("label") or "object")
            if zone:
                bits.append(f"{zone} zone")
            parts.append("— " + " ".join(bits))

        parts.append(f"({event.n_frames} frame{'s' if event.n_frames != 1 else ''})")
        if identity_names:
            parts.append("as " + " & ".join(sorted(set(identity_names))))

        line = " ".join(parts)
        if anomaly_msgs:
            line += " ⚠ " + "; ".join(anomaly_msgs)
        return line

    # -- tier 2: Ollama enrichment --------------------------------------

    def enrich(self, top_events, ollama_url: str | None = None) -> dict[str, str]:
        """Vision-LLM descriptions for the top events: ``{event_id: text}``.

        Quietly returns {} when Ollama is unreachable or the local model
        isn't answering — templates alone are a complete diary.
        """
        if not top_events:
            return {}
        describer = self.describer
        if describer is None:
            try:
                from unkillable.genai.describer import Describer

                describer = Describer(base_url=ollama_url or self.ollama_url)
            except Exception as exc:  # pragma: no cover - config errors only
                log.warning("Describer unavailable, enrichment skipped: %s", exc)
                return {}
        try:
            healthy = bool(describer.health_check())
        except Exception as exc:
            log.warning("Ollama health check failed, enrichment skipped: %s", exc)
            return {}
        if not healthy:
            log.info("Ollama not reachable — tier-2 enrichment skipped (templates only)")
            return {}

        out: dict[str, str] = {}
        for event in top_events:
            if not event.entries:
                continue
            thumb = Path(event.entries[0].thumbnail_path)
            if not thumb.exists():
                continue
            try:
                text = describer.describe(thumb, prompt=_ENRICH_PROMPT)
                # Skip the offline fallback string; keep only real descriptions.
                if text and not text.startswith("Security camera frame from"):
                    out[event.event_id] = text
            except Exception as exc:
                log.warning("Enrichment failed for %s: %s", event.event_id, exc)
        return out


def identity_names_for_event(store_identities, event, date_str: str) -> list[str]:
    """Display names of the identities that appear in an event (all camera IDs)."""
    names: set[str] = set()
    for ident in store_identities:
        for ref in ident.events:
            if ref.startswith(f"{date_str}|{event.event_id}|"):
                names.add(display_name(ident.fingerprint))
                break
    return sorted(names)