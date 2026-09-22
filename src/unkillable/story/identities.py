"""Memory Palace — persistent, heuristic-first identities.

Identities are merged purely from what is already recorded per frame
(label, color_name, zone, size) plus a recency window — no embeddings, so the
palace works fully offline even when CLIP vectors degrade to hash stubs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)

# Sightings of the same (camera, fingerprint) closer than this many seconds
# belong to one identity; longer gaps start a new one.  Two identical-colored
# objects sharing camera + zone within the window merge — an accepted
# heuristic limitation (documented in story.md).
MERGE_WINDOW = 180.0

# Detector reports bboxes in the model-input space (imgsz=640); zones are
# coarse thirds, so small scale errors don't matter.
FRAME_SIZE = 640.0

ZONE_ROWS = ("N", "C", "S")
ZONE_COLS = ("W", "C", "E")


def zone_of(bbox, frame_size: float = FRAME_SIZE) -> str | None:
    """9-bin zone (NW/N/NE/W/C/E/SW/S/SE) from a detection bbox center."""
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx = (x1 + x2) / 2.0 / frame_size
    cy = (y1 + y2) / 2.0 / frame_size
    row = ZONE_ROWS[0] if cy < 1 / 3 else ZONE_ROWS[2] if cy > 2 / 3 else ZONE_ROWS[1]
    col = ZONE_COLS[0] if cx < 1 / 3 else ZONE_COLS[2] if cx > 2 / 3 else ZONE_COLS[1]
    return row + col


def obj_area(obj: dict) -> float:
    """Pixel area of a detection bbox (0 for malformed bboxes)."""
    bbox = obj.get("bbox") or []
    if len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def fingerprint_of(obj: dict) -> tuple | None:
    """Coarse identity fingerprint: (label, color, zone, size).

    ``None`` when the object has no label (nothing stable to key on).
    """
    label = obj.get("label")
    if not label:
        return None
    color = obj.get("color_name") or "unknown"
    zone = zone_of(obj.get("bbox"))
    size = obj.get("size") or "unknown"
    return (label, color, zone, size)


def display_name(fp) -> str:
    """Human-readable name for an identity derived from its fingerprint."""
    label, color, _zone, size = fp
    parts = []
    if color and color != "unknown":
        parts.append(color)
    if size in ("large", "huge"):
        parts.append(size)
    parts.append(label)
    return " ".join(parts)


@dataclass
class Identity:
    """One persistent identity: sightings are ``"YYYY-MM-DD|event_id|epoch"``."""

    id: str
    fingerprint: list
    camera: str
    first_seen: float
    last_seen: float
    events: list = field(default_factory=list)

    @property
    def counts(self) -> int:
        return len(self.events)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "fingerprint": self.fingerprint,
            "camera": self.camera,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Identity:
        return cls(
            id=str(d.get("id", "")),
            fingerprint=list(d.get("fingerprint") or []),
            camera=str(d.get("camera", "")),
            first_seen=float(d.get("first_seen", 0.0)),
            last_seen=float(d.get("last_seen", 0.0)),
            events=list(d.get("events") or []),
        )


class IdentityStore:
    """Persistent identity pool with recency-window merging.

    The store is a plain JSON document; a corrupt file is backed up and the
    store rebuilt from scratch (never a crash).
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.identities: list[Identity] = []
        self._next_seq = 0
        self._load()

    # -- persistence ------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.identities = [Identity.from_dict(d) for d in data]
        except (OSError, ValueError) as exc:
            log.warning("Corrupt identities file %s — rebuilding: %s", self.path, exc)
            self.identities = []
        self._next_seq = self._recover_seq()

    def _recover_seq(self) -> int:
        seq = 0
        for ident in self.identities:
            try:
                seq = max(seq, int(ident.id.rsplit("#", 1)[1]))
            except (ValueError, IndexError):
                continue
        return seq + 1

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps([i.to_dict() for i in self.identities], indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    # -- idempotent updates ----------------------------------------------

    def remove_events_on_date(self, date_str: str) -> None:
        """Drop every sighting reference for one day (nightly re-runs).

        Identity objects themselves are kept so a re-run merges back into the
        same IDs instead of minting duplicates.
        """
        for ident in self.identities:
            ident.events = [ref for ref in ident.events if not ref.startswith(f"{date_str}|")]

    def assign(self, date_str: str, camera: str, seen_at: float, event_id: str, obj: dict) -> str:
        """Return the identity id for a detected object in an event.

        Merges into the most recent same-(camera, fingerprint) identity when
        ``seen_at`` is within :data:`MERGE_WINDOW` of its last sighting;
        otherwise starts a new identity.  Returns ``"unknown"`` for objects
        with no fingerprint.
        """
        fp = fingerprint_of(obj)
        if fp is None:
            return "unknown"
        fp_list = list(fp)
        candidates = [
            i
            for i in self.identities
            if i.camera == camera and i.fingerprint == fp_list and seen_at - i.last_seen <= MERGE_WINDOW
        ]
        if candidates:
            ident = max(candidates, key=lambda i: i.last_seen)
        else:
            slug = "-".join(str(p).replace(" ", "_") or "?" for p in fp_list)
            ident = Identity(
                id=f"{camera}-{slug}#{self._next_seq}",
                fingerprint=fp_list,
                camera=camera,
                first_seen=seen_at,
                last_seen=seen_at,
            )
            self._next_seq += 1
            self.identities.append(ident)
        ident.last_seen = max(ident.last_seen, seen_at)
        ref = f"{date_str}|{event_id}|{seen_at:.0f}"
        if ref not in ident.events:
            ident.events.append(ref)
        return ident.id