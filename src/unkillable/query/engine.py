import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from unkillable.detection.detector import SUPPORTED_LABELS
from unkillable.index.engine import IndexEngine
from unkillable.index.models import IndexEntry
from unkillable.utils.ffmpeg import FFmpegError, FFmpegWrapper
from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)


def parse_time(value) -> float | None:
    """Parse a time filter into epoch seconds.

    Accepts numeric epoch (float/int/str digits) or an ISO/datetime string
    such as ``2026-09-12 21:48:13`` or ``2026-09-12T21:48:13`` (interpreted
    as local time).  Returns ``None`` for empty/invalid input.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.replace(".", "", 1).isdigit():
        try:
            return float(text)
        except ValueError:
            pass
    text = text.replace(" ", "T", 1) if " " in text.strip() and "T" not in text else text
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None

# Canonical vocab used to parse a natural-language query into structure.
_COLOR_ALIASES = {
    "red": "red", "orange": "orange", "brown": "orange", "amber": "orange",
    "yellow": "yellow", "green": "green", "lime": "green", "teal": "green",
    "cyan": "cyan", "blue": "blue", "navy": "blue", "purple": "purple",
    "violet": "purple", "magenta": "purple", "pink": "purple",
    "black": "black", "white": "white", "silver": "gray", "gray": "gray", "grey": "gray",
}
_LABEL_ALIASES: dict[str, str] = {lbl: lbl for lbl in SUPPORTED_LABELS}
_LABEL_ALIASES.update({
    "people": "person", "pedestrian": "person", "pedestrians": "person",
    "cars": "car", "automobile": "car", "automobiles": "car", "vehicle": "car",
    "trucks": "truck", "lorry": "truck", "lorries": "truck", "semi": "truck",
    "buses": "bus", "coach": "bus", "motorcycles": "motorcycle", "bikes": "motorcycle",
    "bicycles": "bicycle", "cyclist": "bicycle", "cyclists": "bicycle",
    "dogs": "dog", "cats": "cat", "birds": "bird",
})
_SIZE_ALIASES = {
    "tiny": "tiny", "small": "small", "smallish": "small",
    "medium": "medium", "mid": "medium", "average": "medium",
    "big": "large", "large": "large", "bigger": "large",
    "huge": "huge", "xl": "huge", "enormous": "huge",
}


@dataclass
class ParsedQuery:
    labels: set[str] = field(default_factory=set)
    colors: set[str] = field(default_factory=set)
    sizes: set[str] = field(default_factory=set)


@dataclass
class QueryResult:
    """A single query match with frame info and optional clip path."""

    entry: IndexEntry
    score: float
    clip_path: str | None = None  # generated clip path
    description: str | None = None  # GenAI description (on-demand)
    score_breakdown: dict | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.entry.id,
            "timestamp": self.entry.timestamp,
            "timestamp_str": self.entry.timestamp_str,
            "wall_time": self.entry.wall_time,
            "wall_time_str": self.entry.wall_time_str,
            "camera": self.entry.camera,
            "source_clip": self.entry.source_clip,
            "thumbnail_path": self.entry.thumbnail_path,
            "motion_score": self.entry.motion_score,
            "labels": self.entry.labels,
            "objects": self.entry.objects,
            "counts": self.entry.counts,
            "tags": self.entry.tags,
            "colors": sorted({o.get("color_name") for o in self.entry.objects if o.get("color_name")}),
            "score": round(self.score, 4),
            "score_breakdown": self.score_breakdown,
            "clip_path": self.clip_path,
            "description": self.description,
            "metadata": self.entry.metadata,
        }


class QueryEngine:
    """Query the video index: embed query, search, generate clips."""

    def __init__(
        self,
        index_engine: IndexEngine | None = None,
        storage_root: Path | str = "storage",
        clip_duration: float = 5.0,
        ffmpeg: FFmpegWrapper | None = None,
    ) -> None:
        self.storage_root = Path(storage_root)
        self.index_engine = index_engine or IndexEngine(storage_root=self.storage_root)
        self.search = self.index_engine.search
        self.clip_duration = clip_duration
        self.ffmpeg = ffmpeg or FFmpegWrapper()
        self.clip_cache_dir = self.storage_root / "query_cache"

    def query(
        self,
        text: str,
        top_k: int = 10,
        labels: list[str] | None = None,
        generate_clips: bool = True,
        start_time: float | None = None,
        end_time: float | None = None,
        sort: str = "relevance",
    ) -> list[QueryResult]:
        """Search the index by natural language query.

        Uses a hybrid scorer: structured label/color/size matches fused with
        semantic similarity and per-frame detection quality.

        ``start_time``/``end_time`` are epoch seconds (or ISO strings, parsed
        by :func:`parse_time`) bounding the frame's absolute wall-clock time;
        frames without a ``wall_time`` are excluded when a range is active.
        ``sort`` is ``relevance`` (hybrid score), ``newest`` or ``oldest``
        (wall-clock time).
        """
        if not text or not text.strip():
            return []

        entries = self.index_engine.load_entries()
        if not entries:
            log.warning("Index is empty")
            return []

        if start_time is not None or end_time is not None:
            entries = [
                e for e in entries
                if e.wall_time is not None
                and (start_time is None or e.wall_time >= start_time)
                and (end_time is None or e.wall_time <= end_time)
            ]
            log.info("Time filter [%s, %s] kept %d entries", start_time, end_time, len(entries))

        parsed = self._parse_query(text)
        semantic_scores = self._semantic_scores(text, entries)

        # Score all entries (per-signal, pre-fusion)
        per_entry: list[tuple[QueryResult, list[float]]] = []
        for entry in entries:
            if not entry.embedding:
                continue
            result, signals = self._score(entry, parsed, semantic_scores[entry.id], labels)
            per_entry.append((result, signals))

        # Reciprocal-rank fusion across relevance signals: each signal
        # contributes 1/(k + rank) per entry, so a frame ranked high on *any*
        # signal is boosted without one dominating.  Ties share a fractional
        # (average) rank, so signals that tie across entries can't bias the
        # outcome by list order — the differing signal decides.
        # NOTE: detection quality is deliberately NOT fused here — a zero
        # detection on the semantically-best frame would cost it a bottom rank
        # that drowns its decisive relevance win.  Quality is a confidence
        # prior, applied afterwards as a multiplier that only reorders
        # near-ties (its 1% span is smaller than one full rank step).
        k = 60.0
        signals_matrix = list(zip(*(s for _, s in per_entry))) if per_entry else []
        fused: dict[int, float] = {}
        for signal_scores in signals_matrix:
            order = sorted(range(len(per_entry)), key=lambda i: signal_scores[i], reverse=True)
            pos = 0
            while pos < len(order):
                end = pos
                while (
                    end + 1 < len(order)
                    and signal_scores[order[end + 1]] == signal_scores[order[pos]]
                ):
                    end += 1
                avg_rank = (pos + end) / 2 + 1  # 1-based average rank of the tie group
                for j in range(pos, end + 1):
                    i = order[j]
                    fused[i] = fused.get(i, 0.0) + 1.0 / (k + avg_rank)
                pos = end + 1
        max_fused = max(fused.values(), default=0.0)

        # Score all entries with fused score (normalized to [0, 1])
        scored: list[tuple[QueryResult, float]] = []
        for i, (result, signals) in enumerate(per_entry):
            score = fused.get(i, 0.0) / max_fused if max_fused > 0 else 0.0
            # Quality prior: ±1% — breaks near-ties toward confident detections
            # without overturning meaningful relevance gaps.
            quality = result.score_breakdown.get("quality", 0.0) if result.score_breakdown else 0.0
            score *= 0.99 + 0.01 * quality
            # Label filter penalizes but doesn't eliminate (as before).
            if labels and not (set(labels) & set(result.entry.labels)):
                score *= 0.20
            result.score = round(score, 4)
            scored.append((result, score))

        # Sort: relevance desc, or by wall-clock time
        if sort == "newest":
            scored.sort(key=lambda x: x[0].entry.wall_time or 0.0, reverse=True)
        elif sort == "oldest":
            scored.sort(key=lambda x: x[0].entry.wall_time or 0.0)
        else:
            scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        # Generate clips on-demand
        results: list[QueryResult] = []
        for result, _score in top:
            if generate_clips:
                result.clip_path = self._generate_clip(result.entry)
            results.append(result)

        log.info("Query '%s' returned %d results", text[:50], len(results))
        return results

    @staticmethod
    def _parse_query(text: str) -> ParsedQuery:
        """Extract structured intent (labels/colors/sizes) from a query string."""
        parsed = ParsedQuery()
        for token in re.findall(r"[a-z]+", text.lower()):
            if token in _COLOR_ALIASES:
                parsed.colors.add(_COLOR_ALIASES[token])
            elif token in _LABEL_ALIASES:
                parsed.labels.add(_LABEL_ALIASES[token])
            elif token in _SIZE_ALIASES:
                parsed.sizes.add(_SIZE_ALIASES[token])
        return parsed

    def _semantic_scores(self, text: str, entries: list[IndexEntry]) -> dict[str, float]:
        """Batch cosine similarity of the query embedding vs every entry.

        Fuses image and text vectors per entry (max of the two): a frame
        matches strongly if either its image or its tags/description text is
        similar to the query.  One NumPy matrix-vector product per vector
        group replaces a per-entry Python cosine loop.  Entries whose vector
        dimension differs from the query are scored 0.0.
        """
        qvec = self.search.embed_text(text)
        qn = np.asarray(qvec, dtype=np.float32)
        qnorm = float(np.linalg.norm(qn))
        scores = {e.id: 0.0 for e in entries}
        if qnorm <= 0.0:
            return scores
        qn = qn / qnorm

        image_by_dim: dict[int, list[tuple[str, list[float], float]]] = {}
        text_by_dim: dict[int, list[tuple[str, list[float], float]]] = {}
        for e in entries:
            # Down-votes demote the text channel (description/tag matches);
            # each down-vote halves its influence on the fused score.
            text_weight = 0.5 ** int(e.metadata.get("feedback_downvotes", 0))
            if e.embedding:
                image_by_dim.setdefault(len(e.embedding), []).append((e.id, e.embedding, 1.0))
            if e.text_embedding:
                text_by_dim.setdefault(len(e.text_embedding), []).append((e.id, e.text_embedding, text_weight))

        for by_dim in (image_by_dim, text_by_dim):
            for dim, items in by_dim.items():
                if dim != len(qvec):
                    continue
                matrix = np.asarray([v for _, v, _w in items], dtype=np.float32)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                sims = (matrix / norms) @ qn
                for (eid, _v, w), s in zip(items, sims):
                    weighted = float(s) * w
                    if weighted > scores[eid]:
                        scores[eid] = weighted
        return scores

    def _score(
        self,
        entry: IndexEntry,
        parsed: ParsedQuery,
        semantic: float,
        extra_labels: list[str] | None = None,
    ) -> tuple[QueryResult, list[float]]:
        """Compute per-signal scores for one entry (fusion happens in query()).

        Signals: [structure, semantic, dominance, quality] for structured
        queries; [semantic, quality] for free-form ones.  ``structure`` is
        computed per detected object: an object only earns credit for query
        attributes it satisfies *itself*.  A frame holding a gray car and a
        purple truck therefore does NOT fully match "gray truck" — no single
        object is both gray and a truck.
        """
        objects = entry.objects or []
        entry_label_set = set(entry.labels)
        entry_colors = {o.get("color_name") for o in objects if o.get("color_name")}
        entry_sizes = {o.get("size") for o in objects if o.get("size")}

        n_labels = len(parsed.labels)
        n_colors = len(parsed.colors)
        n_sizes = len(parsed.sizes)
        has_structure = n_labels or n_colors or n_sizes

        quality = entry.detection_confidence

        if has_structure:
            structure, best_obj = self._object_cooccurrence(entry, parsed)
            dominance = self._dominance(objects, parsed)
            # Relevance signals only — quality is applied as a post-fusion
            # prior in query() (see note there).
            signals = [structure, semantic, dominance]
        else:
            # Free-form query: rely on semantic similarity alone for ranking.
            structure, best_obj, dominance = 0.0, None, 0.0
            signals = [semantic]

        # Perception chips are re-derived from the best-matching object so the
        # displayed breakdown always agrees with the ranking.
        label_match = color_match = size_match = 0.0
        if best_obj is not None:
            if n_labels:
                label_match = sum(1 for lbl in parsed.labels if best_obj.get("label") == lbl) / n_labels
            else:
                label_match = 1.0
            if n_colors:
                color_match = sum(1 for c in parsed.colors if best_obj.get("color_name") == c) / n_colors
            else:
                color_match = 1.0
            if n_sizes:
                size_match = sum(1 for s in parsed.sizes if best_obj.get("size") == s) / n_sizes
            else:
                size_match = 1.0
        elif has_structure:
            # No objects at all: fall back to frame-wide coverage so frames
            # without stored detections still surface next to real matches.
            label_match = len(parsed.labels & entry_label_set) / n_labels if n_labels else 1.0
            color_match = len(parsed.colors & entry_colors) / n_colors if n_colors else 1.0
            size_match = len(parsed.sizes & entry_sizes) / n_sizes if n_sizes else 1.0

        result = QueryResult(
            entry=entry,
            score=0.0,  # fused in query(); RRF needs all entries' ranks
            score_breakdown={
                "labels": sorted(parsed.labels),
                "colors": sorted(parsed.colors),
                "sizes": sorted(parsed.sizes),
                "label_match": round(label_match, 4),
                "color_match": round(color_match, 4),
                "size_match": round(size_match, 4),
                "structure": round(structure, 4),
                "dominance": round(dominance, 4),
                "semantic": round(semantic, 4),
                "quality": round(quality, 4),
            },
        )
        return result, signals

    @staticmethod
    def _object_cooccurrence(
        entry: IndexEntry, parsed: ParsedQuery
    ) -> tuple[float, dict | None]:
        """Return (structure, best_object) — best-object attribute coverage.

        ``structure`` = (types of queried attribute the best object satisfies)
        / (number of queried attribute types), so a "gray truck" query needs
        one object that is both gray and a truck to reach 1.0.
        """
        if not parsed.labels and not parsed.colors and not parsed.sizes:
            return 0.0, None
        n_types = sum(bool(s) for s in (parsed.labels, parsed.colors, parsed.sizes))
        best_credit, best_obj, best_label_hits = 0.0, None, 0
        for obj in entry.objects or []:
            covered = 0
            label_hit = bool(parsed.labels) and obj.get("label") in parsed.labels
            if label_hit:
                covered += 1
            if parsed.colors and obj.get("color_name") in parsed.colors:
                covered += 1
            if parsed.sizes and obj.get("size") in parsed.sizes:
                covered += 1
            if n_types and covered > 0:
                credit = covered / n_types
                if credit > best_credit or (
                    credit == best_credit and label_hit and not best_label_hits
                ):
                    best_credit, best_obj, best_label_hits = credit, obj, int(label_hit)
        return best_credit, best_obj

    @staticmethod
    def _dominance(objects: list[dict], parsed: ParsedQuery) -> float:
        """Smooth boost for frames where matching objects dominate the frame."""
        if not objects:
            return 0.0
        n_types = sum(bool(s) for s in (parsed.labels, parsed.colors, parsed.sizes))
        relevant = 0
        for obj in objects:
            covered = 0
            if parsed.labels and obj.get("label") in parsed.labels:
                covered += 1
            if parsed.colors and obj.get("color_name") in parsed.colors:
                covered += 1
            if parsed.sizes and obj.get("size") in parsed.sizes:
                covered += 1
            if n_types and covered >= n_types:
                relevant += 1
        if not relevant:
            return 0.0
        return math.log1p(relevant) / math.log1p(len(objects) + 1)

    def query_nearest_time(
        self,
        target_time,
        top_k: int = 10,
        labels: list[str] | None = None,
        generate_clips: bool = True,
    ) -> list[QueryResult]:
        """Return entries whose wall-clock time is closest to ``target_time``.

        ``target_time`` may be epoch seconds or an ISO string (parsed by
        :func:`parse_time`).  Entries are ranked by absolute time distance to
        the target — no relevance scoring — so a large ``top_k`` returns the
        whole neighborhood around that moment.  Entries without a
        ``wall_time`` are skipped.  With ``top_k=None`` every entry is
        returned in nearest-first order.  Scores carry ``|Δt|`` seconds
        (rounded), so callers can decide how close is close enough.
        """
        target = parse_time(target_time)
        if target is None:
            log.warning("query_nearest_time: invalid target time %r", target_time)
            return []

        entries = self.index_engine.load_entries()
        if not entries:
            log.warning("Index is empty")
            return []

        scored: list[QueryResult] = []
        for entry in entries:
            if entry.wall_time is None:
                continue
            if labels and not (set(labels) & set(entry.labels)):
                continue
            scored.append(QueryResult(entry=entry, score=round(abs(entry.wall_time - target), 4)))

        scored.sort(key=lambda r: r.score)  # by |Δt| ascending
        if top_k is not None:
            scored = scored[:top_k]

        results: list[QueryResult] = []
        for result in scored:
            if generate_clips:
                result.clip_path = self._generate_clip(result.entry)
            results.append(result)

        log.info("Nearest-time query t=%s returned %d results", target, len(results))
        return results

    def query_by_labels(
        self, labels: list[str], top_k: int = 10
    ) -> list[QueryResult]:
        """Search purely by object labels (no embedding search)."""
        entries = self.index_engine.load_entries()
        if not entries:
            return []

        matches: list[QueryResult] = []
        for entry in entries:
            overlap = len(set(labels) & set(entry.labels))
            if overlap > 0:
                score = overlap / len(labels)
                matches.append(QueryResult(entry=entry, score=score))

        matches.sort(key=lambda r: r.score, reverse=True)
        return matches[:top_k]

    def query_by_time(
        self,
        start_seconds: float,
        end_seconds: float,
        top_k: int = 10,
    ) -> list[QueryResult]:
        """Get all indexed frames within a time range."""
        entries = self.index_engine.load_entries()
        matches: list[QueryResult] = []
        for entry in entries:
            if start_seconds <= entry.timestamp <= end_seconds:
                matches.append(QueryResult(entry=entry, score=1.0))
        return matches[:top_k]

    def _generate_clip(self, entry: IndexEntry, duration: float | None = None) -> str | None:
        """Generate a short video clip centered on the frame timestamp."""
        dur = duration or self.clip_duration
        source = Path(entry.source_clip)
        if not source.is_absolute():
            source = self.storage_root.parent / source
        if not source.exists():
            return None

        self.clip_cache_dir.mkdir(parents=True, exist_ok=True)
        clip_name = f"{entry.id}.mp4"
        clip_path = self.clip_cache_dir / clip_name

        # Center clip on the timestamp
        start = max(0, entry.timestamp - dur / 2)

        try:
            self.ffmpeg.run(
                [
                    "-y",
                    "-ss", str(start),
                    "-i", str(source),
                    "-t", str(dur),
                    "-c", "copy",  # no re-encode — zero cost
                    str(clip_path),
                ],
                timeout=30,
            )
            if clip_path.exists():
                log.debug("Generated clip: %s", clip_path.name)
                return str(clip_path)
        except FFmpegError as exc:
            log.warning("Clip generation failed for %s: %s", entry.id, exc)

        return None

    def describe_entry(self, entry: IndexEntry, ollama_url: str = "http://localhost:11434") -> str:
        """Get a GenAI description for a specific frame (on-demand, lazy)."""
        try:
            from unkillable.genai.describer import Describer

            describer = Describer(base_url=ollama_url)
            return describer.describe(entry.thumbnail_path)
        except Exception as exc:
            log.warning("Description failed for %s: %s", entry.id, exc)
            return f"Description unavailable: {exc}"
