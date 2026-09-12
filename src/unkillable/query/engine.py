import re
from dataclasses import dataclass, field
from pathlib import Path

from unkillable.detection.detector import SUPPORTED_LABELS
from unkillable.index.engine import IndexEngine
from unkillable.index.models import IndexEntry
from unkillable.semantic.search import cosine
from unkillable.utils.ffmpeg import FFmpegError, FFmpegWrapper
from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)

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
    ) -> list[QueryResult]:
        """Search the index by natural language query.

        Uses a hybrid scorer: structured label/color/size matches fused with
        semantic similarity and per-frame detection quality.
        """
        if not text or not text.strip():
            return []

        entries = self.index_engine.load_entries()
        if not entries:
            log.warning("Index is empty")
            return []

        parsed = self._parse_query(text)
        qvec = self.search.embed_text(text)

        # Score all entries
        scored: list[tuple[QueryResult, float]] = []
        for entry in entries:
            if not entry.embedding:
                continue
            result, score = self._score(entry, parsed, qvec, labels)
            scored.append((result, score))

        # Sort by score descending, stable ties by recency of insertion
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

    def _score(
        self,
        entry: IndexEntry,
        parsed: ParsedQuery,
        qvec: list[float] | None,
        extra_labels: list[str] | None = None,
    ) -> tuple[QueryResult, float]:
        """Hybrid score: structure (labels/colors/sizes) + semantics + quality."""
        entry_colors = {o.get("color_name") for o in entry.objects if o.get("color_name")}
        entry_sizes = {o.get("size") for o in entry.objects if o.get("size")}
        entry_label_set = set(entry.labels)

        n_labels = len(parsed.labels)
        n_colors = len(parsed.colors)
        n_sizes = len(parsed.sizes)
        has_structure = n_labels or n_colors or n_sizes

        label_match = (
            len(parsed.labels & entry_label_set) / n_labels if n_labels else 1.0
        )
        color_match = len(parsed.colors & entry_colors) / n_colors if n_colors else 1.0
        size_match = len(parsed.sizes & entry_sizes) / n_sizes if n_sizes else 1.0

        semantic = cosine(qvec, entry.embedding) if qvec and entry.embedding else 0.0
        quality = entry.detection_confidence

        if has_structure:
            # Structure should dominate when the user names attributes, but
            # semantics + quality still break ties between matches.
            score = (
                0.40 * label_match
                + 0.25 * color_match
                + 0.10 * size_match
                + 0.15 * semantic
                + 0.10 * quality
            )
        else:
            # Free-form query: rely on semantic similarity + detection quality.
            score = 0.80 * semantic + 0.20 * quality

        if extra_labels and not (set(extra_labels) & entry_label_set):
            score *= 0.20  # penalize but don't eliminate

        score = max(0.0, min(1.0, score))

        result = QueryResult(
            entry=entry,
            score=round(score, 4),
            score_breakdown={
                "labels": sorted(parsed.labels),
                "colors": sorted(parsed.colors),
                "sizes": sorted(parsed.sizes),
                "label_match": round(label_match, 4),
                "color_match": round(color_match, 4),
                "size_match": round(size_match, 4),
                "semantic": round(semantic, 4),
                "quality": round(quality, 4),
            },
        )
        return result, score

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
