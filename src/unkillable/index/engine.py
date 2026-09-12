import json
import uuid
from pathlib import Path

from unkillable.detection.detector import Detector
from unkillable.index.models import IndexEntry
from unkillable.semantic.search import SemanticSearch
from unkillable.utils.ffmpeg import FFmpegError, FFmpegWrapper
from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)

# Default tracked classes — broad enough to enrich tagging while staying
# meaningful for a monitoring stream.
DEFAULT_TRACK_LABELS = ["person", "car", "truck", "bus", "motorcycle", "bicycle", "dog", "cat"]


def build_tags(objects: list[dict]) -> list[str]:
    """Derive rich tags from a frame's detected objects.

    Produces compound tags ("red car"), plus size qualifiers for conspicuous
    objects ("large truck") and the bare class names.
    """
    tags: set[str] = set()
    for o in objects:
        label = o.get("label")
        if not label:
            continue
        tags.add(label)
        color = o.get("color_name")
        if color and color not in ("unknown",):
            tags.add(f"{color} {label}")
        size = o.get("size")
        if size in ("large", "huge"):
            tags.add(f"{size} {label}")
    return sorted(tags)


class IndexEngine:
    """Index a video clip: extract keyframes, detect objects, embed, store."""

    def __init__(
        self,
        storage_root: Path | str = "storage",
        motion_threshold: float = 0.02,
        detect_labels: list[str] | None = None,
        ffmpeg: FFmpegWrapper | None = None,
        detector: Detector | None = None,
        search: SemanticSearch | None = None,
    ) -> None:
        self.storage_root = Path(storage_root).resolve()
        self.index_dir = self.storage_root / "index"
        self.thumbs_dir = self.index_dir / "thumbnails"
        self.entries_file = self.index_dir / "entries.jsonl"
        self.ffmpeg = ffmpeg or FFmpegWrapper()
        self.motion_threshold = motion_threshold
        self.detect_labels = detect_labels or DEFAULT_TRACK_LABELS
        self.detector = detector or Detector(track_labels=self.detect_labels)
        self.search = search or SemanticSearch(db_path=self.entries_file)

    def ensure_dirs(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)

    def index_clip(self, clip_path: Path | str, camera: str = "default") -> list[IndexEntry]:
        """Index a single video clip — extract keyframes, detect, embed, store."""
        clip = Path(clip_path)
        if not clip.exists():
            log.error("Clip not found: %s", clip)
            return []

        self.ensure_dirs()
        log.info("Indexing clip: %s", clip.name)

        # Step 1: Extract keyframes at motion events
        keyframes = self._extract_keyframes(clip)
        if not keyframes:
            log.info("No keyframes extracted from %s", clip.name)
            return []

        # Step 2: Process each keyframe
        entries: list[IndexEntry] = []
        for ts, frame_path in keyframes:
            entry = self._process_keyframe(clip, ts, frame_path, camera)
            if entry:
                entries.append(entry)

        # Step 3: Persist all entries
        self._save_entries(entries)
        log.info("Indexed %d frames from %s", len(entries), clip.name)
        return entries

    def _extract_keyframes(self, clip: Path) -> list[tuple[float, Path]]:
        """Use FFmpeg scene detection to extract keyframes at motion events."""
        if not self.ffmpeg.is_available():
            log.warning("FFmpeg not available, extracting single frame at 1s")
            thumb = self.thumbs_dir / f"{clip.stem}_001.jpg"
            try:
                self.ffmpeg.extract_thumbnail(str(clip), thumb, "00:00:01", "640:-1")
                return [(1.0, thumb)]
            except FFmpegError:
                return []

        # Run FFmpeg with scene detection + metadata print
        vf = f"select='gt(scene,{self.motion_threshold})',metadata=print:file=-"
        tmp_output = self.index_dir / f"_tmp_{clip.stem}.log"

        try:
            self.ffmpeg.run(
                [
                    "-i", str(clip),
                    "-vf", vf,
                    "-vsync", "vfr",
                    "-f", "null", "-",
                ],
                timeout=120,
            )
        except FFmpegError as exc:
            log.warning("Scene detection failed for %s: %s", clip.name, exc)
            # Fallback: extract 1 frame per second for first 60s
            return self._fallback_extract(clip)

        # Parse timestamps from metadata
        timestamps = self._parse_scene_timestamps(tmp_output)

        # If no scene changes detected, fallback
        if not timestamps:
            return self._fallback_extract(clip)

        # Extract frame at each timestamp
        keyframes: list[tuple[float, Path]] = []
        for i, ts in enumerate(timestamps[:100]):  # cap at 100 keyframes
            thumb = self.thumbs_dir / f"{clip.stem}_f{i:04d}.jpg"
            try:
                ts_str = f"{int(ts) // 3600:02d}:{(int(ts) % 3600) // 60:02d}:{int(ts) % 60:02d}"
                self.ffmpeg.extract_thumbnail(str(clip), thumb, ts_str, "640:-1")
                keyframes.append((ts, thumb))
            except FFmpegError as exc:
                log.warning("Failed to extract frame at %.1fs: %s", ts, exc)

        return keyframes

    def _fallback_extract(self, clip: Path) -> list[tuple[float, Path]]:
        """Fallback: extract 1 frame per 5 seconds for first 60s."""
        keyframes: list[tuple[float, Path]] = []
        for sec in range(0, 60, 5):
            thumb = self.thumbs_dir / f"{clip.stem}_fb{sec:04d}.jpg"
            try:
                ts_str = f"00:00:{sec:02d}"
                self.ffmpeg.extract_thumbnail(str(clip), thumb, ts_str, "640:-1")
                keyframes.append((float(sec), thumb))
            except FFmpegError:
                continue
        return keyframes

    def _parse_scene_timestamps(self, log_path: Path) -> list[float]:
        """Parse FFmpeg metadata output for scene timestamps."""
        # FFmpeg metadata prints pts_time for each frame
        timestamps: list[float] = []
        if not log_path.exists():
            return timestamps
        try:
            content = log_path.read_text()
            for line in content.splitlines():
                if "pts_time:" in line:
                    try:
                        ts = float(line.split("pts_time:")[-1].strip().split()[0])
                        timestamps.append(ts)
                    except (ValueError, IndexError):
                        continue
        except OSError:
            pass
        return timestamps

    def _process_keyframe(
        self, clip: Path, timestamp: float, frame_path: Path, camera: str
    ) -> IndexEntry | None:
        """Run detection + embedding on a single keyframe."""
        if not frame_path.exists():
            return None

        # Object detection
        labels: list[str] = []
        objects: list[dict] = []
        counts: dict[str, int] = {}
        detection_conf: float = 0.0
        try:
            detections = self.detector.detect(frame_path)
            labels = [d.label for d in detections]
            objects = [d.to_dict() for d in detections]
            counts = Detector.counts(detections)
            detection_conf = Detector.mean_confidence(detections)
        except Exception as exc:
            log.debug("Detection failed for %s: %s", frame_path.name, exc)

        tags = build_tags(objects)

        # Embedding (for semantic search)
        embedding: list[float] = []
        try:
            embedding = self.search.embed_image(frame_path)
        except Exception as exc:
            log.debug("Embedding failed for %s: %s", frame_path.name, exc)

        entry = IndexEntry(
            id=f"{clip.stem}_{uuid.uuid4().hex[:8]}",
            timestamp=timestamp,
            source_clip=str(clip.resolve()),
            thumbnail_path=str(frame_path),
            motion_score=0.0,  # TODO: parse from scene detection
            labels=labels,
            objects=objects,
            counts=counts,
            embedding=embedding,
            tags=tags,
            detection_confidence=detection_conf,
            metadata={"camera": camera, "clip_name": clip.name},
        )
        return entry

    def _save_entries(self, entries: list[IndexEntry]) -> None:
        """Append entries to the JSONL index file."""
        self.ensure_dirs()
        try:
            with open(self.entries_file, "a", encoding="utf-8") as f:
                f.writelines(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n" for entry in entries)
        except OSError as exc:
            log.error("Failed to save entries: %s", exc)

    def load_entries(self, limit: int = 1000) -> list[IndexEntry]:
        """Load entries from the index file."""
        if not self.entries_file.exists():
            return []
        entries: list[IndexEntry] = []
        try:
            with open(self.entries_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(IndexEntry.from_dict(json.loads(line)))
        except (OSError, json.JSONDecodeError) as exc:
            log.error("Failed to load entries: %s", exc)
        return entries[-limit:]

    def get_entry(self, entry_id: str) -> IndexEntry | None:
        """Get a single entry by ID."""
        for entry in self.load_entries(limit=10000):
            if entry.id == entry_id:
                return entry
        return None

    def index_count(self) -> int:
        """Return total number of indexed entries."""
        if not self.entries_file.exists():
            return 0
        count = 0
        try:
            with open(self.entries_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except OSError:
            pass
        return count

    @staticmethod
    def clip_counts(entries: list[IndexEntry]) -> dict[str, dict[str, int]]:
        """Aggregate per-clip object totals: {source_clip: {label: count}}.

        Prefers per-frame ``counts``; falls back to ``labels`` (count 1 each)
        for legacy entries that predate counting.
        """
        totals: dict[str, dict[str, int]] = {}
        for entry in entries:
            bucket = totals.setdefault(entry.source_clip, {})
            per_label = entry.counts or {label: 1 for label in entry.labels}
            for label, count in per_label.items():
                bucket[label] = bucket.get(label, 0) + count
        return totals
