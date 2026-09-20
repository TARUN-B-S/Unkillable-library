import json
import re
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
        self._entry_cache: list[IndexEntry] | None = None
        self._entry_cache_key: tuple[int, int] | None = None

    def ensure_dirs(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)

    def index_clip(
        self,
        clip_path: Path | str,
        camera: str = "default",
        wall_time: float | None = None,
    ) -> list[IndexEntry]:
        """Index a single video clip — extract keyframes, detect, embed, store.

        ``wall_time`` (epoch seconds, offset 0 = the clip's first frame) is
        honored verbatim when given; otherwise derived from the clip filename
        (Frigate-style ``<camera>-<epoch>.<fraction>-<id>`` names) and finally
        the clip file's mtime.
        """
        clip = Path(clip_path)
        if not clip.exists():
            log.error("Clip not found: %s", clip)
            return []

        if wall_time is None:
            wall_time = self._derive_wall_time(clip)
        if wall_time is None:
            log.info("No wall-clock time derivable for %s — storing without one", clip.name)

        self.ensure_dirs()
        log.info("Indexing clip: %s", clip.name)

        # Step 1: Extract keyframes at motion events
        keyframes = self._extract_keyframes(clip)
        if not keyframes:
            log.info("No keyframes extracted from %s", clip.name)
            return []

        # Step 2: Process each keyframe (embeddings batched after detection)
        entries: list[IndexEntry] = []
        for ts, frame_path in keyframes:
            entry = self._process_keyframe(clip, ts, frame_path, camera, wall_time)
            if entry:
                entries.append(entry)
        self._embed_entries(entries)

        # Step 3: Persist all entries
        self._save_entries(entries)
        log.info("Indexed %d frames from %s", len(entries), clip.name)
        return entries

    @staticmethod
    def _derive_wall_time(clip: Path) -> float | None:
        """Best-effort wall-clock epoch for a clip: filename epoch → mtime."""
        stem = clip.stem
        match = re.search(r"(?<!\d)(\d{10})(?:\.\d+)?", stem)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        try:
            stat = clip.stat()
            return stat.st_mtime
        except OSError:
            return None

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
        self,
        clip: Path,
        timestamp: float,
        frame_path: Path,
        camera: str,
        wall_time: float | None = None,
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

        entry = IndexEntry(
            id=f"{clip.stem}_{uuid.uuid4().hex[:8]}",
            timestamp=timestamp,
            source_clip=str(clip.resolve()),
            thumbnail_path=str(frame_path),
            wall_time=(wall_time + timestamp) if wall_time is not None else None,
            camera=camera,
            motion_score=0.0,  # TODO: parse from scene detection
            labels=labels,
            objects=objects,
            counts=counts,
            embedding=[],
            tags=tags,
            detection_confidence=detection_conf,
            metadata={"camera": camera, "clip_name": clip.name},
        )
        return entry

    def _embed_entries(self, entries: list[IndexEntry]) -> None:
        """Embed all keyframes in batches — one encoder pass per ~32 frames.

        Sets the image embedding and, when the encoder mode is hash, a
        path-independent lexical fallback vector so queries can still match
        on frame tags/labels (P6).
        """
        if not entries:
            return
        batch_size = 32
        try:
            for start in range(0, len(entries), batch_size):
                batch = entries[start : start + batch_size]
                vecs = self.search.embed_images([e.thumbnail_path for e in batch])
                for entry, vec in zip(batch, vecs):
                    entry.embedding = vec
        except Exception as exc:
            log.debug("Batch embedding failed for %s: %s", entries[0].id, exc)
            for entry in entries:
                entry.embedding = []
        # Hash mode has no true image encoder: the single-call fallback would
        # embed the file *path*, which matches nothing.  Store a lexical
        # vector of the frame's tags/labels instead so text queries rank by
        # token overlap with what was actually detected.
        if self.search._encoder is None:
            for entry in entries:
                entry.embedding = self.search.embed_text(" ".join(entry.tags) or entry.id)
        # Text vectors for dual-embedding search: embed the frame's tags so
        # CLIP text↔text (or lexical↔lexical) can match on descriptions.
        for entry in entries:
            entry.text_embedding = self.search.embed_text(" ".join(entry.tags) or entry.id)

    def _save_entries(self, entries: list[IndexEntry]) -> None:
        """Append entries to the JSONL index file."""
        self.ensure_dirs()
        key_before = self._file_key()
        try:
            with open(self.entries_file, "a", encoding="utf-8") as f:
                f.writelines(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n" for entry in entries)
        except OSError as exc:
            log.error("Failed to save entries: %s", exc)
            return
        # Keep the in-memory cache in sync without a full re-read — but only
        # when it was current before this append; otherwise drop it so the
        # next load re-reads the file (another writer may have added lines).
        if self._entry_cache is not None and self._entry_cache_key == key_before:
            self._entry_cache.extend(entries)
            self._entry_cache_key = self._file_key()
        else:
            self._entry_cache = None
            self._entry_cache_key = None

    @staticmethod
    def _stat_key(path: Path) -> tuple[int, int] | None:
        """Cache-invalidation key for a file: (mtime_ns, size), or None if unreadable."""
        try:
            st = path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def _file_key(self) -> tuple[int, int] | None:
        return self._stat_key(self.entries_file)

    def load_entries(self, limit: int = 1000) -> list[IndexEntry]:
        """Load entries from the index file (cached in memory between calls).

        The parsed entry list is cached and invalidated whenever the file's
        mtime or size changes, so external appends are still picked up.
        """
        if not self.entries_file.exists():
            return []
        key = self._file_key()
        if self._entry_cache is not None and key == self._entry_cache_key:
            return self._entry_cache[-limit:]
        entries: list[IndexEntry] = []
        try:
            with open(self.entries_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(IndexEntry.from_dict(json.loads(line)))
                        except (json.JSONDecodeError, TypeError):
                            log.warning("Skipping corrupt entry line in %s", self.entries_file)
        except OSError as exc:
            log.error("Failed to load entries: %s", exc)
        self._entry_cache = entries
        self._entry_cache_key = key
        return entries[-limit:]

    def get_entry(self, entry_id: str) -> IndexEntry | None:
        """Get a single entry by ID."""
        for entry in self.load_entries(limit=10000):
            if entry.id == entry_id:
                return entry
        return None

    def _rewrite_entries(self, entries: list[IndexEntry]) -> None:
        """Atomically rewrite the whole entries JSONL (update/delete paths)."""
        self.ensure_dirs()
        tmp = self.entries_file.with_suffix(".jsonl.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(e.to_dict(), ensure_ascii=False) + "\n" for e in entries)
            tmp.replace(self.entries_file)
        except OSError as exc:
            log.error("Failed to rewrite entries file: %s", exc)
            raise
        self._entry_cache = None
        self._entry_cache_key = None

    def update_entry_description(self, entry_id: str, description: str) -> None:
        """Persist a GenAI description for an entry and refresh its text vector.

        The description text becomes searchable immediately: the entry's
        ``text_embedding`` is re-derived from the description and the JSONL is
        rewritten in place (atomic tmp-file replace).
        """
        entries = self.load_entries(limit=10**9)
        target = next((e for e in entries if e.id == entry_id), None)
        if target is None:
            raise KeyError(f"Entry not found: {entry_id}")
        target.metadata = {**target.metadata, "description": description}
        target.text_embedding = self.search.embed_text(description)
        self._rewrite_entries(entries)

    def upsert_entry(self, entry: IndexEntry) -> None:
        """Replace an entry by id (or append if new) — used by feedback."""
        entries = self.load_entries(limit=10**9)
        for i, e in enumerate(entries):
            if e.id == entry.id:
                entries[i] = entry
                break
        else:
            entries.append(entry)
        self._rewrite_entries(entries)

    def index_count(self) -> int:
        """Return total number of indexed entries."""
        return len(self.load_entries(limit=10**9))

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
