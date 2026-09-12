import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from unkillable.utils.ffmpeg import FFmpegError, FFmpegWrapper
from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class MotionEvent:
    camera: str
    start_ts: float
    end_ts: float | None = None
    score: float = 0.0


class MotionEngine:
    def __init__(
        self,
        threshold: float = 0.02,
        min_duration: float = 1.0,
        buffer: float = 2.0,
        ffmpeg: FFmpegWrapper | None = None,
        storage_root: Path | str | None = None,
        auto_index: bool = True,
    ) -> None:
        self.threshold = threshold
        self.min_duration = min_duration
        self.buffer = buffer
        self.ffmpeg = ffmpeg or FFmpegWrapper()
        self.storage_root = Path(storage_root) if storage_root else Path("storage")
        self.auto_index = auto_index

    def detect_via_ffmpeg(self, input_src: str, duration: int = 10) -> list[MotionEvent]:
        if not self.ffmpeg.is_available():
            log.warning("FFmpeg not available, falling back to dummy motion")
            return self._dummy_motion(input_src)
        vf = f"select='gt(scene,{self.threshold})',metadata=print"
        output = "-f"
        try:
            result = self.ffmpeg.run(["-i", input_src, "-vf", vf, "-t", str(duration), "-an", "-f", "null", "-"], timeout=duration + 10)
            events = self._parse_metadata(result.stderr)
            log.info("FFmpeg motion detected %d events from %s", len(events), input_src)
            return events
        except FFmpegError as exc:
            log.error("Motion detection failed: %s", exc)
            return self._dummy_motion(input_src)

    def _parse_metadata(self, stderr: str) -> list[MotionEvent]:
        events: list[MotionEvent] = []
        for line in stderr.splitlines():
            if "lavfi.scene_score" in line:
                try:
                    score = float(line.split("lavfi.scene_score=")[-1].split()[0])
                    if score > self.threshold:
                        events.append(MotionEvent(camera="default", start_ts=time.time(), score=score))
                except (ValueError, IndexError):
                    continue
        return events

    def _dummy_motion(self, src: str) -> list[MotionEvent]:
        log.info("Dummy motion: simulating 1 event for %s", src)
        return [MotionEvent(camera="default", start_ts=time.time(), score=0.5)]

    def record_on_motion(self, input_src: str, output: Path, duration: int = 10, camera: str = "default") -> Path | None:
        events = self.detect_via_ffmpeg(input_src, duration)
        if not events:
            log.info("No motion, skipping recording for %s", input_src)
            return None
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.ffmpeg.run(["-y", "-i", input_src, "-t", str(duration), "-c", "copy", str(output)])
            log.info("Recording saved: %s (triggered by %d motion events)", output, len(events))

            # Auto-index the recorded clip
            if self.auto_index and output.exists():
                self._auto_index_clip(output, camera)

            return output
        except FFmpegError as exc:
            log.error("Recording failed: %s", exc)
            raise

    def _auto_index_clip(self, clip_path: Path, camera: str = "default") -> None:
        """Index a recorded clip in the background (best-effort)."""
        try:
            from unkillable.index.engine import IndexEngine

            ie = IndexEngine(storage_root=self.storage_root)
            entries = ie.index_clip(clip_path, camera=camera)
            log.info("Auto-indexed %d frames from %s", len(entries), clip_path.name)
        except Exception as exc:
            log.warning("Auto-index failed for %s: %s", clip_path.name, exc)

    def ffmpeg_motion_filter_cmd(self, input_src: str, output: str) -> list[str]:
        return ["-i", input_src, "-vf", f"select='gt(scene,{self.threshold})',scale=640:360", "-vsync", "vfr", output]
