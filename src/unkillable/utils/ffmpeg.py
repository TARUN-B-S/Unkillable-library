import shutil
import subprocess
from pathlib import Path

from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)


class FFmpegError(RuntimeError):
    pass


class FFmpegWrapper:
    def __init__(self, binary: str = "ffmpeg") -> None:
        self.binary = shutil.which(binary) or binary

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None or Path(self.binary).exists()

    def run(self, args: list[str], timeout: float | None = 120) -> subprocess.CompletedProcess:
        cmd = [self.binary] + args
        log.debug("FFmpeg cmd: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as exc:
            raise FFmpegError(f"FFmpeg binary not found: {self.binary}") from exc
        except subprocess.TimeoutExpired as exc:
            raise FFmpegError(f"FFmpeg timeout after {timeout}s") from exc
        if result.returncode != 0:
            raise FFmpegError(f"FFmpeg failed ({result.returncode}): {result.stderr[:500]}")
        return result

    def probe(self, input_path: str) -> str:
        ffprobe = shutil.which("ffprobe") or "ffprobe"
        try:
            result = subprocess.run([ffprobe, "-v", "error", "-show_entries", "stream=codec_name,width,height,r_frame_rate", "-of", "default=noprint_wrappers=1", input_path], capture_output=True, text=True, timeout=10)
            return result.stdout
        except Exception as exc:
            log.warning("ffprobe failed: %s", exc)
            return ""

    def extract_thumbnail(self, input_src: str, output: Path, timestamp: str = "00:00:01", scale: str = "320:-1") -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        self.run(["-y", "-ss", timestamp, "-i", input_src, "-vframes", "1", "-vf", f"scale={scale}", str(output)])
        if not output.exists():
            raise FFmpegError(f"Thumbnail not created: {output}")
        log.info("Thumbnail extracted: %s", output)
        return output

    def create_test_video(self, output: Path, duration: int = 10, size: str = "1280x720", rate: int = 30) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        self.run(["-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}", "-f", "lavfi", "-i", f"sine=frequency=1000", "-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(output)])
        return output

    def loop_to_rtsp(self, input_file: Path, rtsp_url: str) -> subprocess.Popen:
        if not self.is_available():
            raise FFmpegError("FFmpeg not available for RTSP streaming")
        cmd = [self.binary, "-re", "-stream_loop", "-1", "-i", str(input_file), "-c", "copy", "-f", "rtsp", rtsp_url]
        log.info("Starting FFmpeg RTSP loop: %s -> %s", input_file, rtsp_url)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return proc
        except Exception as exc:
            raise FFmpegError(f"Failed to start RTSP loop: {exc}") from exc
