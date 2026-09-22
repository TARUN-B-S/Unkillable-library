import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

from unkillable.backup import frame_codec
from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)


class RestoreError(RuntimeError):
    pass


def restore_frames(input_dir: str | Path, output: str | Path,
                   nsym: int | None = None, verify_only: bool = False) -> dict:
    src = Path(input_dir)
    manifest_path = src / "manifest.json"
    if not manifest_path.exists():
        raise RestoreError(f"manifest.json not found in {src}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nsym = nsym or int(manifest.get("nsym", 16))
    frames = sorted(src.glob("frame_*.png"))
    if not frames:
        raise RestoreError(f"No frame_*.png in {src}")
    try:
        images = frame_codec.load_frames(frames)
        raw = frame_codec.decode_images(images, nsym=nsym, total_bytes=manifest.get("tarball_bytes"))
    except Exception as exc:
        raise RestoreError(f"Frame decode failed: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    expected = manifest.get("tarball_sha256", "")
    if expected and digest != expected:
        raise RestoreError(f"SHA256 mismatch: got {digest[:16]}.. expected {expected[:16]}.. — data corrupt beyond repair")
    log.info("Restore verified: %d frames, %d bytes, sha %s", len(frames), len(raw), digest[:16])
    if verify_only:
        return {"frames": len(frames), "bytes": len(raw), "sha256": digest, "verified": True}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(path=out)
    except (OSError, tarfile.TarError) as exc:
        raise RestoreError(f"Tar extract failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"frames": len(frames), "bytes": len(raw), "sha256": digest, "output": str(out)}
