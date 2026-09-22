import hashlib
import shutil
import tarfile
import tempfile
from datetime import date
from pathlib import Path

from unkillable.backup import frame_codec
from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class BackupError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _compress_video(src: Path, dst: Path, crf: int = 28) -> bool:
    from unkillable.utils.ffmpeg import FFmpegWrapper

    fw = FFmpegWrapper()
    if not fw.is_available():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        fw.run(["-y", "-i", str(src), "-c:v", "libx264", "-crf", str(crf),
                "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(dst)], timeout=300)
        if dst.exists() and dst.stat().st_size < src.stat().st_size:
            return True
        dst.unlink(missing_ok=True)
        return False
    except Exception as exc:
        log.warning("Video recompress failed %s: %s", src, exc)
        dst.unlink(missing_ok=True)
        return False


def _compress_image(src: Path, dst: Path) -> bool:
    try:
        from PIL import Image

        img = Image.open(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        fmt = "JPEG" if dst.suffix.lower() in (".jpg", ".jpeg") else None
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        if fmt == "JPEG" or dst.suffix.lower() in (".jpg", ".jpeg"):
            img.save(dst, "JPEG", quality=70, optimize=True)
        elif dst.suffix.lower() == ".webp":
            img.save(dst, "WEBP", quality=70, method=4)
        else:
            img.save(dst, "JPEG", quality=70, optimize=True)
            dst2 = dst.with_suffix(".jpg")
            dst.replace(dst2)
            dst = dst2
        if dst.exists() and dst.stat().st_size < src.stat().st_size:
            return True
        dst.unlink(missing_ok=True)
        return False
    except Exception as exc:
        log.warning("Image recompress failed %s: %s", src, exc)
        return False


def stage_storage(source: Path, staging: Path, crf: int = 28) -> int:
    source = Path(source)
    if not source.exists():
        raise BackupError(f"Source not found: {source}")
    count = 0
    for f in source.rglob("*"):
        if not f.is_file():
            continue
        if "backups" in f.parts:
            continue
        rel = f.relative_to(source)
        dst = staging / rel
        ext = f.suffix.lower()
        try:
            if ext in VIDEO_EXTS:
                tmp = dst.with_suffix(".mp4")
                if _compress_video(f, tmp, crf):
                    count += 1
                    continue
            elif ext in IMAGE_EXTS and f.stat().st_size > 50 * 1024:
                if _compress_image(f, dst):
                    count += 1
                    continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            count += 1
        except OSError as exc:
            log.warning("Stage skip %s: %s", f, exc)
    return count


def backup_daily(source: str | Path = "storage", dest: str | Path = "backups",
                 date_str: str | None = None, crf: int = 28, nsym: int = 16,
                 verify: bool = True, delete_source: bool = False) -> dict:
    import json

    src = Path(source)
    if delete_source and not verify:
        raise BackupError("--delete-source requires --verify")
    day = date_str or date.today().isoformat()
    out_dir = Path(dest) / day
    if out_dir.exists() and any(out_dir.glob("frame_*.png")):
        raise BackupError(f"Backup for {day} already exists at {out_dir}")
    with tempfile.TemporaryDirectory(prefix="uklb_stage_") as stage_s, \
            tempfile.TemporaryDirectory(prefix="uklb_tar_") as tar_s:
        staging = Path(stage_s) / "root"
        staging.mkdir(parents=True, exist_ok=True)
        nfiles = stage_storage(src, staging, crf)
        if nfiles == 0:
            raise BackupError(f"Nothing to back up in {src}")
        tar_path = Path(tar_s) / "daily.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(staging, arcname="storage")
        digest = sha256_file(tar_path)
        raw = tar_path.read_bytes()
        frames = frame_codec.encode_bytes(raw, nsym=nsym)
        paths = frame_codec.save_frames(frames, out_dir)
        manifest = {
            "date": day, "files": nfiles, "frames": len(paths),
            "tarball_sha256": digest, "tarball_bytes": len(raw),
            "crf": crf, "nsym": nsym, "width": 1920, "height": 1080, "mode": "L",
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log.info("Backup %s: %d files, %d bytes -> %d frames", day, nfiles, len(raw), len(paths))
        if verify:
            imgs = frame_codec.load_frames(paths)
            try:
                back = frame_codec.decode_images(imgs, nsym=nsym)
            except Exception as exc:
                raise BackupError(f"Verify decode failed: {exc}") from exc
            if hashlib.sha256(back).hexdigest() != digest:
                raise BackupError("Verify failed: sha256 mismatch")
            log.info("Verify OK for %s", day)
        if delete_source:
            removed = 0
            for f in src.rglob("*"):
                if f.is_file():
                    try:
                        f.unlink()
                        removed += 1
                    except OSError as exc:
                        log.warning("Delete skip %s: %s", f, exc)
            for d in sorted([p for p in src.rglob("*") if p.is_dir()], reverse=True):
                try:
                    d.rmdir()
                except OSError:
                    pass
            log.info("Deleted %d files from %s after verified backup", removed, src)
            manifest["deleted_source_files"] = removed
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"dir": str(out_dir), **manifest}
