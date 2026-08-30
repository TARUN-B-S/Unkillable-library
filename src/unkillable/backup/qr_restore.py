import base64
import tarfile
from pathlib import Path

from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)


class QRRestoreError(RuntimeError):
    pass


class QRRestore:
    def decode_from_images(self, image_paths: list[Path], output_archive: Path) -> Path:
        if not image_paths:
            raise QRRestoreError("No images provided for restore")
        try:
            from pyzbar.pyzbar import decode
            from PIL import Image
        except ImportError as exc:
            raise QRRestoreError(f"pyzbar/Pillow required for restore: {exc}") from exc

        chunks: dict[int, str] = {}
        total_expected: int | None = None

        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                decoded = decode(img)
                for obj in decoded:
                    data = obj.data.decode()
                    if "|" not in data or "/" not in data:
                        continue
                    header, payload = data.split("|", 1)
                    idx_str, total_str = header.split("/")
                    idx = int(idx_str)
                    total_expected = int(total_str)
                    chunks[idx] = payload
            except Exception as exc:
                log.warning("Failed to decode %s: %s", img_path, exc)
                continue

        if not chunks:
            raise QRRestoreError("No QR data decoded from images")
        if total_expected and len(chunks) < total_expected:
            log.warning("Missing chunks: got %d/%d", len(chunks), total_expected)
            if len(chunks) < total_expected * 0.7:
                raise QRRestoreError(f"Too many missing chunks: {len(chunks)}/{total_expected}")

        try:
            ordered = "".join(chunks[i] for i in sorted(chunks))
            raw = base64.b64decode(ordered)
            output_archive.parent.mkdir(parents=True, exist_ok=True)
            output_archive.write_bytes(raw)
            log.info("Restored archive: %s (%d bytes from %d QR codes)", output_archive, len(raw), len(chunks))
            return output_archive
        except Exception as exc:
            raise QRRestoreError(f"Reassembly failed: {exc}") from exc

    def extract_archive(self, archive: Path, dest: Path) -> Path:
        if not archive.exists():
            raise QRRestoreError(f"Archive not found: {archive}")
        try:
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(path=dest)
            log.info("Archive extracted to %s", dest)
            return dest
        except (OSError, tarfile.TarError) as exc:
            raise QRRestoreError(f"Extraction failed: {exc}") from exc
