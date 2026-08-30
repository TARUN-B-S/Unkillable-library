import base64
import gzip
import math
import tarfile
import tempfile
from pathlib import Path

from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)


class QRBackupError(RuntimeError):
    pass


class QRBackup:
    def __init__(self, chunk_size: int = 2953, redundancy: float = 0.3, error_correction: str = "M") -> None:
        self.chunk_size = chunk_size
        self.redundancy = redundancy
        self.error_correction = error_correction

    def create_archive(self, sources: list[Path], output: Path) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(output, "w:gz") as tar:
                for src in sources:
                    if not src.exists():
                        log.warning("Source missing, skipping: %s", src)
                        continue
                    tar.add(src, arcname=src.name)
            log.info("Archive created: %s (%d bytes)", output, output.stat().st_size)
            return output
        except (OSError, tarfile.TarError) as exc:
            raise QRBackupError(f"Archive creation failed: {exc}") from exc

    def encode_to_qr_pdf(self, archive: Path, pdf_output: Path, title: str = "THE UNKILLABLE LIBRARY — BACKUP") -> Path:
        if not archive.exists():
            raise QRBackupError(f"Archive not found: {archive}")
        try:
            import qrcode
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import mm
            from reportlab.pdfgen import canvas
        except ImportError as exc:
            raise QRBackupError(f"Missing dependency for QR PDF: {exc}") from exc

        try:
            raw = archive.read_bytes()
            b64 = base64.b64encode(raw).decode()
            chunks = [b64[i : i + self.chunk_size] for i in range(0, len(b64), self.chunk_size)]
            total = len(chunks)
            redundant = math.ceil(total * self.redundancy)
            log.info("Encoding %d chunks + %d redundant -> %d QR codes", total, redundant, total + redundant)
        except OSError as exc:
            raise QRBackupError(f"Failed to read archive: {exc}") from exc

        try:
            pdf_output.parent.mkdir(parents=True, exist_ok=True)
            c = canvas.Canvas(str(pdf_output), pagesize=A4)
            w, h = A4
            cols, rows = 4, 4
            qr_size = 38 * mm
            margin_x = (w - cols * qr_size) / (cols + 1)
            margin_y = 20 * mm

            c.setFont("Helvetica-Bold", 14)
            c.drawCentredString(w / 2, h - 15 * mm, title)
            c.setFont("Helvetica", 8)
            c.drawCentredString(w / 2, h - 22 * mm, f"Archive: {archive.name} | Chunks: {total} | Date: {archive.stat().st_mtime}")

            idx = 0
            for chunk_idx, chunk in enumerate(chunks):
                col = idx % cols
                row = (idx // cols) % rows
                if idx > 0 and idx % (cols * rows) == 0:
                    c.showPage()
                    c.setFont("Helvetica", 8)
                    c.drawCentredString(w / 2, h - 10 * mm, f"{title} — continued")
                x = margin_x + col * (qr_size + margin_x)
                y = h - 35 * mm - (row + 1) * (qr_size + 5 * mm)

                payload = f"{chunk_idx+1}/{total}|{chunk}"
                qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=4, border=2)
                qr.add_data(payload)
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    img.save(tmp.name)
                    tmp_path = tmp.name

                c.drawImage(tmp_path, x, y, width=qr_size, height=qr_size)
                c.setFont("Helvetica", 6)
                c.drawCentredString(x + qr_size / 2, y - 4 * mm, f"QR {chunk_idx+1}/{total}")
                Path(tmp_path).unlink(missing_ok=True)
                idx += 1

            c.setFont("Helvetica", 7)
            c.drawCentredString(w / 2, 12 * mm, "RESTORE: qr-backup --restore <pdf>  or  zbarimg --raw *.jpg > backup.tar.gz")
            c.save()
            log.info("QR PDF created: %s (%d QR codes, %d pages)", pdf_output, total, c.getPageNumber())
            return pdf_output
        except Exception as exc:
            raise QRBackupError(f"QR PDF generation failed: {exc}") from exc
