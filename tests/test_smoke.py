import tempfile
from pathlib import Path

from unkillable.utils.storage import Storage
from unkillable.semantic.search import SemanticSearch
from unkillable.motion.engine import MotionEngine
from unkillable.backup.qr_backup import QRBackup


def test_storage():
    with tempfile.TemporaryDirectory() as d:
        s = Storage(Path(d))
        s.ensure_dirs()
        s.save_event({"camera": "test", "label": "person"})
        assert len(s.load_events()) == 1


def test_semantic_hash():
    with tempfile.TemporaryDirectory() as d:
        ss = SemanticSearch(db_path=Path(d) / "db.jsonl")
        ss.index("1", ss.embed_text("red car"), {"camera": "front"})
        ss.index("2", ss.embed_text("blue dog"), {"camera": "back"})
        results = ss.search("red car", top_k=1)
        assert results[0].id == "1"


def test_motion_dummy():
    m = MotionEngine()
    events = m._dummy_motion("dummy_src")
    assert len(events) == 1


def test_qr_archive():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "hello.txt"
        src.write_text("hello unkillable")
        out = Path(d) / "out.tar.gz"
        pdf = Path(d) / "out.pdf"
        qb = QRBackup()
        qb.create_archive([src], out)
        assert out.exists()
        qb.encode_to_qr_pdf(out, pdf)
        assert pdf.exists()
