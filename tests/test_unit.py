"""Unit tests for the Unkillable Library Python modules.

Tests run against generated ffmpeg commands (string assertions) and
internal logic — no real media files needed.
"""
import hashlib
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════
# Storage
# ═══════════════════════════════════════════════════════════════════════

class TestStorage:
    def test_ensure_dirs(self, tmp_path):
        from unkillable.utils.storage import Storage
        s = Storage(tmp_path)
        s.ensure_dirs()
        assert (tmp_path / "clips").is_dir()
        assert (tmp_path / "thumbnails").is_dir()
        assert (tmp_path / "embeddings").is_dir()

    def test_save_and_load_events(self, tmp_path):
        from unkillable.utils.storage import Storage
        s = Storage(tmp_path)
        s.ensure_dirs()
        s.save_event({"camera": "cam1", "label": "person"})
        s.save_event({"camera": "cam2", "label": "car"})
        events = s.load_events(limit=10)
        assert len(events) == 2
        assert events[0]["camera"] == "cam1"
        assert events[1]["label"] == "car"

    def test_load_events_empty(self, tmp_path):
        from unkillable.utils.storage import Storage
        s = Storage(tmp_path)
        assert s.load_events() == []

    def test_load_events_limit(self, tmp_path):
        from unkillable.utils.storage import Storage
        s = Storage(tmp_path)
        s.ensure_dirs()
        for i in range(5):
            s.save_event({"i": i})
        assert len(s.load_events(limit=2)) == 2

    def test_disk_usage(self, tmp_path):
        from unkillable.utils.storage import Storage
        s = Storage(tmp_path)
        usage = s.disk_usage()
        assert "total_gb" in usage
        assert usage["total_gb"] > 0

    def test_cleanup_old(self, tmp_path):
        from unkillable.utils.storage import Storage
        s = Storage(tmp_path)
        s.ensure_dirs()
        # Create a file
        f = tmp_path / "clips" / "old.mp4"
        f.write_bytes(b"test")
        removed = s.cleanup_old(days=0)  # 0 days = remove everything
        assert removed >= 1
        assert not f.exists()


# ═══════════════════════════════════════════════════════════════════════
# Semantic Search
# ═══════════════════════════════════════════════════════════════════════

class TestSemanticSearch:
    def test_hash_embedding_deterministic(self):
        from unkillable.semantic.search import _hash_embedding
        v1 = _hash_embedding("hello world")
        v2 = _hash_embedding("hello world")
        assert v1 == v2

    def test_hash_embedding_dimension(self):
        from unkillable.semantic.search import _hash_embedding
        v = _hash_embedding("test", dim=512)
        assert len(v) == 512

    def test_hash_embedding_normalized(self):
        from unkillable.semantic.search import _hash_embedding
        v = _hash_embedding("test")
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 0.01

    def test_hash_embedding_different_inputs(self):
        from unkillable.semantic.search import _hash_embedding
        v1 = _hash_embedding("red car")
        v2 = _hash_embedding("blue dog")
        assert v1 != v2

    def test_cosine_similarity(self):
        from unkillable.semantic.search import cosine
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert abs(cosine(a, b) - 1.0) < 0.001

    def test_cosine_orthogonal(self):
        from unkillable.semantic.search import cosine
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine(a, b)) < 0.001

    def test_index_and_search(self, tmp_path):
        from unkillable.semantic.search import SemanticSearch
        db = tmp_path / "db.jsonl"
        ss = SemanticSearch(db_path=db)
        ss.index("1", ss.embed_text("red car"), {"camera": "front"})
        ss.index("2", ss.embed_text("blue dog"), {"camera": "back"})
        results = ss.search("red car", top_k=1)
        assert len(results) == 1
        assert results[0].id == "1"
        assert results[0].score > 0

    def test_search_empty_db(self, tmp_path):
        from unkillable.semantic.search import SemanticSearch
        db = tmp_path / "missing.jsonl"
        ss = SemanticSearch(db_path=db)
        results = ss.search("anything")
        assert results == []

    def test_embed_text_empty_raises(self):
        from unkillable.semantic.search import SemanticSearch
        ss = SemanticSearch()
        with pytest.raises(ValueError):
            ss.embed_text("")

    def test_embed_image_missing_raises(self):
        from unkillable.semantic.search import SemanticSearch
        ss = SemanticSearch()
        with pytest.raises(FileNotFoundError):
            ss.embed_image("/nonexistent/path.jpg")


# ═══════════════════════════════════════════════════════════════════════
# Motion Engine
# ═══════════════════════════════════════════════════════════════════════

class TestMotionEngine:
    def test_dummy_motion(self):
        from unkillable.motion.engine import MotionEngine
        m = MotionEngine()
        events = m._dummy_motion("test_src")
        assert len(events) == 1
        assert events[0].score == 0.5

    def test_parse_metadata(self):
        from unkillable.motion.engine import MotionEngine
        m = MotionEngine(threshold=0.01)
        stderr = (
            "[Parsed_select_2 @ 0x...] lavfi.scene_score=0.05\n"
            "[Parsed_select_2 @ 0x...] lavfi.scene_score=0.01\n"
            "[Parsed_select_2 @ 0x...] lavfi.scene_score=0.15\n"
        )
        events = m._parse_metadata(stderr)
        assert len(events) == 2  # 0.05 and 0.15 pass threshold 0.01, but 0.01 doesn't

    def test_parse_metadata_empty(self):
        from unkillable.motion.engine import MotionEngine
        m = MotionEngine()
        assert m._parse_metadata("") == []
        assert m._parse_metadata("no motion data") == []

    def test_threshold_filtering(self):
        from unkillable.motion.engine import MotionEngine
        m = MotionEngine(threshold=0.5)
        stderr = "lavfi.scene_score=0.3\nlavfi.scene_score=0.6\nlavfi.scene_score=0.49\n"
        events = m._parse_metadata(stderr)
        assert len(events) == 1
        assert events[0].score == 0.6

    def test_motion_filter_cmd(self):
        from unkillable.motion.engine import MotionEngine
        m = MotionEngine(threshold=0.03)
        cmd = m.ffmpeg_motion_filter_cmd("input.mp4", "output.mp4")
        assert "-vf" in cmd
        assert "select='gt(scene,0.03)'" in cmd[cmd.index("-vf") + 1]

    def test_detect_fallback_when_no_ffmpeg(self):
        from unkillable.motion.engine import MotionEngine
        m = MotionEngine()
        m.ffmpeg = MagicMock()
        m.ffmpeg.is_available.return_value = False
        events = m.detect_via_ffmpeg("rtsp://fake", duration=5)
        assert len(events) == 1  # dummy fallback


# ═══════════════════════════════════════════════════════════════════════
# Detector
# ═══════════════════════════════════════════════════════════════════════

class TestDetector:
    def test_invalid_labels_raises(self):
        from unkillable.detection.detector import Detector, DetectorError
        with pytest.raises(DetectorError):
            Detector(track_labels=["invalid_label_xyz"])

    def test_heuristic_returns_empty(self, tmp_path):
        from unkillable.detection.detector import Detector
        # Create a tiny valid image file
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG header
        d = Detector()
        d._yolo = None  # force heuristic mode
        results = d.detect(img)
        assert results == []

    def test_filter_by_label(self):
        from unkillable.detection.detector import Detector, Detection
        d = Detector(track_labels=["person", "car"])
        detections = [
            Detection(label="person", confidence=0.9, bbox=(0, 0, 100, 100)),
            Detection(label="dog", confidence=0.8, bbox=(0, 0, 50, 50)),
            Detection(label="car", confidence=0.7, bbox=(0, 0, 200, 200)),
        ]
        filtered = d.filter_by_label(detections)
        assert len(filtered) == 2
        assert {f.label for f in filtered} == {"person", "car"}

    def test_filter_by_threshold(self):
        from unkillable.detection.detector import Detector, Detection
        d = Detector(track_labels=["person"], threshold=0.8)
        detections = [
            Detection(label="person", confidence=0.9, bbox=(0, 0, 100, 100)),
            Detection(label="person", confidence=0.5, bbox=(0, 0, 50, 50)),
        ]
        filtered = d.filter_by_label(detections)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9


# ═══════════════════════════════════════════════════════════════════════
# QR Backup / Restore
# ═══════════════════════════════════════════════════════════════════════

class TestQRBackup:
    def test_create_archive(self, tmp_path):
        from unkillable.backup.qr_backup import QRBackup
        src = tmp_path / "hello.txt"
        src.write_text("hello unkillable")
        out = tmp_path / "out.tar.gz"
        qb = QRBackup()
        result = qb.create_archive([src], out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_create_archive_missing_source(self, tmp_path):
        from unkillable.backup.qr_backup import QRBackup
        out = tmp_path / "out.tar.gz"
        qb = QRBackup()
        result = qb.create_archive([tmp_path / "nonexistent.txt"], out)
        # Should not crash — missing files are skipped
        assert result.exists()

    def test_encode_to_qr_pdf(self, tmp_path):
        from unkillable.backup.qr_backup import QRBackup
        src = tmp_path / "data.txt"
        src.write_text("test data for QR encoding")
        archive = tmp_path / "archive.tar.gz"
        pdf = tmp_path / "backup.pdf"
        qb = QRBackup()
        qb.create_archive([src], archive)
        result = qb.encode_to_qr_pdf(archive, pdf)
        assert result.exists()
        assert result.suffix == ".pdf"
        assert result.stat().st_size > 1000

    def test_qr_chunk_size(self):
        from unkillable.backup.qr_backup import QRBackup
        qb = QRBackup(chunk_size=2953)
        assert qb.chunk_size == 2953

    def test_qr_redundancy(self):
        from unkillable.backup.qr_backup import QRBackup
        qb = QRBackup(redundancy=0.3)
        assert qb.redundancy == 0.3

    def test_encode_nonexistent_archive_raises(self, tmp_path):
        from unkillable.backup.qr_backup import QRBackup, QRBackupError
        qb = QRBackup()
        with pytest.raises(QRBackupError):
            qb.encode_to_qr_pdf(tmp_path / "nope.tar.gz", tmp_path / "out.pdf")


class TestQRRestore:
    def test_decode_empty_raises(self, tmp_path):
        from unkillable.backup.qr_restore import QRRestore, QRRestoreError
        qr = QRRestore()
        with pytest.raises(QRRestoreError):
            qr.decode_from_images([], tmp_path / "out.tar.gz")

    def test_extract_archive(self, tmp_path):
        from unkillable.backup.qr_backup import QRBackup
        from unkillable.backup.qr_restore import QRRestore
        # Create a test archive
        src = tmp_path / "hello.txt"
        src.write_text("roundtrip test")
        archive = tmp_path / "test.tar.gz"
        QRBackup().create_archive([src], archive)
        # Extract it
        dest = tmp_path / "restored"
        qr = QRRestore()
        result = qr.extract_archive(archive, dest)
        assert result.is_dir()
        assert (result / "hello.txt").read_text() == "roundtrip test"


# ═══════════════════════════════════════════════════════════════════════
# FFmpeg Wrapper
# ═══════════════════════════════════════════════════════════════════════

class TestFFmpegWrapper:
    def test_is_available_when_binary_exists(self):
        from unkillable.utils.ffmpeg import FFmpegWrapper
        fw = FFmpegWrapper(binary="/bin/sh")  # sh always exists
        assert fw.is_available()

    def test_is_not_available_when_missing(self):
        from unkillable.utils.ffmpeg import FFmpegWrapper
        fw = FFmpegWrapper(binary="/nonexistent/ffmpeg")
        assert not fw.is_available()

    def test_run_raises_on_missing_binary(self):
        from unkillable.utils.ffmpeg import FFmpegWrapper, FFmpegError
        fw = FFmpegWrapper(binary="/nonexistent/ffmpeg")
        with pytest.raises(FFmpegError):
            fw.run(["-version"])

    def test_run_raises_on_nonzero_exit(self):
        from unkillable.utils.ffmpeg import FFmpegWrapper, FFmpegError
        fw = FFmpegWrapper(binary="/bin/sh")
        with pytest.raises(FFmpegError):
            fw.run(["-c", "exit 1"])


# ═══════════════════════════════════════════════════════════════════════
# GenAI Describer
# ═══════════════════════════════════════════════════════════════════════

class TestDescriber:
    def test_describe_missing_file_raises(self):
        from unkillable.genai.describer import Describer, DescriberError
        d = Describer()
        with pytest.raises(DescriberError):
            d.describe("/nonexistent/image.jpg")

    def test_fallback_description(self):
        from unkillable.genai.describer import Describer
        d = Describer()
        text = d._fallback_description(Path("test.jpg"))
        assert "test.jpg" in text
        assert "fallback" in text.lower()

    def test_unknown_provider_raises(self):
        from unkillable.genai.describer import Describer, DescriberError
        d = Describer(provider="nonexistent")
        with pytest.raises(DescriberError, match="Unknown provider"):
            d.describe(Path("/dev/null"))


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

class TestCLI:
    def test_cli_help(self):
        from typer.testing import CliRunner
        from unkillable.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Unkillable" in result.output

    def test_storage_info_command(self, tmp_path):
        from typer.testing import CliRunner
        from unkillable.cli import app
        runner = CliRunner()
        storage = tmp_path / "storage"
        storage.mkdir()
        (storage / "clips").mkdir()
        (storage / "thumbnails").mkdir()
        (storage / "embeddings").mkdir()
        result = runner.invoke(app, ["storage-info", "--root", str(storage)])
        assert result.exit_code == 0
