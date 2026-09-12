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

    def test_counts(self):
        from unkillable.detection.detector import Detector, Detection
        detections = [
            Detection(label="person", confidence=0.9, bbox=(0, 0, 10, 10)),
            Detection(label="person", confidence=0.7, bbox=(0, 0, 10, 10)),
            Detection(label="car", confidence=0.8, bbox=(0, 0, 10, 10)),
        ]
        assert Detector.counts(detections) == {"person": 2, "car": 1}
        assert Detector.counts([]) == {}
        assert Detector.counts(None) == {}

    def test_detection_to_dict(self):
        from unkillable.detection.detector import Detection
        d = Detection(label="person", confidence=0.9234, bbox=(1.0, 2.0, 101.0, 202.0),
                      color_hex="#ff0000", color_name="red")
        out = d.to_dict()
        assert out["label"] == "person"
        assert out["confidence"] == 0.9234
        assert out["area"] == 100.0 * 200.0
        assert out["color_hex"] == "#ff0000"
        assert out["color_name"] == "red"

    def test_detection_area(self):
        from unkillable.detection.detector import Detection
        assert Detection(label="x", confidence=0.5, bbox=(0, 0, 10, 20)).area == 200.0
        assert Detection(label="x", confidence=0.5, bbox=(10, 20, 5, 5)).area == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Color extraction
# ═══════════════════════════════════════════════════════════════════════

class TestColorExtraction:
    def _solid(self, bgr, size=100):
        import numpy as np
        return np.full((size, size, 3), bgr, dtype=np.uint8)

    def _pure(self, color):
        rgb = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
               "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
               "orange": (255, 165, 0), "purple": (128, 0, 128)}
        r, g, b = rgb[color]
        return self._solid((b, g, r))

    def test_chromatic_colors(self):
        from unkillable.detection.color import dominant_color
        for color in ("red", "green", "blue", "yellow", "cyan", "orange", "purple"):
            name, hexv = dominant_color(self._pure(color), (0, 0, 100, 100))
            assert name == color, (color, name)
            assert hexv and hexv.startswith("#")

    def test_achromatic(self):
        from unkillable.detection.color import dominant_color
        name, hexv = dominant_color(self._solid((0, 0, 0)), (0, 0, 100, 100))
        assert name == "black"
        name, hexv = dominant_color(self._solid((255, 255, 255)), (0, 0, 100, 100))
        assert name == "white"
        name, hexv = dominant_color(self._solid((128, 128, 128)), (0, 0, 100, 100))
        assert name == "gray"
        assert hexv is not None and hexv.startswith("#")

    def test_bbox_selects_region(self):
        from unkillable.detection.color import dominant_color
        img = self._solid((255, 0, 0)).copy()  # blue full
        img[:, :50] = (0, 0, 255)  # left half red
        name, _ = dominant_color(img, (0, 0, 45, 100))
        assert name == "red"
        name, _ = dominant_color(img, (55, 0, 100, 100))
        assert name == "blue"

    def test_mixed_dominant_wins(self):
        from unkillable.detection.color import dominant_color
        img = self._solid((255, 0, 0)).copy()  # blue full
        img[:, 80:] = (0, 0, 255)  # 80% blue, 20% red
        name, _ = dominant_color(img, (0, 0, 100, 100))
        assert name == "blue"

    def test_tiny_bbox_unknown(self):
        from unkillable.detection.color import dominant_color
        assert dominant_color(self._pure("red"), (0, 0, 2, 2)) == ("unknown", None)

    def test_bbox_clipped_to_image(self):
        from unkillable.detection.color import dominant_color
        img = self._pure("blue")
        name, _ = dominant_color(img, (-50, -50, 500, 500))
        assert name == "blue"


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

    def test_index_command_help(self):
        from typer.testing import CliRunner
        from unkillable.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["index", "--help"])
        assert result.exit_code == 0
        assert "Index" in result.output or "index" in result.output.lower()

    def test_query_command_help(self):
        from typer.testing import CliRunner
        from unkillable.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0
        assert "Query" in result.output or "query" in result.output.lower()

    def test_serve_command_help(self):
        from typer.testing import CliRunner
        from unkillable.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "serve" in result.output.lower() or "server" in result.output.lower()


# ═══════════════════════════════════════════════════════════════════════
# IndexEntry Model
# ═══════════════════════════════════════════════════════════════════════

class TestIndexEntry:
    def test_create_entry(self):
        from unkillable.index.models import IndexEntry
        e = IndexEntry(
            id="clip1_abc123",
            timestamp=42.5,
            source_clip="/path/to/clip.mp4",
            thumbnail_path="/path/to/thumb.jpg",
            motion_score=0.8,
            labels=["person", "car"],
            embedding=[0.1] * 512,
            metadata={"camera": "front"},
        )
        assert e.id == "clip1_abc123"
        assert e.timestamp == 42.5
        assert len(e.labels) == 2
        assert len(e.embedding) == 512

    def test_to_dict_roundtrip(self):
        from unkillable.index.models import IndexEntry
        e = IndexEntry(
            id="test",
            timestamp=60.0,
            source_clip="clip.mp4",
            thumbnail_path="thumb.jpg",
            labels=["dog"],
            embedding=[0.5, 0.3],
        )
        d = e.to_dict()
        e2 = IndexEntry.from_dict(d)
        assert e2.id == e.id
        assert e2.timestamp == e.timestamp
        assert e2.labels == e.labels
        assert e2.embedding == e.embedding

    def test_timestamp_str(self):
        from unkillable.index.models import IndexEntry
        e = IndexEntry(id="a", timestamp=125.0, source_clip="c", thumbnail_path="t")
        assert e.timestamp_str == "02:05"

    def test_timestamp_str_zero(self):
        from unkillable.index.models import IndexEntry
        e = IndexEntry(id="a", timestamp=0.0, source_clip="c", thumbnail_path="t")
        assert e.timestamp_str == "00:00"

    def test_defaults(self):
        from unkillable.index.models import IndexEntry
        e = IndexEntry(id="x", timestamp=1.0, source_clip="c", thumbnail_path="t")
        assert e.labels == []
        assert e.embedding == []
        assert e.motion_score == 0.0
        assert e.metadata == {}

    def test_thumbnail_path_obj(self):
        from unkillable.index.models import IndexEntry
        e = IndexEntry(id="x", timestamp=1.0, source_clip="c", thumbnail_path="/tmp/thumb.jpg")
        assert e.thumbnail_path_obj == Path("/tmp/thumb.jpg")

    def test_from_dict_extra_keys_ignored(self):
        from unkillable.index.models import IndexEntry
        d = {"id": "x", "timestamp": 1.0, "source_clip": "c", "thumbnail_path": "t", "extra": True}
        e = IndexEntry.from_dict(d)
        assert e.id == "x"


# ═══════════════════════════════════════════════════════════════════════
# IndexEngine
# ═══════════════════════════════════════════════════════════════════════

class TestIndexEngine:
    def test_ensure_dirs(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        assert (tmp_path / "index").is_dir()
        assert (tmp_path / "index" / "thumbnails").is_dir()

    def test_index_count_empty(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        assert ie.index_count() == 0

    def test_load_entries_empty(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        assert ie.load_entries() == []

    def test_get_entry_missing(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        assert ie.get_entry("nonexistent") is None

    def test_index_clip_missing_file(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        entries = ie.index_clip(tmp_path / "nonexistent.mp4")
        assert entries == []

    def test_save_and_load_entries(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        entry = IndexEntry(
            id="test_001",
            timestamp=10.0,
            source_clip="clip.mp4",
            thumbnail_path="thumb.jpg",
            labels=["person"],
            embedding=[0.5] * 10,
        )
        ie._save_entries([entry])
        loaded = ie.load_entries()
        assert len(loaded) == 1
        assert loaded[0].id == "test_001"
        assert loaded[0].labels == ["person"]

    def test_save_multiple_entries(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        entries = [
            IndexEntry(id=f"e{i}", timestamp=float(i), source_clip="c", thumbnail_path="t")
            for i in range(5)
        ]
        ie._save_entries(entries)
        loaded = ie.load_entries()
        assert len(loaded) == 5

    def test_index_count_after_save(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        ie._save_entries([IndexEntry(id="x", timestamp=1.0, source_clip="c", thumbnail_path="t")])
        assert ie.index_count() == 1

    def test_parse_scene_timestamps_empty(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        assert ie._parse_scene_timestamps(tmp_path / "missing.log") == []

    def test_parse_scene_timestamps_with_data(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        log_file = tmp_path / "test.log"
        log_file.write_text("pts_time: 1.5\npts_time: 3.2\npts_time: 7.8\n")
        ts = ie._parse_scene_timestamps(log_file)
        assert len(ts) == 3
        assert ts[0] == 1.5
        assert ts[2] == 7.8

    def test_process_keyframe_missing_file(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        result = ie._process_keyframe(Path("clip.mp4"), 1.0, tmp_path / "missing.jpg", "cam")
        assert result is None

    def test_process_keyframe_with_image(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        # Create a tiny image
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        result = ie._process_keyframe(Path("clip.mp4"), 5.0, img, "front")
        assert result is not None
        assert result.timestamp == 5.0
        assert Path(result.source_clip).name == "clip.mp4"
        assert Path(result.source_clip).is_absolute()
        assert "front" in result.metadata["camera"]
        assert len(result.embedding) == 512  # hash embedding dimension

    def test_fallback_extract(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        # Fallback extracts frames even without scene detection
        # With no FFmpeg, it should return empty
        ie.ffmpeg = MagicMock()
        ie.ffmpeg.is_available.return_value = False
        result = ie._fallback_extract(Path("fake.mp4"))
        # Without real FFmpeg, thumbnails won't be created
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════════════
# QueryEngine
# ═══════════════════════════════════════════════════════════════════════

class TestQueryEngine:
    def test_query_empty_index(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.query.engine import QueryEngine
        ie = IndexEngine(storage_root=tmp_path)
        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        results = qe.query("red car")
        assert results == []

    def test_query_empty_string(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.query.engine import QueryEngine
        ie = IndexEngine(storage_root=tmp_path)
        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        results = qe.query("")
        assert results == []

    def test_query_finds_matching_frame(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryEngine
        from unkillable.semantic.search import SemanticSearch

        ie = IndexEngine(storage_root=tmp_path)
        ss = SemanticSearch(db_path=tmp_path / "index" / "entries.jsonl")

        # Manually index entries with embeddings
        ie.ensure_dirs()
        vec_red = ss.embed_text("red car")
        vec_blue = ss.embed_text("blue dog")
        entries = [
            IndexEntry(id="e1", timestamp=10.0, source_clip="c", thumbnail_path="t1", embedding=vec_red, labels=["car"]),
            IndexEntry(id="e2", timestamp=20.0, source_clip="c", thumbnail_path="t2", embedding=vec_blue, labels=["dog"]),
        ]
        ie._save_entries(entries)

        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        results = qe.query("red car", generate_clips=False)
        assert len(results) > 0
        assert results[0].entry.id == "e1"
        assert results[0].score > 0

    def test_query_with_label_filter(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryEngine
        from unkillable.semantic.search import SemanticSearch

        ie = IndexEngine(storage_root=tmp_path)
        ss = SemanticSearch(db_path=tmp_path / "index" / "entries.jsonl")
        ie.ensure_dirs()
        vec = ss.embed_text("vehicle")
        entries = [
            IndexEntry(id="e1", timestamp=1.0, source_clip="c", thumbnail_path="t1", embedding=vec, labels=["car"]),
            IndexEntry(id="e2", timestamp=2.0, source_clip="c", thumbnail_path="t2", embedding=vec, labels=["dog"]),
        ]
        ie._save_entries(entries)

        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        results = qe.query("vehicle", labels=["car"], generate_clips=False)
        # Car label should boost score for e1
        assert len(results) == 2
        # e1 should rank higher due to label match
        car_result = next(r for r in results if r.entry.id == "e1")
        dog_result = next(r for r in results if r.entry.id == "e2")
        assert car_result.score >= dog_result.score

    def test_query_by_labels(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryEngine

        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        entries = [
            IndexEntry(id="e1", timestamp=1.0, source_clip="c", thumbnail_path="t1", labels=["person", "car"]),
            IndexEntry(id="e2", timestamp=2.0, source_clip="c", thumbnail_path="t2", labels=["dog"]),
            IndexEntry(id="e3", timestamp=3.0, source_clip="c", thumbnail_path="t3", labels=["person"]),
        ]
        ie._save_entries(entries)

        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        results = qe.query_by_labels(["person"])
        assert len(results) == 2
        ids = {r.entry.id for r in results}
        assert "e1" in ids
        assert "e3" in ids

    def test_query_by_time(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryEngine

        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        entries = [
            IndexEntry(id="e1", timestamp=5.0, source_clip="c", thumbnail_path="t1"),
            IndexEntry(id="e2", timestamp=15.0, source_clip="c", thumbnail_path="t2"),
            IndexEntry(id="e3", timestamp=25.0, source_clip="c", thumbnail_path="t3"),
        ]
        ie._save_entries(entries)

        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        results = qe.query_by_time(10.0, 20.0)
        assert len(results) == 1
        assert results[0].entry.id == "e2"

    def test_query_result_to_dict(self, tmp_path):
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryResult

        entry = IndexEntry(id="x", timestamp=5.0, source_clip="c", thumbnail_path="t", labels=["car"])
        result = QueryResult(entry=entry, score=0.85, clip_path="/tmp/clip.mp4")
        d = result.to_dict()
        assert d["id"] == "x"
        assert d["score"] == 0.85
        assert d["clip_path"] == "/tmp/clip.mp4"
        assert d["labels"] == ["car"]
        assert d["timestamp_str"] == "00:05"

    def test_generate_clip_missing_source(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryEngine

        ie = IndexEngine(storage_root=tmp_path)
        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        entry = IndexEntry(id="x", timestamp=5.0, source_clip="/nonexistent/clip.mp4", thumbnail_path="t")
        result = qe._generate_clip(entry)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# Flask API
# ═══════════════════════════════════════════════════════════════════════

class TestAPI:
    def test_search_missing_query(self, tmp_path):
        from unkillable.api import app, init_engines
        init_engines(storage_root=tmp_path)
        with app.test_client() as client:
            r = client.get("/api/search")
            assert r.status_code == 400
            assert "error" in r.get_json()

    def test_search_empty_query(self, tmp_path):
        from unkillable.api import app, init_engines
        init_engines(storage_root=tmp_path)
        with app.test_client() as client:
            r = client.get("/api/search?q=")
            assert r.status_code == 400

    def test_search_returns_results(self, tmp_path):
        from unkillable.api import app, init_engines
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        from unkillable.semantic.search import SemanticSearch

        init_engines(storage_root=tmp_path)
        ie = IndexEngine(storage_root=tmp_path)
        ss = SemanticSearch(db_path=tmp_path / "index" / "entries.jsonl")
        ie.ensure_dirs()
        vec = ss.embed_text("red car")
        ie._save_entries([
            IndexEntry(id="e1", timestamp=10.0, source_clip="c", thumbnail_path="t", embedding=vec, labels=["car"]),
        ])

        with app.test_client() as client:
            r = client.get("/api/search?q=red+car&clips=false")
            assert r.status_code == 200
            data = r.get_json()
            assert data["count"] >= 1
            assert data["results"][0]["id"] == "e1"

    def test_search_labels_endpoint(self, tmp_path):
        from unkillable.api import app, init_engines
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry

        init_engines(storage_root=tmp_path)
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        ie._save_entries([
            IndexEntry(id="e1", timestamp=1.0, source_clip="c", thumbnail_path="t", labels=["person"]),
            IndexEntry(id="e2", timestamp=2.0, source_clip="c", thumbnail_path="t", labels=["car"]),
        ])

        with app.test_client() as client:
            r = client.get("/api/search/labels?labels=person")
            assert r.status_code == 200
            data = r.get_json()
            assert data["count"] == 1
            assert data["results"][0]["id"] == "e1"

    def test_search_labels_missing(self, tmp_path):
        from unkillable.api import app, init_engines
        init_engines(storage_root=tmp_path)
        with app.test_client() as client:
            r = client.get("/api/search/labels")
            assert r.status_code == 400

    def test_entries_endpoint(self, tmp_path):
        from unkillable.api import app, init_engines
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry

        init_engines(storage_root=tmp_path)
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        ie._save_entries([
            IndexEntry(id=f"e{i}", timestamp=float(i), source_clip="c", thumbnail_path="t")
            for i in range(3)
        ])

        with app.test_client() as client:
            r = client.get("/api/entries?limit=2")
            assert r.status_code == 200
            data = r.get_json()
            assert data["total"] == 3
            assert data["count"] == 2

    def test_stats_endpoint(self, tmp_path):
        from unkillable.api import app, init_engines
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry

        init_engines(storage_root=tmp_path)
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        ie._save_entries([
            IndexEntry(id="e1", timestamp=1.0, source_clip="clip.mp4", thumbnail_path="t", labels=["person", "car"]),
            IndexEntry(id="e2", timestamp=2.0, source_clip="clip.mp4", thumbnail_path="t", labels=["person"]),
        ])

        with app.test_client() as client:
            r = client.get("/api/stats")
            assert r.status_code == 200
            data = r.get_json()
            assert data["total_frames"] == 2
            assert data["total_clips"] == 1
            assert data["label_counts"]["person"] == 2
            assert data["label_counts"]["car"] == 1

    def test_frame_endpoint_not_found(self, tmp_path):
        from unkillable.api import app, init_engines
        init_engines(storage_root=tmp_path)
        with app.test_client() as client:
            r = client.get("/api/frame/nonexistent")
            assert r.status_code == 404

    def test_clip_endpoint_not_found(self, tmp_path):
        from unkillable.api import app, init_engines
        init_engines(storage_root=tmp_path)
        with app.test_client() as client:
            r = client.get("/api/clip/nonexistent")
            assert r.status_code == 404

    def test_describe_endpoint_not_found(self, tmp_path):
        from unkillable.api import app, init_engines
        init_engines(storage_root=tmp_path)
        with app.test_client() as client:
            r = client.post("/api/describe/nonexistent", json={})
            assert r.status_code == 404

    def test_dashboard_serves_html(self, tmp_path):
        from unkillable.api import app, init_engines
        init_engines(storage_root=tmp_path)
        with app.test_client() as client:
            r = client.get("/")
            assert r.status_code == 200
            assert b"Search" in r.data
            assert b"Live" in r.data


# ═══════════════════════════════════════════════════════════════════════
# IndexEntry Model
# ═══════════════════════════════════════════════════════════════════════

class TestIndexEntry:
    def test_create_entry(self):
        from unkillable.index.models import IndexEntry
        e = IndexEntry(
            id="clip1_abc123",
            timestamp=42.5,
            source_clip="/path/to/clip.mp4",
            thumbnail_path="/path/to/thumb.jpg",
            motion_score=0.8,
            labels=["person", "car"],
            embedding=[0.1] * 512,
            metadata={"camera": "front"},
        )
        assert e.id == "clip1_abc123"
        assert e.timestamp == 42.5
        assert len(e.labels) == 2
        assert len(e.embedding) == 512

    def test_to_dict_roundtrip(self):
        from unkillable.index.models import IndexEntry
        e = IndexEntry(
            id="test",
            timestamp=60.0,
            source_clip="clip.mp4",
            thumbnail_path="thumb.jpg",
            labels=["dog"],
            embedding=[0.5, 0.3],
        )
        d = e.to_dict()
        e2 = IndexEntry.from_dict(d)
        assert e2.id == e.id
        assert e2.timestamp == e.timestamp
        assert e2.labels == e.labels
        assert e2.embedding == e.embedding

    def test_timestamp_str(self):
        from unkillable.index.models import IndexEntry
        e = IndexEntry(id="a", timestamp=125.0, source_clip="c", thumbnail_path="t")
        assert e.timestamp_str == "02:05"

    def test_timestamp_str_zero(self):
        from unkillable.index.models import IndexEntry
        e = IndexEntry(id="a", timestamp=0.0, source_clip="c", thumbnail_path="t")
        assert e.timestamp_str == "00:00"

    def test_defaults(self):
        from unkillable.index.models import IndexEntry
        e = IndexEntry(id="x", timestamp=1.0, source_clip="c", thumbnail_path="t")
        assert e.labels == []
        assert e.embedding == []
        assert e.motion_score == 0.0
        assert e.metadata == {}

    def test_thumbnail_path_obj(self):
        from unkillable.index.models import IndexEntry
        e = IndexEntry(id="x", timestamp=1.0, source_clip="c", thumbnail_path="/tmp/thumb.jpg")
        assert e.thumbnail_path_obj == Path("/tmp/thumb.jpg")

    def test_from_dict_extra_keys_ignored(self):
        from unkillable.index.models import IndexEntry
        d = {"id": "x", "timestamp": 1.0, "source_clip": "c", "thumbnail_path": "t", "extra": True}
        e = IndexEntry.from_dict(d)
        assert e.id == "x"


# ═══════════════════════════════════════════════════════════════════════
# IndexEngine
# ═══════════════════════════════════════════════════════════════════════

class TestIndexEngine:
    def test_ensure_dirs(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        assert (tmp_path / "index").is_dir()
        assert (tmp_path / "index" / "thumbnails").is_dir()

    def test_index_count_empty(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        assert ie.index_count() == 0

    def test_load_entries_empty(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        assert ie.load_entries() == []

    def test_get_entry_missing(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        assert ie.get_entry("nonexistent") is None

    def test_index_clip_missing_file(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        entries = ie.index_clip(tmp_path / "nonexistent.mp4")
        assert entries == []

    def test_save_and_load_entries(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        entry = IndexEntry(
            id="test_001",
            timestamp=10.0,
            source_clip="clip.mp4",
            thumbnail_path="thumb.jpg",
            labels=["person"],
            embedding=[0.5] * 10,
        )
        ie._save_entries([entry])
        loaded = ie.load_entries()
        assert len(loaded) == 1
        assert loaded[0].id == "test_001"
        assert loaded[0].labels == ["person"]

    def test_save_multiple_entries(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        entries = [
            IndexEntry(id=f"e{i}", timestamp=float(i), source_clip="c", thumbnail_path="t")
            for i in range(5)
        ]
        ie._save_entries(entries)
        loaded = ie.load_entries()
        assert len(loaded) == 5

    def test_index_count_after_save(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        ie._save_entries([IndexEntry(id="x", timestamp=1.0, source_clip="c", thumbnail_path="t")])
        assert ie.index_count() == 1

    def test_parse_scene_timestamps_empty(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        assert ie._parse_scene_timestamps(tmp_path / "missing.log") == []

    def test_parse_scene_timestamps_with_data(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        log_file = tmp_path / "test.log"
        log_file.write_text("pts_time: 1.5\npts_time: 3.2\npts_time: 7.8\n")
        ts = ie._parse_scene_timestamps(log_file)
        assert len(ts) == 3
        assert ts[0] == 1.5
        assert ts[2] == 7.8

    def test_process_keyframe_missing_file(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        result = ie._process_keyframe(Path("clip.mp4"), 1.0, tmp_path / "missing.jpg", "cam")
        assert result is None

    def test_process_keyframe_with_image(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        # Create a tiny image
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        result = ie._process_keyframe(Path("clip.mp4"), 5.0, img, "front")
        assert result is not None
        assert result.timestamp == 5.0
        assert Path(result.source_clip).name == "clip.mp4"
        assert Path(result.source_clip).is_absolute()
        assert "front" in result.metadata["camera"]
        assert len(result.embedding) == 512  # hash embedding dimension

    def test_fallback_extract(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        # Fallback extracts frames even without scene detection
        # With no FFmpeg, it should return empty
        ie.ffmpeg = MagicMock()
        ie.ffmpeg.is_available.return_value = False
        result = ie._fallback_extract(Path("fake.mp4"))
        # Without real FFmpeg, thumbnails won't be created
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════════════
# QueryEngine
# ═══════════════════════════════════════════════════════════════════════

class TestQueryEngine:
    def test_query_empty_index(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.query.engine import QueryEngine
        ie = IndexEngine(storage_root=tmp_path)
        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        results = qe.query("red car")
        assert results == []

    def test_query_empty_string(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.query.engine import QueryEngine
        ie = IndexEngine(storage_root=tmp_path)
        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        results = qe.query("")
        assert results == []

    def test_query_finds_matching_frame(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryEngine
        from unkillable.semantic.search import SemanticSearch

        ie = IndexEngine(storage_root=tmp_path)
        ss = SemanticSearch(db_path=tmp_path / "index" / "entries.jsonl")

        # Manually index entries with embeddings
        ie.ensure_dirs()
        vec_red = ss.embed_text("red car")
        vec_blue = ss.embed_text("blue dog")
        entries = [
            IndexEntry(id="e1", timestamp=10.0, source_clip="c", thumbnail_path="t1", embedding=vec_red, labels=["car"]),
            IndexEntry(id="e2", timestamp=20.0, source_clip="c", thumbnail_path="t2", embedding=vec_blue, labels=["dog"]),
        ]
        ie._save_entries(entries)

        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        results = qe.query("red car", generate_clips=False)
        assert len(results) > 0
        assert results[0].entry.id == "e1"
        assert results[0].score > 0

    def test_query_with_label_filter(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryEngine
        from unkillable.semantic.search import SemanticSearch

        ie = IndexEngine(storage_root=tmp_path)
        ss = SemanticSearch(db_path=tmp_path / "index" / "entries.jsonl")
        ie.ensure_dirs()
        vec = ss.embed_text("vehicle")
        entries = [
            IndexEntry(id="e1", timestamp=1.0, source_clip="c", thumbnail_path="t1", embedding=vec, labels=["car"]),
            IndexEntry(id="e2", timestamp=2.0, source_clip="c", thumbnail_path="t2", embedding=vec, labels=["dog"]),
        ]
        ie._save_entries(entries)

        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        results = qe.query("vehicle", labels=["car"], generate_clips=False)
        # Car label should boost score for e1
        assert len(results) == 2
        # e1 should rank higher due to label match
        car_result = next(r for r in results if r.entry.id == "e1")
        dog_result = next(r for r in results if r.entry.id == "e2")
        assert car_result.score >= dog_result.score

    def test_query_by_labels(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryEngine

        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        entries = [
            IndexEntry(id="e1", timestamp=1.0, source_clip="c", thumbnail_path="t1", labels=["person", "car"]),
            IndexEntry(id="e2", timestamp=2.0, source_clip="c", thumbnail_path="t2", labels=["dog"]),
            IndexEntry(id="e3", timestamp=3.0, source_clip="c", thumbnail_path="t3", labels=["person"]),
        ]
        ie._save_entries(entries)

        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        results = qe.query_by_labels(["person"])
        assert len(results) == 2
        ids = {r.entry.id for r in results}
        assert "e1" in ids
        assert "e3" in ids

    def test_query_by_time(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryEngine

        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        entries = [
            IndexEntry(id="e1", timestamp=5.0, source_clip="c", thumbnail_path="t1"),
            IndexEntry(id="e2", timestamp=15.0, source_clip="c", thumbnail_path="t2"),
            IndexEntry(id="e3", timestamp=25.0, source_clip="c", thumbnail_path="t3"),
        ]
        ie._save_entries(entries)

        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        results = qe.query_by_time(10.0, 20.0)
        assert len(results) == 1
        assert results[0].entry.id == "e2"

    def test_query_result_to_dict(self, tmp_path):
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryResult

        entry = IndexEntry(id="x", timestamp=5.0, source_clip="c", thumbnail_path="t", labels=["car"])
        result = QueryResult(entry=entry, score=0.85, clip_path="/tmp/clip.mp4")
        d = result.to_dict()
        assert d["id"] == "x"
        assert d["score"] == 0.85
        assert d["clip_path"] == "/tmp/clip.mp4"
        assert d["labels"] == ["car"]
        assert d["timestamp_str"] == "00:05"

    def test_generate_clip_missing_source(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryEngine

        ie = IndexEngine(storage_root=tmp_path)
        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        entry = IndexEntry(id="x", timestamp=5.0, source_clip="/nonexistent/clip.mp4", thumbnail_path="t")
        result = qe._generate_clip(entry)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# Flask API
# ═══════════════════════════════════════════════════════════════════════

class TestAPI:
    def test_search_missing_query(self, tmp_path):
        from unkillable.api import app, init_engines
        init_engines(storage_root=tmp_path)
        with app.test_client() as client:
            r = client.get("/api/search")
            assert r.status_code == 400
            assert "error" in r.get_json()

    def test_search_empty_query(self, tmp_path):
        from unkillable.api import app, init_engines
        init_engines(storage_root=tmp_path)
        with app.test_client() as client:
            r = client.get("/api/search?q=")
            assert r.status_code == 400

    def test_search_returns_results(self, tmp_path):
        from unkillable.api import app, init_engines
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        from unkillable.semantic.search import SemanticSearch

        init_engines(storage_root=tmp_path)
        ie = IndexEngine(storage_root=tmp_path)
        ss = SemanticSearch(db_path=tmp_path / "index" / "entries.jsonl")
        ie.ensure_dirs()
        vec = ss.embed_text("red car")
        ie._save_entries([
            IndexEntry(id="e1", timestamp=10.0, source_clip="c", thumbnail_path="t", embedding=vec, labels=["car"]),
        ])

        with app.test_client() as client:
            r = client.get("/api/search?q=red+car&clips=false")
            assert r.status_code == 200
            data = r.get_json()
            assert data["count"] >= 1
            assert data["results"][0]["id"] == "e1"

    def test_search_labels_endpoint(self, tmp_path):
        from unkillable.api import app, init_engines
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry

        init_engines(storage_root=tmp_path)
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        ie._save_entries([
            IndexEntry(id="e1", timestamp=1.0, source_clip="c", thumbnail_path="t", labels=["person"]),
            IndexEntry(id="e2", timestamp=2.0, source_clip="c", thumbnail_path="t", labels=["car"]),
        ])

        with app.test_client() as client:
            r = client.get("/api/search/labels?labels=person")
            assert r.status_code == 200
            data = r.get_json()
            assert data["count"] == 1
            assert data["results"][0]["id"] == "e1"

    def test_search_labels_missing(self, tmp_path):
        from unkillable.api import app, init_engines
        init_engines(storage_root=tmp_path)
        with app.test_client() as client:
            r = client.get("/api/search/labels")
            assert r.status_code == 400

    def test_entries_endpoint(self, tmp_path):
        from unkillable.api import app, init_engines
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry

        init_engines(storage_root=tmp_path)
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        ie._save_entries([
            IndexEntry(id=f"e{i}", timestamp=float(i), source_clip="c", thumbnail_path="t")
            for i in range(3)
        ])

        with app.test_client() as client:
            r = client.get("/api/entries?limit=2")
            assert r.status_code == 200
            data = r.get_json()
            assert data["total"] == 3
            assert data["count"] == 2

    def test_stats_endpoint(self, tmp_path):
        from unkillable.api import app, init_engines
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry

        init_engines(storage_root=tmp_path)
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        ie._save_entries([
            IndexEntry(id="e1", timestamp=1.0, source_clip="clip.mp4", thumbnail_path="t", labels=["person", "car"]),
            IndexEntry(id="e2", timestamp=2.0, source_clip="clip.mp4", thumbnail_path="t", labels=["person"]),
        ])

        with app.test_client() as client:
            r = client.get("/api/stats")
            assert r.status_code == 200
            data = r.get_json()
            assert data["total_frames"] == 2
            assert data["total_clips"] == 1
            assert data["label_counts"]["person"] == 2
            assert data["label_counts"]["car"] == 1

    def test_frame_endpoint_not_found(self, tmp_path):
        from unkillable.api import app, init_engines
        init_engines(storage_root=tmp_path)
        with app.test_client() as client:
            r = client.get("/api/frame/nonexistent")
            assert r.status_code == 404

    def test_clip_endpoint_not_found(self, tmp_path):
        from unkillable.api import app, init_engines
        init_engines(storage_root=tmp_path)
        with app.test_client() as client:
            r = client.get("/api/clip/nonexistent")
            assert r.status_code == 404

    def test_describe_endpoint_not_found(self, tmp_path):
        from unkillable.api import app, init_engines
        init_engines(storage_root=tmp_path)
        with app.test_client() as client:
            r = client.post("/api/describe/nonexistent", json={})
            assert r.status_code == 404

    def test_dashboard_serves_html(self, tmp_path):
        from unkillable.api import app, init_engines
        init_engines(storage_root=tmp_path)
        with app.test_client() as client:
            r = client.get("/")
            assert r.status_code == 200
            assert b"Search" in r.data
            assert b"Live" in r.data


# ═══════════════════════════════════════════════════════════════════════
# Rich detections: objects / counts / colors
# ═══════════════════════════════════════════════════════════════════════

class TestClipCounts:
    def test_aggregates_per_clip(self):
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry
        entries = [
            IndexEntry(id="a1", timestamp=1.0, source_clip="a.mp4", thumbnail_path="t",
                       counts={"person": 2, "car": 1}),
            IndexEntry(id="a2", timestamp=2.0, source_clip="a.mp4", thumbnail_path="t",
                       counts={"person": 1}),
            IndexEntry(id="b1", timestamp=1.0, source_clip="b.mp4", thumbnail_path="t",
                       counts={"dog": 1}),
        ]
        totals = IndexEngine.clip_counts(entries)
        assert totals == {"a.mp4": {"person": 3, "car": 1}, "b.mp4": {"dog": 1}}

    def test_empty(self):
        from unkillable.index.engine import IndexEngine
        assert IndexEngine.clip_counts([]) == {}


class TestQueryResultRichFields:
    def test_to_dict_includes_objects_counts_colors(self):
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryResult
        entry = IndexEntry(
            id="x", timestamp=5.0, source_clip="c", thumbnail_path="t", labels=["person"],
            objects=[
                {"label": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10], "area": 100.0,
                 "color_hex": "#ff0000", "color_name": "red"},
                {"label": "person", "confidence": 0.7, "bbox": [0, 0, 5, 5], "area": 25.0,
                 "color_hex": "#0000ff", "color_name": "blue"},
            ],
            counts={"person": 2},
        )
        d = QueryResult(entry=entry, score=0.85, clip_path="/tmp/clip.mp4").to_dict()
        assert d["objects"] == entry.objects
        assert d["counts"] == {"person": 2}
        assert d["colors"] == ["blue", "red"]


class TestAPIRichFields:
    def test_entries_include_objects_counts_colors(self, tmp_path):
        from unkillable.api import app, init_engines
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry

        init_engines(storage_root=tmp_path)
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        ie._save_entries([
            IndexEntry(
                id="e1", timestamp=1.0, source_clip="c", thumbnail_path="t", labels=["person"],
                objects=[{"label": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10], "area": 100.0,
                          "color_hex": "#00ff00", "color_name": "green"}],
                counts={"person": 1},
            ),
        ])

        with app.test_client() as client:
            r = client.get("/api/entries")
            assert r.status_code == 200
            data = r.get_json()
            e = data["entries"][0]
            assert e["objects"][0]["label"] == "person"
            assert e["counts"] == {"person": 1}
            assert e["colors"] == ["green"]

    def test_stats_merges_counts_and_legacy_labels(self, tmp_path):
        from unkillable.api import app, init_engines
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry

        init_engines(storage_root=tmp_path)
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        ie._save_entries([
            IndexEntry(
                id="e1", timestamp=1.0, source_clip="a.mp4", thumbnail_path="t",
                counts={"person": 2, "car": 1},
                objects=[
                    {"label": "person", "confidence": 0.9, "bbox": [0, 0, 1, 1], "area": 1.0,
                     "color_hex": "#ff0000", "color_name": "red"},
                    {"label": "person", "confidence": 0.7, "bbox": [0, 0, 1, 1], "area": 1.0,
                     "color_hex": "#0000ff", "color_name": "blue"},
                    {"label": "car", "confidence": 0.8, "bbox": [0, 0, 1, 1], "area": 1.0,
                     "color_hex": "#00ff00", "color_name": "green"},
                ],
            ),
            IndexEntry(
                id="e2", timestamp=2.0, source_clip="a.mp4", thumbnail_path="t",
                counts={"person": 1},
                objects=[{"label": "person", "confidence": 0.6, "bbox": [0, 0, 1, 1], "area": 1.0,
                          "color_hex": "#ff0000", "color_name": "red"}],
            ),
            # legacy entry — no counts, only labels
            IndexEntry(id="e3", timestamp=1.0, source_clip="b.mp4", thumbnail_path="t", labels=["dog"]),
        ])

        with app.test_client() as client:
            r = client.get("/api/stats")
            assert r.status_code == 200
            data = r.get_json()
            assert data["total_frames"] == 3
            assert data["total_clips"] == 2
            assert data["total_objects"] == 4
            assert data["label_counts"] == {"person": 3, "car": 1, "dog": 1}
            assert data["color_counts"] == {"red": 2, "blue": 1, "green": 1}
            assert data["clip_counts"]["a.mp4"] == {"person": 3, "car": 1}
            assert data["clip_counts"]["b.mp4"] == {"dog": 1}


# ═══════════════════════════════════════════════════════════════════════
# Detection accuracy: size classes, aggregate confidence
# ═══════════════════════════════════════════════════════════════════════

class TestClassifySize:
    def test_boundaries(self):
        from unkillable.detection.detector import classify_size
        # 100x100 frame → area 10000
        assert classify_size((0, 0, 10, 10), 10000) == "tiny"      # 0.01
        assert classify_size((0, 0, 20, 20), 10000) == "small"     # 0.04
        assert classify_size((0, 0, 40, 40), 10000) == "medium"    # 0.16
        assert classify_size((0, 0, 50, 50), 10000) == "large"     # 0.25
        assert classify_size((0, 0, 70, 70), 10000) == "huge"      # 0.49

    def test_zero_frame_area(self):
        from unkillable.detection.detector import classify_size
        assert classify_size((0, 0, 10, 10), 0) == "tiny"

    def test_inverted_bbox_handled(self):
        from unkillable.detection.detector import classify_size
        assert classify_size((10, 10, 0, 0), 10000) == "tiny"


class TestMeanConfidence:
    def test_empty_inputs(self):
        from unkillable.detection.detector import Detector
        assert Detector.mean_confidence(None) == 0.0
        assert Detector.mean_confidence([]) == 0.0

    def test_average(self):
        from unkillable.detection.detector import Detection, Detector
        dets = [
            Detection(label="person", confidence=0.9, bbox=(0, 0, 10, 10)),
            Detection(label="car", confidence=0.7, bbox=(0, 0, 10, 10)),
        ]
        assert Detector.mean_confidence(dets) == 0.8


# ═══════════════════════════════════════════════════════════════════════
# Derived tags
# ═══════════════════════════════════════════════════════════════════════

class TestBuildTags:
    def test_compound_and_size(self):
        from unkillable.index.engine import build_tags
        objects = [
            {"label": "car", "color_name": "red", "size": "large"},
            {"label": "truck", "color_name": "unknown", "size": "small"},
            {"label": "truck", "color_name": "blue", "size": "huge"},
        ]
        tags = build_tags(objects)
        assert tags == ["blue truck", "car", "huge truck", "large car", "red car", "truck"]

    def test_skips_unknown(self):
        from unkillable.index.engine import build_tags
        assert build_tags([{"label": "car", "color_name": "unknown"}]) == ["car"]
        assert build_tags([]) == []
        assert build_tags([{"color_name": "red"}]) == []


# ═══════════════════════════════════════════════════════════════════════
# Hybrid query scoring
# ═══════════════════════════════════════════════════════════════════════

class TestHybridScoring:
    def _entry_path(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        return ie

    def test_parse_query_extracts_structure(self, tmp_path):
        from unkillable.index.engine import IndexEngine
        from unkillable.query.engine import QueryEngine
        qe = QueryEngine(index_engine=IndexEngine(storage_root=tmp_path))
        parsed = qe._parse_query("a big red automobile and two blue bicycles")
        assert parsed.labels == {"car", "bicycle"}
        assert parsed.colors == {"red", "blue"}
        assert parsed.sizes == {"large"}

    def test_color_boost(self, tmp_path):
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryEngine
        from unkillable.semantic.search import SemanticSearch

        ie = self._entry_path(tmp_path)
        ss = SemanticSearch(db_path=tmp_path / "index" / "entries.jsonl")
        entries = [
            IndexEntry(
                id="redcar", timestamp=1.0, source_clip="c", thumbnail_path="t1",
                embedding=ss.embed_text("red car"), labels=["car"],
                objects=[{"label": "car", "confidence": 0.9, "bbox": [0, 0, 10, 10],
                          "area": 100.0, "color_hex": "#ff0000", "color_name": "red"}],
                counts={"car": 1},
            ),
            IndexEntry(
                id="bluedog", timestamp=2.0, source_clip="c", thumbnail_path="t2",
                embedding=ss.embed_text("blue dog"), labels=["dog"],
                objects=[{"label": "dog", "confidence": 0.9, "bbox": [0, 0, 10, 10],
                          "area": 100.0, "color_hex": "#0000ff", "color_name": "blue"}],
                counts={"dog": 1},
            ),
        ]
        ie._save_entries(entries)

        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        results = qe.query("red car", top_k=2, generate_clips=False)
        assert results[0].entry.id == "redcar"
        assert results[0].score > results[1].score
        assert results[0].score_breakdown["label_match"] == 1.0
        assert results[0].score_breakdown["color_match"] == 1.0

    def test_size_boost(self, tmp_path):
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryEngine
        from unkillable.semantic.search import SemanticSearch

        ie = self._entry_path(tmp_path)
        ss = SemanticSearch(db_path=tmp_path / "index" / "entries.jsonl")
        vec = ss.embed_text("dog")
        entries = [
            IndexEntry(
                id="smalldog", timestamp=1.0, source_clip="c", thumbnail_path="t1",
                embedding=vec, labels=["dog"],
                objects=[{"label": "dog", "confidence": 0.8, "bbox": [0, 0, 10, 10],
                          "area": 100.0, "size": "small"}],
                counts={"dog": 1},
            ),
            IndexEntry(
                id="largedog", timestamp=2.0, source_clip="c", thumbnail_path="t2",
                embedding=vec, labels=["dog"],
                objects=[{"label": "dog", "confidence": 0.8, "bbox": [0, 0, 70, 70],
                          "area": 4900.0, "size": "large"}],
                counts={"dog": 1},
            ),
        ]
        ie._save_entries(entries)

        qe = QueryEngine(index_engine=ie, storage_root=tmp_path)
        results = qe.query("small dog", top_k=2, generate_clips=False)
        assert results[0].entry.id == "smalldog"
        assert results[0].score_breakdown["size_match"] == 1.0


# ═══════════════════════════════════════════════════════════════════════
# New IndexEntry fields: tags + detection_confidence end-to-end
# ═══════════════════════════════════════════════════════════════════════

class TestEntryNewFields:
    def test_query_result_to_dict_includes_tags_and_breakdown(self):
        from unkillable.index.models import IndexEntry
        from unkillable.query.engine import QueryResult
        entry = IndexEntry(
            id="x", timestamp=5.0, source_clip="c", thumbnail_path="t",
            labels=["car"], tags=["red car"],
            objects=[{"label": "car", "confidence": 0.85, "bbox": [0, 0, 10, 10],
                      "area": 100.0, "color_hex": "#ff0000", "color_name": "red"}],
            counts={"car": 1}, detection_confidence=0.85,
        )
        d = QueryResult(entry=entry, score=0.9, score_breakdown={"label_match": 1.0}).to_dict()
        assert d["tags"] == ["red car"]
        assert d["score_breakdown"] == {"label_match": 1.0}

    def test_entries_endpoint_exposes_tags_and_confidence(self, tmp_path):
        from unkillable.api import app, init_engines
        from unkillable.index.engine import IndexEngine
        from unkillable.index.models import IndexEntry

        init_engines(storage_root=tmp_path)
        ie = IndexEngine(storage_root=tmp_path)
        ie.ensure_dirs()
        ie._save_entries([
            IndexEntry(
                id="e1", timestamp=1.0, source_clip="c", thumbnail_path="t",
                labels=["car"], tags=["red car"], detection_confidence=0.85,
                objects=[{"label": "car", "confidence": 0.85, "bbox": [0, 0, 10, 10],
                          "area": 100.0, "color_name": "red"}],
                counts={"car": 1},
            ),
        ])

        with app.test_client() as client:
            r = client.get("/api/entries")
            assert r.status_code == 200
            e = r.get_json()["entries"][0]
            assert e["tags"] == ["red car"]
            assert e["detection_confidence"] == 0.85

            r = client.get("/api/stats")
            assert r.status_code == 200
            data = r.get_json()
            assert data["tag_counts"] == {"red car": 1}
            assert data["avg_detection_confidence"] == 0.85
