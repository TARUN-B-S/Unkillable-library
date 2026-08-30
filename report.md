# Report — The Unkillable Library

## Overview
Scaffolded a Python + Docker implementation of the FStival doc: local-only NVR with motion-only recording, AI detection, semantic search, GenAI descriptions, and paper QR backup. Runs on Old PC/desktop (Linux + Docker, CPU-only) with a MediaMTX RTSP test stream via FFmpeg.

## Project Layout
```
unkillable-library/
├── .venv/                     # virtualenv (via virtualenv fallback — ensurepip missing)
├── src/unkillable/
│   ├── motion/engine.py       # FFmpeg select='gt(scene,threshold)' + fallback
│   ├── detection/detector.py  # YOLOv8 if available else heuristic stub
│   ├── semantic/search.py     # CLIP or hash embedding + cosine search
│   ├── genai/describer.py     # Ollama vision LLM client + fallback
│   ├── backup/qr_backup.py    # tar.gz → base64 chunks → QR PDF (reportlab)
│   ├── backup/qr_restore.py   # pyzbar decode → reassemble + extract
│   └── utils/{ffmpeg,storage,logging_config}.py
├── config/config.yml          # Frigate CPU detector, mode: motion, 30d retain
├── config/mediamtx.yml        # RTSP server (path test)
├── docker-compose.yml         # frigate + mediamtx + ollama
├── scripts/generate_test_stream.sh  # FFmpeg testsrc → RTSP loop
├── scripts/setup.sh           # venv + pip + dir creation
├── tests/test_smoke.py        # 4 tests (storage, search, motion, QR)
├── requirements.txt           # pinned via pip freeze
└── storage/{clips,thumbnails,embeddings}/
```

## Features

### 1. Motion-Only Recording
- **Module**: `unkillable.motion.MotionEngine`
- **FFmpeg**: `select='gt(scene,threshold)'` metadata parsing; `record_on_motion` copies only on events
- **Config**: `record.retain.mode: motion`, `motion.threshold: 30` in `config.yml`
- **Error handling**: `FFmpegError`, fallback `_dummy_motion` if binary missing; timeout handling

### 2. AI Object Detection
- **Module**: `unkillable.detection.Detector`
- **Logic**: Loads `ultralytics.YOLO(yolov8n.pt)` if installed; else logs warning and returns heuristic empty (no false positives). Validates labels against Frigate's 60+ types
- **Config**: `objects.track: [person,car,dog,cat,truck,bicycle]`

### 3. Semantic Search (Plain English)
- **Module**: `unkillable.semantic.SemanticSearch`
- **Logic**: `sentence_transformers.CLIP` if available else SHA256 hash vector (512-dim normalized) — cosine similarity, JSONL DB (`storage/embeddings/db.jsonl`)
- **Config**: `semantic_search.model: jinav1, size: small`

### 4. AI-Generated Descriptions
- **Module**: `unkillable.genai.Describer`
- **Logic**: POST to `ollama:11434/api/generate` with base64 image + `qwen3-vl:8b-instruct`; fallback string if offline; `health_check()` probes `/api/tags`

### 5. Paper QR Backup
- **Encode**: `QRBackup.create_archive()` tar.gz + `encode_to_qr_pdf()` base64 split (2953B chunks), 30% redundancy math, reportlab A4 grid 4×4, qrcode ERROR_CORRECT_M
- **Decode**: `QRRestore.decode_from_images()` pyzbar decode, header `idx/total|payload`, base64 reassembly, tar extract
- **Capacity**: ~3KB/page default, dense ~130KB; month ≈ 30–150 pages

## How To Use

### Setup
```bash
./scripts/setup.sh
source .venv/bin/activate
# or: python -m virtualenv .venv (fallback used here)
pip install -r requirements.txt
```

### Docker NVR
```bash
docker compose up -d        # frigate:5000, mediamtx:8554, ollama:11434
./scripts/generate_test_stream.sh              # creates storage/testsrc.mp4 then loops to rtsp://localhost:8554/test
# custom: ./scripts/generate_test_stream.sh myvideo.mp4 rtsp://localhost:8554/test 30
```

### Python CLI (all with logging + error handling)
```bash
PYTHONPATH=src .venv/bin/python -m unkillable.cli --help
PYTHONPATH=src .venv/bin/python -m unkillable.cli motion rtsp://localhost:8554/test --duration 10 --threshold 0.02
PYTHONPATH=src .venv/bin/python -m unkillable.cli thumbnail rtsp://localhost:8554/test --output storage/thumbnails/thumb.jpg
PYTHONPATH=src .venv/bin/python -m unkillable.cli detect storage/thumbnails/thumb.jpg --labels person,car,dog
PYTHONPATH=src .venv/bin/python -m unkillable.cli search "red car leaving" --top-k 5 --db storage/embeddings/db.jsonl
PYTHONPATH=src .venv/bin/python -m unkillable.cli describe storage/thumbnails/thumb.jpg
PYTHONPATH=src .venv/bin/python -m unkillable.cli backup storage/thumbnails,storage/embeddings --output storage/backup.pdf
PYTHONPATH=src .venv/bin/python -m unkillable.cli restore "storage/qr_*.png" --output storage/restored.tar.gz
PYTHONPATH=src .venv/bin/python -m unkillable.cli storage-info --root storage
```

### Programmatic
```python
from pathlib import Path
from unkillable.motion import MotionEngine
from unkillable.semantic import SemanticSearch
from unkillable.backup import QRBackup

MotionEngine(threshold=0.02).detect_via_ffmpeg("rtsp://localhost:8554/test", 10)
ss = SemanticSearch(db_path=Path("storage/embeddings/db.jsonl"))
ss.index("evt1", ss.embed_text("person at door"), {"camera":"front","ts":"2026-08-27T03:42:00Z"})
ss.search("person at door")
QRBackup().encode_to_qr_pdf(Path("storage/tmp.tar.gz"), Path("storage/backup.pdf"))
```

## Logging & Error Handling
- `utils/logging_config.setup_logging(level, log_file)` — stream + optional file, `get_logger(__name__)` per module
- Custom exceptions: `FFmpegError`, `DetectorError`, `StorageError`, `DescriberError`, `QRBackupError`, `QRRestoreError` — all with chaining (`from exc`) and user-friendly messages
- Every I/O wrapped in try/except with `log.warning/error`; fallbacks for missing binaries/models/offline services

## Testing
```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_smoke.py -v  # 4 passed
```

## Hardware Profile
Old PC/desktop, Linux, Docker, CPU detector (`detectors.cpu.type: cpu`), no Coral/GPU, 720p@5fps detect, 30-day motion retain.

## Limitations & Next Steps
- FFmpeg not on host (installed in Frigate container); host fallback is dummy motion — install `ffmpeg` via apt for local CLI tests
- YOLO/CLIP heavy models optional — hash fallback works offline but not semantically accurate; add `ultralytics` + `sentence-transformers` for production
- pyzbar needs `libzbar0` on host (`apt install libzbar0`) for restore from images
- Video clips not in QR backup (by design — too large)
