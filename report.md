# Report — The Unkillable Library

## Overview
Local-only NVR with motion-only recording, AI detection, semantic search, GenAI descriptions, paper QR backup, daily grayscale frame backup, and low-latency live video via MediaMTX HLS + HLS.js. Runs on Old PC/desktop (Linux + Docker, CPU-only) with a MediaMTX RTSP/HLS test stream via FFmpeg bridge from a public camera source.

## Project Layout
```
unkillable-library/
├── .venv/                     # virtualenv (via virtualenv fallback — ensurepip missing)
├── src/unkillable/
│   ├── motion/engine.py       # FFmpeg select='gt(scene,threshold)' + fallback
│   ├── detection/detector.py  # YOLOv8 if available else heuristic stub
│   ├── semantic/search.py     # CLIP or hash embedding + cosine search
│   ├── genai/describer.py     # Ollama vision LLM client + fallback
│   ├── backup/
│   │   ├── qr_backup.py       # tar.gz → base64 chunks → QR PDF (reportlab)
│   │   ├── qr_restore.py      # pyzbar decode → reassemble + extract
│   │   ├── frame_codec.py     # 1920x1080 L-mode RS(255,239) encode/decode + CRC
│   │   ├── daily_backup.py    # FFmpeg CRF-28 → tar → frames + manifest
│   │   └── frame_restore.py   # frames → SHA256 verify → extract
│   ├── cli.py                 # CLI: motion, thumbnail, detect, search, describe, backup, restore, backup-frames, restore-frames, storage-info
│   └── utils/{ffmpeg,storage,logging_config}.py
├── config/config.yml          # Frigate CPU detector, mode: motion, 30d retain
├── config/mediamtx.yml        # RTSP + HLS server (low-latency live)
├── config/mediamtx.yml        # RTSP + HLS server (low-latency live)
├── docker-compose.yml         # frigate + mediamtx + stream-bridge + monitor
├── dashboard/
│   ├── index.html             # Live (HLS.js tuned) + stored events + search UI
│   └── nginx.conf             # Proxy: /frigate/, /api/, /live/, /api/
├── scripts/generate_test_stream.sh  # FFmpeg testsrc → RTSP loop
├── scripts/setup.sh           # venv + pip + dir creation
├── tests/
│   ├── test_smoke.py          # 4 tests (storage, search, motion, QR)
│   └── test_frame_backup.py   # 7 tests (round-trip, RS correction, daily, guards)
├── requirements.txt           # pinned via pip freeze (incl. reedsolo)
├── storage/                   # clips, thumbnails, embeddings, recordings
├── backups/{YYYY-MM-DD}/      # daily grayscale frame sets + manifest.json
├── storage-optimize.md        # Grayscale frame backup design doc
├── report.md / OPINIONS.md    # This file
└── docs/
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

### 6. Daily Grayscale Frame Backup (`storage/` → `backups/YYYY-MM-DD/`)
- **Encode**: `daily_backup.backup_daily()` stages whole `storage/`, recompresses video (H.264 CRF-28 veryfast) and images (JPEG/WebP q70, keeps original if smaller), `tar -czf daily.tar.gz`, `frame_codec.encode_bytes()` splits into ~1.64MB data chunks → RS(255,239) per 239B block + 1 XOR parity frame per 10 data frames, renders lossless `1920x1080` `L`-mode PNGs (1 byte/px, 40px border, 4 corner markers, 64B header ×3: magic `UKLB1`, total/idx, payload len, CRC32) + `manifest.json` (counts, tarball SHA256, crf/nsym)
- **Decode**: `frame_restore.restore_frames()` loads `frame_*.png`, checks corner markers, majority-vote header, RS-decodes, XOR-recovers 1 lost frame/group, CRC32 per frame + SHA256 vs manifest, `tar -xzf` to target; `--delete-source` requires `--verify` pass, source cleared only after verified round-trip
- **Capacity**: ~2.0MB raw/frame, ~1.7MB net after RS; current ~32MB storage → ~12–15 PNGs/day

### 7. Low-Latency Live Video
- **Core Issue**: The live video player (`new Hls()`) was created with zero configuration, causing default HLS.js buffering of 6-24 seconds. Processed clips served unbuffered through nginx (`proxy_buffering off`) appeared before the live stream caught up.
- **Fix 1**: Configured HLS.js with `liveSyncDuration: 0`, `maxBufferLength: 1`, `liveDurationInfinity: true` to chase the live edge (reduced latency from ~6-24s to ~1 segment)
- **Fix 2**: Replaced the direct CDN HLS source with MediaMTX's HLS output (`http://localhost:8888/test/index.m3u8`). MediaMTX serves HLS from the local pipeline (bridge → RTSP → MediaMTX → HLS), so the live view goes through the same path as processed content. Enabled `hls: true` and `webrtc: true` in `config/mediamtx.yml`.
- **Pipeline**: Public HLS source → stream-bridge (ffmpeg HLS→RTSP TCP) → MediaMTX (:8554 RTSP, :8888 HLS) → Frigate (:5000) → nginx (:8080) → browser

## How To Use

### Setup
```bash
./scripts/setup.sh
source .venv/bin/activate
pip install -r requirements.txt
```

### Docker NVR (services up and running)
```bash
docker compose up -d        # mediamtx:8554/8888, frigate:5000/8555, stream-bridge, monitor:8080
# Open http://localhost:8080
```

### Daily Backup
```bash
PYTHONPATH=src .venv/bin/python -m unkillable.cli backup-frames --source storage --dest backups --verify
PYTHONPATH=src .venv/bin/python -m unkillable.cli backup-frames --source storage --dest backups --verify --delete-source
```

### Restore from Frame Backup
```bash
PYTHONPATH=src .venv/bin/python -m unkillable.cli restore-frames backups/2026-09-12 --output storage_restored
PYTHONPATH=src .venv/bin/python -m unkillable.cli restore-frames backups/2026-09-12 --output storage_restored --verify-only
```

### Python CLI
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
from unkillable.backup import QRBackup, backup_daily, restore_frames

MotionEngine(threshold=0.02).detect_via_ffmpeg("rtsp://localhost:8554/test", 10)
ss = SemanticSearch(db_path=Path("storage/embeddings/db.jsonl"))
ss.index("evt1", ss.embed_text("person at door"), {"camera":"front","ts":"2026-08-27T03:42:00Z"})
ss.search("person at door")
QRBackup().encode_to_qr_pdf(Path("storage/tmp.tar.gz"), Path("storage/backup.pdf"))
backup_daily("storage", "backups", date_str="2026-09-12", verify=True)
restore_frames("backups/2026-09-12", "storage_restored")
```

## Logging & Error Handling
- `utils/logging_config.setup_logging(level, log_file)` — stream + optional file, `get_logger(__name__)` per module
- Custom exceptions: `FFmpegError`, `DetectorError`, `StorageError`, `DescriberError`, `QRBackupError`, `QRRestoreError`, `FrameCodecError`, `RestoreError`, `BackupError` — all with chaining (`from exc`) and user-friendly messages
- Every I/O wrapped in try/except with `log.warning/error`; fallbacks for missing binaries/models/offline services

## Testing
```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v  # 11 passed (4 smoke + 7 frame backup)
```

## Hardware Profile
Old PC/desktop, Linux, Docker, CPU detector (`detectors.cpu.type: cpu`), no Coral/GPU, 720p@5fps detect, 30-day motion retain. MediaMTX HLS server on ports 8554/8888, Frigate on 5000/8555, nginx on 8080.

## Limitations & Next Steps
- FFmpeg not on host (installed in Frigate container); host fallback is dummy motion — install `ffmpeg` via apt for local CLI tests
- YOLO/CLIP heavy models optional — hash fallback works offline but not semantically accurate; add `ultralytics` + `sentence-transformers` for production
- pyzbar needs `libzbar0` on host (`apt install libzbar0`) for restore from images
- Video clips not in QR backup (by design — too large); use `backup-frames` for video (H.264 CRF-28, lossy)
- Frame backups must stay bit-exact PNGs; `reedsolo` pure-Python is slow on multi-GB stores — shard small or swap to a C RS backend later
- The `api` service (Python search API with CLIP/ultralytics) requires a full Docker build (~200MB torch) — not currently running but available via `docker compose build api && docker compose up -d api`
- MediaMTX cookie-check redirect can cause issues in some browsers; WebRTC (`webrtc: true`) is available as a sub-second latency alternative