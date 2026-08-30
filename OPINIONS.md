# Opinions & Options Chosen

## Decisions

### 1. Python venv via virtualenv fallback
- **Option A**: `python3 -m venv` — failed (ensurepip missing, no python3.12-venv, no sudo)
- **Chosen**: `virtualenv` via `pip --user` + `~/.local/bin/virtualenv` — works without apt/sudo, creates `.venv` with pip 26.2.1
- **Tradeoff**: Extra bootstrap step, but portable on locked-down Ubuntu

### 2. FFmpeg Role — Hybrid
- **Doc says**: Frigate wraps FFmpeg; title says FFMPEG-FStival
- **Chosen**: Hybrid — Frigate's internal FFmpeg for NVR + explicit `utils/ffmpeg.FFmpegWrapper` for CLI, thumbnails, test stream
- **Why**: Satisfies both Frigate docker path and raw FFmpeg demos (scene detection, scale, RTSP loop)

### 3. Hardware — Old PC, Linux, CPU-only
- **Chosen**: `detectors.cpu.type: cpu`, no `hwaccel`, 1280×720@5fps, 64mb shm
- **Rejected**: Coral (`detectors.coral`), Intel NPU/GPU — not available, would require `privileged: true` already set but no device mount
- **Power**: ~30–50W old desktop — noted, no optimization beyond 5fps

### 4. Test Stream — MediaMTX + FFmpeg loop
- **Chosen**: `bluenviron/mediamtx` + `scripts/generate_test_stream.sh` (testsrc → RTSP `rtsp://localhost:8554/test` with `-stream_loop -1 -c copy`)
- **Rejected**: Direct file input to Frigate — unrealistic; MediaMTX publisher model matches real IP camera RTSP
- **Alternative**: `ffmpeg -f lavfi testsrc` on-the-fly — viable but `Mediamtx` + file loop gives reproducible file + stream

### 5. Motion Detection
- **Chosen**: FFmpeg `select='gt(scene,threshold)'` + metadata parsing; fallback `_dummy_motion` if binary missing
- **Rejected**: OpenCV frame differencing — heavier deps, less faithful to Frigate; kept OpenCV in requirements for future but not required for motion
- **Threshold**: Python default 0.02, Frigate config 30 (different scales — FFmpeg scene 0–1, Frigate 0–100) — intentional separate tuning

### 6. Object Detection
- **Chosen**: `Detector` tries `ultralytics.YOLO(yolov8n.pt)` if installed, else heuristic returning [] (no false positives)
- **Why**: Keeps `requirements.txt` light (no 100MB+ model download by default); production adds `ultralytics` optionally
- **Rejected**: Hard dependency on YOLO — would break offline/low-disk setups

### 7. Semantic Search
- **Chosen**: `sentence_transformers` CLIP if available else SHA256 hash embedding (512-dim normalized) + cosine search, JSONL DB
- **Why**: Real CLIP needs torch + 300MB model; hash fallback is deterministic and proves plumbing without heavy install
- **Rejected**: Jina CLIP download always — too heavy for scaffold; noted as `jinav1` in config but Python fallback is hash

### 8. GenAI Descriptions
- **Chosen**: `Describer` POSTs to Ollama `qwen3-vl:8b-instruct` with base64 image; fallback string if `ConnectionError`
- **Why**: Ollama runs as compose service (`ollama:11434`), `base_url` configurable; offline fallback keeps pipeline alive
- **Rejected**: OpenAI/cloud API — violates "100% local, no cloud"

### 9. QR Backup
- **Chosen**: `qrcode[pil]` + `reportlab` A4 4×4 grid, base64 chunk 2953B, 30% redundancy calc, `pyzbar` for restore
- **Why**: Matches doc capacity (~3KB/page, dense 130KB); `2953` is max QR alphanumeric at ECC M; reportlab gives printable PDF
- **Rejected**: `qr-backup` binary dependency — reimplemented in Python for venv portability
- **Note**: `pyzbar` needs system `libzbar0` (`apt install libzbar0`); install fails gracefully with `QRRestoreError` if missing

### 10. Requirements
- **Chosen**: Pinned `requirements.txt` via `pip freeze` (25 packages) for reproducibility; loose deps in `pyproject.toml`
- **Added after tests**: `annotated-doc`, `markdown-it-py`, `idna`, `charset-normalizer` — missing typer/rich/requests deps discovered at runtime
- **Kept**: `opencv-python-headless` (not strictly needed now) for future motion/frame handling

### 11. Docker Compose
- **Chosen**: `frigate:stable` + `mediamtx:latest` + `ollama:latest`, `shm_size: 64mb`, `privileged: true` for Frigate, tmpfs `/tmp/cache`
- **Rejected**: ` Frigate` extra volumes for Coral — not needed on CPU host
- **Ports**: 5000 (Frigate UI), 8554 (RTSP), 11434 (Ollama) — matches doc + MediaMTX config

### 12. Logging & Errors
- **Chosen**: Central `setup_logging()` with stream + optional file, per-module `get_logger`, custom exception types with `raise ... from exc`
- **Why**: Auditable, no silent failures; every external call (ffmpeg, yolo, ollama, file I/O) wrapped
- **Rejected**: `print()` debugging — use structured logging

## What I'd Do Differently If Real Deployment
- Pin Frigate to `ghcr.io/blakeblackshear/frigate:0.14` not `stable`
- Add `libzbar0` to Dockerfile or document apt step
- Replace hash embeddings with real `jinaai/jina-clip-v1` after confirming RAM (8GB+)
- Add `ffmpeg` to host via `apt` or bundle `imageio-ffmpeg`
- Add retention cron for `storage.cleanup_old(days=30)`
