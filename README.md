# The Unkillable Library

Local-only NVR: motion-only recording, AI object detection, semantic search, paper QR backup, daily grayscale frame backup. Low-latency live video via MediaMTX HLS + HLS.js tuned to chase the live edge.

## Quick Start

```bash
./scripts/setup.sh
source .venv/bin/activate
unkillable --help
docker compose up -d
./scripts/generate_test_stream.sh
```

Open http://localhost:8080

## CLI

```
unkillable motion <rtsp-or-file> --duration 10
unkillable thumbnail <input> --output storage/thumbnails/thumb.jpg
unkillable detect <image.jpg> --labels person,car,dog
unkillable search "red car" --top-k 5
unkillable describe <image.jpg>
unkillable backup storage/thumbnails,storage/embeddings --output storage/backup.pdf
unkillable restore "storage/qr_*.png" --output storage/restored.tar.gz
unkillable backup-frames --source storage --dest backups --verify
unkillable restore-frames backups/2026-09-12 --output storage_restored
```

## Structure

```
unkillable-library/
├── .venv/                 # Python venv (virtualenv fallback)
├── src/unkillable/        # Python package
│   ├── motion/            # FFmpeg scene detection (mode: motion)
│   ├── detection/         # Object detector (YOLO + color extraction)
│   ├── semantic/          # CLIP embeddings + cosine search
│   ├── genai/             # Ollama vision LLM describer
│   ├── backup/            # QR encode/decode (paper backup)
│   └── utils/             # FFmpeg wrapper, storage, logging
├── config/config.yml      # Frigate config (CPU, motion, 30d retain)
├── config/mediamtx.yml    # RTSP + HLS server (low-latency live)
├── docker-compose.yml     # frigate + mediamtx + stream-bridge
├── storage/               # Clips, thumbnails, embeddings
├── backups/               # Daily grayscale frame backups
└── storage-optimize.md report.md OPINIONS.md
```

## Detection model

Indexing uses a YOLO model (Ultralytics). The first run downloads the model
automatically; `setup.sh` pre-warms the cache.

- `UNKILLABLE_YOLO_MODEL` — model file to load (default `yolov8n.pt`, falls back to `yolo11n.pt`, then the built-in heuristic).
- `UNKILLABLE_SKIP_MODEL_DOWNLOAD=1` — skip model loading entirely (offline/CI, heuristic mode).

Each indexed frame records `objects` (label, confidence, bbox, area, dominant
`color_hex`/`color_name`) and `counts` (label → count per frame). `/api/stats`
adds per-clip totals and a color histogram; results expose the same fields for
downstream use (e.g. feeding a color/object string to an LLM).
