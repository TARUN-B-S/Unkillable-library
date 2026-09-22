# The Unkillable Library

Local-only NVR: motion-only recording, AI object detection, semantic search, paper QR backup.

## Quick Start

```bash
./scripts/setup.sh
source .venv/bin/activate
unkillable --help
docker compose up -d
./scripts/generate_test_stream.sh
```

Open http://localhost:5000

## CLI

```
unkillable motion <rtsp-or-file> --duration 10
unkillable thumbnail <input> --output storage/thumbnails/thumb.jpg
unkillable detect <image.jpg> --labels person,car,dog
unkillable search "red car" --top-k 5
unkillable describe <image.jpg>
unkillable backup storage/index,storage/thumbnails --output storage/backup.pdf
unkillable restore "storage/qr_*.png"
unkillable story nightly --top-n 5              # storyteller trio: diary + anomalies + identities
unkillable story diary --date 2026-09-10        # print a day's nightwatch diary
unkillable story search "red car after 8am"     # search the diary text corpus
unkillable story identities                     # list persistent identities
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
├── config/mediamtx.yml    # RTSP server for test stream
├── docker-compose.yml     # Frigate + MediaMTX + Ollama
├── storage/               # Clips, thumbnails, embeddings
└── report.md / OPINIONS.md
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
