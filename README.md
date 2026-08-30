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
unkillable backup storage/thumbnails,storage/embeddings --output storage/backup.pdf
unkillable restore "storage/qr_*.png"
```

## Structure

```
unkillable-library/
├── .venv/                 # Python venv (virtualenv fallback)
├── src/unkillable/        # Python package
│   ├── motion/            # FFmpeg scene detection (mode: motion)
│   ├── detection/         # Object detector (YOLO or heuristic)
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
