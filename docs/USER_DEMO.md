# Unkillable Library — Local Demo Guide

A realistic end-to-end walkthrough of the scaffolded "Unkillable Library": a 100% local NVR
with motion-only recording (Frigate), an AI/detection/search/describe CLI, and paper QR backup.

The goal of this guide is to show exactly what **does** work today. Several headline features
are **demonstrative stubs or optional** — each is labeled below. Read the status legend first.

## Feature status legend

| Label | Meaning |
|---|---|
| **Works as-is** | Fully functional with the default install and no extra downloads. |
| **Demonstrative** | Proves the plumbing/flow, but substitutes a placeholder that is *not* production-quality (e.g. hash "embeddings" instead of real CLIP). |
| **Optional** | Requires an extra install/model that is **not** installed by default. |
| **Unavailable by default** | Present in docs/config but not actually reachable until you install something extra. |
| **Known limitation** | Behavior differs from the marketing docs. |

| Feature | Status | Notes |
|---|---|---|
| `motion` CLI (FFmpeg scene change) | **Works as-is** | Needs host `ffmpeg`; falls back to a fake event if missing. |
| `thumbnail` CLI | **Works as-is** | Needs host `ffmpeg`. |
| `detect` CLI | **Unavailable by default** | Returns **0** detections unless optional `ultralytics` (YOLO) is installed. |
| `search` CLI | **Demonstrative** | Defaults to SHA-256 **hash** embeddings, not real CLIP. Match is exact-text, not meaning. Real `sentence-transformers` is optional. |
| `describe` CLI | **Demonstrative / Optional** | Needs Ollama **and** a manually-pulled vision model. Offline → canned fallback sentence. |
| `backup` CLI (PDF) | **Works as-is** | See **QR redundancy limitation** below. |
| `restore` CLI | **Works as-is, with a system dep** | Needs system `libzbar0` installed via apt. |
| Frigate stack (RTSP + motion recording) | **Works as-is** | See **RTSP record-on-motion limitation** below. |
| Frigate `semantic_search` (config) | **Unavailable by default** | Not exercised by this demo; only the Python CLI `search` is demonstrable. |
| Frigate GenAI (config) | **Unavailable by default** | Separate from the CLI `describe`; needs model pull + wiring. |

## Known limitations (read before you run)

- **Semantic search is a fallback.** `requirements.txt` does not install
  `sentence-transformers` (and `ultralytics` is not there either), so `search` computes a
  deterministic SHA-256 hash vector from the query *text*. Searching `"red car"` only matches
  entries indexed from text that hashes identically — it will **not** find a picture of a red
  car. It demonstrates the search pipeline (embed → cosine → rank), not true semantic search.
  To get real CLIP embeddings, you must separately `pip install sentence-transformers`.
- **QR "redundancy" is not actually written.** The docs promise 30% page-loss tolerance.
  The code computes `30% redundant QR codes` and *logs* it, but never emits them — the PDF
  contains exactly one QR per chunk (`qr_backup.py` only loops over the real chunks). Restore
  will *tolerate* up to 30% missing chunks before erroring, but since there is no second copy
  on paper, losing a page means losing that data for real.
- **RTSP record-on-motion is Frigate's behavior, and the default feed is an external webcam.**
  With `record.retain.mode: motion`, Frigate only writes recordings when it detects motion.
  The default live feed is the **Panama Hummingbird Feeder Cam** (Cornell), a fixed camera with
  sparse motion — great for genuine motion events, but don't expect continuous footage. If you
  need a looped synthetic pattern instead, `scripts/generate_test_stream.sh` still provides the
  old `testsrc` loop.
- **System dependencies.** Two things are not covered by `requirements.txt` alone:
  - host `ffmpeg` (+ `ffprobe`) for the `motion`/`thumbnail` CLI and the test-stream script;
  - system `libzbar0` (`sudo apt install libzbar0`) for `restore`. Without it, restore fails
    with a clean error even though the `pyzbar` pip package is installed.
- **No video in backups (by design).** QR backup covers metadata/thumbnails/embeddings —
  clips are too large for paper and are explicitly excluded.

---

## 0. Prerequisites

- Linux with **Docker Engine + Compose v2**. Frigate needs privileged containers and 64 MB
  shared memory (already set in `docker-compose.yml`).
- `bash`, **Python 3.10+**.
- Host **`ffmpeg` and `ffprobe`** on `PATH` (for the CLI and the test stream):
  ```bash
  ffmpeg -version | head -1
  ```
- `poppler-utils` (`pdftoppm`) — only needed for the PDF→PNG step in the restore demo:
  ```bash
  sudo apt install -y poppler-utils
  ```
- `libzbar0` — only needed for QR restore:
  ```bash
  sudo apt install -y libzbar0
  ```
- ~3–4 GB free disk for images (more if you pull the Ollama vision model).
- Optional (heavy) upgrades you can skip for the core demo:
  - `pip install ultralytics` → real object detection (auto-downloads `yolov8n.pt`, ~6 MB).
  - `pip install sentence-transformers` → real CLIP search (pulls Torch + `clip-ViT-B-32`, multi-GB).

## 1. Setup

Run from the repository root:

```bash
./scripts/setup.sh
source .venv/bin/activate
export PYTHONPATH=src
```

- `setup.sh` creates `.venv` (virtualenv, since `python3 -m venv` is broken on this box),
  installs `requirements.txt`, creates `storage/{clips,thumbnails,embeddings}`, and checks
  that `ffmpeg` and `docker` exist.
- Verify the CLI: **the `unkillable` console script only exists if the package was
  `pip install -e .`; the always-works form is the module:**
  ```bash
  python -m unkillable.cli --help
  ```
  You should see the 8 commands: `motion`, `thumbnail`, `detect`, `search`, `describe`,
  `backup`, `restore`, `storage-info`.

> Tip: if you prefer a short alias, `alias uk="python -m unkillable.cli"`.

## 2. Start the Docker stack

```bash
docker compose up -d
docker compose ps
```

Expected result — containers up (the `stream-bridge` 511NY service is now opt-in via the
`busy-511ny` profile and does not start here):

| Service | Container | Ports | Role |
|---|---|---|---|
| `frigate` | `frigate` | 5000 (UI), 8555 | NVR: motion recording + detection |
| `mediamtx` | `mediamtx` | 8554 (RTSP), 8888 (API) | RTSP server for the live camera |
| `monitor` | `monitor` | 8080 | Dashboard |

First run pulls `ghcr.io/blakeblackshear/frigate:stable`, `bluenviron/mediamtx:latest`, and
`ollama/ollama:latest` — this can take several minutes.

## 3. Feed the live webcam RTSP stream

In a **second terminal** (the bridge streams forever, in the foreground):

```bash
./scripts/bridge_live_stream.sh
```

What it does: resolves the default preset (**Panama Hummingbird Feeder Cam**, a Cornell Lab
live 1080p30 YouTube broadcast) via `yt-dlp`, then remuxes it to `rtsp://localhost:8554/test`
with `ffmpeg`. On a stream drop it re-resolves the short-lived HLS manifest and reconnects.
It requires host `ffmpeg` and `yt-dlp`. See `Live_Bridge.md` for presets and probes.

Offline fallback (synthetic `testsrc` loop, old behavior):

```bash
./scripts/generate_test_stream.sh
```

Verify the stream is live (two quick checks):

```bash
ffprobe -v error -show_entries stream=codec_name,width,height -of csv=p=0 rtsp://localhost:8554/test
curl -s http://localhost:8888/v3/paths/list | python3 -m json.tool | grep -A3 '"test"'
```

Expected: `h264,1920,1080` from ffprobe, and MediaMTX's path API reporting `"test"` with
`"ready": true` (or `publishers: 1`).

## 4. Verify Frigate

Open **http://localhost:5000** in a browser.

- The **LIVE** page should show the `test_camera` receiving the live hummingbird cam
  (read via `rtsp://mediamtx:8554/test`, per `config/config.yml`).
- The **Events** page will be empty or sparse — motion here is real but sparse (see the
  RTSP record-on-motion limitation).
- Watch Frigate attach to the stream:
  ```bash
  docker compose logs -f frigate
  ```
  You should see lines about `test_camera` starting FFmpeg processes (one for `detect`, one for
  `record`). `Ctrl+C` to stop following logs.

If Frigate does detect motion, it writes clips/snapshots into `storage/` (the mounted
`/media/frigate` volume). Inspect growth anytime:

```bash
du -sh storage
find storage -type f -mmin -5 | head
```

Note: Frigate manages its own event data; `storage/events.jsonl` (the Python CLI's event list)
is separate and stays at zero unless you call `Storage.save_event()`.

## 5. CLI walkthrough (all working commands)

Commands below assume the venv is active and `PYTHONPATH=src` (section 1), and that the test
stream from section 3 is running.

### 5.1 Motion detection — **Works as-is**

```bash
python -m unkillable.cli motion rtsp://localhost:8554/test --duration 10 --threshold 0.02
```

Expected output:
```
Motion events: <N>
  score=0.xxx ts=...
  ...
```
`N` is usually several (a live cam with motion — e.g., a hummingbird flying through — or the
`testsrc` fallback pattern changes frames). This uses FFmpeg's
`select='gt(scene,0.02)'` scene filter. If host ffmpeg were missing, you'd instead see one
simulated event (`score=0.500`) from the `_dummy_motion` fallback — that fallback is a
**demonstrative stub**.
With the default live cam, expect **0** events during still stretches and a burst when motion
occurs — that's the cadence we want. The `--probe` flag on `bridge_live_stream.sh` measures it.

Also works against a local file: `python -m unkillable.cli motion storage/testsrc.mp4 --duration 5`.

### 5.2 Thumbnail extraction — **Works as-is**

```bash
python -m unkillable.cli thumbnail rtsp://localhost:8554/test --output storage/thumbnails/thumb.jpg
```

Expected output:
```
Thumbnail: storage/thumbnails/thumb.jpg
```
A 320 px-wide JPEG is written via `ffmpeg -ss 00:00:01 ... -vframes 1 -vf scale=320:-1`.

### 5.3 Object detection — **Unavailable by default**

```bash
python -m unkillable.cli detect storage/thumbnails/thumb.jpg --labels person,car,dog
```

Expected output (default install):
```
Detections: 0
```
This is **not fake success** — the detector deliberately returns no false positives when the
optional `ultralytics` model is absent (`detector.py` falls back to a heuristic that returns
`[]`). To get real detections, install:
```bash
pip install ultralytics   # downloads yolov8n.pt (~6 MB) on first call
python -m unkillable.cli detect storage/thumbnails/thumb.jpg
```
With YOLO, a testsrc thumb may legitimately still return 0 (it's a test pattern, not a photo);
use a real image for a meaningful demo.

### 5.4 Semantic search — **Demonstrative (hash fallback)**

There is no CLI command to index, so populate the embedding DB with the library API first:

```bash
python - <<'PY'
from unkillable.semantic import SemanticSearch
ss = SemanticSearch(db_path="storage/embeddings/db.jsonl")
for i, doc in enumerate(["red car in driveway", "dog in backyard", "person at the front door"]):
    ss.index(f"evt{i}", ss.embed_text(doc), {"camera": "front", "label": f"doc{i}", "ts": "2026-08-27T09:0%d:00Z" % i})
print("indexed 3 documents")
PY
```

Now search **exactly matching** text (exact match works — the hash is deterministic):

```bash
python -m unkillable.cli search "red car in driveway" --top-k 3
```

Expected output:
```
evt0 score=1.000 {"camera": "front", "label": "doc0", "ts": ...}
evt1 score=0.0xx ...
evt2 score=0.0xx ...
```

Expected for a *different* phrase (note the near-zero scores — this is **not** meaning):

```bash
python -m unkillable.cli search "a red vehicle on my driveway"
# evt0/evt1/evt2 score≈0.0xx — hash noise, not semantic relevance
```

If the DB is empty or missing, you'll see:
```
No results (DB empty or no match)
```

> **Honesty box:** with the default hash embedding, only near-identical text scores high.
> "Find the red car" will not find a photo. Real semantic search requires
> `pip install sentence-transformers` (heavy, downloads Torch + `clip-ViT-B-32`); the code
> picks it up automatically if installed.

### 5.5 AI description — **Demonstrative / Optional**

First confirm whether Ollama is reachable and which models exist:

```bash
curl -s http://localhost:11434/api/tags
```

Then try:

```bash
python -m unkillable.cli describe storage/thumbnails/thumb.jpg
```

Three possible outcomes, all truthful:

| Situation | Result |
|---|---|
| Ollama container **down** | `Security camera frame from thumb.jpg — motion event captured (Ollama offline, fallback description).` (canned fallback) |
| Ollama **up**, model **not pulled** | Red failure: `Describe failed: Ollama error: ... model ... not found` |
| Ollama **up**, model pulled | An actual natural-language sentence about the image |

The configured model (`qwen3-vl:8b-instruct`, per `config.yml`) is **not pulled by default**
and is a multi-GB download that runs slowly on CPU. To enable it for real:

```bash
docker exec ollama ollama pull qwen3-vl:8b-instruct   # several GB
python -m unkillable.cli describe storage/thumbnails/thumb.jpg
```

### 5.6 JSON storage info — **Works as-is**

```bash
python -m unkillable.cli storage-info --root storage
```

Expected output:
```
{'total_gb': ..., 'used_gb': ..., 'free_gb': ...}
Events: N
```
`N` counts lines in `storage/events.jsonl`, which is only populated via the Python API
(`Storage.save_event()`), **not** by Frigate — so it will normally be `0`.

## 6. QR backup / restore workflow

This is the flagship loop: **backup to a printable PDF → rasterize its QR pages → restore the
archive from images → extract.**

### 6.1 Create small data to back up

QRs hold ~3 KB per code, so keep the payload small (a real thumbnail JPEG can already exceed
one code; a multi-code PDF is fine but slower). `storage/events.jsonl` from step 5.4 plus a
tiny note works well:

```bash
python - <<'PY'
from unkillable.utils.storage import Storage
s = Storage("storage")
s.save_event({"camera": "front", "label": "person", "notes": "demo event"})
print("wrote storage/events.jsonl")
PY
printf 'Unkillable Library demo backup\nred car 3:42am\n' > storage/demo_notes.txt
```

### 6.2 Backup — **Works as-is** (with the redundancy caveat)

```bash
python -m unkillable.cli backup storage/events.jsonl,storage/demo_notes.txt --output storage/backup.pdf
```

Expected output (log line then result):
```
INFO  ... Encoding N chunks + M redundant -> N+M QR codes
Backup PDF: storage/backup.pdf
```

Open `storage/backup.pdf`: A4 pages with a 4×4 grid of QR codes and a `QR 1/N` caption under
each. Chunks are base64 slices (max ~2953 chars) prefixed with `idx/total|`.

> **QR redundancy limitation:** the log line above prints `+ M redundant`, but those codes are
> **not written to the PDF** — there is exactly one QR per chunk. Handle the printouts like
> their originals: keep all pages, or you lose data.

### 6.3 Restore — **Works as-is** (needs `libzbar0`)

The `restore` command decodes QR codes from **image files** (PNG/JPG — e.g. webcam photos of
the printed pages). For a fully offline loop, rasterize the PDF with `pdftoppm`:

```bash
mkdir -p storage/qr_pages
pdftoppm -png -r 300 storage/backup.pdf storage/qr_pages/page
```

Then decode and reassemble:

```bash
python -m unkillable.cli restore "storage/qr_pages/page-*.png" --output storage/restored.tar.gz
```

Expected output:
```
Restored: storage/restored.tar.gz
```

Verify the round trip is byte-identical to the archive you backed up:

```bash
mkdir -p storage/restored && tar -xzf storage/restored.tar.gz -C storage/restored
cat storage/restored/demo_notes.txt
# Unkillable Library demo backup
# red car 3:42am
```

If `libzbar0` is missing, restore fails cleanly with:
```
Restore failed: pyzbar/Pillow required for restore: ...
```
That's the expected symptom — install `libzbar0` (prerequisites) and retry.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `motion` prints one `score=0.500` event | Host `ffmpeg` missing → dummy fallback | `apt install ffmpeg`; re-run |
| `generate_test_stream.sh` / `bridge_live_stream.sh` says "ffmpeg not found" | No host ffmpeg | Install ffmpeg; the bridge also needs `pip install yt-dlp` for YouTube presets |
| Frigate LIVE shows no camera / black view | RTSP publisher not publishing | Confirm terminal 2 is running `bridge_live_stream.sh`; check `curl http://localhost:8888/v3/paths/list` |
| No Frigate recordings | `mode: motion` + a sparse live cam (e.g. still stretches) | Wait for motion (hummingbird visits), crank `motion.threshold` down (to e.g. 15) in `config/config.yml`, `docker compose restart frigate`; or use the `testsrc` loop to force motion |
| `describe` fails with "Ollama error: ... not found" | Model not pulled | `docker exec ollama ollama pull qwen3-vl:8b-instruct` |
| `describe` returns the canned fallback sentence | Ollama unreachable | `docker compose ps` — is `ollama` running? |
| `detect` always says `Detections: 0` | `ultralytics` not installed | `pip install ultralytics` (optional) |
| `search` scores are ~0.0 for related queries | Hash embedding fallback is not semantic | Install `sentence-transformers`, or phrase queries exactly as indexed |
| `restore` errors about pyzbar | System `libzbar0` missing | `sudo apt install libzbar0` |
| QR PDF encoding slow / many pages | Payload larger than ~3 KB/chunk | Back up small files; expect 1 QR per ~3 KB and 16 QR per A4 page |
| Unexpected Frigate behavior after config edit | `stable` image, not pinned | `docker compose restart frigate`; consider pinning `0.14.x` for production |

## 8. Safe shutdown & cleanup

Stop the demo cleanly (keep your data):

```bash
# 1. Stop the stream in its terminal (Ctrl+C), or from another terminal:
pkill -f bridge_live_stream.sh

# 2. Stop the Docker stack (containers stop; storage/ and ollama_data volume persist):
docker compose down

# 3. Optional full reset — removes containers AND the Ollama volume (destructive):
# docker compose down -v
```

What survives a `docker compose down`: everything on the host under `storage/`
(thumbnails, PDFs, restored archives, test video) and the `ollama_data` volume. To fully reset a
demo run:

```bash
docker compose down -v
rm -f storage/backup.pdf storage/restored.tar.gz storage/testsrc.mp4 storage/demo_notes.txt \
      storage/qr_pages/* storage/thumbnails/* storage/embeddings/db.jsonl storage/events.jsonl
find storage -type d -empty -delete
```

## 9. What the demo does *not* show

- Real semantic search (semantic meaning over thumbnails) — requires `sentence-transformers` and
  a populated embedding index; Frigate's own `semantic_search.jinav1` config is **not** exercised.
- Real object-detection events through Frigate with clips/snapshots (people/cars) — the default
  live cam is a hummingbird feeder; motion bursts are real but brief, and object detection is
  still YOLO-optional. For higher-value detections, probe a street/person preset:
  `./scripts/bridge_live_stream.sh --probe all`.
- GenAI descriptions wired through Frigate events — config exists (`genai`), but only the CLI
  `describe` command is demonstrable, and only after pulling the model.
- Redundant paper backup — the redundant QR set is logged but never printed.