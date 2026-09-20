# Unkillable Library — Demo

**What this is:** a 100% local video surveillance system. It records a camera, detects objects,
and lets you search the footage in plain English — all on your own machine, nothing in the cloud.

**What you'll do in ~10 minutes:** start a live traffic camera → record it → ask
*"show me cars on the road"* → get the exact frames back → have an AI describe them.

---

## 1. What you need

| Requirement | Check with |
|---|---|
| Linux + Docker | `docker --version` |
| Python 3.10+ | `python3 --version` |
| ffmpeg on host | `ffmpeg -version` |

One-time setup:

```bash
./scripts/setup.sh
source .venv/bin/activate
export PYTHONPATH=src
```

---

## 2. Start everything

```bash
./scripts/start.sh
```

This brings up (first run takes a few minutes to pull images):

| Piece | Where | What it does |
|---|---|---|
| MediaMTX | docker | RTSP server the camera streams into |
| Stream bridge | docker | Republishes a **real 511NY traffic camera** into MediaMTX |
| Frigate | docker, UI on :5000 | Detects motion/objects, records clips |
| Search API + dashboard | :5001 / :8080 | Everything you'll interact with |

At the end you'll see the URLs printed. Open **http://localhost:8080** for the dashboard.

> No camera of your own needed — the bundled bridge streams a real public traffic feed.
> The dashboard is also reachable directly at http://localhost:5001.

---

## 3. The 30-second tour

Wait ~1 minute for Frigate to connect and record its first clips, then:

**Search from the dashboard** — open http://localhost:8080, type into the search bar:

```
cars on the road
```

You get the frames that matched, with the detected objects, colors, and match score on each card.
Click any card to see the clip play, the detected objects, and two buttons:

- **✦ Describe with AI** — a vision model looks at the frame and writes what it sees.
  The description is saved and immediately searchable.
- **👍 / 👎** — mark a result relevant or not. Down-voted frames rank lower for similar queries.

**Same thing from the terminal:**

```bash
# Search indexed frames
python -m unkillable.cli query "cars on the road" --no-clips

# Find frames closest to a moment in time
python -m unkillable.cli time "2026-09-12 21:48:13"

# AI description of a specific frame
python -m unkillable.cli describe storage/index/thumbnails/<frame>.jpg
```

---

## 4. Index a clip yourself

Search only finds what's been indexed. Frigate records automatically, but you can feed it
any video file:

```bash
python -m unkillable.cli index storage/clips/latest_clip.mp4 --camera driveway
```

What happens under the hood: ffmpeg picks keyframes at scene changes → YOLO detects objects
(person, car, truck…) with color and size → each frame gets an image embedding *and* a text
embedding of its tags → everything lands in a searchable index.

Then query it:

```bash
python -m unkillable.cli query "a red car in the driveway"
```

Queries like *"two big trucks"* are parsed structurally — they require one object that is
actually both big and a truck, not two separate objects in the same frame.

---

## 5. How ranking works (in one paragraph)

Each frame carries two embeddings: how the image *looks* (CLIP) and what was *detected* in it
(tags, colors, AI description). A query is scored against both, keeping the stronger match.
Structured terms (labels/colors/sizes) must be satisfied by a single object. Final ranking uses
reciprocal-rank fusion instead of hand-tuned weights, so a frame that ranks high on any signal
wins without one dominating. Marking results 👎 halves that frame's text-channel influence —
the system learns from you.

---

## 6. Stop

```bash
./scripts/stop.sh            # stops the API; containers keep running
docker compose down          # stops containers; your storage/ persists
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard empty, "0 results" | Index something first — step 4, or wait for Frigate clips (then `index` them) |
| Search is literal, no meaning | CLIP not installed: `pip install sentence-transformers` (heavy) |
| `detect` says 0 detections | YOLO not installed: `pip install ultralytics` |
| Describe fails / canned text | Ollama not reachable or model missing — check `curl localhost:11434/api/tags` |
| Frigate UI black at :5000 | Stream not publishing — check `docker logs mediamtx` and `stream-bridge` |
| Port 5001 already in use | `serve --port 5002` or stop the old server: `./scripts/stop.sh` |

---

*For the full command reference and internals, see `DOCUMENTATION.md`. For an honest account
of stubs and limitations, see `docs/USER_DEMO.md`.*
