# Live Bridge — curated live webcam feeds for the NVR

The NVR's `test_camera` normally receives whatever someone publishes to
`rtsp://mediamtx:8554/test`. The default demo publisher is now a curated **live webcam
bridge** instead of the synthetic `testsrc` pattern, so Frigate sees real—if sparse—motion
and recordable scenes.

## Quick start

```bash
docker compose up -d            # mediamtx + frigate + monitor
./scripts/bridge_live_stream.sh # streams the default preset (panama-hummer) to RTSP
```

Then verify:

```bash
ffprobe -v error -show_entries stream=codec_name,width,height -of csv=p=0 rtsp://localhost:8554/test
curl -s http://localhost:8888/v3/paths/list | python3 -m json.tool | grep -A3 '"test"'
```

Expected: `h264,1920,1080` and MediaMTX reporting the `test` path as ready.

## Default preset: Panama Hummingbird Feeder Cam

- Source: Cornell Lab / Canopy Tower — `https://www.youtube.com/live/0xFARCdZ3vA`
- Resolved via `yt-dlp` to a live H.264 HLS playlist (`270` = 1080p30 video-only).
- Was live at build time (2026-09-13), 1920x1080 @ 30 fps.
- Fixed camera, feeding platform, occasional hummingbird sorties → sparse motion bursts,
  which is the intended cadence for motion-triggered recording + person-scene demos.

The live HLS manifest URL expires quickly, so the bridge re-resolves it on every loop
iteration (stream drops are transparent).

## The bridge

`scripts/bridge_live_stream.sh` streams any entry from a built-in preset table:

```bash
./scripts/bridge_live_stream.sh --list          # all presets
./scripts/bridge_live_stream.sh --preset truckee-main-street
./scripts/bridge_live_stream.sh --to rtsp://localhost:8554/other
./scripts/bridge_live_stream.sh --probe 2        # cadence check for the selected preset
./scripts/bridge_live_stream.sh --probe all      # compare every preset (slow, hits rate limits)
```

Preset resolution backends:

| backend | how it resolves a source URL | deps |
|---|---|---|
| `yt` | `yt-dlp --get-url -f "270/232/231/229/b[height<=1080]/best[height<=1080]"` | `yt-dlp` |
| `hdontap` | `https://hdontap.com/api/streams/<id>/play/` → JSON `stream_url` (signed HLS, ~12 h) | `curl` + `python3` |
| `dot` | static NY-DOT HLS URL (cache-busted per attempt) | `curl` not needed |

The main loop mirrors the old 511NY pattern: resolve → `ffmpeg -c copy → rtsp_transport tcp`
→ on drop, re-resolve and retry (`sleep 2`).

## Preset catalog

| Preset | Kind | Scene | Cadence |
|---|---|---|---|
| **panama-hummer** (default) | yt | Cornell Panama hummingbird feeder | sparse bursts |
| feederwatch-birds | yt | Cornell FeederWatch cam | sparse |
| fruit-feeder-birds | yt | Cornell Panama fruit feeder (1080p30) | sparse |
| times-square-nyc | yt | 24/7 Times Square | busy |
| fremont-las-vegas | yt | Fremont St Las Vegas | busy |
| truckee-main-street | hdontap | downtown Truckee CA main street | active (observed) |
| temecula-town-square | hdontap | Temecula town square | quiet AM, busy PM |
| holycross-campus | hdontap | College of the Holy Cross campus | activity in class hours |
| julian-downtown | hdontap | Julian CA main street | quiet |
| havasu-london-bridge | hdontap | Lake Havasu London Bridge walkway | tourist flow |
| 511ny-cam | dot | NY DOT highway cam (Newburgh) | busy vehicles |

Human-scene cadence is time-of-day dependent: use `--probe` at demo time and pick the preset
closest to the cadence you want.

## Offline fallback

`scripts/generate_test_stream.sh` still exists: it creates/loops a 10 s `testsrc` MP4 into
the same RTSP path when you have no internet or want a fully synthetic feed.

## Optional busy-tier (NY DOT) in Docker

The old in-container bridge is opt-in so it never collides with the host bridge on the same
path:

```bash
docker compose --profile busy-511ny up -d stream-bridge
```

## Notes & trade-offs

- The bridge runs on the **host** (needs host `ffmpeg`, `yt-dlp`, `curl`, `python3`) and
  publishes to `rtsp://localhost:8554/test`; Frigate reads it at `rtsp://mediamtx:8554/test`
  unchanged.
- Only one publisher may own a MediaMTX path at a time — don't run the host bridge and the
  `busy-511ny` stream-bridge simultaneously.
- YouTube live manifests are short-lived and can rotate qualities; the resolve-on-every-loop
  design handles that.