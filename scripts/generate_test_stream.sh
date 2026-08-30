#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
VIDEO="${1:-$ROOT/storage/testsrc.mp4}"
RTSP_URL="${2:-rtsp://localhost:8554/test}"
DURATION="${3:-10}"

if ! command -v ffmpeg &>/dev/null; then
  echo "ERROR: ffmpeg not found. Install ffmpeg or run inside Docker." >&2
  exit 1
fi

mkdir -p "$(dirname "$VIDEO")"

if [[ ! -f "$VIDEO" ]]; then
  echo "Creating test video: $VIDEO"
  ffmpeg -y -f lavfi -i "testsrc=size=1280x720:rate=30" -f lavfi -i "sine=frequency=1000" -t "$DURATION" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$VIDEO"
fi

echo "Streaming $VIDEO -> $RTSP_URL (loop)"
echo "Press Ctrl+C to stop"
ffmpeg -re -stream_loop -1 -i "$VIDEO" -c copy -rtsp_transport tcp -f rtsp "$RTSP_URL"
