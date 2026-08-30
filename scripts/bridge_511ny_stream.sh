#!/usr/bin/env bash
set -euo pipefail

SOURCE_URL="${1:-https://s52.nysdot.skyvdn.com/rtplive/R10_255/playlist.m3u8}"
DEST_URL="${2:-rtsp://localhost:8554/test}"

if ! command -v ffmpeg &>/dev/null; then
  echo "ERROR: ffmpeg not found." >&2
  exit 1
fi

while true; do
  # Cache-bust the master playlist and start at its newest segment. The loop
  # also recovers automatically when the public HLS publisher rotates variants.
  ffmpeg -hide_banner -loglevel warning \
    -live_start_index -1 \
    -rw_timeout 15000000 \
    -i "${SOURCE_URL}?cachebust=$(date +%s)" \
    -map 0:v:0 -c copy \
    -rtsp_transport tcp -rtsp_flags prefer_tcp \
    -f rtsp "$DEST_URL" || true
  sleep 2
done
