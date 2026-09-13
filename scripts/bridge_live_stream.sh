#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

# Preset table: <name>|<kind>|<source>
#   kind=yt      source is a YouTube watch/live URL (resolved via yt-dlp)
#   kind=hdontap source is an HDOnTap stream id (resolved via /api/streams/<id>/play/)
#   kind=dot     source is a static HLS URL (NY DOT et al.)
PRESETS=(
  "panama-hummer|yt|https://www.youtube.com/watch?v=0xFARCdZ3vA"
  "feederwatch-birds|yt|https://www.youtube.com/watch?v=x10vL6_47Dw"
  "fruit-feeder-birds|yt|https://www.youtube.com/watch?v=WtoxxHADnGk"
  "times-square-nyc|yt|https://www.youtube.com/watch?v=q9aiHJe_oyI"
  "fremont-las-vegas|yt|https://www.youtube.com/watch?v=5XXJN_Ep88E"
  "truckee-main-street|hdontap|657648"
  "temecula-town-square|hdontap|273143"
  "holycross-campus|hdontap|178090"
  "julian-downtown|hdontap|228394"
  "havasu-london-bridge|hdontap|101044"
  "511ny-cam|dot|https://s52.nysdot.skyvdn.com/rtplive/R10_255/playlist.m3u8"
)

PRESET="${UNKILLABLE_STREAM_PRESET:-panama-hummer}"
DEST_URL="rtsp://localhost:8554/test"
PROBE_MINUTES=""
PROBE_THRESHOLD="0.02"
YT_FORMATS="270/232/231/229/b[height<=1080]/best[height<=1080]"

fail() { echo "ERROR: $*" >&2; exit 1; }

digest() { echo "$1" | sed 's|https://www.youtube.com/[^ ]*|YouTube-live|; s|[?&].*||' | cut -c1-72; }

list_presets() {
  printf "%-24s %-10s %s\n" "PRESET" "KIND" "SOURCE"
  for row in "${PRESETS[@]}"; do
    IFS='|' read -r name kind source <<<"$row"
    printf "%-24s %-10s %s\n" "$name" "$kind" "$(digest "$source")"
  done
}

resolve_url() {
  local name="$1" kind="$2" source="$3" url=""
  case "$kind" in
    yt)
      command -v yt-dlp &>/dev/null || fail "'$name': yt-dlp not found (pip install yt-dlp)"
      url="$(yt-dlp --get-url -f "$YT_FORMATS" "$source" 2>/dev/null | head -1 || true)"
      [ -n "$url" ] || fail "'$name': yt-dlp could not extract a live stream from $source"
      ;;
    hdontap)
      command -v curl &>/dev/null || fail "'$name': curl not found"
      command -v python3 &>/dev/null || fail "'$name': python3 not found"
      url="$(curl -sL --max-time 15 "https://hdontap.com/api/streams/$source/play/" \
        -H "User-Agent: Mozilla/5.0" \
        | python3 -c 'import sys,json;print(json.load(sys.stdin)["stream_url"])' 2>/dev/null || true)"
      [ -n "$url" ] || fail "'$name': HDOnTap API returned no stream_url (id=$source)"
      ;;
    dot)
      url="$source"
      ;;
    *)
      fail "'$name': unknown kind '$kind'"
      ;;
  esac
  echo "$url"
}

# Probe a preset's motion cadence via FFmpeg scene detection.
probe_preset() {
  local name="$1" kind="$2" source="$3" duration_min="$4"
  local url events seconds
  command -v ffmpeg &>/dev/null || fail "ffmpeg not found"
  seconds=$((duration_min * 60))
  url="$(resolve_url "$name" "$kind" "$source")" || return 1
  # shellcheck disable=SC2046
  events="$(ffmpeg -hide_banner -loglevel error -nostdin \
    -i "$url" \
    -vf "select='gt(scene,$PROBE_THRESHOLD)',metadata=print:file=-" \
    -an -f null -t "$seconds" - 2>/dev/null | grep -c 'scene_score' || true)"
  printf "%s" "$events"
}

# CADENCE_BAND events per minute is "not very frequent, not very rare".
CADENCE_BAND_LOW="0.05"

usage() {
  cat <<'EOF'
Usage: ./scripts/bridge_live_stream.sh [OPTIONS]

Stream a curated live webcam feed to RTSP for Frigate/the CLI.

Options:
  --preset NAME       Preset to stream (default: UNKILLABLE_STREAM_PRESET or panama-hummer)
  --to URL            RTSP destination (default: rtsp://localhost:8554/test)
  --probe MINUTES     Instead of streaming, measure motion events/min per preset
                      (probe a single preset with --preset, or pass "all")
  --threshold FLOAT   Scene-change threshold for --probe (default: 0.02)
  --list              List available presets and exit

Requires: ffmpeg, yt-dlp (for yt presets), curl + python3 (for hdontap presets).
EOF
}

[[ $# -gt 0 ]] || usage

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) PRESET="$2"; shift 2 ;;
    --to) DEST_URL="$2"; shift 2 ;;
    --probe) PROBE_MINUTES="$2"; shift 2 ;;
    --threshold) PROBE_THRESHOLD="$2"; shift 2 ;;
    --list) list_presets; exit 0 ;;
    --help|-h) usage; exit 0 ;;
    *) fail "unknown argument: $1 (run with --help)" ;;
  esac
done

row=""
for r in "${PRESETS[@]}"; do
  IFS='|' read -r n k s <<<"$r"
  [[ "$n" == "$PRESET" ]] && { row="$r"; break; }
done
[ -n "$row" ] || fail "unknown preset '$PRESET' (see --list)"
IFS='|' read -r PRESET_NAME PRESET_KIND PRESET_SOURCE <<<"$row"

command -v ffmpeg &>/dev/null || fail "ffmpeg not found"

if [[ -n "$PROBE_MINUTES" ]]; then
  echo "Probing motion cadence (threshold=$PROBE_THRESHOLD, ${PROBE_MINUTES}m window):"
  echo "May hit network rate limits when probing 'all'."
  if [[ "$PROBE_MINUTES" == "all" ]]; then
    for r in "${PRESETS[@]}"; do
      IFS='|' read -r n k s <<<"$r"
      events="$(probe_preset "$n" "$k" "$s" "1")"
      printf "  %-24s %3s events/60s\n" "$n" "${events:-0}"
    done
  else
    events="$(probe_preset "$PRESET_NAME" "$PRESET_KIND" "$PRESET_SOURCE" "$PROBE_MINUTES")"
    printf "  %-24s %3s events in %s min (~%.2f/min)\n" \
      "$PRESET_NAME" "${events:-0}" "$PROBE_MINUTES" \
      "$(awk "BEGIN{printf \"%.2f\", (${events:-0})/$PROBE_MINUTES}")"
  fi
  echo "Done. Comparing cadences is the point; pick the preset closest to your target band."
  exit 0
fi

echo "Streaming preset '$PRESET_NAME' ($PRESET_KIND) -> $DEST_URL"
echo "Press Ctrl+C to stop"
while true; do
  url="$(resolve_url "$PRESET_NAME" "$PRESET_KIND" "$PRESET_SOURCE")"
  echo "[$(date '+%H:%M:%S')] resolved source (${#url}b) — starting ffmpeg"
  ffmpeg -hide_banner -loglevel warning -nostdin \
    -fflags nobuffer -flags low_delay \
    -live_start_index -1 \
    -rw_timeout 15000000 \
    -i "$url" \
    -map 0:v:0 -c copy \
    -rtsp_transport tcp -rtsp_flags prefer_tcp \
    -f rtsp "$DEST_URL" || true
  echo "[$(date '+%H:%M:%S')] stream dropped — refreshing source URL and retrying"
  sleep 2
done