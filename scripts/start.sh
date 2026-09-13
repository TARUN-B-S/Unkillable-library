#!/usr/bin/env bash
# Start the full Unkillable Library stack:
#   1. Optional fresh re-index of storage/clips/*.mp4
#   2. Docker stack (frigate, mediamtx, monitor/nginx, stream-bridge)
#   3. Flask search API on :5001 (starts if not already running)
#
# IDs all background processes from this script so `stop.sh` can shut them down.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
LOG_DIR="$ROOT/storage/logs"
API_PORT="${API_PORT:-5001}"
PIDFILE="$ROOT/storage/api.pid"

mkdir -p "$LOG_DIR"

echo "==> Unkillable Library startup ($ROOT)"

# 1) Index source clips when the index is empty.
if [[ ! -s "$ROOT/storage/index/entries.jsonl" ]]; then
  echo "==> Index empty — indexing clips from storage/clips/"
  for clip in "$ROOT"/storage/clips/*.mp4; do
    [[ -e "$clip" ]] || continue
    echo "    indexing $(basename "$clip")"
    "$VENV/bin/unkillable" index "$clip" 2>&1 | tail -1 || true
  done
else
  echo "==> Index present — skipping re-index (delete storage/index/entries.jsonl to rebuild)"
fi

# 2) Docker stack (raises Frigate, Mediamtx, nginx dashboard, stream bridge).
echo "==> Bringing up docker stack"
docker compose -f "$ROOT/docker-compose.yml" up -d

# 3) Search API server
if ss -ltn 2>/dev/null | grep -q ":$API_PORT " || curl -sf "http://localhost:$API_PORT/api/stats" >/dev/null 2>&1; then
  echo "==> API already running on :$API_PORT — skipping start"
else
  echo "==> Starting API server on :$API_PORT"
  if [[ -f "$PIDFILE" ]]; then
    old_pid="$(cat "$PIDFILE")"
    if kill -0 "$old_pid" 2>/dev/null; then
      echo "    killing stale pid $old_pid"
      kill "$old_pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
  fi
  "$VENV/bin/unkillable" serve --port "$API_PORT" \
    >"$LOG_DIR/api.log" 2>&1 &
  echo $! > "$PIDFILE"
  for _ in {1..30}; do
    curl -sf "http://localhost:$API_PORT/api/stats" >/dev/null 2>&1 && break
    sleep 1
  done
fi

echo
echo "==> Up:"
echo "    Dashboard          http://localhost:8080"
echo "    Search API         http://localhost:$API_PORT/api/search?q=trucks+and+colors"
echo "    Stats              http://localhost:$API_PORT/api/stats"
echo "    API log            $LOG_DIR/api.log"