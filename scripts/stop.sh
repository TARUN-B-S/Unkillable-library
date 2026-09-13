#!/usr/bin/env bash
# Stop the Unkillable Library API server started by scripts/start.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="$ROOT/storage/api.pid"

if [[ -f "$PIDFILE" ]]; then
  pid="$(cat "$PIDFILE")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "Stopped API server (pid $pid)"
  else
    echo "PID $pid not running (stale pidfile)"
  fi
  rm -f "$PIDFILE"
else
  echo "No API pidfile found — nothing to stop"
fi

echo "Docker stack left running (stop with: docker compose down)"