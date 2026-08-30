#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"

echo "Setting up Unkillable Library..."

if [[ ! -d "$VENV" ]]; then
  echo "Creating venv..."
  if command -v virtualenv &>/dev/null; then
    virtualenv "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi

echo "Installing requirements..."
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$ROOT/requirements.txt"

echo "Creating storage dirs..."
mkdir -p "$ROOT/storage"/{clips,thumbnails,embeddings}

echo "Checking ffmpeg..."
if command -v ffmpeg &>/dev/null; then echo "ffmpeg: $(ffmpeg -version | head -1)"; else echo "WARNING: ffmpeg not found on host — will be available inside Frigate container"; fi

echo "Checking docker..."
if command -v docker &>/dev/null; then docker --version; else echo "WARNING: docker not found — install Docker to run Frigate/MediaMTX"; fi

echo "Done. Activate venv: source $VENV/bin/activate"
echo "Run: unkillable --help"
