#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "Could not find python3 or python in PATH" >&2
    exit 127
  fi
fi

log "Using Python: $("$PYTHON_BIN" --version 2>&1)"
log "Upgrading pip"
"$PYTHON_BIN" -m pip install --upgrade pip --progress-bar on

log "Installing project dependencies from requirements-datasphere.txt"
"$PYTHON_BIN" -m pip install -r requirements-datasphere.txt --progress-bar on

if [ ! -d upstream_orthrus/.git ]; then
  log "Cloning official Orthrus repository"
  git clone --progress https://github.com/chiennv2000/orthrus upstream_orthrus
else
  log "Official Orthrus repository already exists; skipping clone"
fi

log "Installing local training package in editable mode"
"$PYTHON_BIN" -m pip install -e .

log "Setup complete"
