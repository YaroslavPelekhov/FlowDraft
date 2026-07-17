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

VENV_DIR="${VENV_DIR:-.venv}"

log "Creating isolated virtualenv at ${VENV_DIR}"
if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
  log "stdlib venv is unavailable; falling back to virtualenv"
  rm -rf "$VENV_DIR"
  "$PYTHON_BIN" -m pip install --user virtualenv --progress-bar on
  "$PYTHON_BIN" -m virtualenv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

log "Using venv Python: $("$VENV_PYTHON" --version 2>&1)"
log "Upgrading pip inside virtualenv"
"$VENV_PYTHON" -m pip install --upgrade pip --progress-bar on --no-cache-dir

log "Installing dependencies inside virtualenv"
"$VENV_PYTHON" -m pip install -r requirements-datasphere.txt --progress-bar on --no-cache-dir

if [ ! -d upstream_orthrus/.git ]; then
  log "Cloning official Orthrus repository"
  git clone --progress https://github.com/chiennv2000/orthrus upstream_orthrus
else
  log "Official Orthrus repository already exists; skipping clone"
fi

log "Installing local training package in editable mode"
"$VENV_PYTHON" -m pip install -e . --no-cache-dir

log "Virtualenv setup complete"
