#!/usr/bin/env bash
set -euo pipefail

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

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements-datasphere.txt

if [ ! -d upstream_orthrus/.git ]; then
  git clone https://github.com/chiennv2000/orthrus upstream_orthrus
fi

"$PYTHON_BIN" -m pip install -e .
