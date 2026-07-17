#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Virtualenv not found at ${VENV_DIR}. Run: bash datasphere/setup_venv.sh" >&2
  exit 1
fi

PYTHON_BIN="$VENV_DIR/bin/python" bash datasphere/run_a100.sh
