#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-datasphere/jobs/flowdraft-hardce.yaml}"
DATASPHERE_BIN="${DATASPHERE_BIN:-datasphere}"

if ! command -v "$DATASPHERE_BIN" >/dev/null 2>&1; then
  if [ -x ".venv/bin/datasphere" ]; then
    DATASPHERE_BIN=".venv/bin/datasphere"
  else
    echo "datasphere CLI is not installed. Run: python3 -m venv .venv && .venv/bin/python -m pip install datasphere" >&2
    exit 127
  fi
fi

if ! command -v "$DATASPHERE_BIN" >/dev/null 2>&1 && [ ! -x "$DATASPHERE_BIN" ]; then
  echo "datasphere CLI is not executable: $DATASPHERE_BIN" >&2
  exit 127
fi

if [ -z "${DATASPHERE_PROJECT_ID:-}" ]; then
  echo "DATASPHERE_PROJECT_ID is not set." >&2
  echo "Set it with: export DATASPHERE_PROJECT_ID=<your_project_id>" >&2
  exit 2
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. It is needed for the gated Nemotron dataset." >&2
  echo "Set it with: export HF_TOKEN=hf_..." >&2
  exit 2
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 2
fi

"$DATASPHERE_BIN" project job execute \
  -p "$DATASPHERE_PROJECT_ID" \
  -c "$CONFIG_PATH"
