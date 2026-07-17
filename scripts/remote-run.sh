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

DATASPHERE_AUTH_ARGS=()
if [ -n "${DATASPHERE_OAUTH_TOKEN:-}" ]; then
  DATASPHERE_AUTH_ARGS+=("-t" "$DATASPHERE_OAUTH_TOKEN")
fi
if [ -n "${DATASPHERE_PROFILE:-}" ]; then
  DATASPHERE_AUTH_ARGS+=("--profile" "$DATASPHERE_PROFILE")
fi

if [ -z "${DATASPHERE_PROJECT_ID:-}" ]; then
  echo "DATASPHERE_PROJECT_ID is not set." >&2
  echo "Set it with: export DATASPHERE_PROJECT_ID=<your_project_id>" >&2
  exit 2
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set locally; relying on a DataSphere project secret named HF_TOKEN." >&2
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 2
fi

"$DATASPHERE_BIN" "${DATASPHERE_AUTH_ARGS[@]}" project job execute \
  -p "$DATASPHERE_PROJECT_ID" \
  -c "$CONFIG_PATH"
