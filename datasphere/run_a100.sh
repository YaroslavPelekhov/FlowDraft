#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
log "Preparing packed dataset with MAX_SEQUENCES=${MAX_SEQUENCES:-50000}"
"$PYTHON_BIN" scripts/prepare_dataset.py \
  --model-name Qwen/Qwen3-1.7B \
  --output-dir data/packed_qwen3_1p7b \
  --seq-len 2048 \
  --max-sequences "${MAX_SEQUENCES:-50000}" \
  --shard-size 1024

log "Starting Orthrus Qwen3-1.7B training"
"$PYTHON_BIN" scripts/train_orthrus.py --config configs/a100_80gb.yaml

log "Training script finished"
