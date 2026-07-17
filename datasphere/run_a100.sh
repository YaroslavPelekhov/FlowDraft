#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/prepare_dataset.py \
  --model-name Qwen/Qwen3-1.7B \
  --output-dir data/packed_qwen3_1p7b \
  --seq-len 2048 \
  --max-sequences "${MAX_SEQUENCES:-50000}" \
  --shard-size 1024

python scripts/train_orthrus.py --config configs/a100_80gb.yaml
