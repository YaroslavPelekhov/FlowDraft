#!/usr/bin/env bash
set -euo pipefail

source /opt/supervisor-scripts/utils/environment.sh

cd /workspace/FlowDraft

export PYTHON_BIN=/workspace/flowdraft_venv/bin/python
export TRAIN_MANIFEST=/workspace/flowdraft_data/nemotron_50k/manifest.json
export EVAL_MANIFEST=/workspace/flowdraft_data/nemotron_50k_holdout/manifest.json
export OUT_DIR="${OUT_DIR:-/workspace/flowdraft_runs/flowdraft_v4_full_300}"
export MAX_STEPS="${MAX_STEPS:-300}"
export NUM_ANCHOR_BLOCKS="${NUM_ANCHOR_BLOCKS:-64}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export EVAL_EVERY="${EVAL_EVERY:-50}"
export EVAL_BATCHES="${EVAL_BATCHES:-32}"
export SAVE_EVERY="${SAVE_EVERY:-50}"
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"

exec bash scripts/run_vast_flowdraft_v4.sh
