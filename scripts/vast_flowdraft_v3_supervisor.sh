#!/usr/bin/env bash
set -euo pipefail

source /opt/supervisor-scripts/utils/environment.sh

cd /workspace/FlowDraft

export PYTHON_BIN=/workspace/flowdraft_venv/bin/python
export TRAIN_MANIFEST=/workspace/flowdraft_data/nemotron_50k/manifest.json
export EVAL_MANIFEST=/workspace/flowdraft_data/nemotron_50k_holdout/manifest.json
export OUT_DIR=/workspace/flowdraft_runs/flowdraft_v3_full_600
export MAX_STEPS=600
export SEMIGROUP_START_STEP=300
export NUM_ANCHOR_BLOCKS=16
export EVAL_ANCHOR_BLOCKS=8
export GRADIENT_ACCUMULATION_STEPS=4
export EVAL_EVERY=50
export EVAL_BATCHES=8
export SAVE_EVERY=50
export EARLY_STOPPING_PATIENCE=0
export BENCH_TOKENS=128

exec bash scripts/run_vast_flowdraft_v3.sh
