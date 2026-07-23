#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

PYTHON_BIN="${PYTHON_BIN:-/workspace/flowdraft_venv/bin/python}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/dev/shm/flowdraft_hf_modules}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/flowdraft_xdg}"
export TMPDIR="${TMPDIR:-/tmp/flowdraft_tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/torch_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/triton_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-/workspace/flowdraft_data/nemotron_50k/manifest.json}"
EVAL_MANIFEST="${EVAL_MANIFEST:-/workspace/flowdraft_data/nemotron_50k_holdout/manifest.json}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/workspace/flowdraft_runs/flowdraft_v4_full_300/best}"
OUT_DIR="${OUT_DIR:-/workspace/flowdraft_runs/flowtree_v1_100}"
MAX_STEPS="${MAX_STEPS:-100}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"

if [ ! -f "$INIT_CHECKPOINT/adapter_config.json" ]; then
  log "Missing initialization checkpoint: $INIT_CHECKPOINT"
  exit 2
fi
if [ "$CLEAN_OUTPUT" = "1" ] && [ -d "$OUT_DIR" ]; then
  log "Refusing to overwrite an existing run: $OUT_DIR"
  exit 2
fi
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run.log") 2>&1

log "FlowTree v1 resource preflight"
"$PYTHON_BIN" scripts/inspect_resources.py --paths / /workspace /dev/shm
log "Training FlowTree coverage objective from v4 checkpoint"
"$PYTHON_BIN" scripts/train_flowdraft.py \
  --config configs/flowtree_v1_smoke.yaml \
  --train-manifest "$TRAIN_MANIFEST" \
  --eval-manifest "$EVAL_MANIFEST" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --output-dir "$OUT_DIR" \
  --max-steps "$MAX_STEPS"

CHECKPOINT="$OUT_DIR/best"
if [ ! -f "$CHECKPOINT/adapter_config.json" ]; then
  CHECKPOINT="$OUT_DIR/last"
fi
log "Strict FP32 FlowTree verifier gate"
"$PYTHON_BIN" scripts/smoke_flowtree.py \
  --checkpoint "$CHECKPOINT" \
  --branch-width 2 \
  --branch-depth 3 \
  --max-nodes 256 \
  --dtype fp32
log "FlowTree v1 run complete: $OUT_DIR"
