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
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/workspace/flowdraft_runs/flowdraft_v5_prefix_ecld_2000_r3/best}"
OUT_DIR="${OUT_DIR:-/workspace/flowdraft_runs/hydraflow_advanced_token_5000_r2}"
MAX_STEPS="${MAX_STEPS:-5000}"
CONFIG_PATH="${CONFIG_PATH:-configs/hydraflow_advanced_token_5000.yaml}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_RUN_PATH="${HF_RUN_PATH:-}"

if [ -e "$OUT_DIR" ]; then
  existing_entries="$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -printf '%f\n' | grep -vxE '(run.log|supervisor.log|supervisor.err.log)' || true)"
  if [ -n "$existing_entries" ]; then
    log "Refusing to overwrite existing output: $OUT_DIR"
    exit 2
  fi
fi
if [ ! -f "$INIT_CHECKPOINT/adapter_config.json" ]; then
  log "Missing frozen FlowDraft parent: $INIT_CHECKPOINT"
  exit 2
fi
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run.log") 2>&1

log "Advanced-token feature flow resource preflight"
"$PYTHON_BIN" scripts/inspect_resources.py --paths / /workspace /dev/shm
log "Training one-verifier advanced-token endpoint flow map"
"$PYTHON_BIN" scripts/train_hydraflow.py \
  --config "$CONFIG_PATH" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --train-manifest "$TRAIN_MANIFEST" \
  --eval-manifest "$EVAL_MANIFEST" \
  --output-dir "$OUT_DIR" \
  --max-steps "$MAX_STEPS"

CHECKPOINT="$OUT_DIR/best"
if [ ! -f "$CHECKPOINT/hydraflow_head.safetensors" ]; then
  CHECKPOINT="$OUT_DIR/last"
fi
log "Strict FP32 greedy losslessness gate; TPF counts only frozen-Qwen verifier forwards"
"$PYTHON_BIN" scripts/benchmark_hydraflow.py \
  --checkpoint "$CHECKPOINT" \
  --prompts-jsonl eval_prompts/quick_compare.jsonl \
  --output-jsonl "$OUT_DIR/benchmark_fp32_metrics.jsonl" \
  --summary-json "$OUT_DIR/benchmark_fp32_summary.json" \
  --max-new-tokens 64 \
  --dtype fp32 \
  --attn-implementation eager \
  --require-parity

if [ -n "$HF_REPO_ID" ]; then
  log "Uploading best and last checkpoints to Hugging Face"
  HF_UPLOAD_ARGS=(--run-dir "$OUT_DIR" --repo-id "$HF_REPO_ID")
  if [ -n "$HF_RUN_PATH" ]; then
    HF_UPLOAD_ARGS+=(--run-path "$HF_RUN_PATH")
  fi
  "$PYTHON_BIN" scripts/upload_checkpoints_hf.py "${HF_UPLOAD_ARGS[@]}"
fi
log "Advanced-token feature flow run complete: $OUT_DIR"
