#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

# Budget-matched clean Orthrus baseline for EagleFlow: same packed train and
# holdout data, K=32, 64 anchors, 20k optimizer updates and strict FP32 eval.
PYTHON_BIN="${PYTHON_BIN:-/workspace/flowdraft_venv/bin/python}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/workspace/flowdraft_data/nemotron_50k/manifest.json}"
EVAL_MANIFEST="${EVAL_MANIFEST:-/workspace/flowdraft_data/nemotron_50k_holdout/manifest.json}"
OUT_DIR="${OUT_DIR:-/dev/shm/flowdraft_runs/orthrus_budgetmatch_20000_r2}"
CONFIG_PATH="${CONFIG_PATH:-configs/orthrus_budgetmatch_20000.yaml}"
PROMPT_DIR="${PROMPT_DIR:-/workspace/flowdraft_runs/eagleflow_parallel_continue_20000_r1/paper_eval}"
MAX_STEPS="${MAX_STEPS:-20000}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_RUN_PATH="${HF_RUN_PATH:-}"

export HF_HOME="${HF_HOME:-/dev/shm/flowdraft_hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/dev/shm/flowdraft_hf_modules}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/flowdraft_xdg_cache}"
export TMPDIR="${TMPDIR:-/tmp/flowdraft_tmp}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

existing_entries="$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -printf '%f\n' | grep -vxE '(run.log|supervisor.log|supervisor.err.log)' || true)"
if [ -n "$existing_entries" ]; then
  log "Refusing to overwrite existing output: $OUT_DIR"
  exit 2
fi
if [ ! -f "$TRAIN_MANIFEST" ] || [ ! -f "$EVAL_MANIFEST" ]; then
  log "Missing packed train or holdout manifest"
  exit 2
fi
if [ ! -f "$PROMPT_DIR/aime25_prompts.jsonl" ] || [ ! -f "$PROMPT_DIR/humaneval_prompts.jsonl" ]; then
  log "Missing fixed AIME25/HumanEval prompts: $PROMPT_DIR"
  exit 2
fi

exec > >(tee -a "$OUT_DIR/run.log") 2>&1
log "Orthrus budget-matched resource preflight"
"$PYTHON_BIN" scripts/inspect_resources.py --paths / /workspace /dev/shm
log "Training clean Orthrus forward-KL baseline for $MAX_STEPS optimizer updates"
"$PYTHON_BIN" scripts/train_orthrus.py \
  --config "$CONFIG_PATH" \
  --train-manifest "$TRAIN_MANIFEST" \
  --eval-manifest "$EVAL_MANIFEST" \
  --output-dir "$OUT_DIR" \
  --max-steps "$MAX_STEPS"

run_status="$("$PYTHON_BIN" - "$OUT_DIR/run_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["status"])
PY
)"
if [ "$run_status" != "completed" ]; then
  log "Training ended with status=$run_status; skipping benchmark and Hugging Face upload"
  exit 130
fi

CHECKPOINT="$OUT_DIR/best"
if [ ! -f "$CHECKPOINT/config.json" ]; then
  CHECKPOINT="$OUT_DIR/last"
fi
PAPER_EVAL_DIR="$OUT_DIR/paper_eval"
mkdir -p "$PAPER_EVAL_DIR"
run_task() {
  local task="$1"
  local paper_tpf="$2"
  local paper_speedup="$3"
  "$PYTHON_BIN" scripts/benchmark_orthrus.py \
    --checkpoint "$CHECKPOINT" \
    --prompts-jsonl "$PROMPT_DIR/${task}_prompts.jsonl" \
    --output-jsonl "$PAPER_EVAL_DIR/${task}_fp32_metrics.jsonl" \
    --summary-json "$PAPER_EVAL_DIR/${task}_fp32_summary.json" \
    --max-new-tokens 128 \
    --dtype fp32 \
    --attn-implementation eager \
    --paper-tpf-target "$paper_tpf" \
    --paper-speedup-target "$paper_speedup" \
    --no-official-parity-on-mismatch \
    --require-parity
}
log "Strict FP32 greedy losslessness and efficiency evaluation"
run_task aime25 3.89 4.80
run_task humaneval 2.75 3.07

if [ -n "$HF_REPO_ID" ]; then
  log "Uploading atomic best and last checkpoints to Hugging Face"
  upload_args=(--run-dir "$OUT_DIR" --repo-id "$HF_REPO_ID")
  if [ -n "$HF_RUN_PATH" ]; then
    upload_args+=(--run-path "$HF_RUN_PATH")
  fi
  "$PYTHON_BIN" scripts/upload_checkpoints_hf.py "${upload_args[@]}"
fi
log "Orthrus budget-matched run complete: $OUT_DIR"
