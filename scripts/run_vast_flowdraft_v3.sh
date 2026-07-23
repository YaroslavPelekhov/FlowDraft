#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

PYTHON_BIN="${PYTHON_BIN:-/workspace/flowdraft_venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON_FALLBACK:-python}"
fi

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/dev/shm/flowdraft_hf_modules}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/flowdraft_xdg}"
export TMPDIR="${TMPDIR:-/tmp/flowdraft_tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/torch_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/triton_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HF_MODULES_CACHE" \
  "$XDG_CACHE_HOME" \
  "$TMPDIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-/workspace/flowdraft_data/nemotron_50k/manifest.json}"
EVAL_MANIFEST="${EVAL_MANIFEST:-/workspace/flowdraft_data/nemotron_50k_holdout/manifest.json}"
OUT_DIR="${OUT_DIR:-/workspace/flowdraft_runs/flowdraft_v3_$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_STEPS="${MAX_STEPS:-300}"
SEMIGROUP_START_STEP="${SEMIGROUP_START_STEP:-150}"
NUM_ANCHOR_BLOCKS="${NUM_ANCHOR_BLOCKS:-16}"
EVAL_ANCHOR_BLOCKS="${EVAL_ANCHOR_BLOCKS:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
EVAL_EVERY="${EVAL_EVERY:-50}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"
SAVE_EVERY="${SAVE_EVERY:-50}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
BENCH_TOKENS="${BENCH_TOKENS:-128}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"

if [ "$CLEAN_OUTPUT" = "1" ] && [ -d "$OUT_DIR" ]; then
  log "Refusing to overwrite an existing run: $OUT_DIR"
  exit 2
fi
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run.log") 2>&1

log "FlowDraft v3 resource preflight"
"$PYTHON_BIN" scripts/inspect_resources.py --paths / /workspace /dev/shm

log "Training verifier-aligned stochastic categorical FlowDraft"
"$PYTHON_BIN" scripts/train_flowdraft_v3.py \
  --config configs/flowdraft_v3.yaml \
  --train-manifest "$TRAIN_MANIFEST" \
  --eval-manifest "$EVAL_MANIFEST" \
  --output-dir "$OUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --semigroup-start-step "$SEMIGROUP_START_STEP" \
  --num-anchor-blocks "$NUM_ANCHOR_BLOCKS" \
  --eval-anchor-blocks "$EVAL_ANCHOR_BLOCKS" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --eval-every "$EVAL_EVERY" \
  --eval-batches "$EVAL_BATCHES" \
  --save-every "$SAVE_EVERY" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"

CHECKPOINT="$OUT_DIR/best"
if [ ! -f "$CHECKPOINT/adapter_config.json" ]; then
  CHECKPOINT="$OUT_DIR/last"
fi

log "Strict FP32 greedy losslessness gate"
"$PYTHON_BIN" scripts/benchmark_flowdraft.py \
  --checkpoint "$CHECKPOINT" \
  --prompts-jsonl eval_prompts/quick_compare.jsonl \
  --output-jsonl "$OUT_DIR/benchmark_fp32_lossless_metrics.jsonl" \
  --summary-json "$OUT_DIR/benchmark_fp32_lossless_summary.json" \
  --max-new-tokens 64 \
  --flow-steps 1 \
  --dtype fp32 \
  --attn-implementation eager \
  --parity-margin-threshold 0 \
  --require-parity

log "BF16 throughput benchmark"
"$PYTHON_BIN" scripts/benchmark_flowdraft.py \
  --checkpoint "$CHECKPOINT" \
  --prompts-jsonl eval_prompts/quick_compare.jsonl \
  --output-jsonl "$OUT_DIR/benchmark_bf16_throughput_metrics.jsonl" \
  --summary-json "$OUT_DIR/benchmark_bf16_throughput_summary.json" \
  --max-new-tokens "$BENCH_TOKENS" \
  --flow-steps 1 \
  --dtype bf16 \
  --attn-implementation eager \
  --parity-margin-threshold 0

log "FlowDraft v3 run complete: $OUT_DIR"
