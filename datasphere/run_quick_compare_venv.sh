#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

VENV_DIR="${VENV_DIR:-/tmp/flowdraft_venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Virtualenv not found at ${VENV_DIR}. Run: VENV_DIR=${VENV_DIR} bash datasphere/setup_venv.sh" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-/tmp/flowdraft_hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/flowdraft_hf/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/flowdraft_hf/transformers}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="${DATA_DIR:-/tmp/flowdraft_storage/nemotron_quick_packed}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-/tmp/flowdraft_storage/nemotron_quick_eval_packed}"
OUT_DIR="${OUT_DIR:-/tmp/flowdraft_storage/orthrus_quick2h}"
MAX_SEQUENCES="${MAX_SEQUENCES:-20000}"
EVAL_SEQUENCES="${EVAL_SEQUENCES:-512}"
MAX_STEPS="${MAX_STEPS:-600}"
EPOCHS="${EPOCHS:-1}"
SAVE_EVERY="${SAVE_EVERY:-0}"
EVAL_EVERY="${EVAL_EVERY:-100}"
BENCH_TOKENS="${BENCH_TOKENS:-128}"

log "Preparing ${MAX_SEQUENCES} packed sequences at ${DATA_DIR}"
"$PYTHON_BIN" scripts/prepare_dataset.py \
  --dataset-name nvidia/Nemotron-Post-Training-Dataset-v2 \
  --dataset-config default \
  --splits chat math code \
  --output-dir "$DATA_DIR" \
  --seq-len 2048 \
  --max-sequences "$MAX_SEQUENCES" \
  --shard-size 512

log "Preparing ${EVAL_SEQUENCES} eval packed sequences at ${EVAL_DATA_DIR}"
"$PYTHON_BIN" scripts/prepare_dataset.py \
  --dataset-name nvidia/Nemotron-Post-Training-Dataset-v2 \
  --dataset-config default \
  --splits chat math code \
  --output-dir "$EVAL_DATA_DIR" \
  --seq-len 2048 \
  --max-sequences "$EVAL_SEQUENCES" \
  --shard-size 512

log "Training quick Orthrus checkpoint at ${OUT_DIR}"
"$PYTHON_BIN" scripts/train_orthrus.py \
  --config configs/quick2h.yaml \
  --train-manifest "$DATA_DIR/manifest.json" \
  --eval-manifest "$EVAL_DATA_DIR/manifest.json" \
  --output-dir "$OUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --epochs "$EPOCHS" \
  --save-every "$SAVE_EVERY" \
  --eval-every "$EVAL_EVERY"

log "Benchmarking final checkpoint"
"$PYTHON_BIN" scripts/benchmark_orthrus.py \
  --checkpoint "$OUT_DIR/final" \
  --prompts-jsonl eval_prompts/quick_compare.jsonl \
  --output-jsonl "$OUT_DIR/benchmark_metrics.jsonl" \
  --summary-json "$OUT_DIR/benchmark_summary.json" \
  --max-new-tokens "$BENCH_TOKENS"

if [ -d "$OUT_DIR/best" ]; then
  log "Benchmarking best checkpoint"
  "$PYTHON_BIN" scripts/benchmark_orthrus.py \
    --checkpoint "$OUT_DIR/best" \
    --prompts-jsonl eval_prompts/quick_compare.jsonl \
    --output-jsonl "$OUT_DIR/benchmark_best_metrics.jsonl" \
    --summary-json "$OUT_DIR/benchmark_best_summary.json" \
    --max-new-tokens "$BENCH_TOKENS"
fi

log "Quick compare complete: ${OUT_DIR}/benchmark_summary.json"
