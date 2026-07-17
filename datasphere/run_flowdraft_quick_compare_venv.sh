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
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/dev/shm/flowdraft_hf_modules}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/flowdraft_xdg_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/flowdraft_hf/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/flowdraft_hf/transformers}"
export TMPDIR="${TMPDIR:-/dev/shm/flowdraft_tmp}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$HF_MODULES_CACHE" "$XDG_CACHE_HOME"

DATA_DIR="${DATA_DIR:-/tmp/flowdraft_storage/nemotron_quick_packed}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-/tmp/flowdraft_storage/nemotron_quick_eval_packed}"
OUT_DIR="${OUT_DIR:-/dev/shm/flowdraft_runs/flowdraft_quick2h}"
MAX_SEQUENCES="${MAX_SEQUENCES:-20000}"
EVAL_SEQUENCES="${EVAL_SEQUENCES:-512}"
MAX_STEPS="${MAX_STEPS:-600}"
EPOCHS="${EPOCHS:-1}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-100}"
BENCH_TOKENS="${BENCH_TOKENS:-128}"
FLOW_STEPS="${FLOW_STEPS:-1 2}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.05}"
CONSISTENCY_START_STEP="${CONSISTENCY_START_STEP:-400}"
REBUILD_DATA="${REBUILD_DATA:-0}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"

log "Resource preflight"
"$PYTHON_BIN" scripts/inspect_resources.py --paths / /tmp /dev/shm /home/jupyter || true
OUT_PARENT="$(dirname "$OUT_DIR")"
mkdir -p "$OUT_PARENT"
"$PYTHON_BIN" - "$OUT_PARENT" <<'PY'
import shutil
import sys
path = sys.argv[1]
free_gb = shutil.disk_usage(path).free / (1024**3)
print(f"Output filesystem free space at {path}: {free_gb:.2f} GiB")
if free_gb < 12:
    print(
        "WARNING: best+last full checkpoints may need roughly 8-10 GiB plus write headroom. "
        "Prefer OUT_DIR=/dev/shm/flowdraft_runs/flowdraft_quick2h or free more space.",
        file=sys.stderr,
    )
PY

if [ "$CLEAN_OUTPUT" = "1" ]; then
  log "Removing previous output at ${OUT_DIR}"
  rm -rf "$OUT_DIR"
fi

if [ "$REBUILD_DATA" = "1" ] || [ ! -f "$DATA_DIR/manifest.json" ]; then
  log "Preparing ${MAX_SEQUENCES} packed sequences at ${DATA_DIR}"
  "$PYTHON_BIN" scripts/prepare_dataset.py \
    --dataset-name nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-config default \
    --splits chat math code \
    --output-dir "$DATA_DIR" \
    --seq-len 2048 \
    --max-sequences "$MAX_SEQUENCES" \
    --shard-size 512
else
  log "Reusing packed training data at ${DATA_DIR}"
fi

if [ "$REBUILD_DATA" = "1" ] || [ ! -f "$EVAL_DATA_DIR/manifest.json" ]; then
  log "Preparing ${EVAL_SEQUENCES} eval packed sequences at ${EVAL_DATA_DIR}"
  "$PYTHON_BIN" scripts/prepare_dataset.py \
    --dataset-name nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-config default \
    --splits chat math code \
    --output-dir "$EVAL_DATA_DIR" \
    --seq-len 2048 \
    --max-sequences "$EVAL_SEQUENCES" \
    --shard-size 512
else
  log "Reusing packed eval data at ${EVAL_DATA_DIR}"
fi

log "Training quick FlowDraft checkpoint at ${OUT_DIR}"
"$PYTHON_BIN" scripts/train_flowdraft.py \
  --config configs/flowdraft_quick2h.yaml \
  --train-manifest "$DATA_DIR/manifest.json" \
  --eval-manifest "$EVAL_DATA_DIR/manifest.json" \
  --output-dir "$OUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --epochs "$EPOCHS" \
  --save-every "$SAVE_EVERY" \
  --eval-every "$EVAL_EVERY" \
  --consistency-weight "$CONSISTENCY_WEIGHT" \
  --consistency-start-step "$CONSISTENCY_START_STEP"

for steps in $FLOW_STEPS; do
  log "Benchmarking last FlowDraft checkpoint with flow_steps=${steps}"
  "$PYTHON_BIN" scripts/benchmark_flowdraft.py \
    --checkpoint "$OUT_DIR/last" \
    --prompts-jsonl eval_prompts/quick_compare.jsonl \
    --output-jsonl "$OUT_DIR/benchmark_flow${steps}_metrics.jsonl" \
    --summary-json "$OUT_DIR/benchmark_flow${steps}_summary.json" \
    --max-new-tokens "$BENCH_TOKENS" \
    --flow-steps "$steps"

  if [ -d "$OUT_DIR/best" ]; then
    log "Benchmarking best FlowDraft checkpoint with flow_steps=${steps}"
    "$PYTHON_BIN" scripts/benchmark_flowdraft.py \
      --checkpoint "$OUT_DIR/best" \
      --prompts-jsonl eval_prompts/quick_compare.jsonl \
      --output-jsonl "$OUT_DIR/benchmark_best_flow${steps}_metrics.jsonl" \
      --summary-json "$OUT_DIR/benchmark_best_flow${steps}_summary.json" \
      --max-new-tokens "$BENCH_TOKENS" \
      --flow-steps "$steps"
  fi
done

log "FlowDraft quick compare complete: ${OUT_DIR}"
