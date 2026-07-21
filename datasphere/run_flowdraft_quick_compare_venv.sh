#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

VENV_DIR="${VENV_DIR:-/tmp/flowdraft_venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ ! -x "$PYTHON_BIN" ]; then
  echo "Virtualenv not found at ${VENV_DIR}. Run: VENV_DIR=${VENV_DIR} bash datasphere/setup_venv.sh" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-/tmp/flowdraft_hf}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/dev/shm/flowdraft_hf_modules}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/flowdraft_xdg_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/flowdraft_hf/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/flowdraft_hf/transformers}"
export TMPDIR="${TMPDIR:-/dev/shm/flowdraft_tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/flowdraft_torchinductor}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_MODULES_CACHE" "$XDG_CACHE_HOME"

DATA_DIR="${DATA_DIR:-/tmp/flowdraft_storage/nemotron_quick_packed}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-/tmp/flowdraft_storage/nemotron_quick_eval_packed}"
OUT_DIR="${OUT_DIR:-/dev/shm/flowdraft_runs/flowdraft_quick2h}"
MAX_SEQUENCES="${MAX_SEQUENCES:-20000}"
EVAL_SEQUENCES="${EVAL_SEQUENCES:-512}"
EVAL_SKIP_SEQUENCES="${EVAL_SKIP_SEQUENCES:-$MAX_SEQUENCES}"
MAX_STEPS="${MAX_STEPS:-600}"
EPOCHS="${EPOCHS:-1}"
NUM_ANCHOR_BLOCKS="${NUM_ANCHOR_BLOCKS:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_BATCHES="${EVAL_BATCHES:-32}"
TRAIN_ATTN_IMPLEMENTATION="${TRAIN_ATTN_IMPLEMENTATION:-sdpa}"
BENCH_TOKENS="${BENCH_TOKENS:-128}"
BENCH_DTYPE="${BENCH_DTYPE:-bf16}"
BENCH_ATTN_IMPLEMENTATION="${BENCH_ATTN_IMPLEMENTATION:-sdpa}"
BENCH_REQUIRE_PARITY="${BENCH_REQUIRE_PARITY:-0}"
FLOW_STEPS="${FLOW_STEPS:-1}"
FLOW_STATE_MIN="${FLOW_STATE_MIN:-0.0}"
FLOW_STATE_MAX="${FLOW_STATE_MAX:-0.0}"
FLOW_OBJECTIVE="${FLOW_OBJECTIVE:-ecld}"
DIAGONAL_FRACTION="${DIAGONAL_FRACTION:-0.75}"
FLOW_TIME_CONDITIONING_SCALE="${FLOW_TIME_CONDITIONING_SCALE:-0.05}"
ENDPOINT_TOPK="${ENDPOINT_TOPK:-32}"
TEMPORAL_DIFFERENCE_EPSILON="${TEMPORAL_DIFFERENCE_EPSILON:-0.02}"
TEMPORAL_DRIFT_WEIGHT="${TEMPORAL_DRIFT_WEIGHT:-1.0}"
KL_REDUCTION="${KL_REDUCTION:-tokenmean}"
HARD_CE_WEIGHT="${HARD_CE_WEIGHT:-0.1}"
PREFIX_LOSS_WEIGHT="${PREFIX_LOSS_WEIGHT:-0.25}"
PREFIX_KL_WEIGHT="${PREFIX_KL_WEIGHT:-0.5}"
PREFIX_WEIGHT_DECAY="${PREFIX_WEIGHT_DECAY:-0.9}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.1}"
CONSISTENCY_START_STEP="${CONSISTENCY_START_STEP:-200}"
REBUILD_DATA="${REBUILD_DATA:-0}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"
BENCH_PARITY_ARGS=()
if [ "$BENCH_REQUIRE_PARITY" = "1" ]; then
  BENCH_PARITY_ARGS+=(--require-parity)
fi

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

EVAL_DATA_VALID=0
if [ -f "$EVAL_DATA_DIR/manifest.json" ]; then
  EVAL_DATA_VALID="$($PYTHON_BIN - "$EVAL_DATA_DIR/manifest.json" "$EVAL_SKIP_SEQUENCES" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
print(int(manifest.get("skip_sequences", 0) == int(sys.argv[2])))
PY
)"
fi

if [ "$REBUILD_DATA" = "1" ] || [ "$EVAL_DATA_VALID" != "1" ]; then
  log "Preparing ${EVAL_SEQUENCES} held-out eval sequences after skipping ${EVAL_SKIP_SEQUENCES}"
  rm -rf "$EVAL_DATA_DIR"
  "$PYTHON_BIN" scripts/prepare_dataset.py \
    --dataset-name nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-config default \
    --splits chat math code \
    --output-dir "$EVAL_DATA_DIR" \
    --seq-len 2048 \
    --skip-sequences "$EVAL_SKIP_SEQUENCES" \
    --max-sequences "$EVAL_SEQUENCES" \
    --shard-size 512
else
  log "Reusing non-overlapping packed eval data at ${EVAL_DATA_DIR}"
fi

log "Training quick FlowDraft checkpoint at ${OUT_DIR}"
"$PYTHON_BIN" scripts/train_flowdraft.py \
  --config configs/flowdraft_quick2h.yaml \
  --train-manifest "$DATA_DIR/manifest.json" \
  --eval-manifest "$EVAL_DATA_DIR/manifest.json" \
  --output-dir "$OUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --epochs "$EPOCHS" \
  --num-anchor-blocks "$NUM_ANCHOR_BLOCKS" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --save-every "$SAVE_EVERY" \
  --eval-every "$EVAL_EVERY" \
  --eval-batches "$EVAL_BATCHES" \
  --attn-implementation "$TRAIN_ATTN_IMPLEMENTATION" \
  --kl-reduction "$KL_REDUCTION" \
  --flow-state-min "$FLOW_STATE_MIN" \
  --flow-state-max "$FLOW_STATE_MAX" \
  --flow-objective "$FLOW_OBJECTIVE" \
  --diagonal-fraction "$DIAGONAL_FRACTION" \
  --flow-time-conditioning-scale "$FLOW_TIME_CONDITIONING_SCALE" \
  --endpoint-topk "$ENDPOINT_TOPK" \
  --temporal-difference-epsilon "$TEMPORAL_DIFFERENCE_EPSILON" \
  --temporal-drift-weight "$TEMPORAL_DRIFT_WEIGHT" \
  --hard-ce-weight "$HARD_CE_WEIGHT" \
  --prefix-loss-weight "$PREFIX_LOSS_WEIGHT" \
  --prefix-kl-weight "$PREFIX_KL_WEIGHT" \
  --prefix-weight-decay "$PREFIX_WEIGHT_DECAY" \
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
    --dtype "$BENCH_DTYPE" \
    --attn-implementation "$BENCH_ATTN_IMPLEMENTATION" \
    --flow-steps "$steps" \
    "${BENCH_PARITY_ARGS[@]}"

  if [ -d "$OUT_DIR/best" ]; then
    log "Benchmarking best FlowDraft checkpoint with flow_steps=${steps}"
    "$PYTHON_BIN" scripts/benchmark_flowdraft.py \
      --checkpoint "$OUT_DIR/best" \
      --prompts-jsonl eval_prompts/quick_compare.jsonl \
      --output-jsonl "$OUT_DIR/benchmark_best_flow${steps}_metrics.jsonl" \
      --summary-json "$OUT_DIR/benchmark_best_flow${steps}_summary.json" \
      --max-new-tokens "$BENCH_TOKENS" \
      --dtype "$BENCH_DTYPE" \
      --attn-implementation "$BENCH_ATTN_IMPLEMENTATION" \
      --flow-steps "$steps" \
      "${BENCH_PARITY_ARGS[@]}"
  fi
done

log "FlowDraft quick compare complete: ${OUT_DIR}"
