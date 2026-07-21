#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Export it locally or define it as a DataSphere project secret." >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
JOB_CACHE_DIR="${JOB_CACHE_DIR:-$(pwd)/.job_cache}"
export HF_HOME="${HF_HOME:-$JOB_CACHE_DIR/hf}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/dev/shm/flowdraft_hf_modules}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/flowdraft_xdg_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$JOB_CACHE_DIR/hf/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$JOB_CACHE_DIR/hf/transformers}"
export TMPDIR="${TMPDIR:-/dev/shm/flowdraft_tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/flowdraft_torchinductor}"
mkdir -p \
  "$JOB_CACHE_DIR" \
  "$HF_HOME" \
  "$HF_MODULES_CACHE" \
  "$XDG_CACHE_HOME" \
  "$TMPDIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  outputs

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-/dev/shm/flowdraft_runs/flowdraft_hardce_quick2h}"
RESULTS_DIR="${RESULTS_DIR:-outputs/flowdraft_hardce_quick2h}"
DATA_DIR="${DATA_DIR:-$JOB_CACHE_DIR/flowdraft_storage/nemotron_quick_packed}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$JOB_CACHE_DIR/flowdraft_storage/nemotron_quick_eval_packed}"
MAX_STEPS="${MAX_STEPS:-600}"
NUM_ANCHOR_BLOCKS="${NUM_ANCHOR_BLOCKS:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
MAX_SEQUENCES="${MAX_SEQUENCES:-20000}"
EVAL_SEQUENCES="${EVAL_SEQUENCES:-512}"
FLOW_STEPS="${FLOW_STEPS:-1}"
BENCH_DTYPE="${BENCH_DTYPE:-bf16}"
BENCH_ATTN_IMPLEMENTATION="${BENCH_ATTN_IMPLEMENTATION:-sdpa}"
BENCH_REQUIRE_PARITY="${BENCH_REQUIRE_PARITY:-0}"
HARD_CE_WEIGHT="${HARD_CE_WEIGHT:-0.5}"
KL_REDUCTION="${KL_REDUCTION:-tokenmean}"
PREFIX_LOSS_WEIGHT="${PREFIX_LOSS_WEIGHT:-1.0}"
PREFIX_KL_WEIGHT="${PREFIX_KL_WEIGHT:-0.5}"
PREFIX_WEIGHT_DECAY="${PREFIX_WEIGHT_DECAY:-0.9}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.0}"
CONSISTENCY_START_STEP="${CONSISTENCY_START_STEP:-10000}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_RUN_PATH="${HF_RUN_PATH:-}"

log "Python: $("$PYTHON_BIN" --version 2>&1)"
log "Preparing result directory: ${RESULTS_DIR}"
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

log "Resource preflight"
"$PYTHON_BIN" scripts/inspect_resources.py --paths / /tmp /dev/shm "$(pwd)" | tee "$RESULTS_DIR/preflight.txt" || true
nvidia-smi | tee "$RESULTS_DIR/nvidia-smi.txt" || true

if [ ! -f requirements-datasphere.txt ]; then
  echo "requirements-datasphere.txt not found; run from repository root." >&2
  exit 2
fi

log "Installing local package"
"$PYTHON_BIN" -m pip install -e . --no-cache-dir

if [ ! -d upstream_orthrus/.git ]; then
  log "Cloning official Orthrus repository"
  git clone --progress https://github.com/chiennv2000/orthrus upstream_orthrus
else
  log "Official Orthrus repository already exists"
fi

log "Starting FlowDraft hardCE quick job"
VENV_DIR="" \
PYTHON_BIN="$PYTHON_BIN" \
OUT_DIR="$OUT_DIR" \
DATA_DIR="$DATA_DIR" \
EVAL_DATA_DIR="$EVAL_DATA_DIR" \
MAX_STEPS="$MAX_STEPS" \
NUM_ANCHOR_BLOCKS="$NUM_ANCHOR_BLOCKS" \
GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" \
MAX_SEQUENCES="$MAX_SEQUENCES" \
EVAL_SEQUENCES="$EVAL_SEQUENCES" \
FLOW_STEPS="$FLOW_STEPS" \
BENCH_DTYPE="$BENCH_DTYPE" \
BENCH_ATTN_IMPLEMENTATION="$BENCH_ATTN_IMPLEMENTATION" \
BENCH_REQUIRE_PARITY="$BENCH_REQUIRE_PARITY" \
HF_REPO_ID="$HF_REPO_ID" \
HF_RUN_PATH="$HF_RUN_PATH" \
FLOW_STATE_MIN=0.0 \
FLOW_STATE_MAX=0.0 \
KL_REDUCTION="$KL_REDUCTION" \
HARD_CE_WEIGHT="$HARD_CE_WEIGHT" \
PREFIX_LOSS_WEIGHT="$PREFIX_LOSS_WEIGHT" \
PREFIX_KL_WEIGHT="$PREFIX_KL_WEIGHT" \
PREFIX_WEIGHT_DECAY="$PREFIX_WEIGHT_DECAY" \
CONSISTENCY_WEIGHT="$CONSISTENCY_WEIGHT" \
CONSISTENCY_START_STEP="$CONSISTENCY_START_STEP" \
CLEAN_OUTPUT=1 \
bash datasphere/run_flowdraft_quick_compare_venv.sh

log "Collecting metrics"
find "$OUT_DIR" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.txt' \) -print0 |
  while IFS= read -r -d '' file; do
    cp "$file" "$RESULTS_DIR/"
done

if [ -f "$OUT_DIR/run.log" ]; then
  cp "$OUT_DIR/run.log" "$RESULTS_DIR/"
fi

if [ -d "$OUT_DIR/best" ]; then
  find "$OUT_DIR/best" -maxdepth 1 -type f \( -name 'config.json' -o -name 'generation_config.json' \) -print0 |
    while IFS= read -r -d '' file; do
      mkdir -p "$RESULTS_DIR/best"
      cp "$file" "$RESULTS_DIR/best/"
    done
fi

if [ "$KEEP_CHECKPOINTS" = "1" ]; then
  log "KEEP_CHECKPOINTS=1; copying best/last checkpoints into outputs"
  mkdir -p "$RESULTS_DIR/checkpoints"
  cp -a "$OUT_DIR/best" "$RESULTS_DIR/checkpoints/" 2>/dev/null || true
  cp -a "$OUT_DIR/last" "$RESULTS_DIR/checkpoints/" 2>/dev/null || true
fi

log "Result files"
find "$RESULTS_DIR" -maxdepth 3 -type f -print | sort
log "FlowDraft hardCE job complete"
