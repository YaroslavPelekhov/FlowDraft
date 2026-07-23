#!/usr/bin/env bash
set -euo pipefail

log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }
PYTHON_BIN="${PYTHON_BIN:-/workspace/flowdraft_venv/bin/python}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/dev/shm/flowdraft_hf_modules}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/flowdraft_xdg}"
export TMPDIR="${TMPDIR:-/tmp/flowdraft_tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-/workspace/flowdraft_data/nemotron_50k/manifest.json}"
EVAL_MANIFEST="${EVAL_MANIFEST:-/workspace/flowdraft_data/nemotron_50k_holdout/manifest.json}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/workspace/flowdraft_runs/flowdraft_v4_full_300/best}"
OUT_DIR="${OUT_DIR:-/workspace/flowdraft_runs/simplex_flow_rescue_probe_300_r2}"
BANK_DIR="${BANK_DIR:-/workspace/flowdraft_runs/simplex_flow_rescue_bank_512_r1}"
MAX_STEPS="${MAX_STEPS:-300}"
CALIBRATION_BATCHES="${CALIBRATION_BATCHES:-512}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_RUN_PATH="${HF_RUN_PATH:-}"

if [ -e "$OUT_DIR" ] && [ -n "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -printf x)" ]; then
  log "Refusing to overwrite existing output: $OUT_DIR"; exit 2
fi
if [ ! -f "$INIT_CHECKPOINT/adapter_config.json" ]; then
  log "Missing frozen parent checkpoint: $INIT_CHECKPOINT"; exit 2
fi
mkdir -p "$OUT_DIR"; exec > >(tee -a "$OUT_DIR/run.log") 2>&1
log "SimplexFlow rescue-support resource preflight"
"$PYTHON_BIN" scripts/inspect_resources.py --paths / /workspace /dev/shm
if [ ! -f "$BANK_DIR/rescue_bank.safetensors" ]; then
  log "Calibrating train-only rescue support at $BANK_DIR"
  "$PYTHON_BIN" scripts/build_rescue_candidate_bank.py \
    --init-checkpoint "$INIT_CHECKPOINT" \
    --train-manifest "$TRAIN_MANIFEST" \
    --output-dir "$BANK_DIR" \
    --calibration-batches "$CALIBRATION_BATCHES" \
    --base-candidate-count 96 \
    --rescue-count 32
else
  log "Reusing immutable rescue bank at $BANK_DIR"
fi
log "Training CFM on top-96 plus rescue-32 candidate support"
"$PYTHON_BIN" scripts/train_simplex_flow.py \
  --config configs/simplex_flow_rescue_probe.yaml \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --train-manifest "$TRAIN_MANIFEST" \
  --eval-manifest "$EVAL_MANIFEST" \
  --rescue-bank "$BANK_DIR" \
  --output-dir "$OUT_DIR" \
  --max-steps "$MAX_STEPS"
cp "$BANK_DIR"/rescue_bank_config.json "$OUT_DIR"/rescue_bank_config.json
cp "$BANK_DIR"/calibration_report.json "$OUT_DIR"/rescue_bank_calibration_report.json
CHECKPOINT="$OUT_DIR/best"; [ -f "$CHECKPOINT/simplex_flow.safetensors" ] || CHECKPOINT="$OUT_DIR/last"
log "Strict FP32 greedy losslessness gate with adaptive simplex support"
"$PYTHON_BIN" scripts/benchmark_simplex_flow.py \
  --checkpoint "$CHECKPOINT" \
  --prompts-jsonl eval_prompts/quick_compare.jsonl \
  --output-jsonl "$OUT_DIR/benchmark_fp32_metrics.jsonl" \
  --summary-json "$OUT_DIR/benchmark_fp32_summary.json" \
  --max-new-tokens 64 --flow-steps 2 --dtype fp32 --attn-implementation eager --require-parity
if [ -n "$HF_REPO_ID" ]; then
  log "Uploading best/last checkpoints and protocol to Hugging Face"
  ARGS=(--run-dir "$OUT_DIR" --repo-id "$HF_REPO_ID"); [ -z "$HF_RUN_PATH" ] || ARGS+=(--run-path "$HF_RUN_PATH")
  "$PYTHON_BIN" scripts/upload_checkpoints_hf.py "${ARGS[@]}"
fi
log "SimplexFlow rescue-support probe complete: $OUT_DIR"
