#!/usr/bin/env bash
set -euo pipefail

# Strict greedy efficiency evaluation on fixed official AIME25/HumanEval prompts.
# Task scoring is intentionally separate from this losslessness/throughput run.

PYTHON_BIN="${PYTHON_BIN:-/workspace/flowdraft_venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-/workspace/flowdraft_runs/eagleflow_parallel_continue_20000_r1/best}"
OUT_DIR="${OUT_DIR:-$(dirname "$CHECKPOINT")/paper_eval}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

export HF_HOME="${HF_HOME:-/dev/shm/flowdraft_hf}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/dev/shm/flowdraft_hf_modules}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/flowdraft_xdg_cache}"
mkdir -p "$OUT_DIR"

run_task() {
  local task="$1"
  "$PYTHON_BIN" scripts/prepare_paper_eval_prompts.py \
    --task "$task" \
    --output "$OUT_DIR/${task}_prompts.jsonl"
  "$PYTHON_BIN" scripts/benchmark_eagleflow.py \
    --checkpoint "$CHECKPOINT" \
    --prompts-jsonl "$OUT_DIR/${task}_prompts.jsonl" \
    --output-jsonl "$OUT_DIR/${task}_fp32_metrics.jsonl" \
    --summary-json "$OUT_DIR/${task}_fp32_summary.json" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --dtype fp32 \
    --attn-implementation eager \
    --require-parity
}

run_task aime25
run_task humaneval
