#!/usr/bin/env bash
set -euo pipefail

# Strict, directly comparable efficiency run for the released Orthrus checkpoint.
# It reuses the immutable AIME25/HumanEval prompt files materialized for EagleFlow.

PYTHON_BIN="${PYTHON_BIN:-/workspace/flowdraft_venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-chiennv/Orthrus-Qwen3-1.7B}"
PROMPT_DIR="${PROMPT_DIR:-/workspace/flowdraft_runs/eagleflow_parallel_continue_20000_r1/paper_eval}"
OUT_DIR="${OUT_DIR:-/workspace/flowdraft_runs/official_orthrus_qwen3_1p7b/paper_eval}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

export HF_HOME="${HF_HOME:-/dev/shm/flowdraft_hf}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/dev/shm/flowdraft_hf_modules}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/flowdraft_xdg_cache}"
mkdir -p "$OUT_DIR"

run_task() {
  local task="$1"
  local paper_tpf="$2"
  local paper_speedup="$3"
  "$PYTHON_BIN" scripts/benchmark_orthrus.py \
    --checkpoint "$CHECKPOINT" \
    --prompts-jsonl "$PROMPT_DIR/${task}_prompts.jsonl" \
    --output-jsonl "$OUT_DIR/${task}_fp32_metrics.jsonl" \
    --summary-json "$OUT_DIR/${task}_fp32_summary.json" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --dtype fp32 \
    --attn-implementation eager \
    --paper-tpf-target "$paper_tpf" \
    --paper-speedup-target "$paper_speedup" \
    --no-official-parity-on-mismatch \
    --require-parity
}

run_task aime25 3.89 4.80
run_task humaneval 2.75 3.07
