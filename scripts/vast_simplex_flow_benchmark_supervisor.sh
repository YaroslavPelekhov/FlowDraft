#!/usr/bin/env bash
set -euo pipefail
source /opt/supervisor-scripts/utils/environment.sh
cd /workspace/FlowDraft-cfmproper
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/dev/shm/flowdraft_hf_modules}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/flowdraft_xdg}"
export TMPDIR="${TMPDIR:-/tmp/flowdraft_tmp}"
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN=/workspace/flowdraft_venv/bin/python
OUT_DIR="${OUT_DIR:-/workspace/flowdraft_runs/simplex_flow_probe_300_r1}"
CHECKPOINT="$OUT_DIR/best"
[ -f "$CHECKPOINT/simplex_flow.safetensors" ] || CHECKPOINT="$OUT_DIR/last"
exec "$PYTHON_BIN" scripts/benchmark_simplex_flow.py \
  --checkpoint "$CHECKPOINT" \
  --prompts-jsonl eval_prompts/quick_compare.jsonl \
  --output-jsonl "$OUT_DIR/benchmark_fp32_metrics.jsonl" \
  --summary-json "$OUT_DIR/benchmark_fp32_summary.json" \
  --max-new-tokens 64 \
  --flow-steps 2 \
  --dtype fp32 \
  --attn-implementation eager \
  --require-parity
