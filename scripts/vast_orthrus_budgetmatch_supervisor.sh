#!/usr/bin/env bash
set -euo pipefail

source /opt/supervisor-scripts/utils/environment.sh
cd /workspace/FlowDraft-cfmproper

export OUT_DIR="${OUT_DIR:-/dev/shm/flowdraft_runs/orthrus_budgetmatch_20000_r1}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export CONFIG_PATH="${CONFIG_PATH:-configs/orthrus_budgetmatch_20000.yaml}"
export HF_REPO_ID="${HF_REPO_ID:-Yaroslav574389/FlowDraft-EagleFlow-Qwen3-1.7B}"
export HF_RUN_PATH="${HF_RUN_PATH:-runs/orthrus_budgetmatch_20000_r1}"

exec bash scripts/run_vast_orthrus_budgetmatch.sh
