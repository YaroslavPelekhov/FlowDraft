#!/usr/bin/env bash
set -euo pipefail

source /opt/supervisor-scripts/utils/environment.sh
cd /workspace/FlowDraft-cfmproper

export OUT_DIR="${OUT_DIR:-/workspace/flowdraft_runs/eagleflow_parallel_continue_20000_r1}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export CONFIG_PATH="${CONFIG_PATH:-configs/eagleflow_parallel_continue_20000.yaml}"
export HEAD_INIT="${HEAD_INIT:-/workspace/flowdraft_runs/eagleflow_parallel_refine_3000_r1/best}"

exec bash scripts/run_vast_eagleflow.sh
