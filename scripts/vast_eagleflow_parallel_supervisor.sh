#!/usr/bin/env bash
set -euo pipefail

source /opt/supervisor-scripts/utils/environment.sh
cd /workspace/FlowDraft-cfmproper

export OUT_DIR="${OUT_DIR:-/workspace/flowdraft_runs/eagleflow_parallel_screen_500_r1}"
export MAX_STEPS="${MAX_STEPS:-500}"
export CONFIG_PATH="${CONFIG_PATH:-configs/eagleflow_parallel_screen_500.yaml}"

exec bash scripts/run_vast_eagleflow.sh
