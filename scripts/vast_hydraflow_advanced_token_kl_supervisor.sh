#!/usr/bin/env bash
set -euo pipefail

source /opt/supervisor-scripts/utils/environment.sh
cd /workspace/FlowDraft-cfmproper

export OUT_DIR="${OUT_DIR:-/workspace/flowdraft_runs/hydraflow_advanced_token_kl_5000_r3}"
export CONFIG_PATH="${CONFIG_PATH:-configs/hydraflow_advanced_token_kl_5000.yaml}"
export MAX_STEPS="${MAX_STEPS:-5000}"

exec bash scripts/run_vast_hydraflow_advanced_token.sh
