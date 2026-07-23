#!/usr/bin/env bash
set -euo pipefail

source /workspace/FlowDraft-cfmproper/scripts/environment.sh
cd /workspace/FlowDraft-cfmproper

export OUT_DIR="${OUT_DIR:-/workspace/flowdraft_runs/hydraflow_advanced_token_5000_r1}"
export MAX_STEPS="${MAX_STEPS:-5000}"

exec bash scripts/run_vast_hydraflow_advanced_token.sh
