#!/usr/bin/env bash
set -euo pipefail
source /opt/supervisor-scripts/utils/environment.sh
cd /workspace/FlowDraft-cfmproper
export PYTHON_BIN=/workspace/flowdraft_venv/bin/python
export TRAIN_MANIFEST=/workspace/flowdraft_data/nemotron_50k/manifest.json
export EVAL_MANIFEST=/workspace/flowdraft_data/nemotron_50k_holdout/manifest.json
export INIT_CHECKPOINT=/workspace/flowdraft_runs/flowdraft_v4_full_300/best
export OUT_DIR="${OUT_DIR:-/workspace/flowdraft_runs/dynamic_support_flow_probe_300_r1}"
export MAX_STEPS="${MAX_STEPS:-300}"
exec bash scripts/run_vast_dynamic_support_flow_probe.sh
