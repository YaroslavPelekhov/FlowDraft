# Yandex DataSphere runbook

Target: one NVIDIA A100 80 GB, Qwen3-1.7B, bfloat16.

1. Create a DataSphere project with an A100 80 GB resource.
2. Clone this repository into the notebook filesystem.
3. Accept the gated Hugging Face dataset terms for `nvidia/Nemotron-Post-Training-Dataset-v2`.
4. Set `HF_TOKEN` in the DataSphere secrets or notebook environment.
5. Run:

```bash
git clone <YOUR_REPO_URL> FlowDraft
cd FlowDraft
bash datasphere/setup_venv.sh
bash datasphere/run_a100_venv.sh
```

The default A100 config uses 50k packed 2048-token sequences, 32 Orthrus anchor blocks per sequence, and 10k optimizer steps. This is intentionally smaller than the paper's 0.96B-token, 2-epoch setup, because the paper used a single 8xH200 node. Increase `MAX_SEQUENCES`, `num_anchor_blocks`, and `max_steps` when you want a closer reproduction and can afford the runtime.

For a shorter comparable run with benchmark output:

```bash
VENV_DIR=/tmp/flowdraft_venv bash datasphere/setup_venv.sh
HF_TOKEN=hf_... VENV_DIR=/tmp/flowdraft_venv bash datasphere/run_quick_compare_venv.sh
```

Results are written to `/dev/shm/flowdraft_runs/orthrus_quick2h` by default. The run keeps only `best/` by quick eval KL and `last/` for the latest weights, then benchmarks both when `best/` exists. Existing packed data is reused unless `REBUILD_DATA=1`; old model outputs are removed only when `CLEAN_OUTPUT=1`.

Single-prompt inference from the best checkpoint:

```bash
/tmp/flowdraft_venv/bin/python scripts/infer_orthrus.py \
  --checkpoint /dev/shm/flowdraft_runs/orthrus_quick2h/best \
  --prompt "Solve: if a rectangle has length 12 and width 7, what is its area?" \
  --max-new-tokens 256
```

To run the matched FlowDraft MVP experiment after the Orthrus baseline:

```bash
HF_TOKEN=hf_... VENV_DIR=/tmp/flowdraft_venv CLEAN_OUTPUT=1 \
  bash datasphere/run_flowdraft_quick_compare_venv.sh
```

It writes to `/dev/shm/flowdraft_runs/flowdraft_quick2h` and benchmarks both one-jump and two-jump flow drafting. Compare `benchmark_best_flow1_summary.json` and `benchmark_best_flow2_summary.json` against the Orthrus `benchmark_best_summary.json`.
