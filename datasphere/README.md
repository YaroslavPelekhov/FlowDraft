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

Results are written to `/tmp/flowdraft_storage/orthrus_quick2h`. The run saves both `best/` by quick eval KL and `final/`, then benchmarks both when `best/` exists.
