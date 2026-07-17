# Orthrus Training Reconstruction for Qwen3-1.7B

This project reconstructs a practical training pipeline for the smallest Orthrus model, Qwen3-1.7B, on top of the official inference repository:

- Official code: https://github.com/chiennv2000/orthrus
- Paper: https://arxiv.org/abs/2605.12825
- Target hardware: one NVIDIA A100 80 GB in Yandex DataSphere

The official repository currently ships the model architecture and inference path, but not the public training pipeline. This repo adds the missing training layer and clones `upstream_orthrus/` from the official code during setup.

## What Is Reconstructed

From the paper and upstream code:

- Base model: `Qwen/Qwen3-1.7B`
- Orthrus block size: `K = 32`
- Context length: `L = 2048`
- Diffusion training signal: forward KL from the frozen AR teacher distribution
- Training data domains: balanced `chat`, `math`, `code` from `nvidia/Nemotron-Post-Training-Dataset-v2`
- Frozen backbone: by default only `q_proj_diff`, `k_proj_diff`, and `v_proj_diff` are trainable, matching the paper's statement that `WQdiff`, `WKdiff`, and `WVdiff` are updated
- Training mask: each diffusion block attends to clean AR cache before its anchor and bidirectionally inside its own corrupted block

The paper-scale run uses 471,952 packed sequences, 2 epochs, 256 anchor blocks per sequence, and was trained on a single 8xH200 node. The A100 config in this repo intentionally reduces anchor blocks and total sequences for a run that is realistic to iterate on with one GPU.

## Repository Layout

```text
upstream_orthrus/          official Orthrus clone, created by datasphere/setup.sh
orthrus_training/          training utilities
scripts/prepare_dataset.py packs Nemotron examples into fixed token shards
scripts/train_orthrus.py   KL distillation training loop
scripts/evaluate_lossless.py greedy AR vs Orthrus parity check
configs/                  smoke, A100, and paper-hparam configs
datasphere/               Yandex DataSphere setup and run scripts
```

## Quick Start

```bash
python -m pip install -r requirements-datasphere.txt
git clone https://github.com/chiennv2000/orthrus upstream_orthrus
python -m pip install -e .
```

Prepare a small smoke dataset:

```bash
python scripts/prepare_dataset.py \
  --output-dir data/smoke_packed \
  --seq-len 2048 \
  --max-sequences 8 \
  --shard-size 8
```

Run a short training smoke test:

```bash
python scripts/train_orthrus.py --config configs/smoke.yaml
```

Run the practical A100 recipe:

```bash
bash datasphere/run_a100.sh
```

For Yandex DataSphere, the cleaner isolated option is:

```bash
bash datasphere/setup_venv.sh
bash datasphere/run_a100_venv.sh
```

Check greedy lossless parity after training:

```bash
python scripts/evaluate_lossless.py \
  --checkpoint outputs/orthrus-qwen3-1p7b-a100/final \
  --max-new-tokens 128
```

## A100 Recipe

`configs/a100_80gb.yaml` uses:

- `batch_size: 1`
- `gradient_accumulation_steps: 32`
- `num_anchor_blocks: 32`
- `block_size: 32`
- `max_steps: 10000`
- `learning_rate: 2e-4`
- `warmup_ratio: 0.05`
- `dtype: bf16`

This is a compute-aware reconstruction, not a claim that one A100 reproduces the full paper checkpoint in the same wall time. To move toward the paper setup, increase:

1. `--max-sequences` in dataset preparation toward `471952`
2. `num_anchor_blocks` toward `256`
3. `epochs` toward `2`

## Notes

- The Nemotron dataset is gated on Hugging Face. Accept its terms and set `HF_TOKEN`.
- The mask token id is `151669`, matching the released Orthrus Qwen3-1.7B config. It is an unused Qwen3 vocabulary row, not a tokenizer-emitted text token.
- Saved checkpoints include `modeling_orthrus.py` copied from the official upstream repository and can be loaded with `trust_remote_code=True`.
- The train loop uses full-vocabulary soft KL. This is heavier than hard-label CE but matches the paper's finding that soft distillation gives better acceptance speed.
