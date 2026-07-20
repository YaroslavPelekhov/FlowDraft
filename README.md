# Orthrus Training Reconstruction for Qwen3-1.7B

This project reconstructs a practical training pipeline for the smallest Orthrus model, Qwen3-1.7B, on top of the official inference repository:

- Official code: https://github.com/chiennv2000/orthrus
- Paper: https://arxiv.org/abs/2605.12825
- Target hardware: one NVIDIA A100 80 GB in Yandex DataSphere

The official repository currently ships the model architecture and inference path, but not the public training pipeline. This repo adds the missing training layer and clones `upstream_orthrus/` from the official code during setup.

For the paper-grade training/evaluation protocol, see
[`docs/official_benchmarks.md`](/Users/yaroslavpelehov/Downloads/FlowDraft/docs/official_benchmarks.md) and
[`configs/eval/orthrus_paper_suite.yaml`](/Users/yaroslavpelehov/Downloads/FlowDraft/configs/eval/orthrus_paper_suite.yaml).

## What Is Reconstructed

From the paper and upstream code:

- Base model: `Qwen/Qwen3-1.7B`
- Orthrus block size: `K = 32`
- Context length: `L = 2048`
- Diffusion training signal: forward KL from the frozen AR teacher distribution
- Training data domains: balanced `chat`, `math`, `code` from `nvidia/Nemotron-Post-Training-Dataset-v2`
- Frozen backbone: by default only `q_proj_diff`, `k_proj_diff`, and `v_proj_diff` are trainable, matching the paper's statement that `WQdiff`, `WKdiff`, and `WVdiff` are updated
- Training mask: each diffusion block attends to clean AR cache before its anchor and bidirectionally inside its own corrupted block

The paper-scale run uses 600k training examples, 2 epochs, 256 anchor blocks per packed sequence, and was trained on a single 8xH200 node. The A100 config in this repo intentionally reduces anchor blocks and total sequences for a run that is realistic to iterate on with one GPU.

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

## Quick Comparable Run

For a result in roughly a couple of hours on one A100 80 GB, use the quick compare path. It trains a small Orthrus-Qwen3-1.7B checkpoint and immediately benchmarks AR vs Orthrus greedy decoding.

```bash
VENV_DIR=/tmp/flowdraft_venv bash datasphere/setup_venv.sh
HF_TOKEN=hf_... VENV_DIR=/tmp/flowdraft_venv bash datasphere/run_quick_compare_venv.sh
```

Defaults:

- `MAX_SEQUENCES=20000`
- `MAX_STEPS=600`
- `BENCH_TOKENS=128`
- `BENCH_DTYPE=bf16`, `BENCH_ATTN_IMPLEMENTATION=sdpa` for fast smoke benchmarking
- `BENCH_REQUIRE_PARITY=0`, set to `1` for strict lossless CI-style checks
- `REBUILD_DATA=0`, so existing packed data is reused
- `CLEAN_OUTPUT=0`, set to `1` for a fresh output directory
- packed data under `/tmp/flowdraft_storage`
- model outputs under `/dev/shm/flowdraft_runs/orthrus_quick2h` by default

Outputs:

- `/dev/shm/flowdraft_runs/orthrus_quick2h/train_metrics.jsonl`
- `/dev/shm/flowdraft_runs/orthrus_quick2h/best`
- `/dev/shm/flowdraft_runs/orthrus_quick2h/best_metrics.json`
- `/dev/shm/flowdraft_runs/orthrus_quick2h/last`
- `/dev/shm/flowdraft_runs/orthrus_quick2h/last_metrics.json`
- `/dev/shm/flowdraft_runs/orthrus_quick2h/benchmark_metrics.jsonl`
- `/dev/shm/flowdraft_runs/orthrus_quick2h/benchmark_summary.json`
- `/dev/shm/flowdraft_runs/orthrus_quick2h/benchmark_best_summary.json`

You can shrink it for a fast check:

```bash
MAX_SEQUENCES=1000 MAX_STEPS=32 BENCH_TOKENS=64 \
  VENV_DIR=/tmp/flowdraft_venv bash datasphere/run_quick_compare_venv.sh
```

The training loop writes periodic train metrics, saves `best/` by lowest quick eval KL loss, and saves `last/` every `SAVE_EVERY` steps plus at the end. It never writes numbered checkpoint directories in the quick path, so only `best/` and `last/` exist at any time. The benchmark reports exact greedy parity, AR tokens/sec, Orthrus tokens/sec, speedup, Orthrus tokens per forward pass, acceptance length statistics, and ratios/gaps against the Qwen3-1.7B paper target speedup of 4.25x. This is not paper-scale training, but it gives reproducible numbers for comparing checkpoints and deciding whether a longer run is worth it.

For paper-grade greedy lossless checks, benchmark in deterministic strict mode:

```bash
BENCH_DTYPE=fp32 BENCH_ATTN_IMPLEMENTATION=eager BENCH_REQUIRE_PARITY=1 \
  VENV_DIR=/tmp/flowdraft_venv bash datasphere/run_quick_compare_venv.sh
```

`eval_prompts/quick_compare.jsonl` is still only a small sanity set. The official
paper suite is GSM8K, MATH-500, AIME24, AIME25, HumanEval, MBPP, Pseudo2code,
and LiveCodeBench-v5.

Run inference from the current best checkpoint:

```bash
/tmp/flowdraft_venv/bin/python scripts/infer_orthrus.py \
  --checkpoint /dev/shm/flowdraft_runs/orthrus_quick2h/best \
  --prompt "Solve: if a rectangle has length 12 and width 7, what is its area?" \
  --max-new-tokens 256
```

Use `--mode ar` for the sequential baseline and `--mode diffusion` for Orthrus generation.

## FlowDraft MVP

FlowDraft replaces the Orthrus masked-diffusion drafting call with a categorical flow-map style endpoint drafter while keeping the frozen AR backbone and AR verifier unchanged. The first MVP trains the same trainable diffusion projections with endpoint teacher distillation from the frozen AR head, plus an optional late consistency term.

Run the matched 600-step quick comparison:

```bash
HF_TOKEN=hf_... VENV_DIR=/tmp/flowdraft_venv CLEAN_OUTPUT=1 \
  bash datasphere/run_flowdraft_quick_compare_venv.sh
```

Defaults:

- output under `/dev/shm/flowdraft_runs/flowdraft_quick2h`
- same packed Nemotron data as the Orthrus quick run
- `MAX_STEPS=600`
- `block_size=32`
- `BENCH_DTYPE=bf16`, `BENCH_ATTN_IMPLEMENTATION=sdpa` for fast smoke benchmarking
- `BENCH_REQUIRE_PARITY=0`, set to `1` with `BENCH_DTYPE=fp32 BENCH_ATTN_IMPLEMENTATION=eager` for strict lossless checks
- `FLOW_STATE_MIN=0.0`, `FLOW_STATE_MAX=0.0` for one-jump drafting from the masked prior
- `HARD_CE_WEIGHT=0.5` to bias the endpoint drafter toward greedy top-1 agreement
- `CONSISTENCY_WEIGHT=0.0` in the default quick run; enable it only after the CE baseline is healthy
- `FLOW_STEPS="1"` for the primary one-jump benchmark
- only `best/` and `last/` checkpoints are kept

Compare against the Orthrus quick baseline:

```bash
cat /dev/shm/flowdraft_runs/orthrus_quick2h/benchmark_best_summary.json
cat /dev/shm/flowdraft_runs/flowdraft_quick2h/benchmark_best_flow1_summary.json
```

The main numbers are `mean_speedup`, `mean_flowdraft_tpf`, `mean_acceptance`, and `parity_rate`. A useful first success criterion is FlowDraft `flow_steps=1` beating the Orthrus quick baseline's TPF/speedup without reducing parity.

For paper-grade FlowDraft parity:

```bash
BENCH_DTYPE=fp32 BENCH_ATTN_IMPLEMENTATION=eager BENCH_REQUIRE_PARITY=1 \
  FLOW_STEPS=1 VENV_DIR=/tmp/flowdraft_venv bash datasphere/run_flowdraft_quick_compare_venv.sh
```

Inspect available disk/GPU resources before a longer run:

```bash
/tmp/flowdraft_venv/bin/python scripts/inspect_resources.py --paths / /tmp /dev/shm /home/jupyter
```

For a fresh quick run that reuses packed data but deletes old model outputs:

```bash
HF_TOKEN=hf_... VENV_DIR=/tmp/flowdraft_venv CLEAN_OUTPUT=1 \
  bash datasphere/run_quick_compare_venv.sh
```

Check greedy lossless parity after training:

```bash
python scripts/evaluate_lossless.py \
  --checkpoint /dev/shm/flowdraft_runs/orthrus_quick2h/best \
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

1. `--max-sequences` in dataset preparation toward the paper-scale 600k-example regime
2. `num_anchor_blocks` toward `256`
3. `epochs` toward `2`

## Notes

- The Nemotron dataset is gated on Hugging Face. Accept its terms and set `HF_TOKEN`.
- The mask token id is `151669`, matching the released Orthrus Qwen3-1.7B config. It is an unused Qwen3 vocabulary row, not a tokenizer-emitted text token.
- Saved checkpoints include `modeling_orthrus.py` copied from the official upstream repository and can be loaded with `trust_remote_code=True`.
- The train loop uses full-vocabulary soft KL. This is heavier than hard-label CE but matches the paper's finding that soft distillation gives better acceptance speed.
