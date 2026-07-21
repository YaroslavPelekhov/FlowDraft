# Official Orthrus / FlowDraft benchmark protocol

This file is the repository source of truth for separating paper-grade
reproduction from quick sanity checks.

## Paper sources

- Orthrus paper: https://arxiv.org/abs/2605.12825
- Orthrus code release: https://github.com/chiennv2000/orthrus
- Training dataset card: https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2

The paper evaluates Qwen3-1.7B, Qwen3-4B, and Qwen3-8B. For this project the
primary target is the smallest model, `Qwen/Qwen3-1.7B`.

## Official training recipe

The public paper specifies the following training setup for Orthrus:

- Dataset: `nvidia/Nemotron-Post-Training-Dataset-v2`
- Domains: `chat`, `math`, `code`
- Sampling ratio: uniform `1:1:1`
- Packed context length: `2048` tokens
- Parallel block size: `K = 32`
- Anchor blocks per packed sequence: `256`
- Training examples: `600000`
- Epochs: `2`
- Backbone: frozen AR Qwen3 model
- Trainable parameters: injected diffusion attention projections
- Objective: forward KL from the frozen AR teacher distribution
- Precision: `bf16`
- Scheduler: cosine with `0.05` warmup ratio
- Gradient clipping: `1.0`
- Paper hardware: one node with `8xH200`

FlowDraft keeps the same frozen AR backbone, same block size, same verifier, and
same data. The experimental change is only the drafter:

- Baseline Orthrus: single-step masked diffusion drafter
- FlowDraft: time-conditioned categorical endpoint flow-map drafter
- FlowDraft objective: diagonal AR-teacher endpoint distillation plus
  off-diagonal ECLD; the primary first point is `FLOW_STEPS=1`

## Official evaluation suite

The paper reports efficiency on these zero-shot tasks:

| Task | Primary source to use | Status in this repo |
|---|---|---|
| GSM8K | `openai/gsm8k` | needs official harness |
| MATH-500 | `HuggingFaceH4/MATH-500` | needs official harness |
| AIME24 | AIME 2024 set, e.g. `HuggingFaceH4/aime_2024` | needs official harness |
| AIME25 | AIME 2025 set, e.g. `math-ai/aime25` | needs official harness |
| HumanEval | `openai/human-eval` / `openai/openai_humaneval` | needs code harness |
| MBPP | Google Research MBPP, e.g. `google-research-datasets/mbpp` | needs code harness |
| Pseudo2code | LongProc `pseudo_to_code` | needs LongProc runner |
| LiveCodeBench-v5 | LiveCodeBench official runner | needs official runner |

`eval_prompts/quick_compare.jsonl` is not an official benchmark. It is only a
small generated prompt set for smoke testing training/inference health.

## Required paper-grade metrics

Every reported accelerated run must pass the lossless gate first:

- Greedy parity: `parity_rate = 1.0`
- Strict parity is required in the same dtype/backend used for the reported
  throughput. `fp32/eager` is an additional numerical audit, not a substitute
  for parity in the measured `bf16/sdpa` run.

After parity passes, report:

- Cycle-weighted average accepted length
- Aggregate tokens per forward pass (`total tokens / total forwards`)
- Aggregate wall-clock throughput speedup versus sequential AR
- AR tokens/sec and accelerated tokens/sec
- Task accuracy or pass rate inherited from the base AR run

## Paper target numbers for Qwen3-1.7B

For greedy decoding (`T = 0`), Table 1 reports:

| Task | TPF | Speedup |
|---|---:|---:|
| GSM8K | 4.20 | 4.37x |
| MATH-500 | 4.71 | 4.74x |
| AIME24 | 4.33 | 5.65x |
| AIME25 | 3.89 | 4.80x |
| HumanEval | 2.75 | 3.07x |
| MBPP | 2.76 | 3.07x |
| Pseudo2code | 4.60 | 4.90x |
| LiveCodeBench-v5 | 3.86 | 5.87x |
| Average | 3.89 | 4.25x |

For diverse sampling (`T = 1`), Table 1 reports:

| Task | TPF | Speedup |
|---|---:|---:|
| GSM8K | 3.88 | 4.04x |
| MATH-500 | 4.47 | 4.50x |
| AIME24 | 3.84 | 5.01x |
| AIME25 | 3.55 | 4.38x |
| HumanEval | 2.60 | 2.90x |
| MBPP | 2.88 | 3.20x |
| Pseudo2code | 4.37 | 4.65x |
| LiveCodeBench-v5 | 3.58 | 5.44x |
| Average | 3.65 | 4.27x |

## Strict benchmark commands

Orthrus baseline:

```bash
/tmp/flowdraft_venv/bin/python scripts/benchmark_orthrus.py \
  --checkpoint /dev/shm/flowdraft_runs/orthrus_quick2h/best \
  --prompts-jsonl eval_prompts/quick_compare.jsonl \
  --summary-json /dev/shm/flowdraft_runs/orthrus_quick2h/strict_summary.json \
  --max-new-tokens 128 \
  --dtype fp32 \
  --attn-implementation eager \
  --require-parity
```

FlowDraft one-step drafter:

```bash
/tmp/flowdraft_venv/bin/python scripts/benchmark_flowdraft.py \
  --checkpoint /dev/shm/flowdraft_runs/flowdraft_quick2h/best \
  --prompts-jsonl eval_prompts/quick_compare.jsonl \
  --summary-json /dev/shm/flowdraft_runs/flowdraft_quick2h/strict_flow1_summary.json \
  --max-new-tokens 128 \
  --flow-steps 1 \
  --dtype fp32 \
  --attn-implementation eager \
  --require-parity
```

For automated quick scripts, set:

```bash
BENCH_DTYPE=fp32 BENCH_ATTN_IMPLEMENTATION=eager BENCH_REQUIRE_PARITY=1
```

## Next implementation target

The next non-smoke milestone is a single official evaluator wrapper that runs
the same prompt construction through AR, Orthrus, and FlowDraft:

1. Generate outputs for each official benchmark task.
2. Enforce `parity_rate = 1.0` per task before reporting speed.
3. Save per-task JSONL metrics and one summary table with the paper targets.
4. Keep `eval_prompts/quick_compare.jsonl` only as a development sanity check.
