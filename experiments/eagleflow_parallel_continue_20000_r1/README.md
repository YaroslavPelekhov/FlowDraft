# EagleFlow Parallel Continuation, 20k Steps

This archive records the validated continuation run used for the current
EagleFlow result.  It stores configuration, train/holdout metrics, fixed prompt
sets, and strict FP32/eager efficiency measurements.  Checkpoint tensors are
intentionally excluded from Git; their local archive lives under
`outputs/archived_weights/eagleflow_parallel_continue_20000_r1/` and has been
mirrored to the private Hugging Face model repository recorded in
`hf_upload_manifest.json`.

## Training

- Method: parallel attention-conditioned endpoint Flow Map (`EagleFlow`)
- Base verifier: frozen parent `flowdraft_v5_prefix_ecld_2000_r3/best`
- Head warm start: `eagleflow_parallel_refine_3000_r1/best`
- Dataset: `nvidia/Nemotron-Post-Training-Dataset-v2`, packed 2,048-token
  sequences from `chat`, `math`, and `code`
- Train pool: 50,000 sequences; disjoint holdout: 2,048 following sequences
- Budget: 20,000 optimizer steps; selection metric: holdout greedy-prefix
  acceptance; selected checkpoint: step 14,500

## Strict Efficiency Results

All accelerated outputs are compared token-for-token with greedy AR outputs.
The benchmark uses `fp32`, eager attention, and 128 generated tokens per prompt.
These are losslessness and efficiency measurements, not AIME accuracy or
HumanEval pass@1 scores.

| Prompt source | Prompts | Parity | Aggregate speedup | Verifier TPF | Mean acceptance |
|---|---:|---:|---:|---:|---:|
| `math-ai/aime25` | 30 | 100% | 1.493x | 1.761 | 0.776 |
| `openai/openai_humaneval` | 164 | 100% | 1.394x | 1.647 | 0.664 |

The exact source revisions, prompts, per-prompt measurements, and summaries
are under `paper_eval/`.  Regenerate them with
`scripts/run_vast_eagleflow_paper_eval.sh`.
