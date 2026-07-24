# Budget-Matched Orthrus Baseline

Clean Orthrus reconstruction trained from `Qwen/Qwen3-1.7B` with the same packed-data budget as the validated FlowDraft run. The frozen AR verifier is identical in every efficiency measurement.

## Training protocol

- Objective: Orthrus forward-KL AR distillation only.
- Train / holdout: 49,750 / 2,048 distinct packed Nemotron sequences; disjointness is checked before training.
- Budget: 20,000 optimizer updates, batch size 1, 64 anchor blocks, block size 32.
- Validation: 32 fixed holdout batches with fixed CUDA anchor seed 4284; `best` is selected by greedy prefix acceptance.
- Best checkpoint: step 1,000, validation prefix acceptance 0.324707.
- Inference: FP32, eager attention, greedy decoding, 128-token cap, strict custom parity required.

## Fixed-Prompt Efficiency Comparison

TPF is aggregate generated tokens divided by aggregate frozen-verifier forward passes. Speedup is aggregate AR wall-clock time divided by aggregate accelerated wall-clock time. These are efficiency measurements only; task accuracy/pass@1 is not scored here.

| Method | AIME25 TPF | AIME25 speedup | AIME25 parity | HumanEval TPF | HumanEval speedup | HumanEval parity |
|---|---:|---:|---:|---:|---:|---:|
| FlowDraft (ours, endpoint Flow Map) | 1.761 | 1.493x | 100% | 1.647 | 1.394x | 100% |
| Orthrus (ours, clean budget-matched reconstruction) | 1.371 | 1.276x | 100% | 1.591 | 1.476x | 100% |
| Orthrus (released reference checkpoint) | 3.459 | 3.124x | 100% | 4.559 | 4.090x | 100% |

Mean accepted draft length is retained in the raw per-prompt metrics, but it is not a cross-method headline metric. FlowDraft uses a lightweight external drafter, while Orthrus' proposal loop incurs an additional model pass. Thus the same accepted length has a different forward-pass cost; TPF and wall-clock speedup are the comparable quantities.

The released checkpoint is a reference implementation, not a budget-matched training comparison: it was trained by the authors at substantially larger scale. It is evaluated on the same fixed prompts and inference backend as the two local methods.

## Artifacts

- `run_config.json`, `run_manifest.json`: exact training configuration and status.
- `train_metrics.jsonl`: all train and fixed-holdout observations.
- `best_metrics.json`, `last_metrics.json`: selected checkpoint and final state.
- `paper_eval/`: raw per-prompt results and aggregate summaries for AIME25 and HumanEval.
- `hf_upload_manifest.json`: private Hugging Face archive locations for atomic `best` and `last` weights.
