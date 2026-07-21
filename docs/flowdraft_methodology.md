# FlowDraft methodology

This document defines what may be called FlowDraft in this repository and the
minimum protocol for reporting its results.

## Method definition

FlowDraft keeps the frozen Qwen3 autoregressive path and the Orthrus verifier.
Only `q_proj_diff`, `k_proj_diff`, and `v_proj_diff` are trainable. Therefore,
verification remains lossless: greedy output must be token-identical to the
frozen AR path, independently of drafter quality.

The drafter is a categorical endpoint flow map, not repeated masked decoding.
For source and target times `0 <= s <= t <= 1`, it predicts a simplex endpoint
`pi_s,t(x_s)` and applies

```text
X_s,t(x_s) = x_s + (t - s) / (1 - s) * (pi_s,t(x_s) - x_s).
```

The implementation has the defining CFM components:

- explicit `(s,t)` sinusoidal conditioning of the partial denoiser;
- straight-line categorical states between the mask prior and clean tokens;
- diagonal endpoint training at `s=t`;
- off-diagonal endpoint-consistent Lagrangian distillation (ECLD);
- one- or few-jump inference through the same endpoint transport equation;
- unchanged frozen AR verification after drafting.

## Objective

Training is staged.

1. Teacher-matching stage: all anchors use diagonal states. The primary loss is
   forward KL from the frozen AR distribution. Prefix-weighted KL and a small
   hard-label CE term emphasize early positions without replacing the soft
   teacher objective.
2. Flow-map stage: 75% of blocks remain diagonal and 25% use `s<t`. The
   off-diagonal objective is

```text
L_ECLD = 4 * CE(stopgrad(pi_t,t(X_s,t)), pi_s,t(x_s))
         + 2 * gamma^2 * ||d pi_s,t(x_s) / dt||^2,
gamma = (t - s) / (1 - s).
```

The default loss is

```text
L = L_teacher_KL
    + 0.10 * L_hard_CE
    + 0.25 * L_prefix_CE
    + 0.50 * L_prefix_teacher_KL
    + 0.10 * L_ECLD.
```

This choice follows two empirical findings: Orthrus reports better drafting
from soft teacher KL than hard CE, while Categorical Flow Maps reports stable
low-NFE generation from ECLD with a 0.75 diagonal fraction.

## Single-A100 approximations

Two numerical approximations are explicit in run metadata:

- The Qwen vocabulary simplex is projected to embedding space using a
  renormalized top-32 distribution. A full `151k x hidden_size` expectation at
  every block position is too expensive for one A100.
- The ECLD temporal derivative is a forward finite difference. Full-model JVP
  through the released FlexAttention kernel is not supported by the target
  PyTorch stack.

These approximations do not change the endpoint flow-map definition or the AR
verifier. They must be disclosed when comparing against the original CFM paper.

## Data protocol

- Training data: `nvidia/Nemotron-Post-Training-Dataset-v2`, round-robin
  `chat/math/code`, packed to 2048 tokens.
- Validation must continue after the training range using
  `--skip-sequences`; it must never restart the stream from sequence zero.
- `train_flowdraft.py` hashes packed rows and refuses to train if train and
  validation contain any identical sequence.
- Benchmark prompts must not be used for checkpoint selection.

## Reporting protocol

A benchmark row is valid only when accelerated output is token-identical to AR.
Report the lossless gate before efficiency:

```text
parity_rate = identical AR/FlowDraft outputs / all prompts = 1.0.
```

Aggregate efficiency exactly over the run:

```text
TPF = total generated tokens / total FlowDraft forward passes
speedup = aggregate FlowDraft tokens/s / aggregate AR tokens/s.
```

Do not use the unweighted mean of per-prompt TPF as the headline metric. Always
also report task accuracy/pass@1, max-token truncation count, hardware, dtype,
attention backend, flow jumps, prompt count, and raw JSONL outputs.

Paper comparisons use task-specific Qwen3-1.7B targets. The Table 1 average is
not a valid target for an individual task. `eval_prompts/quick_compare.jsonl`
is a smoke test and must not be described as an official benchmark.

## Current improvement path

The next statistically meaningful ablations are:

1. teacher KL only versus prefix-weighted KL;
2. CFM diagonal training versus ECLD;
3. top-k endpoint projection at 16, 32, and 64;
4. one jump versus two jumps at equal measured forward-pass cost;
5. offline training versus verification-error replay on draft-induced states.

The last item follows speculative-drafter literature: offline distillation can
plateau because inference visits states induced by drafter errors. It should be
implemented as a separate stage after the current ECLD run, not mixed into the
first controlled comparison.

## Primary sources

- Orthrus: https://arxiv.org/abs/2605.12825
- Categorical Flow Maps: https://arxiv.org/abs/2602.12233
- Flow Maps via Self-Distillation: https://arxiv.org/abs/2505.18825
- DistillSpec: https://openreview.net/forum?id=rsY6J3ZaTF
- Draft-OPD: https://arxiv.org/abs/2605.29343
