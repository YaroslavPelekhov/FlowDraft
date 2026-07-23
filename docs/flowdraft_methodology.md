# FlowDraft methodology

This document defines what may be called FlowDraft in this repository and the
minimum protocol for reporting its results.

## Method definition

FlowDraft keeps the frozen Qwen3 autoregressive path and the Orthrus verifier.
The baseline trains only `q_proj_diff`, `k_proj_diff`, and `v_proj_diff`.
`cfm_v2` additionally trains a small residual state adapter that maps continuous
categorical states into the frozen Qwen embedding space. Therefore,
verification remains lossless: greedy output must be token-identical to the
frozen AR path, independently of drafter quality.

The drafter is a categorical endpoint flow map, not repeated masked decoding.
For source and target times `0 <= s <= t <= 1`, it predicts a simplex endpoint
`pi_s,t(x_s)` and applies

```text
X_s,t(x_s) = x_s + (t - s) / (1 - s) * (pi_s,t(x_s) - x_s).
```

The `cfm_v2` implementation has the defining CFM components:

- RMS-normalized residual state adapter with FiLM `(s,t)` conditioning;
- straight-line categorical states between the mask prior and clean endpoints;
- diagonal endpoint training at `s=t`;
- off-diagonal endpoint-consistent Lagrangian distillation (ECLD);
- exact full-vocabulary endpoint expectation for off-diagonal transport;
- explicit `(s,t)=(0,1)` supervision for the one-jump inference boundary;
- one- or few-jump inference through the same endpoint transport equation;
- unchanged frozen AR verification after drafting.

## Objective

Training is staged, but the primary CFM ablation isolates the two terms below.

1. Teacher-VFM stage: all anchors use diagonal states `x_t`. The primary loss
   is forward KL from the frozen AR distribution.
2. Flow-map stage: 50% of blocks remain diagonal, 37.5% are exact
   one-jump `(0,1)` proposals, and 12.5% use interior `s<t` pairs. The
   exact boundary proposals are supervised directly by the frozen AR teacher
   and a prefix-weighted CE term. ECLD is applied only to interior pairs:

```text
L_ECLD = 4 * w_t * CE(stopgrad(pi_t,t(X_s,t)), pi_s,t(x_s))
         + 2 * gamma^2 * ||d pi_s,t(x_s) / dt||^2,
gamma = (t - s) / (1 - s).
```

The default stage-2 loss is

```text
L = L_diagonal_teacher + lambda_direct * L_one_jump_teacher
    + lambda_prefix * L_one_jump_prefix + lambda_ECLD * L_ECLD.
```

`w_t=(1-t)^-2` is clamped at a denominator of `0.05` in the paper-style
configuration. A uniform-weight variant is retained only as an explicit
stability ablation. The boundary pairs are excluded from ECLD because its
time weight is singular at `t=1`. Every run records the unweighted loss and
the weighted contribution of each term, so a scale mismatch is visible before
benchmarking.

## Single-A100 approximations

One numerical approximation is explicit in run metadata:

- The ECLD temporal derivative is a boundary-safe finite difference. Full-model JVP
  through the released FlexAttention kernel is not supported by the target
  PyTorch stack.

The former top-32 simplex approximation is retained only for legacy checkpoint
reproduction. `cfm_v2` uses a chunked complete-vocabulary expectation. The
finite-difference derivative must still be disclosed when comparing against the
original CFM paper.

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
3. paper-clamped versus uniform ECLD time weights;
4. one jump versus two jumps at equal measured forward-pass cost;
5. base CFM versus a separately introduced prefix-acceptance auxiliary;
6. offline training versus verification-error replay on draft-induced states.

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
