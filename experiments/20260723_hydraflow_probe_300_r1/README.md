# HydraFlow Self-Conditioned Feature Probe (2026-07-23)

This probe tests a sequentially-dependent alternative to the direct CacheFlow
endpoint predictor. The 33,628,160-parameter HydraFlow head transports the
verified final Qwen feature through 31 latent transitions. Each transition is
conditioned on the previous predicted continuous token embedding. The frozen
Qwen vocabulary head is then applied once to the complete trajectory, and one
ordinary Qwen causal pass verifies the candidate block.

## Protocol

Only HydraFlow is trainable; the parent is the frozen
`flowdraft_v4_full_300/best` adapter. Training used 49,750 packed sequences
and content-disjoint model selection used 2,048 packed sequences. The intended
budget was 300 updates, but early stopping selected step 100 after three later
holdout checks failed to improve it. The model used scheduled teacher forcing
from 1.0 to 0.1 and was evaluated in fully self-conditioned mode.

The final benchmark uses FP32 eager attention, 10 independent prompts, and 64
generated tokens per non-EOS prompt. It requires exact greedy parity. TPF counts
only prefill and frozen-Qwen verifier forwards; head calls are recorded
separately and wall-clock includes their full cost.

## Result

| Metric | Result |
| --- | ---: |
| Greedy parity | 100% (10/10) |
| Best held-out prefix acceptance | 0.1055 at step 100 |
| Aggregate target-model TPF | 1.078 |
| Aggregate wall-clock speedup | 0.698x |
| Weighted accepted draft tokens | 0.083 |
| Median accepted draft tokens | 0.0 |
| Frozen-Qwen forwards | 587 |
| HydraFlow head calls | 577 |

HydraFlow improves the held-out proxy over the 3.7M direct CacheFlow head but
does not translate into useful independent-prompt acceptance. Its sequential
Python-level latent rollout also imposes enough overhead to make wall-clock
slower than AR. This is a negative result: the implementation is lossless but
is not an inference acceleration.

## Artifacts

The remote run directory retains `best/` and `last/` 64 MB HydraFlow weights.
Git records their architecture configs, raw train metrics, and strict benchmark
metrics. The launcher uploads both weight directories to a private Hugging Face
model repository whenever `HF_TOKEN` and `HF_REPO_ID` are configured. This
environment had no HF token, so the automatic upload did not occur.
