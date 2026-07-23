# Long Prefix-Aligned Categorical Flow Run (v5 r3)

## Purpose

This is the long, verifier-anchored Orthrus-style categorical flow-map
experiment.  It trains the diffusion drafter against the frozen Qwen verifier
on disjoint packed Nemotron splits, with prefix-weighted teacher matching and
endpoint consistency.

## Protocol

- Train/evaluation manifests: 49,750 / 2,048 packed sequences, checked
  disjoint before training.
- Training: 128 anchors, gradient accumulation 8, 2,000 optimizer steps.
- The run resumed from the valid v5 r2 checkpoint at step 200.  Its ECLD
  partition was adjusted so every FlexAttention sub-problem has a supported
  token multiple.
- Checkpoints: only atomic `best` and `last` directories are kept remotely.
  Model weight files are intentionally not committed to Git.

## Best Holdout Result

The selected checkpoint is step 1,800:

- greedy accepted length: 2.1938
- first-token accuracy: 0.9375

## Strict Inference Gate

The eager FP32 Quick-10 benchmark used the frozen verifier to emit every
accepted token and compared complete greedy continuations against plain AR:

- parity: 100% (10/10 prompts)
- aggregate TPF: 1.3943
- aggregate wall-clock speedup: 1.2965x
- weighted accepted draft tokens: 1.8468

## Conclusion

The training objective improves acceptance, but this architecture still makes
one full diffusion-Qwen pass plus one full AR-verifier pass per draft cycle.
The second full model pass caps end-to-end throughput well below the 2x target.
It is retained as a reproducible negative throughput result and motivates the
one-verifier advanced-token feature-flow experiments.
