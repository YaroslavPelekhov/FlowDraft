# Local-Simplex CFM Probe

**Status:** completed on 2026-07-23, A100 40 GB, commit `0cfeeed` for training and `c5abe10` for the benchmark retry.

This is a categorical flow-map experiment rather than a hidden-state regressor:

- the frozen parent Orthrus drafter provides a context-conditioned top-128 token candidate simplex for every draft position;
- a 9.67M-parameter bidirectional partial denoiser learns the CFM endpoint map on that simplex;
- the objective is root endpoint CE, diagonal VFM CE, endpoint consistency (ECLD), temporal drift, and prefix-survival weighting;
- deployment uses one frozen Orthrus diffusion pass, two inexpensive simplex-flow steps, then the ordinary frozen AR verifier.

## Protocol

- Train: 49,750 packed Nemotron sequences; holdout: 2,048 disjoint packed sequences.
- Requested cap: 300 optimizer steps. Early stopping selected `best` at step 50 and stopped at step 200 after three non-improving holdout evaluations.
- Strict inference: FP32, eager attention, ten generated sanity prompts, 64 generated tokens per prompt, greedy decoding, and parity required.
- TPF counts frozen-Qwen forwards. The small CFM head calls are reported separately.

## Result

| Metric | Value |
| --- | ---: |
| Holdout best greedy accepted prefix | 1.672 |
| Holdout first-token accuracy at best | 92.6% |
| Holdout candidate coverage at best | 81.6% |
| Greedy parity | 100% (10/10) |
| Aggregate TPF | 1.153 |
| Aggregate wall-clock speedup | 1.049x |
| Weighted accepted length | 1.488 |
| Frozen-Qwen forwards | 549 |
| Local CFM head calls | 270 |

The results are a valid lossless CFM proof of implementation, but not a successful speedup result: the two flow-map steps add enough latency that the gain is only 4.9%. The largest visible limitation is candidate coverage: on the holdout, roughly 18% of frozen AR targets were not in the top-128 support selected by the parent drafter, so a local-simplex refiner cannot recover them.

The committed files contain the full protocol, training curve, per-prompt measurements, and logs. `best` and `last` binary head weights are intentionally excluded from Git; they remain in the remote run directory and should be uploaded to a private Hugging Face model repository using a currently configured HF credential.
