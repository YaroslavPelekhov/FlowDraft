# Residual dynamic-support CFM screen (negative control)

**Purpose.** This run tested whether a semantic retrieval residual over the
parent top-96 vocabulary support could improve the categorical-flow drafter
without changing the lossless Orthrus verifier.

**Protocol.** The run used 49,750 packed Nemotron training sequences and a
disjoint 2,048-sequence holdout. The selected checkpoint is the best holdout
greedy-prefix checkpoint. The final benchmark is strict eager FP32 greedy
decoding on the versioned ten-prompt sanity suite; every generated token is
still emitted only after the frozen AR verifier accepts it.

| Metric | Result |
| --- | ---: |
| Best holdout greedy prefix | 1.566 (step 120) |
| Candidate coverage at best | 82.2% |
| Greedy parity | 100% (10/10) |
| Aggregate TPF | 1.153 |
| Aggregate wall-clock speedup | 1.023x |
| Mean accepted draft tokens | 1.488 |

**Conclusion.** Retrieval enlarged the candidate support but did not improve
accepted prefixes relative to the simpler fixed-support CFM. More importantly,
the method still pays one frozen Qwen diffusion pass plus one verifier pass per
cycle. It is preserved as a reproducible negative result and is not a route to
the 2x target.

Files are copied directly from the remote run; `best` and `last` weights remain
on the GPU volume under `/workspace/flowdraft_runs/residual_support_flow_screen_160_r2`.
