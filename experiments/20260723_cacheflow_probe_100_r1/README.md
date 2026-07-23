# CacheFlow One-Pass Trajectory Probe (2026-07-23)

This experiment tests a deliberately cheap alternative to a second Qwen
forward pass. A 3,691,776-parameter conditional endpoint head receives the
last exact final hidden state, the verified anchor-token embedding, and a
continuous source trajectory. It proposes the final hidden-state trajectory
for the next 31 tokens in parallel. The frozen Qwen LM head turns those states
into token candidates, and one ordinary causal Qwen pass verifies the complete
candidate block.

## Protocol

Only the new head is trainable. The frozen parent is
`flowdraft_v4_full_300/best`. Training uses 49,750 packed sequences and model
selection uses 2,048 content-disjoint packed sequences. The run used 100
optimizer steps. The final benchmark uses FP32 eager attention, 10 independent
prompts, 64 generated tokens per non-EOS prompt, and strict greedy parity.

`target_model_tpf` counts prefill plus every ordinary frozen-Qwen verifier
forward. `cacheflow_head_calls` is reported separately and wall-clock timing
includes it. Therefore no small-head compute is hidden from the measured
speedup.

## Result

The lossless guard works, but this endpoint predictor is not accurate enough
to be useful after only 100 updates:

| Metric | Result |
| --- | ---: |
| Greedy parity | 100% (10/10) |
| Aggregate target-model TPF | 1.064 |
| Aggregate wall-clock speedup | 0.916x |
| Weighted accepted draft tokens | 0.068 |
| Median accepted draft tokens | 0.0 |
| Frozen-Qwen forwards | 595 |
| CacheFlow head calls | 585 |

The independent held-out proxy reached greedy-prefix acceptance `0.0859` at
step 100. It is retained as a diagnostic only. The prompt benchmark shows that
it did not translate into useful generation-time acceptance. This probe is a
negative result for direct final-hidden trajectory prediction; it must not be
presented as an acceleration result.

## Artifacts

The remote run directory preserves the two 7.1 MB CacheFlow head checkpoints
under `best/` and `last/`. Git intentionally stores only their architecture
configs plus the compact protocol and metrics. The launcher now uploads both
weight directories to a private Hugging Face model repository when `HF_TOKEN`
and `HF_REPO_ID` are configured; this particular remote environment had no
Hugging Face token, so it did not publish them automatically. The parent
FlowDraft adapter is referenced in the run manifest and must be restored
alongside either head.
