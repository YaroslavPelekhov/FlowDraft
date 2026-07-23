# R2Flow Fixed-Point Probe (2026-07-23)

This is the first residual-refined FlowDraft experiment. It starts from the
frozen `flowdraft_v4_full_300/best` adapter and trains only a 6,870,274
parameter block-local corrector for 100 optimizer steps on 49,750 packed train
sequences. The 2,048 packed evaluation sequences are disjoint by content hash.

## Method

For a flow proposal `y0`, the frozen AR verifier defines `y1 = J(y0)` and
`y2 = J(y1)`. The corrector receives the first verifier's final hidden states,
candidate-token embeddings, residual-token embeddings, and top-2 margin. It is
trained to predict `y2` in one parallel update. Greedy decoding still performs
an independent final AR verification and emits only the matching prefix.

## Result

The independent FP32 quick benchmark has 10 prompts and 64 generated tokens
per non-EOS prompt. It reached exact greedy parity on all 10 prompts:

| Metric | Result |
| --- | ---: |
| Greedy parity | 100% (10/10) |
| Aggregate TPF | 1.170 |
| Aggregate wall-clock speedup | 1.066x |
| Mean final accepted length | 1.926 |
| Median final accepted length | 2.0 |
| Counted R2Flow forwards | 541 |

The packed holdout selection metric reached 12.922 accepted tokens at step 100,
but it did **not** transfer to the independent prompt benchmark. That number is
therefore retained as a diagnostic of the `J^2` imitation task, not reported as
an inference result. The final decoder counts the flow pass, the residual
verifier pass, and the final lossless verifier pass; no compute is hidden.

## Artifacts

`best/` and `last/` on the remote run directory contain the two small R2Flow
corrector checkpoints. They reference the immutable frozen parent FlowDraft
adapter listed in `run_config.json`. The metric files copied here are sufficient
to reproduce the protocol after restoring those two parent artifacts.
