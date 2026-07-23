# Adaptive-Support Local-Simplex CFM Probe

**Status:** completed on 2026-07-23, A100 40 GB, training code at commit `72dca40`.

This is a controlled ablation of the local-simplex categorical flow-map drafter. It retains the same 9.67M-parameter endpoint CFM, objective, frozen FlowDraft parent, and lossless AR verifier as the preceding probe, while changing only the candidate support:

- positions start with the parent drafter's top-96 token candidates;
- up to 32 train-derived rescue candidates are added from a frozen lookup table;
- the lookup maps a parent top-1 token to frequent AR targets that were missed by the top-96 support during an independent training-split calibration pass;
- all rescue-token logits come from the parent pass already required by Orthrus; this adds no frozen-Qwen forward pass.

The goal was to address the earlier support-coverage bottleneck without turning the CFM into a second autoregressive model.

## Protocol

- Calibration: 507,904 anchored positions from the training split only. Base top-96 coverage was 80.77%; the table has 32 rescue slots for 2,114 populated parent-top-1 keys.
- Train: 49,750 packed Nemotron sequences. Holdout: 2,048 disjoint packed sequences, never used in calibration or fitting.
- Objective: root endpoint CE, diagonal VFM CE, endpoint consistency (ECLD), temporal drift, and prefix-survival weighting.
- Requested cap: 300 optimizer steps. Early stopping selected `best` at step 100 and stopped at step 250 after three non-improving holdout evaluations.
- Strict inference: FP32, eager attention, ten generated sanity prompts, 64 generated tokens per prompt, greedy decoding, and exact AR parity required.
- TPF counts frozen-Qwen forward passes; local CFM-head calls are recorded separately.

## Result

| Metric | Adaptive support | Fixed top-128 baseline |
| --- | ---: | ---: |
| Holdout best greedy accepted prefix | 1.586 | 1.672 |
| Holdout first-token accuracy at best | 93.0% | 92.6% |
| Holdout candidate coverage at best | 81.1% | 81.6% |
| Greedy parity | 100% (10/10) | 100% (10/10) |
| Aggregate TPF | 1.149 | 1.153 |
| Aggregate wall-clock speedup | 1.022x | 1.049x |
| Weighted accepted length | 1.478 | 1.488 |
| Frozen-Qwen forwards | 551 | 549 |
| Local CFM-head calls | 270 | 270 |

The strict losslessness gate passes, so the mechanism is methodologically valid for greedy decoding. It is nevertheless a negative result: the top-1-conditioned rescue table did not transfer to the held-out contexts and reduced both holdout prefix acceptance and aggregate speed. This rules out this simple static rescue mapping as the main way to close the local-simplex support gap.

The committed files contain the calibration report, exact run configuration, holdout curve, per-prompt measurements, and full log. Binary `best` and `last` weights, including the rescue bank, remain outside Git in the remote run directory and should be uploaded to a private Hugging Face model repository once a remote HF credential is configured.
