# Identity-Aware Dynamic-Support CFM Probe

**Status:** completed on 2026-07-23, A100 40 GB. The first launch (`r1`) stopped before step one due to a BF16 retrieval-path dtype bug; `r2` is the corrected, complete run recorded here.

This probe tested whether the local categorical flow map could overcome the rank-only limitation of the previous head. It conditions endpoint transport on frozen Orthrus diffusion hidden states and semantic token codes, and retrieves 32 dynamic support vertices from the whole frozen vocabulary alongside 96 parent-ranked vertices.

## Protocol

- Parent: frozen FlowDraft/Orthrus Qwen3-1.7B checkpoint.
- Train: 49,750 packed Nemotron sequences. Holdout: 2,048 disjoint packed sequences.
- Head: 11.83M trainable parameters; 32-dimensional deterministic random projection of frozen token embeddings for full-vocabulary retrieval.
- Objective: root endpoint CE, diagonal VFM CE, ECLD, temporal drift, prefix weighting, and hard-negative retrieval loss.
- Requested cap: 300 optimizer steps. Best holdout prefix metric was selected at step 50; early stopping stopped at step 200.
- Strict inference: FP32, eager attention, greedy decoding, 10 generated sanity prompts, 64 output tokens per prompt, exact AR parity required.

## Result

| Metric | Dynamic support | Fixed top-128 CFM |
| --- | ---: | ---: |
| Best holdout accepted prefix | 1.691 | 1.672 |
| Best holdout support coverage | 81.5% | 81.6% |
| Greedy parity | 100% (10/10) | 100% (10/10) |
| Aggregate TPF | 1.124 | 1.153 |
| Aggregate wall-clock speedup | 1.004x | 1.049x |
| Weighted accepted length | 1.421 | 1.488 |
| Frozen-Qwen forwards | 563 | 549 |

The method is lossless but unsuccessful as an acceleration method. It supplied a small holdout-prefix improvement, yet its dynamic retrieval did not materially raise support coverage and increased proposal cost. More importantly, the initial query was random and displaced some parent candidates before it was trained. The follow-up design uses a zero-initialized low-rank residual so its first step is exactly the top-128 parent baseline, then learns dynamic support only when the residual has evidence to improve it.

The archive contains the exact configuration, train/holdout measurements, per-prompt benchmark, and full log. Binary `best` and `last` checkpoints remain in `/workspace/flowdraft_runs/dynamic_support_flow_probe_300_r2` on the GPU host and are excluded from Git.
