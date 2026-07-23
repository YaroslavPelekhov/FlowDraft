#!/usr/bin/env python
"""Calibrate a train-only rescue support for local categorical flow maps."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orthrus_training.candidate_support import RescueCandidateBank
from orthrus_training.data import PackedTokenDataset
from orthrus_training.modeling import dtype_from_string, load_flowdraft_adapter
from orthrus_training.simplex_flow_data import collect_base_draft_logits


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--calibration-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-anchor-blocks", type=int, default=32)
    parser.add_argument("--base-candidate-count", type=int, default=96)
    parser.add_argument("--rescue-count", type=int, default=32)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Rescue-support calibration requires CUDA")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing bank: {output}")
    if args.base_candidate_count < 1 or args.rescue_count < 1:
        raise ValueError("base-candidate-count and rescue-count must be positive")
    output.mkdir(parents=True, exist_ok=True)
    dtype = dtype_from_string(args.dtype)
    model, metadata, _ = load_flowdraft_adapter(
        args.init_checkpoint, args.upstream_dir, dtype, args.attn_implementation
    )
    model.to("cuda", dtype=dtype).train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = DataLoader(
        PackedTokenDataset(args.train_manifest), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    total, covered = 0, 0
    progress = tqdm(total=args.calibration_batches, desc="rescue-bank calibration")
    for batch_index, raw in enumerate(loader):
        if batch_index >= args.calibration_batches:
            break
        logits, teacher = collect_base_draft_logits(
            model, raw.to("cuda", non_blocking=True), args.num_anchor_blocks, generator
        )
        ids = logits.topk(args.base_candidate_count, dim=-1).indices
        matches = ids.eq(teacher.unsqueeze(-1))
        covered += int(matches.any(dim=-1).sum().item())
        total += teacher.numel()
        miss = ~matches.any(dim=-1)
        keys = ids[..., 0][miss].detach().cpu().tolist()
        targets = teacher[miss].detach().cpu().tolist()
        for key, target in zip(keys, targets):
            counts[int(key)][int(target)] += 1
        progress.update(1)
    progress.close()
    vocab_size = int(model.config.vocab_size)
    table = torch.full((vocab_size, args.rescue_count), -1, dtype=torch.int32)
    populated = 0
    for key, counter in counts.items():
        selected = [token for token, _ in counter.most_common(args.rescue_count)]
        table[key, : len(selected)] = torch.tensor(selected, dtype=torch.int32)
        populated += 1
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "parent_top1_to_frequent_missed_ar_token",
        "calibration_batches": min(args.calibration_batches, batch_index + 1),
        "num_anchor_blocks": args.num_anchor_blocks,
        "base_candidate_count": args.base_candidate_count,
        "rescue_count": args.rescue_count,
        "parent_topk_coverage": covered / total if total else 0.0,
        "calibration_positions": total,
        "populated_parent_tokens": populated,
        "train_manifest": str(Path(args.train_manifest).resolve()),
        "base_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "parent_adapter_metadata": metadata,
    }
    bank = RescueCandidateBank(table=table, base_candidate_count=args.base_candidate_count)
    bank.save(output, report)
    write_json(output / "calibration_report.json", report)
    print("RESCUE_BANK " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
