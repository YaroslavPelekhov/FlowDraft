"""Candidate-support augmentation for local categorical flow maps.

The flow stays on a discrete simplex, but the simplex is allowed to include a
small train-derived rescue set in addition to the parent drafter's top-k.  No
extra Qwen call is needed: rescue-token logits are gathered from the draft
logits already computed for the ordinary Orthrus proposal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


@dataclass
class RescueCandidateBank:
    """A deterministic mapping from parent top-1 token to rescue token ids."""

    table: torch.Tensor
    base_candidate_count: int

    @property
    def rescue_count(self) -> int:
        return int(self.table.shape[1])

    @property
    def vocab_size(self) -> int:
        return int(self.table.shape[0])

    def to(self, device: torch.device | str) -> "RescueCandidateBank":
        return RescueCandidateBank(self.table.to(device=device), self.base_candidate_count)

    def lookup(self, base_top1: torch.Tensor) -> torch.Tensor:
        if base_top1.numel() and int(base_top1.max()) >= self.vocab_size:
            raise ValueError("parent candidate id exceeds rescue-bank vocabulary")
        return self.table[base_top1.long()].long()

    def save(self, directory: str | Path, metadata: dict | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file({"rescue_token_table": self.table.cpu().contiguous()}, directory / "rescue_bank.safetensors")
        payload = {
            "format": "flowdraft_rescue_candidate_bank_v1",
            "vocab_size": self.vocab_size,
            "rescue_count": self.rescue_count,
            "base_candidate_count": self.base_candidate_count,
        }
        if metadata:
            payload.update(metadata)
        with (directory / "rescue_bank_config.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def load(cls, directory: str | Path, device: torch.device | str = "cpu") -> "RescueCandidateBank":
        directory = Path(directory)
        with (directory / "rescue_bank_config.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("format") != "flowdraft_rescue_candidate_bank_v1":
            raise ValueError(f"Unsupported rescue bank: {metadata.get('format')}")
        table = load_file(directory / "rescue_bank.safetensors")["rescue_token_table"].to(device=device)
        expected = (int(metadata["vocab_size"]), int(metadata["rescue_count"]))
        if tuple(table.shape) != expected:
            raise ValueError(f"Rescue-bank table shape {tuple(table.shape)} != {expected}")
        return cls(table=table, base_candidate_count=int(metadata["base_candidate_count"]))


def select_candidate_support(
    logits: torch.Tensor,
    candidate_count: int,
    rescue_bank: RescueCandidateBank | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a unique fixed-size simplex support and its frozen draft logits.

    The ordering is: parent top-k, train-derived rescues for the parent top-1,
    then additional parent-ranked tokens to fill duplicates.  This is entirely
    deterministic and only gathers from ``logits``; it never invokes a model.
    """

    if logits.dim() < 2 or candidate_count < 2:
        raise ValueError("logits must be [..., vocab] and candidate_count must be >=2")
    vocab_size = logits.shape[-1]
    if candidate_count > vocab_size:
        raise ValueError("candidate_count cannot exceed vocabulary size")
    if rescue_bank is None:
        values, ids = logits.topk(candidate_count, dim=-1)
        return values, ids
    if rescue_bank.vocab_size != vocab_size:
        raise ValueError("rescue bank and logits use different vocabularies")
    base_count = rescue_bank.base_candidate_count
    if base_count <= 0 or base_count > candidate_count:
        raise ValueError("invalid rescue-bank base_candidate_count")
    raw_count = min(vocab_size, candidate_count + rescue_bank.rescue_count)
    raw_ids = logits.topk(raw_count, dim=-1).indices
    base_ids = raw_ids[..., :base_count]
    rescue_ids = rescue_bank.lookup(base_ids[..., 0])
    pool = torch.cat([base_ids, rescue_ids, raw_ids], dim=-1)
    pool_width = pool.shape[-1]

    # Remove repeats while retaining the first occurrence in the priority order.
    equality = pool.unsqueeze(-1).eq(pool.unsqueeze(-2))
    earlier = torch.tril(
        torch.ones((pool_width, pool_width), dtype=torch.bool, device=pool.device), diagonal=-1
    )
    duplicate = (equality & earlier).any(dim=-1)
    valid = pool.ge(0) & ~duplicate
    ranks = valid.to(torch.long).cumsum(dim=-1)
    wanted = torch.arange(1, candidate_count + 1, device=pool.device).view(
        *((1,) * (pool.dim() - 1)), candidate_count, 1
    )
    selected_indices = (ranks.unsqueeze(-2) == wanted).to(torch.int64).argmax(dim=-1)
    ids = pool.gather(-1, selected_indices)
    if torch.any(ids < 0):
        raise RuntimeError("candidate pool did not contain enough valid token ids")
    values = logits.gather(-1, ids)
    return values, ids


def select_dynamic_candidate_support(
    logits: torch.Tensor,
    candidate_count: int,
    dynamic_ids: torch.Tensor,
    base_candidate_count: int,
    candidate_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge parent top-k with context-specific retrieved token ids.

    Candidate order is deliberately stable: the parent distribution first,
    learned retrieval next, then additional parent-ranked tokens to replace
    duplicates. The result remains a fixed-size categorical simplex.
    """

    if logits.dim() < 2 or candidate_count < 2:
        raise ValueError("logits must be [..., vocab] and candidate_count must be >=2")
    if not 0 < base_candidate_count < candidate_count:
        raise ValueError("base_candidate_count must be in (0, candidate_count)")
    if dynamic_ids.shape[:-1] != logits.shape[:-1] or dynamic_ids.dim() != logits.dim():
        raise ValueError("dynamic_ids must match logits except for its final width")
    if dynamic_ids.dtype != torch.long or dynamic_ids.shape[-1] == 0:
        raise ValueError("dynamic_ids must be non-empty torch.long ids")
    if candidate_logits is not None and candidate_logits.shape != logits.shape:
        raise ValueError("candidate_logits must match logits")
    value_logits = logits if candidate_logits is None else candidate_logits
    vocab_size = logits.shape[-1]
    if candidate_count > vocab_size:
        raise ValueError("candidate_count cannot exceed vocabulary size")
    if int(dynamic_ids.min()) < 0 or int(dynamic_ids.max()) >= vocab_size:
        raise ValueError("dynamic_ids contain an out-of-vocabulary token")

    fallback_count = min(vocab_size, candidate_count + dynamic_ids.shape[-1])
    fallback_ids = logits.topk(fallback_count, dim=-1).indices
    base_ids = fallback_ids[..., :base_candidate_count]
    pool = torch.cat([base_ids, dynamic_ids, fallback_ids[..., base_candidate_count:]], dim=-1)
    width = pool.shape[-1]
    equality = pool.unsqueeze(-1).eq(pool.unsqueeze(-2))
    earlier = torch.tril(torch.ones((width, width), dtype=torch.bool, device=pool.device), diagonal=-1)
    duplicate = (equality & earlier).any(dim=-1)
    valid = ~duplicate
    ranks = valid.to(torch.long).cumsum(dim=-1)
    wanted = torch.arange(1, candidate_count + 1, device=pool.device).view(
        *((1,) * (pool.dim() - 1)), candidate_count, 1
    )
    selected_indices = (ranks.unsqueeze(-2) == wanted).to(torch.int64).argmax(dim=-1)
    ids = pool.gather(-1, selected_indices)
    values = value_logits.gather(-1, ids)
    return values, ids
