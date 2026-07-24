from __future__ import annotations

import json
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PackedShard:
    path: Path
    num_sequences: int
    seq_len: int


class PackedTokenDataset(Dataset):
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        base_dir = self.manifest_path.parent
        self.seq_len = int(manifest["seq_len"])
        self.shards = [
            PackedShard(
                path=(base_dir / shard["file"]).resolve(),
                num_sequences=int(shard["num_sequences"]),
                seq_len=int(shard["seq_len"]),
            )
            for shard in manifest["shards"]
        ]
        self._arrays: list[np.memmap | None] = [None] * len(self.shards)
        self._offsets = np.cumsum([0] + [s.num_sequences for s in self.shards])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            index = len(self) + index
        shard_idx = int(np.searchsorted(self._offsets, index, side="right") - 1)
        local_idx = index - int(self._offsets[shard_idx])
        arr = self._array(shard_idx)
        return torch.from_numpy(np.array(arr[local_idx], dtype=np.int64, copy=True))

    def _array(self, shard_idx: int) -> np.memmap:
        arr = self._arrays[shard_idx]
        if arr is None:
            shard = self.shards[shard_idx]
            arr = np.load(shard.path, mmap_mode="r")
            if arr.shape != (shard.num_sequences, shard.seq_len):
                raise ValueError(f"Shard shape mismatch for {shard.path}: {arr.shape}")
            self._arrays[shard_idx] = arr
        return arr


def packed_sequence_hashes(manifest_path: str | Path) -> set[bytes]:
    """Hash every packed row without loading complete shards into memory."""

    dataset = PackedTokenDataset(manifest_path)
    hashes: set[bytes] = set()
    for shard_idx, shard in enumerate(dataset.shards):
        array = dataset._array(shard_idx)
        for row in array:
            hashes.add(hashlib.blake2b(memoryview(row), digest_size=16).digest())
    return hashes


def assert_disjoint_packed_manifests(
    train_manifest: str | Path,
    eval_manifest: str | Path,
) -> tuple[int, int]:
    """Fail fast when fixed packed sequences leak from training into eval."""

    train_hashes = packed_sequence_hashes(train_manifest)
    eval_hashes = packed_sequence_hashes(eval_manifest)
    overlap = train_hashes & eval_hashes
    if overlap:
        raise ValueError(
            f"Train/eval leakage: found {len(overlap)} identical packed sequences. "
            "Rebuild eval data with --skip-sequences set past the training range."
        )
    return len(train_hashes), len(eval_hashes)


def sample_anchor_positions(
    batch_size: int,
    seq_len: int,
    block_size: int,
    num_blocks: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if seq_len <= block_size:
        raise ValueError(f"seq_len={seq_len} must be larger than block_size={block_size}")
    max_anchor = seq_len - block_size
    return torch.randint(
        0,
        max_anchor + 1,
        (batch_size, num_blocks),
        device=device,
        generator=generator,
    )


def make_diffusion_batch(
    input_ids: torch.Tensor,
    anchors: torch.Tensor,
    block_size: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build flattened Orthrus diffusion blocks.

    Returns:
      diffusion_ids: [B, num_blocks * K], anchor token plus K-1 masks per block.
      position_ids: [B, num_blocks * K], original rotary positions for each block token.
      causal_limit: [B, num_blocks * K], max AR cache index visible to each query.
      teacher_positions: [B, num_blocks * (K-1)], AR logits rows for next-token targets.
      target_ids: [B, num_blocks * (K-1)], hard labels for diagnostics.
    """

    batch_size, _ = input_ids.shape
    num_blocks = anchors.shape[1]
    device = input_ids.device

    offsets = torch.arange(block_size, device=device)
    block_positions = anchors.unsqueeze(-1) + offsets.view(1, 1, block_size)
    clean_blocks = torch.gather(input_ids, 1, block_positions.reshape(batch_size, -1))
    clean_blocks = clean_blocks.reshape(batch_size, num_blocks, block_size)

    diffusion_blocks = torch.full_like(clean_blocks, mask_token_id)
    diffusion_blocks[:, :, 0] = clean_blocks[:, :, 0]
    diffusion_ids = diffusion_blocks.reshape(batch_size, num_blocks * block_size)
    position_ids = block_positions.reshape(batch_size, num_blocks * block_size)

    causal_per_block = (anchors - 1).clamp_min(-1)
    causal_limit = causal_per_block.unsqueeze(-1).expand(-1, -1, block_size)
    causal_limit = causal_limit.reshape(batch_size, num_blocks * block_size)

    pred_offsets = torch.arange(block_size - 1, device=device)
    teacher_positions = anchors.unsqueeze(-1) + pred_offsets.view(1, 1, block_size - 1)
    teacher_positions = teacher_positions.reshape(batch_size, num_blocks * (block_size - 1))

    target_positions = teacher_positions + 1
    target_ids = torch.gather(input_ids, 1, target_positions)
    return diffusion_ids, position_ids, causal_limit, teacher_positions, target_ids


def iter_round_robin(iterators: dict[str, Iterator], order: list[str]) -> Iterator[tuple[str, dict]]:
    while True:
        random.shuffle(order)
        for split in order:
            yield split, next(iterators[split])
