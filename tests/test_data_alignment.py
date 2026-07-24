import json

import numpy as np
import pytest
import torch

from orthrus_training.data import (
    assert_disjoint_packed_manifests,
    make_diffusion_batch,
    sample_anchor_positions,
)


def test_make_diffusion_batch_alignment():
    input_ids = torch.arange(20).view(1, 20)
    anchors = torch.tensor([[2, 10]])

    diffusion_ids, position_ids, causal_limit, teacher_positions, target_ids = make_diffusion_batch(
        input_ids=input_ids,
        anchors=anchors,
        block_size=4,
        mask_token_id=99,
    )

    assert diffusion_ids.tolist() == [[2, 99, 99, 99, 10, 99, 99, 99]]
    assert position_ids.tolist() == [[2, 3, 4, 5, 10, 11, 12, 13]]
    assert causal_limit.tolist() == [[1, 1, 1, 1, 9, 9, 9, 9]]
    assert teacher_positions.tolist() == [[2, 3, 4, 10, 11, 12]]
    assert target_ids.tolist() == [[3, 4, 5, 11, 12, 13]]


def test_seeded_anchor_positions_are_reproducible():
    first = sample_anchor_positions(
        batch_size=2,
        seq_len=20,
        block_size=4,
        num_blocks=3,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(4284),
    )
    second = sample_anchor_positions(
        batch_size=2,
        seq_len=20,
        block_size=4,
        num_blocks=3,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(4284),
    )

    assert torch.equal(first, second)


def _write_manifest(directory, values):
    directory.mkdir()
    array = np.asarray(values, dtype=np.int32)
    np.save(directory / "train-00000.npy", array)
    manifest = {
        "seq_len": int(array.shape[1]),
        "shards": [
            {
                "file": "train-00000.npy",
                "num_sequences": int(array.shape[0]),
                "seq_len": int(array.shape[1]),
            }
        ],
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_disjoint_manifest_guard_detects_exact_packed_leakage(tmp_path):
    train = _write_manifest(tmp_path / "train", [[1, 2, 3], [4, 5, 6]])
    clean_eval = _write_manifest(tmp_path / "clean_eval", [[7, 8, 9]])
    leaked_eval = _write_manifest(tmp_path / "leaked_eval", [[4, 5, 6]])

    assert assert_disjoint_packed_manifests(train, clean_eval) == (2, 1)
    with pytest.raises(ValueError, match="Train/eval leakage"):
        assert_disjoint_packed_manifests(train, leaked_eval)
