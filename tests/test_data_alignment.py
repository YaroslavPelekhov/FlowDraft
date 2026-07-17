import torch

from orthrus_training.data import make_diffusion_batch


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
