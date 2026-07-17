import torch

from orthrus_training.flowdraft import make_flowdraft_batch, make_flowdraft_inputs_embeds


class DummyInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 8)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyInner()


def test_flowdraft_batch_and_embeddings_shape():
    input_ids = torch.arange(16).view(1, 16)
    anchors = torch.tensor([[2, 8]])
    clean_blocks, position_ids, causal_limit, teacher_positions, target_ids = make_flowdraft_batch(
        input_ids=input_ids,
        anchors=anchors,
        block_size=4,
    )

    assert clean_blocks.shape == (1, 2, 4)
    assert position_ids.shape == (1, 8)
    assert causal_limit.shape == (1, 8)
    assert teacher_positions.tolist() == [[2, 3, 4, 8, 9, 10]]
    assert target_ids.tolist() == [[3, 4, 5, 9, 10, 11]]

    state_mix = torch.zeros((1, 2, 1, 1))
    embeds = make_flowdraft_inputs_embeds(
        model=DummyModel(),
        clean_blocks=clean_blocks,
        mask_token_id=31,
        state_mix=state_mix,
    )
    assert embeds.shape == (1, 8, 8)
