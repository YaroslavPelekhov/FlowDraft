import torch

from orthrus_training.flowdraft import make_flowdraft_batch, make_flowdraft_inputs_embeds
from orthrus_training.losses import (
    prefix_acceptance_metrics,
    prefix_survival_cross_entropy,
    prefix_survival_weights,
)


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


def test_prefix_survival_weights_prioritize_early_tokens():
    weights = prefix_survival_weights(block_size=5, decay=0.9, device=torch.device("cpu"))

    assert weights.shape == (4,)
    assert torch.all(weights[:-1] > weights[1:])
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-6)


def test_prefix_survival_loss_penalizes_early_error_more_than_late_error():
    target_ids = torch.tensor([[0, 1, 2, 3]])
    early_wrong = torch.full((1, 4, 5), -4.0)
    late_wrong = torch.full((1, 4, 5), -4.0)

    for pos, target in enumerate(target_ids[0].tolist()):
        early_wrong[0, pos, target] = 4.0
        late_wrong[0, pos, target] = 4.0

    early_wrong[0, 0, 0] = -4.0
    early_wrong[0, 0, 4] = 4.0
    late_wrong[0, 3, 3] = -4.0
    late_wrong[0, 3, 4] = 4.0

    early_loss = prefix_survival_cross_entropy(early_wrong, target_ids, block_size=5, decay=0.9)
    late_loss = prefix_survival_cross_entropy(late_wrong, target_ids, block_size=5, decay=0.9)

    assert early_loss > late_loss


def test_greedy_prefix_acceptance_stops_at_first_error():
    target_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    logits = torch.full((1, 8, 5), -4.0)
    predictions = [0, 1, 4, 3, 0, 4, 2, 3]
    for position, prediction in enumerate(predictions):
        logits[0, position, prediction] = 4.0

    metrics = prefix_acceptance_metrics(logits, target_ids, block_size=5)

    # The two blocks accept 2 and 1 tokens before their first mismatch.
    assert torch.isclose(metrics["greedy_prefix_acceptance"], torch.tensor(1.5))
    assert torch.isclose(metrics["first_token_acc"], torch.tensor(1.0))
