import torch

from orthrus_training.candidate_support import (
    RescueCandidateBank,
    select_candidate_support,
    select_dynamic_candidate_support,
)


def test_rescue_support_includes_missed_token_and_is_unique():
    logits = torch.tensor([[[10.0, 9.0, 8.0, 7.0, 6.0, 5.0]]])
    table = torch.full((6, 2), -1, dtype=torch.int32)
    # Parent top-1 is token 0. Rescue token 5 would otherwise be outside top-4.
    table[0] = torch.tensor([5, 1], dtype=torch.int32)
    bank = RescueCandidateBank(table=table, base_candidate_count=2)
    values, ids = select_candidate_support(logits, candidate_count=4, rescue_bank=bank)
    assert values.shape == (1, 1, 4)
    assert ids.tolist()[0][0][:2] == [0, 1]
    assert 5 in ids.tolist()[0][0]
    assert len(set(ids.tolist()[0][0])) == 4


def test_plain_support_matches_torch_topk():
    logits = torch.randn(2, 3, 11)
    values, ids = select_candidate_support(logits, candidate_count=5)
    expected_values, expected_ids = logits.topk(5, dim=-1)
    assert torch.equal(ids, expected_ids)
    assert torch.equal(values, expected_values)


def test_dynamic_support_keeps_parent_prefix_and_retrieved_ids():
    logits = torch.tensor([[[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]]])
    dynamic = torch.tensor([[[6, 1, 5]]])
    values, ids = select_dynamic_candidate_support(
        logits, candidate_count=5, dynamic_ids=dynamic, base_candidate_count=2
    )
    assert values.shape == (1, 1, 5)
    assert ids.tolist()[0][0][:2] == [0, 1]
    assert 6 in ids.tolist()[0][0]
    assert 5 in ids.tolist()[0][0]
    assert len(set(ids.tolist()[0][0])) == 5
