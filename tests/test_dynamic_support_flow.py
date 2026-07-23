import torch

from orthrus_training.simplex_flow import DynamicSupportSimplexFlowRefiner, make_token_codebook


def test_dynamic_support_head_shapes_and_retrieval():
    torch.manual_seed(7)
    head = DynamicSupportSimplexFlowRefiner(
        block_size=5,
        candidate_count=6,
        draft_hidden_size=12,
        hidden_size=16,
        token_code_dim=4,
        num_layers=1,
        num_heads=4,
    )
    embedding = torch.randn(23, 12)
    codebook = make_token_codebook(embedding, code_dim=4, seed=11)
    base_logits = torch.randn(2, 3, 4, 6)
    candidate_ids = torch.randint(0, 23, (2, 3, 4, 6))
    hidden = torch.randn(2, 3, 4, 12)
    state = torch.full_like(base_logits, 1.0 / 6)
    source = torch.zeros(2, 3, 4)
    target = torch.ones(2, 3, 4)
    endpoint = head(base_logits, state, source, target, candidate_ids, hidden, codebook)
    assert endpoint.shape == base_logits.shape
    assert torch.allclose(endpoint.sum(dim=-1), torch.ones_like(endpoint[..., 0]), atol=1e-5)
    scores = head.retrieval_scores(hidden, codebook)
    assert scores.shape == (2, 3, 4, 23)
    assert torch.isfinite(scores).all()
    assert torch.equal(scores, torch.zeros_like(scores))
