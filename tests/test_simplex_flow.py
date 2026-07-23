import torch

from orthrus_training.simplex_flow import (
    SimplexFlowRefiner,
    local_simplex_path,
    simplex_flow_step,
)


def test_local_simplex_path_interpolates_uniform_and_endpoint():
    targets = torch.tensor([[[0, 2]]])
    zeros = torch.zeros_like(targets, dtype=torch.float32)
    ones = torch.ones_like(zeros)
    source = local_simplex_path(targets, zeros, candidates=4)
    endpoint = local_simplex_path(targets, ones, candidates=4)
    assert torch.allclose(source, torch.full_like(source, 0.25))
    assert endpoint.argmax(dim=-1).equal(targets)
    assert torch.allclose(endpoint.sum(dim=-1), torch.ones_like(ones))


def test_simplex_flow_refiner_is_valid_endpoint_and_zero_initial_residual():
    head = SimplexFlowRefiner(block_size=4, candidate_count=5, hidden_size=40, num_layers=1, num_heads=5)
    base = torch.randn(2, 3, 3, 5)
    state = torch.full_like(base, 0.2)
    times = torch.zeros(2, 3, 3)
    endpoint = head(base, state, times, torch.ones_like(times))
    assert endpoint.shape == base.shape
    assert torch.allclose(endpoint.sum(dim=-1), torch.ones_like(times), atol=1e-5)
    assert torch.allclose(endpoint, torch.softmax(base, dim=-1), atol=1e-5)
    transported = simplex_flow_step(state, endpoint, times, torch.ones_like(times))
    assert torch.allclose(transported, endpoint, atol=1e-5)
