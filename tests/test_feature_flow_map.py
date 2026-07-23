import torch

from orthrus_training.feature_flow_map import FeatureFlowMapHead, feature_flow_interpolate, feature_flow_source


def test_feature_flow_map_shapes_and_diagonal():
    torch.manual_seed(0)
    head = FeatureFlowMapHead(hidden_size=16, block_size=5, latent_size=24, num_layers=1, num_heads=4)
    context = torch.randn(2, 3, 16)
    anchors = torch.randn(2, 3, 16)
    source = feature_flow_source(context, 4, torch.Generator().manual_seed(5))
    context_source = feature_flow_source(context, 4, mode="context")
    target = torch.randn_like(source)
    time = torch.full((2, 3), 0.4)
    state = feature_flow_interpolate(source, target, time)
    output = head(context, anchors, state, time)
    assert source.shape == (2, 3, 4, 16)
    assert torch.equal(context_source[:, :, 0], context)
    assert output.shape == source.shape
    assert torch.allclose(feature_flow_interpolate(source, target, torch.zeros_like(time)), source)
    assert torch.allclose(feature_flow_interpolate(source, target, torch.ones_like(time)), target)
