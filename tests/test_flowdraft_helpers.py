import torch

from orthrus_training.flowdraft import (
    add_flow_time_conditioning,
    exact_endpoint_embeddings,
    flow_map_step_size,
    make_flowdraft_batch,
    make_flowdraft_inputs_embeds,
    make_endpoint_blocks,
    sample_categorical_source_tokens,
    sample_cfm_time_pairs,
    topk_endpoint_embeddings,
    transport_categorical_state,
)
from orthrus_training.modeling import FlowDraftStateAdapter
from orthrus_training.residual_flow import ResidualFlowCorrector, verifier_margin
from orthrus_training.cacheflow import CacheFlowTrajectoryHead, flow_source_from_context
from orthrus_training.hydraflow import HydraFlowDrafter
from orthrus_training.eagleflow import EagleFlowDrafter, ParallelEagleFlowDrafter
from orthrus_training.flowtree import (
    ancestor_matrix,
    build_flowtree,
    greedy_path_coverage,
    soft_topk_coverage_loss,
)
from orthrus_training.losses import (
    bounded_jsd_distillation,
    prefix_acceptance_metrics,
    prefix_survival_cross_entropy,
    prefix_survival_weights,
    verifier_aligned_losses,
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


def test_cfm_time_pairs_have_diagonal_and_ordered_off_diagonal_blocks():
    torch.manual_seed(7)
    source, target, diagonal = sample_cfm_time_pairs(
        batch_size=2,
        num_blocks=8,
        diagonal_fraction=0.75,
        device=torch.device("cpu"),
    )

    assert diagonal.sum(dim=1).tolist() == [6, 6]
    assert torch.allclose(source[diagonal], target[diagonal])
    assert torch.all(target[~diagonal] > source[~diagonal])


def test_cfm_time_pairs_reserve_exact_one_jump_pairs():
    torch.manual_seed(11)
    source, target, diagonal = sample_cfm_time_pairs(
        batch_size=2,
        num_blocks=8,
        diagonal_fraction=0.75,
        one_jump_fraction=0.5,
        device=torch.device("cpu"),
    )

    one_jump = (source.squeeze(-1).squeeze(-1) == 0.0) & (target.squeeze(-1).squeeze(-1) == 1.0)
    assert torch.all(~one_jump | ~diagonal)
    assert one_jump.sum(dim=1).tolist() == [1, 1]


def test_endpoint_flow_map_reaches_endpoint_at_t_one():
    source = torch.tensor([[[[0.25, 0.75]]]])
    endpoint = torch.tensor([[[[0.9, 0.1]]]])
    transported = transport_categorical_state(source, endpoint, 0.4, 1.0)

    assert torch.allclose(flow_map_step_size(torch.tensor(0.4), torch.tensor(1.0)), torch.tensor(1.0))
    assert torch.allclose(transported, endpoint)


def test_topk_endpoint_embeddings_preserve_simplex_average():
    logits = torch.tensor([[[0.0, 2.0, 1.0]]])
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    projected = topk_endpoint_embeddings(logits, embeddings, topk=3)
    expected = torch.softmax(logits, dim=-1) @ embeddings

    assert torch.allclose(projected, expected, atol=1e-6)


def test_exact_endpoint_embeddings_match_full_softmax_expectation():
    logits = torch.tensor([[[0.0, 2.0, 1.0, -1.0]]])
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0], [-1.0, 1.0]])
    projected = exact_endpoint_embeddings(logits, embeddings, vocab_chunk_size=2)
    expected = torch.softmax(logits, dim=-1) @ embeddings

    assert torch.allclose(projected, expected, atol=1e-6)


def test_flow_time_conditioning_is_explicit_and_shape_stable():
    inputs = torch.ones((1, 8, 8))
    source = torch.zeros((1, 2, 1, 1))
    target = torch.ones((1, 2, 1, 1)) * 0.5
    conditioned = add_flow_time_conditioning(inputs, source, target, block_size=4, scale=0.1)

    assert conditioned.shape == inputs.shape
    assert not torch.equal(conditioned, inputs)


def test_state_adapter_is_an_identity_at_initialization():
    adapter = FlowDraftStateAdapter(hidden_size=8, bottleneck_size=4)
    inputs = torch.randn((1, 8, 8))
    time = torch.randn((1, 2, 8))

    assert torch.allclose(adapter(inputs, time, block_size=4), inputs)


def test_uniform_categorical_source_is_stochastic_and_preserves_anchor():
    clean = torch.tensor([[[3, 4, 5, 6], [8, 9, 10, 11]]])
    generator = torch.Generator().manual_seed(123)
    source = sample_categorical_source_tokens(
        clean,
        vocab_size=32,
        mask_token_id=31,
        prior="uniform",
        generator=generator,
    )

    assert source.shape == clean.shape
    assert torch.equal(source[:, :, 0], clean[:, :, 0])
    assert not torch.equal(source[:, :, 1:], clean[:, :, 1:])


def test_simplex_state_uses_sampled_source_vertex():
    model = DummyModel()
    clean = torch.tensor([[[3, 4, 5, 6]]])
    source = torch.tensor([[[3, 7, 8, 9]]])
    mix = torch.zeros((1, 1, 1, 1))
    embeds = make_flowdraft_inputs_embeds(
        model=model,
        clean_blocks=clean,
        mask_token_id=31,
        state_mix=mix,
        source_token_ids=source,
    ).reshape(1, 1, 4, 8)

    assert torch.allclose(embeds, model.model.embed_tokens(source))


def test_endpoint_blocks_join_anchor_and_verifier_targets():
    anchors = torch.tensor([[[4], [8]]])
    targets = torch.tensor([[[10, 11, 12], [20, 21, 22]]])

    blocks = make_endpoint_blocks(anchors, targets)

    assert blocks.tolist() == [[[4, 10, 11, 12], [8, 20, 21, 22]]]


def test_flowtree_builds_shared_prefixes_and_covers_teacher_path():
    logits = torch.full((4, 8), -5.0)
    logits[0, 3] = 5.0
    logits[0, 4] = 4.0
    logits[1, 2] = 5.0
    logits[1, 1] = 4.0
    logits[2, 6] = 5.0
    logits[3, 7] = 5.0
    tree = build_flowtree(9, logits, branch_width=2, branch_depth=2, max_nodes=32)

    assert tree.num_nodes == 15
    assert greedy_path_coverage(tree, torch.tensor([3, 2, 6, 7])) == 4
    visibility = ancestor_matrix(tree.parents)
    assert bool(visibility[0, 0])
    assert not bool(visibility[1, 2])
    assert bool(visibility[-1, 0])


def test_flowtree_coverage_surrogate_rewards_teacher_in_branch_budget():
    targets = torch.tensor([[1, 2, 3]])
    covered = torch.full((1, 3, 6), -4.0)
    uncovered = covered.clone()
    for pos, target in enumerate(targets[0].tolist()):
        covered[0, pos, target] = 4.0
        uncovered[0, pos, (target + 1) % 6] = 4.0
        uncovered[0, pos, target] = -4.0

    assert soft_topk_coverage_loss(covered, targets, branch_width=2, branch_depth=2) < 1e-3
    assert soft_topk_coverage_loss(uncovered, targets, branch_width=2, branch_depth=2) > 0.1


def test_bounded_jsd_is_zero_for_identical_logits_and_bounded():
    student = torch.tensor([[[2.0, 0.0, -1.0]]])
    identical = bounded_jsd_distillation(student, student)
    different = bounded_jsd_distillation(student, -student)

    assert torch.isclose(identical, torch.tensor(0.0), atol=1e-7)
    assert 0.0 < different < torch.log(torch.tensor(2.0))


def test_residual_flow_corrector_is_zero_update_at_initialization():
    corrector = ResidualFlowCorrector(
        hidden_size=8,
        block_size=5,
        bottleneck_size=12,
        num_layers=1,
        num_heads=3,
    )
    hidden = torch.randn((2, 8, 8))
    candidate = torch.randn_like(hidden)
    residual = torch.randn_like(hidden)
    logits = torch.randn((2, 8, 11))

    correction = corrector(hidden, candidate, residual, verifier_margin(logits))

    assert correction.shape == hidden.shape
    assert torch.allclose(correction, torch.zeros_like(hidden))


def test_verifier_aligned_loss_stops_acceptance_at_first_mismatch():
    verifier = torch.full((1, 4, 5), -4.0)
    draft = torch.full((1, 4, 5), -4.0)
    verifier_targets = [0, 1, 2, 3]
    draft_targets = [0, 1, 4, 3]
    for position, target in enumerate(verifier_targets):
        verifier[0, position, target] = 4.0
    for position, target in enumerate(draft_targets):
        draft[0, position, target] = 4.0

    losses = verifier_aligned_losses(draft, verifier, block_size=5)

    assert torch.isclose(
        losses["greedy_prefix_acceptance"], torch.tensor(2.0)
    )
    assert losses["first_rejected_mask"].tolist() == [[False, False, True, False]]
    assert torch.isfinite(losses["reverse_kl"])


def test_cacheflow_endpoint_shape_is_block_parallel_and_finite():
    head = CacheFlowTrajectoryHead(
        hidden_size=8,
        block_size=5,
        latent_size=16,
        num_layers=1,
        num_heads=4,
    )
    context = torch.randn((2, 3, 8))
    anchor = torch.randn((2, 3, 8))
    source = flow_source_from_context(context, prediction_length=4, generator=torch.Generator().manual_seed(7))

    endpoint = head(context, anchor, source)

    assert endpoint.shape == (2, 3, 4, 8)
    assert torch.isfinite(endpoint).all()


def test_hydraflow_self_conditioned_rollout_has_a_correlated_trajectory():
    head = HydraFlowDrafter(hidden_size=8, block_size=5, state_size=16, num_layers=1)
    context = torch.randn((2, 3, 8))
    anchor = torch.randn((2, 3, 8))
    teacher = torch.randn((2, 3, 4, 8))

    free_hidden, free_embeddings = head.rollout(context, anchor)
    forced_hidden, forced_embeddings = head.rollout(context, anchor, teacher, teacher_forcing_ratio=1.0)

    assert free_hidden.shape == forced_hidden.shape == (2, 3, 4, 8)
    assert free_embeddings.shape == forced_embeddings.shape == (2, 3, 4, 8)
    assert not torch.allclose(free_hidden[:, :, 1:], forced_hidden[:, :, 1:])


def test_eagleflow_attention_trajectory_uses_feature_and_token_feedback():
    head = EagleFlowDrafter(hidden_size=8, block_size=5, state_size=8, num_layers=2, num_heads=2)
    context = torch.randn((2, 3, 8))
    anchor = torch.randn((2, 3, 8))
    teacher_embeddings = torch.randn((2, 3, 4, 8))
    teacher_features = torch.randn((2, 3, 4, 8))

    free_hidden, free_embeddings = head.rollout(context, anchor)
    forced_hidden, forced_embeddings = head.rollout(
        context,
        anchor,
        teacher_embeddings=teacher_embeddings,
        teacher_features=teacher_features,
        teacher_forcing_ratio=1.0,
    )

    assert free_hidden.shape == forced_hidden.shape == (2, 3, 4, 8)
    assert free_embeddings.shape == forced_embeddings.shape == (2, 3, 4, 8)
    assert torch.isfinite(free_hidden).all()
    assert not torch.allclose(free_hidden[:, :, 1:], forced_hidden[:, :, 1:])


def test_parallel_eagleflow_generates_all_block_endpoints_in_one_call():
    head = ParallelEagleFlowDrafter(hidden_size=8, block_size=5, state_size=8, num_layers=2, num_heads=2)
    context = torch.randn((2, 3, 8))
    anchor = torch.randn((2, 3, 8))

    hidden, embeddings = head.rollout(context, anchor)

    assert hidden.shape == embeddings.shape == (2, 3, 4, 8)
    assert torch.isfinite(hidden).all()
