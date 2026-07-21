from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def select_endpoint_logits(diffusion_logits: torch.Tensor, block_size: int) -> torch.Tensor:
    batch_size, flat_len, vocab_size = diffusion_logits.shape
    num_blocks = flat_len // block_size
    logits = diffusion_logits.reshape(batch_size, num_blocks, block_size, vocab_size)
    return logits[:, :, : block_size - 1, :].reshape(batch_size, num_blocks * (block_size - 1), vocab_size)


def make_flowdraft_batch(
    input_ids: torch.Tensor,
    anchors: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build block metadata for conditional flow-map drafting.

    The first token in each block is the AR-verified anchor. The drafter predicts
    the next K-1 tokens from endpoint logits at positions 0..K-2.
    """

    batch_size, _ = input_ids.shape
    num_blocks = anchors.shape[1]
    device = input_ids.device

    offsets = torch.arange(block_size, device=device)
    block_positions = anchors.unsqueeze(-1) + offsets.view(1, 1, block_size)
    clean_blocks = torch.gather(input_ids, 1, block_positions.reshape(batch_size, -1))
    clean_blocks = clean_blocks.reshape(batch_size, num_blocks, block_size)
    position_ids = block_positions.reshape(batch_size, num_blocks * block_size)

    causal_per_block = (anchors - 1).clamp_min(-1)
    causal_limit = causal_per_block.unsqueeze(-1).expand(-1, -1, block_size)
    causal_limit = causal_limit.reshape(batch_size, num_blocks * block_size)

    pred_offsets = torch.arange(block_size - 1, device=device)
    teacher_positions = anchors.unsqueeze(-1) + pred_offsets.view(1, 1, block_size - 1)
    teacher_positions = teacher_positions.reshape(batch_size, num_blocks * (block_size - 1))

    target_positions = teacher_positions + 1
    target_ids = torch.gather(input_ids, 1, target_positions)
    return clean_blocks, position_ids, causal_limit, teacher_positions, target_ids


def sample_flow_state_mix(
    batch_size: int,
    num_blocks: int,
    min_mix: float,
    max_mix: float,
    device: torch.device,
) -> torch.Tensor:
    if min_mix < 0.0 or max_mix > 1.0 or min_mix > max_mix:
        raise ValueError(f"Invalid flow state range: min={min_mix}, max={max_mix}")
    if min_mix == max_mix:
        return torch.full((batch_size, num_blocks, 1, 1), min_mix, device=device)
    return torch.empty((batch_size, num_blocks, 1, 1), device=device).uniform_(min_mix, max_mix)


def sample_cfm_time_pairs(
    batch_size: int,
    num_blocks: int,
    diagonal_fraction: float,
    device: torch.device,
    logit_mean: float = -0.4,
    logit_std: float = 1.0,
    min_time: float = 0.0,
    max_time: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample diagonal VFM and off-diagonal CFM time pairs per block.

    This follows Categorical Flow Maps: a fixed fraction is trained on s=t,
    while the remainder uses s<t. Times follow the paper's logit-normal law.
    """

    if not 0.0 < diagonal_fraction <= 1.0:
        raise ValueError("diagonal_fraction must be in (0, 1]")
    if min_time < 0.0 or max_time >= 1.0 or min_time > max_time:
        raise ValueError(f"Invalid time interval [{min_time}, {max_time}]")

    shape = (batch_size, num_blocks, 1, 1)
    raw_s = torch.randn(shape, device=device) * logit_std + logit_mean
    raw_t = torch.randn(shape, device=device) * logit_std + logit_mean
    source_time = torch.sigmoid(raw_s).clamp(min_time, max_time)
    target_time = torch.sigmoid(raw_t).clamp(min_time, max_time)

    diagonal_count = max(1, round(num_blocks * diagonal_fraction))
    diagonal_count = min(num_blocks, diagonal_count)
    diagonal_mask = torch.zeros((batch_size, num_blocks), dtype=torch.bool, device=device)
    diagonal_mask[:, :diagonal_count] = True

    # Randomize which anchors carry diagonal/off-diagonal supervision.
    order = torch.rand((batch_size, num_blocks), device=device).argsort(dim=1)
    diagonal_mask = torch.gather(diagonal_mask, 1, order)
    lo = torch.minimum(source_time, target_time)
    hi = torch.maximum(source_time, target_time)
    source_time = torch.where(diagonal_mask[..., None, None], lo, lo)
    target_time = torch.where(diagonal_mask[..., None, None], lo, hi)

    # Avoid a zero-width off-diagonal interval after finite-precision clamping.
    source_time = torch.where(
        diagonal_mask[..., None, None], source_time, source_time.clamp_max(max_time - 1e-3)
    )
    target_time = torch.where(
        diagonal_mask[..., None, None],
        source_time,
        torch.maximum(target_time, source_time + 1e-3).clamp_max(max_time),
    )
    return source_time, target_time, diagonal_mask


def flow_map_step_size(
    source_time: torch.Tensor | float,
    target_time: torch.Tensor | float,
    denominator_floor: float = 0.05,
) -> torch.Tensor:
    """Return gamma=(t-s)/(1-s), with the CFM denominator clamp."""

    source = torch.as_tensor(source_time)
    target = torch.as_tensor(target_time, device=source.device, dtype=source.dtype)
    if torch.any(target < source):
        raise ValueError("target_time must be greater than or equal to source_time")
    return (target - source) / (1.0 - source).clamp_min(denominator_floor)


def transport_categorical_state(
    source_state: torch.Tensor,
    endpoint_state: torch.Tensor,
    source_time: torch.Tensor | float,
    target_time: torch.Tensor | float,
    denominator_floor: float = 0.05,
) -> torch.Tensor:
    """Apply the endpoint-parameterized categorical flow map in state space."""

    gamma = flow_map_step_size(source_time, target_time, denominator_floor)
    gamma = gamma.to(device=source_state.device, dtype=source_state.dtype)
    while gamma.dim() < source_state.dim():
        gamma = gamma.unsqueeze(-1)
    return source_state + gamma * (endpoint_state - source_state)


def topk_endpoint_embeddings(
    logits: torch.Tensor,
    embedding_weight: torch.Tensor,
    topk: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Project a sparse simplex endpoint into token embedding space.

    Full-vocabulary probability-by-embedding products are prohibitively large
    for Qwen3. This preserves gradients through the retained simplex mass and
    renormalizes it, making the approximation explicit and deterministic.
    """

    if topk <= 0:
        raise ValueError("topk must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values, indices = torch.topk(logits.float() / temperature, min(topk, logits.shape[-1]), dim=-1)
    probabilities = F.softmax(values, dim=-1).to(dtype=embedding_weight.dtype)
    embeddings = F.embedding(indices, embedding_weight)
    return (probabilities.unsqueeze(-1) * embeddings).sum(dim=-2)


def sinusoidal_flow_time_embedding(
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Fixed magnitude-stable conditioning for the partial denoiser pi(s,t)."""

    if hidden_size < 4:
        raise ValueError("hidden_size must be at least 4")
    pair_size = hidden_size // 2
    half = max(1, pair_size // 2)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=source_time.device, dtype=torch.float32)
        / max(half - 1, 1)
    )

    def encode(value: torch.Tensor) -> torch.Tensor:
        angles = value.float() * frequencies
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if encoded.shape[-1] < pair_size:
            encoded = F.pad(encoded, (0, pair_size - encoded.shape[-1]))
        return encoded[..., :pair_size]

    source = encode(source_time)
    target = encode(target_time)
    result = torch.cat([source, target], dim=-1)
    if result.shape[-1] < hidden_size:
        result = F.pad(result, (0, hidden_size - result.shape[-1]))
    return result[..., :hidden_size].to(dtype=dtype)


def add_flow_time_conditioning(
    inputs_embeds: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    """Add explicit (s,t) conditioning to every token in each draft block."""

    if scale == 0.0:
        return inputs_embeds
    batch_size, flat_length, hidden_size = inputs_embeds.shape
    if flat_length % block_size != 0:
        raise ValueError(f"Input length {flat_length} is not divisible by block size {block_size}")
    num_blocks = flat_length // block_size
    expected = (batch_size, num_blocks)
    if source_time.shape[:2] != expected or target_time.shape[:2] != expected:
        raise ValueError(f"Expected time tensors beginning with {expected}")

    source = source_time.reshape(batch_size, num_blocks, 1)
    target = target_time.reshape(batch_size, num_blocks, 1)
    time_embed = sinusoidal_flow_time_embedding(source, target, hidden_size, inputs_embeds.dtype)
    time_embed = time_embed.unsqueeze(2).expand(-1, -1, block_size, -1)

    blocks = inputs_embeds.reshape(batch_size, num_blocks, block_size, hidden_size)
    rms = blocks.float().square().mean(dim=-1, keepdim=True).sqrt().mean(dim=(1, 2), keepdim=True)
    conditioned = blocks + scale * rms.to(blocks.dtype) * time_embed
    return conditioned.reshape_as(inputs_embeds)


def make_flowdraft_inputs_embeds(
    model,
    clean_blocks: torch.Tensor,
    mask_token_id: int,
    state_mix: torch.Tensor | float,
    state_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create continuous categorical-flow states as token embeddings.

    Position 0 in each block is always the known AR anchor token. Positions 1..K-1
    are an interpolation from the mask/prior embedding to either clean endpoints
    during training or self-conditioned endpoint tokens during inference.
    """

    embed_tokens = model.model.embed_tokens
    mask_blocks = torch.full_like(clean_blocks, mask_token_id)
    endpoint_blocks = clean_blocks if state_token_ids is None else state_token_ids

    mask_embeds = embed_tokens(mask_blocks)
    endpoint_embeds = embed_tokens(endpoint_blocks)
    if isinstance(state_mix, torch.Tensor):
        state_mix = state_mix.to(dtype=mask_embeds.dtype)
    mixed_embeds = (1.0 - state_mix) * mask_embeds + state_mix * endpoint_embeds

    anchor_embeds = embed_tokens(clean_blocks[:, :, :1])
    mixed_embeds = torch.cat([anchor_embeds, mixed_embeds[:, :, 1:, :]], dim=2)
    batch_size, num_blocks, block_size, hidden_size = mixed_embeds.shape
    return mixed_embeds.reshape(batch_size, num_blocks * block_size, hidden_size)


def make_discrete_flowdraft_state(
    anchor_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor | None,
    diff_len: int,
    mask_token_id: int,
) -> torch.Tensor:
    state_ids = torch.full(
        (anchor_token_ids.shape[0], diff_len),
        mask_token_id,
        dtype=anchor_token_ids.dtype,
        device=anchor_token_ids.device,
    )
    state_ids[:, :1] = anchor_token_ids
    if draft_token_ids is not None and diff_len > 1:
        state_ids[:, 1:] = draft_token_ids[:, : diff_len - 1]
    return state_ids
