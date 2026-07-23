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
    one_jump_fraction: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample diagonal VFM and off-diagonal CFM time pairs per block.

    This follows Categorical Flow Maps: a fixed fraction is trained on s=t,
    while the remainder uses s<t. Times follow the paper's logit-normal law.
    """

    if not 0.0 < diagonal_fraction <= 1.0:
        raise ValueError("diagonal_fraction must be in (0, 1]")
    if min_time < 0.0 or max_time >= 1.0 or min_time > max_time:
        raise ValueError(f"Invalid time interval [{min_time}, {max_time}]")
    if not 0.0 <= one_jump_fraction <= 1.0:
        raise ValueError("one_jump_fraction must be in [0, 1]")

    shape = (batch_size, num_blocks, 1, 1)
    raw_s = torch.randn(shape, device=device) * logit_std + logit_mean
    raw_t = torch.randn(shape, device=device) * logit_std + logit_mean
    source_time = torch.sigmoid(raw_s).clamp(min_time, max_time)
    target_time = torch.sigmoid(raw_t).clamp(min_time, max_time)
    diagonal_time = torch.sigmoid(
        torch.randn(shape, device=device) * logit_std + logit_mean
    ).clamp(min_time, max_time)

    diagonal_count = max(1, round(num_blocks * diagonal_fraction))
    diagonal_count = min(num_blocks, diagonal_count)
    diagonal_mask = torch.zeros((batch_size, num_blocks), dtype=torch.bool, device=device)
    diagonal_mask[:, :diagonal_count] = True

    # Randomize which anchors carry diagonal/off-diagonal supervision.
    order = torch.rand((batch_size, num_blocks), device=device).argsort(dim=1)
    diagonal_mask = torch.gather(diagonal_mask, 1, order)
    lo = torch.minimum(source_time, target_time)
    hi = torch.maximum(source_time, target_time)
    # The VFM term is evaluated at x_t.  Sampling a separate diagonal t is
    # essential: using min(s, t) would bias the diagonal loss toward noise.
    source_time = torch.where(diagonal_mask[..., None, None], diagonal_time, lo)
    target_time = torch.where(diagonal_mask[..., None, None], diagonal_time, hi)

    # Avoid a zero-width off-diagonal interval after finite-precision clamping.
    source_time = torch.where(
        diagonal_mask[..., None, None], source_time, source_time.clamp_max(max_time - 1e-3)
    )
    target_time = torch.where(
        diagonal_mask[..., None, None],
        source_time,
        torch.maximum(target_time, source_time + 1e-3).clamp_max(max_time),
    )

    if one_jump_fraction > 0.0 and diagonal_count < num_blocks:
        off_count = num_blocks - diagonal_count
        one_jump_count = min(off_count, round(off_count * one_jump_fraction))
        if one_jump_count:
            scores = torch.rand((batch_size, num_blocks), device=device)
            scores = scores.masked_fill(diagonal_mask, -1.0)
            selected = scores.topk(one_jump_count, dim=1).indices
            one_jump_mask = torch.zeros_like(diagonal_mask)
            one_jump_mask.scatter_(1, selected, True)
            source_time = torch.where(one_jump_mask[..., None, None], torch.zeros_like(source_time), source_time)
            target_time = torch.where(one_jump_mask[..., None, None], torch.ones_like(target_time), target_time)
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


@torch.no_grad()
def exact_endpoint_embeddings(
    logits: torch.Tensor,
    embedding_weight: torch.Tensor,
    temperature: float = 1.0,
    vocab_chunk_size: int = 8192,
) -> torch.Tensor:
    """Compute E_{v~softmax(logits)}[embedding(v)] without top-k truncation.

    CFM transports a simplex-valued endpoint.  Renormalizing the top-k logits
    changes that endpoint and invalidates the ECLD target.  This chunked
    computation keeps the complete vocabulary while avoiding a second dense
    probability tensor the size of the logits.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    if logits.shape[-1] != embedding_weight.shape[0]:
        raise ValueError("logits vocabulary and embedding vocabulary must match")

    original_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    flat_logits = logits.detach().float().reshape(-1, vocab_size) / temperature
    row_max = flat_logits.max(dim=-1, keepdim=True).values
    numerator = torch.zeros(
        (flat_logits.shape[0], embedding_weight.shape[1]),
        device=logits.device,
        dtype=torch.float32,
    )
    denominator = torch.zeros((flat_logits.shape[0], 1), device=logits.device, dtype=torch.float32)
    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        weights = torch.exp(flat_logits[:, start:end] - row_max)
        denominator += weights.sum(dim=-1, keepdim=True)
        numerator += weights @ embedding_weight[start:end].float()
    result = numerator / denominator.clamp_min(torch.finfo(numerator.dtype).tiny)
    return result.reshape(*original_shape, embedding_weight.shape[1]).to(dtype=embedding_weight.dtype)


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


def condition_flowdraft_state(
    model,
    inputs_embeds: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    """Use the CFM adapter when enabled, otherwise preserve legacy checkpoints."""

    if not bool(getattr(model.config, "flowdraft_state_adapter", False)):
        return add_flow_time_conditioning(
            inputs_embeds, source_time, target_time, block_size=block_size, scale=scale
        )
    hidden_size = inputs_embeds.shape[-1]
    batch_size = inputs_embeds.shape[0]
    num_blocks = inputs_embeds.shape[1] // block_size
    time_embed = sinusoidal_flow_time_embedding(
        source_time.reshape(batch_size, num_blocks, 1),
        target_time.reshape(batch_size, num_blocks, 1),
        hidden_size,
        inputs_embeds.dtype,
    )
    return model.flowdraft_state_adapter(inputs_embeds, time_embed, block_size)


def make_flowdraft_inputs_embeds(
    model,
    clean_blocks: torch.Tensor,
    mask_token_id: int,
    state_mix: torch.Tensor | float,
    state_token_ids: torch.Tensor | None = None,
    source_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create continuous categorical-flow states as token embeddings.

    Position 0 in each block is always the known AR anchor token. Positions 1..K-1
    are an interpolation from the mask/prior embedding to either clean endpoints
    during training or self-conditioned endpoint tokens during inference.
    """

    embed_tokens = model.model.embed_tokens
    mask_blocks = (
        torch.full_like(clean_blocks, mask_token_id)
        if source_token_ids is None
        else source_token_ids
    )
    if mask_blocks.shape != clean_blocks.shape:
        raise ValueError(
            f"source_token_ids must have shape {tuple(clean_blocks.shape)}, "
            f"got {tuple(mask_blocks.shape)}"
        )
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


def sample_categorical_source_tokens(
    clean_blocks: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    prior: str = "uniform",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the stochastic x_0 vertices used by the categorical flow.

    A fixed mask is a degenerate source distribution and cannot represent a
    non-trivial generative flow. Uniform one-hot vertices provide a genuine
    stochastic simplex prior while keeping the state projection exact:
    E(x_t)=(1-t)E(x_0)+tE(x_1).
    """

    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than one")
    if prior == "mask":
        source = torch.full_like(clean_blocks, mask_token_id)
    elif prior == "uniform":
        source = torch.randint(
            low=0,
            high=vocab_size,
            size=clean_blocks.shape,
            device=clean_blocks.device,
            dtype=clean_blocks.dtype,
            generator=generator,
        )
    else:
        raise ValueError(f"Unsupported categorical source prior: {prior}")

    # Position zero is the AR-verified anchor, not part of the random source.
    source[:, :, :1] = clean_blocks[:, :, :1]
    return source


def make_endpoint_blocks(
    anchor_token_ids: torch.Tensor,
    endpoint_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Join verified anchors and K-1 endpoint tokens into K-token blocks."""

    if anchor_token_ids.dim() != 3 or anchor_token_ids.shape[-1] != 1:
        raise ValueError("anchor_token_ids must have shape [batch, blocks, 1]")
    if endpoint_token_ids.dim() != 3:
        raise ValueError("endpoint_token_ids must have shape [batch, blocks, K-1]")
    if endpoint_token_ids.shape[:2] != anchor_token_ids.shape[:2]:
        raise ValueError("anchor and endpoint block dimensions must match")
    return torch.cat([anchor_token_ids, endpoint_token_ids], dim=-1)


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
