from __future__ import annotations

import torch


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
