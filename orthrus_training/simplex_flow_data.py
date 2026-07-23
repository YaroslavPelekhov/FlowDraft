"""Frozen Orthrus teacher collection shared by CFM support calibration/training."""

from __future__ import annotations

import torch

from orthrus_training.flowdraft import (
    condition_flowdraft_state,
    make_flowdraft_batch,
    make_flowdraft_inputs_embeds,
    sample_categorical_source_tokens,
    select_endpoint_logits,
)
from orthrus_training.data import sample_anchor_positions


@torch.no_grad()
def collect_base_draft_logits(
    model,
    input_ids: torch.Tensor,
    num_blocks: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return parent draft logits and frozen AR teacher ids by independently anchored block."""

    block_size = int(model.config.block_size)
    anchors = sample_anchor_positions(input_ids.shape[0], input_ids.shape[1], block_size, num_blocks, input_ids.device)
    clean, positions, limits, teacher_positions, _ = make_flowdraft_batch(input_ids, anchors, block_size)
    source = sample_categorical_source_tokens(
        clean,
        int(model.config.vocab_size),
        int(model.config.mask_token_id),
        prior=str(getattr(model.config, "flowdraft_source_prior", "uniform")),
        generator=generator,
    )
    source_time = torch.zeros((input_ids.shape[0], num_blocks, 1, 1), device=input_ids.device)
    target_time = torch.ones_like(source_time)
    flow_inputs = make_flowdraft_inputs_embeds(
        model, clean, int(model.config.mask_token_id), source_time, source_token_ids=source
    )
    flow_inputs = condition_flowdraft_state(
        model,
        flow_inputs,
        source_time,
        target_time,
        block_size,
        float(getattr(model.config, "flowdraft_time_conditioning_scale", 0.0)),
    )
    context = model(input_ids=input_ids, use_cache=True, is_diffusion_pass=False)
    teacher = torch.gather(
        context.logits,
        1,
        teacher_positions.unsqueeze(-1).expand(-1, -1, context.logits.shape[-1]),
    ).argmax(dim=-1)
    draft = model(
        inputs_embeds=flow_inputs,
        position_ids=positions,
        past_key_values=context.past_key_values,
        use_cache=False,
        is_diffusion_pass=True,
        causal_limit=limits,
        ar_seq_len=input_ids.shape[1],
    )
    shape = (input_ids.shape[0], num_blocks, block_size - 1)
    return select_endpoint_logits(draft.logits, block_size).reshape(*shape, -1), teacher.reshape(shape)
