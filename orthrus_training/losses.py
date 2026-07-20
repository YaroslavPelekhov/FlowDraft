from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_logits(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    batch_size, count = positions.shape
    vocab_size = logits.shape[-1]
    expanded = positions.unsqueeze(-1).expand(batch_size, count, vocab_size)
    return torch.gather(logits, dim=1, index=expanded)


def forward_kl_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    weights: torch.Tensor | None = None,
    reduction: str = "batchmean",
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if reduction not in {"batchmean", "tokenmean"}:
        raise ValueError(f"Unsupported KL reduction: {reduction}")

    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    if weights is not None:
        per_token = per_token * _broadcast_position_weights(weights, per_token)

    if reduction == "batchmean":
        loss = per_token.sum() / student_logits.shape[0]
    else:
        if weights is None:
            loss = per_token.mean()
        else:
            normalizer = _broadcast_position_weights(weights, per_token).expand_as(per_token).sum().clamp_min(1e-8)
            loss = per_token.sum() / normalizer
    return loss * (temperature**2)


def _broadcast_position_weights(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if weights.dim() == 1:
        if values.shape[-1] % weights.numel() != 0:
            raise ValueError(
                f"Cannot repeat {weights.numel()} position weights over {values.shape[-1]} values"
            )
        repeats = values.shape[-1] // weights.numel()
        weights = weights.repeat(repeats)
    return weights.to(device=values.device, dtype=values.dtype).reshape((1,) * (values.dim() - 1) + (-1,))


def prefix_survival_weights(
    block_size: int,
    decay: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Weights that make early draft positions count for all later prefix survival.

    Orthrus/FlowDraft verifies a block left-to-right and stops at the first
    mismatch. A mistake at draft position 0 therefore destroys every longer
    accepted prefix, while a mistake near the end only affects the tail. These
    weights are the reverse cumulative sum of discounted prefix rewards,
    normalized to mean one so the loss scale stays close to ordinary CE.
    """

    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    if decay <= 0.0 or decay > 1.0:
        raise ValueError("decay must be in (0, 1]")

    steps = block_size - 1
    rewards = decay ** torch.arange(steps, device=device, dtype=dtype)
    weights = torch.flip(torch.cumsum(torch.flip(rewards, dims=[0]), dim=0), dims=[0])
    return weights / weights.mean().clamp_min(1e-8)


def weighted_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    per_token = F.cross_entropy(
        logits.reshape(-1, vocab_size).float(),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape_as(target_ids)
    if weights is None:
        return per_token.mean()

    broadcast = _broadcast_position_weights(weights, per_token)
    return (per_token * broadcast).sum() / broadcast.expand_as(per_token).sum().clamp_min(1e-8)


def prefix_survival_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    block_size: int,
    decay: float,
) -> torch.Tensor:
    weights = prefix_survival_weights(
        block_size=block_size,
        decay=decay,
        device=logits.device,
        dtype=torch.float32,
    )
    return weighted_cross_entropy(logits, target_ids, weights=weights)


@torch.no_grad()
def prefix_acceptance_metrics(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    block_size: int,
) -> dict[str, torch.Tensor]:
    steps = block_size - 1
    if target_ids.shape[-1] % steps != 0:
        raise ValueError(f"Target length {target_ids.shape[-1]} is not divisible by {steps}")

    vocab_size = logits.shape[-1]
    num_groups = target_ids.numel() // steps
    flat_logits = logits.reshape(num_groups, steps, vocab_size).float()
    flat_targets = target_ids.reshape(num_groups, steps)
    preds = flat_logits.argmax(dim=-1)
    nll = F.cross_entropy(
        flat_logits.reshape(-1, vocab_size),
        flat_targets.reshape(-1),
        reduction="none",
    ).reshape(num_groups, steps)
    prefix_prob = torch.exp((-torch.cumsum(nll, dim=-1)).clamp(min=-30.0, max=0.0))
    return {
        "first_token_acc": (preds[:, 0] == flat_targets[:, 0]).float().mean(),
        "first_token_ce": nll[:, 0].mean(),
        "prefix_expected_acceptance": prefix_prob.sum(dim=-1).mean(),
    }


@torch.no_grad()
def token_accuracy(student_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    preds = student_logits.argmax(dim=-1)
    return (preds == target_ids).float().mean()
