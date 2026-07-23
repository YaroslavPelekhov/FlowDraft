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
    correct = (preds == flat_targets).float()
    greedy_prefix = torch.cumprod(correct, dim=-1).sum(dim=-1)
    nll = F.cross_entropy(
        flat_logits.reshape(-1, vocab_size),
        flat_targets.reshape(-1),
        reduction="none",
    ).reshape(num_groups, steps)
    prefix_prob = torch.exp((-torch.cumsum(nll, dim=-1)).clamp(min=-30.0, max=0.0))
    return {
        "first_token_acc": correct[:, 0].mean(),
        "first_token_ce": nll[:, 0].mean(),
        "greedy_prefix_acceptance": greedy_prefix.mean(),
        "prefix_expected_acceptance": prefix_prob.sum(dim=-1).mean(),
    }


@torch.no_grad()
def token_accuracy(student_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    preds = student_logits.argmax(dim=-1)
    return (preds == target_ids).float().mean()


def bounded_jsd_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bounded Jensen-Shannon distillation for semigroup consistency."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_log = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    student_prob = student_log.exp()
    teacher_prob = teacher_log.exp()
    mixture_log = torch.logaddexp(student_log, teacher_log) - torch.log(
        torch.tensor(2.0, device=student_logits.device)
    )
    per_token = 0.5 * (
        (student_prob * (student_log - mixture_log)).sum(dim=-1)
        + (teacher_prob * (teacher_log - mixture_log)).sum(dim=-1)
    )
    if weights is not None:
        broadcast = _broadcast_position_weights(weights, per_token)
        return (
            per_token * broadcast
        ).sum() / broadcast.expand_as(per_token).sum().clamp_min(1e-8) * (temperature**2)
    return per_token.mean() * (temperature**2)


def verifier_aligned_losses(
    draft_logits: torch.Tensor,
    verifier_logits: torch.Tensor,
    block_size: int,
    temperature: float = 1.0,
    rejected_decay: float = 0.8,
    reverse_kl_clip: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Losses aligned with left-to-right speculative acceptance.

    Accepted positions preserve the target distribution with forward KL. The
    first rejected token and its suffix use clipped reverse KL, while top-1 CE
    directly repairs the decisions that determine greedy prefix survival.
    """

    if draft_logits.shape != verifier_logits.shape:
        raise ValueError("draft and verifier logits must have identical shapes")
    if not 0.0 < rejected_decay <= 1.0:
        raise ValueError("rejected_decay must be in (0, 1]")
    if reverse_kl_clip < 0.0:
        raise ValueError("reverse_kl_clip must be non-negative")

    steps = block_size - 1
    if draft_logits.shape[1] % steps:
        raise ValueError("token length must be divisible by block_size - 1")
    batch_size, _, vocab_size = draft_logits.shape
    num_blocks = draft_logits.shape[1] // steps
    draft = draft_logits.reshape(batch_size, num_blocks, steps, vocab_size).float()
    verifier = verifier_logits.detach().reshape_as(draft).float()

    target_ids = verifier.argmax(dim=-1)
    draft_ids = draft.argmax(dim=-1)
    matches = draft_ids.eq(target_ids)
    accepted = matches.to(torch.int64).cumprod(dim=-1).bool()
    alive_before = torch.cat(
        [
            torch.ones_like(accepted[:, :, :1]),
            accepted[:, :, :-1],
        ],
        dim=-1,
    )
    first_rejected = alive_before & ~matches

    draft_log = F.log_softmax(draft / temperature, dim=-1)
    verifier_log = F.log_softmax(verifier / temperature, dim=-1)
    draft_prob = draft_log.exp()
    verifier_prob = verifier_log.exp()

    forward_per_token = (verifier_prob * (verifier_log - draft_log)).sum(dim=-1)
    accepted_weights = alive_before.float()
    forward_kl = (forward_per_token * accepted_weights).sum() / accepted_weights.sum().clamp_min(1.0)

    if reverse_kl_clip > 0:
        # Mix in a small detached proposal mass instead of flooring every
        # vocabulary entry. Per-entry flooring is almost uniform for a 150k
        # vocabulary and destroys the verifier signal.
        verifier_for_reverse = (
            (1.0 - reverse_kl_clip) * verifier_prob
            + reverse_kl_clip * draft_prob.detach()
        )
        verifier_reverse_log = verifier_for_reverse.clamp_min(1e-12).log()
    else:
        verifier_reverse_log = verifier_log
    reverse_per_token = (draft_prob * (draft_log - verifier_reverse_log)).sum(dim=-1)

    positions = torch.arange(steps, device=draft.device).view(1, 1, steps)
    first_index = torch.where(
        first_rejected,
        positions,
        torch.full_like(positions, steps),
    ).amin(dim=-1, keepdim=True)
    suffix_offset = positions - first_index
    rejected_weights = torch.where(
        suffix_offset >= 0,
        rejected_decay ** suffix_offset.float(),
        torch.zeros_like(suffix_offset, dtype=torch.float32),
    )
    rejected_weights = rejected_weights * (first_index < steps)
    reverse_kl = (reverse_per_token * rejected_weights).sum() / rejected_weights.sum().clamp_min(1.0)

    top1_per_token = F.cross_entropy(
        draft.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape(batch_size, num_blocks, steps)
    top1_weights = alive_before.float()
    top1_ce = (top1_per_token * top1_weights).sum() / top1_weights.sum().clamp_min(1.0)

    return {
        "forward_kl": forward_kl * (temperature**2),
        "reverse_kl": reverse_kl * (temperature**2),
        "top1_ce": top1_ce,
        "target_ids": target_ids.reshape(batch_size, num_blocks * steps),
        "accepted_mask": accepted.reshape(batch_size, num_blocks * steps),
        "first_rejected_mask": first_rejected.reshape(batch_size, num_blocks * steps),
        "greedy_prefix_acceptance": accepted.float().sum(dim=-1).mean(),
        "first_token_accuracy": matches[:, :, 0].float().mean(),
    }
