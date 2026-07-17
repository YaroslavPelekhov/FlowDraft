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
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return loss * (temperature**2)


@torch.no_grad()
def token_accuracy(student_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    preds = student_logits.argmax(dim=-1)
    return (preds == target_ids).float().mean()
