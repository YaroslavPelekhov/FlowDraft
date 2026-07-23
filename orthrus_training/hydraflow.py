"""Self-conditioned latent feature flow for one-pass speculative drafting.

The module transports a verified context feature across a draft block with a
small recurrent latent state.  Unlike independent multi-token heads, each
position consumes a continuous prediction of the preceding token embedding.
The frozen Qwen output head is applied once, in parallel, after the trajectory
has been generated.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class LatentFlowBlock(nn.Module):
    def __init__(self, state_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(state_size)
        self.mlp = nn.Sequential(
            nn.Linear(state_size, state_size * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(state_size * 4, state_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.mlp(self.norm(state))


class HydraFlowDrafter(nn.Module):
    """A sequentially-dependent, continuous endpoint flow over draft features."""

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        state_size: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or block_size < 2 or state_size <= 0 or num_layers <= 0:
            raise ValueError("hidden_size, block_size, state_size, and num_layers must be positive")
        self.hidden_size = hidden_size
        self.block_size = block_size
        self.prediction_length = block_size - 1
        self.state_size = state_size
        self.num_layers = num_layers
        self.context_norm = nn.LayerNorm(hidden_size)
        self.anchor_norm = nn.LayerNorm(hidden_size)
        self.context_proj = nn.Linear(hidden_size, state_size)
        self.anchor_proj = nn.Linear(hidden_size, state_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_proj = nn.Linear(hidden_size, state_size)
        self.position = nn.Parameter(torch.zeros(1, self.prediction_length, state_size))
        self.transition = nn.GRUCell(state_size, state_size)
        self.blocks = nn.ModuleList(LatentFlowBlock(state_size, dropout) for _ in range(num_layers))
        self.output_norm = nn.LayerNorm(state_size)
        self.source_endpoint = nn.Linear(state_size, hidden_size)
        self.flow_input_norm = nn.LayerNorm(hidden_size)
        self.flow_input_proj = nn.Linear(hidden_size, state_size)
        self.flow_time = nn.Sequential(nn.Linear(1, state_size), nn.SiLU(), nn.Linear(state_size, state_size))
        self.flow_block = LatentFlowBlock(state_size, dropout)
        self.flow_norm = nn.LayerNorm(state_size)
        self.hidden_endpoint = nn.Linear(state_size, hidden_size)
        self.embedding_endpoint = nn.Linear(state_size, hidden_size)

    def rollout(
        self,
        context_hidden: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        feedback_embedding_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        flow_targets: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce a correlated future trajectory without a target-model pass.

        ``teacher_embeddings`` has shape ``[B, blocks, K-1, H]`` during
        training.  At deployment it is absent and the predicted endpoint feeds
        the next latent transition.  A discrete ``feedback_embedding_fn`` can
        turn each endpoint into an advanced-token embedding from the frozen
        vocabulary; this matches inference more closely than feeding an
        unconstrained continuous vector back into the recurrence.

        With ``flow_targets`` and ``flow_time`` the endpoint map is trained on
        the categorical-flow interpolant ``(1-t) source + t target``.  The
        deployed path always uses the source at ``t=0``.
        """

        if context_hidden.shape != anchor_embeddings.shape:
            raise ValueError("context_hidden and anchor_embeddings must have identical shapes")
        if context_hidden.dim() != 3 or context_hidden.shape[-1] != self.hidden_size:
            raise ValueError("context_hidden must have shape [B, blocks, hidden_size]")
        expected = context_hidden.shape[:2] + (self.prediction_length, self.hidden_size)
        if teacher_embeddings is not None and teacher_embeddings.shape != expected:
            raise ValueError(f"Expected teacher embeddings {expected}, got {tuple(teacher_embeddings.shape)}")
        if flow_targets is not None and flow_targets.shape != expected:
            raise ValueError(f"Expected flow targets {expected}, got {tuple(flow_targets.shape)}")
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")

        batch_size, blocks, _ = context_hidden.shape
        if flow_time is None:
            flow_time = torch.zeros((batch_size, blocks, 1), device=context_hidden.device, dtype=context_hidden.dtype)
        if flow_time.shape != (batch_size, blocks, 1):
            raise ValueError("flow_time must have shape [B, blocks, 1]")
        state = torch.tanh(
            self.context_proj(self.context_norm(context_hidden))
            + self.anchor_proj(self.anchor_norm(anchor_embeddings))
        ).reshape(batch_size * blocks, self.state_size)
        current = anchor_embeddings
        hidden_outputs, embedding_outputs = [], []
        for position in range(self.prediction_length):
            transition_input = self.input_proj(self.input_norm(current)).reshape(batch_size * blocks, self.state_size)
            state = self.transition(transition_input + self.position[:, position, :], state)
            for block in self.blocks:
                state = block(state)
            normalized = self.output_norm(state)
            source = self.source_endpoint(normalized).reshape(batch_size, blocks, self.hidden_size)
            interpolant = source if flow_targets is None else (
                (1.0 - flow_time) * source + flow_time * flow_targets[:, :, position, :]
            )
            flow_state = normalized + self.flow_input_proj(self.flow_input_norm(interpolant)).reshape(batch_size * blocks, self.state_size)
            flow_state = flow_state + self.flow_time(flow_time.reshape(batch_size * blocks, 1))
            flow_state = self.flow_block(flow_state)
            hidden = source + self.hidden_endpoint(self.flow_norm(flow_state)).reshape(batch_size, blocks, self.hidden_size)
            embedding = self.embedding_endpoint(normalized).reshape(batch_size, blocks, self.hidden_size)
            hidden_outputs.append(hidden)
            embedding_outputs.append(embedding)
            predicted = feedback_embedding_fn(hidden.detach()) if feedback_embedding_fn is not None else embedding.detach()
            if teacher_embeddings is None or teacher_forcing_ratio <= 0.0:
                current = predicted
            elif teacher_forcing_ratio >= 1.0:
                current = teacher_embeddings[:, :, position, :]
            else:
                use_teacher = torch.rand((batch_size, blocks, 1), device=current.device) < teacher_forcing_ratio
                current = torch.where(use_teacher, teacher_embeddings[:, :, position, :], predicted)
        return torch.stack(hidden_outputs, dim=2), torch.stack(embedding_outputs, dim=2)
