"""Self-conditioned latent feature flow for one-pass speculative drafting.

The module transports a verified context feature across a draft block with a
small recurrent latent state.  Unlike independent multi-token heads, each
position consumes a continuous prediction of the preceding token embedding.
The frozen Qwen output head is applied once, in parallel, after the trajectory
has been generated.
"""

from __future__ import annotations

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
        self.hidden_endpoint = nn.Linear(state_size, hidden_size)
        self.embedding_endpoint = nn.Linear(state_size, hidden_size)

    def rollout(
        self,
        context_hidden: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce a correlated future trajectory without a target-model pass.

        ``teacher_embeddings`` has shape ``[B, blocks, K-1, H]`` during
        training.  At deployment it is absent and the predicted continuous
        endpoint embedding feeds the next latent transition.
        """

        if context_hidden.shape != anchor_embeddings.shape:
            raise ValueError("context_hidden and anchor_embeddings must have identical shapes")
        if context_hidden.dim() != 3 or context_hidden.shape[-1] != self.hidden_size:
            raise ValueError("context_hidden must have shape [B, blocks, hidden_size]")
        expected = context_hidden.shape[:2] + (self.prediction_length, self.hidden_size)
        if teacher_embeddings is not None and teacher_embeddings.shape != expected:
            raise ValueError(f"Expected teacher embeddings {expected}, got {tuple(teacher_embeddings.shape)}")
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")

        batch_size, blocks, _ = context_hidden.shape
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
            hidden = self.hidden_endpoint(normalized).reshape(batch_size, blocks, self.hidden_size)
            embedding = self.embedding_endpoint(normalized).reshape(batch_size, blocks, self.hidden_size)
            hidden_outputs.append(hidden)
            embedding_outputs.append(embedding)
            if teacher_embeddings is None:
                current = embedding
            else:
                current = (
                    teacher_forcing_ratio * teacher_embeddings[:, :, position, :]
                    + (1.0 - teacher_forcing_ratio) * embedding
                )
        return torch.stack(hidden_outputs, dim=2), torch.stack(embedding_outputs, dim=2)
