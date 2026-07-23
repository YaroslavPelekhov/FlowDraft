"""Low-rank conditional flow over future final-layer trajectories.

CacheFlow deliberately avoids a second large-model pass.  It maps a cached
context feature, the already verified anchor token, and a continuous source
trajectory to a complete future final-hidden-state trajectory.  The frozen LM
head converts the trajectory to a correlated token proposal; a single regular
AR verifier remains responsible for every emitted token.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CacheFlowTrajectoryHead(nn.Module):
    """Small block-parallel endpoint flow in Qwen final-hidden-state space."""

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        latent_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or block_size < 2 or latent_size <= 0:
            raise ValueError("hidden_size, block_size, and latent_size must be positive")
        if latent_size % num_heads:
            raise ValueError("latent_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.block_size = block_size
        self.prediction_length = block_size - 1
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_proj = nn.Linear(hidden_size * 2, latent_size)
        self.source_norm = nn.LayerNorm(hidden_size)
        self.source_proj = nn.Linear(hidden_size, latent_size)
        self.position = nn.Parameter(torch.zeros(1, self.prediction_length, latent_size))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_size,
            nhead=num_heads,
            dim_feedforward=latent_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(latent_size)
        self.endpoint_proj = nn.Linear(latent_size, hidden_size)

    def forward(
        self,
        context_hidden: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        source_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Predict final hidden states for every token after the anchor.

        ``context_hidden`` and ``anchor_embeddings`` are ``[B, blocks, H]``;
        ``source_trajectory`` is ``[B, blocks, K-1, H]``.
        """

        if context_hidden.shape != anchor_embeddings.shape:
            raise ValueError("context_hidden and anchor_embeddings must have identical shapes")
        if context_hidden.shape[-1] != self.hidden_size:
            raise ValueError("Unexpected hidden size")
        expected_source = context_hidden.shape[:2] + (self.prediction_length, self.hidden_size)
        if source_trajectory.shape != expected_source:
            raise ValueError(f"Expected source trajectory {expected_source}, got {tuple(source_trajectory.shape)}")
        batch_size, num_blocks, _ = context_hidden.shape
        context = self.context_proj(torch.cat([context_hidden, anchor_embeddings], dim=-1)).unsqueeze(2)
        source = self.source_proj(self.source_norm(source_trajectory))
        states = source + context + self.position
        states = states.reshape(batch_size * num_blocks, self.prediction_length, self.latent_size)
        states = self.mixer(states)
        endpoint = self.endpoint_proj(self.output_norm(states))
        return endpoint.reshape(batch_size, num_blocks, self.prediction_length, self.hidden_size)


def flow_source_like(target_hidden: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample a scale-matched continuous source for a one-jump hidden flow."""

    if target_hidden.dim() != 4:
        raise ValueError("target_hidden must have shape [B, blocks, K-1, H]")
    rms = target_hidden.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    return torch.randn(
        target_hidden.shape,
        device=target_hidden.device,
        dtype=target_hidden.dtype,
        generator=generator,
    ) * rms.to(dtype=target_hidden.dtype)


def flow_source_from_context(
    context_hidden: torch.Tensor,
    prediction_length: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the deployment-time source without observing future states."""

    if context_hidden.dim() != 3:
        raise ValueError("context_hidden must have shape [B, blocks, H]")
    rms = context_hidden.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    shape = context_hidden.shape[:2] + (prediction_length, context_hidden.shape[-1])
    return torch.randn(shape, device=context_hidden.device, dtype=context_hidden.dtype, generator=generator) * rms.unsqueeze(2).to(context_hidden.dtype)
