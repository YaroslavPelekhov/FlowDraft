"""Residual-conditioned flow primitives for lossless AR verification.

R2Flow treats the frozen autoregressive verifier as a fixed-point oracle.  A
candidate block ``y0`` is verified once to obtain ``y1 = J(y0)``.  The small
corrector consumes the verifier hidden states and the discrete residual
``y1 - y0`` and proposes an approximation to ``J(J(y0))``.  A second frozen AR
verification pass remains the only authority that can emit tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ResidualFlowCorrector(nn.Module):
    """A block-local residual flow map operating on frozen verifier states.

    The output projection is initialized to zero.  Consequently the initial
    distribution exactly reproduces the first verifier pass; optimization can
    only learn a correction, never replace the frozen AR oracle outright.
    """

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        bottleneck_size: int = 384,
        num_layers: int = 2,
        num_heads: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or block_size < 2 or bottleneck_size <= 0:
            raise ValueError("hidden_size, block_size, and bottleneck_size must be positive")
        if bottleneck_size % num_heads:
            raise ValueError("bottleneck_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.block_size = block_size
        self.prediction_length = block_size - 1
        self.bottleneck_size = bottleneck_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Frozen hidden state, candidate embedding, verifier-token embedding,
        # and verifier margin.  The residual is explicit rather than inferred.
        self.input_norm = nn.LayerNorm(hidden_size * 3 + 1)
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_size * 3 + 1, bottleneck_size),
            nn.SiLU(),
            nn.Linear(bottleneck_size, bottleneck_size),
        )
        self.position = nn.Parameter(torch.zeros(1, self.prediction_length, bottleneck_size))
        layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_size,
            nhead=num_heads,
            dim_feedforward=bottleneck_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(bottleneck_size)
        self.output_proj = nn.Linear(bottleneck_size, hidden_size)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        verifier_hidden: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        residual_embeddings: torch.Tensor,
        verifier_margin: torch.Tensor,
    ) -> torch.Tensor:
        """Return a correction to frozen verifier hidden states.

        All tensors use flattened ``[batch, blocks * (K - 1), ...]`` layout.
        The returned tensor has the same shape as ``verifier_hidden``.
        """

        if verifier_hidden.shape != candidate_embeddings.shape or verifier_hidden.shape != residual_embeddings.shape:
            raise ValueError("hidden and embedding tensors must have matching shapes")
        if verifier_hidden.shape[-1] != self.hidden_size:
            raise ValueError("Unexpected verifier hidden size")
        if verifier_margin.shape != verifier_hidden.shape[:-1] + (1,):
            raise ValueError("verifier_margin must have shape [batch, positions, 1]")
        batch_size, flat_length, _ = verifier_hidden.shape
        if flat_length % self.prediction_length:
            raise ValueError("ResidualFlowCorrector input must contain complete draft blocks")

        num_blocks = flat_length // self.prediction_length
        features = torch.cat(
            [verifier_hidden, candidate_embeddings, residual_embeddings, verifier_margin], dim=-1
        )
        states = self.input_proj(self.input_norm(features))
        states = states.reshape(batch_size * num_blocks, self.prediction_length, self.bottleneck_size)
        states = states + self.position
        states = self.mixer(states)
        correction = self.output_proj(self.output_norm(states))
        return correction.reshape_as(verifier_hidden)


def verifier_margin(logits: torch.Tensor) -> torch.Tensor:
    """Return a normalized top-2 confidence feature without retaining a vocab map."""

    if logits.shape[-1] < 2:
        raise ValueError("Verifier logits need at least two vocabulary entries")
    top_two = logits.float().topk(k=2, dim=-1).values
    margin = (top_two[..., 0] - top_two[..., 1]) / math.sqrt(float(logits.shape[-1]))
    return margin.unsqueeze(-1).to(dtype=logits.dtype)


def corrector_logits(
    corrector: ResidualFlowCorrector,
    lm_head: nn.Module,
    verifier_hidden: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    residual_embeddings: torch.Tensor,
    verifier_logits: torch.Tensor,
) -> torch.Tensor:
    """Project a residual-flow update through the frozen tied output head."""

    delta = corrector(
        verifier_hidden,
        candidate_embeddings,
        residual_embeddings,
        verifier_margin(verifier_logits),
    )
    return lm_head(verifier_hidden + delta)
