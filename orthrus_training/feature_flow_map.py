"""Conditional feature-space categorical flow maps for one-verifier drafting.

The map operates on the frozen model's final hidden-state trajectory.  Its
endpoint is projected through the frozen LM head to categorical proposals, and
the ordinary AR model verifies every proposal before it can be emitted.  The
head is deliberately block-parallel: it never runs a second Qwen forward pass.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _time_features(time: torch.Tensor, width: int) -> torch.Tensor:
    """Smooth Fourier features for a scalar CFM time in ``[0, 1]``."""

    if time.dim() != 2:
        raise ValueError("time must have shape [batch, blocks]")
    half = max(1, width // 2)
    frequencies = torch.exp(
        torch.linspace(0.0, math.log(1000.0), half, device=time.device, dtype=torch.float32)
    )
    phases = time.float().unsqueeze(-1) * frequencies * (2.0 * math.pi)
    features = torch.cat([phases.sin(), phases.cos()], dim=-1)
    if features.shape[-1] < width:
        features = torch.nn.functional.pad(features, (0, width - features.shape[-1]))
    return features[..., :width]


class FeatureFlowMapHead(nn.Module):
    """Endpoint map ``pi(x_s, s -> 1 | h_context, x_anchor)``.

    ``x_s`` is a continuous trajectory in the frozen model's feature space. A
    training diagonal is formed by interpolating a deployment-time source and
    the teacher-forced final hidden states.  At inference we always use
    ``s=0`` and the same context-scaled source distribution.
    """

    format_name = "feature_flow_map_v1"

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        latent_size: int = 768,
        num_layers: int = 4,
        num_heads: int = 12,
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
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, latent_size),
            nn.GELU(),
            nn.Linear(latent_size, latent_size),
        )
        self.state_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, latent_size))
        self.time_proj = nn.Sequential(
            nn.Linear(latent_size, latent_size), nn.SiLU(), nn.Linear(latent_size, latent_size)
        )
        self.position = nn.Parameter(torch.empty(1, self.prediction_length, latent_size))
        nn.init.normal_(self.position, std=0.02)
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
        self.output = nn.Sequential(nn.LayerNorm(latent_size), nn.Linear(latent_size, hidden_size))

    def forward(
        self,
        context_hidden: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a complete endpoint trajectory.

        Args:
            context_hidden: ``[B, A, H]`` exact cached AR features before the
                verified anchor token.
            anchor_embeddings: ``[B, A, H]`` for that verified token.
            state: ``[B, A, K-1, H]`` continuous CFM state.
            time: ``[B, A]`` current CFM time.
        """

        if context_hidden.shape != anchor_embeddings.shape:
            raise ValueError("context_hidden and anchor_embeddings must match")
        expected = context_hidden.shape[:2] + (self.prediction_length, self.hidden_size)
        if state.shape != expected:
            raise ValueError(f"Expected state {expected}, got {tuple(state.shape)}")
        if time.shape != context_hidden.shape[:2]:
            raise ValueError("time must match [batch, anchors]")
        condition = self.condition_proj(torch.cat([context_hidden, anchor_embeddings], dim=-1)).unsqueeze(2)
        # Fourier phases are accumulated in FP32 for numerical stability, but
        # the learned projection follows the head's BF16/FP32 execution dtype.
        time_features = self.time_proj(_time_features(time, self.latent_size).to(dtype=state.dtype)).unsqueeze(2)
        states = self.state_proj(state) + condition + time_features + self.position
        batch, anchors = context_hidden.shape[:2]
        states = states.reshape(batch * anchors, self.prediction_length, self.latent_size)
        endpoint = self.output(self.mixer(states))
        return endpoint.reshape(batch, anchors, self.prediction_length, self.hidden_size)


def feature_flow_source(
    context_hidden: torch.Tensor,
    prediction_length: int,
    generator: torch.Generator | None = None,
    mode: str = "gaussian",
) -> torch.Tensor:
    """Create a deployment-identical conditional CFM source trajectory.

    ``context`` is a deterministic conditional base distribution for greedy
    decoding: it retains the AR feature geometry instead of forcing the map to
    denoise irrelevant random coordinates before it can predict token one.
    ``gaussian`` remains available for stochastic-source ablations.
    """

    if context_hidden.dim() != 3:
        raise ValueError("context_hidden must have shape [B, anchors, H]")
    shape = context_hidden.shape[:2] + (prediction_length, context_hidden.shape[-1])
    if mode == "context":
        return context_hidden.unsqueeze(2).expand(shape)
    if mode != "gaussian":
        raise ValueError(f"Unsupported feature-flow source mode: {mode}")
    rms = context_hidden.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    return torch.randn(shape, device=context_hidden.device, dtype=context_hidden.dtype, generator=generator) * rms.unsqueeze(2).to(context_hidden.dtype)


def feature_flow_interpolate(source: torch.Tensor, target: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Sample the straight conditional path used by endpoint flow matching."""

    if source.shape != target.shape or source.dim() != 4:
        raise ValueError("source and target must have identical [B, A, K-1, H] shapes")
    if time.shape != source.shape[:2]:
        raise ValueError("time must match [B, A]")
    alpha = time.to(dtype=source.dtype).unsqueeze(-1).unsqueeze(-1)
    return source.lerp(target, alpha)
