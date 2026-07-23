"""Attention-conditioned endpoint Flow Map drafter.

This module keeps the useful EAGLE conditioning pattern: every draft position
is represented by the pair ``(AR feature, advanced-token embedding)``.  A
small causal attention stack correlates the positions inside a proposed block;
the final endpoint is additionally trained as a conditional one-step flow map.
Only the t=0 endpoint path is used at decoding time.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenAttention(nn.Module):
    """A compact causal attention layer over feature/token pairs."""

    def __init__(self, hidden_size: int, state_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if state_size % num_heads:
            raise ValueError("state_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_heads = num_heads
        self.head_dim = state_size // num_heads
        self.feature_norm = nn.LayerNorm(hidden_size)
        self.token_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size * 2, state_size, bias=False)
        self.k_proj = nn.Linear(hidden_size * 2, state_size, bias=False)
        self.v_proj = nn.Linear(hidden_size * 2, state_size, bias=False)
        self.o_proj = nn.Linear(state_size, hidden_size, bias=False)
        self.post_norm = nn.LayerNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, state_size * 2, bias=False)
        self.up_proj = nn.Linear(hidden_size, state_size * 2, bias=False)
        self.down_proj = nn.Linear(state_size * 2, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        feature: torch.Tensor,
        token_embedding: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Advance one position and append its projected K/V to ``cache``."""

        batch = feature.shape[0]
        pair = torch.cat((self.feature_norm(feature), self.token_norm(token_embedding)), dim=-1)
        query = self.q_proj(pair).reshape(batch, self.num_heads, self.head_dim)
        key = self.k_proj(pair).reshape(batch, self.num_heads, self.head_dim)
        value = self.v_proj(pair).reshape(batch, self.num_heads, self.head_dim)
        if cache is None:
            keys, values = key.unsqueeze(2), value.unsqueeze(2)
        else:
            keys = torch.cat((cache[0], key.unsqueeze(2)), dim=2)
            values = torch.cat((cache[1], value.unsqueeze(2)), dim=2)
        scores = torch.einsum("bhd,bhld->bhl", query, keys) * (self.head_dim ** -0.5)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(feature.dtype)
        attended = torch.einsum("bhl,bhld->bhd", weights, values).reshape(batch, self.state_size)
        feature = feature + self.dropout(self.o_proj(attended))
        normalized = self.post_norm(feature)
        feature = feature + self.dropout(self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized)))
        return feature, (keys, values)


class EagleFlowDrafter(nn.Module):
    """EAGLE-conditioned feature trajectory with a t=0 endpoint Flow Map."""

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        state_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
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
        self.num_heads = num_heads
        self.input_norm = nn.LayerNorm(hidden_size)
        self.position = nn.Parameter(torch.zeros(1, self.prediction_length, hidden_size))
        self.layers = nn.ModuleList(
            FeatureTokenAttention(hidden_size, state_size, num_heads, dropout) for _ in range(num_layers)
        )
        self.source_norm = nn.LayerNorm(hidden_size)
        self.source_residual = nn.Linear(hidden_size, hidden_size, bias=False)
        self.flow_norm = nn.LayerNorm(hidden_size * 2)
        self.flow_time = nn.Sequential(
            nn.Linear(1, state_size), nn.SiLU(), nn.Linear(state_size, hidden_size)
        )
        self.flow_hidden = nn.Sequential(
            nn.Linear(hidden_size * 2, state_size), nn.SiLU(), nn.Linear(state_size, hidden_size)
        )
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.embedding_endpoint = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.source_residual.weight)
        nn.init.zeros_(self.flow_hidden[-1].weight)
        nn.init.zeros_(self.flow_hidden[-1].bias)
        nn.init.zeros_(self.flow_time[-1].weight)
        nn.init.zeros_(self.flow_time[-1].bias)

    def rollout(
        self,
        context_hidden: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor | None = None,
        teacher_features: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        feedback_embedding_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        flow_targets: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a correlated feature endpoint for every position in a block."""

        if context_hidden.shape != anchor_embeddings.shape:
            raise ValueError("context_hidden and anchor_embeddings must have identical shapes")
        if context_hidden.dim() != 3 or context_hidden.shape[-1] != self.hidden_size:
            raise ValueError("context_hidden must have shape [B, blocks, hidden_size]")
        expected = context_hidden.shape[:2] + (self.prediction_length, self.hidden_size)
        for name, value in (("teacher_embeddings", teacher_embeddings), ("teacher_features", teacher_features), ("flow_targets", flow_targets)):
            if value is not None and value.shape != expected:
                raise ValueError(f"Expected {name} shape {expected}, got {tuple(value.shape)}")
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")

        batch_size, blocks, _ = context_hidden.shape
        if flow_time is None:
            flow_time = torch.zeros((batch_size, blocks, 1), device=context_hidden.device, dtype=context_hidden.dtype)
        if flow_time.shape != (batch_size, blocks, 1):
            raise ValueError("flow_time must have shape [B, blocks, 1]")
        flat = batch_size * blocks
        feature = self.input_norm(context_hidden).reshape(flat, self.hidden_size)
        token = anchor_embeddings.reshape(flat, self.hidden_size)
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.num_layers
        hidden_outputs, embedding_outputs = [], []
        for position in range(self.prediction_length):
            feature = feature + self.position[:, position, :]
            for layer_index, layer in enumerate(self.layers):
                feature, caches[layer_index] = layer(feature, token, caches[layer_index])
            source = feature + self.source_residual(self.source_norm(feature))
            source = source.reshape(batch_size, blocks, self.hidden_size)
            interpolant = source if flow_targets is None else (
                (1.0 - flow_time) * source + flow_time * flow_targets[:, :, position, :]
            )
            flow_pair = torch.cat((source, interpolant), dim=-1)
            flow_update = self.flow_hidden(self.flow_norm(flow_pair))
            flow_update = flow_update + self.flow_time(flow_time.reshape(flat, 1)).reshape_as(flow_update)
            hidden = source + flow_update
            embedding = self.embedding_endpoint(self.embedding_norm(feature)).reshape(batch_size, blocks, self.hidden_size)
            hidden_outputs.append(hidden)
            embedding_outputs.append(embedding)
            predicted_token = feedback_embedding_fn(hidden.detach()) if feedback_embedding_fn is not None else embedding.detach()
            predicted_feature = hidden.detach()
            if teacher_embeddings is not None and teacher_features is not None and teacher_forcing_ratio > 0.0:
                if teacher_forcing_ratio >= 1.0:
                    token = teacher_embeddings[:, :, position, :].reshape(flat, self.hidden_size)
                    feature = teacher_features[:, :, position, :].reshape(flat, self.hidden_size)
                else:
                    use_teacher = torch.rand((batch_size, blocks, 1), device=feature.device) < teacher_forcing_ratio
                    token = torch.where(use_teacher, teacher_embeddings[:, :, position, :], predicted_token).reshape(flat, self.hidden_size)
                    feature = torch.where(use_teacher, teacher_features[:, :, position, :], predicted_feature).reshape(flat, self.hidden_size)
            else:
                token = predicted_token.reshape(flat, self.hidden_size)
                feature = predicted_feature.reshape(flat, self.hidden_size)
        return torch.stack(hidden_outputs, dim=2), torch.stack(embedding_outputs, dim=2)


class ParallelFeatureTokenAttention(nn.Module):
    """Causal block attention evaluated in one fused operation."""

    def __init__(self, hidden_size: int, state_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if state_size % num_heads:
            raise ValueError("state_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_heads = num_heads
        self.head_dim = state_size // num_heads
        self.feature_norm = nn.LayerNorm(hidden_size)
        self.token_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size * 2, state_size, bias=False)
        self.k_proj = nn.Linear(hidden_size * 2, state_size, bias=False)
        self.v_proj = nn.Linear(hidden_size * 2, state_size, bias=False)
        self.o_proj = nn.Linear(state_size, hidden_size, bias=False)
        self.post_norm = nn.LayerNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, state_size * 2, bias=False)
        self.up_proj = nn.Linear(hidden_size, state_size * 2, bias=False)
        self.down_proj = nn.Linear(state_size * 2, hidden_size, bias=False)
        self.dropout = dropout

    def forward(self, feature: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
        batch, length, _ = feature.shape
        pair = torch.cat((self.feature_norm(feature), self.token_norm(token_embedding)), dim=-1)
        query = self.q_proj(pair).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(pair).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(pair).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, length, self.state_size)
        feature = feature + F.dropout(self.o_proj(attended), self.dropout, self.training)
        normalized = self.post_norm(feature)
        update = self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))
        return feature + F.dropout(update, self.dropout, self.training)


class ParallelEagleFlowDrafter(nn.Module):
    """One-pass block Flow Map with parallel continuous token refinement.

    The head never calls the frozen vocabulary projection internally.  Instead,
    each attention layer shifts its own continuous token endpoint one position
    right.  Thus all K-1 future positions are drafted in one head pass while
    still receiving an autoregressive-like advanced-token signal.
    """

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        state_size: int = 768,
        num_layers: int = 3,
        num_heads: int = 12,
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
        self.num_heads = num_heads
        self.position = nn.Parameter(torch.zeros(1, self.prediction_length, hidden_size))
        self.source_tokens = nn.Parameter(torch.zeros(1, self.prediction_length - 1, hidden_size))
        self.context_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList(
            ParallelFeatureTokenAttention(hidden_size, state_size, num_heads, dropout) for _ in range(num_layers)
        )
        self.token_norm = nn.LayerNorm(hidden_size)
        self.embedding_endpoint = nn.Linear(hidden_size, hidden_size, bias=False)
        self.source_norm = nn.LayerNorm(hidden_size)
        self.source_residual = nn.Linear(hidden_size, hidden_size, bias=False)
        self.flow_norm = nn.LayerNorm(hidden_size * 2)
        self.flow_time = nn.Sequential(nn.Linear(1, state_size), nn.SiLU(), nn.Linear(state_size, hidden_size))
        self.flow_hidden = nn.Sequential(nn.Linear(hidden_size * 2, state_size), nn.SiLU(), nn.Linear(state_size, hidden_size))
        nn.init.normal_(self.source_tokens, std=0.02)
        nn.init.zeros_(self.source_residual.weight)
        nn.init.zeros_(self.flow_hidden[-1].weight)
        nn.init.zeros_(self.flow_hidden[-1].bias)
        nn.init.zeros_(self.flow_time[-1].weight)
        nn.init.zeros_(self.flow_time[-1].bias)

    def rollout(
        self,
        context_hidden: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor | None = None,
        teacher_features: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        feedback_embedding_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        flow_targets: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del teacher_features, feedback_embedding_fn
        if context_hidden.shape != anchor_embeddings.shape:
            raise ValueError("context_hidden and anchor_embeddings must have identical shapes")
        if context_hidden.dim() != 3 or context_hidden.shape[-1] != self.hidden_size:
            raise ValueError("context_hidden must have shape [B, blocks, hidden_size]")
        expected = context_hidden.shape[:2] + (self.prediction_length, self.hidden_size)
        if flow_targets is not None and flow_targets.shape != expected:
            raise ValueError(f"Expected flow_targets shape {expected}, got {tuple(flow_targets.shape)}")
        if teacher_embeddings is not None and teacher_embeddings.shape != expected:
            raise ValueError(f"Expected teacher_embeddings shape {expected}, got {tuple(teacher_embeddings.shape)}")
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")
        batch_size, blocks, _ = context_hidden.shape
        if flow_time is None:
            flow_time = torch.zeros((batch_size, blocks, 1), device=context_hidden.device, dtype=context_hidden.dtype)
        if flow_time.shape != (batch_size, blocks, 1):
            raise ValueError("flow_time must have shape [B, blocks, 1]")
        flat = batch_size * blocks
        feature = self.context_norm(context_hidden).reshape(flat, 1, self.hidden_size)
        feature = feature.expand(-1, self.prediction_length, -1) + self.position
        anchor = anchor_embeddings.reshape(flat, 1, self.hidden_size)
        token = torch.cat((anchor, self.source_tokens.expand(flat, -1, -1)), dim=1)
        teacher_shift = None
        if teacher_embeddings is not None:
            targets = teacher_embeddings.reshape(flat, self.prediction_length, self.hidden_size)
            teacher_shift = torch.cat((anchor, targets[:, :-1, :]), dim=1)
        for layer in self.layers:
            feature = layer(feature, token)
            continuous = self.embedding_endpoint(self.token_norm(feature))
            predicted_shift = torch.cat((anchor, continuous[:, :-1, :]), dim=1)
            if teacher_shift is None or teacher_forcing_ratio <= 0.0:
                token = predicted_shift
            elif teacher_forcing_ratio >= 1.0:
                token = teacher_shift
            else:
                use_teacher = torch.rand((flat, 1, 1), device=feature.device) < teacher_forcing_ratio
                token = torch.where(use_teacher, teacher_shift, predicted_shift)
        source = feature + self.source_residual(self.source_norm(feature))
        source = source.reshape(batch_size, blocks, self.prediction_length, self.hidden_size)
        interpolant = source if flow_targets is None else (1.0 - flow_time.unsqueeze(2)) * source + flow_time.unsqueeze(2) * flow_targets
        flow_pair = torch.cat((source, interpolant), dim=-1)
        flow_update = self.flow_hidden(self.flow_norm(flow_pair))
        flow_update = flow_update + self.flow_time(flow_time.reshape(flat, 1)).reshape(batch_size, blocks, 1, self.hidden_size)
        hidden = source + flow_update
        return hidden, continuous.reshape(batch_size, blocks, self.prediction_length, self.hidden_size)
