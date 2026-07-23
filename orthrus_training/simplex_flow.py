"""A small categorical flow map on a frozen drafter's local candidate simplex.

The full Qwen vocabulary is too large to evolve repeatedly at inference.  This
module keeps the K candidates proposed by a frozen Orthrus pass and learns an
endpoint map on their probability simplex.  It therefore adds correlated
multi-step CFM refinement without an additional Qwen forward pass.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def local_simplex_path(target_index: torch.Tensor, time: torch.Tensor, candidates: int) -> torch.Tensor:
    """Return x_t=(1-t)u+t*delta_y on the local categorical simplex."""

    if target_index.shape != time.shape:
        raise ValueError("target_index and time must have the same shape")
    state = torch.full(
        (*target_index.shape, candidates),
        1.0 / candidates,
        device=target_index.device,
        dtype=time.dtype,
    )
    endpoint = torch.nn.functional.one_hot(target_index, candidates).to(dtype=time.dtype)
    return (1.0 - time.unsqueeze(-1)) * state + time.unsqueeze(-1) * endpoint


def simplex_flow_step(state: torch.Tensor, endpoint: torch.Tensor, source_time: torch.Tensor, target_time: torch.Tensor) -> torch.Tensor:
    """CFM endpoint transport: x_{s,t}=x_s+(t-s)/(1-s)*(pi-x_s)."""

    gamma = (target_time - source_time) / (1.0 - source_time).clamp_min(0.05)
    return state + gamma.unsqueeze(-1).to(dtype=state.dtype) * (endpoint - state)


class SimplexFlowRefiner(nn.Module):
    """Contextual partial denoiser pi_phi(s,t,x_s) over a top-k candidate set."""

    def __init__(
        self,
        block_size: int,
        candidate_count: int = 128,
        hidden_size: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if block_size < 2 or candidate_count < 2:
            raise ValueError("block_size must be >=2 and candidate_count must be >=2")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.block_size = block_size
        self.prediction_length = block_size - 1
        self.candidate_count = candidate_count
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Base logits give each local simplex a context-conditioned coordinate
        # system; x_s is the actual categorical state transported by the map.
        self.input_norm = nn.LayerNorm(candidate_count * 2 + 4)
        self.input_proj = nn.Linear(candidate_count * 2 + 4, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, self.prediction_length, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.delta = nn.Linear(hidden_size, candidate_count)
        # Initial endpoint equals the frozen Orthrus local distribution.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self,
        base_logits: torch.Tensor,
        state: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
    ) -> torch.Tensor:
        """Return endpoint probabilities for [B, blocks, K-1, candidates]."""

        expected = base_logits.shape
        if len(expected) != 4 or expected[-1] != self.candidate_count:
            raise ValueError(f"Expected base_logits [..., {self.candidate_count}], got {tuple(expected)}")
        if state.shape != expected:
            raise ValueError("state must match base_logits")
        if expected[2] != self.prediction_length:
            raise ValueError("candidate positions do not match block size")
        if source_time.shape != expected[:-1] or target_time.shape != expected[:-1]:
            raise ValueError("time tensors must have shape base_logits.shape[:-1]")

        normalized_logits = base_logits.float() - base_logits.float().mean(dim=-1, keepdim=True)
        normalized_logits = normalized_logits / normalized_logits.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        phase = torch.stack(
            [
                torch.sin(math.pi * source_time),
                torch.cos(math.pi * source_time),
                torch.sin(math.pi * target_time),
                torch.cos(math.pi * target_time),
            ],
            dim=-1,
        ).to(dtype=base_logits.dtype)
        features = torch.cat([normalized_logits.to(dtype=base_logits.dtype), state, phase], dim=-1)
        batch, blocks, positions, _ = features.shape
        features = features.reshape(batch * blocks, positions, -1)
        hidden = self.input_proj(self.input_norm(features)) + self.position
        hidden = self.blocks(hidden)
        correction = self.delta(self.output_norm(hidden)).reshape_as(base_logits)
        return torch.softmax(base_logits.float() + correction.float(), dim=-1).to(dtype=base_logits.dtype)


@torch.no_grad()
def make_token_codebook(
    embedding_weight: torch.Tensor,
    code_dim: int,
    seed: int,
) -> torch.Tensor:
    """Create a deterministic low-dimensional semantic code for every token.

    Random projection preserves enough of the frozen embedding geometry for
    fast maximum-inner-product retrieval, while avoiding a second full-size LM
    head. The codebook is derived solely from the frozen parent checkpoint and
    is therefore reproducible from ``seed`` rather than trainable state.
    """

    if embedding_weight.ndim != 2 or code_dim < 4:
        raise ValueError("embedding_weight must be [vocab, hidden] and code_dim >= 4")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(
        (embedding_weight.shape[1], code_dim), generator=generator, dtype=torch.float32
    ) / math.sqrt(code_dim)
    projection = projection.to(device=embedding_weight.device)
    codes = embedding_weight.detach().float() @ projection
    return torch.nn.functional.normalize(codes, dim=-1)


class DynamicSupportSimplexFlowRefiner(nn.Module):
    """Identity-aware CFM with a learned dynamic vocabulary-support query.

    Unlike a rank-only local simplex, this head observes the frozen diffusion
    feature and a semantic code for each candidate token. A lightweight query
    retrieves context-specific extra vertices before endpoint transport; the
    CFM itself still evolves only a small categorical simplex.
    """

    def __init__(
        self,
        block_size: int,
        candidate_count: int,
        draft_hidden_size: int,
        hidden_size: int = 512,
        token_code_dim: int = 32,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if block_size < 2 or candidate_count < 2:
            raise ValueError("block_size and candidate_count must be >= 2")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.block_size = block_size
        self.prediction_length = block_size - 1
        self.candidate_count = candidate_count
        self.draft_hidden_size = draft_hidden_size
        self.hidden_size = hidden_size
        self.token_code_dim = token_code_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.context_norm = nn.LayerNorm(draft_hidden_size)
        self.context_proj = nn.Linear(draft_hidden_size, hidden_size)
        self.retrieval = nn.Sequential(
            nn.LayerNorm(draft_hidden_size),
            nn.Linear(draft_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, token_code_dim),
        )
        self.input_norm = nn.LayerNorm(candidate_count * 2 + token_code_dim + 4)
        self.input_proj = nn.Linear(candidate_count * 2 + token_code_dim + 4, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, self.prediction_length, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.rank_delta = nn.Linear(hidden_size, candidate_count)
        self.identity_query = nn.Linear(hidden_size, token_code_dim)
        nn.init.zeros_(self.rank_delta.weight)
        nn.init.zeros_(self.rank_delta.bias)
        nn.init.zeros_(self.identity_query.weight)
        nn.init.zeros_(self.identity_query.bias)

    def retrieval_query(self, draft_hidden: torch.Tensor) -> torch.Tensor:
        if draft_hidden.shape[-1] != self.draft_hidden_size:
            raise ValueError("draft_hidden has the wrong feature dimension")
        return torch.nn.functional.normalize(self.retrieval(draft_hidden.float()), dim=-1)

    def retrieval_scores(self, draft_hidden: torch.Tensor, token_codebook: torch.Tensor) -> torch.Tensor:
        if token_codebook.shape[-1] != self.token_code_dim:
            raise ValueError("token_codebook has the wrong feature dimension")
        query = self.retrieval_query(draft_hidden)
        flat = query.reshape(-1, self.token_code_dim)
        scores = flat @ token_codebook.float().transpose(0, 1)
        return scores.reshape(*query.shape[:-1], token_codebook.shape[0])

    def forward(
        self,
        base_logits: torch.Tensor,
        state: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        candidate_ids: torch.Tensor,
        draft_hidden: torch.Tensor,
        token_codebook: torch.Tensor,
    ) -> torch.Tensor:
        expected = base_logits.shape
        if len(expected) != 4 or expected[-1] != self.candidate_count:
            raise ValueError(f"Expected base_logits [..., {self.candidate_count}], got {tuple(expected)}")
        if state.shape != expected or candidate_ids.shape != expected:
            raise ValueError("state and candidate_ids must match base_logits")
        if draft_hidden.shape[:-1] != expected[:-1] or expected[2] != self.prediction_length:
            raise ValueError("draft_hidden positions do not match the CFM simplex")
        if source_time.shape != expected[:-1] or target_time.shape != expected[:-1]:
            raise ValueError("time tensors must match base_logits without candidates")
        if int(candidate_ids.min()) < 0 or int(candidate_ids.max()) >= token_codebook.shape[0]:
            raise ValueError("candidate_ids are outside the token codebook")

        candidate_codes = torch.nn.functional.embedding(candidate_ids, token_codebook).to(dtype=base_logits.dtype)
        base_distribution = torch.softmax(base_logits.float(), dim=-1).to(dtype=base_logits.dtype)
        candidate_summary = (base_distribution.unsqueeze(-1) * candidate_codes).sum(dim=-2)
        normalized_logits = base_logits.float() - base_logits.float().mean(dim=-1, keepdim=True)
        normalized_logits = normalized_logits / normalized_logits.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        phase = torch.stack(
            [
                torch.sin(math.pi * source_time),
                torch.cos(math.pi * source_time),
                torch.sin(math.pi * target_time),
                torch.cos(math.pi * target_time),
            ],
            dim=-1,
        ).to(dtype=base_logits.dtype)
        features = torch.cat(
            [normalized_logits.to(dtype=base_logits.dtype), state, candidate_summary, phase], dim=-1
        )
        batch, blocks, positions, _ = features.shape
        features = features.reshape(batch * blocks, positions, -1)
        context = self.context_proj(self.context_norm(draft_hidden)).reshape(batch * blocks, positions, -1)
        hidden = self.input_proj(self.input_norm(features)) + context + self.position
        hidden = self.blocks(hidden)
        hidden = self.output_norm(hidden).reshape(batch, blocks, positions, self.hidden_size)
        rank_correction = self.rank_delta(hidden)
        identity_query = self.identity_query(hidden).to(dtype=candidate_codes.dtype)
        identity_correction = (identity_query.unsqueeze(-2) * candidate_codes).sum(dim=-1)
        logits = base_logits.float() + rank_correction.float() + identity_correction.float()
        return torch.softmax(logits, dim=-1).to(dtype=base_logits.dtype)
