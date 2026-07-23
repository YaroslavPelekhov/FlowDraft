from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DIFFUSION_TRAINABLE_MARKERS = (
    "q_proj_diff",
    "k_proj_diff",
    "v_proj_diff",
)


class FlowDraftStateAdapter(nn.Module):
    """Map continuous categorical states into Qwen's embedding space.

    The frozen backbone was pretrained on discrete token embeddings.  CFM states
    are convex combinations of categorical vertices, so they need a small,
    trainable and time-conditioned adapter before entering the diffusion path.
    The residual branch starts at zero, preserving the original Orthrus
    initialization while allowing the adapter to learn the continuous geometry.
    """

    def __init__(self, hidden_size: int, bottleneck_size: int = 256, eps: float = 1e-6):
        super().__init__()
        if hidden_size <= 0 or bottleneck_size <= 0:
            raise ValueError("hidden_size and bottleneck_size must be positive")
        self.hidden_size = hidden_size
        self.eps = eps
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.film = nn.Linear(hidden_size, bottleneck_size * 2)
        self.up = nn.Linear(bottleneck_size, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs_embeds: torch.Tensor, time_embed: torch.Tensor, block_size: int) -> torch.Tensor:
        batch_size, flat_length, hidden_size = inputs_embeds.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {hidden_size}")
        if flat_length % block_size:
            raise ValueError("FlowDraft state length must be divisible by block_size")

        num_blocks = flat_length // block_size
        if time_embed.shape != (batch_size, num_blocks, hidden_size):
            raise ValueError(
                f"Expected time embedding {(batch_size, num_blocks, hidden_size)}, got {tuple(time_embed.shape)}"
            )

        blocks = inputs_embeds.reshape(batch_size, num_blocks, block_size, hidden_size)
        rms = blocks.float().square().mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        normalized = blocks / rms.to(blocks.dtype)
        hidden = self.down(normalized)
        film = self.film(time_embed).unsqueeze(2)
        scale, shift = film.chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(hidden * (1.0 + scale) + shift)
        return (blocks + self.up(hidden)).reshape_as(inputs_embeds)


def attach_flowdraft_state_adapter(model: torch.nn.Module, bottleneck_size: int = 256) -> None:
    """Attach the adapter once so it participates in adapter checkpoints."""

    if hasattr(model, "flowdraft_state_adapter"):
        return
    hidden_size = int(model.config.hidden_size)
    model.flowdraft_state_adapter = FlowDraftStateAdapter(hidden_size, bottleneck_size)


def set_flowdraft_state_adapter_trainable(model: torch.nn.Module, enabled: bool) -> None:
    """Keep legacy Orthrus adapters binary-compatible with their old checkpoints."""

    if not hasattr(model, "flowdraft_state_adapter"):
        raise ValueError("Model has no FlowDraft state adapter")
    for parameter in model.flowdraft_state_adapter.parameters():
        parameter.requires_grad = enabled


def _generate_causal_verifier_mask(
    B: int,
    H: int,
    diffusion_length: int,
    ar_len: int,
    block_size: int,
    causal_limit: torch.Tensor,
    sparse_block_size: int = 128,
):
    """Block-parallel AR mask used to obtain verifier logits during training."""

    upstream = sys.modules["upstream_orthrus_model"]

    def verifier_mask_fn(b, h, q_idx, kv_idx):
        is_kv_ar = kv_idx < ar_len
        valid_ar = is_kv_ar & (kv_idx <= causal_limit[b, q_idx])

        draft_kv_idx = kv_idx - ar_len
        q_block_id = q_idx // block_size
        kv_block_id = draft_kv_idx // block_size
        q_offset = q_idx % block_size
        kv_offset = draft_kv_idx % block_size
        valid_draft = (
            (~is_kv_ar)
            & (q_block_id == kv_block_id)
            & (kv_offset <= q_offset)
        )
        return valid_ar | valid_draft

    return upstream.create_block_mask(
        verifier_mask_fn,
        B=B,
        H=H,
        Q_LEN=diffusion_length,
        KV_LEN=ar_len + diffusion_length,
        BLOCK_SIZE=sparse_block_size,
    )


@contextmanager
def parallel_ar_verifier_mode(model: torch.nn.Module):
    """Run the dual-pass kernel with frozen AR projections and a causal block mask.

    The official Orthrus module only exposes a bidirectional diffusion pass for
    training. This scoped adapter reuses the same shared KV cache to evaluate
    many proposed blocks in parallel with the frozen Qwen verifier. It restores
    every module reference before gradients are computed.
    """

    upstream = sys.modules.get("upstream_orthrus_model")
    if upstream is None:
        raise RuntimeError("The upstream Orthrus module has not been imported")

    original_mask_builder = upstream.generate_dual_pass_mask
    swaps: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []
    projection_pairs = (
        ("q_proj_diff", "q_proj"),
        ("k_proj_diff", "k_proj"),
        ("v_proj_diff", "v_proj"),
        ("o_proj_diff", "o_proj"),
        ("q_norm_diff", "q_norm"),
        ("k_norm_diff", "k_norm"),
    )
    try:
        upstream.generate_dual_pass_mask = _generate_causal_verifier_mask
        for layer in model.model.layers:
            attention = layer.self_attn
            for diffusion_name, ar_name in projection_pairs:
                original = getattr(attention, diffusion_name)
                swaps.append((attention, diffusion_name, original))
                setattr(attention, diffusion_name, getattr(attention, ar_name))
        yield
    finally:
        for module, name, original in reversed(swaps):
            setattr(module, name, original)
        upstream.generate_dual_pass_mask = original_mask_builder


@torch.no_grad()
def parallel_verifier_logits(
    model: torch.nn.Module,
    proposed_blocks: torch.Tensor,
    position_ids: torch.Tensor,
    causal_limit: torch.Tensor,
    past_key_values,
    ar_seq_len: int,
) -> torch.Tensor:
    """Evaluate K-token proposals with frozen AR QKV in one block-parallel pass."""

    batch_size, num_blocks, block_size = proposed_blocks.shape
    flat_blocks = proposed_blocks.reshape(batch_size, num_blocks * block_size)
    with parallel_ar_verifier_mode(model):
        outputs = model(
            input_ids=flat_blocks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            is_diffusion_pass=True,
            causal_limit=causal_limit,
            ar_seq_len=ar_seq_len,
        )
    logits = outputs.logits.reshape(batch_size, num_blocks, block_size, -1)
    return logits[:, :, : block_size - 1, :].reshape(
        batch_size, num_blocks * (block_size - 1), -1
    )


@contextmanager
def flowtree_ar_verifier_mode(model: torch.nn.Module, visibility: torch.Tensor):
    """Expose a parent-aware AR verifier mask for one packed FlowTree.

    ``visibility[q, k]`` is true precisely when tree token ``k`` lies on the
    inclusive root-to-query path.  This makes every node's logits identical to
    an AR rollout on its own branch, subject only to floating-point kernels.
    """

    upstream = sys.modules.get("upstream_orthrus_model")
    if upstream is None:
        raise RuntimeError("The upstream Orthrus module has not been imported")
    if visibility.dim() != 2 or visibility.shape[0] != visibility.shape[1]:
        raise ValueError("visibility must have shape [tree_nodes, tree_nodes]")

    original_mask_builder = upstream.generate_dual_pass_mask
    swaps: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []

    def generate_tree_mask(B, H, diffusion_length, ar_len, block_size, causal_limit, sparse_block_size=128):
        if diffusion_length != visibility.shape[0]:
            raise ValueError("FlowTree visibility does not match diffusion length")

        def tree_mask_fn(b, h, q_idx, kv_idx):
            is_kv_ar = kv_idx < ar_len
            safe_query_idx = q_idx.clamp(0, visibility.shape[0] - 1)
            valid_query = q_idx < visibility.shape[0]
            valid_ar = is_kv_ar & valid_query & (kv_idx <= causal_limit[b, safe_query_idx])
            draft_kv_idx = kv_idx - ar_len
            # FlexAttention evaluates both sides of ``&``. Clamp prompt-KV
            # indices before the table lookup, then gate them out below.
            safe_tree_idx = draft_kv_idx.clamp(0, visibility.shape[1] - 1)
            valid_tree = (~is_kv_ar) & valid_query & visibility[safe_query_idx, safe_tree_idx]
            return valid_ar | valid_tree

        return upstream.create_block_mask(
            tree_mask_fn,
            B=B,
            H=H,
            Q_LEN=diffusion_length,
            KV_LEN=ar_len + diffusion_length,
            BLOCK_SIZE=sparse_block_size,
        )

    projection_pairs = (
        ("q_proj_diff", "q_proj"),
        ("k_proj_diff", "k_proj"),
        ("v_proj_diff", "v_proj"),
        ("o_proj_diff", "o_proj"),
        ("q_norm_diff", "q_norm"),
        ("k_norm_diff", "k_norm"),
    )
    try:
        upstream.generate_dual_pass_mask = generate_tree_mask
        for layer in model.model.layers:
            attention = layer.self_attn
            for diffusion_name, ar_name in projection_pairs:
                original = getattr(attention, diffusion_name)
                swaps.append((attention, diffusion_name, original))
                setattr(attention, diffusion_name, getattr(attention, ar_name))
        yield
    finally:
        for module, name, original in reversed(swaps):
            setattr(module, name, original)
        upstream.generate_dual_pass_mask = original_mask_builder


@torch.no_grad()
def flowtree_verifier_logits(
    model: torch.nn.Module,
    tree_token_ids: torch.Tensor,
    tree_position_ids: torch.Tensor,
    visibility: torch.Tensor,
    causal_limit: torch.Tensor,
    past_key_values,
    ar_seq_len: int,
) -> torch.Tensor:
    """Verify every FlowTree node under its own shared-prefix AR context."""

    if tree_token_ids.dim() != 2 or tree_token_ids.shape[0] != 1:
        raise ValueError("FlowTree verifier currently supports one request at a time")
    was_training = model.training
    # Orthrus creates its FlexAttention mask only in train mode. Its Qwen path
    # has no dropout, and this call stays under no_grad.
    model.train()
    try:
        with flowtree_ar_verifier_mode(model, visibility):
            outputs = model(
                input_ids=tree_token_ids,
                position_ids=tree_position_ids,
                past_key_values=past_key_values,
                use_cache=False,
                is_diffusion_pass=True,
                causal_limit=causal_limit,
                ar_seq_len=ar_seq_len,
            )
    finally:
        if not was_training:
            model.eval()
    return outputs.logits


def import_upstream_orthrus(upstream_dir: str | Path):
    upstream_dir = Path(upstream_dir).resolve()
    model_file = upstream_dir / "src" / "model.py"
    if not model_file.exists():
        raise FileNotFoundError(f"Cannot find upstream Orthrus model at {model_file}")

    module_name = "upstream_orthrus_model"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    patch_flex_attention_compat(module)
    return module


def patch_flex_attention_compat(module) -> None:
    """Make upstream Orthrus FlexAttention usable on DataSphere's torch 2.5.x.

    The official repo targets a newer FlexAttention API and passes
    kernel_options={"BACKEND": "FLASH"}. In torch 2.5.x this can be emitted into
    Triton code as a bare FLASH symbol and fail at compile time. The default
    FlexAttention backend is sufficient for the reconstructed training loop.
    """

    def fused_flex_attention_compat(q, k, v, mask=None):
        return module._compiled_flex_attention(
            q,
            k,
            v,
            block_mask=mask,
            enable_gqa=True,
        )

    module.fused_flex_attention = fused_flex_attention_compat


def build_orthrus_from_qwen(
    base_model_name_or_path: str,
    upstream_dir: str | Path,
    block_size: int,
    mask_token_id: int,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    device_map: str | dict | None = None,
    flowdraft_adapter_bottleneck: int = 256,
):
    """Create an OrthrusLM initialized from a Qwen3 CausalLM checkpoint."""

    upstream = import_upstream_orthrus(upstream_dir)
    base_config = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)
    base_config.block_size = block_size
    base_config.mask_token_id = mask_token_id
    base_config.architectures = ["OrthrusLM"]
    base_config.auto_map = {"AutoModelForCausalLM": "modeling_orthrus.OrthrusLM"}
    base_config._attn_implementation = attn_implementation

    orthrus = upstream.OrthrusLM(base_config)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        device_map=device_map,
    )

    missing, unexpected = orthrus.load_state_dict(base.state_dict(), strict=False)
    initialize_diffusion_from_ar(orthrus)
    attach_flowdraft_state_adapter(orthrus, flowdraft_adapter_bottleneck)
    freeze_non_diffusion_parameters(orthrus)

    del base
    return orthrus, {"missing": missing, "unexpected": unexpected}


def load_flowdraft_adapter(
    checkpoint_dir: str | Path,
    upstream_dir: str | Path,
    dtype: torch.dtype,
    attn_implementation: str,
):
    """Reconstruct a frozen Orthrus model and load a trainable-only FlowDraft checkpoint."""

    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "adapter_config.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("format") != "flowdraft_trainable_v1":
        raise ValueError(f"Unsupported adapter format: {metadata.get('format')}")

    model, load_info = build_orthrus_from_qwen(
        base_model_name_or_path=metadata["base_model"],
        upstream_dir=upstream_dir,
        block_size=int(metadata["block_size"]),
        mask_token_id=int(metadata["mask_token_id"]),
        dtype=dtype,
        attn_implementation=attn_implementation,
        flowdraft_adapter_bottleneck=int(metadata.get("flowdraft_adapter_bottleneck", 256)),
    )
    adapter_state = load_file(checkpoint_dir / "adapter_model.safetensors")
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    unexpected = [name for name in unexpected if name not in load_info.get("unexpected", [])]
    if unexpected:
        raise ValueError(f"Unexpected adapter weights: {unexpected}")
    expected_names = set(metadata.get("trainable_parameter_names", []))
    if expected_names and expected_names != set(adapter_state):
        raise ValueError("Adapter weight names do not match adapter_config.json")

    for key in (
        "flowdraft_cfm",
        "flowdraft_objective",
        "flowdraft_time_conditioning_scale",
        "flowdraft_endpoint_topk",
        "flowdraft_endpoint_transport",
        "flowdraft_state_adapter",
        "flowdraft_adapter_bottleneck",
        "flowdraft_one_jump_fraction",
        "flowdraft_source_prior",
        "flowdraft_source_seed",
    ):
        if key in metadata:
            setattr(model.config, key, metadata[key])
    return model, metadata, {"base": load_info, "adapter_missing": missing}


def load_trainable_initialization(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    allow_missing_trainables: bool = False,
) -> list[str]:
    """Load only currently trainable tensors from a full or adapter checkpoint."""

    checkpoint_dir = Path(checkpoint_dir)
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    full_path = checkpoint_dir / "model.safetensors"
    source_path = adapter_path if adapter_path.exists() else full_path
    if not source_path.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors or model.safetensors in {checkpoint_dir}")

    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    selected = {}
    with safe_open(source_path, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        missing = sorted(trainable_names - available)
        if missing and not allow_missing_trainables:
            raise ValueError(f"Initialization checkpoint is missing trainable weights: {missing[:5]}")
        for name in sorted(trainable_names & available):
            selected[name] = handle.get_tensor(name)
    _, unexpected = model.load_state_dict(selected, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected initialization weights: {unexpected}")
    return sorted(selected)


def initialize_diffusion_from_ar(model: torch.nn.Module) -> None:
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.q_proj_diff.load_state_dict(attn.q_proj.state_dict())
        attn.k_proj_diff.load_state_dict(attn.k_proj.state_dict())
        attn.v_proj_diff.load_state_dict(attn.v_proj.state_dict())
        attn.o_proj_diff.load_state_dict(attn.o_proj.state_dict())
        attn.q_norm_diff.load_state_dict(attn.q_norm.state_dict())
        attn.k_norm_diff.load_state_dict(attn.k_norm.state_dict())


def freeze_non_diffusion_parameters(
    model: torch.nn.Module,
    trainable_markers: Iterable[str] = DIFFUSION_TRAINABLE_MARKERS,
) -> None:
    markers = tuple(trainable_markers)
    for name, param in model.named_parameters():
        param.requires_grad = any(marker in name for marker in markers)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def dtype_from_string(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")
