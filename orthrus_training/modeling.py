from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DIFFUSION_TRAINABLE_MARKERS = (
    "q_proj_diff",
    "k_proj_diff",
    "v_proj_diff",
)


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
    ):
        if key in metadata:
            setattr(model.config, key, metadata[key])
    return model, metadata, {"base": load_info, "adapter_missing": missing}


def load_trainable_initialization(model: torch.nn.Module, checkpoint_dir: str | Path) -> list[str]:
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
        if missing:
            raise ValueError(f"Initialization checkpoint is missing trainable weights: {missing[:5]}")
        for name in sorted(trainable_names):
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
