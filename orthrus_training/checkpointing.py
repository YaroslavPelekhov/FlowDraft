from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def _write_orthrus_checkpoint(model, tokenizer, output_dir: Path, upstream_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    config_path = output_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["architectures"] = ["OrthrusLM"]
    config["auto_map"] = {"AutoModelForCausalLM": "modeling_orthrus.OrthrusLM"}
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")

    src_model = Path(upstream_dir) / "src" / "model.py"
    shutil.copy2(src_model, output_dir / "modeling_orthrus.py")


def _replace_directory(src_dir: Path, dst_dir: Path) -> None:
    backup_dir = dst_dir.with_name(f".{dst_dir.name}.previous")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    if dst_dir.exists():
        dst_dir.rename(backup_dir)

    try:
        src_dir.rename(dst_dir)
    except Exception:
        if backup_dir.exists() and not dst_dir.exists():
            backup_dir.rename(dst_dir)
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def save_orthrus_checkpoint(model, tokenizer, output_dir: str | Path, upstream_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    try:
        _write_orthrus_checkpoint(model, tokenizer, tmp_dir, upstream_dir)
        _replace_directory(tmp_dir, output_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def _write_trainable_checkpoint(
    model,
    output_dir: Path,
    base_model: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable_state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable_state:
        raise ValueError("Model has no trainable parameters to checkpoint")
    save_file(trainable_state, output_dir / "adapter_model.safetensors")
    metadata = {
        "format": "flowdraft_trainable_v1",
        "base_model": base_model,
        "block_size": int(model.config.block_size),
        "mask_token_id": int(model.config.mask_token_id),
        "flowdraft_cfm": bool(getattr(model.config, "flowdraft_cfm", False)),
        "flowdraft_objective": getattr(model.config, "flowdraft_objective", "legacy"),
        "flowdraft_time_conditioning_scale": float(
            getattr(model.config, "flowdraft_time_conditioning_scale", 0.0)
        ),
        "flowdraft_endpoint_topk": int(getattr(model.config, "flowdraft_endpoint_topk", 32)),
        "trainable_parameter_names": sorted(trainable_state),
    }
    with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_trainable_checkpoint(model, output_dir: str | Path, base_model: str) -> None:
    """Atomically save only FlowDraft's trainable projections and reconstruction metadata."""

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    try:
        _write_trainable_checkpoint(model, tmp_dir, base_model)
        _replace_directory(tmp_dir, output_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def save_training_state(
    output_dir: str | Path,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    epoch: int,
) -> None:
    state_dir = Path(output_dir) / "trainer_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "epoch": epoch,
        },
        state_dir / "state.pt",
    )
