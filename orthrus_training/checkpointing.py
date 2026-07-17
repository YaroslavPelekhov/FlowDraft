from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch


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
