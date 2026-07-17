from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch


def save_orthrus_checkpoint(model, tokenizer, output_dir: str | Path, upstream_dir: str | Path) -> None:
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
