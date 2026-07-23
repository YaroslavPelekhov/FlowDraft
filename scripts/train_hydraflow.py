#!/usr/bin/env python
"""Train a self-conditioned latent HydraFlow drafter on a frozen Qwen verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orthrus_training.data import PackedTokenDataset, assert_disjoint_packed_manifests, sample_anchor_positions
from orthrus_training.hydraflow import HydraFlowDrafter
from orthrus_training.losses import gather_logits, prefix_acceptance_metrics, prefix_survival_cross_entropy
from orthrus_training.modeling import dtype_from_string, load_flowdraft_adapter


def parse_args():
    parser = argparse.ArgumentParser(description="Train a sequentially-dependent latent HydraFlow drafter.")
    parser.add_argument("--config")
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-anchor-blocks", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-loss-weight", type=float, default=0.25)
    parser.add_argument("--embedding-loss-weight", type=float, default=0.25)
    parser.add_argument("--prefix-loss-weight", type=float, default=1.0)
    parser.add_argument("--prefix-weight-decay", type=float, default=0.9)
    parser.add_argument("--state-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--teacher-forcing-start", type=float, default=1.0)
    parser.add_argument("--teacher-forcing-end", type=float, default=0.1)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-anchor-blocks", type=int, default=32)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    cli = {item.dest for item in parser._actions if item.dest != "help" and any(flag in sys.argv[1:] for flag in item.option_strings)}
    return args, cli


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_config(args, values: dict, cli: set[str]):
    for key, value in values.items():
        name = key.replace("-", "_")
        if hasattr(args, name) and name not in cli:
            setattr(args, name, value)
    return args


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(head: HydraFlowDrafter, directory: Path, config: dict) -> None:
    temporary, previous = directory.with_name(f".{directory.name}.tmp"), directory.with_name(f".{directory.name}.previous")
    if temporary.exists():
        shutil.rmtree(temporary)
    if previous.exists():
        shutil.rmtree(previous)
    temporary.mkdir(parents=True)
    save_file({name: value.detach().cpu().contiguous() for name, value in head.state_dict().items()}, temporary / "hydraflow_head.safetensors")
    write_json(temporary / "hydraflow_config.json", config)
    if directory.exists():
        directory.rename(previous)
    try:
        temporary.rename(directory)
    except Exception:
        if previous.exists() and not directory.exists():
            previous.rename(directory)
        raise
    if previous.exists():
        shutil.rmtree(previous)


@torch.no_grad()
def collect_teacher(model, input_ids: torch.Tensor, num_blocks: int) -> dict:
    block_size = int(model.config.block_size)
    anchors = sample_anchor_positions(input_ids.shape[0], input_ids.shape[1], block_size, num_blocks, input_ids.device).clamp_min(1)
    outputs = model(input_ids=input_ids, use_cache=False, output_hidden_states=True, is_diffusion_pass=False)
    final_hidden = outputs.hidden_states[-1]
    offsets = torch.arange(block_size - 1, device=input_ids.device)
    prediction_positions = anchors.unsqueeze(-1) + offsets
    context_positions = anchors - 1
    context_hidden = torch.gather(final_hidden, 1, context_positions.unsqueeze(-1).expand(-1, -1, final_hidden.shape[-1]))
    target_hidden = torch.gather(
        final_hidden,
        1,
        prediction_positions.reshape(input_ids.shape[0], -1).unsqueeze(-1).expand(-1, -1, final_hidden.shape[-1]),
    ).reshape(input_ids.shape[0], num_blocks, block_size - 1, final_hidden.shape[-1])
    anchor_tokens = torch.gather(input_ids, 1, anchors)
    teacher_logits = gather_logits(outputs.logits, prediction_positions.reshape(input_ids.shape[0], -1))
    target_tokens = teacher_logits.argmax(dim=-1).reshape(input_ids.shape[0], num_blocks, block_size - 1)
    return {
        "context_hidden": context_hidden,
        "anchor_embeddings": model.model.embed_tokens(anchor_tokens).detach(),
        "target_hidden": target_hidden,
        "target_embeddings": model.model.embed_tokens(target_tokens).detach(),
        "target_tokens": target_tokens.reshape(input_ids.shape[0], -1),
    }


def teacher_forcing_ratio(args, step: int) -> float:
    progress = (step - 1) / max(args.max_steps - 1, 1)
    return args.teacher_forcing_start + progress * (args.teacher_forcing_end - args.teacher_forcing_start)


def forward_head(model, head, teacher: dict, ratio: float):
    hidden, embeddings = head.rollout(
        teacher["context_hidden"], teacher["anchor_embeddings"], teacher["target_embeddings"], ratio
    )
    logits = model.lm_head(hidden.reshape(-1, hidden.shape[-1])).reshape(hidden.shape[0], -1, int(model.config.vocab_size))
    return hidden, embeddings, logits


@torch.no_grad()
def evaluate(model, head, dataloader, args) -> dict:
    head.eval()
    hidden_losses, embedding_losses, prefixes, first_tokens = [], [], [], []
    for index, raw in enumerate(dataloader):
        if index >= args.eval_batches:
            break
        teacher = collect_teacher(model, raw.to(device="cuda", non_blocking=True), args.eval_anchor_blocks)
        hidden, embeddings, logits = forward_head(model, head, teacher, ratio=0.0)
        hidden_rms = teacher["target_hidden"].float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        embedding_rms = teacher["target_embeddings"].float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        hidden_losses.append(float(((hidden.float() - teacher["target_hidden"].float()) / hidden_rms).square().mean().cpu()))
        embedding_losses.append(float(((embeddings.float() - teacher["target_embeddings"].float()) / embedding_rms).square().mean().cpu()))
        metrics = prefix_acceptance_metrics(logits, teacher["target_tokens"], int(model.config.block_size))
        prefixes.append(float(metrics["greedy_prefix_acceptance"].cpu()))
        first_tokens.append(float(metrics["first_token_acc"].cpu()))
    head.train()
    return {
        "eval_hidden_mse": sum(hidden_losses) / len(hidden_losses),
        "eval_embedding_mse": sum(embedding_losses) / len(embedding_losses),
        "eval_greedy_prefix_acceptance": sum(prefixes) / len(prefixes),
        "eval_first_token_acc": sum(first_tokens) / len(first_tokens),
        "eval_batches": len(prefixes),
    }


def main() -> None:
    parsed, cli = parse_args()
    args = merge_config(parsed, load_config(parsed.config), cli)
    if not torch.cuda.is_available():
        raise RuntimeError("HydraFlow training requires CUDA")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and {item.name for item in output_dir.iterdir()} - {"run.log"}:
        raise FileExistsError(f"Refusing to overwrite existing experiment: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_unique, eval_unique = assert_disjoint_packed_manifests(args.train_manifest, args.eval_manifest)
    dtype = dtype_from_string(args.dtype)
    model, parent_metadata, _ = load_flowdraft_adapter(args.init_checkpoint, args.upstream_dir, dtype, args.attn_implementation)
    model.to(device="cuda", dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = HydraFlowDrafter(int(model.config.hidden_size), int(model.config.block_size), args.state_size, args.num_layers, args.dropout).to(device="cuda", dtype=dtype).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(args.max_steps * args.warmup_ratio)), args.max_steps)
    train_loader = DataLoader(PackedTokenDataset(args.train_manifest), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(PackedTokenDataset(args.eval_manifest), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    checkpoint_config = {
        "format": "hydraflow_latent_drafter_v1", "method": "self_conditioned_latent_feature_flow",
        "objective": "prefix_survival_plus_hidden_and_embedding_endpoint_regression",
        "base_flowdraft_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "base_flowdraft_adapter_sha256": sha256_file(Path(args.init_checkpoint) / "adapter_model.safetensors"),
        "block_size": head.block_size, "hidden_size": head.hidden_size, "state_size": head.state_size,
        "num_layers": head.num_layers,
        "inference": "continuous latent rollout, one batched lm_head, one ordinary frozen AR verifier pass",
    }
    run_config = vars(args) | {"method": checkpoint_config["method"], "parent_adapter_metadata": parent_metadata, "train_unique_sequences": train_unique, "eval_unique_sequences": eval_unique, "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "run_manifest.json", {"status": "running", "run_config": run_config, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
    print(f"hydraflow trainable={sum(item.numel() for item in head.parameters()):,} train_unique={train_unique} eval_unique={eval_unique}", flush=True)
    best_metric, best_step, stale, step = float("-inf"), None, 0, 0
    metrics_file = (output_dir / "train_metrics.jsonl").open("w", encoding="utf-8")
    progress = tqdm(total=args.max_steps, desc="hydraflow optimizer steps")
    try:
        for raw in train_loader:
            step += 1
            ratio = teacher_forcing_ratio(args, step)
            teacher = collect_teacher(model, raw.to(device="cuda", non_blocking=True), args.num_anchor_blocks)
            hidden, embeddings, logits = forward_head(model, head, teacher, ratio)
            hidden_rms = teacher["target_hidden"].float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            embedding_rms = teacher["target_embeddings"].float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            hidden_loss = ((hidden.float() - teacher["target_hidden"].float()) / hidden_rms).square().mean()
            embedding_loss = ((embeddings.float() - teacher["target_embeddings"].float()) / embedding_rms).square().mean()
            prefix_loss = prefix_survival_cross_entropy(logits, teacher["target_tokens"], int(model.config.block_size), args.prefix_weight_decay)
            loss = args.hidden_loss_weight * hidden_loss + args.embedding_loss_weight * embedding_loss + args.prefix_loss_weight * prefix_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); progress.update(1)
            metrics = prefix_acceptance_metrics(logits.detach(), teacher["target_tokens"], int(model.config.block_size))
            record = {"step": step, "loss": float(loss.detach().cpu()), "hidden_mse": float(hidden_loss.detach().cpu()), "embedding_mse": float(embedding_loss.detach().cpu()), "prefix_loss": float(prefix_loss.detach().cpu()), "greedy_prefix": float(metrics["greedy_prefix_acceptance"].cpu()), "first_token_acc": float(metrics["first_token_acc"].cpu()), "teacher_forcing_ratio": ratio, "lr": scheduler.get_last_lr()[0], "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
            metrics_file.write(json.dumps(record) + "\n"); metrics_file.flush()
            if step % args.log_every == 0:
                print("TRAIN " + json.dumps(record), flush=True)
            if step % args.eval_every == 0 or step == args.max_steps:
                evaluation = evaluate(model, head, eval_loader, args) | {"step": step}
                metrics_file.write(json.dumps(evaluation) + "\n"); metrics_file.flush(); print("EVAL " + json.dumps(evaluation), flush=True)
                metric = evaluation["eval_greedy_prefix_acceptance"]
                if metric > best_metric:
                    best_metric, best_step, stale = metric, step, 0
                    save_checkpoint(head, output_dir / "best", checkpoint_config | {"best_step": best_step, "best_metric": best_metric})
                else:
                    stale += 1
                if args.early_stopping_patience and stale >= args.early_stopping_patience:
                    break
            if step >= args.max_steps:
                break
    finally:
        progress.close(); metrics_file.close()
    save_checkpoint(head, output_dir / "last", checkpoint_config | {"last_step": step})
    write_json(output_dir / "best_metrics.json", {"best_step": best_step, "best_metric": best_metric})
    write_json(output_dir / "last_metrics.json", {"last_step": step})
    write_json(output_dir / "run_manifest.json", {"status": "completed", "best_step": best_step, "best_metric": best_metric, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "run_config": run_config})
    print(f"HYDRAFLOW_COMPLETE best_step={best_step} best_metric={best_metric:.6f}", flush=True)


if __name__ == "__main__":
    main()
