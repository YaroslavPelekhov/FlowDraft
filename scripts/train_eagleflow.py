#!/usr/bin/env python
"""Train the attention-conditioned endpoint Flow Map drafter on a frozen verifier."""

from __future__ import annotations

import argparse
import json
import signal
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orthrus_training.data import PackedTokenDataset, assert_disjoint_packed_manifests
from orthrus_training.eagleflow import EagleFlowDrafter
from orthrus_training.losses import forward_kl_distillation, prefix_acceptance_metrics, prefix_survival_cross_entropy
from orthrus_training.modeling import dtype_from_string, load_flowdraft_adapter
from train_hydraflow import (
    advanced_token_embeddings,
    collect_teacher,
    load_config,
    merge_config,
    sha256_file,
    teacher_forcing_ratio,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an EAGLE-conditioned endpoint Flow Map drafter.")
    parser.add_argument("--config")
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-anchor-blocks", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-loss-weight", type=float, default=0.25)
    parser.add_argument("--embedding-loss-weight", type=float, default=0.05)
    parser.add_argument("--prefix-loss-weight", type=float, default=1.0)
    parser.add_argument("--kl-loss-weight", type=float, default=0.0)
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--prefix-weight-decay", type=float, default=0.9)
    parser.add_argument("--state-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--teacher-forcing-start", type=float, default=1.0)
    parser.add_argument("--teacher-forcing-end", type=float, default=0.0)
    parser.add_argument("--teacher-forcing-decay-ratio", type=float, default=0.35)
    parser.add_argument("--feedback-mode", choices=("continuous", "advanced_token"), default="advanced_token")
    parser.add_argument("--flow-diagonal-fraction", type=float, default=0.25)
    parser.add_argument("--flow-consistency-weight", type=float, default=0.05)
    parser.add_argument("--flow-diagonal-loss-weight", type=float, default=0.10)
    parser.add_argument("--flow-time-min", type=float, default=0.05)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-anchor-blocks", type=int, default=32)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=281)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    cli = {
        item.dest for item in parser._actions
        if item.dest != "help" and any(flag in sys.argv[1:] for flag in item.option_strings)
    }
    return args, cli


def save_checkpoint(head: EagleFlowDrafter, directory: Path, config: dict) -> None:
    temporary = directory.with_name(f".{directory.name}.tmp")
    previous = directory.with_name(f".{directory.name}.previous")
    if temporary.exists():
        shutil.rmtree(temporary)
    if previous.exists():
        shutil.rmtree(previous)
    temporary.mkdir(parents=True)
    state = {name: value.detach().cpu().contiguous() for name, value in head.state_dict().items()}
    save_file(state, temporary / "eagleflow_head.safetensors")
    write_json(temporary / "eagleflow_config.json", config)
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


def forward_head(
    model,
    head: EagleFlowDrafter,
    teacher: dict,
    ratio: float,
    feedback_mode: str,
    flow_targets: torch.Tensor | None = None,
    flow_time: torch.Tensor | None = None,
):
    feedback = advanced_token_embeddings if feedback_mode == "advanced_token" else None
    hidden, embeddings = head.rollout(
        teacher["context_hidden"],
        teacher["anchor_embeddings"],
        teacher_embeddings=teacher["target_embeddings"],
        teacher_features=teacher["target_hidden"],
        teacher_forcing_ratio=ratio,
        feedback_embedding_fn=(lambda value: feedback(model, value)) if feedback is not None else None,
        flow_targets=flow_targets,
        flow_time=flow_time,
    )
    logits = model.lm_head(hidden.reshape(-1, hidden.shape[-1])).reshape(
        hidden.shape[0], -1, int(model.config.vocab_size)
    )
    return hidden, embeddings, logits


@torch.no_grad()
def evaluate(model, head: EagleFlowDrafter, dataloader, args) -> dict:
    head.eval()
    hidden_losses, embedding_losses, kl_losses, prefixes, first_tokens = [], [], [], [], []
    for index, raw in enumerate(dataloader):
        if index >= args.eval_batches:
            break
        teacher = collect_teacher(model, raw.to(device="cuda", non_blocking=True), args.eval_anchor_blocks)
        hidden, embeddings, logits = forward_head(model, head, teacher, 0.0, args.feedback_mode)
        hidden_rms = teacher["target_hidden"].float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        embedding_rms = teacher["target_embeddings"].float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        hidden_losses.append(float(((hidden.float() - teacher["target_hidden"].float()) / hidden_rms).square().mean().cpu()))
        embedding_losses.append(float(((embeddings.float() - teacher["target_embeddings"].float()) / embedding_rms).square().mean().cpu()))
        if args.kl_loss_weight:
            kl_losses.append(float(forward_kl_distillation(logits, teacher["teacher_logits"], args.kl_temperature, reduction="tokenmean").cpu()))
        metrics = prefix_acceptance_metrics(logits, teacher["target_tokens"], int(model.config.block_size))
        prefixes.append(float(metrics["greedy_prefix_acceptance"].cpu()))
        first_tokens.append(float(metrics["first_token_acc"].cpu()))
    head.train()
    return {
        "eval_hidden_mse": sum(hidden_losses) / len(hidden_losses),
        "eval_embedding_mse": sum(embedding_losses) / len(embedding_losses),
        "eval_kl": sum(kl_losses) / len(kl_losses) if kl_losses else None,
        "eval_greedy_prefix_acceptance": sum(prefixes) / len(prefixes),
        "eval_first_token_acc": sum(first_tokens) / len(first_tokens),
        "eval_batches": len(prefixes),
    }


def main() -> None:
    parsed, cli = parse_args()
    args = merge_config(parsed, load_config(parsed.config), cli)
    if not torch.cuda.is_available():
        raise RuntimeError("EagleFlow training requires CUDA")
    if args.state_size % args.num_heads:
        raise ValueError("state_size must be divisible by num_heads")
    if not 0.0 <= args.flow_diagonal_fraction <= 1.0:
        raise ValueError("flow_diagonal_fraction must be in [0, 1]")
    if not 0.0 <= args.flow_time_min < 1.0:
        raise ValueError("flow_time_min must be in [0, 1)")
    if not 0.0 < args.teacher_forcing_decay_ratio <= 1.0:
        raise ValueError("teacher_forcing_decay_ratio must be in (0, 1]")
    if args.kl_loss_weight < 0.0 or args.kl_temperature <= 0.0:
        raise ValueError("kl_loss_weight must be non-negative and kl_temperature positive")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    allowed = {"run.log", "supervisor.log", "supervisor.err.log"}
    if output_dir.exists() and {item.name for item in output_dir.iterdir()} - allowed:
        raise FileExistsError(f"Refusing to overwrite existing experiment: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_unique, eval_unique = assert_disjoint_packed_manifests(args.train_manifest, args.eval_manifest)
    dtype = dtype_from_string(args.dtype)
    model, parent_metadata, _ = load_flowdraft_adapter(args.init_checkpoint, args.upstream_dir, dtype, args.attn_implementation)
    model.to(device="cuda", dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = EagleFlowDrafter(
        int(model.config.hidden_size), int(model.config.block_size), args.state_size,
        args.num_layers, args.num_heads, args.dropout,
    ).to(device="cuda", dtype=dtype).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(args.max_steps * args.warmup_ratio)), args.max_steps)
    optimizer.zero_grad(set_to_none=True)
    train_loader = DataLoader(PackedTokenDataset(args.train_manifest), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(PackedTokenDataset(args.eval_manifest), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    checkpoint_config = {
        "format": "eagleflow_endpoint_drafter_v1",
        "method": "attention_conditioned_endpoint_flow_map",
        "objective": "prefix_survival_plus_eagle_feature_token_trajectory_plus_endpoint_flow_consistency",
        "base_flowdraft_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "base_flowdraft_adapter_sha256": sha256_file(Path(args.init_checkpoint) / "adapter_model.safetensors"),
        "block_size": head.block_size, "hidden_size": head.hidden_size, "state_size": head.state_size,
        "num_layers": head.num_layers, "num_heads": head.num_heads, "feedback_mode": args.feedback_mode,
        "inference": "time-zero endpoint flow rollout with advanced-token feedback and one frozen AR verifier pass",
    }
    run_config = vars(args) | {
        "method": checkpoint_config["method"], "parent_adapter_metadata": parent_metadata,
        "train_unique_sequences": train_unique, "eval_unique_sequences": eval_unique,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "run_manifest.json", {"status": "running", "run_config": run_config, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
    print(f"eagleflow trainable={sum(value.numel() for value in head.parameters()):,} train_unique={train_unique} eval_unique={eval_unique}", flush=True)
    best_metric, best_step, stale, step = float("-inf"), None, 0, 0
    stop_requested = False

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"EAGLEFLOW_STOP_REQUESTED signal={signum}; finishing the current optimizer step", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    metrics_file = (output_dir / "train_metrics.jsonl").open("w", encoding="utf-8")
    progress = tqdm(total=args.max_steps, desc="eagleflow optimizer steps")
    try:
        for raw in train_loader:
            step += 1
            ratio = teacher_forcing_ratio(args, step)
            teacher = collect_teacher(model, raw.to(device="cuda", non_blocking=True), args.num_anchor_blocks)
            hidden, embeddings, logits = forward_head(model, head, teacher, ratio, args.feedback_mode)
            hidden_rms = teacher["target_hidden"].float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            embedding_rms = teacher["target_embeddings"].float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            hidden_loss = ((hidden.float() - teacher["target_hidden"].float()) / hidden_rms).square().mean()
            embedding_loss = ((embeddings.float() - teacher["target_embeddings"].float()) / embedding_rms).square().mean()
            prefix_loss = prefix_survival_cross_entropy(logits, teacher["target_tokens"], int(model.config.block_size), args.prefix_weight_decay)
            kl_loss = forward_kl_distillation(logits, teacher["teacher_logits"], args.kl_temperature, reduction="tokenmean") if args.kl_loss_weight else torch.zeros((), device=hidden.device)
            diagonal_loss, consistency_loss = torch.zeros((), device=hidden.device), torch.zeros((), device=hidden.device)
            if args.flow_diagonal_fraction and torch.rand((), device=hidden.device) < args.flow_diagonal_fraction:
                flow_time = torch.empty((*teacher["context_hidden"].shape[:2], 1), device=hidden.device, dtype=hidden.dtype).uniform_(args.flow_time_min, 1.0)
                diagonal_hidden, _, _ = forward_head(model, head, teacher, 1.0, args.feedback_mode, teacher["target_hidden"], flow_time)
                diagonal_loss = ((diagonal_hidden.float() - teacher["target_hidden"].float()) / hidden_rms).square().mean()
                consistency_loss = ((diagonal_hidden.float() - hidden.detach().float()) / hidden_rms).square().mean()
            loss = (
                args.hidden_loss_weight * hidden_loss + args.embedding_loss_weight * embedding_loss
                + args.prefix_loss_weight * prefix_loss + args.kl_loss_weight * kl_loss
                + args.flow_diagonal_loss_weight * diagonal_loss + args.flow_consistency_weight * consistency_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); progress.update(1)
            metrics = prefix_acceptance_metrics(logits.detach(), teacher["target_tokens"], int(model.config.block_size))
            record = {
                "step": step, "loss": float(loss.detach().cpu()), "hidden_mse": float(hidden_loss.detach().cpu()),
                "embedding_mse": float(embedding_loss.detach().cpu()), "prefix_loss": float(prefix_loss.detach().cpu()),
                "kl": float(kl_loss.detach().cpu()), "flow_diagonal_mse": float(diagonal_loss.detach().cpu()),
                "flow_consistency_mse": float(consistency_loss.detach().cpu()),
                "greedy_prefix": float(metrics["greedy_prefix_acceptance"].cpu()), "first_token_acc": float(metrics["first_token_acc"].cpu()),
                "teacher_forcing_ratio": ratio, "lr": scheduler.get_last_lr()[0],
                "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            }
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
            if step >= args.max_steps or stop_requested:
                break
    finally:
        progress.close(); metrics_file.close()
    save_checkpoint(head, output_dir / "last", checkpoint_config | {"last_step": step})
    write_json(output_dir / "best_metrics.json", {"best_step": best_step, "best_metric": best_metric})
    write_json(output_dir / "last_metrics.json", {"last_step": step})
    status = "interrupted" if stop_requested else "completed"
    write_json(output_dir / "run_manifest.json", {"status": status, "best_step": best_step, "best_metric": best_metric, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "run_config": run_config})
    print(f"EAGLEFLOW_COMPLETE status={status} best_step={best_step} best_metric={best_metric:.6f}", flush=True)


if __name__ == "__main__":
    main()
