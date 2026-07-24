#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orthrus_training.checkpointing import save_orthrus_checkpoint, save_training_state
from orthrus_training.data import (
    PackedTokenDataset,
    assert_disjoint_packed_manifests,
    make_diffusion_batch,
    sample_anchor_positions,
)
from orthrus_training.losses import (
    forward_kl_distillation,
    gather_logits,
    prefix_acceptance_metrics,
    token_accuracy,
)
from orthrus_training.modeling import (
    build_orthrus_from_qwen,
    count_parameters,
    dtype_from_string,
    load_tokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Orthrus diffusion heads by AR soft distillation.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train-manifest", default="data/packed_qwen3_1p7b/manifest.json")
    parser.add_argument("--eval-manifest", default=None)
    parser.add_argument("--output-dir", default="outputs/orthrus-qwen3-1p7b-a100")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--mask-token-id", type=int, default=151669)
    parser.add_argument("--num-anchor-blocks", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-anchor-blocks", type=int, default=8)
    parser.add_argument(
        "--best-metric",
        choices=("eval_loss", "eval_greedy_prefix_acceptance"),
        default="eval_loss",
        help="Validation metric used to select the atomic best checkpoint.",
    )
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-trainer-state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    cli_keys = {
        action.dest
        for action in parser._actions
        if action.dest != "help"
        and any(option in sys.argv[1:] for option in action.option_strings)
    }
    return args, cli_keys


def load_config_file(path: str | None) -> dict:
    if path is None:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(args, config: dict, cli_keys: set[str]):
    for key, value in config.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr) and attr not in cli_keys:
            setattr(args, attr, value)
    return args


def select_student_logits(diffusion_logits: torch.Tensor, block_size: int) -> torch.Tensor:
    batch_size, flat_len, vocab_size = diffusion_logits.shape
    num_blocks = flat_len // block_size
    logits = diffusion_logits.reshape(batch_size, num_blocks, block_size, vocab_size)
    return logits[:, :, : block_size - 1, :].reshape(batch_size, num_blocks * (block_size - 1), vocab_size)


@torch.no_grad()
def evaluate_distillation(
    model,
    dataloader: DataLoader,
    device: torch.device,
    block_size: int,
    mask_token_id: int,
    num_anchor_blocks: int,
    temperature: float,
    max_batches: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses = []
    accuracies = []
    prefix_acceptances = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        input_ids = batch.to(device=device, non_blocking=True)
        anchors = sample_anchor_positions(
            batch_size=input_ids.shape[0],
            seq_len=input_ids.shape[1],
            block_size=block_size,
            num_blocks=num_anchor_blocks,
            device=device,
        )
        diffusion_ids, diff_position_ids, causal_limit, teacher_positions, target_ids = make_diffusion_batch(
            input_ids=input_ids,
            anchors=anchors,
            block_size=block_size,
            mask_token_id=mask_token_id,
        )

        ar_outputs = model(input_ids=input_ids, use_cache=True, is_diffusion_pass=False)
        teacher_logits = gather_logits(ar_outputs.logits, teacher_positions).detach()
        diff_outputs = model(
            input_ids=diffusion_ids,
            position_ids=diff_position_ids,
            past_key_values=ar_outputs.past_key_values,
            use_cache=False,
            is_diffusion_pass=True,
            causal_limit=causal_limit,
            ar_seq_len=input_ids.shape[1],
        )
        student_logits = select_student_logits(diff_outputs.logits, block_size)
        losses.append(float(forward_kl_distillation(student_logits, teacher_logits, temperature=temperature).cpu()))
        accuracies.append(float(token_accuracy(student_logits, target_ids).cpu()))
        prefix_metrics = prefix_acceptance_metrics(student_logits, target_ids, block_size)
        prefix_acceptances.append(float(prefix_metrics["greedy_prefix_acceptance"].cpu()))

    if was_training:
        model.train()

    return {
        "eval_loss": sum(losses) / len(losses) if losses else float("inf"),
        "eval_top1": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "eval_greedy_prefix_acceptance": (
            sum(prefix_acceptances) / len(prefix_acceptances) if prefix_acceptances else 0.0
        ),
        "eval_batches": len(losses),
    }


def main() -> None:
    parsed_args, cli_keys = parse_args()
    args = merge_config(parsed_args, load_config_file(parsed_args.config), cli_keys)
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("This training loop expects a CUDA GPU with FlexAttention support.")

    device = torch.device("cuda")
    dtype = dtype_from_string(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.base_model)
    train_unique = None
    eval_unique = None
    if args.eval_manifest:
        train_unique, eval_unique = assert_disjoint_packed_manifests(
            args.train_manifest,
            args.eval_manifest,
        )
    dataset = PackedTokenDataset(args.train_manifest)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_dataloader = None
    if args.eval_manifest:
        eval_dataset = PackedTokenDataset(args.eval_manifest)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    model, load_info = build_orthrus_from_qwen(
        base_model_name_or_path=args.base_model,
        upstream_dir=args.upstream_dir,
        block_size=args.block_size,
        mask_token_id=args.mask_token_id,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.to(device=device, dtype=dtype)
    model.train()
    model.config.use_cache = True
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.compile:
        model = torch.compile(model)

    total_params, trainable_params = count_parameters(model)
    trainable_ratio = trainable_params / total_params
    print(f"parameters total={total_params:,} trainable={trainable_params:,} ratio={trainable_ratio:.2%}")
    print(f"load missing={len(load_info['missing'])} unexpected={len(load_info['unexpected'])}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = min(args.max_steps, args.epochs * updates_per_epoch)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    metadata = vars(args) | {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
        "train_unique_sequences": train_unique,
        "eval_unique_sequences": eval_unique,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "running",
                "run_config": metadata,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    global_step = 0
    running_loss = 0.0
    running_acc = 0.0
    best_metric = float("inf") if args.best_metric == "eval_loss" else float("-inf")
    best_step = None
    best_values: dict[str, float] = {}
    stop_requested = False

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"ORTHRUS_STOP_REQUESTED signal={signum}; finishing the current optimizer step", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started_at = time.perf_counter()
    metrics_path = output_dir / "train_metrics.jsonl"
    metrics_file = metrics_path.open("a", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)

    try:
        progress = tqdm(total=total_steps, desc="optimizer steps")
        for epoch in range(args.epochs):
            for micro_step, batch in enumerate(dataloader):
                input_ids = batch.to(device=device, non_blocking=True)
                anchors = sample_anchor_positions(
                    batch_size=input_ids.shape[0],
                    seq_len=input_ids.shape[1],
                    block_size=args.block_size,
                    num_blocks=args.num_anchor_blocks,
                    device=device,
                )
                diffusion_ids, diff_position_ids, causal_limit, teacher_positions, target_ids = make_diffusion_batch(
                    input_ids=input_ids,
                    anchors=anchors,
                    block_size=args.block_size,
                    mask_token_id=args.mask_token_id,
                )

                with torch.no_grad():
                    ar_outputs = model(input_ids=input_ids, use_cache=True, is_diffusion_pass=False)
                    teacher_logits = gather_logits(ar_outputs.logits, teacher_positions).detach()
                    past_key_values = ar_outputs.past_key_values

                diff_outputs = model(
                    input_ids=diffusion_ids,
                    position_ids=diff_position_ids,
                    past_key_values=past_key_values,
                    use_cache=False,
                    is_diffusion_pass=True,
                    causal_limit=causal_limit,
                    ar_seq_len=input_ids.shape[1],
                )
                student_logits = select_student_logits(diff_outputs.logits, args.block_size)
                loss = forward_kl_distillation(student_logits, teacher_logits, temperature=args.temperature)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

                running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
                running_acc += float(token_accuracy(student_logits.detach(), target_ids).cpu())

                if (micro_step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    progress.update(1)

                    if global_step % args.log_every == 0:
                        denom = args.log_every * args.gradient_accumulation_steps
                        avg_loss = running_loss / denom
                        avg_acc = running_acc / denom
                        lr = scheduler.get_last_lr()[0]
                        elapsed = time.perf_counter() - started_at
                        record = {
                            "step": global_step,
                            "epoch": epoch,
                            "loss": avg_loss,
                            "top1": avg_acc,
                            "lr": lr,
                            "elapsed_seconds": elapsed,
                            "optimizer_steps_per_second": global_step / elapsed if elapsed > 0 else 0.0,
                            "peak_memory_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
                        }
                        print(
                            f"step={global_step} loss={avg_loss:.5f} top1={avg_acc:.4f} "
                            f"lr={lr:.3e} peak_gb={record['peak_memory_gb']:.2f}"
                        )
                        metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                        metrics_file.flush()
                        running_loss = 0.0
                        running_acc = 0.0

                    if eval_dataloader is not None and args.eval_every > 0 and global_step % args.eval_every == 0:
                        eval_record = evaluate_distillation(
                            model=model,
                            dataloader=eval_dataloader,
                            device=device,
                            block_size=args.block_size,
                            mask_token_id=args.mask_token_id,
                            num_anchor_blocks=args.eval_anchor_blocks,
                            temperature=args.temperature,
                            max_batches=args.eval_batches,
                        )
                        eval_record.update(
                            {
                                "step": global_step,
                                "epoch": epoch,
                                "kind": "eval",
                                "best_metric_name": args.best_metric,
                                "best_metric": best_metric,
                            }
                        )
                        print(
                            f"eval step={global_step} loss={eval_record['eval_loss']:.5f} "
                            f"top1={eval_record['eval_top1']:.4f} "
                            f"prefix={eval_record['eval_greedy_prefix_acceptance']:.4f}"
                        )
                        metrics_file.write(json.dumps(eval_record, sort_keys=True) + "\n")
                        metrics_file.flush()

                        candidate_metric = eval_record[args.best_metric]
                        is_better = (
                            candidate_metric < best_metric
                            if args.best_metric == "eval_loss"
                            else candidate_metric > best_metric
                        )
                        if is_better:
                            best_metric = candidate_metric
                            best_step = global_step
                            best_values = {
                                "best_eval_loss": eval_record["eval_loss"],
                                "best_eval_top1": eval_record["eval_top1"],
                                "best_eval_greedy_prefix_acceptance": eval_record[
                                    "eval_greedy_prefix_acceptance"
                                ],
                            }
                            save_orthrus_checkpoint(model, tokenizer, output_dir / "best", args.upstream_dir)
                            with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                                json.dump(
                                    {
                                        "best_step": best_step,
                                        "best_metric_name": args.best_metric,
                                        "best_metric": best_metric,
                                        **best_values,
                                        "checkpoint_written": True,
                                    },
                                    f,
                                    indent=2,
                                    sort_keys=True,
                                )
                                f.write("\n")

                    if args.save_every > 0 and global_step % args.save_every == 0:
                        last_dir = output_dir / "last"
                        save_orthrus_checkpoint(model, tokenizer, last_dir, args.upstream_dir)
                        with (output_dir / "last_metrics.json").open("w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "last_step": global_step,
                                    "best_step": best_step,
                                    "best_metric_name": args.best_metric,
                                    "best_metric": best_metric if best_step is not None else None,
                                    **best_values,
                                },
                                f,
                                indent=2,
                                sort_keys=True,
                            )
                            f.write("\n")
                        if args.save_trainer_state:
                            save_training_state(last_dir, optimizer, scheduler, global_step, epoch)

                    if global_step >= total_steps:
                        break

                if global_step >= total_steps or stop_requested:
                    break

            if global_step >= total_steps or stop_requested:
                break

        progress.close()
        save_orthrus_checkpoint(model, tokenizer, output_dir / "last", args.upstream_dir)
        with (output_dir / "last_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "last_step": global_step,
                    "best_step": best_step,
                    "best_metric_name": args.best_metric,
                    "best_metric": best_metric if best_step is not None else None,
                    **best_values,
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")
        if args.save_trainer_state:
            save_training_state(output_dir / "last", optimizer, scheduler, global_step, args.epochs)
        with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "interrupted" if stop_requested else "completed",
                    "best_step": best_step,
                    "best_metric_name": args.best_metric,
                    "best_metric": best_metric if best_step is not None else None,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "run_config": metadata,
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")
    finally:
        metrics_file.close()


if __name__ == "__main__":
    main()
