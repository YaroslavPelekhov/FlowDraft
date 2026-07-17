#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orthrus_training.checkpointing import save_orthrus_checkpoint, save_training_state
from orthrus_training.data import PackedTokenDataset, sample_anchor_positions
from orthrus_training.flowdraft import (
    make_flowdraft_batch,
    make_flowdraft_inputs_embeds,
    sample_flow_state_mix,
    select_endpoint_logits,
)
from orthrus_training.losses import forward_kl_distillation, gather_logits, token_accuracy
from orthrus_training.modeling import (
    build_orthrus_from_qwen,
    count_parameters,
    dtype_from_string,
    load_tokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FlowDraft endpoint drafter inside Orthrus verifier.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train-manifest", default="data/packed_qwen3_1p7b/manifest.json")
    parser.add_argument("--eval-manifest", default=None)
    parser.add_argument("--output-dir", default="/dev/shm/flowdraft_runs/flowdraft_quick2h")
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
    parser.add_argument("--flow-state-min", type=float, default=0.0)
    parser.add_argument("--flow-state-max", type=float, default=0.5)
    parser.add_argument("--hard-ce-weight", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--consistency-start-step", type=int, default=400)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-anchor-blocks", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=100)
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


def flowdraft_forward(
    model,
    input_ids: torch.Tensor,
    block_size: int,
    mask_token_id: int,
    num_anchor_blocks: int,
    state_min: float,
    state_max: float,
    device: torch.device,
):
    anchors = sample_anchor_positions(
        batch_size=input_ids.shape[0],
        seq_len=input_ids.shape[1],
        block_size=block_size,
        num_blocks=num_anchor_blocks,
        device=device,
    )
    clean_blocks, position_ids, causal_limit, teacher_positions, target_ids = make_flowdraft_batch(
        input_ids=input_ids,
        anchors=anchors,
        block_size=block_size,
    )
    state_mix = sample_flow_state_mix(
        batch_size=input_ids.shape[0],
        num_blocks=anchors.shape[1],
        min_mix=state_min,
        max_mix=state_max,
        device=device,
    )
    flow_inputs = make_flowdraft_inputs_embeds(
        model=model,
        clean_blocks=clean_blocks,
        mask_token_id=mask_token_id,
        state_mix=state_mix,
    )

    ar_outputs = model(input_ids=input_ids, use_cache=True, is_diffusion_pass=False)
    teacher_logits = gather_logits(ar_outputs.logits, teacher_positions).detach()
    diff_outputs = model(
        inputs_embeds=flow_inputs,
        position_ids=position_ids,
        past_key_values=ar_outputs.past_key_values,
        use_cache=False,
        is_diffusion_pass=True,
        causal_limit=causal_limit,
        ar_seq_len=input_ids.shape[1],
    )
    student_logits = select_endpoint_logits(diff_outputs.logits, block_size)
    return {
        "student_logits": student_logits,
        "teacher_logits": teacher_logits,
        "target_ids": target_ids,
        "clean_blocks": clean_blocks,
        "position_ids": position_ids,
        "causal_limit": causal_limit,
        "past_key_values": ar_outputs.past_key_values,
    }


def consistency_loss(
    model,
    first_logits: torch.Tensor,
    clean_blocks: torch.Tensor,
    position_ids: torch.Tensor,
    causal_limit: torch.Tensor,
    past_key_values,
    block_size: int,
    mask_token_id: int,
    ar_seq_len: int,
    temperature: float,
) -> torch.Tensor:
    batch_size, num_blocks, _ = clean_blocks.shape
    with torch.no_grad():
        endpoint_ids = clean_blocks.clone()
        endpoint_ids[:, :, 1:] = first_logits.argmax(dim=-1).reshape(batch_size, num_blocks, block_size - 1)
        late_mix = torch.full((batch_size, num_blocks, 1, 1), 0.75, device=clean_blocks.device)

    late_inputs = make_flowdraft_inputs_embeds(
        model=model,
        clean_blocks=clean_blocks,
        mask_token_id=mask_token_id,
        state_mix=late_mix,
        state_token_ids=endpoint_ids,
    )
    late_outputs = model(
        inputs_embeds=late_inputs,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=False,
        is_diffusion_pass=True,
        causal_limit=causal_limit,
        ar_seq_len=ar_seq_len,
    )
    late_logits = select_endpoint_logits(late_outputs.logits, block_size)
    return forward_kl_distillation(late_logits, first_logits.detach(), temperature=temperature)


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
    hard_ce_losses = []
    accuracies = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        input_ids = batch.to(device=device, non_blocking=True)
        out = flowdraft_forward(
            model=model,
            input_ids=input_ids,
            block_size=block_size,
            mask_token_id=mask_token_id,
            num_anchor_blocks=num_anchor_blocks,
            state_min=0.0,
            state_max=0.0,
            device=device,
        )
        losses.append(float(forward_kl_distillation(out["student_logits"], out["teacher_logits"], temperature).cpu()))
        hard_ce_losses.append(
            float(
                F.cross_entropy(
                    out["student_logits"].reshape(-1, out["student_logits"].shape[-1]).float(),
                    out["target_ids"].reshape(-1),
                ).cpu()
            )
        )
        accuracies.append(float(token_accuracy(out["student_logits"], out["target_ids"]).cpu()))

    if was_training:
        model.train()

    return {
        "eval_loss": sum(losses) / len(losses) if losses else float("inf"),
        "eval_hard_ce": sum(hard_ce_losses) / len(hard_ce_losses) if hard_ce_losses else float("inf"),
        "eval_top1": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "eval_batches": len(losses),
    }


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


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
    model.config.flowdraft_training = True
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
        "method": "flowdraft_ce",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
    }
    write_json(output_dir / "run_config.json", metadata)

    global_step = 0
    running_loss = 0.0
    running_ce_loss = 0.0
    running_hard_ce_loss = 0.0
    running_consistency_loss = 0.0
    running_acc = 0.0
    best_eval_loss = float("inf")
    best_step = None
    best_eval_top1 = None
    started_at = time.perf_counter()
    metrics_file = (output_dir / "train_metrics.jsonl").open("a", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)

    try:
        progress = tqdm(total=total_steps, desc="flowdraft optimizer steps")
        for epoch in range(args.epochs):
            for micro_step, batch in enumerate(dataloader):
                input_ids = batch.to(device=device, non_blocking=True)
                out = flowdraft_forward(
                    model=model,
                    input_ids=input_ids,
                    block_size=args.block_size,
                    mask_token_id=args.mask_token_id,
                    num_anchor_blocks=args.num_anchor_blocks,
                    state_min=args.flow_state_min,
                    state_max=args.flow_state_max,
                    device=device,
                )
                ce_loss = forward_kl_distillation(
                    out["student_logits"],
                    out["teacher_logits"],
                    temperature=args.temperature,
                )
                hard_ce_loss = F.cross_entropy(
                    out["student_logits"].reshape(-1, out["student_logits"].shape[-1]).float(),
                    out["target_ids"].reshape(-1),
                )
                consistency = torch.zeros((), device=device)
                if args.consistency_weight > 0 and global_step >= args.consistency_start_step:
                    consistency = consistency_loss(
                        model=model,
                        first_logits=out["student_logits"],
                        clean_blocks=out["clean_blocks"],
                        position_ids=out["position_ids"],
                        causal_limit=out["causal_limit"],
                        past_key_values=out["past_key_values"],
                        block_size=args.block_size,
                        mask_token_id=args.mask_token_id,
                        ar_seq_len=input_ids.shape[1],
                        temperature=args.temperature,
                    )
                loss = ce_loss + args.hard_ce_weight * hard_ce_loss + args.consistency_weight * consistency
                (loss / args.gradient_accumulation_steps).backward()

                running_loss += float(loss.detach().cpu())
                running_ce_loss += float(ce_loss.detach().cpu())
                running_hard_ce_loss += float(hard_ce_loss.detach().cpu())
                running_consistency_loss += float(consistency.detach().cpu())
                running_acc += float(token_accuracy(out["student_logits"].detach(), out["target_ids"]).cpu())

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
                        avg_ce = running_ce_loss / denom
                        avg_hard_ce = running_hard_ce_loss / denom
                        avg_consistency = running_consistency_loss / denom
                        avg_acc = running_acc / denom
                        lr = scheduler.get_last_lr()[0]
                        elapsed = time.perf_counter() - started_at
                        record = {
                            "step": global_step,
                            "epoch": epoch,
                            "loss": avg_loss,
                            "teacher_kl": avg_ce,
                            "hard_ce": avg_hard_ce,
                            "hard_ce_weight": args.hard_ce_weight,
                            "consistency_loss": avg_consistency,
                            "top1": avg_acc,
                            "lr": lr,
                            "elapsed_seconds": elapsed,
                            "optimizer_steps_per_second": global_step / elapsed if elapsed > 0 else 0.0,
                            "peak_memory_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
                        }
                        print(
                            f"step={global_step} loss={avg_loss:.5f} teacher_kl={avg_ce:.5f} "
                            f"hard_ce={avg_hard_ce:.5f} consistency={avg_consistency:.5f} top1={avg_acc:.4f} "
                            f"lr={lr:.3e} peak_gb={record['peak_memory_gb']:.2f}"
                        )
                        metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                        metrics_file.flush()
                        running_loss = 0.0
                        running_ce_loss = 0.0
                        running_hard_ce_loss = 0.0
                        running_consistency_loss = 0.0
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
                        eval_record.update({"step": global_step, "epoch": epoch, "kind": "eval"})
                        print(
                            f"eval step={global_step} loss={eval_record['eval_loss']:.5f} "
                            f"top1={eval_record['eval_top1']:.4f}"
                        )
                        metrics_file.write(json.dumps(eval_record, sort_keys=True) + "\n")
                        metrics_file.flush()

                        if eval_record["eval_loss"] < best_eval_loss:
                            best_eval_loss = eval_record["eval_loss"]
                            best_step = global_step
                            best_eval_top1 = eval_record["eval_top1"]
                            save_orthrus_checkpoint(model, tokenizer, output_dir / "best", args.upstream_dir)
                            write_json(
                                output_dir / "best_metrics.json",
                                {
                                    "best_step": best_step,
                                    "best_eval_loss": best_eval_loss,
                                    "best_eval_top1": best_eval_top1,
                                    "checkpoint_written": True,
                                    "method": "flowdraft_ce",
                                },
                            )

                    if args.save_every > 0 and global_step % args.save_every == 0:
                        last_dir = output_dir / "last"
                        save_orthrus_checkpoint(model, tokenizer, last_dir, args.upstream_dir)
                        write_json(
                            output_dir / "last_metrics.json",
                            {
                                "last_step": global_step,
                                "best_step": best_step,
                                "best_eval_loss": best_eval_loss if best_step is not None else None,
                                "method": "flowdraft_ce",
                            },
                        )
                        if args.save_trainer_state:
                            save_training_state(last_dir, optimizer, scheduler, global_step, epoch)

                    if global_step >= total_steps:
                        break

                if global_step >= total_steps:
                    break

            if global_step >= total_steps:
                break

        progress.close()
        save_orthrus_checkpoint(model, tokenizer, output_dir / "last", args.upstream_dir)
        write_json(
            output_dir / "last_metrics.json",
            {
                "last_step": global_step,
                "best_step": best_step,
                "best_eval_loss": best_eval_loss if best_step is not None else None,
                "method": "flowdraft_ce",
            },
        )
        if args.save_trainer_state:
            save_training_state(output_dir / "last", optimizer, scheduler, global_step, args.epochs)
    finally:
        metrics_file.close()


if __name__ == "__main__":
    main()
