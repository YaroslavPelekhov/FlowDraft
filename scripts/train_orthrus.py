#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orthrus_training.checkpointing import save_orthrus_checkpoint, save_training_state
from orthrus_training.data import PackedTokenDataset, make_diffusion_batch, sample_anchor_positions
from orthrus_training.losses import forward_kl_distillation, gather_logits, token_accuracy
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
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_config_file(path: str | None) -> dict:
    if path is None:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(args, config: dict):
    for key, value in config.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr):
            setattr(args, attr, value)
    return args


def select_student_logits(diffusion_logits: torch.Tensor, block_size: int) -> torch.Tensor:
    batch_size, flat_len, vocab_size = diffusion_logits.shape
    num_blocks = flat_len // block_size
    logits = diffusion_logits.reshape(batch_size, num_blocks, block_size, vocab_size)
    return logits[:, :, : block_size - 1, :].reshape(batch_size, num_blocks * (block_size - 1), vocab_size)


def main() -> None:
    parsed_args = parse_args()
    args = merge_config(parsed_args, load_config_file(parsed_args.config))
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
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    global_step = 0
    running_loss = 0.0
    running_acc = 0.0
    optimizer.zero_grad(set_to_none=True)

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
                    print(f"step={global_step} loss={avg_loss:.5f} top1={avg_acc:.4f} lr={lr:.3e}")
                    running_loss = 0.0
                    running_acc = 0.0

                if global_step % args.save_every == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step:06d}"
                    save_orthrus_checkpoint(model, tokenizer, ckpt_dir, args.upstream_dir)
                    save_training_state(ckpt_dir, optimizer, scheduler, global_step, epoch)

                if global_step >= total_steps:
                    break

        if global_step >= total_steps:
            break

    progress.close()
    save_orthrus_checkpoint(model, tokenizer, output_dir / "final", args.upstream_dir)
    save_training_state(output_dir / "final", optimizer, scheduler, global_step, args.epochs)


if __name__ == "__main__":
    main()
