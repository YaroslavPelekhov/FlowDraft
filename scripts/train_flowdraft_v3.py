#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orthrus_training.checkpointing import save_trainable_checkpoint
from orthrus_training.data import (
    PackedTokenDataset,
    assert_disjoint_packed_manifests,
    sample_anchor_positions,
)
from orthrus_training.flowdraft import (
    condition_flowdraft_state,
    flow_map_step_size,
    make_endpoint_blocks,
    make_flowdraft_batch,
    make_flowdraft_inputs_embeds,
    sample_categorical_source_tokens,
    sample_cfm_time_pairs,
    select_endpoint_logits,
    topk_endpoint_embeddings,
)
from orthrus_training.losses import (
    bounded_jsd_distillation,
    forward_kl_distillation,
    verifier_aligned_losses,
    weighted_cross_entropy,
)
from orthrus_training.modeling import (
    build_orthrus_from_qwen,
    count_parameters,
    dtype_from_string,
    load_tokenizer,
    parallel_verifier_logits,
    set_flowdraft_state_adapter_trainable,
)


def parse_args() -> tuple[argparse.Namespace, set[str]]:
    parser = argparse.ArgumentParser(
        description="Train verifier-aligned stochastic categorical FlowDraft."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--mask-token-id", type=int, default=151669)
    parser.add_argument("--source-prior", choices=["uniform", "mask"], default="uniform")
    parser.add_argument("--source-seed", type=int, default=2718)
    parser.add_argument("--num-anchor-blocks", type=int, default=16)
    parser.add_argument("--eval-anchor-blocks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--diagonal-fraction", type=float, default=0.5)
    parser.add_argument("--one-jump-fraction", type=float, default=0.5)
    parser.add_argument("--flow-time-logit-mean", type=float, default=-0.4)
    parser.add_argument("--flow-time-logit-std", type=float, default=1.0)
    parser.add_argument("--flow-time-max", type=float, default=0.95)
    parser.add_argument("--flow-time-conditioning-scale", type=float, default=0.05)
    parser.add_argument("--flow-state-adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flow-adapter-bottleneck", type=int, default=256)
    parser.add_argument("--endpoint-topk", type=int, default=32)
    parser.add_argument("--endpoint-kl-weight", type=float, default=1.0)
    parser.add_argument("--endpoint-top1-weight", type=float, default=0.5)
    parser.add_argument("--proposal-forward-kl-weight", type=float, default=1.0)
    parser.add_argument("--proposal-reverse-kl-weight", type=float, default=0.5)
    parser.add_argument("--proposal-top1-weight", type=float, default=1.0)
    parser.add_argument("--rejected-decay", type=float, default=0.8)
    parser.add_argument("--reverse-kl-clip", type=float, default=0.01)
    parser.add_argument("--semigroup-weight", type=float, default=0.1)
    parser.add_argument("--semigroup-start-step", type=int, default=150)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    cli_keys = {
        action.dest
        for action in parser._actions
        if action.dest != "help"
        and any(option in sys.argv[1:] for option in action.option_strings)
    }
    return args, cli_keys


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_config(args: argparse.Namespace, values: dict, cli_keys: set[str]) -> argparse.Namespace:
    for key, value in values.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr) and attr not in cli_keys:
            setattr(args, attr, value)
    return args


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: str | Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(root: Path) -> dict:
    def value(*parts: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *parts], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": value("rev-parse", "HEAD"),
        "branch": value("branch", "--show-current"),
        "status": value("status", "--short"),
    }


def select_block_values(
    values: torch.Tensor,
    block_mask: torch.Tensor,
    block_width: int,
) -> torch.Tensor:
    batch_size, num_blocks = block_mask.shape
    tail = values.shape[2:] if values.dim() > 2 else ()
    blocked = values.reshape(batch_size, num_blocks, block_width, *tail)
    selected = block_mask.sum(dim=1)
    if not torch.all(selected == selected[0]):
        raise ValueError("Every batch item must select the same number of blocks")
    return blocked[block_mask].reshape(batch_size, int(selected[0]), block_width, *tail)


def diffusion_logits_from_embeds(
    model,
    raw_inputs: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    position_ids: torch.Tensor,
    causal_limit: torch.Tensor,
    past_key_values,
    ar_seq_len: int,
    block_size: int,
    time_scale: float,
) -> torch.Tensor:
    conditioned = condition_flowdraft_state(
        model,
        raw_inputs,
        source_time,
        target_time,
        block_size=block_size,
        scale=time_scale,
    )
    outputs = model(
        inputs_embeds=conditioned,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=False,
        is_diffusion_pass=True,
        causal_limit=causal_limit,
        ar_seq_len=ar_seq_len,
    )
    return select_endpoint_logits(outputs.logits, block_size)


def state_inputs(
    model,
    endpoint_blocks: torch.Tensor,
    source_tokens: torch.Tensor,
    source_time: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    return make_flowdraft_inputs_embeds(
        model=model,
        clean_blocks=endpoint_blocks,
        mask_token_id=mask_token_id,
        state_mix=source_time,
        source_token_ids=source_tokens,
    )


def prepare_onpolicy_batch(model, input_ids: torch.Tensor, args, source_generator):
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    anchors = sample_anchor_positions(
        batch_size=batch_size,
        seq_len=seq_len,
        block_size=args.block_size,
        num_blocks=args.num_anchor_blocks,
        device=device,
    )
    clean_blocks, position_ids, causal_limit, _, _ = make_flowdraft_batch(
        input_ids, anchors, args.block_size
    )
    source_tokens = sample_categorical_source_tokens(
        clean_blocks,
        vocab_size=int(model.config.vocab_size),
        mask_token_id=args.mask_token_id,
        prior=args.source_prior,
        generator=source_generator,
    )
    zero = torch.zeros((batch_size, anchors.shape[1], 1, 1), device=device)
    one = torch.ones_like(zero)

    with torch.no_grad():
        ar_outputs = model(
            input_ids=input_ids,
            use_cache=True,
            is_diffusion_pass=False,
            logits_to_keep=1,
        )
    proposal_inputs = state_inputs(
        model, clean_blocks, source_tokens, zero, args.mask_token_id
    )
    proposal_logits = diffusion_logits_from_embeds(
        model,
        proposal_inputs,
        zero,
        one,
        position_ids,
        causal_limit,
        ar_outputs.past_key_values,
        seq_len,
        args.block_size,
        args.flow_time_conditioning_scale,
    )
    draft_tokens = proposal_logits.detach().argmax(dim=-1).reshape(
        batch_size, anchors.shape[1], args.block_size - 1
    )
    proposed_blocks = make_endpoint_blocks(clean_blocks[:, :, :1], draft_tokens)
    verifier_logits = parallel_verifier_logits(
        model,
        proposed_blocks,
        position_ids,
        causal_limit,
        ar_outputs.past_key_values,
        seq_len,
    )
    verifier_ids = verifier_logits.argmax(dim=-1).reshape(
        batch_size, anchors.shape[1], args.block_size - 1
    )
    endpoint_blocks = make_endpoint_blocks(clean_blocks[:, :, :1], verifier_ids)
    return {
        "clean_blocks": clean_blocks,
        "source_tokens": source_tokens,
        "endpoint_blocks": endpoint_blocks,
        "position_ids": position_ids,
        "causal_limit": causal_limit,
        "past_key_values": ar_outputs.past_key_values,
        "proposal_logits": proposal_logits,
        "verifier_logits": verifier_logits,
        "ar_seq_len": seq_len,
    }


def semigroup_psd_loss(
    model,
    direct_logits: torch.Tensor,
    raw_source_inputs: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    diagonal_mask: torch.Tensor,
    batch,
    args,
) -> torch.Tensor:
    """Stop-gradient two-step composition target for F_{s,t}=F_{u,t} o F_{s,u}."""

    off_diagonal = ~diagonal_mask
    if not torch.any(off_diagonal):
        return direct_logits.new_zeros(())
    midpoint = 0.5 * (source_time + target_time)
    batch_size, num_blocks, block_size = batch["endpoint_blocks"].shape
    hidden_size = raw_source_inputs.shape[-1]

    with torch.no_grad():
        first_leg_logits = diffusion_logits_from_embeds(
            model,
            raw_source_inputs,
            source_time,
            midpoint,
            batch["position_ids"],
            batch["causal_limit"],
            batch["past_key_values"],
            batch["ar_seq_len"],
            block_size,
            args.flow_time_conditioning_scale,
        )
        endpoint_embeds = topk_endpoint_embeddings(
            first_leg_logits,
            model.model.embed_tokens.weight,
            topk=args.endpoint_topk,
            temperature=args.temperature,
        ).reshape(batch_size, num_blocks, block_size - 1, hidden_size)
        source_blocks = raw_source_inputs.reshape(
            batch_size, num_blocks, block_size, hidden_size
        )
        gamma = flow_map_step_size(source_time, midpoint).to(source_blocks.dtype)
        transported = source_blocks[:, :, 1:, :] + gamma * (
            endpoint_embeds - source_blocks[:, :, 1:, :]
        )
        midpoint_inputs = torch.cat(
            [source_blocks[:, :, :1, :], transported], dim=2
        ).reshape_as(raw_source_inputs)
        composed_logits = diffusion_logits_from_embeds(
            model,
            midpoint_inputs,
            midpoint,
            target_time,
            batch["position_ids"],
            batch["causal_limit"],
            batch["past_key_values"],
            batch["ar_seq_len"],
            block_size,
            args.flow_time_conditioning_scale,
        )

    direct_selected = select_block_values(
        direct_logits, off_diagonal, block_size - 1
    ).reshape(batch_size, -1, direct_logits.shape[-1])
    composed_selected = select_block_values(
        composed_logits, off_diagonal, block_size - 1
    ).reshape_as(direct_selected)
    return bounded_jsd_distillation(
        direct_selected, composed_selected, temperature=args.temperature
    )


def compute_losses(model, input_ids: torch.Tensor, args, global_step: int, source_generator):
    batch = prepare_onpolicy_batch(model, input_ids, args, source_generator)
    batch_size = input_ids.shape[0]
    num_blocks = batch["endpoint_blocks"].shape[1]
    source_time, target_time, diagonal_mask = sample_cfm_time_pairs(
        batch_size=batch_size,
        num_blocks=num_blocks,
        diagonal_fraction=args.diagonal_fraction,
        device=input_ids.device,
        logit_mean=args.flow_time_logit_mean,
        logit_std=args.flow_time_logit_std,
        max_time=args.flow_time_max,
        one_jump_fraction=args.one_jump_fraction,
    )
    raw_inputs = state_inputs(
        model,
        batch["endpoint_blocks"],
        batch["source_tokens"],
        source_time,
        args.mask_token_id,
    )
    direct_logits = diffusion_logits_from_embeds(
        model,
        raw_inputs,
        source_time,
        target_time,
        batch["position_ids"],
        batch["causal_limit"],
        batch["past_key_values"],
        batch["ar_seq_len"],
        args.block_size,
        args.flow_time_conditioning_scale,
    )
    endpoint_kl = forward_kl_distillation(
        direct_logits,
        batch["verifier_logits"],
        temperature=args.temperature,
        reduction="tokenmean",
    )
    teacher_ids = batch["verifier_logits"].argmax(dim=-1)
    position_weights = torch.flip(
        torch.cumsum(
            torch.flip(
                args.rejected_decay
                ** torch.arange(
                    args.block_size - 1,
                    device=input_ids.device,
                    dtype=torch.float32,
                ),
                dims=[0],
            ),
            dim=0,
        ),
        dims=[0],
    )
    position_weights = position_weights / position_weights.mean().clamp_min(1e-8)
    endpoint_top1 = weighted_cross_entropy(
        direct_logits, teacher_ids, weights=position_weights
    )
    proposal = verifier_aligned_losses(
        batch["proposal_logits"],
        batch["verifier_logits"],
        block_size=args.block_size,
        temperature=args.temperature,
        rejected_decay=args.rejected_decay,
        reverse_kl_clip=args.reverse_kl_clip,
    )
    semigroup = direct_logits.new_zeros(())
    if args.semigroup_weight > 0 and global_step >= args.semigroup_start_step:
        semigroup = semigroup_psd_loss(
            model,
            direct_logits,
            raw_inputs,
            source_time,
            target_time,
            diagonal_mask,
            batch,
            args,
        )

    loss = (
        args.endpoint_kl_weight * endpoint_kl
        + args.endpoint_top1_weight * endpoint_top1
        + args.proposal_forward_kl_weight * proposal["forward_kl"]
        + args.proposal_reverse_kl_weight * proposal["reverse_kl"]
        + args.proposal_top1_weight * proposal["top1_ce"]
        + args.semigroup_weight * semigroup
    )
    return {
        "loss": loss,
        "endpoint_kl": endpoint_kl,
        "endpoint_top1_ce": endpoint_top1,
        "proposal_forward_kl": proposal["forward_kl"],
        "proposal_reverse_kl": proposal["reverse_kl"],
        "proposal_top1_ce": proposal["top1_ce"],
        "semigroup_jsd": semigroup,
        "greedy_prefix_acceptance": proposal["greedy_prefix_acceptance"],
        "first_token_accuracy": proposal["first_token_accuracy"],
    }


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: torch.device, args) -> dict:
    # Keep training=True so the upstream model applies the independent block
    # FlexAttention mask. Qwen3-1.7B has no active dropout in this path.
    model.train()
    generator = torch.Generator(device=device).manual_seed(args.source_seed + 1)
    totals: dict[str, float] = {}
    batches = 0
    for batch in dataloader:
        if batches >= args.eval_batches:
            break
        input_ids = batch.to(device=device, non_blocking=True)
        metrics = compute_losses(
            model,
            input_ids,
            args,
            global_step=max(args.semigroup_start_step - 1, 0),
            source_generator=generator,
        )
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        batches += 1
    return {
        f"eval_{key}": value / batches for key, value in totals.items()
    } | {"eval_batches": batches}


def main() -> None:
    parsed, cli_keys = parse_args()
    config_values = load_config(parsed.config)
    args = merge_config(parsed, config_values, cli_keys)
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("FlowDraft v3 training requires CUDA")
    if args.block_size < 2:
        raise ValueError("block_size must be at least two")

    device = torch.device("cuda")
    dtype = dtype_from_string(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PackedTokenDataset(args.train_manifest)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = None
    if args.eval_manifest:
        train_unique, eval_unique = assert_disjoint_packed_manifests(
            args.train_manifest, args.eval_manifest
        )
        print(
            f"data disjointness verified train_unique={train_unique} "
            f"eval_unique={eval_unique}"
        )
        eval_loader = DataLoader(
            PackedTokenDataset(args.eval_manifest),
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
        flowdraft_adapter_bottleneck=args.flow_adapter_bottleneck,
    )
    model.to(device=device, dtype=dtype)
    model.train()
    model.config.use_cache = True
    model.config.flowdraft_cfm = True
    model.config.flowdraft_objective = "verifier_anchored_psd_v3"
    model.config.flowdraft_time_conditioning_scale = args.flow_time_conditioning_scale
    model.config.flowdraft_endpoint_topk = args.endpoint_topk
    model.config.flowdraft_endpoint_transport = "topk"
    model.config.flowdraft_state_adapter = args.flow_state_adapter
    model.config.flowdraft_adapter_bottleneck = args.flow_adapter_bottleneck
    model.config.flowdraft_source_prior = args.source_prior
    model.config.flowdraft_source_seed = args.source_seed
    set_flowdraft_state_adapter_trainable(model, args.flow_state_adapter)

    total_parameters, trainable_parameters = count_parameters(model)
    print(
        f"parameters total={total_parameters:,} trainable={trainable_parameters:,} "
        f"ratio={trainable_parameters / total_parameters:.2%}"
    )
    print(
        f"load missing={len(load_info['missing'])} "
        f"unexpected={len(load_info['unexpected'])}"
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_steps = min(args.max_steps, args.epochs * updates_per_epoch)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(total_steps * args.warmup_ratio),
        total_steps,
    )

    root = Path(__file__).resolve().parents[1]
    manifest = {
        "status": "running",
        "method": "flowdraft_verifier_anchored_categorical_flow_map_v3",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "git": git_metadata(root),
        "config": config_values,
        "args": vars(args),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "eval_manifest_sha256": sha256_file(args.eval_manifest),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(
        output_dir / "run_config.json",
        {
            "method": manifest["method"],
            "base_reference": "frozen_qwen3_1p7b_ar",
            "source_distribution": f"categorical_{args.source_prior}_one_hot",
            "state_path": "linear_simplex_interpolant_projected_by_frozen_embedding",
            "endpoint_target": "frozen_qwen_verifier_on_draft_generated_prefix",
            "consistency": "stop_gradient_semigroup_psd_bounded_jsd",
            "hard_labels": "frozen_qwen_top1_only",
            "dataset_tokens_used_as_hard_labels": False,
            "lossless_verifier_trainable": False,
            **vars(args),
        },
    )

    generator = torch.Generator(device=device).manual_seed(args.source_seed)
    optimizer.zero_grad(set_to_none=True)
    metrics_handle = (output_dir / "train_metrics.jsonl").open(
        "a", encoding="utf-8"
    )
    global_step = 0
    best_acceptance = float("-inf")
    best_step = None
    stale_evals = 0
    running: dict[str, float] = {}
    running_microsteps = 0
    started = time.perf_counter()
    stopped_early = False

    try:
        progress = tqdm(total=total_steps, desc="FlowDraft v3 optimizer steps")
        for epoch in range(args.epochs):
            for micro_step, raw_batch in enumerate(train_loader):
                input_ids = raw_batch.to(device=device, non_blocking=True)
                metrics = compute_losses(
                    model, input_ids, args, global_step, generator
                )
                (metrics["loss"] / args.gradient_accumulation_steps).backward()
                for key, value in metrics.items():
                    running[key] = running.get(key, 0.0) + float(
                        value.detach().cpu()
                    )
                running_microsteps += 1

                if (micro_step + 1) % args.gradient_accumulation_steps:
                    continue
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    args.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)

                if global_step % args.log_every == 0:
                    record = {
                        key: value / running_microsteps
                        for key, value in running.items()
                    }
                    record.update(
                        {
                            "kind": "train",
                            "step": global_step,
                            "epoch": epoch,
                            "lr": scheduler.get_last_lr()[0],
                            "elapsed_seconds": time.perf_counter() - started,
                            "peak_memory_gb": torch.cuda.max_memory_allocated(device)
                            / (1024**3),
                        }
                    )
                    print(
                        f"step={global_step} loss={record['loss']:.4f} "
                        f"proposal_fkl={record['proposal_forward_kl']:.4f} "
                        f"proposal_rkl={record['proposal_reverse_kl']:.4f} "
                        f"top1_ce={record['proposal_top1_ce']:.4f} "
                        f"psd_jsd={record['semigroup_jsd']:.5f} "
                        f"prefix={record['greedy_prefix_acceptance']:.3f} "
                        f"first={record['first_token_accuracy']:.3f} "
                        f"peak_gb={record['peak_memory_gb']:.2f}",
                        flush=True,
                    )
                    metrics_handle.write(json.dumps(record, sort_keys=True) + "\n")
                    metrics_handle.flush()
                    running = {}
                    running_microsteps = 0

                if (
                    eval_loader is not None
                    and args.eval_every > 0
                    and global_step % args.eval_every == 0
                ):
                    eval_record = evaluate(model, eval_loader, device, args)
                    eval_record.update(
                        {"kind": "eval", "step": global_step, "epoch": epoch}
                    )
                    print(
                        f"eval step={global_step} "
                        f"loss={eval_record['eval_loss']:.4f} "
                        f"prefix={eval_record['eval_greedy_prefix_acceptance']:.3f} "
                        f"first={eval_record['eval_first_token_accuracy']:.3f}",
                        flush=True,
                    )
                    metrics_handle.write(
                        json.dumps(eval_record, sort_keys=True) + "\n"
                    )
                    metrics_handle.flush()
                    acceptance = eval_record["eval_greedy_prefix_acceptance"]
                    if acceptance > best_acceptance:
                        best_acceptance = acceptance
                        best_step = global_step
                        stale_evals = 0
                        save_trainable_checkpoint(
                            model, output_dir / "best", args.base_model
                        )
                        write_json(
                            output_dir / "best_metrics.json",
                            {
                                "best_step": best_step,
                                "eval_greedy_prefix_acceptance": best_acceptance,
                                **eval_record,
                            },
                        )
                    else:
                        stale_evals += 1
                    if (
                        args.early_stopping_patience > 0
                        and stale_evals >= args.early_stopping_patience
                    ):
                        stopped_early = True

                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_trainable_checkpoint(
                        model, output_dir / "last", args.base_model
                    )
                    write_json(
                        output_dir / "last_metrics.json",
                        {
                            "last_step": global_step,
                            "best_step": best_step,
                            "best_eval_greedy_prefix_acceptance": (
                                best_acceptance if best_step is not None else None
                            ),
                        },
                    )

                if stopped_early or global_step >= total_steps:
                    break
            if stopped_early or global_step >= total_steps:
                break
        progress.close()

        if args.save_final:
            save_trainable_checkpoint(model, output_dir / "last", args.base_model)
            write_json(
                output_dir / "last_metrics.json",
                {
                    "last_step": global_step,
                    "best_step": best_step,
                    "best_eval_greedy_prefix_acceptance": (
                        best_acceptance if best_step is not None else None
                    ),
                    "stopped_early": stopped_early,
                },
            )
        manifest.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "global_step": global_step,
                "best_step": best_step,
                "stopped_early": stopped_early,
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
    finally:
        metrics_handle.close()


if __name__ == "__main__":
    main()
