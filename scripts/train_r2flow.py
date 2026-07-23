#!/usr/bin/env python
"""Train a residual-refined Flow Map against a frozen AR fixed-point oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
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
from orthrus_training.flowdraft import (
    condition_flowdraft_state,
    make_endpoint_blocks,
    make_flowdraft_batch,
    make_flowdraft_inputs_embeds,
    sample_categorical_source_tokens,
    select_endpoint_logits,
)
from orthrus_training.losses import prefix_acceptance_metrics, prefix_survival_cross_entropy
from orthrus_training.modeling import dtype_from_string, load_flowdraft_adapter
from orthrus_training.residual_flow import ResidualFlowCorrector, corrector_logits
from orthrus_training.modeling import parallel_verifier_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Train R2Flow as a verifier-residual fixed-point corrector.")
    parser.add_argument("--config")
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--init-checkpoint", required=True, help="Frozen one-step FlowDraft adapter checkpoint.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-anchor-blocks", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--prefix-loss-weight", type=float, default=1.0)
    parser.add_argument("--prefix-weight-decay", type=float, default=0.88)
    parser.add_argument("--bottleneck-size", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-anchor-blocks", type=int, default=16)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--source-seed", type=int, default=4242)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    cli_keys = {
        action.dest
        for action in parser._actions
        if action.dest != "help" and any(option in sys.argv[1:] for option in action.option_strings)
    }
    return args, cli_keys


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_config(args, values: dict, cli_keys: set[str]):
    for key, value in values.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr) and attr not in cli_keys:
            setattr(args, attr, value)
    return args


def sha256_file(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def checkpoint_config(args, corrector: ResidualFlowCorrector, parent_sha256: str | None) -> dict:
    return {
        "format": "r2flow_residual_corrector_v1",
        "method": "residual_refined_categorical_flow_map",
        "objective": "J_squared_fixed_point_distillation_plus_prefix_survival",
        "base_flowdraft_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "base_flowdraft_adapter_sha256": parent_sha256,
        "block_size": corrector.block_size,
        "hidden_size": corrector.hidden_size,
        "bottleneck_size": corrector.bottleneck_size,
        "num_layers": corrector.num_layers,
        "num_heads": corrector.num_heads,
        "target_operator": "J(J(y0)) where J is the frozen parallel AR verifier",
        "lossless_decoding": "A separate final AR verifier still emits only its matching prefix.",
    }


def save_corrector_checkpoint(
    corrector: ResidualFlowCorrector,
    output_dir: Path,
    config: dict,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    backup_dir = output_dir.with_name(f".{output_dir.name}.previous")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    tmp_dir.mkdir(parents=True)
    state = {name: value.detach().cpu().contiguous() for name, value in corrector.state_dict().items()}
    save_file(state, tmp_dir / "r2flow_corrector.safetensors")
    write_json(tmp_dir / "r2flow_config.json", config)
    if output_dir.exists():
        output_dir.rename(backup_dir)
    try:
        tmp_dir.rename(output_dir)
    except Exception:
        if backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


@torch.no_grad()
def collect_fixed_point_targets(
    model,
    input_ids: torch.Tensor,
    num_anchor_blocks: int,
    source_generator: torch.Generator,
):
    """Collect y0 -> J(y0) -> J^2(y0) without differentiating through Qwen."""

    block_size = int(model.config.block_size)
    anchors = sample_anchor_positions(
        batch_size=input_ids.shape[0],
        seq_len=input_ids.shape[1],
        block_size=block_size,
        num_blocks=num_anchor_blocks,
        device=input_ids.device,
    )
    clean_blocks, position_ids, causal_limit, _, _ = make_flowdraft_batch(input_ids, anchors, block_size)
    source_blocks = sample_categorical_source_tokens(
        clean_blocks,
        vocab_size=int(model.config.vocab_size),
        mask_token_id=int(model.config.mask_token_id),
        prior=str(getattr(model.config, "flowdraft_source_prior", "uniform")),
        generator=source_generator,
    )
    source_time = torch.zeros((input_ids.shape[0], num_anchor_blocks, 1, 1), device=input_ids.device)
    target_time = torch.ones_like(source_time)
    flow_inputs = make_flowdraft_inputs_embeds(
        model=model,
        clean_blocks=clean_blocks,
        mask_token_id=int(model.config.mask_token_id),
        state_mix=source_time,
        source_token_ids=source_blocks,
    )
    flow_inputs = condition_flowdraft_state(
        model,
        flow_inputs,
        source_time,
        target_time,
        block_size=block_size,
        scale=float(getattr(model.config, "flowdraft_time_conditioning_scale", 0.0)),
    )

    context = model(input_ids=input_ids, use_cache=True, is_diffusion_pass=False)
    draft_outputs = model(
        inputs_embeds=flow_inputs,
        position_ids=position_ids,
        past_key_values=context.past_key_values,
        use_cache=False,
        is_diffusion_pass=True,
        causal_limit=causal_limit,
        ar_seq_len=input_ids.shape[1],
    )
    draft_logits = select_endpoint_logits(draft_outputs.logits, block_size)
    y0 = draft_logits.argmax(dim=-1).reshape(input_ids.shape[0], num_anchor_blocks, block_size - 1)
    proposed_blocks = make_endpoint_blocks(clean_blocks[:, :, :1], y0)

    verifier0_logits, verifier0_hidden = parallel_verifier_outputs(
        model=model,
        proposed_blocks=proposed_blocks,
        position_ids=position_ids,
        causal_limit=causal_limit,
        past_key_values=context.past_key_values,
        ar_seq_len=input_ids.shape[1],
        output_hidden_states=True,
    )
    if verifier0_hidden is None:
        raise RuntimeError("R2Flow requires final verifier hidden states")
    y1 = verifier0_logits.argmax(dim=-1).reshape_as(y0)
    verifier1_logits, _ = parallel_verifier_outputs(
        model=model,
        proposed_blocks=make_endpoint_blocks(clean_blocks[:, :, :1], y1),
        position_ids=position_ids,
        causal_limit=causal_limit,
        past_key_values=context.past_key_values,
        ar_seq_len=input_ids.shape[1],
        output_hidden_states=False,
    )
    y2 = verifier1_logits.argmax(dim=-1)
    return {
        "draft_logits": draft_logits,
        "y0": y0.reshape(input_ids.shape[0], -1),
        "y1": y1.reshape(input_ids.shape[0], -1),
        "y2": y2,
        "anchor_tokens": clean_blocks[:, :, :1],
        "position_ids": position_ids,
        "causal_limit": causal_limit,
        "past_key_values": context.past_key_values,
        "ar_seq_len": input_ids.shape[1],
        "verifier0_logits": verifier0_logits,
        "verifier0_hidden": verifier0_hidden,
    }


def correct_logits(model, corrector: ResidualFlowCorrector, batch: dict) -> torch.Tensor:
    candidate_embeddings = model.model.embed_tokens(batch["y0"]).detach()
    residual_embeddings = model.model.embed_tokens(batch["y1"]).detach()
    return corrector_logits(
        corrector=corrector,
        lm_head=model.lm_head,
        verifier_hidden=batch["verifier0_hidden"].detach(),
        candidate_embeddings=candidate_embeddings,
        residual_embeddings=residual_embeddings,
        verifier_logits=batch["verifier0_logits"].detach(),
    )


@torch.no_grad()
def evaluate(
    model,
    corrector: ResidualFlowCorrector,
    dataloader: DataLoader,
    args,
    source_generator: torch.Generator,
) -> dict:
    corrector.eval()
    baseline_acceptance = []
    corrected_acceptance = []
    corrected_first_token_acc = []
    target_prefix_acceptance = []
    for index, batch in enumerate(dataloader):
        if index >= args.eval_batches:
            break
        collected = collect_fixed_point_targets(
            model, batch.to(device="cuda", non_blocking=True), args.eval_anchor_blocks, source_generator
        )
        logits = correct_logits(model, corrector, collected)
        baseline = prefix_acceptance_metrics(
            collected["draft_logits"], collected["y1"], block_size=int(model.config.block_size)
        )
        target = prefix_acceptance_metrics(logits, collected["y2"], block_size=int(model.config.block_size))
        corrected_tokens = logits.argmax(dim=-1)
        corrected_blocks = make_endpoint_blocks(
            collected["anchor_tokens"],
            corrected_tokens.reshape(batch.shape[0], args.eval_anchor_blocks, -1),
        )
        final_logits, _ = parallel_verifier_outputs(
            model=model,
            proposed_blocks=corrected_blocks,
            position_ids=collected["position_ids"],
            causal_limit=collected["causal_limit"],
            past_key_values=collected["past_key_values"],
            ar_seq_len=int(collected["ar_seq_len"]),
            output_hidden_states=False,
        )
        final_targets = final_logits.argmax(dim=-1)
        final = prefix_acceptance_metrics(logits, final_targets, block_size=int(model.config.block_size))
        baseline_acceptance.append(float(baseline["greedy_prefix_acceptance"].cpu()))
        target_prefix_acceptance.append(float(target["greedy_prefix_acceptance"].cpu()))
        corrected_first_token_acc.append(float(final["first_token_acc"].cpu()))
        corrected_acceptance.append(float(final["greedy_prefix_acceptance"].cpu()))
    corrector.train()
    return {
        "eval_baseline_greedy_prefix_acceptance": sum(baseline_acceptance) / len(baseline_acceptance),
        "eval_corrected_greedy_prefix_acceptance": sum(corrected_acceptance) / len(corrected_acceptance),
        "eval_target_j2_prefix_acceptance": sum(target_prefix_acceptance) / len(target_prefix_acceptance),
        "eval_corrected_first_token_acc": sum(corrected_first_token_acc) / len(corrected_first_token_acc),
        "eval_batches": len(corrected_acceptance),
    }


def main() -> None:
    parsed, cli_keys = parse_args()
    config_values = load_config(parsed.config)
    args = merge_config(parsed, config_values, cli_keys)
    if not torch.cuda.is_available():
        raise RuntimeError("R2Flow training requires CUDA")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        existing = {path.name for path in output_dir.iterdir()}
        # The launcher opens its durable stdout log before invoking Python.  It
        # is safe to retain that single file; any training artifact means this
        # is a real prior run and must never be overwritten.
        if existing - {"run.log"}:
            raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_unique, eval_unique = assert_disjoint_packed_manifests(args.train_manifest, args.eval_manifest)
    dtype = dtype_from_string(args.dtype)
    model, parent_metadata, _ = load_flowdraft_adapter(
        checkpoint_dir=args.init_checkpoint,
        upstream_dir=args.upstream_dir,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.to(device="cuda", dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    corrector = ResidualFlowCorrector(
        hidden_size=int(model.config.hidden_size),
        block_size=int(model.config.block_size),
        bottleneck_size=args.bottleneck_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device="cuda", dtype=dtype)
    corrector.train()
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(args.max_steps * args.warmup_ratio)),
        num_training_steps=args.max_steps,
    )
    train_loader = DataLoader(
        PackedTokenDataset(args.train_manifest), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        PackedTokenDataset(args.eval_manifest), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    parent_sha256 = sha256_file(Path(args.init_checkpoint) / "adapter_model.safetensors")
    ckpt_config = checkpoint_config(args, corrector, parent_sha256)
    run_config = vars(args) | {
        "method": ckpt_config["method"],
        "objective": ckpt_config["objective"],
        "parent_adapter_metadata": parent_metadata,
        "train_unique_sequences": train_unique,
        "eval_unique_sequences": eval_unique,
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "run_manifest.json", {
        "status": "running", "run_config": run_config, "python": sys.version,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    })
    print(
        f"r2flow parameters trainable={sum(p.numel() for p in corrector.parameters()):,} "
        f"parent={args.init_checkpoint} train_unique={train_unique} eval_unique={eval_unique}", flush=True
    )

    source_generator = torch.Generator(device="cuda").manual_seed(args.source_seed)
    eval_generator = torch.Generator(device="cuda").manual_seed(args.source_seed + 1)
    metrics_file = (output_dir / "train_metrics.jsonl").open("w", encoding="utf-8")
    best_metric = float("-inf")
    best_step = None
    stale_evaluations = 0
    global_step = 0
    accumulation_step = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, desc="r2flow optimizer steps")
    try:
        while global_step < args.max_steps:
            for raw_batch in train_loader:
                collected = collect_fixed_point_targets(
                    model, raw_batch.to(device="cuda", non_blocking=True), args.num_anchor_blocks, source_generator
                )
                logits = correct_logits(model, corrector, collected)
                hard_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), collected["y2"].reshape(-1))
                prefix_loss = prefix_survival_cross_entropy(
                    logits, collected["y2"], block_size=int(model.config.block_size), decay=args.prefix_weight_decay
                )
                loss = (hard_ce + args.prefix_loss_weight * prefix_loss) / args.gradient_accumulation_steps
                loss.backward()
                accumulation_step += 1
                if accumulation_step < args.gradient_accumulation_steps:
                    continue
                accumulation_step = 0
                torch.nn.utils.clip_grad_norm_(corrector.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)
                metrics = prefix_acceptance_metrics(logits.detach(), collected["y2"], int(model.config.block_size))
                baseline = prefix_acceptance_metrics(
                    collected["draft_logits"], collected["y1"], int(model.config.block_size)
                )
                record = {
                    "step": global_step,
                    "loss": float(loss.detach().cpu()) * args.gradient_accumulation_steps,
                    "hard_ce": float(hard_ce.detach().cpu()),
                    "prefix_loss": float(prefix_loss.detach().cpu()),
                    "j0_to_j1_prefix": float(baseline["greedy_prefix_acceptance"].cpu()),
                    "corrector_to_j2_prefix": float(metrics["greedy_prefix_acceptance"].cpu()),
                    "corrector_first_token_acc": float(metrics["first_token_acc"].cpu()),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                }
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                if global_step % args.log_every == 0:
                    print("TRAIN " + json.dumps(record), flush=True)
                if global_step % args.eval_every == 0 or global_step == args.max_steps:
                    evaluation = evaluate(model, corrector, eval_loader, args, eval_generator)
                    evaluation["step"] = global_step
                    metrics_file.write(json.dumps(evaluation) + "\n")
                    metrics_file.flush()
                    print("EVAL " + json.dumps(evaluation), flush=True)
                    metric = evaluation["eval_corrected_greedy_prefix_acceptance"]
                    if metric > best_metric:
                        best_metric = metric
                        best_step = global_step
                        stale_evaluations = 0
                        save_corrector_checkpoint(corrector, output_dir / "best", ckpt_config | {"best_step": best_step, "best_metric": best_metric})
                    else:
                        stale_evaluations += 1
                    if args.early_stopping_patience and stale_evaluations >= args.early_stopping_patience:
                        print(f"early stopping at step={global_step}", flush=True)
                        global_step = args.max_steps
                        break
                if global_step >= args.max_steps:
                    break
            if global_step >= args.max_steps:
                break
    finally:
        progress.close()
        metrics_file.close()
    save_corrector_checkpoint(corrector, output_dir / "last", ckpt_config | {"last_step": global_step})
    write_json(output_dir / "best_metrics.json", {"best_step": best_step, "best_metric": best_metric})
    write_json(output_dir / "last_metrics.json", {"last_step": global_step, "seconds": time.time()})
    write_json(output_dir / "run_manifest.json", {
        "status": "completed", "best_step": best_step, "best_metric": best_metric,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "run_config": run_config,
    })
    print(f"R2FLOW_COMPLETE best_step={best_step} best_metric={best_metric:.6f}", flush=True)


if __name__ == "__main__":
    main()
