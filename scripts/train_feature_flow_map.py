#!/usr/bin/env python
"""Train a block-parallel feature flow map with a frozen lossless verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from orthrus_training.data import PackedTokenDataset, assert_disjoint_packed_manifests, sample_anchor_positions
from orthrus_training.feature_flow_map import FeatureFlowMapHead, feature_flow_interpolate, feature_flow_source
from orthrus_training.losses import forward_kl_distillation, gather_logits, prefix_acceptance_metrics, prefix_survival_cross_entropy
from orthrus_training.modeling import dtype_from_string, load_flowdraft_adapter


def parse_args():
    parser = argparse.ArgumentParser(description="Train a cached-feature categorical flow map.")
    parser.add_argument("--config")
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-anchor-blocks", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--root-hidden-weight", type=float, default=0.35)
    parser.add_argument("--diagonal-hidden-weight", type=float, default=0.35)
    parser.add_argument("--teacher-kl-weight", type=float, default=0.35)
    parser.add_argument("--prefix-loss-weight", type=float, default=1.5)
    parser.add_argument("--consistency-weight", type=float, default=0.15)
    parser.add_argument("--prefix-weight-decay", type=float, default=0.90)
    parser.add_argument("--latent-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-anchor-blocks", type=int, default=64)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--source-seed", type=int, default=901)
    parser.add_argument("--source-mode", choices=("gaussian", "context"), default="gaussian")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    explicit = {a.dest for a in parser._actions if a.dest != "help" and any(o in sys.argv[1:] for o in a.option_strings)}
    return args, explicit


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_config(args, values: dict, explicit: set[str]):
    for key, value in values.items():
        name = key.replace("-", "_")
        if hasattr(args, name) and name not in explicit:
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


def save_checkpoint(head: FeatureFlowMapHead, directory: Path, config: dict) -> None:
    """Atomically retain a single complete best or last head checkpoint."""

    temporary = directory.with_name(f".{directory.name}.tmp")
    previous = directory.with_name(f".{directory.name}.previous")
    for candidate in (temporary, previous):
        if candidate.exists():
            shutil.rmtree(candidate)
    temporary.mkdir(parents=True)
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in head.state_dict().items()},
        temporary / "feature_flow_map.safetensors",
    )
    write_json(temporary / "feature_flow_map_config.json", config)
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
def collect_teacher(model, input_ids: torch.Tensor, num_blocks: int) -> dict[str, torch.Tensor]:
    """Gather teacher-forced AR feature paths at independent valid anchors."""

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
    return {
        "context_hidden": context_hidden,
        "anchor_tokens": anchor_tokens,
        "target_hidden": target_hidden,
        "teacher_logits": teacher_logits,
    }


def normalised_feature_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rms = target.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    return ((prediction.float() - target.float()) / rms).square().mean()


def root_prediction(head, model, teacher: dict[str, torch.Tensor], source_generator: torch.Generator, source_mode: str):
    source = feature_flow_source(teacher["context_hidden"], head.prediction_length, source_generator, mode=source_mode)
    time = torch.zeros(teacher["context_hidden"].shape[:2], device=source.device, dtype=source.dtype)
    endpoint = head(teacher["context_hidden"], model.model.embed_tokens(teacher["anchor_tokens"]).detach(), source, time)
    logits = model.lm_head(endpoint.reshape(-1, endpoint.shape[-1])).reshape(endpoint.shape[0], -1, int(model.config.vocab_size))
    return source, endpoint, logits


def training_losses(head, model, teacher, args, source_generator) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    source, root_endpoint, logits = root_prediction(head, model, teacher, source_generator, args.source_mode)
    target = teacher["target_hidden"]
    root_hidden = normalised_feature_mse(root_endpoint, target)
    teacher_kl = forward_kl_distillation(logits, teacher["teacher_logits"], reduction="tokenmean")
    prefix = prefix_survival_cross_entropy(logits, teacher["teacher_logits"].argmax(dim=-1), head.block_size, args.prefix_weight_decay)

    diagonal_time = torch.rand(target.shape[:2], device=target.device, generator=source_generator, dtype=target.dtype)
    diagonal_state = feature_flow_interpolate(source, target, diagonal_time)
    diagonal_endpoint = head(
        teacher["context_hidden"], model.model.embed_tokens(teacher["anchor_tokens"]).detach(), diagonal_state, diagonal_time
    )
    diagonal_hidden = normalised_feature_mse(diagonal_endpoint, target)

    # Endpoint consistency: a detached root endpoint should be recovered after
    # transporting the source part-way along the learned map.
    consistency_time = torch.rand(target.shape[:2], device=target.device, generator=source_generator, dtype=target.dtype) * 0.8 + 0.1
    transported = feature_flow_interpolate(source, root_endpoint.detach(), consistency_time)
    consistent_endpoint = head(
        teacher["context_hidden"], model.model.embed_tokens(teacher["anchor_tokens"]).detach(), transported, consistency_time
    )
    consistency = normalised_feature_mse(consistent_endpoint, root_endpoint.detach())
    loss = (
        args.root_hidden_weight * root_hidden
        + args.diagonal_hidden_weight * diagonal_hidden
        + args.teacher_kl_weight * teacher_kl
        + args.prefix_loss_weight * prefix
        + args.consistency_weight * consistency
    )
    return {
        "loss": loss, "root_hidden": root_hidden, "diagonal_hidden": diagonal_hidden,
        "teacher_kl": teacher_kl, "prefix_loss": prefix, "consistency": consistency, "logits": logits,
    }, root_endpoint


@torch.no_grad()
def evaluate(model, head, dataloader, args, source_generator) -> dict:
    head.eval()
    feature_losses, kls, prefixes, firsts = [], [], [], []
    for index, raw in enumerate(dataloader):
        if index >= args.eval_batches:
            break
        teacher = collect_teacher(model, raw.to(device="cuda", non_blocking=True), args.eval_anchor_blocks)
        _, endpoint, logits = root_prediction(head, model, teacher, source_generator, args.source_mode)
        feature_losses.append(float(normalised_feature_mse(endpoint, teacher["target_hidden"]).cpu()))
        kls.append(float(forward_kl_distillation(logits, teacher["teacher_logits"], reduction="tokenmean").cpu()))
        metrics = prefix_acceptance_metrics(logits, teacher["teacher_logits"].argmax(dim=-1), head.block_size)
        prefixes.append(float(metrics["greedy_prefix_acceptance"].cpu()))
        firsts.append(float(metrics["first_token_acc"].cpu()))
    head.train()
    return {
        "eval_feature_mse": sum(feature_losses) / len(feature_losses),
        "eval_teacher_kl": sum(kls) / len(kls),
        "eval_greedy_prefix_acceptance": sum(prefixes) / len(prefixes),
        "eval_first_token_acc": sum(firsts) / len(firsts),
        "eval_batches": len(prefixes),
    }


def main() -> None:
    parsed, explicit = parse_args()
    args = merge_config(parsed, load_config(parsed.config), explicit)
    if not torch.cuda.is_available():
        raise RuntimeError("FeatureFlowMap training requires CUDA")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and {path.name for path in output_dir.iterdir()} - {"run.log"}:
        raise FileExistsError(f"Refusing to overwrite existing experiment: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_unique, eval_unique = assert_disjoint_packed_manifests(args.train_manifest, args.eval_manifest)
    dtype = dtype_from_string(args.dtype)
    model, parent_metadata, _ = load_flowdraft_adapter(args.init_checkpoint, args.upstream_dir, dtype, args.attn_implementation)
    model.to(device="cuda", dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = FeatureFlowMapHead(
        hidden_size=int(model.config.hidden_size), block_size=int(model.config.block_size), latent_size=args.latent_size,
        num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
    ).to(device="cuda", dtype=dtype).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(args.max_steps * args.warmup_ratio)), args.max_steps)
    train_loader = DataLoader(PackedTokenDataset(args.train_manifest), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(PackedTokenDataset(args.eval_manifest), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    checkpoint_config = {
        "format": FeatureFlowMapHead.format_name,
        "method": "conditional_endpoint_categorical_feature_flow_map",
        "base_flowdraft_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "base_flowdraft_adapter_sha256": sha256_file(Path(args.init_checkpoint) / "adapter_model.safetensors"),
        "block_size": head.block_size, "hidden_size": head.hidden_size, "latent_size": head.latent_size,
        "num_layers": head.num_layers, "num_heads": head.num_heads,
        "source_mode": args.source_mode,
        "inference": "block-parallel feature flow map followed by exactly one ordinary frozen AR verifier pass",
    }
    run_config = vars(args) | {"method": checkpoint_config["method"], "parent_adapter_metadata": parent_metadata, "train_unique_sequences": train_unique, "eval_unique_sequences": eval_unique, "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "run_manifest.json", {"status": "running", "run_config": run_config, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
    print(f"feature_flow_map trainable={sum(p.numel() for p in head.parameters()):,} train_unique={train_unique} eval_unique={eval_unique}", flush=True)
    source_generator = torch.Generator(device="cuda").manual_seed(args.source_seed)
    eval_generator = torch.Generator(device="cuda").manual_seed(args.source_seed + 1)
    best_metric, best_step, stale, step = float("-inf"), None, 0, 0
    metrics_file = (output_dir / "train_metrics.jsonl").open("w", encoding="utf-8")
    progress = tqdm(total=args.max_steps, desc="feature-flow optimizer steps")
    try:
        for raw in train_loader:
            teacher = collect_teacher(model, raw.to(device="cuda", non_blocking=True), args.num_anchor_blocks)
            losses, _ = training_losses(head, model, teacher, args, source_generator)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            step += 1; progress.update(1)
            metrics = prefix_acceptance_metrics(losses["logits"].detach(), teacher["teacher_logits"].argmax(dim=-1), head.block_size)
            record = {name: float(value.detach().cpu()) for name, value in losses.items() if name != "logits"}
            record |= {"step": step, "greedy_prefix": float(metrics["greedy_prefix_acceptance"].cpu()), "first_token_acc": float(metrics["first_token_acc"].cpu()), "lr": scheduler.get_last_lr()[0], "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
            metrics_file.write(json.dumps(record) + "\n"); metrics_file.flush()
            if step % args.log_every == 0:
                print("TRAIN " + json.dumps(record), flush=True)
            if step % args.eval_every == 0 or step == args.max_steps:
                evaluation = evaluate(model, head, eval_loader, args, eval_generator) | {"step": step}
                metrics_file.write(json.dumps(evaluation) + "\n"); metrics_file.flush(); print("EVAL " + json.dumps(evaluation), flush=True)
                metric = evaluation["eval_greedy_prefix_acceptance"]
                if metric > best_metric:
                    best_metric, best_step, stale = metric, step, 0
                    save_checkpoint(head, output_dir / "best", checkpoint_config | {"best_step": best_step, "best_metric": best_metric})
                else:
                    stale += 1
                if args.early_stopping_patience and stale >= args.early_stopping_patience:
                    print(f"Early stop at step={step}; best_step={best_step}", flush=True)
                    break
            if step >= args.max_steps:
                break
    finally:
        progress.close(); metrics_file.close()
    save_checkpoint(head, output_dir / "last", checkpoint_config | {"last_step": step})
    write_json(output_dir / "best_metrics.json", {"best_step": best_step, "best_metric": best_metric})
    write_json(output_dir / "last_metrics.json", {"last_step": step})
    write_json(output_dir / "run_manifest.json", {"status": "completed", "best_step": best_step, "best_metric": best_metric, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "run_config": run_config})
    print(f"FEATURE_FLOW_COMPLETE best_step={best_step} best_metric={best_metric:.6f}", flush=True)


if __name__ == "__main__":
    main()
