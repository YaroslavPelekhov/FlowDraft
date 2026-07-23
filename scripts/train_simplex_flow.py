#!/usr/bin/env python
"""Train a proper local-simplex Categorical Flow Map over frozen Orthrus candidates."""

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

from orthrus_training.candidate_support import RescueCandidateBank, select_candidate_support
from orthrus_training.data import PackedTokenDataset, assert_disjoint_packed_manifests
from orthrus_training.losses import prefix_survival_weights
from orthrus_training.modeling import dtype_from_string, load_flowdraft_adapter
from orthrus_training.simplex_flow import SimplexFlowRefiner, local_simplex_path, simplex_flow_step
from orthrus_training.simplex_flow_data import collect_base_draft_logits


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-anchor-blocks", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=128)
    parser.add_argument(
        "--rescue-bank",
        default=None,
        help="Optional train-derived rescue candidate bank; copied into best and last checkpoints.",
    )
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--root-ce-weight", type=float, default=1.0)
    parser.add_argument("--diagonal-ce-weight", type=float, default=0.5)
    parser.add_argument("--ecld-weight", type=float, default=0.25)
    parser.add_argument("--temporal-weight", type=float, default=0.02)
    parser.add_argument("--prefix-weight-decay", type=float, default=0.9)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-anchor-blocks", type=int, default=32)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=311)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    cli = {action.dest for action in parser._actions if action.dest != "help" and any(flag in sys.argv[1:] for flag in action.option_strings)}
    return args, cli


def read_config(path):
    if not path:
        return {}
    import yaml
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge(args, values, cli):
    for key, value in values.items():
        key = key.replace("-", "_")
        if hasattr(args, key) and key not in cli:
            setattr(args, key, value)
    return args


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    head: SimplexFlowRefiner,
    directory: Path,
    config: dict,
    rescue_bank: RescueCandidateBank | None = None,
) -> None:
    temporary = directory.with_name(f".{directory.name}.tmp")
    previous = directory.with_name(f".{directory.name}.previous")
    for candidate in (temporary, previous):
        if candidate.exists():
            shutil.rmtree(candidate)
    temporary.mkdir(parents=True)
    save_file({name: value.detach().cpu().contiguous() for name, value in head.state_dict().items()}, temporary / "simplex_flow.safetensors")
    if rescue_bank is not None:
        rescue_bank.save(temporary)
    write_json(temporary / "simplex_flow_config.json", config)
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
def collect_candidates(
    model,
    input_ids: torch.Tensor,
    num_blocks: int,
    candidate_count: int,
    generator: torch.Generator,
    rescue_bank: RescueCandidateBank | None = None,
):
    logits, teacher = collect_base_draft_logits(model, input_ids, num_blocks, generator)
    values, ids = select_candidate_support(logits, candidate_count, rescue_bank)
    matches = ids.eq(teacher.unsqueeze(-1))
    covered = matches.any(dim=-1)
    local_target = matches.to(torch.int64).argmax(dim=-1)
    return values, ids, local_target, covered


def weighted_masked_ce(probabilities, targets, covered, prefix_weights):
    nll = -probabilities.float().clamp_min(1e-8).log().gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    weights = prefix_weights.view(1, 1, -1) * covered.float()
    return (nll * weights).sum() / weights.sum().clamp_min(1.0)


def sample_times(shape, device):
    # CFM's logit-normal schedule, clipped away from the singular endpoint.
    diagonal = torch.sigmoid(torch.randn(shape, device=device) - 0.4).clamp(0.02, 0.95)
    source = torch.sigmoid(torch.randn(shape, device=device) - 0.4).clamp(0.0, 0.90)
    target = source + (1.0 - source) * torch.sigmoid(torch.randn(shape, device=device) - 0.4)
    return diagonal, source, target.clamp_max(0.98)


def flow_losses(head, values, targets, covered, prefix_weights, args):
    shape = values.shape[:-1]
    candidates = values.shape[-1]
    uniform = torch.full_like(values, 1.0 / candidates)
    root_source = torch.zeros(shape, device=values.device)
    root_target = torch.ones_like(root_source)
    root = head(values, uniform, root_source, root_target)
    root_ce = weighted_masked_ce(root, targets, covered, prefix_weights)

    diagonal_t, source_t, target_t = sample_times(shape, values.device)
    diagonal_state = local_simplex_path(targets, diagonal_t, candidates).to(dtype=values.dtype)
    diagonal = head(values, diagonal_state, diagonal_t, diagonal_t)
    diagonal_ce = weighted_masked_ce(diagonal, targets, covered, prefix_weights)

    source_state = local_simplex_path(targets, source_t, candidates).to(dtype=values.dtype)
    endpoint = head(values, source_state, source_t, target_t)
    transported = simplex_flow_step(source_state, endpoint, source_t, target_t).detach()
    with torch.no_grad():
        target_endpoint = head(values, transported, target_t, target_t)
    ecld = -(target_endpoint.float() * endpoint.float().clamp_min(1e-8).log()).sum(dim=-1)
    ecld = (ecld * covered.float()).sum() / covered.float().sum().clamp_min(1.0)

    eps = 0.02
    next_t = (target_t + eps).clamp_max(0.99)
    next_endpoint = head(values, source_state, source_t, next_t)
    temporal = ((next_endpoint.float() - endpoint.float()) / (next_t - target_t).unsqueeze(-1).clamp_min(1e-3)).square().sum(dim=-1)
    temporal = (temporal * covered.float()).sum() / covered.float().sum().clamp_min(1.0)
    loss = args.root_ce_weight * root_ce + args.diagonal_ce_weight * diagonal_ce + args.ecld_weight * ecld + args.temporal_weight * temporal
    return loss, {"root_ce": root_ce, "diagonal_ce": diagonal_ce, "ecld": ecld, "temporal": temporal, "root": root}


@torch.no_grad()
def evaluate(model, head, loader, args, generator, rescue_bank=None):
    head.eval()
    prefixes, firsts, coverages = [], [], []
    weights = prefix_survival_weights(int(model.config.block_size), args.prefix_weight_decay, torch.device("cuda"))
    for index, raw in enumerate(loader):
        if index >= args.eval_batches:
            break
        values, _, targets, covered = collect_candidates(
            model, raw.to("cuda", non_blocking=True), args.eval_anchor_blocks,
            args.candidate_count, generator, rescue_bank,
        )
        time = torch.zeros(values.shape[:-1], device=values.device)
        proposal = head(values, torch.full_like(values, 1.0 / values.shape[-1]), time, torch.ones_like(time)).argmax(dim=-1)
        correct = proposal.eq(targets) & covered
        prefix = correct.to(torch.int64).cumprod(dim=-1).sum(dim=-1).float()
        prefixes.append(float(prefix.mean().cpu()))
        firsts.append(float(correct[:, :, 0].float().mean().cpu()))
        coverages.append(float(covered.float().mean().cpu()))
    head.train()
    return {"eval_greedy_prefix_acceptance": sum(prefixes) / max(1, len(prefixes)), "eval_first_token_acc": sum(firsts) / max(1, len(firsts)), "eval_candidate_coverage": sum(coverages) / max(1, len(coverages)), "eval_batches": len(prefixes)}


def main():
    parsed, cli = parse_args()
    args = merge(parsed, read_config(parsed.config), cli)
    if not torch.cuda.is_available():
        raise RuntimeError("SimplexFlow requires CUDA")
    set_seed(args.seed)
    output = Path(args.output_dir)
    if output.exists() and {item.name for item in output.iterdir()} - {"run.log"}:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_unique, eval_unique = assert_disjoint_packed_manifests(args.train_manifest, args.eval_manifest)
    dtype = dtype_from_string(args.dtype)
    model, parent_metadata, _ = load_flowdraft_adapter(args.init_checkpoint, args.upstream_dir, dtype, args.attn_implementation)
    # Orthrus selects its block-parallel diffusion mask from training mode.
    # Parameters remain frozen and every backbone call below is no-grad.
    model.to("cuda", dtype=dtype).train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rescue_bank = RescueCandidateBank.load(args.rescue_bank, device="cuda") if args.rescue_bank else None
    if rescue_bank is not None and rescue_bank.base_candidate_count >= args.candidate_count:
        raise ValueError("rescue bank requires candidate_count greater than its base_candidate_count")
    head = SimplexFlowRefiner(int(model.config.block_size), args.candidate_count, args.hidden_size, args.num_layers, args.num_heads, args.dropout).to("cuda", dtype=dtype).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, round(args.max_steps * args.warmup_ratio)), args.max_steps)
    train_loader = DataLoader(PackedTokenDataset(args.train_manifest), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(PackedTokenDataset(args.eval_manifest), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    config = vars(args) | {
        "format": "simplex_categorical_flow_map_v1", "method": "local_simplex_endpoint_cfm_refiner",
        "objective": "root_endpoint_ce + diagonal_vfm_ce + ECLD + temporal_drift",
        "base_flowdraft_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "base_flowdraft_adapter_sha256": sha256(Path(args.init_checkpoint) / "adapter_model.safetensors"),
        "block_size": int(model.config.block_size), "candidate_count": args.candidate_count,
        "rescue_bank": str(Path(args.rescue_bank).resolve()) if args.rescue_bank else None,
        "candidate_support": "parent_topk_only" if rescue_bank is None else "parent_topk_plus_train_derived_rescue_tokens",
        "inference": "one frozen Orthrus diffusion pass, cheap local-simplex CFM steps, one frozen AR verifier; strict greedy verifier preserves parity",
        "parent_adapter_metadata": parent_metadata, "train_unique_sequences": train_unique, "eval_unique_sequences": eval_unique,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "run_config.json", config)
    write_json(output / "run_manifest.json", {"status": "running", "config": config, "gpu": torch.cuda.get_device_name(0)})
    print(f"simplex_flow trainable={sum(p.numel() for p in head.parameters()):,} train_unique={train_unique} eval_unique={eval_unique}", flush=True)
    weights = prefix_survival_weights(int(model.config.block_size), args.prefix_weight_decay, torch.device("cuda"))
    train_generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    eval_generator = torch.Generator(device="cuda").manual_seed(args.seed + 2)
    best, best_step, stale, step = float("-inf"), None, 0, 0
    metrics_file = (output / "train_metrics.jsonl").open("w", encoding="utf-8")
    progress = tqdm(total=args.max_steps, desc="simplex-cfm optimizer steps")
    try:
        for raw in train_loader:
            step += 1
            values, _, targets, covered = collect_candidates(
                model, raw.to("cuda", non_blocking=True), args.num_anchor_blocks,
                args.candidate_count, train_generator, rescue_bank,
            )
            loss, terms = flow_losses(head, values, targets, covered, weights, args)
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); progress.update(1)
            root_pred = terms["root"].detach().argmax(dim=-1)
            record = {"step": step, "loss": float(loss.detach().cpu()), "root_ce": float(terms["root_ce"].detach().cpu()), "diagonal_ce": float(terms["diagonal_ce"].detach().cpu()), "ecld": float(terms["ecld"].detach().cpu()), "temporal": float(terms["temporal"].detach().cpu()), "candidate_coverage": float(covered.float().mean().cpu()), "root_first_acc": float(((root_pred[:, :, 0] == targets[:, :, 0]) & covered[:, :, 0]).float().mean().cpu()), "lr": scheduler.get_last_lr()[0], "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
            metrics_file.write(json.dumps(record) + "\n"); metrics_file.flush()
            if step % args.log_every == 0:
                print("TRAIN " + json.dumps(record), flush=True)
            if step % args.eval_every == 0 or step == args.max_steps:
                evaluation = evaluate(model, head, eval_loader, args, eval_generator, rescue_bank) | {"step": step}
                metrics_file.write(json.dumps(evaluation) + "\n"); metrics_file.flush(); print("EVAL " + json.dumps(evaluation), flush=True)
                metric = evaluation["eval_greedy_prefix_acceptance"]
                if metric > best:
                    best, best_step, stale = metric, step, 0
                    save_checkpoint(
                        head, output / "best", config | {"best_step": best_step, "best_metric": best}, rescue_bank
                    )
                else:
                    stale += 1
                if args.early_stopping_patience and stale >= args.early_stopping_patience:
                    print(f"Early stop at step {step}", flush=True); break
            if step >= args.max_steps:
                break
    finally:
        progress.close(); metrics_file.close()
    save_checkpoint(head, output / "last", config | {"last_step": step}, rescue_bank)
    manifest = json.loads((output / "run_manifest.json").read_text())
    manifest.update({"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "best_step": best_step, "best_metric": best, "last_step": step})
    write_json(output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
