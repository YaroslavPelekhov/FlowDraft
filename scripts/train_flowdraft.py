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

from orthrus_training.checkpointing import (
    save_orthrus_checkpoint,
    save_trainable_checkpoint,
    save_training_state,
)
from orthrus_training.data import (
    PackedTokenDataset,
    assert_disjoint_packed_manifests,
    sample_anchor_positions,
)
from orthrus_training.flowdraft import (
    condition_flowdraft_state,
    exact_endpoint_embeddings,
    flow_map_step_size,
    make_flowdraft_batch,
    make_flowdraft_inputs_embeds,
    sample_categorical_source_tokens,
    sample_cfm_time_pairs,
    sample_flow_state_mix,
    select_endpoint_logits,
    topk_endpoint_embeddings,
    transport_categorical_state,
)
from orthrus_training.losses import (
    forward_kl_distillation,
    gather_logits,
    prefix_acceptance_metrics,
    prefix_survival_cross_entropy,
    prefix_survival_weights,
    token_accuracy,
)
from orthrus_training.modeling import (
    build_orthrus_from_qwen,
    count_parameters,
    dtype_from_string,
    load_trainable_initialization,
    load_tokenizer,
    set_flowdraft_state_adapter_trainable,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FlowDraft endpoint drafter inside Orthrus verifier.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train-manifest", default="data/packed_qwen3_1p7b/manifest.json")
    parser.add_argument("--eval-manifest", default=None)
    parser.add_argument("--allow-eval-overlap", action="store_true")
    parser.add_argument("--output-dir", default="/dev/shm/flowdraft_runs/flowdraft_quick2h")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional full or trainable checkpoint used to initialize the diffusion projections.",
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--mask-token-id", type=int, default=151669)
    parser.add_argument("--source-prior", choices=["uniform", "mask"], default="uniform")
    parser.add_argument("--source-seed", type=int, default=2718)
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
    parser.add_argument("--kl-reduction", choices=["batchmean", "tokenmean"], default="batchmean")
    parser.add_argument("--flow-state-min", type=float, default=0.0)
    parser.add_argument("--flow-state-max", type=float, default=0.5)
    parser.add_argument("--flow-objective", choices=["legacy", "ecld"], default="ecld")
    parser.add_argument("--diagonal-fraction", type=float, default=0.75)
    parser.add_argument("--flow-time-logit-mean", type=float, default=-0.4)
    parser.add_argument("--flow-time-logit-std", type=float, default=1.0)
    parser.add_argument("--flow-time-max", type=float, default=0.95)
    parser.add_argument("--flow-time-conditioning-scale", type=float, default=0.1)
    parser.add_argument("--endpoint-topk", type=int, default=32)
    parser.add_argument("--endpoint-transport", choices=["topk", "dense"], default="topk")
    parser.add_argument("--endpoint-vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--flow-state-adapter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flow-adapter-bottleneck", type=int, default=256)
    parser.add_argument("--one-jump-fraction", type=float, default=0.0)
    parser.add_argument(
        "--direct-endpoint-teacher-weight",
        type=float,
        default=1.0,
        help="Weight for frozen-AR KL on exact CFM (0 -> 1) proposal states.",
    )
    parser.add_argument("--ecld-time-weight", choices=["uniform", "paper_clamped"], default="uniform")
    parser.add_argument("--temporal-difference-epsilon", type=float, default=0.02)
    parser.add_argument("--temporal-drift-weight", type=float, default=1.0)
    parser.add_argument("--hard-ce-weight", type=float, default=0.5)
    parser.add_argument("--prefix-loss-weight", type=float, default=0.0)
    parser.add_argument("--prefix-kl-weight", type=float, default=0.0)
    parser.add_argument(
        "--direct-prefix-kl-weight",
        type=float,
        default=0.0,
        help="Prefix-survival-weighted soft AR KL on exact (0 -> 1) proposal states.",
    )
    parser.add_argument("--prefix-weight-decay", type=float, default=0.9)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--consistency-start-step", type=int, default=400)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-anchor-blocks", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=1701)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "Stop after this many consecutive holdout evaluations without an improvement "
            "in --best-metric. Set to 0 to disable early stopping."
        ),
    )
    parser.add_argument(
        "--best-metric",
        choices=["eval_greedy_prefix_acceptance", "eval_prefix_expected_acceptance", "eval_loss"],
        default="eval_greedy_prefix_acceptance",
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--checkpoint-format", choices=["trainable", "full"], default="trainable")
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-trainer-state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-missing-init-trainables", action=argparse.BooleanOptionalAction, default=False)
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
    flow_objective: str = "legacy",
    diagonal_fraction: float = 0.75,
    time_logit_mean: float = -0.4,
    time_logit_std: float = 1.0,
    time_max: float = 0.95,
    time_conditioning_scale: float = 0.1,
    one_jump_fraction: float = 0.0,
    source_prior: str = "uniform",
    source_generator: torch.Generator | None = None,
    source_time: torch.Tensor | None = None,
    target_time: torch.Tensor | None = None,
    force_one_jump: bool = False,
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
    source_blocks = sample_categorical_source_tokens(
        clean_blocks,
        vocab_size=int(model.config.vocab_size),
        mask_token_id=mask_token_id,
        prior=source_prior,
        generator=source_generator,
    )
    diagonal_mask = torch.ones(
        (input_ids.shape[0], anchors.shape[1]), dtype=torch.bool, device=device
    )
    if flow_objective == "ecld":
        if force_one_jump:
            shape = (input_ids.shape[0], anchors.shape[1], 1, 1)
            source_time = torch.zeros(shape, device=device)
            target_time = torch.ones(shape, device=device)
            diagonal_mask = torch.zeros(
                (input_ids.shape[0], anchors.shape[1]), dtype=torch.bool, device=device
            )
        elif source_time is None or target_time is None:
            source_time, target_time, diagonal_mask = sample_cfm_time_pairs(
                batch_size=input_ids.shape[0],
                num_blocks=anchors.shape[1],
                diagonal_fraction=diagonal_fraction,
                device=device,
                logit_mean=time_logit_mean,
                logit_std=time_logit_std,
                max_time=time_max,
                one_jump_fraction=one_jump_fraction,
            )
        else:
            diagonal_mask = torch.isclose(source_time, target_time).reshape(
                input_ids.shape[0], anchors.shape[1]
            )
        state_mix = source_time
    else:
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
        source_token_ids=source_blocks,
    )
    if flow_objective == "ecld":
        flow_inputs = condition_flowdraft_state(
            model,
            flow_inputs,
            source_time,
            target_time,
            block_size=block_size,
            scale=time_conditioning_scale,
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
    one_jump_mask = (
        ~diagonal_mask
        & torch.isclose(source_time.squeeze(-1).squeeze(-1), torch.zeros((), device=device))
        & torch.isclose(target_time.squeeze(-1).squeeze(-1), torch.ones((), device=device))
    ) if flow_objective == "ecld" else torch.zeros_like(diagonal_mask)
    return {
        "student_logits": student_logits,
        "teacher_logits": teacher_logits,
        "target_ids": target_ids,
        "clean_blocks": clean_blocks,
        "source_blocks": source_blocks,
        "position_ids": position_ids,
        "causal_limit": causal_limit,
        "past_key_values": ar_outputs.past_key_values,
        "source_time": source_time,
        "target_time": target_time,
        "diagonal_mask": diagonal_mask,
        "one_jump_mask": one_jump_mask,
    }


def select_block_values(values: torch.Tensor, block_mask: torch.Tensor, block_width: int) -> torch.Tensor:
    batch_size, num_blocks = block_mask.shape
    tail = values.shape[2:] if values.dim() > 2 else ()
    blocked = values.reshape(batch_size, num_blocks, block_width, *tail)
    selected_per_batch = block_mask.sum(dim=1)
    if not torch.all(selected_per_batch == selected_per_batch[0]):
        raise ValueError("Each batch item must select the same number of blocks")
    return blocked[block_mask].reshape(batch_size, int(selected_per_batch[0]), block_width, *tail)


def ecld_consistency_loss(
    model,
    first_logits: torch.Tensor,
    clean_blocks: torch.Tensor,
    source_blocks: torch.Tensor,
    position_ids: torch.Tensor,
    causal_limit: torch.Tensor,
    past_key_values,
    block_size: int,
    mask_token_id: int,
    ar_seq_len: int,
    temperature: float,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    diagonal_mask: torch.Tensor,
    one_jump_mask: torch.Tensor,
    endpoint_topk: int,
    endpoint_transport: str,
    endpoint_vocab_chunk_size: int,
    ecld_time_weight: str,
    time_conditioning_scale: float,
    temporal_difference_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finite-difference ECLD on off-diagonal CFM blocks.

    Dense transport uses the complete vocabulary simplex. The temporal
    derivative is finite-difference because the upstream attention stack does
    not support a full-model JVP on the target hardware.
    """

    # Exact (0 -> 1) blocks are supervised directly by the frozen AR teacher.
    # Applying the temporal ECLD weight at t=1 would create a boundary
    # singularity and drown out the proposal objective.
    off_diagonal = ~diagonal_mask & ~one_jump_mask
    if not torch.any(off_diagonal):
        zero = first_logits.new_zeros(())
        return zero, zero, zero

    batch_size = clean_blocks.shape[0]
    off_blocks = int(off_diagonal.sum(dim=1)[0].item())
    selected_clean = clean_blocks[off_diagonal].reshape(batch_size, off_blocks, block_size)
    selected_source_blocks = source_blocks[off_diagonal].reshape(
        batch_size, off_blocks, block_size
    )
    selected_positions = position_ids.reshape(batch_size, -1, block_size)[off_diagonal].reshape(
        batch_size, off_blocks * block_size
    )
    selected_limits = causal_limit.reshape(batch_size, -1, block_size)[off_diagonal].reshape(
        batch_size, off_blocks * block_size
    )
    selected_source = source_time[off_diagonal].reshape(batch_size, off_blocks, 1, 1)
    selected_target = target_time[off_diagonal].reshape(batch_size, off_blocks, 1, 1)
    selected_logits = select_block_values(first_logits, off_diagonal, block_size - 1)

    source_inputs = make_flowdraft_inputs_embeds(
        model=model,
        clean_blocks=selected_clean,
        mask_token_id=mask_token_id,
        state_mix=selected_source,
        source_token_ids=selected_source_blocks,
    ).reshape(batch_size, off_blocks, block_size, -1)
    if endpoint_transport == "dense":
        endpoint_embeds = exact_endpoint_embeddings(
            selected_logits,
            model.model.embed_tokens.weight,
            temperature=temperature,
            vocab_chunk_size=endpoint_vocab_chunk_size,
        )
    else:
        endpoint_embeds = topk_endpoint_embeddings(
            selected_logits,
            model.model.embed_tokens.weight,
            topk=endpoint_topk,
            temperature=temperature,
        )
    transported_draft = transport_categorical_state(
        source_inputs[:, :, 1:, :],
        endpoint_embeds,
        selected_source,
        selected_target,
    )
    transported_inputs = torch.cat([source_inputs[:, :, :1, :], transported_draft], dim=2)
    transported_inputs = transported_inputs.reshape(batch_size, off_blocks * block_size, -1).detach()
    transported_inputs = condition_flowdraft_state(
        model,
        transported_inputs,
        selected_target,
        selected_target,
        block_size=block_size,
        scale=time_conditioning_scale,
    )

    with torch.no_grad():
        target_outputs = model(
            inputs_embeds=transported_inputs,
            position_ids=selected_positions,
            past_key_values=past_key_values,
            use_cache=False,
            is_diffusion_pass=True,
            causal_limit=selected_limits,
            ar_seq_len=ar_seq_len,
        )
        target_logits = select_endpoint_logits(target_outputs.logits, block_size).reshape_as(selected_logits)
        endpoint_targets = F.softmax(target_logits.float() / temperature, dim=-1)

    endpoint_log_probs = F.log_softmax(selected_logits.float() / temperature, dim=-1)
    endpoint_ce = -(endpoint_targets * endpoint_log_probs).sum(dim=-1)
    if ecld_time_weight == "paper_clamped":
        block_weights = (1.0 - selected_target.squeeze(-1).squeeze(-1)).clamp_min(0.05).pow(-2)
        token_weights = block_weights.unsqueeze(-1).expand(-1, -1, block_size - 1).reshape_as(endpoint_ce)
        endpoint_consistency = (token_weights * endpoint_ce).mean() * (temperature**2)
    elif ecld_time_weight == "uniform":
        endpoint_consistency = endpoint_ce.mean() * (temperature**2)
    else:
        raise ValueError(f"Unsupported ECLD time weight: {ecld_time_weight}")

    # Use a backward difference at t=1.  The one-jump boundary pairs include
    # exactly t=1, where a forward finite difference would reverse direction.
    backward_difference = selected_target >= 0.999
    forward_target = (selected_target + temporal_difference_epsilon).clamp_max(0.999)
    backward_target = (selected_target - temporal_difference_epsilon).clamp_min(selected_source)
    next_target = torch.where(backward_difference, backward_target, forward_target)
    delta = (next_target - selected_target).abs().clamp_min(1e-3)
    next_inputs = source_inputs.reshape(batch_size, off_blocks * block_size, -1)
    next_inputs = condition_flowdraft_state(
        model,
        next_inputs,
        selected_source,
        next_target,
        block_size=block_size,
        scale=time_conditioning_scale,
    )
    next_outputs = model(
        inputs_embeds=next_inputs,
        position_ids=selected_positions,
        past_key_values=past_key_values,
        use_cache=False,
        is_diffusion_pass=True,
        causal_limit=selected_limits,
        ar_seq_len=ar_seq_len,
    )
    next_logits = select_endpoint_logits(next_outputs.logits, block_size).reshape_as(selected_logits)
    current_probs = F.softmax(selected_logits.float() / temperature, dim=-1)
    next_probs = F.softmax(next_logits.float() / temperature, dim=-1)
    derivative = torch.where(
        backward_difference,
        (current_probs - next_probs) / delta,
        (next_probs - current_probs) / delta,
    )
    drift_per_token = derivative.square().sum(dim=-1)
    gamma = flow_map_step_size(selected_source, selected_target).reshape(batch_size, off_blocks, 1)
    temporal_drift = (gamma.square() * drift_per_token).mean()
    combined = 4.0 * endpoint_consistency + 2.0 * temporal_drift
    return combined, endpoint_consistency, temporal_drift


@torch.no_grad()
def evaluate_distillation(
    model,
    dataloader: DataLoader,
    device: torch.device,
    block_size: int,
    mask_token_id: int,
    num_anchor_blocks: int,
    temperature: float,
    kl_reduction: str,
    prefix_weight_decay: float,
    max_batches: int,
    flow_objective: str,
    time_conditioning_scale: float,
    source_prior: str,
    source_seed: int,
) -> dict[str, float]:
    was_training = model.training
    # The upstream Orthrus implementation selects the independent diffusion
    # block mask through its training flag. Qwen has no active dropout in this
    # path, so evaluation must retain train mode and disable gradients instead.
    model.train()
    losses = []
    hard_ce_losses = []
    prefix_losses = []
    accuracies = []
    first_token_accuracies = []
    first_token_ces = []
    greedy_prefix_acceptances = []
    prefix_expected_acceptances = []
    source_generator = torch.Generator(device=device).manual_seed(source_seed)

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
            flow_objective=flow_objective,
            time_conditioning_scale=time_conditioning_scale,
            force_one_jump=flow_objective == "ecld",
            source_prior=source_prior,
            source_generator=source_generator,
        )
        teacher_ids = out["teacher_logits"].argmax(dim=-1)
        losses.append(
            float(
                forward_kl_distillation(
                    out["student_logits"],
                    out["teacher_logits"],
                    temperature,
                    reduction=kl_reduction,
                ).cpu()
            )
        )
        hard_ce_losses.append(
            float(
                F.cross_entropy(
                    out["student_logits"].reshape(-1, out["student_logits"].shape[-1]).float(),
                    teacher_ids.reshape(-1),
                ).cpu()
            )
        )
        prefix_losses.append(
            float(
                prefix_survival_cross_entropy(
                    out["student_logits"],
                    teacher_ids,
                    block_size=block_size,
                    decay=prefix_weight_decay,
                ).cpu()
            )
        )
        accuracies.append(float(token_accuracy(out["student_logits"], teacher_ids).cpu()))
        prefix_metrics = prefix_acceptance_metrics(out["student_logits"], teacher_ids, block_size)
        first_token_accuracies.append(float(prefix_metrics["first_token_acc"].cpu()))
        first_token_ces.append(float(prefix_metrics["first_token_ce"].cpu()))
        greedy_prefix_acceptances.append(float(prefix_metrics["greedy_prefix_acceptance"].cpu()))
        prefix_expected_acceptances.append(float(prefix_metrics["prefix_expected_acceptance"].cpu()))

    if not was_training:
        model.eval()

    return {
        "eval_loss": sum(losses) / len(losses) if losses else float("inf"),
        "eval_hard_ce": sum(hard_ce_losses) / len(hard_ce_losses) if hard_ce_losses else float("inf"),
        "eval_prefix_ce": sum(prefix_losses) / len(prefix_losses) if prefix_losses else float("inf"),
        "eval_top1": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "eval_first_token_acc": sum(first_token_accuracies) / len(first_token_accuracies)
        if first_token_accuracies
        else 0.0,
        "eval_first_token_ce": sum(first_token_ces) / len(first_token_ces) if first_token_ces else float("inf"),
        "eval_greedy_prefix_acceptance": sum(greedy_prefix_acceptances) / len(greedy_prefix_acceptances)
        if greedy_prefix_acceptances
        else 0.0,
        "eval_prefix_expected_acceptance": sum(prefix_expected_acceptances) / len(prefix_expected_acceptances)
        if prefix_expected_acceptances
        else 0.0,
        "eval_batches": len(losses),
    }


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def sha256_file(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(repo_root: Path) -> dict:
    def git_value(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": git_value("rev-parse", "HEAD"),
        "branch": git_value("branch", "--show-current"),
        "status": git_value("status", "--short"),
    }


def write_run_manifest(
    output_dir: Path,
    args,
    config_path: str | None,
    config_values: dict,
    repo_root: Path,
) -> None:
    manifest = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git": git_metadata(repo_root),
        "config_file": {
            "path": str(Path(config_path).resolve()) if config_path else None,
            "sha256": sha256_file(config_path) if config_path else None,
            "resolved": config_values,
        },
        "train_manifest": {
            "path": str(Path(args.train_manifest).resolve()),
            "sha256": sha256_file(args.train_manifest),
        },
        "eval_manifest": {
            "path": str(Path(args.eval_manifest).resolve()) if args.eval_manifest else None,
            "sha256": sha256_file(args.eval_manifest) if args.eval_manifest else None,
        },
        "args": vars(args),
    }
    write_json(output_dir / "run_manifest.json", manifest)


def save_model_checkpoint(model, tokenizer, output_dir: Path, args) -> None:
    if args.checkpoint_format == "trainable":
        save_trainable_checkpoint(model, output_dir, args.base_model)
    else:
        save_orthrus_checkpoint(model, tokenizer, output_dir, args.upstream_dir)


def main() -> None:
    parsed_args, cli_keys = parse_args()
    config_values = load_config_file(parsed_args.config)
    args = merge_config(parsed_args, config_values, cli_keys)
    set_seed(args.seed)

    if args.flow_objective == "ecld" and args.consistency_weight > 0:
        off_diagonal_blocks = args.num_anchor_blocks - round(
            args.num_anchor_blocks * args.diagonal_fraction
        )
        off_diagonal_length = off_diagonal_blocks * args.block_size
        if off_diagonal_length <= 0 or off_diagonal_length % 128 != 0:
            raise ValueError(
                "ECLD off-diagonal diffusion length must be a positive multiple of 128 "
                "for the upstream FlexAttention kernel; got "
                f"({args.num_anchor_blocks} - round({args.num_anchor_blocks} * "
                f"{args.diagonal_fraction})) * {args.block_size} = {off_diagonal_length}."
            )

    if not torch.cuda.is_available():
        raise RuntimeError("This training loop expects a CUDA GPU with FlexAttention support.")

    device = torch.device("cuda")
    dtype = dtype_from_string(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        output_dir,
        args,
        args.config,
        config_values,
        Path(__file__).resolve().parents[1],
    )

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
        if not args.allow_eval_overlap:
            train_unique, eval_unique = assert_disjoint_packed_manifests(
                args.train_manifest, args.eval_manifest
            )
            print(f"data disjointness verified train_unique={train_unique} eval_unique={eval_unique}")
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
        flowdraft_adapter_bottleneck=args.flow_adapter_bottleneck,
    )
    initialized_names = []
    if args.init_checkpoint:
        initialized_names = load_trainable_initialization(
            model,
            args.init_checkpoint,
            allow_missing_trainables=args.allow_missing_init_trainables,
        )
        print(
            f"initialized trainable projections from {args.init_checkpoint} "
            f"tensors={len(initialized_names)}"
        )
    model.to(device=device, dtype=dtype)
    model.train()
    model.config.use_cache = True
    model.config.flowdraft_training = True
    model.config.flowdraft_cfm = args.flow_objective == "ecld"
    model.config.flowdraft_time_conditioning_scale = args.flow_time_conditioning_scale
    model.config.flowdraft_endpoint_topk = args.endpoint_topk
    model.config.flowdraft_endpoint_transport = args.endpoint_transport
    model.config.flowdraft_state_adapter = args.flow_state_adapter
    model.config.flowdraft_adapter_bottleneck = args.flow_adapter_bottleneck
    model.config.flowdraft_one_jump_fraction = args.one_jump_fraction
    model.config.flowdraft_objective = args.flow_objective
    model.config.flowdraft_source_prior = args.source_prior
    model.config.flowdraft_source_seed = args.source_seed
    set_flowdraft_state_adapter_trainable(model, args.flow_state_adapter)
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
        "method": "flowdraft_teacher_forced_categorical_flow_map_v4" if args.flow_state_adapter else (
            "flowdraft_categorical_flow_map" if args.flow_objective == "ecld" else "flowdraft_legacy"
        ),
        "objective": (
            "teacher_forced_diagonal_vfm_plus_soft_prefix_kl_plus_off_diagonal_ecld"
            if args.flow_objective == "ecld"
            else "endpoint_teacher_distillation_plus_prefix_survival"
        ),
        "simplex_projection": "complete_vocabulary_expectation"
        if args.endpoint_transport == "dense"
        else f"renormalized_topk_{args.endpoint_topk}",
        "temporal_derivative": "finite_difference_boundary_safe" if args.flow_objective == "ecld" else None,
        "initialized_trainable_tensors": len(initialized_names),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
    }
    write_json(output_dir / "run_config.json", metadata)

    global_step = 0
    running_loss = 0.0
    running_ce_loss = 0.0
    running_direct_teacher_kl = 0.0
    running_direct_teacher_contribution = 0.0
    running_hard_ce_loss = 0.0
    running_prefix_loss = 0.0
    running_direct_prefix_contribution = 0.0
    running_prefix_kl_loss = 0.0
    running_direct_prefix_kl = 0.0
    running_direct_prefix_kl_contribution = 0.0
    running_consistency_loss = 0.0
    running_consistency_contribution = 0.0
    running_endpoint_consistency = 0.0
    running_temporal_drift = 0.0
    running_acc = 0.0
    running_first_acc = 0.0
    running_first_ce = 0.0
    running_greedy_prefix_acceptance = 0.0
    running_prefix_expected_acceptance = 0.0
    best_eval_loss = float("inf")
    best_metric_value = float("inf") if args.best_metric == "eval_loss" else float("-inf")
    best_step = None
    best_eval_top1 = None
    evaluations_without_improvement = 0
    stopped_early = False
    started_at = time.perf_counter()
    metrics_file = (output_dir / "train_metrics.jsonl").open("a", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)
    source_generator = torch.Generator(device=device).manual_seed(args.source_seed)

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
                    flow_objective=args.flow_objective,
                    # Stage 1 always contains diagonal VFM and exact 0 -> 1
                    # teacher proposals. Stage 2 replaces half of the latter
                    # with interior ECLD pairs; it must not remove the direct
                    # proposal signal during warm-up.
                    diagonal_fraction=args.diagonal_fraction,
                    time_logit_mean=args.flow_time_logit_mean,
                    time_logit_std=args.flow_time_logit_std,
                    time_max=args.flow_time_max,
                    time_conditioning_scale=args.flow_time_conditioning_scale,
                    one_jump_fraction=(
                        args.one_jump_fraction
                        if global_step >= args.consistency_start_step
                        else 1.0
                    ),
                    source_prior=args.source_prior,
                    source_generator=source_generator,
                )

                supervised_mask = out["diagonal_mask"] if args.flow_objective == "ecld" else torch.ones_like(
                    out["diagonal_mask"]
                )
                supervised_logits = select_block_values(
                    out["student_logits"], supervised_mask, args.block_size - 1
                ).reshape(input_ids.shape[0], -1, out["student_logits"].shape[-1])
                supervised_teacher = select_block_values(
                    out["teacher_logits"], supervised_mask, args.block_size - 1
                ).reshape_as(supervised_logits)
                supervised_targets = select_block_values(
                    out["teacher_logits"].argmax(dim=-1), supervised_mask, args.block_size - 1
                ).reshape(input_ids.shape[0], -1)
                ce_loss = forward_kl_distillation(
                    supervised_logits,
                    supervised_teacher,
                    temperature=args.temperature,
                    reduction=args.kl_reduction,
                )
                hard_ce_loss = F.cross_entropy(
                    supervised_logits.reshape(-1, supervised_logits.shape[-1]).float(),
                    supervised_targets.reshape(-1),
                )
                prefix_loss = prefix_survival_cross_entropy(
                    supervised_logits,
                    supervised_targets,
                    block_size=args.block_size,
                    decay=args.prefix_weight_decay,
                )
                prefix_kl_loss = torch.zeros((), device=device)
                if args.prefix_kl_weight > 0:
                    prefix_weights = prefix_survival_weights(
                        block_size=args.block_size,
                        decay=args.prefix_weight_decay,
                        device=device,
                        dtype=torch.float32,
                    )
                    prefix_kl_loss = forward_kl_distillation(
                        supervised_logits,
                        supervised_teacher,
                        temperature=args.temperature,
                        weights=prefix_weights,
                        reduction=args.kl_reduction,
                    )
                direct_teacher_kl = torch.zeros((), device=device)
                direct_prefix_loss = torch.zeros((), device=device)
                direct_prefix_kl = torch.zeros((), device=device)
                direct_top1 = torch.zeros((), device=device)
                direct_prefix_metrics = {
                    "first_token_acc": torch.zeros((), device=device),
                    "first_token_ce": torch.zeros((), device=device),
                    "greedy_prefix_acceptance": torch.zeros((), device=device),
                    "prefix_expected_acceptance": torch.zeros((), device=device),
                }
                if args.flow_objective == "ecld" and torch.any(out["one_jump_mask"]):
                    direct_logits = select_block_values(
                        out["student_logits"], out["one_jump_mask"], args.block_size - 1
                    ).reshape(input_ids.shape[0], -1, out["student_logits"].shape[-1])
                    direct_teacher = select_block_values(
                        out["teacher_logits"], out["one_jump_mask"], args.block_size - 1
                    ).reshape_as(direct_logits)
                    direct_targets = select_block_values(
                        out["teacher_logits"].argmax(dim=-1), out["one_jump_mask"], args.block_size - 1
                    ).reshape(input_ids.shape[0], -1)
                    direct_teacher_kl = forward_kl_distillation(
                        direct_logits,
                        direct_teacher,
                        temperature=args.temperature,
                        reduction=args.kl_reduction,
                    )
                    direct_prefix_loss = prefix_survival_cross_entropy(
                        direct_logits,
                        direct_targets,
                        block_size=args.block_size,
                        decay=args.prefix_weight_decay,
                    )
                    direct_prefix_kl = forward_kl_distillation(
                        direct_logits,
                        direct_teacher,
                        temperature=args.temperature,
                        weights=prefix_survival_weights(
                            block_size=args.block_size,
                            decay=args.prefix_weight_decay,
                            device=device,
                            dtype=torch.float32,
                        ),
                        reduction=args.kl_reduction,
                    )
                    direct_top1 = token_accuracy(direct_logits.detach(), direct_targets)
                    direct_prefix_metrics = prefix_acceptance_metrics(
                        direct_logits.detach(), direct_targets, args.block_size
                    )
                consistency = torch.zeros((), device=device)
                endpoint_consistency = torch.zeros((), device=device)
                temporal_drift = torch.zeros((), device=device)
                if (
                    args.flow_objective == "ecld"
                    and args.consistency_weight > 0
                    and global_step >= args.consistency_start_step
                ):
                    consistency, endpoint_consistency, temporal_drift = ecld_consistency_loss(
                        model=model,
                        first_logits=out["student_logits"],
                        clean_blocks=out["clean_blocks"],
                        source_blocks=out["source_blocks"],
                        position_ids=out["position_ids"],
                        causal_limit=out["causal_limit"],
                        past_key_values=out["past_key_values"],
                        block_size=args.block_size,
                        mask_token_id=args.mask_token_id,
                        ar_seq_len=input_ids.shape[1],
                        temperature=args.temperature,
                        source_time=out["source_time"],
                        target_time=out["target_time"],
                        diagonal_mask=out["diagonal_mask"],
                        one_jump_mask=out["one_jump_mask"],
                        endpoint_topk=args.endpoint_topk,
                        endpoint_transport=args.endpoint_transport,
                        endpoint_vocab_chunk_size=args.endpoint_vocab_chunk_size,
                        ecld_time_weight=args.ecld_time_weight,
                        time_conditioning_scale=args.flow_time_conditioning_scale,
                        temporal_difference_epsilon=args.temporal_difference_epsilon,
                    )
                loss = (
                    ce_loss
                    + args.direct_endpoint_teacher_weight * direct_teacher_kl
                    + args.hard_ce_weight * hard_ce_loss
                    + args.prefix_loss_weight * direct_prefix_loss
                    + args.prefix_kl_weight * prefix_kl_loss
                    + args.direct_prefix_kl_weight * direct_prefix_kl
                    + args.consistency_weight
                    * (4.0 * endpoint_consistency + 2.0 * args.temporal_drift_weight * temporal_drift)
                )
                direct_teacher_contribution = args.direct_endpoint_teacher_weight * direct_teacher_kl
                direct_prefix_contribution = args.prefix_loss_weight * direct_prefix_loss
                direct_prefix_kl_contribution = args.direct_prefix_kl_weight * direct_prefix_kl
                consistency_contribution = args.consistency_weight * (
                    4.0 * endpoint_consistency + 2.0 * args.temporal_drift_weight * temporal_drift
                )
                (loss / args.gradient_accumulation_steps).backward()

                running_loss += float(loss.detach().cpu())
                running_ce_loss += float(ce_loss.detach().cpu())
                running_direct_teacher_kl += float(direct_teacher_kl.detach().cpu())
                running_direct_teacher_contribution += float(direct_teacher_contribution.detach().cpu())
                running_hard_ce_loss += float(hard_ce_loss.detach().cpu())
                running_prefix_loss += float(direct_prefix_loss.detach().cpu())
                running_direct_prefix_contribution += float(direct_prefix_contribution.detach().cpu())
                running_prefix_kl_loss += float(prefix_kl_loss.detach().cpu())
                running_direct_prefix_kl += float(direct_prefix_kl.detach().cpu())
                running_direct_prefix_kl_contribution += float(direct_prefix_kl_contribution.detach().cpu())
                running_consistency_loss += float(consistency.detach().cpu())
                running_consistency_contribution += float(consistency_contribution.detach().cpu())
                running_endpoint_consistency += float(endpoint_consistency.detach().cpu())
                running_temporal_drift += float(temporal_drift.detach().cpu())
                reported_top1 = direct_top1 if args.flow_objective == "ecld" else token_accuracy(
                    supervised_logits.detach(), supervised_targets
                )
                reported_prefix_metrics = direct_prefix_metrics if args.flow_objective == "ecld" else prefix_acceptance_metrics(
                    supervised_logits.detach(), supervised_targets, args.block_size
                )
                running_acc += float(reported_top1.cpu())
                running_first_acc += float(reported_prefix_metrics["first_token_acc"].cpu())
                running_first_ce += float(reported_prefix_metrics["first_token_ce"].cpu())
                running_greedy_prefix_acceptance += float(reported_prefix_metrics["greedy_prefix_acceptance"].cpu())
                running_prefix_expected_acceptance += float(reported_prefix_metrics["prefix_expected_acceptance"].cpu())

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
                        avg_direct_teacher_kl = running_direct_teacher_kl / denom
                        avg_direct_teacher_contribution = running_direct_teacher_contribution / denom
                        avg_hard_ce = running_hard_ce_loss / denom
                        avg_prefix = running_prefix_loss / denom
                        avg_direct_prefix_contribution = running_direct_prefix_contribution / denom
                        avg_prefix_kl = running_prefix_kl_loss / denom
                        avg_direct_prefix_kl = running_direct_prefix_kl / denom
                        avg_direct_prefix_kl_contribution = running_direct_prefix_kl_contribution / denom
                        avg_consistency = running_consistency_loss / denom
                        avg_consistency_contribution = running_consistency_contribution / denom
                        avg_endpoint_consistency = running_endpoint_consistency / denom
                        avg_temporal_drift = running_temporal_drift / denom
                        avg_acc = running_acc / denom
                        avg_first_acc = running_first_acc / denom
                        avg_first_ce = running_first_ce / denom
                        avg_greedy_prefix_acceptance = running_greedy_prefix_acceptance / denom
                        avg_prefix_expected_acceptance = running_prefix_expected_acceptance / denom
                        lr = scheduler.get_last_lr()[0]
                        elapsed = time.perf_counter() - started_at
                        record = {
                            "step": global_step,
                            "epoch": epoch,
                            "loss": avg_loss,
                            "teacher_kl": avg_ce,
                            "direct_endpoint_teacher_kl": avg_direct_teacher_kl,
                            "direct_endpoint_teacher_weight": args.direct_endpoint_teacher_weight,
                            "direct_endpoint_teacher_contribution": avg_direct_teacher_contribution,
                            "kl_reduction": args.kl_reduction,
                            "hard_ce": avg_hard_ce,
                            "hard_ce_weight": args.hard_ce_weight,
                            "prefix_ce": avg_prefix,
                            "prefix_loss_weight": args.prefix_loss_weight,
                            "prefix_contribution": avg_direct_prefix_contribution,
                            "prefix_kl": avg_prefix_kl,
                            "prefix_kl_weight": args.prefix_kl_weight,
                            "direct_prefix_kl": avg_direct_prefix_kl,
                            "direct_prefix_kl_weight": args.direct_prefix_kl_weight,
                            "direct_prefix_kl_contribution": avg_direct_prefix_kl_contribution,
                            "prefix_weight_decay": args.prefix_weight_decay,
                            "consistency_loss": avg_consistency,
                            "consistency_contribution": avg_consistency_contribution,
                            "endpoint_consistency_ce": avg_endpoint_consistency,
                            "temporal_drift": avg_temporal_drift,
                            "top1": avg_acc,
                            "first_token_acc": avg_first_acc,
                            "first_token_ce": avg_first_ce,
                            "greedy_prefix_acceptance": avg_greedy_prefix_acceptance,
                            "prefix_expected_acceptance": avg_prefix_expected_acceptance,
                            "lr": lr,
                            "elapsed_seconds": elapsed,
                            "optimizer_steps_per_second": global_step / elapsed if elapsed > 0 else 0.0,
                            "peak_memory_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
                        }
                        print(
                            f"step={global_step} loss={avg_loss:.5f} teacher_kl={avg_ce:.5f} "
                            f"direct_kl={avg_direct_teacher_kl:.5f} "
                            f"direct_term={avg_direct_teacher_contribution:.5f} "
                            f"hard_ce={avg_hard_ce:.5f} prefix_ce={avg_prefix:.5f} "
                            f"prefix_kl={avg_prefix_kl:.5f} direct_prefix_kl={avg_direct_prefix_kl:.5f} "
                            f"prefix_term={avg_direct_prefix_contribution:.5f} "
                            f"consistency={avg_consistency:.5f} consistency_term={avg_consistency_contribution:.5f} "
                            f"endpoint_ce={avg_endpoint_consistency:.5f} drift={avg_temporal_drift:.5f} "
                            f"top1={avg_acc:.4f} first={avg_first_acc:.4f} "
                            f"greedy_prefix={avg_greedy_prefix_acceptance:.2f} "
                            f"expected_prefix={avg_prefix_expected_acceptance:.2f} "
                            f"lr={lr:.3e} peak_gb={record['peak_memory_gb']:.2f}"
                        )
                        metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                        metrics_file.flush()
                        running_loss = 0.0
                        running_ce_loss = 0.0
                        running_direct_teacher_kl = 0.0
                        running_direct_teacher_contribution = 0.0
                        running_hard_ce_loss = 0.0
                        running_prefix_loss = 0.0
                        running_direct_prefix_contribution = 0.0
                        running_prefix_kl_loss = 0.0
                        running_direct_prefix_kl = 0.0
                        running_direct_prefix_kl_contribution = 0.0
                        running_consistency_loss = 0.0
                        running_consistency_contribution = 0.0
                        running_endpoint_consistency = 0.0
                        running_temporal_drift = 0.0
                        running_acc = 0.0
                        running_first_acc = 0.0
                        running_first_ce = 0.0
                        running_greedy_prefix_acceptance = 0.0
                        running_prefix_expected_acceptance = 0.0

                    if eval_dataloader is not None and args.eval_every > 0 and global_step % args.eval_every == 0:
                        eval_cuda_device = device.index if device.index is not None else torch.cuda.current_device()
                        with torch.random.fork_rng(devices=[eval_cuda_device]):
                            torch.manual_seed(args.eval_seed)
                            torch.cuda.manual_seed(args.eval_seed)
                            eval_record = evaluate_distillation(
                                model=model,
                                dataloader=eval_dataloader,
                                device=device,
                                block_size=args.block_size,
                                mask_token_id=args.mask_token_id,
                                num_anchor_blocks=args.eval_anchor_blocks,
                                temperature=args.temperature,
                                kl_reduction=args.kl_reduction,
                                prefix_weight_decay=args.prefix_weight_decay,
                                max_batches=args.eval_batches,
                                flow_objective=args.flow_objective,
                                time_conditioning_scale=args.flow_time_conditioning_scale,
                                source_prior=args.source_prior,
                                source_seed=args.source_seed + 1,
                            )
                        eval_record.update({"step": global_step, "epoch": epoch, "kind": "eval"})
                        print(
                            f"eval step={global_step} loss={eval_record['eval_loss']:.5f} "
                            f"top1={eval_record['eval_top1']:.4f} "
                            f"first={eval_record['eval_first_token_acc']:.4f} "
                            f"greedy_prefix={eval_record['eval_greedy_prefix_acceptance']:.2f} "
                            f"expected_prefix={eval_record['eval_prefix_expected_acceptance']:.2f}"
                        )
                        metrics_file.write(json.dumps(eval_record, sort_keys=True) + "\n")
                        metrics_file.flush()

                        candidate_value = eval_record[args.best_metric]
                        improved = (
                            candidate_value < best_metric_value
                            if args.best_metric == "eval_loss"
                            else candidate_value > best_metric_value
                        )
                        if improved:
                            evaluations_without_improvement = 0
                            best_metric_value = candidate_value
                            best_eval_loss = eval_record["eval_loss"]
                            best_step = global_step
                            best_eval_top1 = eval_record["eval_top1"]
                            save_model_checkpoint(model, tokenizer, output_dir / "best", args)
                            write_json(
                                output_dir / "best_metrics.json",
                                {
                                    "best_step": best_step,
                                    "best_metric": args.best_metric,
                                    "best_metric_value": best_metric_value,
                                    "best_eval_loss": best_eval_loss,
                                    "best_eval_top1": best_eval_top1,
                                    "best_eval_first_token_acc": eval_record["eval_first_token_acc"],
                                    "best_eval_greedy_prefix_acceptance": eval_record[
                                        "eval_greedy_prefix_acceptance"
                                    ],
                                    "best_eval_prefix_expected_acceptance": eval_record[
                                        "eval_prefix_expected_acceptance"
                                    ],
                                    "checkpoint_written": True,
                                    "method": metadata["method"],
                                },
                            )
                        else:
                            evaluations_without_improvement += 1

                        if (
                            args.early_stopping_patience > 0
                            and evaluations_without_improvement >= args.early_stopping_patience
                        ):
                            stopped_early = True
                            early_stop_record = {
                                "step": global_step,
                                "epoch": epoch,
                                "kind": "early_stop",
                                "best_step": best_step,
                                "best_metric": args.best_metric,
                                "best_metric_value": best_metric_value,
                                "evaluations_without_improvement": evaluations_without_improvement,
                                "early_stopping_patience": args.early_stopping_patience,
                            }
                            print(
                                f"early stopping at step={global_step}: no {args.best_metric} improvement "
                                f"for {evaluations_without_improvement} holdout evaluations; "
                                f"best_step={best_step}"
                            )
                            metrics_file.write(json.dumps(early_stop_record, sort_keys=True) + "\n")
                            metrics_file.flush()

                    if args.save_every > 0 and global_step % args.save_every == 0:
                        last_dir = output_dir / "last"
                        save_model_checkpoint(model, tokenizer, last_dir, args)
                        write_json(
                            output_dir / "last_metrics.json",
                            {
                                "last_step": global_step,
                                "best_step": best_step,
                                "best_metric": args.best_metric,
                                "best_metric_value": best_metric_value if best_step is not None else None,
                                "best_eval_loss": best_eval_loss if best_step is not None else None,
                                "method": metadata["method"],
                            },
                        )
                        if args.save_trainer_state:
                            save_training_state(last_dir, optimizer, scheduler, global_step, epoch)

                    if stopped_early or global_step >= total_steps:
                        break

                if stopped_early or global_step >= total_steps:
                    break

            if stopped_early or global_step >= total_steps:
                break

        progress.close()
        if args.save_final:
            save_model_checkpoint(model, tokenizer, output_dir / "last", args)
            write_json(
                output_dir / "last_metrics.json",
                {
                    "last_step": global_step,
                    "best_step": best_step,
                    "best_metric": args.best_metric,
                    "best_metric_value": best_metric_value if best_step is not None else None,
                    "best_eval_loss": best_eval_loss if best_step is not None else None,
                    "method": "flowdraft_categorical_flow_map",
                    "stopped_early": stopped_early,
                    "early_stopping_patience": args.early_stopping_patience,
                    "evaluations_without_improvement": evaluations_without_improvement,
                },
            )
            if args.save_trainer_state:
                save_training_state(output_dir / "last", optimizer, scheduler, global_step, args.epochs)
        manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "global_step": global_step,
                "best_step": best_step,
                "stopped_early": stopped_early,
                "early_stopping_patience": args.early_stopping_patience,
                "evaluations_without_improvement": evaluations_without_improvement,
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
    finally:
        metrics_file.close()


if __name__ == "__main__":
    main()
