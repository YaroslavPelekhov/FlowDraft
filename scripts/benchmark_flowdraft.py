#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path


def cache_dir(env_name: str, preferred: str, fallback: str) -> Path:
    raw = os.environ.get(env_name)
    path = Path(raw or preferred)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        path = Path(fallback)
        path.mkdir(parents=True, exist_ok=True)
    return path


default_tmp = cache_dir("TMPDIR", "/dev/shm/flowdraft_tmp", "/tmp/flowdraft_tmp")
os.environ["TMPDIR"] = str(default_tmp)
tempfile.tempdir = str(default_tmp)
os.environ["HF_HOME"] = str(cache_dir("HF_HOME", "/tmp/flowdraft_hf", "/tmp/flowdraft_hf"))
os.environ["HF_MODULES_CACHE"] = str(
    cache_dir("HF_MODULES_CACHE", "/dev/shm/flowdraft_hf_modules", "/tmp/flowdraft_hf_modules")
)
os.environ["XDG_CACHE_HOME"] = str(
    cache_dir("XDG_CACHE_HOME", "/dev/shm/flowdraft_xdg_cache", "/tmp/flowdraft_xdg_cache")
)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orthrus_training.flowdraft import (
    condition_flowdraft_state,
    exact_endpoint_embeddings,
    make_discrete_flowdraft_state,
    sample_categorical_source_tokens,
    topk_endpoint_embeddings,
    transport_categorical_state,
)
from orthrus_training.modeling import load_flowdraft_adapter, load_tokenizer


PAPER_GREEDY_TARGETS = {
    "gsm8k": {"tpf": 4.20, "speedup": 4.37},
    "math500": {"tpf": 4.71, "speedup": 4.74},
    "aime24": {"tpf": 4.33, "speedup": 5.65},
    "aime25": {"tpf": 3.89, "speedup": 4.80},
    "humaneval": {"tpf": 2.75, "speedup": 3.07},
    "mbpp": {"tpf": 2.76, "speedup": 3.07},
    "pseudo2code": {"tpf": 4.60, "speedup": 4.90},
    "livecodebench_v5": {"tpf": 3.86, "speedup": 5.87},
    "average": {"tpf": 3.89, "speedup": 4.25},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark AR vs FlowDraft greedy lossless decoding.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--prompts-jsonl", default="eval_prompts/quick_compare.jsonl")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--flow-steps", type=int, default=1)
    parser.add_argument("--warmup-prompts", type=int, default=1)
    # Exact parity is sensitive to block-vs-token numerical differences in
    # fused SDPA kernels. Use eager attention unless throughput is requested
    # explicitly with --attn-implementation sdpa.
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--benchmark-task", choices=sorted(PAPER_GREEDY_TARGETS), default=None)
    parser.add_argument("--paper-speedup-target", type=float, default=None)
    parser.add_argument("--paper-tpf-target", type=float, default=None)
    parser.add_argument("--require-parity", action="store_true")
    parser.add_argument(
        "--parity-margin-threshold",
        type=float,
        default=0.5,
        help=(
            "Recompute the earliest verifier decision below this top-2 logit "
            "margin with a sequential AR step. This guards BF16 block-vs-token ties."
        ),
    )
    return parser.parse_args()


def load_prompts(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row:
                row["id"] = f"prompt_{idx:04d}"
            rows.append(row)
    return rows


def encode_prompt(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    return encoded.to(device)


def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def first_mismatch(a: torch.Tensor, b: torch.Tensor) -> int | None:
    min_len = min(a.shape[1], b.shape[1])
    mismatch = (a[:, :min_len] != b[:, :min_len]).nonzero()
    if len(mismatch) > 0:
        return int(mismatch[0, 1].item())
    if a.shape[1] != b.shape[1]:
        return min_len
    return None


@torch.inference_mode()
def generate_ar_greedy(model, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None):
    past_key_values = DynamicCache(config=model.config)
    output_ids = [input_ids]
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

    synchronize(input_ids.device)
    start = time.perf_counter()
    outputs = model(input_ids=input_ids, position_ids=position_ids, past_key_values=past_key_values, use_cache=True)
    forward_passes = 1
    next_token = sample_greedy(outputs.logits[:, -1, :])

    generated = []
    for step in range(max_new_tokens):
        generated.append(next_token.view(1, 1))
        if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
            break
        if step == max_new_tokens - 1:
            break
        pos = torch.tensor([[input_ids.shape[1] + step]], device=input_ids.device)
        outputs = model(
            input_ids=next_token.view(1, 1),
            position_ids=pos,
            past_key_values=past_key_values,
            use_cache=True,
        )
        forward_passes += 1
        next_token = sample_greedy(outputs.logits[:, -1, :])

    synchronize(input_ids.device)
    elapsed = time.perf_counter() - start
    return torch.cat(output_ids + generated, dim=1), elapsed, forward_passes


@torch.inference_mode()
def generate_flowdraft_greedy(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    flow_steps: int,
    parity_margin_threshold: float = 0.5,
):
    device = input_ids.device
    block_size = model.config.block_size
    mask_token_id = model.config.mask_token_id
    past_key_values = DynamicCache(config=model.config)
    source_prior = str(getattr(model.config, "flowdraft_source_prior", "mask"))
    source_seed = int(getattr(model.config, "flowdraft_source_seed", 17))
    source_generator = torch.Generator(device=device).manual_seed(source_seed)

    max_length = input_ids.shape[1] + max_new_tokens
    output_ids = torch.full((1, max_length + block_size), mask_token_id, dtype=torch.long, device=device)
    output_ids[:, : input_ids.shape[1]] = input_ids

    synchronize(device)
    start = time.perf_counter()
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    outputs = model(input_ids=input_ids, position_ids=position_ids, past_key_values=past_key_values)
    forward_passes = 1

    start_idx = input_ids.shape[1]
    next_token = sample_greedy(outputs.logits[:, -1, :])
    output_ids[:, start_idx] = next_token
    generated_count = 1
    acceptance_lengths: list[int] = []
    verification_margins: list[list[float]] = []

    if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
        synchronize(device)
        return (
            output_ids[:, : start_idx + 1],
            time.perf_counter() - start,
            forward_passes,
            acceptance_lengths,
            verification_margins,
        )

    while generated_count < max_new_tokens and start_idx < max_length - 1:
        diff_len = min(block_size, max_length - start_idx)
        diff_position_ids = torch.arange(start_idx, start_idx + diff_len, device=device).unsqueeze(0)
        draft_tokens = None
        state_ids = None
        raw_state_embeds = None
        use_cfm = bool(getattr(model.config, "flowdraft_cfm", False))
        time_scale = float(getattr(model.config, "flowdraft_time_conditioning_scale", 0.0))
        endpoint_topk = int(getattr(model.config, "flowdraft_endpoint_topk", 32))

        for flow_index in range(max(1, flow_steps)):
            if use_cfm:
                if raw_state_embeds is None:
                    state_ids = make_discrete_flowdraft_state(
                        anchor_token_ids=output_ids[:, start_idx : start_idx + 1],
                        draft_token_ids=None,
                        diff_len=diff_len,
                        mask_token_id=mask_token_id,
                    )
                    if source_prior != "mask":
                        source_blocks = state_ids.reshape(1, 1, diff_len)
                        state_ids = sample_categorical_source_tokens(
                            source_blocks,
                            vocab_size=int(model.config.vocab_size),
                            mask_token_id=mask_token_id,
                            prior=source_prior,
                            generator=source_generator,
                        ).reshape(1, diff_len)
                    raw_state_embeds = model.model.embed_tokens(state_ids)
                source_time_value = flow_index / max(1, flow_steps)
                target_time_value = (flow_index + 1) / max(1, flow_steps)
                source_time = torch.full((1, 1, 1, 1), source_time_value, device=device)
                target_time = torch.full((1, 1, 1, 1), target_time_value, device=device)
                conditioned = condition_flowdraft_state(
                    model,
                    raw_state_embeds,
                    source_time,
                    target_time,
                    block_size=diff_len,
                    scale=time_scale,
                )
                diff_outputs = model(
                    inputs_embeds=conditioned,
                    position_ids=diff_position_ids,
                    past_key_values=past_key_values,
                    use_cache=False,
                    is_diffusion_pass=True,
                    ar_seq_len=start_idx,
                )
            else:
                state_ids = make_discrete_flowdraft_state(
                    anchor_token_ids=output_ids[:, start_idx : start_idx + 1],
                    draft_token_ids=draft_tokens,
                    diff_len=diff_len,
                    mask_token_id=mask_token_id,
                )
                diff_outputs = model(
                    input_ids=state_ids,
                    position_ids=diff_position_ids,
                    past_key_values=past_key_values,
                    use_cache=False,
                    is_diffusion_pass=True,
                    ar_seq_len=start_idx,
                )
            forward_passes += 1
            if diff_len > 1:
                endpoint_logits = diff_outputs.logits[:, :-1, :]
                draft_tokens = sample_greedy(endpoint_logits)
                if use_cfm and flow_index + 1 < max(1, flow_steps):
                    endpoint_transport = getattr(model.config, "flowdraft_endpoint_transport", "topk")
                    if endpoint_transport == "dense":
                        endpoint_embeds = exact_endpoint_embeddings(
                            endpoint_logits,
                            model.model.embed_tokens.weight,
                        )
                    else:
                        endpoint_embeds = topk_endpoint_embeddings(
                            endpoint_logits,
                            model.model.embed_tokens.weight,
                            topk=endpoint_topk,
                        )
                    transported = transport_categorical_state(
                        raw_state_embeds[:, 1:, :],
                        endpoint_embeds,
                        source_time.squeeze(-1),
                        target_time.squeeze(-1),
                    )
                    raw_state_embeds = torch.cat([raw_state_embeds[:, :1, :], transported], dim=1)
            else:
                draft_tokens = torch.empty((1, 0), dtype=torch.long, device=device)

        if draft_tokens is not None and draft_tokens.numel():
            proposed_block = torch.cat([output_ids[:, start_idx : start_idx + 1], draft_tokens], dim=1)
        else:
            proposed_block = output_ids[:, start_idx : start_idx + 1]
        ar_outputs = model(
            input_ids=proposed_block,
            position_ids=diff_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            is_diffusion_pass=False,
        )
        forward_passes += 1
        ar_tokens = sample_greedy(ar_outputs.logits)

        if draft_tokens is not None and draft_tokens.numel():
            matches = draft_tokens == ar_tokens[:, :-1]
            acceptance_len = int(matches.cumprod(dim=1).sum(dim=1)[0].item())
        else:
            acceptance_len = 0
        top_two = ar_outputs.logits.float().topk(k=2, dim=-1).values
        token_margins = top_two[:, :, 0] - top_two[:, :, 1]
        sequential_next_token = None
        candidate_margins = token_margins[0, : acceptance_len + 1]
        if parity_margin_threshold > 0:
            uncertain = (candidate_margins < parity_margin_threshold).nonzero()
            if len(uncertain) > 0:
                # Crop to immediately before the token whose logits determine
                # the uncertain decision, then reproduce the ordinary AR call.
                fallback_index = int(uncertain[0, 0].item())
                acceptance_len = fallback_index
                cache_length_before_context_token = start_idx + fallback_index
                past_key_values.crop(cache_length_before_context_token)
                context_token = proposed_block[:, fallback_index : fallback_index + 1]
                context_position = torch.tensor(
                    [[start_idx + fallback_index]], device=device
                )
                sequential_outputs = model(
                    input_ids=context_token,
                    position_ids=context_position,
                    past_key_values=past_key_values,
                    use_cache=True,
                    is_diffusion_pass=False,
                )
                forward_passes += 1
                sequential_next_token = sample_greedy(
                    sequential_outputs.logits[:, -1, :]
                )

        acceptance_lengths.append(acceptance_len)
        verification_margins.append(
            token_margins[0, : acceptance_len + 1].cpu().tolist()
        )

        next_token = (
            sequential_next_token
            if sequential_next_token is not None
            else ar_tokens[:, acceptance_len]
        )
        end_idx = start_idx + acceptance_len + 1
        accepted_block = proposed_block[:, : acceptance_len + 1]

        eos_positions = (accepted_block == eos_token_id).nonzero() if eos_token_id is not None else []
        if len(eos_positions) > 0:
            eos_offset = int(eos_positions[0, -1].item())
            output_ids[:, start_idx : start_idx + eos_offset + 1] = accepted_block[:, : eos_offset + 1]
            synchronize(device)
            return (
                output_ids[:, : start_idx + eos_offset + 1],
                time.perf_counter() - start,
                forward_passes,
                acceptance_lengths,
                verification_margins,
            )

        output_ids[:, start_idx:end_idx] = accepted_block
        generated_count += acceptance_len
        start_idx = end_idx
        past_key_values.crop(start_idx)

        if generated_count < max_new_tokens and start_idx < max_length:
            output_ids[:, start_idx] = next_token
            generated_count += 1
            if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                synchronize(device)
                return (
                    output_ids[:, : start_idx + 1],
                    time.perf_counter() - start,
                    forward_passes,
                    acceptance_lengths,
                    verification_margins,
                )

    synchronize(device)
    elapsed = time.perf_counter() - start
    return (
        output_ids[:, :max_length],
        elapsed,
        forward_passes,
        acceptance_lengths,
        verification_margins,
    )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
    return float(values[idx])


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype.lower() in {"bf16", "bfloat16"} and device.type == "cuda" else torch.float32

    checkpoint_path = Path(args.checkpoint)
    if (checkpoint_path / "adapter_config.json").exists():
        model, adapter_metadata, _ = load_flowdraft_adapter(
            checkpoint_path,
            upstream_dir=args.upstream_dir,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
        )
        tokenizer = load_tokenizer(adapter_metadata["base_model"])
        model = model.to(device=device, dtype=dtype).eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            dtype=dtype,
            device_map=str(device),
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        ).eval()

    rows = []
    prompts = load_prompts(args.prompts_jsonl)

    for row in prompts[: args.warmup_prompts]:
        input_ids = encode_prompt(tokenizer, row["prompt"], model.device)
        _ = generate_ar_greedy(model, input_ids, min(16, args.max_new_tokens), tokenizer.eos_token_id)
        _ = generate_flowdraft_greedy(
            model,
            input_ids,
            min(16, args.max_new_tokens),
            tokenizer.eos_token_id,
            args.flow_steps,
            args.parity_margin_threshold,
        )

    for row in prompts:
        input_ids = encode_prompt(tokenizer, row["prompt"], model.device)
        ar_ids, ar_seconds, ar_forwards = generate_ar_greedy(
            model, input_ids, args.max_new_tokens, tokenizer.eos_token_id
        )
        flow_ids, flow_seconds, flow_forwards, acceptance, verification_margins = generate_flowdraft_greedy(
            model,
            input_ids,
            args.max_new_tokens,
            tokenizer.eos_token_id,
            args.flow_steps,
            args.parity_margin_threshold,
        )

        ar_new = int(ar_ids.shape[1] - input_ids.shape[1])
        flow_new = int(flow_ids.shape[1] - input_ids.shape[1])
        parity = bool(torch.equal(ar_ids, flow_ids))
        mismatch_at = first_mismatch(ar_ids, flow_ids)
        record = {
            "id": row["id"],
            "prompt": row["prompt"],
            "parity_ok": parity,
            "first_mismatch_at": mismatch_at,
            "first_mismatch_generated_offset": (
                mismatch_at - input_ids.shape[1] if mismatch_at is not None else None
            ),
            "ar_mismatch_token_id": (
                int(ar_ids[0, mismatch_at].item())
                if mismatch_at is not None and mismatch_at < ar_ids.shape[1]
                else None
            ),
            "flowdraft_mismatch_token_id": (
                int(flow_ids[0, mismatch_at].item())
                if mismatch_at is not None and mismatch_at < flow_ids.shape[1]
                else None
            ),
            "ar_tokens": ar_new,
            "flowdraft_tokens": flow_new,
            "ar_seconds": ar_seconds,
            "flowdraft_seconds": flow_seconds,
            "ar_tokens_per_sec": ar_new / ar_seconds if ar_seconds > 0 else 0.0,
            "flowdraft_tokens_per_sec": flow_new / flow_seconds if flow_seconds > 0 else 0.0,
            "speedup": (flow_new / flow_seconds) / (ar_new / ar_seconds)
            if ar_seconds > 0 and flow_seconds > 0 and ar_new
            else 0.0,
            "ar_forward_passes": ar_forwards,
            "flowdraft_forward_passes": flow_forwards,
            "flowdraft_tpf": flow_new / flow_forwards if flow_forwards else 0.0,
            "flow_steps": args.flow_steps,
            "parity_margin_threshold": args.parity_margin_threshold,
            "acceptance_mean": statistics.mean(acceptance) if acceptance else 0.0,
            "acceptance_p50": percentile([float(x) for x in acceptance], 0.50),
            "acceptance_p90": percentile([float(x) for x in acceptance], 0.90),
            "acceptance_cycles": len(acceptance),
            "acceptance_sum": sum(acceptance),
            "acceptance_lengths": acceptance,
            "verification_margins": verification_margins,
            "reached_max_new_tokens": ar_new >= args.max_new_tokens,
        }
        rows.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    if args.output_jsonl:
        with Path(args.output_jsonl).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    inferred_tasks = {str(row.get("task", "")).lower() for row in prompts if row.get("task")}
    benchmark_task = args.benchmark_task
    if benchmark_task is None and len(inferred_tasks) == 1:
        candidate = inferred_tasks.pop().replace("-", "_")
        benchmark_task = candidate if candidate in PAPER_GREEDY_TARGETS else None
    paper_targets = PAPER_GREEDY_TARGETS.get(benchmark_task, {})
    paper_speedup_target = args.paper_speedup_target
    paper_tpf_target = args.paper_tpf_target
    if paper_speedup_target is None:
        paper_speedup_target = paper_targets.get("speedup")
    if paper_tpf_target is None:
        paper_tpf_target = paper_targets.get("tpf")

    total_ar_tokens = sum(row["ar_tokens"] for row in rows)
    total_flow_tokens = sum(row["flowdraft_tokens"] for row in rows)
    total_ar_seconds = sum(row["ar_seconds"] for row in rows)
    total_flow_seconds = sum(row["flowdraft_seconds"] for row in rows)
    total_flow_forwards = sum(row["flowdraft_forward_passes"] for row in rows)
    total_acceptance_cycles = sum(row["acceptance_cycles"] for row in rows)
    total_accepted_tokens = sum(row["acceptance_sum"] for row in rows)

    summary = {
        "num_prompts": len(rows),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "benchmark_task": benchmark_task,
        "flow_steps": args.flow_steps,
        "parity_margin_threshold": args.parity_margin_threshold,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "parity_rate": sum(1 for row in rows if row["parity_ok"]) / len(rows) if rows else 0.0,
        "aggregate_speedup": (
            (total_flow_tokens / total_flow_seconds) / (total_ar_tokens / total_ar_seconds)
            if total_ar_tokens and total_flow_tokens and total_ar_seconds > 0 and total_flow_seconds > 0
            else 0.0
        ),
        "prompt_mean_speedup": statistics.mean(row["speedup"] for row in rows) if rows else 0.0,
        "mean_ar_tokens_per_sec": statistics.mean(row["ar_tokens_per_sec"] for row in rows) if rows else 0.0,
        "mean_flowdraft_tokens_per_sec": statistics.mean(row["flowdraft_tokens_per_sec"] for row in rows) if rows else 0.0,
        "aggregate_flowdraft_tpf": total_flow_tokens / total_flow_forwards if total_flow_forwards else 0.0,
        "prompt_mean_flowdraft_tpf": statistics.mean(row["flowdraft_tpf"] for row in rows) if rows else 0.0,
        "weighted_mean_acceptance": (
            total_accepted_tokens / total_acceptance_cycles if total_acceptance_cycles else 0.0
        ),
        "prompt_mean_acceptance": statistics.mean(row["acceptance_mean"] for row in rows) if rows else 0.0,
        "p50_acceptance": percentile([row["acceptance_p50"] for row in rows], 0.50),
        "p90_acceptance": percentile([row["acceptance_p90"] for row in rows], 0.90),
        "total_ar_tokens": total_ar_tokens,
        "total_flowdraft_tokens": total_flow_tokens,
        "total_ar_seconds": total_ar_seconds,
        "total_flowdraft_seconds": total_flow_seconds,
        "total_flowdraft_forward_passes": total_flow_forwards,
        "num_reached_max_new_tokens": sum(row["reached_max_new_tokens"] for row in rows),
        "paper_speedup_target": paper_speedup_target,
        "paper_tpf_target": paper_tpf_target,
    }
    if paper_speedup_target is not None:
        summary["speedup_vs_paper_target"] = summary["aggregate_speedup"] / paper_speedup_target
        summary["speedup_gap_to_paper_target"] = paper_speedup_target - summary["aggregate_speedup"]
    if paper_tpf_target is not None:
        summary["tpf_vs_paper_target"] = summary["aggregate_flowdraft_tpf"] / paper_tpf_target
        summary["tpf_gap_to_paper_target"] = paper_tpf_target - summary["aggregate_flowdraft_tpf"]
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)

    if args.summary_json:
        with Path(args.summary_json).open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")

    if args.require_parity and summary["parity_rate"] < 1.0:
        raise SystemExit(f"Parity check failed: parity_rate={summary['parity_rate']:.6f}")


if __name__ == "__main__":
    main()
