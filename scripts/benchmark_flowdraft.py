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

from orthrus_training.flowdraft import make_discrete_flowdraft_state


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark AR vs FlowDraft greedy lossless decoding.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts-jsonl", default="eval_prompts/quick_compare.jsonl")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--flow-steps", type=int, default=1)
    parser.add_argument("--warmup-prompts", type=int, default=1)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--paper-speedup-target", type=float, default=4.25)
    parser.add_argument("--paper-tpf-target", type=float, default=3.89)
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

    elapsed = time.perf_counter() - start
    return torch.cat(output_ids + generated, dim=1), elapsed, forward_passes


@torch.inference_mode()
def generate_flowdraft_greedy(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    flow_steps: int,
):
    device = input_ids.device
    block_size = model.config.block_size
    mask_token_id = model.config.mask_token_id
    past_key_values = DynamicCache(config=model.config)

    max_length = input_ids.shape[1] + max_new_tokens
    output_ids = torch.full((1, max_length + block_size), mask_token_id, dtype=torch.long, device=device)
    output_ids[:, : input_ids.shape[1]] = input_ids

    start = time.perf_counter()
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    outputs = model(input_ids=input_ids, position_ids=position_ids, past_key_values=past_key_values)
    forward_passes = 1

    start_idx = input_ids.shape[1]
    next_token = sample_greedy(outputs.logits[:, -1, :])
    output_ids[:, start_idx] = next_token
    generated_count = 1
    acceptance_lengths: list[int] = []

    if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
        return output_ids[:, : start_idx + 1], time.perf_counter() - start, forward_passes, acceptance_lengths

    while generated_count < max_new_tokens and start_idx < max_length - 1:
        diff_len = min(block_size, max_length - start_idx)
        diff_position_ids = torch.arange(start_idx, start_idx + diff_len, device=device).unsqueeze(0)
        draft_tokens = None
        state_ids = None

        for _ in range(max(1, flow_steps)):
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
                draft_tokens = sample_greedy(diff_outputs.logits[:, :-1, :])
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
        acceptance_lengths.append(acceptance_len)

        next_token = ar_tokens[:, acceptance_len]
        end_idx = start_idx + acceptance_len + 1
        accepted_block = proposed_block[:, : acceptance_len + 1]

        eos_positions = (accepted_block == eos_token_id).nonzero() if eos_token_id is not None else []
        if len(eos_positions) > 0:
            eos_offset = int(eos_positions[0, -1].item())
            output_ids[:, start_idx : start_idx + eos_offset + 1] = accepted_block[:, : eos_offset + 1]
            return output_ids[:, : start_idx + eos_offset + 1], time.perf_counter() - start, forward_passes, acceptance_lengths

        output_ids[:, start_idx:end_idx] = accepted_block
        generated_count += acceptance_len
        start_idx = end_idx
        past_key_values.crop(start_idx)

        if generated_count < max_new_tokens and start_idx < max_length:
            output_ids[:, start_idx] = next_token
            generated_count += 1
            if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                return output_ids[:, : start_idx + 1], time.perf_counter() - start, forward_passes, acceptance_lengths

    elapsed = time.perf_counter() - start
    return output_ids[:, :max_length], elapsed, forward_passes, acceptance_lengths


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
            model, input_ids, min(16, args.max_new_tokens), tokenizer.eos_token_id, args.flow_steps
        )

    for row in prompts:
        input_ids = encode_prompt(tokenizer, row["prompt"], model.device)
        ar_ids, ar_seconds, ar_forwards = generate_ar_greedy(
            model, input_ids, args.max_new_tokens, tokenizer.eos_token_id
        )
        flow_ids, flow_seconds, flow_forwards, acceptance = generate_flowdraft_greedy(
            model, input_ids, args.max_new_tokens, tokenizer.eos_token_id, args.flow_steps
        )

        ar_new = int(ar_ids.shape[1] - input_ids.shape[1])
        flow_new = int(flow_ids.shape[1] - input_ids.shape[1])
        parity = bool(torch.equal(ar_ids, flow_ids))
        record = {
            "id": row["id"],
            "prompt": row["prompt"],
            "parity_ok": parity,
            "first_mismatch_at": first_mismatch(ar_ids, flow_ids),
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
            "acceptance_mean": statistics.mean(acceptance) if acceptance else 0.0,
            "acceptance_p50": percentile([float(x) for x in acceptance], 0.50),
            "acceptance_p90": percentile([float(x) for x in acceptance], 0.90),
            "acceptance_cycles": len(acceptance),
        }
        rows.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    if args.output_jsonl:
        with Path(args.output_jsonl).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "num_prompts": len(rows),
        "flow_steps": args.flow_steps,
        "parity_rate": sum(1 for row in rows if row["parity_ok"]) / len(rows) if rows else 0.0,
        "mean_speedup": statistics.mean(row["speedup"] for row in rows) if rows else 0.0,
        "mean_ar_tokens_per_sec": statistics.mean(row["ar_tokens_per_sec"] for row in rows) if rows else 0.0,
        "mean_flowdraft_tokens_per_sec": statistics.mean(row["flowdraft_tokens_per_sec"] for row in rows) if rows else 0.0,
        "mean_flowdraft_tpf": statistics.mean(row["flowdraft_tpf"] for row in rows) if rows else 0.0,
        "mean_acceptance": statistics.mean(row["acceptance_mean"] for row in rows) if rows else 0.0,
        "p50_acceptance": percentile([row["acceptance_p50"] for row in rows], 0.50),
        "p90_acceptance": percentile([row["acceptance_p90"] for row in rows], 0.90),
        "paper_speedup_target": args.paper_speedup_target,
        "paper_tpf_target": args.paper_tpf_target,
    }
    summary["speedup_vs_paper_target"] = (
        summary["mean_speedup"] / args.paper_speedup_target if args.paper_speedup_target > 0 else 0.0
    )
    summary["tpf_vs_paper_target"] = (
        summary["mean_flowdraft_tpf"] / args.paper_tpf_target if args.paper_tpf_target > 0 else 0.0
    )
    summary["speedup_gap_to_paper_target"] = args.paper_speedup_target - summary["mean_speedup"]
    summary["tpf_gap_to_paper_target"] = args.paper_tpf_target - summary["mean_flowdraft_tpf"]
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)

    if args.summary_json:
        with Path(args.summary_json).open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
