#!/usr/bin/env python
"""Strict eager-FP32 greedy benchmark for one-verifier feature flow maps."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_flowdraft import encode_prompt, first_mismatch, generate_ar_greedy, load_prompts, percentile, sample_greedy, synchronize
from orthrus_training.feature_flow_map import FeatureFlowMapHead, feature_flow_source
from orthrus_training.modeling import dtype_from_string, load_flowdraft_adapter, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a lossless FeatureFlowMap decoder.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--summary-json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--warmup-prompts", type=int, default=1)
    parser.add_argument("--require-parity", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_head(path: str | Path, device: torch.device, dtype: torch.dtype):
    path = Path(path)
    with (path / "feature_flow_map_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("format") != FeatureFlowMapHead.format_name:
        raise ValueError(f"Unsupported feature flow checkpoint format: {config.get('format')}")
    head = FeatureFlowMapHead(
        hidden_size=int(config["hidden_size"]), block_size=int(config["block_size"]),
        latent_size=int(config["latent_size"]), num_layers=int(config["num_layers"]), num_heads=int(config["num_heads"]),
    )
    head.load_state_dict(load_file(path / "feature_flow_map.safetensors"), strict=True)
    return head.to(device=device, dtype=dtype).eval(), config


@torch.inference_mode()
def generate_feature_flow_greedy(model, head, input_ids, max_new_tokens: int, eos_token_id: int | None):
    device = input_ids.device
    block_size, mask_id = int(model.config.block_size), int(model.config.mask_token_id)
    past = DynamicCache(config=model.config)
    source_generator = torch.Generator(device=device).manual_seed(901)
    max_length = input_ids.shape[1] + max_new_tokens
    output_ids = torch.full((1, max_length + block_size), mask_id, device=device, dtype=torch.long)
    output_ids[:, :input_ids.shape[1]] = input_ids
    synchronize(device); started = time.perf_counter()
    prefill = model(
        input_ids=input_ids, position_ids=torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
        past_key_values=past, use_cache=True, output_hidden_states=True, is_diffusion_pass=False,
    )
    target_forwards, head_calls = 1, 0
    context_hidden = prefill.hidden_states[-1][:, -1, :]
    start_idx, generated, output_length = input_ids.shape[1], 1, input_ids.shape[1] + 1
    output_ids[:, start_idx] = sample_greedy(prefill.logits[:, -1, :])
    acceptances: list[int] = []
    if eos_token_id is not None and int(output_ids[0, start_idx]) == int(eos_token_id):
        synchronize(device)
        return output_ids[:, :output_length], time.perf_counter() - started, target_forwards, head_calls, acceptances
    while generated < max_new_tokens and start_idx < max_length - 1:
        diff_len = min(block_size, max_length - start_idx)
        context = context_hidden.reshape(1, 1, -1)
        source = feature_flow_source(context, head.prediction_length, source_generator)
        time_zero = torch.zeros((1, 1), device=device, dtype=source.dtype)
        endpoint = head(context, model.model.embed_tokens(output_ids[:, start_idx:start_idx + 1]).reshape(1, 1, -1), source, time_zero)
        proposal = sample_greedy(model.lm_head(endpoint.reshape(-1, endpoint.shape[-1]))).reshape(1, -1)[:, :diff_len - 1]
        head_calls += 1
        proposed_block = torch.cat([output_ids[:, start_idx:start_idx + 1], proposal], dim=1)
        verifier = model(
            input_ids=proposed_block, position_ids=torch.arange(start_idx, start_idx + diff_len, device=device).unsqueeze(0),
            past_key_values=past, use_cache=True, output_hidden_states=True, is_diffusion_pass=False,
        )
        target_forwards += 1
        verifier_tokens = sample_greedy(verifier.logits)
        matches = proposal == verifier_tokens[:, :-1]
        accepted = int(matches.cumprod(dim=1).sum(dim=1)[0].item()) if proposal.numel() else 0
        acceptances.append(accepted)
        accepted_block, end_idx = proposed_block[:, :accepted + 1], start_idx + accepted + 1
        output_ids[:, start_idx:end_idx] = accepted_block
        output_length = end_idx
        eos_positions = (accepted_block == eos_token_id).nonzero() if eos_token_id is not None else []
        if len(eos_positions) > 0:
            output_length = start_idx + int(eos_positions[0, -1].item()) + 1
            break
        # This state was produced by the verifier for the accepted prefix.
        context_hidden = verifier.hidden_states[-1][:, accepted, :]
        next_token = verifier_tokens[:, accepted]
        generated += accepted; start_idx = end_idx; past.crop(start_idx)
        if generated < max_new_tokens and start_idx < max_length:
            output_ids[:, start_idx] = next_token
            generated += 1; output_length = start_idx + 1
            if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                break
    synchronize(device)
    return output_ids[:, :output_length], time.perf_counter() - started, target_forwards, head_calls, acceptances


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_string(args.dtype)
    head, config = load_head(args.checkpoint, device, dtype)
    parent = args.base_checkpoint or config["base_flowdraft_checkpoint"]
    model, metadata, _ = load_flowdraft_adapter(parent, args.upstream_dir, dtype, args.attn_implementation)
    model = model.to(device=device, dtype=dtype).eval()
    tokenizer, prompts = load_tokenizer(metadata["base_model"]), load_prompts(args.prompts_jsonl)
    for row in prompts[:args.warmup_prompts]:
        generate_feature_flow_greedy(model, head, encode_prompt(tokenizer, row["prompt"], device), min(16, args.max_new_tokens), tokenizer.eos_token_id)
    rows = []
    for row in prompts:
        input_ids = encode_prompt(tokenizer, row["prompt"], device)
        ar_ids, ar_seconds, ar_forwards = generate_ar_greedy(model, input_ids, args.max_new_tokens, tokenizer.eos_token_id)
        flow_ids, flow_seconds, target_forwards, head_calls, acceptance = generate_feature_flow_greedy(model, head, input_ids, args.max_new_tokens, tokenizer.eos_token_id)
        ar_new, flow_new = int(ar_ids.shape[1] - input_ids.shape[1]), int(flow_ids.shape[1] - input_ids.shape[1])
        record = {
            "id": row["id"], "parity_ok": bool(torch.equal(ar_ids, flow_ids)), "first_mismatch_at": first_mismatch(ar_ids, flow_ids),
            "ar_tokens": ar_new, "feature_flow_tokens": flow_new, "ar_seconds": ar_seconds, "feature_flow_seconds": flow_seconds,
            "speedup": ((flow_new / flow_seconds) / (ar_new / ar_seconds)) if ar_seconds and flow_seconds and ar_new else 0.0,
            "ar_forward_passes": ar_forwards, "target_model_forward_passes": target_forwards, "feature_flow_head_calls": head_calls,
            "target_model_tpf": flow_new / target_forwards if target_forwards else 0.0, "acceptance": acceptance,
            "acceptance_mean": statistics.mean(acceptance) if acceptance else 0.0,
        }
        rows.append(record); print(json.dumps(record), flush=True)
    total_ar_tokens, total_flow_tokens = sum(row["ar_tokens"] for row in rows), sum(row["feature_flow_tokens"] for row in rows)
    total_ar_seconds, total_flow_seconds = sum(row["ar_seconds"] for row in rows), sum(row["feature_flow_seconds"] for row in rows)
    total_forwards, total_head_calls = sum(row["target_model_forward_passes"] for row in rows), sum(row["feature_flow_head_calls"] for row in rows)
    accepted = [value for row in rows for value in row["acceptance"]]
    summary = {
        "method": "conditional endpoint categorical feature flow map", "num_prompts": len(rows), "checkpoint": str(Path(args.checkpoint).resolve()),
        "base_checkpoint": str(parent), "dtype": args.dtype,
        "parity_rate": sum(row["parity_ok"] for row in rows) / len(rows) if rows else 0.0,
        "aggregate_speedup": ((total_flow_tokens / total_flow_seconds) / (total_ar_tokens / total_ar_seconds)) if total_ar_seconds and total_flow_seconds else 0.0,
        "aggregate_target_model_tpf": total_flow_tokens / total_forwards if total_forwards else 0.0,
        "weighted_acceptance": sum(accepted) / len(accepted) if accepted else 0.0,
        "p50_acceptance": percentile([float(value) for value in accepted], 0.5),
        "total_target_model_forward_passes": total_forwards, "total_feature_flow_head_calls": total_head_calls,
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    if args.output_jsonl:
        with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
            handle.write("\n".join(json.dumps(row) for row in rows) + "\n")
    if args.summary_json:
        write_json(Path(args.summary_json), summary)
    if args.require_parity and summary["parity_rate"] != 1.0:
        raise SystemExit(f"Parity check failed: {summary['parity_rate']:.6f}")


if __name__ == "__main__":
    main()
