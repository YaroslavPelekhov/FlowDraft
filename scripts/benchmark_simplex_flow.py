#!/usr/bin/env python
"""Strict greedy benchmark for a local-simplex Categorical Flow Map refiner."""

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

from benchmark_flowdraft import (
    encode_prompt,
    first_mismatch,
    generate_ar_greedy,
    load_prompts,
    percentile,
    sample_greedy,
    synchronize,
)
from orthrus_training.flowdraft import (
    condition_flowdraft_state,
    make_discrete_flowdraft_state,
    sample_categorical_source_tokens,
)
from orthrus_training.modeling import dtype_from_string, load_flowdraft_adapter, load_tokenizer
from orthrus_training.simplex_flow import SimplexFlowRefiner, simplex_flow_step


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--summary-json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--flow-steps", type=int, default=2)
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--warmup-prompts", type=int, default=1)
    parser.add_argument("--require-parity", action="store_true")
    parser.add_argument("--parity-margin-threshold", type=float, default=0.5)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_head(path: str | Path, device: torch.device, dtype: torch.dtype):
    path = Path(path)
    with (path / "simplex_flow_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("format") != "simplex_categorical_flow_map_v1":
        raise ValueError(f"Unsupported SimplexFlow checkpoint: {config.get('format')}")
    head = SimplexFlowRefiner(
        block_size=int(config["block_size"]), candidate_count=int(config["candidate_count"]),
        hidden_size=int(config["hidden_size"]), num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]), dropout=float(config.get("dropout", 0.0)),
    )
    head.load_state_dict(load_file(path / "simplex_flow.safetensors"), strict=True)
    return head.to(device=device, dtype=dtype).eval(), config


@torch.inference_mode()
def generate_simplex_flow_greedy(model, head, input_ids, max_new_tokens, eos_token_id, flow_steps, parity_margin_threshold):
    device = input_ids.device
    block_size, mask_id = int(model.config.block_size), int(model.config.mask_token_id)
    source_prior = str(getattr(model.config, "flowdraft_source_prior", "uniform"))
    source_generator = torch.Generator(device=device).manual_seed(int(getattr(model.config, "flowdraft_source_seed", 17)))
    past = DynamicCache(config=model.config)
    max_length = input_ids.shape[1] + max_new_tokens
    output_ids = torch.full((1, max_length + block_size), mask_id, dtype=torch.long, device=device)
    output_ids[:, : input_ids.shape[1]] = input_ids
    synchronize(device); started = time.perf_counter()
    prefill = model(input_ids=input_ids, position_ids=torch.arange(input_ids.shape[1], device=device).unsqueeze(0), past_key_values=past, use_cache=True)
    qwen_forwards, head_calls = 1, 0
    start_idx = input_ids.shape[1]
    output_ids[:, start_idx] = sample_greedy(prefill.logits[:, -1, :])
    generated, output_length, acceptance = 1, start_idx + 1, []
    if eos_token_id is not None and int(output_ids[0, start_idx]) == int(eos_token_id):
        synchronize(device)
        return output_ids[:, :output_length], time.perf_counter() - started, qwen_forwards, head_calls, acceptance

    while generated < max_new_tokens and start_idx < max_length - 1:
        diff_len = min(block_size, max_length - start_idx)
        positions = torch.arange(start_idx, start_idx + diff_len, device=device).unsqueeze(0)
        state_ids = make_discrete_flowdraft_state(output_ids[:, start_idx:start_idx + 1], None, diff_len, mask_id)
        if source_prior != "mask":
            state_ids = sample_categorical_source_tokens(
                state_ids.reshape(1, 1, diff_len), int(model.config.vocab_size), mask_id,
                prior=source_prior, generator=source_generator,
            ).reshape(1, diff_len)
        raw = model.model.embed_tokens(state_ids)
        source_time = torch.zeros((1, 1, 1, 1), device=device)
        target_time = torch.ones_like(source_time)
        conditioned = condition_flowdraft_state(
            model, raw, source_time, target_time, diff_len,
            float(getattr(model.config, "flowdraft_time_conditioning_scale", 0.0)),
        )
        draft = model(
            inputs_embeds=conditioned, position_ids=positions, past_key_values=past,
            use_cache=False, is_diffusion_pass=True, ar_seq_len=start_idx,
        )
        qwen_forwards += 1
        if diff_len > 1:
            values, candidates = draft.logits[:, :-1, :].topk(head.candidate_count, dim=-1)
            if diff_len == block_size:
                state = torch.full_like(values, 1.0 / head.candidate_count)
                flow_values = values.reshape(1, 1, diff_len - 1, head.candidate_count)
                for index in range(max(1, flow_steps)):
                    s = torch.full(values.shape[:-1], index / max(1, flow_steps), device=device)
                    t = torch.full(values.shape[:-1], (index + 1) / max(1, flow_steps), device=device)
                    endpoint = head(
                        flow_values,
                        state.reshape(1, 1, diff_len - 1, -1),
                        s.reshape(1, 1, diff_len - 1),
                        t.reshape(1, 1, diff_len - 1),
                    ).reshape_as(values)
                    state = simplex_flow_step(state, endpoint, s, t)
                    head_calls += 1
                proposal = candidates.gather(-1, state.argmax(dim=-1, keepdim=True)).squeeze(-1)
            else:
                # The head is trained for a fixed K-1 simplex.  At a short
                # terminal block keep the parent proposal instead of padding
                # fake positions into its correlated flow state.
                proposal = candidates[:, :, 0]
        else:
            proposal = torch.empty((1, 0), dtype=torch.long, device=device)
        proposed = torch.cat([output_ids[:, start_idx:start_idx + 1], proposal], dim=1)
        verifier = model(input_ids=proposed, position_ids=positions, past_key_values=past, use_cache=True, is_diffusion_pass=False)
        qwen_forwards += 1
        verifier_tokens = sample_greedy(verifier.logits)
        matches = proposal.eq(verifier_tokens[:, :-1])
        accepted = int(matches.cumprod(dim=1).sum(dim=1)[0].item()) if proposal.numel() else 0
        margins = verifier.logits.float().topk(2, dim=-1).values
        margins = margins[:, :, 0] - margins[:, :, 1]
        sequential = None
        uncertain = (margins[0, :accepted + 1] < parity_margin_threshold).nonzero() if parity_margin_threshold > 0 else []
        if len(uncertain) > 0:
            accepted = int(uncertain[0, 0].item())
            past.crop(start_idx + accepted)
            sequential = model(
                input_ids=proposed[:, accepted:accepted + 1],
                position_ids=torch.tensor([[start_idx + accepted]], device=device), past_key_values=past,
                use_cache=True, is_diffusion_pass=False,
            )
            qwen_forwards += 1
            next_token = sample_greedy(sequential.logits[:, -1, :])
        else:
            next_token = verifier_tokens[:, accepted]
        acceptance.append(accepted)
        accepted_block = proposed[:, :accepted + 1]
        eos_positions = (accepted_block == eos_token_id).nonzero() if eos_token_id is not None else []
        if len(eos_positions) > 0:
            output_length = start_idx + int(eos_positions[0, -1].item()) + 1
            output_ids[:, start_idx:output_length] = accepted_block[:, : output_length - start_idx]
            break
        end_idx = start_idx + accepted + 1
        output_ids[:, start_idx:end_idx] = accepted_block
        output_length = end_idx
        generated += accepted
        start_idx = end_idx
        past.crop(start_idx)
        if generated < max_new_tokens and start_idx < max_length:
            output_ids[:, start_idx] = next_token
            generated += 1; output_length = start_idx + 1
            if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                break
    synchronize(device)
    return output_ids[:, :output_length], time.perf_counter() - started, qwen_forwards, head_calls, acceptance


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_string(args.dtype)
    head, config = load_head(args.checkpoint, device, dtype)
    parent = args.base_checkpoint or config["base_flowdraft_checkpoint"]
    model, metadata, _ = load_flowdraft_adapter(parent, args.upstream_dir, dtype, args.attn_implementation)
    model = model.to(device=device, dtype=dtype).eval()
    tokenizer, prompts = load_tokenizer(metadata["base_model"]), load_prompts(args.prompts_jsonl)
    for row in prompts[:args.warmup_prompts]:
        generate_simplex_flow_greedy(model, head, encode_prompt(tokenizer, row["prompt"], device), min(16, args.max_new_tokens), tokenizer.eos_token_id, args.flow_steps, args.parity_margin_threshold)
    rows = []
    for row in prompts:
        encoded = encode_prompt(tokenizer, row["prompt"], device)
        ar_ids, ar_seconds, ar_forwards = generate_ar_greedy(model, encoded, args.max_new_tokens, tokenizer.eos_token_id)
        draft_ids, seconds, forwards, head_calls, accepts = generate_simplex_flow_greedy(model, head, encoded, args.max_new_tokens, tokenizer.eos_token_id, args.flow_steps, args.parity_margin_threshold)
        ar_new, draft_new = int(ar_ids.shape[1] - encoded.shape[1]), int(draft_ids.shape[1] - encoded.shape[1])
        record = {"id": row["id"], "parity_ok": bool(torch.equal(ar_ids, draft_ids)), "first_mismatch_at": first_mismatch(ar_ids, draft_ids), "ar_tokens": ar_new, "simplex_flow_tokens": draft_new, "ar_seconds": ar_seconds, "simplex_flow_seconds": seconds, "speedup": ((draft_new / seconds) / (ar_new / ar_seconds)) if ar_new and ar_seconds and seconds else 0.0, "ar_forward_passes": ar_forwards, "qwen_forward_passes": forwards, "simplex_head_calls": head_calls, "tpf": draft_new / forwards if forwards else 0.0, "acceptance": accepts, "acceptance_mean": statistics.mean(accepts) if accepts else 0.0}
        rows.append(record); print(json.dumps(record), flush=True)
    total_ar_tokens, total_tokens = sum(x["ar_tokens"] for x in rows), sum(x["simplex_flow_tokens"] for x in rows)
    total_ar_seconds, total_seconds = sum(x["ar_seconds"] for x in rows), sum(x["simplex_flow_seconds"] for x in rows)
    total_forwards = sum(x["qwen_forward_passes"] for x in rows)
    accepted = [v for x in rows for v in x["acceptance"]]
    summary = {"method": "local-simplex categorical flow map", "checkpoint": str(Path(args.checkpoint).resolve()), "base_checkpoint": str(parent), "num_prompts": len(rows), "flow_steps": args.flow_steps, "parity_rate": sum(x["parity_ok"] for x in rows) / len(rows) if rows else 0.0, "aggregate_speedup": ((total_tokens / total_seconds) / (total_ar_tokens / total_ar_seconds)) if total_tokens and total_seconds and total_ar_seconds else 0.0, "aggregate_tpf": total_tokens / total_forwards if total_forwards else 0.0, "weighted_acceptance": sum(accepted) / len(accepted) if accepted else 0.0, "total_qwen_forward_passes": total_forwards, "total_simplex_head_calls": sum(x["simplex_head_calls"] for x in rows)}
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
