#!/usr/bin/env python
"""Strict greedy losslessness benchmark for residual-refined FlowDraft."""

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
from orthrus_training.flowdraft import (
    condition_flowdraft_state,
    make_discrete_flowdraft_state,
    sample_categorical_source_tokens,
)
from orthrus_training.modeling import dtype_from_string, load_flowdraft_adapter, load_tokenizer
from orthrus_training.residual_flow import ResidualFlowCorrector, corrector_logits


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark lossless R2Flow decoding against greedy AR.")
    parser.add_argument("--checkpoint", required=True, help="R2Flow best/last corrector directory")
    parser.add_argument("--base-checkpoint", default=None, help="Optional override for the parent FlowDraft adapter")
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--warmup-prompts", type=int, default=1)
    parser.add_argument("--require-parity", action="store_true")
    return parser.parse_args()


def load_corrector(path: str | Path, device: torch.device, dtype: torch.dtype):
    path = Path(path)
    with (path / "r2flow_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("format") != "r2flow_residual_corrector_v1":
        raise ValueError(f"Unsupported R2Flow checkpoint: {config.get('format')}")
    corrector = ResidualFlowCorrector(
        hidden_size=int(config["hidden_size"]),
        block_size=int(config["block_size"]),
        bottleneck_size=int(config["bottleneck_size"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
    )
    missing, unexpected = corrector.load_state_dict(load_file(path / "r2flow_corrector.safetensors"), strict=True)
    if missing or unexpected:
        raise ValueError(f"Invalid R2Flow weights missing={missing} unexpected={unexpected}")
    return corrector.to(device=device, dtype=dtype).eval(), config


@torch.inference_mode()
def draft_block(model, output_ids, start_idx: int, diff_len: int, past_key_values, source_generator):
    """Run the frozen one-step categorical FlowDraft proposal."""

    device = output_ids.device
    block_size = int(model.config.block_size)
    state_ids = make_discrete_flowdraft_state(
        anchor_token_ids=output_ids[:, start_idx : start_idx + 1],
        draft_token_ids=None,
        diff_len=diff_len,
        mask_token_id=int(model.config.mask_token_id),
    )
    source_prior = str(getattr(model.config, "flowdraft_source_prior", "mask"))
    if source_prior != "mask":
        state_ids = sample_categorical_source_tokens(
            state_ids.reshape(1, 1, diff_len),
            vocab_size=int(model.config.vocab_size),
            mask_token_id=int(model.config.mask_token_id),
            prior=source_prior,
            generator=source_generator,
        ).reshape(1, diff_len)
    source_time = torch.zeros((1, 1, 1, 1), device=device)
    target_time = torch.ones_like(source_time)
    embeds = model.model.embed_tokens(state_ids)
    conditioned = condition_flowdraft_state(
        model,
        embeds,
        source_time,
        target_time,
        block_size=diff_len,
        scale=float(getattr(model.config, "flowdraft_time_conditioning_scale", 0.0)),
    )
    positions = torch.arange(start_idx, start_idx + diff_len, device=device).unsqueeze(0)
    outputs = model(
        inputs_embeds=conditioned,
        position_ids=positions,
        past_key_values=past_key_values,
        use_cache=False,
        is_diffusion_pass=True,
        ar_seq_len=start_idx,
    )
    return sample_greedy(outputs.logits[:, :-1, :]) if diff_len > 1 else torch.empty((1, 0), dtype=torch.long, device=device)


@torch.inference_mode()
def generate_r2flow_greedy(model, corrector, input_ids, max_new_tokens: int, eos_token_id: int | None):
    device = input_ids.device
    block_size = int(model.config.block_size)
    mask_token_id = int(model.config.mask_token_id)
    past_key_values = DynamicCache(config=model.config)
    source_generator = torch.Generator(device=device).manual_seed(int(getattr(model.config, "flowdraft_source_seed", 17)))
    max_length = input_ids.shape[1] + max_new_tokens
    output_ids = torch.full((1, max_length + block_size), mask_token_id, dtype=torch.long, device=device)
    output_ids[:, : input_ids.shape[1]] = input_ids

    synchronize(device)
    started = time.perf_counter()
    prompt_positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    prefill = model(input_ids=input_ids, position_ids=prompt_positions, past_key_values=past_key_values, use_cache=True)
    forward_passes = 1
    start_idx = input_ids.shape[1]
    next_token = sample_greedy(prefill.logits[:, -1, :])
    output_ids[:, start_idx] = next_token
    generated_count = 1
    output_length = start_idx + 1
    first_pass_acceptance: list[int] = []
    final_acceptance: list[int] = []

    if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
        synchronize(device)
        return output_ids[:, :output_length], time.perf_counter() - started, forward_passes, first_pass_acceptance, final_acceptance

    while generated_count < max_new_tokens and start_idx < max_length - 1:
        diff_len = min(block_size, max_length - start_idx)
        proposal = draft_block(model, output_ids, start_idx, diff_len, past_key_values, source_generator)
        forward_passes += 1
        proposed_block = torch.cat([output_ids[:, start_idx : start_idx + 1], proposal], dim=1)
        positions = torch.arange(start_idx, start_idx + diff_len, device=device).unsqueeze(0)
        first = model(
            input_ids=proposed_block,
            position_ids=positions,
            past_key_values=past_key_values,
            use_cache=True,
            is_diffusion_pass=False,
            output_hidden_states=diff_len == block_size,
        )
        forward_passes += 1
        first_tokens = sample_greedy(first.logits)
        if proposal.numel():
            first_match = proposal == first_tokens[:, :-1]
            first_pass_acceptance.append(int(first_match.cumprod(dim=1).sum(dim=1)[0].item()))

        use_corrector = diff_len == block_size
        if use_corrector:
            verifier_hidden = first.hidden_states[-1][:, :-1, :]
            corrected_logits = corrector_logits(
                corrector=corrector,
                lm_head=model.lm_head,
                verifier_hidden=verifier_hidden,
                candidate_embeddings=model.model.embed_tokens(proposal),
                residual_embeddings=model.model.embed_tokens(first_tokens[:, :-1]),
                verifier_logits=first.logits[:, :-1, :],
            )
            corrected_tokens = sample_greedy(corrected_logits)
            past_key_values.crop(start_idx)
            final_block = torch.cat([output_ids[:, start_idx : start_idx + 1], corrected_tokens], dim=1)
            final = model(
                input_ids=final_block,
                position_ids=positions,
                past_key_values=past_key_values,
                use_cache=True,
                is_diffusion_pass=False,
            )
            forward_passes += 1
            final_tokens = sample_greedy(final.logits)
            matches = corrected_tokens == final_tokens[:, :-1]
            acceptance_len = int(matches.cumprod(dim=1).sum(dim=1)[0].item())
            accepted_block = final_block[:, : acceptance_len + 1]
            next_token = final_tokens[:, acceptance_len]
        else:
            final_block = proposed_block
            final_tokens = first_tokens
            acceptance_len = first_pass_acceptance[-1] if first_pass_acceptance else 0
            accepted_block = final_block[:, : acceptance_len + 1]
            next_token = final_tokens[:, acceptance_len]
        final_acceptance.append(acceptance_len)
        end_idx = start_idx + acceptance_len + 1
        output_ids[:, start_idx:end_idx] = accepted_block
        output_length = end_idx
        eos_positions = (accepted_block == eos_token_id).nonzero() if eos_token_id is not None else []
        if len(eos_positions) > 0:
            eos_offset = int(eos_positions[0, -1].item())
            output_length = start_idx + eos_offset + 1
            break
        generated_count += acceptance_len
        start_idx = end_idx
        past_key_values.crop(start_idx)
        if generated_count < max_new_tokens and start_idx < max_length:
            output_ids[:, start_idx] = next_token
            generated_count += 1
            output_length = start_idx + 1
            if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                break
    synchronize(device)
    return output_ids[:, :output_length], time.perf_counter() - started, forward_passes, first_pass_acceptance, final_acceptance


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_string(args.dtype)
    corrector, config = load_corrector(args.checkpoint, device, dtype)
    base_checkpoint = args.base_checkpoint or config["base_flowdraft_checkpoint"]
    model, metadata, _ = load_flowdraft_adapter(
        checkpoint_dir=base_checkpoint,
        upstream_dir=args.upstream_dir,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model = model.to(device=device, dtype=dtype).eval()
    tokenizer = load_tokenizer(metadata["base_model"])
    prompts = load_prompts(args.prompts_jsonl)
    for row in prompts[: args.warmup_prompts]:
        ids = encode_prompt(tokenizer, row["prompt"], device)
        generate_r2flow_greedy(model, corrector, ids, min(16, args.max_new_tokens), tokenizer.eos_token_id)

    rows = []
    for row in prompts:
        input_ids = encode_prompt(tokenizer, row["prompt"], device)
        ar_ids, ar_seconds, ar_forwards = generate_ar_greedy(model, input_ids, args.max_new_tokens, tokenizer.eos_token_id)
        r2_ids, r2_seconds, r2_forwards, first_acceptance, final_acceptance = generate_r2flow_greedy(
            model, corrector, input_ids, args.max_new_tokens, tokenizer.eos_token_id
        )
        ar_new = int(ar_ids.shape[1] - input_ids.shape[1])
        r2_new = int(r2_ids.shape[1] - input_ids.shape[1])
        parity = bool(torch.equal(ar_ids, r2_ids))
        mismatch = first_mismatch(ar_ids, r2_ids)
        record = {
            "id": row["id"], "parity_ok": parity, "first_mismatch_at": mismatch,
            "ar_tokens": ar_new, "r2flow_tokens": r2_new,
            "ar_seconds": ar_seconds, "r2flow_seconds": r2_seconds,
            "speedup": ((r2_new / r2_seconds) / (ar_new / ar_seconds)) if ar_seconds and r2_seconds and ar_new else 0.0,
            "ar_forward_passes": ar_forwards, "r2flow_forward_passes": r2_forwards,
            "r2flow_tpf": r2_new / r2_forwards if r2_forwards else 0.0,
            "first_verifier_acceptance": first_acceptance,
            "final_verifier_acceptance": final_acceptance,
            "final_acceptance_mean": statistics.mean(final_acceptance) if final_acceptance else 0.0,
        }
        rows.append(record)
        print(json.dumps(record), flush=True)
    total_ar_tokens = sum(r["ar_tokens"] for r in rows)
    total_r2_tokens = sum(r["r2flow_tokens"] for r in rows)
    total_ar_seconds = sum(r["ar_seconds"] for r in rows)
    total_r2_seconds = sum(r["r2flow_seconds"] for r in rows)
    total_r2_forwards = sum(r["r2flow_forward_passes"] for r in rows)
    all_final = [value for row in rows for value in row["final_verifier_acceptance"]]
    summary = {
        "method": "R2Flow residual-refined categorical flow draft",
        "num_prompts": len(rows), "checkpoint": str(Path(args.checkpoint).resolve()),
        "base_checkpoint": str(base_checkpoint), "dtype": args.dtype,
        "parity_rate": sum(r["parity_ok"] for r in rows) / len(rows) if rows else 0.0,
        "aggregate_speedup": ((total_r2_tokens / total_r2_seconds) / (total_ar_tokens / total_ar_seconds)) if total_ar_seconds and total_r2_seconds else 0.0,
        "aggregate_tpf": total_r2_tokens / total_r2_forwards if total_r2_forwards else 0.0,
        "weighted_final_acceptance": sum(all_final) / len(all_final) if all_final else 0.0,
        "p50_final_acceptance": percentile([float(v) for v in all_final], 0.5),
        "total_r2flow_forward_passes": total_r2_forwards,
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    if args.output_jsonl:
        with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    if args.summary_json:
        with Path(args.summary_json).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
    if args.require_parity and summary["parity_rate"] != 1.0:
        raise SystemExit(f"Parity check failed: {summary['parity_rate']:.6f}")


if __name__ == "__main__":
    main()
