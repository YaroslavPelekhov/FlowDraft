#!/usr/bin/env python
"""Strict greedy benchmark for the identity-aware dynamic-support CFM."""

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

from benchmark_flowdraft import encode_prompt, first_mismatch, generate_ar_greedy, load_prompts, sample_greedy, synchronize
from orthrus_training.candidate_support import select_dynamic_candidate_support
from orthrus_training.flowdraft import condition_flowdraft_state, make_discrete_flowdraft_state, sample_categorical_source_tokens
from orthrus_training.modeling import dtype_from_string, load_flowdraft_adapter, load_tokenizer
from orthrus_training.simplex_flow import DynamicSupportSimplexFlowRefiner, make_token_codebook, simplex_flow_step


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
    with (path / "dynamic_support_flow_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("format") != "dynamic_support_categorical_flow_map_v1":
        raise ValueError(f"Unsupported checkpoint: {config.get('format')}")
    head = DynamicSupportSimplexFlowRefiner(
        int(config["block_size"]), int(config["candidate_count"]), int(config["draft_hidden_size"]),
        int(config["hidden_size"]), int(config["token_code_dim"]), int(config["num_layers"]),
        int(config["num_heads"]), float(config.get("dropout", 0.0)),
    )
    head.load_state_dict(load_file(path / "dynamic_support_flow.safetensors"), strict=True)
    return head.to(device=device, dtype=dtype).eval(), config


@torch.inference_mode()
def generate_dynamic_support_greedy(model, head, codebook, config, input_ids, max_new_tokens, eos_token_id, flow_steps, margin):
    device = input_ids.device
    block_size, mask_id = int(model.config.block_size), int(model.config.mask_token_id)
    source_prior = str(getattr(model.config, "flowdraft_source_prior", "uniform"))
    generator = torch.Generator(device=device).manual_seed(int(getattr(model.config, "flowdraft_source_seed", 17)))
    past = DynamicCache(config=model.config)
    max_length = input_ids.shape[1] + max_new_tokens
    output = torch.full((1, max_length + block_size), mask_id, dtype=torch.long, device=device)
    output[:, : input_ids.shape[1]] = input_ids
    synchronize(device); started = time.perf_counter()
    prefill = model(input_ids=input_ids, position_ids=torch.arange(input_ids.shape[1], device=device).unsqueeze(0), past_key_values=past, use_cache=True)
    forwards, head_calls = 1, 0
    start_idx = input_ids.shape[1]
    output[:, start_idx] = sample_greedy(prefill.logits[:, -1, :])
    generated, length, acceptance = 1, start_idx + 1, []
    if eos_token_id is not None and int(output[0, start_idx]) == int(eos_token_id):
        synchronize(device)
        return output[:, :length], time.perf_counter() - started, forwards, head_calls, acceptance

    while generated < max_new_tokens and start_idx < max_length - 1:
        diff_len = min(block_size, max_length - start_idx)
        positions = torch.arange(start_idx, start_idx + diff_len, device=device).unsqueeze(0)
        state_ids = make_discrete_flowdraft_state(output[:, start_idx:start_idx + 1], None, diff_len, mask_id)
        if source_prior != "mask":
            state_ids = sample_categorical_source_tokens(
                state_ids.reshape(1, 1, diff_len), int(model.config.vocab_size), mask_id,
                prior=source_prior, generator=generator,
            ).reshape(1, diff_len)
        raw = model.model.embed_tokens(state_ids)
        source_time = torch.zeros((1, 1, 1, 1), device=device)
        conditioned = condition_flowdraft_state(
            model, raw, source_time, torch.ones_like(source_time), diff_len,
            float(getattr(model.config, "flowdraft_time_conditioning_scale", 0.0)),
        )
        draft = model(inputs_embeds=conditioned, position_ids=positions, past_key_values=past, use_cache=False, is_diffusion_pass=True, ar_seq_len=start_idx)
        forwards += 1
        if diff_len > 1:
            base_logits = draft.logits[:, :-1, :]
            hidden = draft.hidden_states[-1][:, :-1, :]
            scores = head.retrieval_scores(hidden, codebook)
            retrieved = scores.topk(int(config["retrieval_count"]), dim=-1).indices
            values, candidates = select_dynamic_candidate_support(
                base_logits, int(config["candidate_count"]), retrieved, int(config["base_candidate_count"])
            )
            if diff_len == block_size:
                state = torch.full_like(values, 1.0 / head.candidate_count)
                values4 = values.reshape(1, 1, diff_len - 1, head.candidate_count)
                ids4 = candidates.reshape(1, 1, diff_len - 1, head.candidate_count)
                hidden4 = hidden.reshape(1, 1, diff_len - 1, -1)
                for index in range(max(1, flow_steps)):
                    source = torch.full(values.shape[:-1], index / max(1, flow_steps), device=device)
                    target = torch.full(values.shape[:-1], (index + 1) / max(1, flow_steps), device=device)
                    endpoint = head(
                        values4, state.reshape(1, 1, diff_len - 1, -1), source.reshape(1, 1, diff_len - 1),
                        target.reshape(1, 1, diff_len - 1), ids4, hidden4, codebook,
                    ).reshape_as(values)
                    state = simplex_flow_step(state, endpoint, source, target)
                    head_calls += 1
                proposal = candidates.gather(-1, state.argmax(dim=-1, keepdim=True)).squeeze(-1)
            else:
                proposal = candidates[:, :, 0]
        else:
            proposal = torch.empty((1, 0), dtype=torch.long, device=device)
        proposed = torch.cat([output[:, start_idx:start_idx + 1], proposal], dim=1)
        verifier = model(input_ids=proposed, position_ids=positions, past_key_values=past, use_cache=True, is_diffusion_pass=False)
        forwards += 1
        verifier_tokens = sample_greedy(verifier.logits)
        matches = proposal.eq(verifier_tokens[:, :-1])
        accepted = int(matches.cumprod(dim=1).sum(dim=1)[0].item()) if proposal.numel() else 0
        margins = verifier.logits.float().topk(2, dim=-1).values
        uncertain = (margins[0, :accepted + 1, 0] - margins[0, :accepted + 1, 1] < margin).nonzero() if margin > 0 else []
        if len(uncertain) > 0:
            accepted = int(uncertain[0, 0].item())
            past.crop(start_idx + accepted)
            sequential = model(input_ids=proposed[:, accepted:accepted + 1], position_ids=torch.tensor([[start_idx + accepted]], device=device), past_key_values=past, use_cache=True, is_diffusion_pass=False)
            forwards += 1
            next_token = sample_greedy(sequential.logits[:, -1, :])
        else:
            next_token = verifier_tokens[:, accepted]
        acceptance.append(accepted)
        accepted_block = proposed[:, :accepted + 1]
        eos_positions = (accepted_block == eos_token_id).nonzero() if eos_token_id is not None else []
        if len(eos_positions) > 0:
            length = start_idx + int(eos_positions[0, -1].item()) + 1
            output[:, start_idx:length] = accepted_block[:, : length - start_idx]
            break
        end_idx = start_idx + accepted + 1
        output[:, start_idx:end_idx] = accepted_block
        length, generated, start_idx = end_idx, generated + accepted, end_idx
        past.crop(start_idx)
        if generated < max_new_tokens and start_idx < max_length:
            output[:, start_idx] = next_token
            generated += 1; length = start_idx + 1
            if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                break
    synchronize(device)
    return output[:, :length], time.perf_counter() - started, forwards, head_calls, acceptance


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_string(args.dtype)
    head, config = load_head(args.checkpoint, device, dtype)
    parent = args.base_checkpoint or config["base_flowdraft_checkpoint"]
    model, metadata, _ = load_flowdraft_adapter(parent, args.upstream_dir, dtype, args.attn_implementation)
    model = model.to(device=device, dtype=dtype).eval()
    codebook = make_token_codebook(model.model.embed_tokens.weight, int(config["token_code_dim"]), int(config["token_code_seed"]))
    tokenizer, prompts = load_tokenizer(metadata["base_model"]), load_prompts(args.prompts_jsonl)
    for row in prompts[:args.warmup_prompts]:
        generate_dynamic_support_greedy(model, head, codebook, config, encode_prompt(tokenizer, row["prompt"], device), min(16, args.max_new_tokens), tokenizer.eos_token_id, args.flow_steps, args.parity_margin_threshold)
    rows = []
    for row in prompts:
        encoded = encode_prompt(tokenizer, row["prompt"], device)
        ar_ids, ar_seconds, ar_forwards = generate_ar_greedy(model, encoded, args.max_new_tokens, tokenizer.eos_token_id)
        draft_ids, seconds, forwards, head_calls, accepts = generate_dynamic_support_greedy(model, head, codebook, config, encoded, args.max_new_tokens, tokenizer.eos_token_id, args.flow_steps, args.parity_margin_threshold)
        ar_new, draft_new = int(ar_ids.shape[1] - encoded.shape[1]), int(draft_ids.shape[1] - encoded.shape[1])
        record = {"id": row["id"], "parity_ok": bool(torch.equal(ar_ids, draft_ids)), "first_mismatch_at": first_mismatch(ar_ids, draft_ids), "ar_tokens": ar_new, "dynamic_support_tokens": draft_new, "ar_seconds": ar_seconds, "dynamic_support_seconds": seconds, "speedup": ((draft_new / seconds) / (ar_new / ar_seconds)) if ar_new and ar_seconds and seconds else 0.0, "ar_forward_passes": ar_forwards, "qwen_forward_passes": forwards, "cfm_head_calls": head_calls, "tpf": draft_new / forwards if forwards else 0.0, "acceptance": accepts, "acceptance_mean": statistics.mean(accepts) if accepts else 0.0}
        rows.append(record); print(json.dumps(record), flush=True)
    total_ar_tokens, total_tokens = sum(x["ar_tokens"] for x in rows), sum(x["dynamic_support_tokens"] for x in rows)
    total_ar_seconds, total_seconds = sum(x["ar_seconds"] for x in rows), sum(x["dynamic_support_seconds"] for x in rows)
    total_forwards = sum(x["qwen_forward_passes"] for x in rows)
    accepted = [value for row in rows for value in row["acceptance"]]
    summary = {"method": "identity-aware dynamic-support categorical flow map", "checkpoint": str(Path(args.checkpoint).resolve()), "base_checkpoint": str(parent), "num_prompts": len(rows), "flow_steps": args.flow_steps, "parity_rate": sum(x["parity_ok"] for x in rows) / len(rows) if rows else 0.0, "aggregate_speedup": ((total_tokens / total_seconds) / (total_ar_tokens / total_ar_seconds)) if total_tokens and total_seconds and total_ar_seconds else 0.0, "aggregate_tpf": total_tokens / total_forwards if total_forwards else 0.0, "weighted_acceptance": sum(accepted) / len(accepted) if accepted else 0.0, "total_qwen_forward_passes": total_forwards, "total_cfm_head_calls": sum(x["cfm_head_calls"] for x in rows)}
    print("SUMMARY " + json.dumps(summary), flush=True)
    if args.output_jsonl:
        Path(args.output_jsonl).write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    if args.summary_json:
        write_json(Path(args.summary_json), summary)
    if args.require_parity and summary["parity_rate"] != 1.0:
        raise SystemExit(f"Parity check failed: {summary['parity_rate']:.6f}")


if __name__ == "__main__":
    main()
