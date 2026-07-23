#!/usr/bin/env python
"""Strict single-block FlowTree smoke test on a frozen FlowDraft checkpoint."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orthrus_training.flowdraft import (
    condition_flowdraft_state,
    make_discrete_flowdraft_state,
    sample_categorical_source_tokens,
)
from orthrus_training.flowtree import build_flowtree, leaf_paths
from orthrus_training.modeling import load_flowdraft_adapter, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strict FlowTree verifier smoke test.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream-dir", default="upstream_orthrus")
    parser.add_argument("--prompt", default="Write a Python function that returns the square of an integer.")
    parser.add_argument("--branch-width", type=int, default=2)
    parser.add_argument("--branch-depth", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    return parser.parse_args()


def encode_prompt(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    return encoded.to(device)


@torch.inference_mode()
def draft_endpoint(model, anchor: torch.Tensor, start_idx: int, past_key_values) -> torch.Tensor:
    """One CFM endpoint pass, matching the deployment FlowDraft proposal path."""

    device = anchor.device
    diff_len = int(model.config.block_size)
    mask_token_id = int(model.config.mask_token_id)
    source_prior = str(getattr(model.config, "flowdraft_source_prior", "mask"))
    source_seed = int(getattr(model.config, "flowdraft_source_seed", 17))
    state_ids = make_discrete_flowdraft_state(anchor, None, diff_len, mask_token_id)
    if source_prior != "mask":
        state_ids = sample_categorical_source_tokens(
            state_ids.reshape(1, 1, diff_len),
            vocab_size=int(model.config.vocab_size),
            mask_token_id=mask_token_id,
            prior=source_prior,
            generator=torch.Generator(device=device).manual_seed(source_seed),
        ).reshape(1, diff_len)
    source = torch.zeros((1, 1, 1, 1), device=device)
    target = torch.ones((1, 1, 1, 1), device=device)
    embeds = condition_flowdraft_state(
        model,
        model.model.embed_tokens(state_ids),
        source,
        target,
        block_size=diff_len,
        scale=float(getattr(model.config, "flowdraft_time_conditioning_scale", 0.0)),
    )
    positions = torch.arange(start_idx, start_idx + diff_len, device=device).unsqueeze(0)
    # The cache contains tokens before ``anchor``. Diffusion is bidirectional
    # within its proposal block, exactly as in ordinary FlowDraft inference.
    output = model(
        inputs_embeds=embeds,
        position_ids=positions,
        past_key_values=past_key_values,
        use_cache=False,
        is_diffusion_pass=True,
        ar_seq_len=start_idx,
    )
    return output.logits[0, :-1]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model, metadata, _ = load_flowdraft_adapter(
        args.checkpoint,
        upstream_dir=args.upstream_dir,
        dtype=dtype,
        attn_implementation="eager",
    )
    tokenizer = load_tokenizer(metadata["base_model"])
    model = model.to(device=device, dtype=dtype).eval()
    prompt_ids = encode_prompt(tokenizer, args.prompt, device)
    past = DynamicCache(config=model.config)
    prefill = model(input_ids=prompt_ids, past_key_values=past, use_cache=True)
    anchor = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    start_idx = int(prompt_ids.shape[1])

    # The proposal path reads this cache but never mutates it.
    endpoint_logits = draft_endpoint(model, anchor, start_idx, past)
    tree = build_flowtree(
        int(anchor.item()),
        endpoint_logits,
        branch_width=args.branch_width,
        branch_depth=args.branch_depth,
        max_nodes=args.max_nodes,
    )
    paths = leaf_paths(tree)
    path_tokens = tree.token_ids[paths]
    path_len = int(path_tokens.shape[1])
    branch_cache = copy.deepcopy(past)
    branch_cache.batch_repeat_interleave(path_tokens.shape[0])
    positions = (
        start_idx + torch.arange(path_len, device=device)
    ).unsqueeze(0).expand(path_tokens.shape[0], -1)
    # This is an ordinary causal AR forward, batched over FlowTree leaves.
    # It is the conservative v1 verifier and has no custom attention kernel.
    branch_output = model(
        input_ids=path_tokens,
        position_ids=positions,
        past_key_values=branch_cache,
        use_cache=True,
        is_diffusion_pass=False,
    )
    branch_tokens = branch_output.logits.argmax(dim=-1)
    matches = path_tokens[:, 1:] == branch_tokens[:, :-1]
    accepted = matches.to(torch.int64).cumprod(dim=-1).sum(dim=-1)
    selected_branch = int(accepted.argmax().item())
    acceptance = int(accepted[selected_branch].item())

    # Strictly compare the winning batched branch with an ordinary single-path
    # replay before allowing this verifier to become a decoding primitive.
    replay_cache = copy.deepcopy(past)
    replay = model(
        input_ids=path_tokens[selected_branch : selected_branch + 1],
        position_ids=positions[selected_branch : selected_branch + 1],
        past_key_values=replay_cache,
        use_cache=True,
        is_diffusion_pass=False,
    )
    exact = bool(torch.equal(branch_tokens[selected_branch], replay.logits[0].argmax(dim=-1)))
    mismatch = (branch_tokens[selected_branch] != replay.logits[0].argmax(dim=-1)).nonzero(as_tuple=False).flatten()
    mismatch_at = int(mismatch[0]) if mismatch.numel() else None
    print(
        "FLOWTREE_SMOKE "
        f"nodes={tree.num_nodes} leaves={path_tokens.shape[0]} "
        f"best_accepted={acceptance} batched_vs_replay_parity={exact}"
    )
    print(
        f"FLOWTREE_PATH selected_leaf={selected_branch} "
        f"batched_tokens={branch_tokens[selected_branch].tolist()} "
        f"replay_tokens={replay.logits[0].argmax(dim=-1).tolist()} mismatch_at={mismatch_at}"
    )
    if not exact:
        raise SystemExit("FlowTree verifier disagrees with linear AR replay")


if __name__ == "__main__":
    main()
