#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_PROMPTS = [
    "Solve: if a rectangle has length 12 and width 7, what is its area?",
    "Write a Python function that returns the Fibonacci sequence up to n terms.",
    "Explain why the sky appears blue in two short paragraphs.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Check greedy lossless parity between AR and Orthrus modes.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts-jsonl", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--attn-implementation", default="eager")
    return parser.parse_args()


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompts.append(row["prompt"] if isinstance(row, dict) else str(row))
    return prompts


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()

    mismatches = 0
    for prompt in load_prompts(args.prompts_jsonl):
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(model.device)

        started = time.perf_counter()
        ar = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            use_diffusion_mode=False,
        )
        ar_time = time.perf_counter() - started

        started = time.perf_counter()
        diffusion = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            use_diffusion_mode=True,
        )
        diffusion_time = time.perf_counter() - started

        same = torch.equal(ar, diffusion)
        mismatches += int(not same)
        print(
            json.dumps(
                {
                    "prompt": prompt,
                    "same": same,
                    "ar_tokens": int(ar.shape[1] - input_ids.shape[1]),
                    "diffusion_tokens": int(diffusion.shape[1] - input_ids.shape[1]),
                    "ar_seconds": ar_time,
                    "diffusion_seconds": diffusion_time,
                },
                ensure_ascii=False,
            )
        )

    if mismatches:
        raise SystemExit(f"{mismatches} prompts did not match")


if __name__ == "__main__":
    main()
