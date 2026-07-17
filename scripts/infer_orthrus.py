#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
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
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description="Run interactive inference with an Orthrus checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--mode", choices=["diffusion", "ar"], default="diffusion")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def read_prompt(args) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Pass --prompt, --prompt-file, or pipe a prompt on stdin.")


def torch_dtype(name: str, device: str):
    if device != "cuda":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def encode_prompt(tokenizer, prompt: str, system_prompt: str, use_chat_template: bool, device) -> torch.Tensor:
    if use_chat_template:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        encoded = tokenizer(prompt, return_tensors="pt")

    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    return encoded.to(device)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch_dtype(args.dtype, device),
        device_map=device,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()

    prompt = read_prompt(args).strip()
    input_ids = encode_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        system_prompt=args.system_prompt,
        use_chat_template=not args.no_chat_template,
        device=model.device,
    )

    streamer = None
    if not args.no_stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    started = time.perf_counter()
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        use_diffusion_mode=args.mode == "diffusion",
        streamer=streamer,
    )
    elapsed = time.perf_counter() - started

    new_tokens = int(output_ids.shape[1] - input_ids.shape[1])
    generated_ids = output_ids[:, input_ids.shape[1] :]
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if args.no_stream:
        print(text)
    else:
        print()

    print(
        "INFERENCE "
        + json.dumps(
            {
                "checkpoint": args.checkpoint,
                "mode": args.mode,
                "new_tokens": new_tokens,
                "seconds": elapsed,
                "tokens_per_second": new_tokens / elapsed if elapsed > 0 else 0.0,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
