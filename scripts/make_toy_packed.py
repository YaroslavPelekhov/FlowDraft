#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orthrus_training.modeling import load_tokenizer


TEXTS = [
    "User: Solve 12 times 7. Assistant: 84.",
    "User: Write a Python function for Fibonacci numbers. Assistant: def fib(n): return [0, 1][:n].",
    "User: Explain why the sky is blue. Assistant: Rayleigh scattering makes shorter blue wavelengths scatter more.",
    "User: Translate hello to Russian. Assistant: privet.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create a tiny packed dataset for environment smoke tests.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", default="/tmp/flowdraft_storage/toy_packed")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-sequences", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model_name)

    ids: list[int] = []
    while len(ids) < args.seq_len * args.num_sequences:
        for text in TEXTS:
            ids.extend(tokenizer.encode(text, add_special_tokens=False))
            if tokenizer.eos_token_id is not None:
                ids.append(tokenizer.eos_token_id)

    arr = np.asarray(ids[: args.seq_len * args.num_sequences], dtype=np.int32)
    arr = arr.reshape(args.num_sequences, args.seq_len)
    np.save(output_dir / "toy-00000.npy", arr)

    manifest = {
        "dataset_name": "toy",
        "model_name": args.model_name,
        "seq_len": args.seq_len,
        "num_sequences": args.num_sequences,
        "num_tokens": args.num_sequences * args.seq_len,
        "shards": [
            {
                "file": "toy-00000.npy",
                "num_sequences": args.num_sequences,
                "seq_len": args.seq_len,
            }
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
