#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orthrus_training.modeling import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Pack Nemotron chat/math/code examples into fixed token shards.")
    parser.add_argument("--dataset-name", default="nvidia/Nemotron-Post-Training-Dataset-v2")
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--splits", nargs="+", default=["chat", "math", "code"])
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", default="data/packed_qwen3_1p7b")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-sequences", type=int, default=50000)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_split(dataset_name: str, dataset_config: str | None, split: str, streaming: bool, token: str | None):
    try:
        return load_dataset(dataset_name, dataset_config, split=split, streaming=streaming, token=token)
    except Exception:
        if dataset_config and dataset_config != "SFT":
            return load_dataset(dataset_name, "SFT", split=split, streaming=streaming, token=token)
        raise


def example_to_text(example: dict, tokenizer) -> str:
    messages = example.get("messages")
    if messages:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    for key in ("text", "content", "prompt"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return json.dumps(example, ensure_ascii=False)


def save_shard(output_dir: Path, shard_idx: int, rows: list[np.ndarray], manifest: dict) -> None:
    arr = np.stack(rows).astype(np.int32, copy=False)
    filename = f"train-{shard_idx:05d}.npy"
    np.save(output_dir / filename, arr)
    manifest["shards"].append(
        {
            "file": filename,
            "num_sequences": int(arr.shape[0]),
            "seq_len": int(arr.shape[1]),
        }
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model_name)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    datasets = {
        split: iter(load_split(args.dataset_name, args.dataset_config, split, args.streaming, hf_token))
        for split in args.splits
    }

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "splits": args.splits,
        "model_name": args.model_name,
        "seq_len": args.seq_len,
        "shards": [],
    }

    buffer: list[int] = []
    rows: list[np.ndarray] = []
    shard_idx = 0
    produced = 0
    split_cursor = 0

    progress = tqdm(total=args.max_sequences, desc="packed sequences")
    while produced < args.max_sequences:
        split = args.splits[split_cursor % len(args.splits)]
        split_cursor += 1
        example = next(datasets[split])
        text = example_to_text(example, tokenizer)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)
        buffer.extend(token_ids)

        while len(buffer) >= args.seq_len and produced < args.max_sequences:
            rows.append(np.asarray(buffer[: args.seq_len], dtype=np.int32))
            del buffer[: args.seq_len]
            produced += 1
            progress.update(1)

            if len(rows) >= args.shard_size:
                save_shard(output_dir, shard_idx, rows, manifest)
                rows = []
                shard_idx += 1

    if rows:
        save_shard(output_dir, shard_idx, rows, manifest)

    progress.close()
    manifest["num_sequences"] = produced
    manifest["num_tokens"] = produced * args.seq_len
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
