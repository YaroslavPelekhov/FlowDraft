#!/usr/bin/env python
"""Materialize fixed AIME25 or HumanEval prompts for efficiency benchmarks.

The output is JSONL consumed by the strict FlowDraft benchmark.  It deliberately
keeps task targets as metadata: this script prepares generation prompts only;
task scoring belongs to the respective official harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


TASKS = {
    "aime25": {
        "dataset": "math-ai/aime25",
        "config": "default",
        "split": "test",
    },
    "humaneval": {
        "dataset": "openai/openai_humaneval",
        "config": "openai_humaneval",
        "split": "test",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare official benchmark records as fixed JSONL prompts.")
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default="main", help="Dataset revision; use a commit SHA to pin externally.")
    return parser.parse_args()


def dataset_sha(dataset: str, revision: str) -> str:
    """Resolve the revision once and store it with every prompt."""
    return HfApi().dataset_info(dataset, revision=revision).sha


def aime_prompt(problem: str) -> str:
    return (
        "Solve the following AIME 2025 problem. Show concise reasoning and end with "
        "the integer answer from 000 to 999.\n\nProblem:\n"
        f"{problem}"
    )


def humaneval_prompt(prompt: str) -> str:
    return (
        "Complete the following Python function. Return only executable Python code "
        "that continues the given function, without Markdown fences or explanation.\n\n"
        f"{prompt}"
    )


def main() -> None:
    args = parse_args()
    spec = TASKS[args.task]
    sha = dataset_sha(spec["dataset"], args.revision)
    dataset = load_dataset(
        spec["dataset"],
        spec["config"],
        split=spec["split"],
        revision=sha,
    )

    rows: list[dict] = []
    for index, example in enumerate(dataset):
        common = {
            "benchmark": args.task,
            "dataset": spec["dataset"],
            "dataset_revision": sha,
            "dataset_split": spec["split"],
        }
        if args.task == "aime25":
            rows.append({
                **common,
                "id": f"AIME25/{example['id']}",
                "prompt": aime_prompt(example["problem"]),
                "reference_answer": str(example["answer"]),
            })
        else:
            rows.append({
                **common,
                "id": str(example["task_id"]),
                "prompt": humaneval_prompt(example["prompt"]),
                "entry_point": str(example["entry_point"]),
                "canonical_solution": str(example["canonical_solution"]),
                "test": str(example["test"]),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"task": args.task, "examples": len(rows), "revision": sha, "output": str(output)}))


if __name__ == "__main__":
    main()
