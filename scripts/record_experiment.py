#!/usr/bin/env python3
"""Copy reproducibility artifacts from a run into a Git-trackable record."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_FILES = {
    "best_metrics.json",
    "last_metrics.json",
    "run_config.json",
    "run_manifest.json",
    "checkpoint_index.jsonl",
    "train_metrics.jsonl",
    "run.log",
    "preflight.txt",
    "nvidia-smi.txt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Completed run directory.")
    parser.add_argument("--record-id", default=None, help="Git record directory name.")
    parser.add_argument("--output-root", default="experiments", help="Git-tracked record root.")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Additional file name or relative path to copy from the run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source_dir).resolve()
    if not source.is_dir():
        raise SystemExit(f"Run directory does not exist: {source}")

    repo_root = Path(__file__).resolve().parents[1]
    record_id = args.record_id or source.name
    destination = (repo_root / args.output_root / record_id).resolve()
    if destination == repo_root or repo_root not in destination.parents:
        raise SystemExit("Record destination must be inside the repository")
    destination.mkdir(parents=True, exist_ok=True)

    relative_files = set(ALLOWED_FILES) | set(args.include)
    relative_files.update(
        candidate.name
        for candidate in source.iterdir()
        if candidate.is_file() and candidate.suffix in {".json", ".jsonl", ".txt"}
    )
    copied: list[dict[str, object]] = []
    for relative in sorted(relative_files):
        source_file = source / relative
        if not source_file.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
        copied.append(
            {
                "path": relative,
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )

    record = {
        "record_id": record_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir_name": source.name,
        "files": copied,
        "git": {
            "commit": git_value(repo_root, "rev-parse", "HEAD"),
            "branch": git_value(repo_root, "branch", "--show-current"),
            "status": git_value(repo_root, "status", "--short"),
        },
    }
    with (destination / "record.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Recorded {len(copied)} reproducibility files at {destination}")


if __name__ == "__main__":
    main()
