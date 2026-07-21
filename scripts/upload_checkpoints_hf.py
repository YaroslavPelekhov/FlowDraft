#!/usr/bin/env python3
"""Upload best/last checkpoints and run metadata to a private HF model repo."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID"))
    parser.add_argument("--run-path", default=os.environ.get("HF_RUN_PATH"))
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--commit-message", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.repo_id:
        raise SystemExit("Set --repo-id or HF_REPO_ID")
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    missing = [name for name in ("best", "last") if not (run_dir / name).is_dir()]
    if missing:
        raise SystemExit(f"Missing checkpoint directories: {', '.join(missing)}")

    run_path = args.run_path or f"runs/{run_dir.name}"
    manifest = {
        "uploaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "run_path": run_path,
        "run_dir": run_dir.name,
        "checkpoint_paths": [f"{run_path}/best", f"{run_path}/last"],
    }
    with (run_dir / "hf_upload_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    message = args.commit_message or f"Archive FlowDraft run {run_dir.name}"
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(run_dir),
        path_in_repo=run_path,
        commit_message=message,
        ignore_patterns=["*.tmp", "trainer_state/*"],
    )

    print(f"Uploaded best/last and run metadata to {args.repo_id}/{run_path}")


if __name__ == "__main__":
    main()
