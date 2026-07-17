#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def disk_info(path: str) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "path": path,
        "total_gb": usage.total / (1024**3),
        "used_gb": usage.used / (1024**3),
        "free_gb": usage.free / (1024**3),
        "free_ratio": usage.free / usage.total if usage.total else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect disk and GPU resources for FlowDraft runs.")
    parser.add_argument("--paths", nargs="+", default=["/", "/tmp", "/dev/shm", "/home/jupyter"])
    parser.add_argument("--json", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = []
    for path in args.paths:
        if Path(path).exists():
            paths.append(disk_info(path))

    cuda = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        cuda["device_count"] = torch.cuda.device_count()
        cuda["devices"] = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            cuda["devices"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                }
            )

    report = {"disk": paths, "cuda": cuda}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("Disk:")
    for row in paths:
        print(
            f"  {row['path']}: free={row['free_gb']:.2f} GiB "
            f"used={row['used_gb']:.2f}/{row['total_gb']:.2f} GiB"
        )
    print("CUDA:")
    print(f"  available={cuda['available']}")
    for device in cuda.get("devices", []):
        print(f"  [{device['index']}] {device['name']} memory={device['total_memory_gb']:.2f} GiB")


if __name__ == "__main__":
    main()
