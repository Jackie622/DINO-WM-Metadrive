#!/usr/bin/env python3
"""Check which DLP scene JSON bundles are complete."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


REQUIRED_SUFFIXES = {
    "agents": "_agents.json",
    "frames": "_frames.json",
    "instances": "_instances.json",
    "obstacles": "_obstacles.json",
    "scene": "_scene.json",
}


def discover_prefixes(data_root: Path) -> dict[str, set[str]]:
    found: dict[str, set[str]] = defaultdict(set)
    for path in data_root.glob("*.json"):
        for name, suffix in REQUIRED_SUFFIXES.items():
            if path.name.endswith(suffix):
                prefix = path.name[: -len(suffix)]
                found[prefix].add(name)
                break
    return dict(sorted(found.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dlp_dataset/data", type=Path)
    args = parser.parse_args()

    data_root = args.data_root
    if not data_root.exists():
        print(f"[DLP] data root not found: {data_root}")
        return 1

    prefixes = discover_prefixes(data_root)
    if not prefixes:
        print(f"[DLP] no scene JSON files found under {data_root}")
        return 1

    complete = []
    print(f"[DLP] data root: {data_root.resolve()}")
    for prefix, pieces in prefixes.items():
        missing = sorted(set(REQUIRED_SUFFIXES) - pieces)
        status = "OK" if not missing else "MISSING " + ",".join(missing)
        print(f"{prefix}: {status}")
        if not missing:
            complete.append(prefix)

    print(f"[DLP] complete scenes: {len(complete)}/{len(prefixes)}")
    if complete:
        print("[DLP] load prefixes:")
        for prefix in complete:
            print(f"  {data_root / prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
