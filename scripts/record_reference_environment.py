#!/usr/bin/env python3
"""Record objective provenance for an Agent-prepared reference environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    distributions = sorted(
        (
            {"name": item.metadata["Name"], "version": item.version}
            for item in importlib.metadata.distributions()
            if item.metadata["Name"]
        ),
        key=lambda item: item["name"].lower(),
    )
    record = {
        "schema_version": "1.0",
        "python": platform.python_version(),
        "dependency_lock": {
            "path": args.dependency_lock.name,
            "sha256": sha256(args.dependency_lock),
        },
        "isolation": {
            "kind": "agent_prepared",
            "environment_id": hashlib.sha256(sys.executable.encode()).hexdigest()[:24],
            "system_site_packages": True,
        },
        "installed_distributions": distributions,
        "runtime_network_access": False,
        "network_enforcement": ["adapter_socket_guard"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
