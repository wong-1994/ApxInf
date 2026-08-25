#!/usr/bin/env python3
"""Capture machine-readable NVIDIA Thor qualification environment evidence."""
import json
import platform
import subprocess
from pathlib import Path

import torch

def command(*args: str) -> str:
    completed = subprocess.run(args, text=True, capture_output=True, timeout=10)
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError(f"environment command failed: {' '.join(args)}")
    return completed.stdout.strip()

def main() -> None:
    properties = torch.cuda.get_device_properties(0)
    driver_path = Path("/proc/driver/nvidia/version")
    driver = driver_path.read_text(encoding="utf-8").strip()
    power_mode = command("nvpmodel", "-q")
    telemetry_run = subprocess.run(
        ("timeout", "1", "tegrastats", "--interval", "100"),
        text=True, capture_output=True, timeout=3,
    )
    telemetry = telemetry_run.stdout.strip().splitlines()[-1] if telemetry_run.stdout.strip() else ""
    if "@" not in telemetry or "CPU [" not in telemetry:
        raise RuntimeError("tegrastats did not report temperature and clocks")
    print(json.dumps({
        "available": torch.cuda.is_available(), "device": properties.name,
        "capability": [properties.major, properties.minor], "total_memory": properties.total_memory,
        "driver": driver, "cuda": torch.version.cuda,
        "libraries": {"torch": torch.__version__, "cudnn": torch.backends.cudnn.version()},
        "kernel_build": platform.release(), "power_mode": power_mode,
        "clocks_and_temperature": telemetry,
    }, sort_keys=True))

if __name__ == "__main__":
    main()
