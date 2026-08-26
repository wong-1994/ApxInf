#!/usr/bin/env python3
"""Deterministic GR00T N1.7 BF16 parity and latency check on Thor."""

import argparse
import time

import numpy as np

import apxinf_py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("fixture")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    fixture = np.load(args.fixture)
    tokens = fixture["token_ids"].reshape(-1).astype(np.uint32)
    attention = np.ones(tokens.shape, dtype=np.uint8)
    image_mask = (tokens == 151655).astype(np.uint8)
    inputs = (
        fixture["pixel_values"].astype(np.float32),
        fixture["grid_thw"].astype(np.uint32),
        tokens,
        attention,
        image_mask,
        fixture["state"].reshape(1, 132).astype(np.float32),
        0,
        fixture["noise"].reshape(40, 132).astype(np.float32),
    )
    expected = fixture["action"].reshape(40, 132).astype(np.float32)
    model = apxinf_py.Model.load("Gr00tN1d7", args.checkpoint, precision="bf16")
    for _ in range(args.warmup):
        model.infer_groot(*inputs)
    outputs = []
    times = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        outputs.append(np.asarray(model.infer_groot(*inputs)))
        times.append(1000.0 * (time.perf_counter() - started))
    output = outputs[0]
    delta = output - expected
    repeat_max = max(float(np.max(np.abs(value - output))) for value in outputs)
    print(
        "GR00T_N1D7_PARITY",
        f"shape={output.shape}",
        f"max_abs={np.max(np.abs(delta)):.8g}",
        f"mean_abs={np.mean(np.abs(delta)):.8g}",
        f"relative_l2={np.linalg.norm(delta) / np.linalg.norm(expected):.8g}",
        f"repeat_max_abs={repeat_max:.8g}",
        f"latency_ms={times}",
    )


if __name__ == "__main__":
    main()
