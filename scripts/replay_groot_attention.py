#!/usr/bin/env python3
"""Replay captured GR00T attention variants through the ApxInf CUDA ABI."""

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--causal-variants", type=int, nargs="*", default=[])
    args = parser.parse_args()

    metadata = json.loads((args.capture_dir / "attention-capture.json").read_text())
    arrays = np.load(args.capture_dir / "attention-capture.npz")
    library = ctypes.CDLL(str(args.library))
    replay = library.apxinf_cross_sdpa_bf16
    replay.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint32] * 6 + [ctypes.c_float, ctypes.c_void_p]
    replay.restype = ctypes.c_int
    stream = torch.cuda.current_stream().cuda_stream
    comparisons = []

    for variant in metadata["variants"]:
        index = variant["id"]
        prefix = f"attention.variant.{index}"
        q = torch.from_numpy(arrays[f"{prefix}.q"]).to(device="cuda", dtype=torch.bfloat16)
        k = torch.from_numpy(arrays[f"{prefix}.k"]).to(device="cuda", dtype=torch.bfloat16)
        v = torch.from_numpy(arrays[f"{prefix}.v"]).to(device="cuda", dtype=torch.bfloat16)
        # Captures are [batch, heads, tokens, dim]; the ABI is [tokens, heads, dim].
        q = q[0].permute(1, 0, 2).contiguous()
        k = k[0].permute(1, 0, 2).contiguous()
        v = v[0].permute(1, 0, 2).contiguous()
        output = torch.empty_like(q)
        mask = None
        if variant["mask_shape"] is not None:
            captured_mask = arrays[f"{prefix}.mask"]
            mask = torch.from_numpy(captured_mask[0, 0, 0].astype(np.uint8)).cuda()
        status = replay(
            q.data_ptr(), k.data_ptr(), v.data_ptr(),
            0 if mask is None else mask.data_ptr(), output.data_ptr(),
            q.shape[0], k.shape[0], q.shape[1], k.shape[1], q.shape[2],
            int(index in args.causal_variants),
            1.0 / (q.shape[2] ** 0.5), stream,
        )
        if status != 0:
            raise RuntimeError(f"variant {index}: CUDA status {status}")
        torch.cuda.synchronize()
        actual = output.float().cpu().numpy()
        reference = arrays[f"{prefix}.output"][0]
        if variant["provider"].startswith("transformers"):
            reference = reference.reshape(actual.shape)
        else:
            reference = reference.transpose(1, 0, 2)
        difference = np.abs(actual - reference)
        tolerance = args.atol + args.rtol * np.abs(reference)
        comparisons.append({
            "variant": index,
            "provider": variant["provider"],
            "q_shape": variant["q_shape"],
            "k_shape": variant["k_shape"],
            "mask_shape": variant["mask_shape"],
            "causal": index in args.causal_variants,
            "max_abs_error": float(difference.max()),
            "max_rel_error": float((difference / np.maximum(np.abs(reference), 1e-12)).max()),
            "max_tolerance_excess": float((difference - tolerance).max()),
            "passed": bool(np.all(difference <= tolerance)),
        })

    result = {
        "schema_version": "1.0",
        "source_capture": str(args.capture_dir),
        "comparison_rule": "abs(actual-reference) <= atol + rtol*abs(reference)",
        "atol": args.atol,
        "rtol": args.rtol,
        "causal_variants": args.causal_variants,
        "comparisons": comparisons,
        "passed": all(item["passed"] for item in comparisons),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
