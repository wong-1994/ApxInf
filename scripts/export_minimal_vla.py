#!/usr/bin/env python3
"""Deterministically export the private minimal VLA source weights.

The source layout follows training-framework Linear convention ``[out, in]``.
The Rust loader records and applies the corresponding transpose at load time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


PARAMETERS = {
    "action_projection.weight": ([2, 2], [1.0, 0.0, 0.0, 1.0], "transpose [out,in] to [in,out]"),
    "token_embedding.weight": ([4, 2], [0.0, 0.0, 0.25, -0.25, 0.5, 0.5, -0.5, 0.25], "identity"),
    "vision_projection.weight": ([2, 3], [0.5, 0.0, -0.5, 0.0, 0.25, 0.25], "transpose [out,in] to [in,out]"),
}


def bf16_bytes(values: list[float]) -> bytes:
    return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16) for value in values)


def export(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {"__metadata__": {"format": "pt", "model_type": "minimal_vla"}}
    payload = bytearray()
    evidence = []
    for name in sorted(PARAMETERS):
        shape, values, transformation = PARAMETERS[name]
        data = bf16_bytes(values)
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, len(payload)]}
        evidence.append({"parameter": name, "dtype": "BF16", "shape": shape, "transformation": transformation, "sha256": hashlib.sha256(data).hexdigest()})
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    weights = struct.pack("<Q", len(encoded)) + encoded + payload
    weights_path = output / "model.safetensors"
    weights_path.write_bytes(weights)
    config = {"model_type": "minimal_vla", "type": "minimal_vla", "image_size": 1, "num_views": 1, "action_horizon": 1, "action_dim": 2, "vocab_size": 4, "max_token_len": 1}
    (output / "config.json").write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
    manifest = {"format": "apxinf-minimal-vla-export-v1", "deterministic_seed": 0, "source_layout": "synthetic-pytorch", "requested_tuples": [{"target": "nvidia-thor", "precision": "bf16"}], "parameters": evidence, "parameter_count": len(evidence), "weights_sha256": hashlib.sha256(weights).hexdigest()}
    (output / "export-manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    export(parser.parse_args().output)


if __name__ == "__main__":
    main()
