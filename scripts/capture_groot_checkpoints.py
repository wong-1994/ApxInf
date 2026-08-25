#!/usr/bin/env python3
"""Capture fixed GR00T backbone/action checkpoints for differential replay."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


def load_adapter(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("apxinf_groot_checkpoint_adapter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def array(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().contiguous().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter = load_adapter(args.source_root / "apxinf_reference.py")
    adapter.set_seed(0)
    model = adapter.load(str(args.checkpoint))
    records: dict[str, np.ndarray] = {}
    handles = []
    counters: dict[str, int] = {}

    def tensor_output(output):
        if torch.is_tensor(output):
            return output
        if hasattr(output, "data") and isinstance(output.data, dict):
            return output.data.get("backbone_features")
        if isinstance(output, dict):
            return output.get("backbone_features")
        return None

    def capture_once(name: str):
        def hook(_module, _inputs, output):
            value = tensor_output(output)
            if torch.is_tensor(value):
                records[name] = array(value)
            if hasattr(output, "data") and isinstance(output.data, dict):
                for mask_name in ("backbone_attention_mask", "image_mask"):
                    mask = output.data.get(mask_name)
                    if torch.is_tensor(mask):
                        records[f"{name}.{mask_name}"] = array(mask)
        return hook

    def capture_per_flow(name: str):
        def hook(_module, _inputs, output):
            value = tensor_output(output)
            if not torch.is_tensor(value):
                return
            step = counters.get(name, 0)
            counters[name] = step + 1
            records[f"flow.step.{step}.{name}"] = array(value)
        return hook

    handles.append(model.backbone.register_forward_hook(capture_once("backbone.hidden")))
    handles.append(model.action_head.vlln.register_forward_hook(capture_once("backbone.vlln")))
    for index, block in enumerate(model.action_head.vl_self_attention.transformer_blocks):
        handles.append(block.register_forward_hook(capture_once(f"backbone.vl_block.{index}")))
    handles.append(model.action_head.state_encoder.register_forward_hook(capture_once("state.embedding")))
    handles.append(model.action_head.action_encoder.register_forward_hook(capture_per_flow("action.embedding")))
    handles.append(model.action_head.model.timestep_encoder.register_forward_hook(capture_per_flow("timestep.embedding")))
    for index, block in enumerate(model.action_head.model.transformer_blocks):
        handles.append(block.register_forward_hook(capture_per_flow(f"dit.block.{index}")))
    handles.append(model.action_head.model.proj_out_2.register_forward_hook(capture_per_flow("velocity.predecode")))
    handles.append(model.action_head.action_decoder.register_forward_hook(capture_per_flow("velocity.decoded")))

    profile = {"name": "libero-256", "inputs": {"image": [1, 1, 256, 256, 3], "state": [1, 1, 7]}}
    inputs = adapter.preprocess(profile)
    for name, value in inputs.items():
        if torch.is_tensor(value):
            records[f"input.{name}"] = array(value)
    output = adapter.infer(model, inputs)
    records["normalized_actions"] = array(output["actions"])
    for handle in handles:
        handle.remove()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "checkpoint-capture.npz", **records)
    metadata = {
        "schema_version": "1.0", "seed": 0,
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in records.items()},
    }
    (args.output / "checkpoint-capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"arrays": len(records), "flow_steps": counters.get("action.embedding", 0)}))


if __name__ == "__main__":
    main()
