#!/usr/bin/env python3
"""Export the selected GR00T operator capture tensors as simple f32 files."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--inspection", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    arrays = np.load(args.capture)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embodiment = int(arrays["module.action_head.state_encoder.layer1.input.1"].reshape(-1)[0])
    exported = {}

    def emit(alias: str, value: np.ndarray) -> None:
        value = np.ascontiguousarray(value, dtype=np.float32)
        (args.output_dir / f"{alias}.f32").write_bytes(value.tobytes())
        exported[alias] = list(value.shape)

    conv = "module.backbone.model.model.visual.patch_embed.proj"
    emit("conv_input", arrays[f"{conv}.input.0"].reshape(256, -1))
    emit("conv_weight", arrays[f"{conv}.parameter.weight"].reshape(1024, -1).T)
    emit("conv_bias", arrays[f"{conv}.parameter.bias"])
    emit("conv_output", arrays[f"{conv}.output"].reshape(256, 1024))

    def linear(alias: str, module: str) -> None:
        emit(f"{alias}_input", arrays[f"module.{module}.input.0"].reshape(-1, arrays[f"module.{module}.input.0"].shape[-1]))
        emit(f"{alias}_weight", arrays[f"module.{module}.parameter.W"][embodiment])
        emit(f"{alias}_bias", arrays[f"module.{module}.parameter.b"][embodiment])
        emit(f"{alias}_output", arrays[f"module.{module}.output"].reshape(-1, arrays[f"module.{module}.output"].shape[-1]))

    linear("state1", "action_head.state_encoder.layer1")
    linear("state2", "action_head.state_encoder.layer2")
    linear("action1", "action_head.action_encoder.W1")
    linear("action2", "action_head.action_encoder.W2")
    linear("action3", "action_head.action_encoder.W3")
    linear("decoder1", "action_head.action_decoder.layer1")
    linear("decoder2", "action_head.action_decoder.layer2")
    emit("state_output", arrays["module.action_head.state_encoder.output"].reshape(-1, 1536))
    emit("decoder_output", arrays["module.action_head.action_decoder.output"].reshape(-1, 132))
    if args.inspection is not None:
        inspection = json.loads(args.inspection.read_text())
        case = next(item for item in inspection["profiles"] if item["profile"] == "libero-256" and item["seed"] == 0)
        for name in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "state"):
            emit(f"observation_{name}", np.asarray(case["inputs"][name]["data"]))
        emit("observation_noise", arrays["flow.step.0.action_input"].reshape(40, 132))
        emit("observation_reference_actions", arrays["normalized_actions"].reshape(40, 132))
    manifest = {"schema_version": "1.0", "embodiment": embodiment, "arrays": exported}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
