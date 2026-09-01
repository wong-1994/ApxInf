#!/usr/bin/env python3
"""Load WallOSS and run one raw-observation inference in-process.

The checkpoint model predicts a ``[10, 26]`` normalized action chunk.  Pass
``--action-dim`` only when the checkpoint's leading channels and selected
normalizer key match the robot's deployable action convention.

Requires the ``apxinf_py`` CUDA binding and ``pip install -e
'python/apxinf[walloss]'``.

    python examples/walloss_policy_infer.py \
        --model-dir /path/to/wall-oss-0.5 --action-dim 7
"""

from __future__ import annotations

import argparse
import pathlib

from _common import synthetic_observation  # noqa: E402 (path shim in _common)

from apxinf import WallossPolicy


def _csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(items) != 2:
        raise argparse.ArgumentTypeError("expected exactly two comma-separated values")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp8"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--norm-key", default="x2_normal")
    parser.add_argument(
        "--action-dim",
        type=int,
        default=0,
        help="deployable action width; 0 keeps all 26 model channels",
    )
    parser.add_argument(
        "--image-keys",
        type=_csv,
        default=("observation/image", "observation/wrist_image"),
    )
    parser.add_argument(
        "--camera-names",
        type=_csv,
        default=("face_view", "right_wrist_view"),
        help="checkpoint-trained camera semantics, in image-key order",
    )
    parser.add_argument("--calibration", type=pathlib.Path)
    parser.add_argument("--tactics", type=pathlib.Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="pick up the block")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = WallossPolicy.from_pretrained(
        args.model_dir,
        device=args.device,
        precision=args.precision,
        norm_key=args.norm_key,
        action_dim=(args.action_dim or None),
        image_keys=args.image_keys,
        camera_names=args.camera_names,
        calibration=args.calibration,
        tactics=args.tactics,
        seed=args.seed,
    )
    try:
        print("metadata:", policy.metadata)
        observation = synthetic_observation(
            image_keys=policy.metadata["image_keys"],
            state_dim=policy.action_dim,
            prompt=args.prompt,
            seed=args.seed,
        )
        result = policy.infer(observation)
        actions = result["actions"]
        print(f"actions: shape={actions.shape} dtype={actions.dtype}")
        print(f"timing:  {result['timing']}")
        print(f"first action vector: {actions[0]}")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
