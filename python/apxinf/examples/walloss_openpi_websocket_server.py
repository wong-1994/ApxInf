#!/usr/bin/env python3
"""Serve WallOSS through ApxInf's OpenPI-compatible websocket transport."""

from __future__ import annotations

import argparse
import logging

from apxinf import WallossPolicy
from apxinf.serving import WebsocketPolicyServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp8"), default="bf16")
    parser.add_argument(
        "--tactics",
        help="optional tactics JSON generated for the current GPU and kernel build",
    )
    parser.add_argument("--norm-key", default="x2_normal")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument(
        "--image-keys", default="observation/image,observation/wrist_image",
        help="two comma-separated RGB uint8 observation keys",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    image_keys = tuple(key.strip() for key in args.image_keys.split(",") if key.strip())
    policy = WallossPolicy.from_pretrained(
        args.model_dir,
        device=args.device,
        precision=args.precision,
        tactics=args.tactics,
        norm_key=args.norm_key,
        action_dim=args.action_dim,
        image_keys=image_keys,
        seed=args.seed,
        metadata={"protocol": "openpi.websocket_policy", "precision": args.precision},
    )
    WebsocketPolicyServer(policy, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
