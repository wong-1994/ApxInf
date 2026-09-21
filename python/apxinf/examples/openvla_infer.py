#!/usr/bin/env python3
"""Run the original OpenVLA checkpoint through the native CUDA policy."""
import argparse
from PIL import Image
from apxinf import AutoPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--unnorm-key", default="bridge_orig")
    args = parser.parse_args()
    policy = AutoPolicy.from_pretrained(
        args.model_dir, precision="bf16", unnorm_key=args.unnorm_key
    )
    result = policy.infer(
        {"image": Image.open(args.image).convert("RGB"), "prompt": args.prompt}
    )
    print("tokens:", result["token_ids"])
    print("actions:", result["actions"])
    print("timing:", result["timing"])


if __name__ == "__main__":
    main()
