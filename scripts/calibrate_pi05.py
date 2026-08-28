#!/usr/bin/env python3
"""Build a self-describing PI0.5 static-FP8 profile from business Observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping

import numpy as np


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_APXINF_PKG = _REPO_ROOT / "python" / "apxinf"
if _APXINF_PKG.is_dir() and str(_APXINF_PKG) not in sys.path:
    sys.path.insert(0, str(_APXINF_PKG))

SCHEMA = "apxinf.pi05.fp8-calibration.v1"
FP8_FORMAT = "e4m3fn"
STATISTIC = "absmax"
SCALE_RULE = "max(amax*margin/448,1e-8)"
FP8_MAX = 448.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a strict PI0.5 FP8 profile from Observation NPZ files."
    )
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument(
        "--input",
        action="append",
        type=pathlib.Path,
        default=[],
        help="Observation NPZ containing configured image keys, prompt, and optional state",
    )
    parser.add_argument(
        "--zero-fixture",
        action="store_true",
        help="explicitly labeled non-production synthetic Observation",
    )
    parser.add_argument("--data-id", help="stable representative-dataset identifier")
    parser.add_argument(
        "--source-revision",
        help="source commit/version (required when calibration runs outside a Git checkout)",
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-key", action="append", default=[])
    parser.add_argument("--num-views", type=int)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--state-key", default="observation/state")
    parser.add_argument("--discrete-state", action="store_true")
    parser.add_argument("--state-norm-key", default="state")
    parser.add_argument("--tokenizer-path", type=pathlib.Path)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--margin", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def validate_args(args):
    if not args.model_dir.is_dir():
        raise ValueError(f"model directory does not exist: {args.model_dir}")
    if bool(args.input) == bool(args.zero_fixture):
        raise ValueError("pass one or more --input files, or --zero-fixture (but not both)")
    if not np.isfinite(args.margin) or args.margin < 1.0:
        raise ValueError("--margin must be finite and >= 1")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.num_views is not None and args.num_views < 1:
        raise ValueError("--num-views must be positive")
    if args.num_views is not None and not args.image_key:
        raise ValueError("--num-views requires one --image-key per view")
    if args.image_key and args.num_views is not None and len(args.image_key) != args.num_views:
        raise ValueError("--num-views must equal the number of --image-key values")
    if args.action_horizon is not None and args.action_horizon < 1:
        raise ValueError("--action-horizon must be positive")
    if args.data_id is not None and not args.data_id.strip():
        raise ValueError("--data-id must not be empty")
    if args.input and args.data_id is not None and args.data_id.startswith("synthetic:"):
        raise ValueError("representative --input data cannot use a synthetic: --data-id")
    missing = [path for path in args.input if not path.is_file()]
    if missing:
        raise ValueError(f"calibration input does not exist: {missing[0]}")
    checkpoint = args.checkpoint or args.model_dir / "model.safetensors"
    if not checkpoint.exists():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    output = args.output or args.model_dir / "calibration.json"
    if output.exists() and not args.force:
        raise ValueError(f"output already exists (pass --force to replace it): {output}")
    return output, checkpoint


def _decode_npz_value(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return np.ascontiguousarray(array)


def load_observations(args, policy) -> Iterable[Mapping[str, object]]:
    """Load raw Observations; preprocessing remains exclusively in Pi05Policy."""
    if args.zero_fixture:
        observation = {
            key: np.zeros((policy.model.image_size, policy.model.image_size, 3), np.uint8)
            for key in policy.image_keys
        }
        observation[policy.prompt_key] = "synthetic calibration fixture"
        if policy.discrete_state:
            tokenize = policy.input_pipeline["tokenize"]
            state_normalizer = getattr(tokenize, "state_normalizer", None)
            state_width = getattr(state_normalizer, "width", policy.action_dim)
            observation[policy.state_key] = np.zeros(state_width, np.float32)
        yield observation
        return
    for path in args.input:
        with np.load(path, allow_pickle=False) as sample:
            yield {name: _decode_npz_value(sample[name]) for name in sample.files}


def deterministic_noise(policy, seed: int, sample_index: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, sample_index]))
    return np.ascontiguousarray(
        rng.standard_normal((policy.model.action_horizon, policy.model.action_dim)),
        dtype=np.float32,
    )


def merge_records(aggregate, records):
    for name, value in records.items():
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"native calibration returned invalid amax for {name}: {value}")
        aggregate[name] = max(aggregate.get(name, 0.0), value)


def _hash_files(paths: Iterable[pathlib.Path], root: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item.relative_to(root))):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(checkpoint: pathlib.Path) -> str:
    if checkpoint.is_dir():
        files = list(checkpoint.rglob("*.safetensors"))
        root = checkpoint
    elif checkpoint.name.endswith(".index.json"):
        index = json.loads(checkpoint.read_text())
        shards = sorted(set(index.get("weight_map", {}).values()))
        files = [checkpoint, *(checkpoint.parent / name for name in shards)]
        root = checkpoint.parent
    else:
        files = [checkpoint]
        root = checkpoint.parent
    if not files or any(not path.is_file() for path in files):
        raise ValueError(f"cannot resolve checkpoint files from {checkpoint}")
    return "sha256:" + _hash_files(files, root)


def calibration_data_identity(paths: Iterable[pathlib.Path], explicit) -> str:
    if explicit:
        return explicit
    paths = [path.resolve() for path in paths]
    if not paths:
        return "synthetic:zero-observation-v1"
    common = pathlib.Path(os.path.commonpath(paths))
    if common.is_file():
        common = common.parent
    return "sha256:" + _hash_files(paths, common)


def source_revision(explicit=None) -> str:
    if explicit is not None:
        if not explicit.strip() or explicit == "unknown":
            raise ValueError("--source-revision must identify a real commit or release")
        return explicit
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"], cwd=_REPO_ROOT, check=False
        ).returncode != 0
        return revision + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(
            "cannot determine source revision; pass --source-revision explicitly"
        ) from error


def calibration_document(
    records,
    *,
    margin,
    sample_count,
    bootstrap,
    required_sites,
    checkpoint,
    data_identity,
    seed,
    device,
    revision=None,
):
    required = tuple(required_sites)
    observed = set(records)
    missing = sorted(set(required) - observed)
    unknown = sorted(observed - set(required))
    if missing or unknown:
        raise ValueError(f"calibration site coverage mismatch: missing={missing}, unknown={unknown}")
    scales = {}
    for name in sorted(records):
        amax = float(records[name])
        if bootstrap and name == "vision.patch_input":
            amax = max(amax, 1.0 / margin)
        if bootstrap and name == "action.input":
            amax = max(amax, 5.0 / margin)
        scales[name] = {
            "amax": amax,
            "scale": max(amax * margin / FP8_MAX, 1.0e-8),
        }
    return {
        "schema": SCHEMA,
        "model": {"family": "pi05", "checkpoint": checkpoint},
        "quantization": {
            "format": FP8_FORMAT,
            "statistic": STATISTIC,
            "scale_rule": SCALE_RULE,
            "margin": margin,
        },
        "calibration_data": {
            "identity": data_identity,
            "kind": "synthetic-zero-fixture" if bootstrap else "representative",
            "production": not bootstrap,
            "sample_count": sample_count,
        },
        "seed_policy": {
            "algorithm": "numpy-pcg64-seed-sequence-v1",
            "base_seed": seed,
            "sample_sequence": "[base_seed,sample_index]",
        },
        "source_revision": source_revision(revision),
        "device": {"requested": device, "host": platform.platform()},
        "plan": {"sites": list(required)},
        "observed_sites": sorted(observed),
        "scales": scales,
    }


def write_profile(output: pathlib.Path, document, *, force: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if force else "x"
    try:
        with output.open(mode) as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise ValueError(
            f"output already exists (pass --force to replace it): {output}"
        ) from error


def main(argv=None):
    args = parse_args(argv)
    output, checkpoint = validate_args(args)
    from apxinf import Pi05Policy

    policy_options = {
        "checkpoint": checkpoint,
        "device": args.device,
        "precision": "bf16",
        "seed": args.seed,
        "prompt_key": args.prompt_key,
        "state_key": args.state_key,
        "discrete_state": args.discrete_state,
        "state_norm_key": args.state_norm_key,
    }
    if args.image_key:
        policy_options["image_keys"] = tuple(args.image_key)
        policy_options["num_views"] = args.num_views or len(args.image_key)
    if args.tokenizer_path is not None:
        policy_options["tokenizer_path"] = args.tokenizer_path
    if args.action_horizon is not None:
        policy_options["action_horizon"] = args.action_horizon
    policy = Pi05Policy.from_pretrained(args.model_dir, **policy_options)
    aggregate = {}
    sample_count = 0
    try:
        required_sites = policy.model._calibration_plan()
        for sample_count, observation in enumerate(load_observations(args, policy), start=1):
            noise = deterministic_noise(policy, args.seed, sample_count - 1)
            merge_records(
                aggregate, policy.calibrate_observation(observation, noise=noise)
            )
    finally:
        policy.close()
    document = calibration_document(
        aggregate,
        margin=args.margin,
        sample_count=sample_count,
        bootstrap=args.zero_fixture,
        required_sites=required_sites,
        checkpoint=checkpoint_identity(checkpoint),
        data_identity=calibration_data_identity(args.input, args.data_id),
        seed=args.seed,
        device=args.device,
        revision=args.source_revision,
    )
    write_profile(output, document, force=args.force)
    print(f"wrote {len(aggregate)} activation scales from {sample_count} sample(s): {output}")
    if args.zero_fixture:
        print("warning: synthetic profile is non-production; calibrate representative data")


if __name__ == "__main__":
    main()
