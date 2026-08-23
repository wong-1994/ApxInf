#!/usr/bin/env python3
"""Verify a trusted source's private Canonical VLA implementation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path
from typing import Any


TRANSFORMATION_CATEGORIES = {
    "transpose",
    "split",
    "concatenation",
    "packing",
    "mask",
    "conditioning",
    "cache",
    "schedule",
}
STATE_CATEGORIES = {"mask", "conditioning", "cache", "schedule"}


def load_reference_support() -> Any:
    path = Path(__file__).with_name("reference_adapter.py")
    spec = importlib.util.spec_from_file_location("apxinf_private_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the private Reference Adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def value_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_string_list(value: Any, path: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value) or any(
        not isinstance(item, str) or not item for item in value
    ):
        requirement = "a non-empty string array" if nonempty else "a string array"
        raise ValueError(f"{path} must be {requirement}")
    return value


def validate_manifest(
    manifest: Any,
    inventory: dict[str, Any],
    classification: dict[str, Any],
) -> None:
    required = {
        "parameter_mapping",
        "transformations",
        "semantic_rewrites",
        "state_semantics",
        "branches",
        "intermediate_mapping",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("canonicalization manifest has missing or unknown fields")

    transformations = manifest["transformations"]
    if not isinstance(transformations, list) or not transformations:
        raise ValueError("transformations must be a non-empty array")
    transformation_ids: set[str] = set()
    for index, transformation in enumerate(transformations):
        path = f"transformations[{index}]"
        if not isinstance(transformation, dict):
            raise ValueError(f"{path} must be an object")
        required_fields = {
            "id",
            "category",
            "rewrite",
            "assumptions",
            "source_paths",
            "target_paths",
            "description",
        }
        if set(transformation) != required_fields:
            raise ValueError(f"{path} has missing or unknown fields")
        identifier = transformation["id"]
        if not isinstance(identifier, str) or not identifier or identifier in transformation_ids:
            raise ValueError(f"{path}.id must be a unique non-empty string")
        transformation_ids.add(identifier)
        if transformation["category"] not in TRANSFORMATION_CATEGORIES:
            raise ValueError(f"{path}.category is not a declared transformation kind")
        if transformation["rewrite"] not in {"algebraic", "numerical_equivalence"}:
            raise ValueError(f"{path}.rewrite must declare its proof mode")
        assumptions = require_string_list(
            transformation["assumptions"], f"{path}.assumptions"
        )
        if transformation["rewrite"] == "algebraic" and not assumptions:
            raise ValueError(f"{path}.assumptions must record algebraic assumptions")
        require_string_list(
            transformation["source_paths"], f"{path}.source_paths", nonempty=True
        )
        require_string_list(
            transformation["target_paths"], f"{path}.target_paths", nonempty=True
        )
        if not isinstance(transformation["description"], str) or not transformation[
            "description"
        ]:
            raise ValueError(f"{path}.description must be a non-empty string")

    mappings = manifest["parameter_mapping"]
    if not isinstance(mappings, list):
        raise ValueError("parameter_mapping must be an array")
    mapped_sources = []
    for index, mapping in enumerate(mappings):
        path = f"parameter_mapping[{index}]"
        if not isinstance(mapping, dict) or set(mapping) != {
            "source",
            "targets",
            "transformation_ids",
        }:
            raise ValueError(f"{path} has missing or unknown fields")
        if not isinstance(mapping["source"], str) or not mapping["source"]:
            raise ValueError(f"{path}.source must be a non-empty string")
        mapped_sources.append(mapping["source"])
        require_string_list(mapping["targets"], f"{path}.targets", nonempty=True)
        references = require_string_list(
            mapping["transformation_ids"], f"{path}.transformation_ids"
        )
        if not set(references) <= transformation_ids:
            raise ValueError(f"{path} references an unknown transformation")
    expected_sources = [parameter["name"] for parameter in inventory["parameters"]]
    if len(mapped_sources) != len(set(mapped_sources)) or set(mapped_sources) != set(
        expected_sources
    ):
        missing = sorted(set(expected_sources) - set(mapped_sources))
        unknown = sorted(set(mapped_sources) - set(expected_sources))
        raise ValueError(
            "parameter mapping must consume every source parameter exactly once; "
            f"unmapped={missing}, unknown={unknown}"
        )

    expected_rewrites = {
        (item["capability"], item["observed"], item["canonical"])
        for item in classification["classifications"]
        if item["classification"] == "canonicalizable"
    }
    declared_rewrites = set()
    rewrites = manifest["semantic_rewrites"]
    if not isinstance(rewrites, list):
        raise ValueError("semantic_rewrites must be an array")
    for index, rewrite in enumerate(rewrites):
        path = f"semantic_rewrites[{index}]"
        if not isinstance(rewrite, dict) or set(rewrite) != {
            "capability",
            "source",
            "canonical",
            "transformation_ids",
        }:
            raise ValueError(f"{path} has missing or unknown fields")
        declared_rewrites.add(
            (rewrite["capability"], rewrite["source"], rewrite["canonical"])
        )
        references = require_string_list(
            rewrite["transformation_ids"], f"{path}.transformation_ids", nonempty=True
        )
        if not set(references) <= transformation_ids:
            raise ValueError(f"{path} references an unknown transformation")
    if declared_rewrites != expected_rewrites:
        raise ValueError("semantic rewrites do not cover every canonicalizable capability")

    state_semantics = manifest["state_semantics"]
    if not isinstance(state_semantics, list):
        raise ValueError("state_semantics must be an array")
    categories = set()
    for index, state in enumerate(state_semantics):
        path = f"state_semantics[{index}]"
        if not isinstance(state, dict) or set(state) != {
            "category",
            "source",
            "canonical",
            "transformation_ids",
        }:
            raise ValueError(f"{path} has missing or unknown fields")
        categories.add(state["category"])
        references = require_string_list(
            state["transformation_ids"], f"{path}.transformation_ids", nonempty=True
        )
        if not set(references) <= transformation_ids:
            raise ValueError(f"{path} references an unknown transformation")
    if categories != STATE_CATEGORIES or len(state_semantics) != len(STATE_CATEGORIES):
        raise ValueError("state semantics must cover mask, conditioning, cache, and schedule")

    branches = manifest["branches"]
    if not isinstance(branches, list):
        raise ValueError("branches must be an array")
    branch_sources = []
    for index, branch in enumerate(branches):
        path = f"branches[{index}]"
        if not isinstance(branch, dict) or set(branch) != {
            "source",
            "disposition",
            "transformation_ids",
        }:
            raise ValueError(f"{path} has missing or unknown fields")
        branch_sources.append(branch["source"])
        if branch["disposition"] not in {"preserved", "transformed"}:
            raise ValueError(f"{path}.disposition must explain the source branch")
        references = require_string_list(
            branch["transformation_ids"], f"{path}.transformation_ids"
        )
        if not set(references) <= transformation_ids:
            raise ValueError(f"{path} references an unknown transformation")
    expected_branches = [
        branch.get("name", f"dynamic_branches[{index}]")
        for index, branch in enumerate(inventory["dynamic_branches"])
    ]
    if len(branch_sources) != len(set(branch_sources)) or set(branch_sources) != set(
        expected_branches
    ):
        raise ValueError("every source branch must have an explained disposition")

    intermediates = manifest["intermediate_mapping"]
    if not isinstance(intermediates, list) or not intermediates:
        raise ValueError("intermediate_mapping must select at least one checkpoint")
    for index, mapping in enumerate(intermediates):
        if not isinstance(mapping, dict) or set(mapping) != {"source", "canonical"}:
            raise ValueError(f"intermediate_mapping[{index}] is invalid")
        if any(not isinstance(mapping[key], str) or not mapping[key] for key in mapping):
            raise ValueError(f"intermediate_mapping[{index}] paths must be strings")


def compare_values(
    source: Any, canonical: Any, absolute: float, relative: float
) -> tuple[bool, float, float]:
    if isinstance(source, dict) and isinstance(canonical, dict):
        if set(source) != set(canonical):
            return False, 0.0, 0.0
        results = [
            compare_values(source[key], canonical[key], absolute, relative)
            for key in source
        ]
    elif isinstance(source, list) and isinstance(canonical, list):
        if len(source) != len(canonical):
            return False, 0.0, 0.0
        results = [
            compare_values(left, right, absolute, relative)
            for left, right in zip(source, canonical)
        ]
    elif (
        isinstance(source, (int, float))
        and not isinstance(source, bool)
        and isinstance(canonical, (int, float))
        and not isinstance(canonical, bool)
    ):
        absolute_error = abs(float(source) - float(canonical))
        relative_error = absolute_error / abs(float(source)) if source else absolute_error
        return (
            absolute_error <= absolute + relative * abs(float(source)),
            absolute_error,
            relative_error,
        )
    else:
        return source == canonical, 0.0, 0.0
    if not results:
        return True, 0.0, 0.0
    return (
        all(result[0] for result in results),
        max(result[1] for result in results),
        max(result[2] for result in results),
    )


def comparison(
    scope: str,
    source_path: str,
    canonical_path: str,
    source: Any,
    canonical: Any,
    absolute: float,
    relative: float,
) -> dict[str, Any]:
    passed, max_absolute_error, max_relative_error = compare_values(
        source, canonical, absolute, relative
    )
    return {
        "scope": scope,
        "source_path": source_path,
        "canonical_path": canonical_path,
        "passed": passed,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
    }


def verify(args: argparse.Namespace) -> None:
    support = load_reference_support()
    support.disable_runtime_network()
    try:
        adapter = support.ReferenceAdapter(args.source_root, args.entrypoint)
        source_model = adapter.load(str(args.checkpoint))
        canonical_source_model = adapter.load(str(args.checkpoint))
        canonical_model = adapter._call("canonicalize", canonical_source_model)
        manifest = adapter._call("canonicalization_manifest")
        inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        classification = json.loads(args.classification.read_text(encoding="utf-8"))
        profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
        thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
        validate_manifest(manifest, inventory, classification)
        absolute = float(thresholds["absolute"])
        relative = float(thresholds["relative"])

        trace_cases = []
        evidence_cases = []
        for profile in profiles:
            for seed in (0, 1):
                support.set_deterministic_seed(adapter._module, seed)
                source_inputs = adapter.preprocess(profile)
                source_output = adapter.infer(source_model, source_inputs)
                source_intermediates = adapter.capture_intermediates(
                    source_model, source_inputs
                )
                source_postprocessed = adapter.postprocess(source_output)

                support.set_deterministic_seed(adapter._module, seed)
                canonical_inputs = adapter.preprocess(profile)
                canonical_output = adapter._call(
                    "canonical_infer", canonical_model, canonical_inputs
                )
                canonical_intermediates = adapter._call(
                    "canonical_capture_intermediates",
                    canonical_model,
                    canonical_inputs,
                )
                canonical_postprocessed = adapter._call(
                    "canonical_postprocess", canonical_output
                )

                source_intermediates_json = support.json_capture(source_intermediates)
                canonical_intermediates_json = support.json_capture(
                    canonical_intermediates
                )
                source_output_json = support.json_capture(source_output)
                canonical_output_json = support.json_capture(canonical_output)
                source_postprocessed_json = support.json_capture(source_postprocessed)
                canonical_postprocessed_json = support.json_capture(
                    canonical_postprocessed
                )
                comparisons = []
                for mapping in manifest["intermediate_mapping"]:
                    comparisons.append(
                        comparison(
                            "intermediates",
                            f"intermediates.{mapping['source']}",
                            f"intermediates.{mapping['canonical']}",
                            source_intermediates_json[mapping["source"]],
                            canonical_intermediates_json[mapping["canonical"]],
                            absolute,
                            relative,
                        )
                    )
                comparisons.extend(
                    [
                        comparison(
                            "normalized_actions",
                            "output.actions",
                            "output.actions",
                            source_output_json["actions"],
                            canonical_output_json["actions"],
                            absolute,
                            relative,
                        ),
                        comparison(
                            "postprocessed_actions",
                            "postprocessed.actions",
                            "postprocessed.actions",
                            source_postprocessed_json["actions"],
                            canonical_postprocessed_json["actions"],
                            absolute,
                            relative,
                        ),
                    ]
                )
                trace_cases.append(
                    {
                        "profile": profile.get("name"),
                        "seed": seed,
                        "inputs": support.json_capture(canonical_inputs),
                        "output": canonical_output_json,
                        "intermediates": canonical_intermediates_json,
                        "postprocessed": canonical_postprocessed_json,
                    }
                )
                evidence_cases.append(
                    {
                        "profile": profile.get("name"),
                        "seed": seed,
                        "comparisons": comparisons,
                        "passed": all(item["passed"] for item in comparisons),
                    }
                )

        canonical_semantics = {
            item["capability"]: item["canonical"]
            for item in classification["classifications"]
            if item["path"].startswith("capability_facts.")
        }
        trace = {
            "schema_version": "1.0",
            "port_id": args.port_id,
            "mode": "canonicalized",
            "contract": classification["contract"],
            "canonical_semantics": canonical_semantics,
            "cases": trace_cases,
        }
        failures = sum(
            not item["passed"]
            for case in evidence_cases
            for item in case["comparisons"]
        )
        evidence = {
            "schema_version": "1.0",
            "port_id": args.port_id,
            "mode": "canonicalized",
            "contract": classification["contract"],
            "source_inventory_sha256": value_sha256(inventory),
            "canonical_trace_sha256": value_sha256(trace),
            "thresholds": thresholds,
            "parameter_mapping": manifest["parameter_mapping"],
            "transformations": manifest["transformations"],
            "semantic_rewrites": manifest["semantic_rewrites"],
            "state_semantics": manifest["state_semantics"],
            "branches": manifest["branches"],
            "intermediate_mapping": manifest["intermediate_mapping"],
            "cases": evidence_cases,
            "summary": {
                "cases": len(evidence_cases),
                "comparisons": sum(
                    len(case["comparisons"]) for case in evidence_cases
                ),
                "failures": failures,
            },
        }
        write_json(args.trace, trace)
        write_json(args.evidence, evidence)
        if failures:
            write_json(
                args.result,
                {
                    "status": "gap",
                    "gaps": [
                        {
                            "kind": "numerical_mismatch",
                            "path": comparison["canonical_path"],
                            "message": (
                                f"{comparison['scope']} exceeds correctness thresholds"
                            ),
                        }
                        for case in evidence_cases
                        for comparison in case["comparisons"]
                        if not comparison["passed"]
                    ],
                    "summary": evidence["summary"],
                },
            )
        else:
            write_json(
                args.result,
                {"status": "success", "summary": evidence["summary"]},
            )
    except Exception as error:
        write_json(
            args.result,
            {
                "status": "gap",
                "gaps": [
                    {
                        "kind": "incomplete_canonicalization",
                        "path": "canonicalization_manifest",
                        "message": str(error),
                    }
                ],
                "summary": {"cases": 0, "comparisons": 0, "failures": 1},
                "traceback": traceback.format_exc(),
            },
        )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("--source-root", type=Path, required=True)
    command.add_argument("--entrypoint", required=True)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--profiles", type=Path, required=True)
    command.add_argument("--inventory", type=Path, required=True)
    command.add_argument("--classification", type=Path, required=True)
    command.add_argument("--thresholds", type=Path, required=True)
    command.add_argument("--trace", type=Path, required=True)
    command.add_argument("--evidence", type=Path, required=True)
    command.add_argument("--result", type=Path, required=True)
    command.add_argument("--port-id", required=True)
    return command


if __name__ == "__main__":
    verify(parser().parse_args())
