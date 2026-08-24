#!/usr/bin/env python3
"""Verify a trusted source's private Canonical VLA implementation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
    canonical_parameters: list[dict[str, Any]],
    expected_canonical_parameters: list[dict[str, Any]],
    canonical_aliases: list[list[str]],
    canonical_tied_weights: list[list[str]],
) -> None:
    required = {
        "parameter_mapping",
        "transformations",
        "semantic_rewrites",
        "state_semantics",
        "branches",
        "intermediate_mapping",
        "preprocessing_mapping",
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
    mapped_targets = []
    source_targets: dict[str, list[str]] = {}
    source_parameters = {
        parameter["name"]: parameter for parameter in inventory["parameters"]
    }
    target_parameters = {
        parameter["name"]: parameter for parameter in canonical_parameters
    }
    expected_target_parameters = {
        parameter["name"]: parameter for parameter in expected_canonical_parameters
    }
    for index, mapping in enumerate(mappings):
        path = f"parameter_mapping[{index}]"
        if not isinstance(mapping, dict) or set(mapping) != {
            "sources",
            "targets",
            "transformation_ids",
        }:
            raise ValueError(f"{path} has missing or unknown fields")
        sources = require_string_list(
            mapping["sources"], f"{path}.sources", nonempty=True
        )
        targets = require_string_list(
            mapping["targets"], f"{path}.targets", nonempty=True
        )
        mapped_sources.extend(sources)
        mapped_targets.extend(targets)
        for source in sources:
            source_targets[source] = targets
        references = require_string_list(
            mapping["transformation_ids"], f"{path}.transformation_ids"
        )
        if not set(references) <= transformation_ids:
            raise ValueError(f"{path} references an unknown transformation")
        if not set(sources) <= set(source_parameters):
            raise ValueError(f"{path} references an unknown source parameter")
        if not set(targets) <= set(target_parameters):
            raise ValueError(f"{path} references an unknown canonical parameter")
        source_elements = sum(
            math.prod(source_parameters[name]["shape"]) for name in sources
        )
        target_elements = sum(
            math.prod(target_parameters[name]["shape"]) for name in targets
        )
        source_dtypes = {source_parameters[name]["dtype"] for name in sources}
        target_dtypes = {target_parameters[name]["dtype"] for name in targets}
        if source_dtypes != target_dtypes:
            raise ValueError(f"{path} does not preserve the mapped parameter dtype")
        source_shapes = [source_parameters[name]["shape"] for name in sources]
        target_shapes = [target_parameters[name]["shape"] for name in targets]
        source_hashes = [source_parameters[name]["data_sha256"] for name in sources]
        expected_hashes = [
            expected_target_parameters[name]["data_sha256"]
            for name in targets
            if name in expected_target_parameters
        ]
        structure_changed = (
            len(sources) != len(targets)
            or source_shapes != target_shapes
            or source_hashes != expected_hashes
        )
        structural_ids = {
            transformation["id"]
            for transformation in transformations
            if transformation["category"]
            in {"transpose", "split", "concatenation", "packing"}
        }
        if structure_changed and not set(references) & structural_ids:
            raise ValueError(
                f"{path} must cite a structural parameter transformation"
            )
        if source_elements != target_elements:
            raise ValueError(f"{path} does not preserve the mapped parameter size")
        for target in targets:
            expected = expected_target_parameters.get(target)
            actual = target_parameters[target]
            if expected is None:
                raise ValueError(f"{path} does not produce canonical parameter {target}")
            if (
                expected["shape"] != actual["shape"]
                or expected["dtype"] != actual["dtype"]
                or expected["data_sha256"] != actual["data_sha256"]
            ):
                raise ValueError(
                    f"{path} canonical parameter value does not match {target}"
                )
    expected_sources = list(source_parameters)
    if len(mapped_sources) != len(set(mapped_sources)) or set(mapped_sources) != set(
        expected_sources
    ):
        missing = sorted(set(expected_sources) - set(mapped_sources))
        unknown = sorted(set(mapped_sources) - set(expected_sources))
        raise ValueError(
            "parameter mapping must consume every source parameter exactly once; "
            f"unmapped={missing}, unknown={unknown}"
        )
    expected_targets = list(target_parameters)
    if len(mapped_targets) != len(set(mapped_targets)) or set(mapped_targets) != set(
        expected_targets
    ):
        missing = sorted(set(expected_targets) - set(mapped_targets))
        unknown = sorted(set(mapped_targets) - set(expected_targets))
        raise ValueError(
            "parameter mapping must consume every canonical parameter exactly once; "
            f"unmapped={missing}, unknown={unknown}"
        )

    def require_preserved_groups(
        kind: str, source_groups: list[list[str]], canonical_groups: list[list[str]]
    ) -> None:
        canonical_sets = [set(group) for group in canonical_groups]
        for source_group in source_groups:
            projected = {
                target
                for source in source_group
                if source in source_targets
                for target in source_targets[source]
            }
            if len(projected) > 1 and not any(
                projected <= canonical_group for canonical_group in canonical_sets
            ):
                raise ValueError(f"canonical parameters do not preserve source {kind}")

    source_parameter_names = set(source_parameters)
    source_aliases = [
        [name for name in group if name in source_parameter_names]
        for group in inventory["aliases"]
    ]
    require_preserved_groups(
        "aliases",
        [group for group in source_aliases if len(group) > 1],
        canonical_aliases,
    )
    require_preserved_groups(
        "tied weights", inventory["tied_weights"], canonical_tied_weights
    )

    def projected_groups(source_groups: list[list[str]]) -> set[frozenset[str]]:
        return {
            frozenset(
                target
                for source in group
                if source in source_targets
                for target in source_targets[source]
            )
            for group in source_groups
            if len(group) > 1
        }

    structural_ids = {
        transformation["id"]
        for transformation in transformations
        if transformation["category"] in {"split", "concatenation", "packing"}
    }
    target_mapping = {
        target: mapping
        for mapping in mappings
        for target in mapping["targets"]
    }

    def require_explained_relation_change(
        kind: str,
        expected_groups: set[frozenset[str]],
        actual_groups: set[frozenset[str]],
    ) -> None:
        changed_targets = set().union(*(expected_groups ^ actual_groups))
        unexplained = sorted(
            target
            for target in changed_targets
            if target not in target_mapping
            or not (
                set(target_mapping[target]["transformation_ids"])
                & structural_ids
            )
        )
        if unexplained:
            raise ValueError(
                f"canonical {kind} changes require split, concatenation, or "
                f"packing evidence for targets: {unexplained}"
            )

    require_explained_relation_change(
        "aliases",
        projected_groups([group for group in source_aliases if len(group) > 1]),
        {frozenset(group) for group in canonical_aliases},
    )
    require_explained_relation_change(
        "tied weights",
        projected_groups(inventory["tied_weights"]),
        {frozenset(group) for group in canonical_tied_weights},
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
    transformed_state_categories = {
        item["category"]
        for item in transformations
        if item["category"] in STATE_CATEGORIES
    }
    if categories != transformed_state_categories or len(state_semantics) != len(
        transformed_state_categories
    ):
        raise ValueError("state semantics must cover every declared state transformation")

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
            branch["transformation_ids"],
            f"{path}.transformation_ids",
            nonempty=branch["disposition"] == "transformed",
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

    preprocessing = manifest["preprocessing_mapping"]
    if not isinstance(preprocessing, list) or not preprocessing:
        raise ValueError("preprocessing_mapping must select at least one input boundary")
    preprocessing_sources = []
    preprocessing_targets = []
    for index, mapping in enumerate(preprocessing):
        path = f"preprocessing_mapping[{index}]"
        if not isinstance(mapping, dict) or set(mapping) != {
            "source",
            "canonical",
            "transformation_ids",
        }:
            raise ValueError(f"{path} is invalid")
        if any(
            not isinstance(mapping[key], str) or not mapping[key]
            for key in ("source", "canonical")
        ):
            raise ValueError(f"{path} paths must be strings")
        for key in ("source", "canonical"):
            mapping_path = mapping[key]
            if mapping_path != "inputs" and not mapping_path.startswith("inputs."):
                raise ValueError(f"{path}.{key} must start with inputs")
        preprocessing_sources.append(mapping["source"])
        preprocessing_targets.append(mapping["canonical"])
        references = require_string_list(
            mapping["transformation_ids"], f"{path}.transformation_ids"
        )
        if not set(references) <= transformation_ids:
            raise ValueError(f"{path} references an unknown transformation")
    if len(preprocessing_sources) != len(set(preprocessing_sources)):
        raise ValueError("preprocessing mappings cannot duplicate source paths")
    if len(preprocessing_targets) != len(set(preprocessing_targets)):
        raise ValueError("preprocessing mappings cannot duplicate canonical paths")
    if "inputs" not in preprocessing_sources or "inputs" not in preprocessing_targets:
        raise ValueError("preprocessing mappings must cover the complete inputs boundary")


def compare_values(
    source: Any, canonical: Any, absolute: float, relative: float
) -> tuple[bool, float | None, float | None, str | None]:
    if isinstance(source, dict) and isinstance(canonical, dict):
        if set(source) != set(canonical):
            return False, None, None, "object_keys"
        results = [
            compare_values(source[key], canonical[key], absolute, relative)
            for key in source
        ]
    elif isinstance(source, list) and isinstance(canonical, list):
        if len(source) != len(canonical):
            return False, None, None, "array_length"
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
            None,
        )
    else:
        if source == canonical:
            return True, 0.0, 0.0, None
        return False, None, None, "value_or_type"
    if not results:
        return True, 0.0, 0.0, None
    mismatch_reason = next(
        (result[3] for result in results if result[3] is not None), None
    )
    if mismatch_reason is not None:
        return False, None, None, mismatch_reason
    absolute_errors = [result[1] for result in results if result[1] is not None]
    relative_errors = [result[2] for result in results if result[2] is not None]
    return (
        all(result[0] for result in results),
        max(absolute_errors) if absolute_errors else None,
        max(relative_errors) if relative_errors else None,
        mismatch_reason,
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
    passed, max_absolute_error, max_relative_error, mismatch_reason = compare_values(
        source, canonical, absolute, relative
    )
    return {
        "scope": scope,
        "source_path": source_path,
        "canonical_path": canonical_path,
        "passed": passed,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
        "mismatch_reason": mismatch_reason,
    }


def mapped_value(root: Any, path: str) -> Any:
    if path == "inputs":
        return root
    if not path.startswith("inputs."):
        raise ValueError(f"preprocessing mapping path must start with inputs: {path}")
    value = root
    for part in path.split(".")[1:]:
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"preprocessing mapping path does not exist: {path}")
        value = value[part]
    return value


def verify(args: argparse.Namespace) -> None:
    support = load_reference_support()
    support.disable_runtime_network()
    try:
        adapter = support.ReferenceAdapter(args.source_root, args.entrypoint)
        adapter.set_seed(0)
        source_model = adapter.load(str(args.checkpoint))
        adapter.set_seed(0)
        canonical_source_model = adapter.load(str(args.checkpoint))
        adapter.set_seed(0)
        canonical_model = adapter.canonicalize(canonical_source_model)
        manifest = adapter.canonicalization_manifest()
        inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        classification = json.loads(args.classification.read_text(encoding="utf-8"))
        profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
        thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
        canonical_parameter_values = support.named_values(
            canonical_model, "named_parameters"
        )
        canonical_parameters = [
            support.tensor_record(name, value)
            for name, value in canonical_parameter_values
        ]
        expected_canonical_parameters = []
        for mapping in manifest["parameter_mapping"]:
            try:
                mapped_values = adapter.canonical_parameter_values(
                    canonical_source_model, mapping
                )
            except KeyError as error:
                raise ValueError(
                    f"parameter mapping references an unknown canonical parameter: {error}"
                ) from error
            for name, value in mapped_values.items():
                expected_canonical_parameters.append(
                    support.tensor_record(name, value)
                )
        canonical_aliases = support.alias_groups(canonical_parameter_values)
        canonical_tied_weights = support.tied_weight_groups(
            canonical_parameter_values
        )
        validate_manifest(
            manifest,
            inventory,
            classification,
            canonical_parameters,
            expected_canonical_parameters,
            canonical_aliases,
            canonical_tied_weights,
        )
        absolute = float(thresholds["absolute"])
        relative = float(thresholds["relative"])

        trace_cases = []
        evidence_cases = []
        for profile in profiles:
            for seed in (0, 1):
                adapter.set_seed(seed)
                source_inputs = adapter.preprocess(profile)
                source_output = adapter.infer(source_model, source_inputs)
                source_output_json = support.json_capture(source_output)
                source_intermediates = adapter.capture_intermediates(
                    source_model, source_inputs
                )
                source_intermediates_json = support.json_capture(
                    source_intermediates
                )
                source_postprocessed = adapter.postprocess(source_output)
                source_postprocessed_json = support.json_capture(source_postprocessed)

                adapter.set_seed(seed)
                transform_inputs = adapter.preprocess(profile)
                expected_canonical_inputs = adapter.canonicalize_preprocessed_inputs(
                    transform_inputs
                )

                adapter.set_seed(seed)
                canonical_inputs = adapter.canonical_preprocess(profile)
                canonical_inputs_json = support.json_capture(canonical_inputs)
                canonical_output = adapter.canonical_infer(
                    canonical_model, canonical_inputs
                )
                canonical_output_json = support.json_capture(canonical_output)
                canonical_intermediates = adapter.canonical_capture_intermediates(
                    canonical_model, canonical_inputs
                )
                canonical_intermediates_json = support.json_capture(
                    canonical_intermediates
                )
                canonical_postprocessed = adapter.canonical_postprocess(
                    canonical_output
                )
                canonical_postprocessed_json = support.json_capture(
                    canonical_postprocessed
                )

                expected_canonical_inputs_json = support.json_capture(
                    expected_canonical_inputs
                )
                comparisons = [
                    comparison(
                        "preprocessed_inputs",
                        mapping["source"],
                        mapping["canonical"],
                        mapped_value(
                            expected_canonical_inputs_json, mapping["source"]
                        ),
                        mapped_value(canonical_inputs_json, mapping["canonical"]),
                        absolute,
                        relative,
                    )
                    for mapping in manifest["preprocessing_mapping"]
                ]
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
                        "inputs": canonical_inputs_json,
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

        trace = support.canonical_trace_document(
            args.port_id, "canonicalized", classification, trace_cases
        )
        evidence = support.canonical_evidence_document(
            port_id=args.port_id,
            mode="canonicalized",
            classification=classification,
            inventory=inventory,
            trace=trace,
            thresholds=thresholds,
            parameter_mapping=manifest["parameter_mapping"],
            canonical_parameters=canonical_parameters,
            canonical_aliases=canonical_aliases,
            canonical_tied_weights=canonical_tied_weights,
            transformations=manifest["transformations"],
            cases=evidence_cases,
            manifest_evidence={
                "semantic_rewrites": manifest["semantic_rewrites"],
                "state_semantics": manifest["state_semantics"],
                "branches": manifest["branches"],
                "intermediate_mapping": manifest["intermediate_mapping"],
                "preprocessing_mapping": manifest["preprocessing_mapping"],
            },
        )
        failures = evidence["summary"]["failures"]
        support.write_json(args.trace, trace)
        support.write_json(args.evidence, evidence)
        if failures:
            support.write_json(
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
            support.write_json(
                args.result,
                {"status": "success", "summary": evidence["summary"]},
            )
    except Exception as error:
        support.write_json(
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
