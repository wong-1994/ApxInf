#!/usr/bin/env python3
"""Run Intake and canonical-equivalence Preflight for an ApxInf VLA Port."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import secrets
import subprocess
import sys
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reference_adapter_template import (
    canonical_evidence_document,
    canonical_trace_document,
)
from porting_core import (
    CORRECTNESS_FAILURE,
    ENVIRONMENT_FAILURE,
    FamilyPack,
    INVALID_INPUT,
    KERNEL_GAP,
    MISSING_INPUT,
    PortOutcome,
    PortingCore,
    REFERENCE_LOAD_FAILURE,
    REFERENCE_TRACE_FAILURE,
    SUCCESS,
    UNSUPPORTED_SEMANTICS,
    UNSUPPORTED_TARGET,
    ArtifactStore,
    VLA_FAMILY_PACK,
    select_family_pack,
    resume_report,
    validate_requested_tuple,
)
from kernel_coverage import KernelCoverageError, analyze_kernel_coverage
from portable_bundle import BundleError, create_bundle, merge_bundle
from publication import PublicationError, prepare_publication


REQUEST_SCHEMA_VERSION = "1.0"
REFERENCE_ADAPTER_CONTRACT_VERSION = "1.0"
TARGETS = {"thor", "orin"}
PRECISIONS = {"bf16", "fp8", "int8_w8a8"}


IntakeOutcome = PortOutcome


@dataclass(frozen=True, order=True)
class ContractVersion:
    major: int
    minor: int

    @classmethod
    def parse(cls, value: Any) -> ContractVersion | None:
        if not isinstance(value, str):
            return None
        parts = value.split(".")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            return None
        return cls(*(int(part) for part in parts))


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def reference_adapter_template() -> Path:
    return repository_root() / "scripts" / "reference_adapter_template.py"


def canonical_adapter_template() -> Path:
    return repository_root() / "scripts" / "canonical_adapter_template.py"


def default_capability_contract(version: str) -> Path:
    return repository_root() / "contracts" / f"vla-capability-contract-{version}.json"


def default_kernel_capabilities() -> Path:
    return repository_root() / "contracts" / "kernel-capabilities-1.0.json"


def port_dir_is_unsafe(port_dir: Path, source: Path | None = None) -> bool:
    protected_roots = [repository_root()]
    if source is not None:
        protected_roots.append(source.resolve())
    return any(port_dir.is_relative_to(root) for root in protected_roots)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        encoded_path = relative.as_posix().encode("utf-8")
        if path.is_symlink():
            digest.update(b"L\0" + encoded_path + b"\0")
            digest.update(os.readlink(path).encode("utf-8"))
        elif path.is_file():
            digest.update(b"F\0" + encoded_path + b"\0")
            with path.open("rb") as source_file:
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def apxinf_source_sha256() -> str:
    digest = hashlib.sha256()
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root(),
        check=True,
        capture_output=True,
    )
    for relative_bytes in result.stdout.split(b"\0"):
        if not relative_bytes:
            continue
        relative = relative_bytes.decode("utf-8")
        path = repository_root() / relative
        digest.update(relative_bytes + b"\0")
        if path.is_file():
            digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def initialize(args: argparse.Namespace) -> int:
    try:
        family_pack = select_family_pack(args.family)
    except ValueError as error:
        print(error, file=sys.stderr)
        return INVALID_INPUT.code
    port_dir = args.port_dir.resolve()
    source = args.source.resolve() if args.source is not None else None
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    if source is not None and not source.is_dir():
        print(f"source directory does not exist: {source}", file=sys.stderr)
        return MISSING_INPUT.code
    if checkpoint is not None and not checkpoint.is_file():
        print(f"checkpoint does not exist: {checkpoint}", file=sys.stderr)
        return MISSING_INPUT.code
    if port_dir_is_unsafe(port_dir, source):
        print("port directory must be outside source checkouts", file=sys.stderr)
        return INVALID_INPUT.code

    reference_values = (args.reference_entrypoint, args.dependency_lock)
    if any(reference_values) and not all(reference_values):
        print(
            "--reference-entrypoint and --dependency-lock must be supplied together",
            file=sys.stderr,
        )
        return INVALID_INPUT.code
    if all(reference_values) and source is None:
        print("reference inspection requires --source", file=sys.stderr)
        return INVALID_INPUT.code
    if source is not None and all(reference_values):
        for option, relative_path in (
            ("reference entrypoint", args.reference_entrypoint),
            ("dependency lock", args.dependency_lock),
        ):
            candidate = (source / relative_path).resolve()
            if not candidate.is_relative_to(source) or not candidate.is_file():
                print(f"{option} does not exist inside source: {relative_path}", file=sys.stderr)
                return MISSING_INPUT.code

    source_revision = args.source_revision
    source_name = source.name if source is not None else port_dir.name
    revision_suffix = source_revision[:8] if source_revision else "draft"

    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "port_id": f"{source_name}-{revision_suffix}",
        "model_family": family_pack.family,
        "source": {
            "path": str(source) if source is not None else None,
            "revision": source_revision,
            "sha256": source_sha256(source) if source is not None else None,
        },
        "checkpoint": {
            "path": str(checkpoint) if checkpoint is not None else None,
            "sha256": file_sha256(checkpoint) if checkpoint is not None else None,
        },
        "reference": {
            "entrypoint": args.reference_entrypoint,
            "dependency_lock": args.dependency_lock,
            "network_access": False,
        },
        "capability_contract_version": family_pack.default_contract_version,
        "representative_profiles": [{"name": None, "inputs": {}}],
        "requested_targets": [
            {
                "target": None,
                "precision": None,
                "latency_goal": {"p50_ms": None, "p95_ms": None},
            }
        ],
        "correctness_thresholds": {"absolute": None, "relative": None},
        "tuning_budgets": [{"target": None, "seconds": None}],
        "user_environment_declarations": {},
    }
    request_path = port_dir / "request.json"
    if request_path.exists():
        print(f"request already exists: {request_path}", file=sys.stderr)
        return 3
    write_json(request_path, request)
    if all(reference_values):
        adapter_path = port_dir / "private" / "reference_adapter.py"
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_text(
            reference_adapter_template().read_text(encoding="utf-8"), encoding="utf-8"
        )
    print(request_path)
    return 0


def environment_facts() -> dict[str, str]:
    return {
        "os": platform.system().lower(),
        "arch": platform.machine().lower(),
        "python": platform.python_version(),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def reject_non_json_number(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant: {value}")


def parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"number is outside the finite JSON range: {value}")
    return parsed


def failed_report(
    request: dict[str, Any], outcome: IntakeOutcome, issues: list[dict[str, str]]
) -> dict[str, Any]:
    safe_request = json_safe(request)
    if not isinstance(safe_request.get("port_id"), str):
        safe_request["port_id"] = None
    if not isinstance(safe_request.get("schema_version"), str):
        safe_request["schema_version"] = None
    family = safe_request.get("model_family")
    pack = select_family_pack(family) if family in {"vla"} else VLA_FAMILY_PACK
    return PortingCore.failed(safe_request, pack, outcome, issues, environment_facts()).report


def unsupported_target_issues(request: dict[str, Any]) -> list[dict[str, str]]:
    issues = []
    try:
        pack = select_family_pack(request.get("model_family"))
    except ValueError as error:
        return [{"path": "model_family", "message": str(error)}]
    for index, item in enumerate(request.get("requested_targets", [])):
        pair = (item.get("target"), item.get("precision"))
        if None in pair:
            continue
        try:
            validate_requested_tuple(pack, pair[0], pair[1])
        except ValueError as error:
            message = str(error)
            issues.append({"path": f"requested_targets[{index}]", "message": message})
    return issues


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def add_unknown_field_issues(
    issues: list[dict[str, str]], value: Any, allowed: set[str], path: str
) -> None:
    if not isinstance(value, dict):
        return
    for field in sorted(value.keys() - allowed):
        issues.append(
            {
                "path": f"{path}.{field}",
                "message": "field is not allowed by the request schema",
            }
        )


def schema_issues(request: Any) -> list[dict[str, str]]:
    if not isinstance(request, dict):
        return [{"path": "$", "message": "request must be a JSON object"}]

    issues = []
    allowed_fields = {
        "schema_version",
        "port_id",
        "model_family",
        "source",
        "checkpoint",
        "reference",
        "capability_contract_version",
        "representative_profiles",
        "requested_targets",
        "correctness_thresholds",
        "tuning_budgets",
        "user_environment_declarations",
    }
    required_fields = {
        "schema_version", "port_id", "model_family", "capability_contract_version"
    }
    for field in sorted(required_fields - request.keys()):
        issues.append({"path": field, "message": "field is required by the request schema"})
    add_unknown_field_issues(
        issues,
        request,
        allowed_fields,
        "$",
    )
    if (
        "schema_version" in request
        and request.get("schema_version") != REQUEST_SCHEMA_VERSION
    ):
        issues.append(
            {
                "path": "schema_version",
                "message": f"must equal {REQUEST_SCHEMA_VERSION}",
            }
        )
    for field in ("port_id",):
        if field in request and not isinstance(request[field], str):
            issues.append({"path": field, "message": "must be a string"})
    if "model_family" in request:
        try:
            select_family_pack(request["model_family"])
        except ValueError as error:
            issues.append({"path": "model_family", "message": str(error)})
    contract_version = request.get("capability_contract_version")
    if "capability_contract_version" in request and (
        ContractVersion.parse(contract_version) is None
    ):
        issues.append(
            {
                "path": "capability_contract_version",
                "message": "must be an exact major.minor version",
            }
        )
    for field in ("source", "checkpoint", "reference", "correctness_thresholds"):
        if field in request and not isinstance(request[field], dict):
            issues.append({"path": field, "message": "must be an object"})
    source = request.get("source")
    if isinstance(source, dict):
        for field in ("path", "revision"):
            value = source.get(field)
            if value is not None and not isinstance(value, str):
                issues.append({"path": f"source.{field}", "message": "must be a string"})
        digest = source.get("sha256")
        if digest is not None and not valid_sha256(digest):
            issues.append(
                {"path": "source.sha256", "message": "must be a lowercase SHA-256"}
            )
    checkpoint = request.get("checkpoint")
    if isinstance(checkpoint, dict):
        path_value = checkpoint.get("path")
        if path_value is not None and not isinstance(path_value, str):
            issues.append({"path": "checkpoint.path", "message": "must be a string"})
        digest = checkpoint.get("sha256")
        if digest is not None and not valid_sha256(digest):
            issues.append(
                {"path": "checkpoint.sha256", "message": "must be a lowercase SHA-256"}
            )
    add_unknown_field_issues(
        issues, request.get("source"), {"path", "revision", "sha256"}, "source"
    )
    add_unknown_field_issues(
        issues, request.get("checkpoint"), {"path", "sha256"}, "checkpoint"
    )
    reference = request.get("reference")
    if isinstance(reference, dict):
        entrypoint = reference.get("entrypoint")
        dependency_lock = reference.get("dependency_lock")
        for field, value in (
            ("entrypoint", entrypoint),
            ("dependency_lock", dependency_lock),
        ):
            if value is not None and (
                not isinstance(value, str)
                or not value
                or Path(value).is_absolute()
                or ".." in Path(value).parts
            ):
                issues.append(
                    {
                        "path": f"reference.{field}",
                        "message": "must be a source-relative path",
                    }
                )
        if (entrypoint is None) != (dependency_lock is None):
            issues.append(
                {
                    "path": "reference",
                    "message": "entrypoint and dependency_lock must be provided together",
                }
            )
        if reference.get("network_access") is not False:
            issues.append(
                {
                    "path": "reference.network_access",
                    "message": "must be false for trusted source execution",
                }
            )
        if entrypoint is not None:
            source_values = source if isinstance(source, dict) else {}
            checkpoint_values = checkpoint if isinstance(checkpoint, dict) else {}
            for path, value in (
                ("source.path", source_values.get("path")),
                ("source.revision", source_values.get("revision")),
                ("source.sha256", source_values.get("sha256")),
                ("checkpoint.path", checkpoint_values.get("path")),
                ("checkpoint.sha256", checkpoint_values.get("sha256")),
            ):
                if not value:
                    issues.append(
                        {
                            "path": path,
                            "message": "is required for reference inspection",
                        }
                    )
    add_unknown_field_issues(
        issues,
        reference,
        {"entrypoint", "dependency_lock", "network_access"},
        "reference",
    )
    add_unknown_field_issues(
        issues,
        request.get("correctness_thresholds"),
        {"absolute", "relative"},
        "correctness_thresholds",
    )

    profiles = request.get("representative_profiles")
    if profiles is not None and not isinstance(profiles, list):
        issues.append({"path": "representative_profiles", "message": "must be an array"})
    elif isinstance(profiles, list):
        for index, profile in enumerate(profiles):
            path = f"representative_profiles[{index}]"
            if not isinstance(profile, dict):
                issues.append({"path": path, "message": "must be an object"})
                continue
            add_unknown_field_issues(issues, profile, {"name", "inputs"}, path)
            if profile.get("name") is not None and not isinstance(profile.get("name"), str):
                issues.append({"path": f"{path}.name", "message": "must be a non-empty string"})
            inputs = profile.get("inputs")
            if not isinstance(inputs, dict):
                issues.append({"path": f"{path}.inputs", "message": "must be an object"})
            elif any(
                not isinstance(shape, list)
                or not shape
                or any(
                    not isinstance(dimension, int)
                    or isinstance(dimension, bool)
                    or dimension <= 0
                    for dimension in shape
                )
                for shape in inputs.values()
            ):
                issues.append(
                    {
                        "path": f"{path}.inputs",
                        "message": "shapes must contain positive integer dimensions",
                    }
                )

    requested = request.get("requested_targets")
    if requested is not None and not isinstance(requested, list):
        issues.append({"path": "requested_targets", "message": "must be an array"})
    elif isinstance(requested, list):
        for index, item in enumerate(requested):
            path = f"requested_targets[{index}]"
            if not isinstance(item, dict):
                issues.append({"path": path, "message": "must be an object"})
                continue
            add_unknown_field_issues(
                issues, item, {"target", "precision", "latency_goal"}, path
            )
            target = item.get("target")
            if target is not None and (
                not isinstance(target, str) or target not in TARGETS
            ):
                issues.append({"path": f"{path}.target", "message": "must be thor or orin"})
            precision = item.get("precision")
            if precision is not None and (
                not isinstance(precision, str) or precision not in PRECISIONS
            ):
                issues.append(
                    {
                        "path": f"{path}.precision",
                        "message": "must be bf16, fp8, or int8_w8a8",
                    }
                )
            goal = item.get("latency_goal")
            if goal is not None and not isinstance(goal, dict):
                issues.append({"path": f"{path}.latency_goal", "message": "must be an object"})
            elif isinstance(goal, dict):
                add_unknown_field_issues(
                    issues, goal, {"p50_ms", "p95_ms"}, f"{path}.latency_goal"
                )
                p50 = goal.get("p50_ms")
                p95 = goal.get("p95_ms")
                if p50 is not None and (not is_number(p50) or p50 <= 0):
                    issues.append(
                        {
                            "path": f"{path}.latency_goal",
                            "message": (
                                "p50_ms must be positive and p95_ms must be at least p50_ms"
                            ),
                        }
                    )
                elif p95 is not None and (
                    not is_number(p95) or (is_number(p50) and p95 < p50)
                ):
                    issues.append(
                        {
                            "path": f"{path}.latency_goal",
                            "message": (
                                "p50_ms must be positive and p95_ms must be at least p50_ms"
                            ),
                        }
                    )

    thresholds = request.get("correctness_thresholds")
    if isinstance(thresholds, dict):
        for field in ("absolute", "relative"):
            value = thresholds.get(field)
            if value is not None and (not is_number(value) or value < 0):
                issues.append(
                    {
                        "path": f"correctness_thresholds.{field}",
                        "message": "must be a non-negative number",
                    }
                )

    budgets = request.get("tuning_budgets")
    if budgets is not None and not isinstance(budgets, list):
        issues.append({"path": "tuning_budgets", "message": "must be an array"})
    elif isinstance(budgets, list):
        for index, budget in enumerate(budgets):
            path = f"tuning_budgets[{index}]"
            if not isinstance(budget, dict):
                issues.append({"path": path, "message": "must be an object"})
                continue
            add_unknown_field_issues(issues, budget, {"target", "seconds"}, path)
            target = budget.get("target")
            if target is not None and (
                not isinstance(target, str) or target not in TARGETS
            ):
                issues.append({"path": f"{path}.target", "message": "must be thor or orin"})
            if budget.get("seconds") is not None and (
                not is_number(budget.get("seconds")) or budget.get("seconds", 0) <= 0
            ):
                issues.append({"path": f"{path}.seconds", "message": "must be positive"})

    declarations = request.get("user_environment_declarations")
    if declarations is not None and not isinstance(declarations, dict):
        issues.append(
            {"path": "user_environment_declarations", "message": "must be an object"}
        )
    return issues


def normalize_request(request: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(request)
    normalized.setdefault("source", {})
    normalized.setdefault("checkpoint", {})
    normalized.setdefault(
        "reference",
        {"entrypoint": None, "dependency_lock": None, "network_access": False},
    )
    normalized.setdefault("representative_profiles", [])
    normalized.setdefault("requested_targets", [])
    normalized.setdefault("correctness_thresholds", {})
    normalized.setdefault("tuning_budgets", [])
    normalized.setdefault("user_environment_declarations", {})
    return normalized


def missing_fact_warnings(request: dict[str, Any]) -> list[dict[str, str]]:
    warnings = []
    required_values = (
        ("source.path", request.get("source", {}).get("path")),
        ("source.revision", request.get("source", {}).get("revision")),
        ("source.sha256", request.get("source", {}).get("sha256")),
        ("checkpoint.path", request.get("checkpoint", {}).get("path")),
        ("checkpoint.sha256", request.get("checkpoint", {}).get("sha256")),
        (
            "correctness_thresholds.absolute",
            request.get("correctness_thresholds", {}).get("absolute"),
        ),
        (
            "correctness_thresholds.relative",
            request.get("correctness_thresholds", {}).get("relative"),
        ),
    )
    for path, value in required_values:
        if value is None or value == "" or value == []:
            warnings.append(
                {
                    "path": path,
                    "message": "not provided; related guarantees are unavailable",
                }
            )

    profiles = request.get("representative_profiles", [])
    if not profiles:
        warnings.append(
            {
                "path": "representative_profiles",
                "message": "not provided; correctness coverage is unavailable",
            }
        )
    for index, profile in enumerate(profiles):
        for field in ("name", "inputs"):
            value = profile.get(field)
            if value is None or value == "" or value == {}:
                warnings.append(
                    {
                        "path": f"representative_profiles[{index}].{field}",
                        "message": "not provided; correctness coverage is unavailable",
                    }
                )

    requested = request.get("requested_targets", [])
    if not requested:
        warnings.append(
            {
                "path": "requested_targets",
                "message": "not provided; no target/precision work is guaranteed",
            }
        )
    for index, item in enumerate(requested):
        for field in ("target", "precision"):
            if not item.get(field):
                warnings.append(
                    {
                        "path": f"requested_targets[{index}].{field}",
                        "message": "not provided; target qualification is unavailable",
                    }
                )
        goal = item.get("latency_goal", {})
        for field in ("p50_ms", "p95_ms"):
            if goal.get(field) is None:
                warnings.append(
                    {
                        "path": f"requested_targets[{index}].latency_goal.{field}",
                        "message": "not provided; performance is not guaranteed",
                    }
                )

    budgets = request.get("tuning_budgets", [])
    if not budgets:
        warnings.append(
            {
                "path": "tuning_budgets",
                "message": "not provided; tuning is not guaranteed",
            }
        )
    for index, budget in enumerate(budgets):
        for field in ("target", "seconds"):
            if budget.get(field) is None:
                warnings.append(
                    {
                        "path": f"tuning_budgets[{index}].{field}",
                        "message": "not provided; tuning is not guaranteed",
                    }
                )
    requested_targets = {
        item.get("target") for item in requested if item.get("target") is not None
    }
    budget_targets = {
        budget.get("target") for budget in budgets if budget.get("target") is not None
    }
    for target in sorted(requested_targets - budget_targets):
        warnings.append(
            {
                "path": f"tuning_budgets[{target}]",
                "message": f"a tuning budget is required for requested target {target}",
            }
        )
    return warnings


def provenance_issues(
    request: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    missing = []
    invalid = []
    source_path = request["source"].get("path")
    source_digest = request["source"].get("sha256")
    checkpoint_path = request["checkpoint"].get("path")
    checkpoint_digest = request["checkpoint"].get("sha256")
    source = Path(source_path) if source_path else None
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if source is not None and not source.is_dir():
        missing.append(
            {"path": "source.path", "message": "source directory no longer exists"}
        )
    elif source is not None and source_digest and source_sha256(source) != source_digest:
        invalid.append(
            {
                "path": "source.sha256",
                "message": "source content no longer matches its pinned SHA-256",
            }
        )
    if checkpoint is not None and not checkpoint.is_file():
        missing.append(
            {"path": "checkpoint.path", "message": "checkpoint no longer exists"}
        )
    elif (
        checkpoint is not None
        and checkpoint_digest
        and file_sha256(checkpoint) != checkpoint_digest
    ):
        invalid.append(
            {
                "path": "checkpoint.sha256",
                "message": "checkpoint content no longer matches its pinned SHA-256",
            }
        )
    return missing, invalid


def artifact_record(
    port_dir: Path,
    path: Path,
    request: dict[str, Any],
    environment_path: Path,
    extra_upstream: dict[str, Path] | None = None,
) -> dict[str, Any]:
    pack = select_family_pack(request["model_family"])
    request = copy.deepcopy(request)
    environment_sha = file_sha256(environment_path)
    contract_path = (extra_upstream or {}).get("capability_contract")
    if contract_path is None:
        contract_path = default_capability_contract(
            request["capability_contract_version"]
        )
    contract_sha = (
        file_sha256(contract_path)
        if contract_path.is_file()
        else value_sha256(request["capability_contract_version"])
    )
    kernel_path = (extra_upstream or {}).get(
        "kernel_capabilities", default_kernel_capabilities()
    )
    source_dependent = path.name in {
        "environment.json",
        "source_inventory.json",
        "inspection.json",
    }
    kernel_dependent = path.name in {
        "kernel_coverage.json",
        "kernel_gap_handoff.json",
    }
    contract_dependent = path.name in {
        "capability_classification.json",
        "capability_gap_report.json",
    }
    request["dependency_fingerprints"] = {
        "source_sha256": request["source"]["sha256"] if source_dependent else None,
        "checkpoint_sha256": (
            request["checkpoint"]["sha256"] if source_dependent else None
        ),
        "apxinf_source_sha256": apxinf_source_sha256(),
        "kernel_build_sha256": (
            file_sha256(kernel_path) if kernel_dependent else None
        ),
        "environment_sha256": environment_sha if source_dependent else None,
        "capability_contract_sha256": (
            contract_sha if contract_dependent else None
        ),
        "documentation_sha256": None,
        "target_environment_sha256": (
            {
                f"{item['target']}/{item['precision']}": environment_sha
                for item in request.get("requested_targets", [])
            }
            if kernel_dependent
            else {}
        ),
    }
    store = ArtifactStore(
        port_dir,
        request,
        pack,
        Path(__file__).resolve(),
        port_dir / "private" / "reference_adapter.py",
    )
    return store.record(path, environment_path, extra_upstream)


def resume_port(args: argparse.Namespace) -> int:
    port_dir = args.port_dir.resolve()
    try:
        request = json.loads((port_dir / "request.json").read_text(encoding="utf-8"))
        report = json.loads((port_dir / "report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"cannot resume Port: {error}", file=sys.stderr)
        return MISSING_INPUT.code
    current: dict[str, dict[str, Any]] = {}
    for name, recorded in report.get("artifacts", {}).items():
        refreshed = copy.deepcopy(recorded)
        fingerprints = refreshed["fingerprints"]
        dependency_paths = refreshed.get("dependency_paths", {})
        payload_path = port_dir / recorded.get("path", "")
        if payload_path.is_file():
            fingerprints["content_sha256"] = file_sha256(payload_path)
        else:
            fingerprints["content_sha256"] = None
        source_path = request.get("source", {}).get("path")
        checkpoint_path = request.get("checkpoint", {}).get("path")
        if fingerprints.get("source_sha256") is not None:
            fingerprints["source_sha256"] = (
                source_sha256(Path(source_path)) if source_path else None
            )
        if fingerprints.get("checkpoint_sha256") is not None:
            fingerprints["checkpoint_sha256"] = (
                file_sha256(Path(checkpoint_path)) if checkpoint_path else None
            )
        fingerprints["apxinf_source_sha256"] = apxinf_source_sha256()
        for tool_name in list(fingerprints.get("tool_sha256", {})):
            tool_path = dependency_paths.get(tool_name)
            fingerprints["tool_sha256"][tool_name] = (
                file_sha256(Path(tool_path))
                if tool_path and Path(tool_path).is_file()
                else None
            )
        environment_path = dependency_paths.get("environment")
        if environment_path and fingerprints.get("environment_sha256") is not None:
            environment_digest = (
                file_sha256(Path(environment_path))
                if Path(environment_path).is_file()
                else None
            )
            fingerprints["environment_sha256"] = environment_digest
            fingerprints["target_environment_sha256"] = {
                key: environment_digest
                for key in fingerprints.get("target_environment_sha256", {})
            }
        kernel_path = Path(
            dependency_paths.get(
                "kernel_capabilities", str(default_kernel_capabilities())
            )
        )
        if fingerprints.get("kernel_build_sha256") is not None:
            fingerprints["kernel_build_sha256"] = (
                file_sha256(kernel_path) if kernel_path.is_file() else None
            )
        contract_path = Path(
            dependency_paths.get(
                "capability_contract",
                str(
                    default_capability_contract(
                        request["capability_contract_version"]
                    )
                ),
            )
        )
        if fingerprints.get("capability_contract_sha256") is not None:
            fingerprints["capability_contract_sha256"] = (
                file_sha256(contract_path)
                if contract_path.is_file()
                else value_sha256(request["capability_contract_version"])
            )
        if fingerprints.get("documentation_sha256") is not None:
            fingerprints["documentation_sha256"] = file_sha256(
                repository_root() / "doc" / "porting-workflow.md"
            )
        upstream = fingerprints.get("upstream_sha256", {})
        if "request" in upstream:
            upstream["request"] = file_sha256(port_dir / "request.json")
        for dependency_name in list(upstream):
            dependency = report.get("artifacts", {}).get(dependency_name)
            if dependency is not None:
                dependency_path = port_dir / dependency["path"]
                upstream[dependency_name] = (
                    file_sha256(dependency_path) if dependency_path.is_file() else None
                )
            elif dependency_name in dependency_paths:
                dependency_path = Path(dependency_paths[dependency_name])
                upstream[dependency_name] = (
                    file_sha256(dependency_path) if dependency_path.is_file() else None
                )
        current[name] = refreshed
    resumed = resume_report(report, current)
    write_json(port_dir / "report.json", resumed)
    print(port_dir / "report.json")
    return SUCCESS.code


def value_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_input_issue(
    inventory: dict[str, Any], capture: dict[str, Any]
) -> dict[str, str] | None:
    cases = capture.get("profiles", [])
    seed_zero_inputs = [case.get("inputs") for case in cases if case.get("seed") == 0]
    if len({value_sha256(value) for value in seed_zero_inputs}) < 2:
        return {
            "path": "representative_profiles",
            "message": (
                "canonical equivalence requires at least two distinct preprocessed "
                "representative inputs"
            ),
        }
    if inventory.get("stochastic_inputs"):
        seeded_inputs_differ = any(
            value_sha256(cases[index].get("inputs"))
            != value_sha256(cases[index + 1].get("inputs"))
            for index in range(0, len(cases) - 1, 2)
            if cases[index].get("seed") == 0 and cases[index + 1].get("seed") == 1
        )
        if not seeded_inputs_differ:
            return {
                "path": "stochastic_inputs",
                "message": (
                    "declared stochastic inputs must exercise at least two random seeds"
                ),
            }
    return None


def write_direct_canonical_evidence(
    port_dir: Path,
    request: dict[str, Any],
    inventory: dict[str, Any],
    capture: dict[str, Any],
    classification: dict[str, Any],
    environment_path: Path,
    upstream_paths: dict[str, Path],
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    trace = canonical_trace_document(
        request["port_id"], "direct", classification, capture["profiles"], inventory
    )
    trace_path = port_dir / "private" / "canonical_trace.json"
    write_json(trace_path, trace)
    cases = []
    for case in trace["cases"]:
        comparisons = [
            {
                "scope": scope,
                "source_path": source_path,
                "canonical_path": source_path,
                "passed": True,
                "max_absolute_error": 0.0,
                "max_relative_error": 0.0,
                "mismatch_reason": None,
            }
            for scope, source_path in (
                ("preprocessed_inputs", "inputs"),
                ("intermediates", "intermediates"),
                ("normalized_actions", "output.actions"),
                ("postprocessed_actions", "postprocessed.actions"),
            )
        ]
        cases.append(
            {
                "profile": case["profile"],
                "seed": case["seed"],
                "comparisons": comparisons,
                "passed": True,
            }
        )
    parameter_mapping = [
        {
            "sources": [parameter["name"]],
            "targets": [parameter["name"]],
            "transformation_ids": [],
        }
        for parameter in inventory["parameters"]
    ]
    evidence = canonical_evidence_document(
        port_id=request["port_id"],
        mode="direct",
        classification=classification,
        inventory=inventory,
        trace=trace,
        thresholds=request["correctness_thresholds"],
        parameter_mapping=parameter_mapping,
        canonical_parameters=inventory["parameters"],
        canonical_aliases=inventory["aliases"],
        canonical_tied_weights=inventory["tied_weights"],
        transformations=[],
        cases=cases,
    )
    evidence_path = port_dir / "private" / "canonical_equivalence.json"
    write_json(evidence_path, evidence)
    artifacts = {
        "canonical_trace": artifact_record(
            port_dir, trace_path, request, environment_path, upstream_paths
        ),
        "canonical_equivalence": artifact_record(
            port_dir,
            evidence_path,
            request,
            environment_path,
            {**upstream_paths, "canonical_trace": trace_path},
        ),
    }
    return evidence["summary"], artifacts


def previous_capability_contract_path(path: Path, version: str) -> Path:
    candidates = (
        path.parent / f"vla-capability-contract-{version}.json",
        path.parent / f"capability-contract-{version}.json",
        default_capability_contract(version),
    )
    for candidate in candidates:
        if candidate != path and candidate.is_file():
            return candidate
    raise ValueError(f"previous capability contract {version} is unavailable")


def capability_change_is_additive(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> bool:
    if before is None:
        return after is not None and not after["required"]
    if after is None:
        return False
    return (
        before["required"] == after["required"]
        and before["cardinality"] == after["cardinality"]
        and set(before["supported"]) <= set(after["supported"])
        and set(before["canonicalizable"].items())
        <= set(after["canonicalizable"].items())
    )


def validate_contract_delta(
    previous: dict[str, Any], current: dict[str, Any]
) -> None:
    previous_rules = previous["capabilities"]
    current_rules = current["capabilities"]
    changes = current["revision"]["changes"]
    declared = {change["capability"]: change["kind"] for change in changes}
    if len(declared) != len(changes):
        raise ValueError("contract changes must name each capability once")
    changed = {
        capability
        for capability in set(previous_rules) | set(current_rules)
        if previous_rules.get(capability) != current_rules.get(capability)
    }
    if set(declared) != changed:
        raise ValueError(
            "contract revision changes must exactly name changed capabilities"
        )

    additive_changes = {
        capability: capability_change_is_additive(
            previous_rules.get(capability), current_rules.get(capability)
        )
        for capability in changed
    }
    if current["revision"]["kind"] == "breaking":
        if additive_changes and all(additive_changes.values()):
            raise ValueError("additive-only contract changes must increment minor")
        return
    for capability in changed:
        if not additive_changes[capability]:
            raise ValueError(
                "additive contracts may only add supported or canonicalizable values"
            )


def load_capability_contract(
    path: Path, expected_version: str, pack: FamilyPack | None = None
) -> dict[str, Any]:
    pack = pack or select_family_pack("vla")
    contract = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_non_json_number,
        parse_float=parse_finite_float,
    )
    if not isinstance(contract, dict):
        raise ValueError("capability contract must be a JSON object")
    expected_fields = {
        "schema_version",
        "contract_version",
        "family",
        "revision",
        "capabilities",
    }
    if set(contract) != expected_fields:
        raise ValueError("capability contract has missing or unknown top-level fields")
    if contract.get("schema_version") != "1.0":
        raise ValueError("capability contract schema_version must equal 1.0")
    if contract.get("family") != pack.contract_family:
        raise ValueError(
            f"capability contract family must be {pack.contract_family}"
        )
    if contract.get("contract_version") != expected_version:
        raise ValueError(
            "capability contract version does not match the exact request pin"
        )
    published_path = default_capability_contract(expected_version)
    if published_path.is_file() and path.resolve() != published_path.resolve():
        published = json.loads(
            published_path.read_text(encoding="utf-8"),
            parse_constant=reject_non_json_number,
            parse_float=parse_finite_float,
        )
        if value_sha256(published) != value_sha256(contract):
            raise ValueError(
                "published capability contract content is immutable at a fixed version"
            )
    version = ContractVersion.parse(expected_version)
    if version is None:
        raise ValueError("capability contract version must be major.minor")
    revision = contract.get("revision")
    if not isinstance(revision, dict):
        raise ValueError("capability contract must declare revision metadata")
    if set(revision) != {"kind", "previous_version", "changes"}:
        raise ValueError("capability contract revision has missing or unknown fields")
    kind = revision.get("kind")
    previous_version = revision.get("previous_version")
    changes = revision.get("changes")
    if not isinstance(changes, list):
        raise ValueError("capability contract revision changes must be an array")
    if kind == "initial":
        if previous_version is not None or changes:
            raise ValueError("an initial contract cannot declare previous changes")
    elif kind in {"additive", "breaking"}:
        previous = ContractVersion.parse(previous_version)
        if previous is None:
            raise ValueError("updated contracts require a previous major.minor version")
        if any(
            not isinstance(change, dict)
            or set(change) != {"capability", "kind"}
            or not isinstance(change.get("capability"), str)
            or not change.get("capability")
            for change in changes
        ):
            raise ValueError("contract changes must declare capability and kind")
        change_kinds = {change.get("kind") for change in changes}
        if not change_kinds <= {"additive", "changed", "removed"}:
            raise ValueError("contract change kind is invalid")
        if kind == "additive":
            if version.major != previous.major or version.minor <= previous.minor:
                raise ValueError("additive contracts must increment the minor version")
            if not changes or change_kinds != {"additive"}:
                raise ValueError("additive contracts may declare only additive changes")
        elif version.major <= previous.major:
            raise ValueError("breaking contracts must increment the major version")
        elif not change_kinds.intersection({"changed", "removed"}):
            raise ValueError(
                "breaking contracts must declare changed or removed semantics"
            )
    else:
        raise ValueError("capability contract revision kind is invalid")
    capabilities = contract.get("capabilities")
    if not isinstance(capabilities, dict) or not capabilities:
        raise ValueError("capability contract must declare capabilities")
    missing_capabilities = sorted(pack.required_capabilities - capabilities.keys())
    if missing_capabilities:
        raise ValueError(
            "capability contract is missing required capabilities: "
            + ", ".join(missing_capabilities)
        )
    for name, rule in capabilities.items():
        if not isinstance(name, str) or not isinstance(rule, dict):
            raise ValueError("capability rules must be named JSON objects")
        if set(rule) != {
            "required",
            "cardinality",
            "supported",
            "canonicalizable",
        }:
            raise ValueError(f"capability {name} has missing or unknown fields")
        supported = rule.get("supported")
        if not isinstance(supported, list) or any(
            not isinstance(value, str) or not value for value in supported
        ):
            raise ValueError(f"capability {name} must declare supported values")
        canonicalizable = rule.get("canonicalizable")
        if not isinstance(canonicalizable, dict) or any(
            not isinstance(source, str)
            or not source
            or not isinstance(target, str)
            or not target
            for source, target in canonicalizable.items()
        ):
            raise ValueError(
                f"capability {name} must declare canonicalizable mappings"
            )
        if not set(canonicalizable.values()) <= set(supported):
            raise ValueError(
                f"capability {name} canonical targets must be supported values"
            )
        if not isinstance(rule.get("required"), bool):
            raise ValueError(f"capability {name} required must be boolean")
        if rule.get("cardinality") != "exactly_one":
            raise ValueError(f"capability {name} cardinality must be exactly_one")
        if name in pack.required_capabilities and not rule["required"]:
            raise ValueError(f"core capability {name} must remain required")
    for change in changes:
        capability = change["capability"]
        change_kind = change["kind"]
        if change_kind in {"additive", "changed"} and capability not in capabilities:
            raise ValueError(
                f"contract change for {capability} has no capability declaration"
            )
        if change_kind == "removed" and capability in capabilities:
            raise ValueError(
                f"removed capability {capability} must not remain declared"
            )
    if kind != "initial":
        previous_path = previous_capability_contract_path(path, previous_version)
        previous_contract = load_capability_contract(
            previous_path, previous_version
        )
        validate_contract_delta(previous_contract, contract)
    return contract


def classification_record(
    path: str,
    capability: str,
    observed: Any,
    classification: str,
    canonical: str | None,
    reason: str,
    evidence_paths: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "path": path,
        "capability": capability,
        "observed": observed,
        "classification": classification,
        "canonical": canonical,
        "reason": reason,
        "evidence_paths": evidence_paths or [path],
    }


def inventory_capability_evidence(
    inventory: dict[str, Any],
) -> tuple[dict[str, list[tuple[str, str]]], list[dict[str, Any]]]:
    evidence: dict[str, list[tuple[str, str]]] = {}
    issues: list[dict[str, Any]] = []

    def add(capability: str, path: str, value: str) -> None:
        evidence.setdefault(capability, []).append((path, value))

    if isinstance(inventory.get("input_schema"), list) and inventory["input_schema"]:
        add("shape_profiles", "input_schema", "finite")

    normalization = inventory.get("normalization")
    if isinstance(normalization, dict) and "model" in normalization:
        value = normalization["model"]
        if isinstance(value, str) and value:
            add("normalization", "normalization.model", value)
        else:
            issues.append(
                classification_record(
                    "normalization.model",
                    "normalization",
                    value,
                    "unsupported",
                    None,
                    "model normalization must be a non-empty string",
                )
            )

    schedules = inventory.get("schedules", [])
    if isinstance(schedules, list):
        for index, schedule in enumerate(schedules):
            path = f"schedules[{index}].kind"
            if isinstance(schedule, dict) and isinstance(schedule.get("kind"), str):
                add("schedules", path, schedule["kind"])
            else:
                issues.append(
                    classification_record(
                        path,
                        "schedules",
                        schedule,
                        "unsupported",
                        None,
                        "schedule semantics are unexplained",
                    )
                )

    branches = inventory.get("dynamic_branches", [])
    if isinstance(branches, list) and not branches:
        add("control_flow", "dynamic_branches", "static")
    elif isinstance(branches, list):
        for index, branch in enumerate(branches):
            path = f"dynamic_branches[{index}].kind"
            if isinstance(branch, dict) and isinstance(branch.get("kind"), str):
                add("control_flow", path, branch["kind"])
            else:
                issues.append(
                    classification_record(
                        f"dynamic_branches[{index}]",
                        "control_flow",
                        branch,
                        "unsupported",
                        None,
                        "dynamic control flow is unexplained",
                    )
                )

    for index, trace in enumerate(inventory.get("operator_traces", [])):
        if not isinstance(trace, dict) or "semantic_capabilities" not in trace:
            continue
        semantics = trace["semantic_capabilities"]
        if not isinstance(semantics, dict):
            issues.append(
                classification_record(
                    f"operator_traces[{index}].semantic_capabilities",
                    "operator_traces",
                    semantics,
                    "unsupported",
                    None,
                    "operator semantic capabilities must be an object",
                )
            )
            continue
        for capability, value in semantics.items():
            path = f"operator_traces[{index}].semantic_capabilities.{capability}"
            values = value if isinstance(value, list) else [value]
            for observed in values:
                if isinstance(observed, str) and observed:
                    add(capability, path, observed)
                else:
                    issues.append(
                        classification_record(
                            path,
                            capability,
                            observed,
                            "unsupported",
                            None,
                            "operator semantic capability must be a non-empty string",
                        )
                    )
    return evidence, issues


def classify_capabilities(
    inventory: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    raw_facts = inventory.get("capability_facts", {})
    facts = raw_facts if isinstance(raw_facts, dict) else {}
    rules = contract["capabilities"]
    evidence, classifications = inventory_capability_evidence(inventory)
    if not isinstance(raw_facts, dict):
        classifications.append(
            classification_record(
                "capability_facts",
                "capability_facts",
                raw_facts,
                "unsupported",
                None,
                "capability facts must be a JSON object",
            )
        )
    for capability in sorted(set(rules) | set(facts) | set(evidence)):
        observed_values = facts.get(capability, [])
        rule = rules.get(capability)
        capability_evidence = evidence.get(capability, [])
        if not isinstance(observed_values, list):
            classifications.append(
                classification_record(
                    f"capability_facts.{capability}",
                    capability,
                    observed_values,
                    "unsupported",
                    None,
                    "capability observations must be an array",
                )
            )
            continue
        if rule is None:
            values = observed_values or [None]
            for index, observed in enumerate(values):
                classifications.append(
                    classification_record(
                        f"capability_facts.{capability}[{index}]",
                        capability,
                        observed,
                        "unsupported",
                        None,
                        "capability is not declared by the contract",
                    )
                )
            continue
        if not observed_values and not rule["required"] and not capability_evidence:
            continue
        if not observed_values:
            classifications.append(
                classification_record(
                    f"capability_facts.{capability}",
                    capability,
                    None,
                    "unsupported",
                    None,
                    "required capability fact is unknown",
                    [path for path, _ in capability_evidence]
                    or [f"capability_facts.{capability}"],
                )
            )
            continue
        string_values = [
            observed for observed in observed_values if isinstance(observed, str)
        ]
        if len(string_values) != len(observed_values):
            classifications.append(
                classification_record(
                    f"capability_facts.{capability}",
                    capability,
                    observed_values,
                    "unsupported",
                    None,
                    "capability observations must be non-empty strings",
                )
            )
        if len(set(string_values)) != 1:
            classifications.append(
                classification_record(
                    f"capability_facts.{capability}",
                    capability,
                    observed_values,
                    "unsupported",
                    None,
                    "capability observations are contradictory",
                )
            )
        evidence_values = {value for _, value in capability_evidence}
        evidence_matches = evidence_values == set(string_values)
        if (
            capability == "shape_profiles"
            and evidence_values == {"finite"}
            and set(string_values) in ({"static"}, {"finite"})
        ):
            evidence_matches = True
        if evidence_values and not evidence_matches:
            classifications.append(
                classification_record(
                    capability_evidence[0][0],
                    capability,
                    sorted(evidence_values),
                    "unsupported",
                    None,
                    "inventory semantics contradict declared capability facts",
                    [path for path, _ in capability_evidence]
                    + [f"capability_facts.{capability}"],
                )
            )
        for index, observed in enumerate(observed_values):
            if not isinstance(observed, str) or not observed:
                classification = "unsupported"
                canonical = None
                reason = "capability observation must be a non-empty string"
            elif observed in rule["supported"]:
                classification = "supported"
                canonical = observed
                reason = "declared supported semantic"
            elif observed in rule["canonicalizable"]:
                classification = "canonicalizable"
                canonical = rule["canonicalizable"][observed]
                reason = "declared canonicalizable semantic"
            else:
                classification = "unsupported"
                canonical = None
                reason = "semantic is not declared supported or canonicalizable"
            classifications.append(
                classification_record(
                    f"capability_facts.{capability}[{index}]",
                    capability,
                    observed,
                    classification,
                    canonical,
                    reason,
                    [f"capability_facts.{capability}[{index}]"]
                    + [
                        path
                        for path, evidence_value in capability_evidence
                        if evidence_value == observed
                        or (
                            capability == "shape_profiles"
                            and evidence_value == "finite"
                            and observed == "static"
                        )
                    ],
                )
            )
    for index, observed in enumerate(inventory.get("custom_operators", [])):
        classifications.append(
            classification_record(
                f"custom_operators[{index}]",
                "custom_operators",
                observed,
                "unsupported",
                None,
                "unexplained custom operator is outside the contract",
            )
        )
    counts = {
        kind: sum(item["classification"] == kind for item in classifications)
        for kind in ("supported", "canonicalizable", "unsupported")
    }
    return {
        "schema_version": "1.0",
        "contract": {
            "version": contract["contract_version"],
            "sha256": value_sha256(contract),
        },
        "source_inventory_sha256": value_sha256(inventory),
        "classifications": classifications,
        "summary": counts,
        "dependency_fingerprints": {
            capability: value_sha256(rule)
            for capability, rule in sorted(rules.items())
            if capability in facts
        },
    }


def offline_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
        "PIP_CONFIG_FILE",
        "PIP_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS",
        "PIP_INDEX_URL",
        "PIP_TRUSTED_HOST",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "APXINF_REFERENCE_NETWORK": "disabled",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "PIP_NO_INDEX": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def run_reference_inspection(
    port_dir: Path, request: dict[str, Any]
) -> tuple[IntakeOutcome, list[dict[str, str]], dict[str, dict[str, Any]]]:
    source = Path(request["source"]["path"])
    checkpoint = Path(request["checkpoint"]["path"])
    reference = request["reference"]
    entrypoint = reference["entrypoint"]
    dependency_lock = reference["dependency_lock"]
    entrypoint_path = (source / entrypoint).resolve()
    lock_path = (source / dependency_lock).resolve()
    for issue_path, path in (
        ("reference.entrypoint", entrypoint_path),
        ("reference.dependency_lock", lock_path),
    ):
        if not path.is_relative_to(source.resolve()) or not path.is_file():
            issue = {"path": issue_path, "message": "declared reference file is missing"}
            return MISSING_INPUT, [issue], {}

    private_dir = port_dir / "private"
    adapter_path = private_dir / "reference_adapter.py"
    if not adapter_path.is_file():
        issue = {
            "path": "artifacts.reference_adapter",
            "message": "generated private Reference Adapter is missing; re-run init",
        }
        return MISSING_INPUT, [issue], {}

    environment_dir = private_dir / "reference_environment"
    environment_id = secrets.token_hex(12)
    virtual_environment = environment_dir / "venvs" / environment_id
    environment_path = environment_dir / "environment.json"
    try:
        venv.EnvBuilder(with_pip=True).create(virtual_environment)
        python = virtual_environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        lock_text = lock_path.read_text(encoding="utf-8")
        requirements = [
            line
            for line in lock_text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        active_lock_text = "\n".join(requirements).lower()
        if "http://" in active_lock_text or "https://" in active_lock_text:
            raise RuntimeError("dependency lock contains a network URL")
        if requirements:
            installed = subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--require-hashes",
                    "--disable-pip-version-check",
                    "--no-input",
                    "-r",
                    str(lock_path),
                ],
                cwd=source,
                env=offline_environment(),
                capture_output=True,
                text=True,
                check=False,
                timeout=300,
            )
            if installed.returncode != 0:
                message = installed.stderr.strip() or installed.stdout.strip()
                raise RuntimeError(f"locked dependency installation failed: {message}")
        listed = subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "list",
                "--format=json",
                "--disable-pip-version-check",
            ],
            cwd=source,
            env=offline_environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if listed.returncode != 0:
            message = listed.stderr.strip() or listed.stdout.strip()
            raise RuntimeError(f"could not inventory locked environment: {message}")
        installed_distributions = sorted(
            json.loads(listed.stdout), key=lambda distribution: distribution["name"].lower()
        )
        environment = {
            "schema_version": "1.0",
            "python": platform.python_version(),
            "dependency_lock": {
                "path": dependency_lock,
                "sha256": file_sha256(lock_path),
            },
            "isolation": {
                "kind": "venv",
                "environment_id": environment_id,
                "system_site_packages": False,
            },
            "installed_distributions": installed_distributions,
            "runtime_network_access": False,
            "network_enforcement": ["offline_environment", "python_socket_guard"],
        }
        write_json(environment_path, environment)
    except (
        OSError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        RuntimeError,
    ) as error:
        issue = {
            "path": "reference.environment",
            "message": f"could not prepare locked reference environment: {error}",
        }
        return ENVIRONMENT_FAILURE, [issue], {}

    profiles_path = private_dir / "reference_profiles.json"
    inventory_path = private_dir / "source_inventory.json"
    capture_path = private_dir / "captures" / "inspection.json"
    result_path = (
        private_dir / "inspection_results" / f"{secrets.token_hex(12)}.json"
    )
    profiles_path.write_text(
        json.dumps(request["representative_profiles"], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    command = [
        str(python),
        "-I",
        "-B",
        str(adapter_path),
        "--source-root",
        str(source),
        "--entrypoint",
        entrypoint,
        "--checkpoint",
        str(checkpoint),
        "--profiles",
        str(profiles_path),
        "--inventory",
        str(inventory_path),
        "--capture",
        str(capture_path),
        "--result",
        str(result_path),
        "--source-revision",
        request["source"]["revision"],
        "--source-sha256",
        request["source"]["sha256"],
        "--checkpoint-sha256",
        request["checkpoint"]["sha256"],
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=source,
            env=offline_environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"reference process exited {completed.returncode}: {detail}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError, RuntimeError) as error:
        issue = {
            "path": "reference.environment",
            "message": f"could not execute Reference Adapter: {error}",
        }
        return ENVIRONMENT_FAILURE, [issue], {
            "reference_adapter": artifact_record(
                port_dir, adapter_path, request, environment_path
            ),
            "reference_environment": artifact_record(
                port_dir, environment_path, request, environment_path
            ),
        }

    status = result.get("status")
    def record(path: Path) -> dict[str, Any]:
        return artifact_record(port_dir, path, request, environment_path)

    artifacts = {
        "reference_adapter": record(adapter_path),
        "reference_environment": record(environment_path),
    }
    if status == "load_failure":
        issue = {
            "path": "reference.load",
            "message": f"trusted source could not be loaded: {result.get('message', 'unknown error')}",
        }
        return REFERENCE_LOAD_FAILURE, [issue], artifacts
    if status == "trace_failure":
        issue = {
            "path": "reference.trace",
            "message": f"trusted source could not be traced: {result.get('message', 'unknown error')}",
        }
        return REFERENCE_TRACE_FAILURE, [issue], artifacts
    if status != "success" or not inventory_path.is_file() or not capture_path.is_file():
        issue = {
            "path": "reference.trace",
            "message": "Reference Adapter did not produce complete inspection artifacts",
        }
        return REFERENCE_TRACE_FAILURE, [issue], artifacts

    try:
        current_source_sha256 = source_sha256(source)
    except OSError as error:
        issue = {
            "path": "source.sha256",
            "message": f"could not verify source after inspection: {error}",
        }
        return ENVIRONMENT_FAILURE, [issue], artifacts
    if current_source_sha256 != request["source"]["sha256"]:
        issue = {
            "path": "source.sha256",
            "message": "trusted source changed while reference inspection was running",
        }
        return INVALID_INPUT, [issue], artifacts

    artifacts.update(
        {
            "source_inventory": record(inventory_path),
            "private_capture": record(capture_path),
        }
    )
    return SUCCESS, [], artifacts


def run_canonical_verification(
    port_dir: Path,
    request: dict[str, Any],
    inventory_path: Path,
    classification_path: Path,
    environment_path: Path,
) -> tuple[
    IntakeOutcome,
    list[dict[str, str]],
    dict[str, dict[str, Any]],
    dict[str, int],
    list[dict[str, str]],
]:
    private_dir = port_dir / "private"
    adapter_path = private_dir / "canonical_adapter.py"
    adapter_path.write_text(
        canonical_adapter_template().read_text(encoding="utf-8"), encoding="utf-8"
    )
    profiles_path = private_dir / "reference_profiles.json"
    thresholds_path = private_dir / "correctness_thresholds.json"
    write_json(thresholds_path, request["correctness_thresholds"])
    trace_path = private_dir / "canonical_trace.json"
    evidence_path = private_dir / "canonical_equivalence.json"
    result_path = private_dir / "canonical_results" / f"{secrets.token_hex(12)}.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment_id = environment["isolation"]["environment_id"]
    virtual_environment = environment_path.parent / "venvs" / environment_id
    python = virtual_environment / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    command = [
        str(python),
        "-I",
        "-B",
        str(adapter_path),
        "--source-root",
        request["source"]["path"],
        "--entrypoint",
        request["reference"]["entrypoint"],
        "--checkpoint",
        request["checkpoint"]["path"],
        "--profiles",
        str(profiles_path),
        "--inventory",
        str(inventory_path),
        "--classification",
        str(classification_path),
        "--thresholds",
        str(thresholds_path),
        "--trace",
        str(trace_path),
        "--evidence",
        str(evidence_path),
        "--result",
        str(result_path),
        "--port-id",
        request["port_id"],
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=Path(request["source"]["path"]),
            env=offline_environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"Canonical Adapter exited {completed.returncode}: {detail}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError, RuntimeError) as error:
        issue = {
            "path": "canonicalization.environment",
            "message": f"could not execute Canonical Adapter: {error}",
        }
        return ENVIRONMENT_FAILURE, [issue], {}, {
            "cases": 0,
            "comparisons": 0,
            "failures": 1,
        }, []

    upstream = {
        "source_inventory": inventory_path,
        "capability_classification": classification_path,
    }
    artifacts = {
        "canonical_adapter": artifact_record(
            port_dir, adapter_path, request, environment_path, upstream
        )
    }
    execution_upstream = {**upstream, "canonical_adapter": adapter_path}
    if trace_path.is_file():
        artifacts["canonical_trace"] = artifact_record(
            port_dir, trace_path, request, environment_path, execution_upstream
        )
    if evidence_path.is_file():
        artifacts["canonical_equivalence"] = artifact_record(
            port_dir,
            evidence_path,
            request,
            environment_path,
            {**execution_upstream, "canonical_trace": trace_path},
        )
    try:
        current_source_sha256 = source_sha256(Path(request["source"]["path"]))
    except OSError as error:
        issue = {
            "path": "source.sha256",
            "message": f"could not verify source after canonicalization: {error}",
        }
        return ENVIRONMENT_FAILURE, [issue], artifacts, {
            "cases": 0,
            "comparisons": 0,
            "failures": 1,
        }, []
    if current_source_sha256 != request["source"]["sha256"]:
        issue = {
            "path": "source.sha256",
            "message": "trusted source changed while canonicalization was running",
        }
        return INVALID_INPUT, [issue], artifacts, {
            "cases": 0,
            "comparisons": 0,
            "failures": 1,
        }, []
    summary = result.get(
        "summary", {"cases": 0, "comparisons": 0, "failures": 1}
    )
    if result.get("status") == "success":
        return SUCCESS, [], artifacts, summary, []
    gaps = result.get("gaps")
    if not isinstance(gaps, list) or not gaps:
        gaps = [
            {
                "kind": "incomplete_canonicalization",
                "path": "canonicalization",
                "message": "Canonical Adapter did not produce a complete result",
            }
        ]
    issues = [
        {"path": gap.get("path", "canonicalization"), "message": gap["message"]}
        for gap in gaps
    ]
    return CORRECTNESS_FAILURE, issues, artifacts, summary, gaps


def run_port(args: argparse.Namespace) -> int:
    port_dir = args.port_dir.resolve()
    if port_dir_is_unsafe(port_dir):
        print("port directory must be outside source checkouts", file=sys.stderr)
        return 3
    request_path = port_dir / "request.json"
    try:
        request = json.loads(
            request_path.read_text(encoding="utf-8"),
            parse_constant=reject_non_json_number,
            parse_float=parse_finite_float,
        )
    except (json.JSONDecodeError, ValueError) as error:
        detail = error.msg if isinstance(error, json.JSONDecodeError) else str(error)
        issue = {"path": "request.json", "message": f"invalid JSON: {detail}"}
        write_json(port_dir / "report.json", failed_report({}, INVALID_INPUT, [issue]))
        print(issue["message"], file=sys.stderr)
        return INVALID_INPUT.code
    except OSError as error:
        issue = {"path": "request.json", "message": f"cannot read request: {error}"}
        write_json(port_dir / "report.json", failed_report({}, MISSING_INPUT, [issue]))
        print(issue["message"], file=sys.stderr)
        return MISSING_INPUT.code

    issues = schema_issues(request)
    if issues:
        safe_request = request if isinstance(request, dict) else {}
        report = failed_report(safe_request, INVALID_INPUT, issues)
        write_json(port_dir / "report.json", report)
        print(issues[0]["message"], file=sys.stderr)
        return INVALID_INPUT.code

    request = normalize_request(request)
    source_path = request.get("source", {}).get("path")
    source = Path(source_path) if isinstance(source_path, str) and source_path else None
    if port_dir_is_unsafe(port_dir, source):
        print("port directory must be outside source checkouts", file=sys.stderr)
        return INVALID_INPUT.code

    warnings = missing_fact_warnings(request)

    issues = unsupported_target_issues(request)
    if issues:
        report = failed_report(request, UNSUPPORTED_TARGET, issues)
        write_json(port_dir / "report.json", report)
        print(issues[0]["message"], file=sys.stderr)
        return UNSUPPORTED_TARGET.code

    family_pack = select_family_pack(request["model_family"])
    contract_path = (
        args.capability_contract.resolve()
        if args.capability_contract is not None
        else default_capability_contract(request["capability_contract_version"])
    )
    contract = None
    if request["reference"].get("entrypoint") is not None:
        try:
            contract = load_capability_contract(
                contract_path,
                request["capability_contract_version"],
                family_pack,
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            issue = {
                "path": "capability_contract",
                "message": f"could not evaluate capability contract: {error}",
            }
            report = failed_report(request, INVALID_INPUT, [issue])
            write_json(port_dir / "report.json", report)
            print(issue["message"], file=sys.stderr)
            return INVALID_INPUT.code

    try:
        missing_provenance, invalid_provenance = provenance_issues(request)
    except OSError as error:
        issues = [
            {
                "path": "provenance",
                "message": f"could not verify source or checkpoint: {error}",
            }
        ]
        report = failed_report(request, ENVIRONMENT_FAILURE, issues)
        write_json(port_dir / "report.json", report)
        print(issues[0]["message"], file=sys.stderr)
        return ENVIRONMENT_FAILURE.code
    if missing_provenance:
        report = failed_report(request, MISSING_INPUT, missing_provenance)
        write_json(port_dir / "report.json", report)
        print(missing_provenance[0]["message"], file=sys.stderr)
        return MISSING_INPUT.code
    if invalid_provenance:
        report = failed_report(request, INVALID_INPUT, invalid_provenance)
        write_json(port_dir / "report.json", report)
        print(invalid_provenance[0]["message"], file=sys.stderr)
        return INVALID_INPUT.code

    core = PortingCore(request, family_pack, environment_facts(), warnings)
    core.pass_stage("intake")
    core.finish(
        SUCCESS,
        "Intake passed with warnings" if warnings else "Intake passed",
    )
    report = core.report
    if request["reference"].get("entrypoint") is not None:
        outcome, inspection_issues, artifacts = run_reference_inspection(
            port_dir, request
        )
        core.add_artifacts(artifacts)
        report["reference_inspection"] = {
            "status": "passed" if outcome == SUCCESS else "failed",
            "adapter_contract_version": REFERENCE_ADAPTER_CONTRACT_VERSION,
        }
        if outcome == SUCCESS:
            report["exit"]["message"] = (
                "Intake and source inspection passed with warnings"
                if warnings
                else "Intake and source inspection passed"
            )
            inventory_path = port_dir / artifacts["source_inventory"]["path"]
            try:
                inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
                assert contract is not None
                classification = classify_capabilities(inventory, contract)
                classification_path = (
                    port_dir / "private" / "capability_classification.json"
                )
                previous_classification = (
                    json.loads(classification_path.read_text(encoding="utf-8"))
                    if classification_path.is_file()
                    else None
                )
                if previous_classification is not None:
                    previous_contract = previous_classification.get("contract", {})
                    if (
                        previous_contract.get("version")
                        == classification["contract"]["version"]
                        and previous_contract.get("sha256")
                        != classification["contract"]["sha256"]
                    ):
                        raise ValueError(
                            "pinned capability contract content changed "
                            "without a new version"
                        )
                    previous_fingerprints = previous_classification.get(
                        "dependency_fingerprints", {}
                    )
                else:
                    previous_fingerprints = {}
                current_fingerprints = classification["dependency_fingerprints"]
                classification["invalidated_capabilities"] = (
                    sorted(
                        capability
                        for capability in set(previous_fingerprints)
                        | set(current_fingerprints)
                        if previous_fingerprints.get(capability)
                        != current_fingerprints.get(capability)
                    )
                    if previous_classification is not None
                    else []
                )
            except (OSError, json.JSONDecodeError, ValueError) as error:
                issue = {
                    "path": "capability_contract",
                    "message": f"could not evaluate capability contract: {error}",
                }
                report["exit"] = {
                    "code": INVALID_INPUT.code,
                    "category": INVALID_INPUT.category,
                    "message": issue["message"],
                }
                report["stages"]["preflight"] = "failed"
                report["issues"] = [issue]
                write_json(port_dir / "report.json", report)
                print(issue["message"], file=sys.stderr)
                return INVALID_INPUT.code
            write_json(classification_path, classification)
            report["artifacts"]["capability_classification"] = artifact_record(
                port_dir,
                classification_path,
                request,
                port_dir / artifacts["reference_environment"]["path"],
                {
                    "source_inventory": inventory_path,
                    "capability_contract": contract_path,
                },
            )
            core.start_stage("preflight")
            report["capability_assessment"] = {
                "status": "passed",
                "contract_version": contract["contract_version"],
                **classification["summary"],
            }
            core.set_gate(
                "capability_contract",
                "passed",
                classification["summary"],
            )
            report["exit"]["message"] = "Preflight capability assessment passed"
            if classification["summary"]["unsupported"]:
                gaps = [
                    item
                    for item in classification["classifications"]
                    if item["classification"] == "unsupported"
                ]
                gap = {
                    "schema_version": "1.0",
                    "port_id": request["port_id"],
                    "category": UNSUPPORTED_SEMANTICS.category,
                    "contract": classification["contract"],
                    "source_inventory_sha256": classification[
                        "source_inventory_sha256"
                    ],
                    "gaps": gaps,
                }
                gap_path = port_dir / "private" / "capability_gap_report.json"
                write_json(gap_path, gap)
                report["artifacts"]["gap_report"] = artifact_record(
                    port_dir,
                    gap_path,
                    request,
                    port_dir / artifacts["reference_environment"]["path"],
                    {
                        "source_inventory": inventory_path,
                        "capability_contract": contract_path,
                        "capability_classification": classification_path,
                    },
                )
                report["stages"]["preflight"] = "blocked"
                core.set_gate("capability_contract", "blocked", classification["summary"])
                report["capability_assessment"]["status"] = "blocked"
                report["exit"] = {
                    "code": UNSUPPORTED_SEMANTICS.code,
                    "category": UNSUPPORTED_SEMANTICS.category,
                    "message": "source semantics fall outside the Capability Contract",
                }
                report["issues"] = [
                    {"path": item["path"], "message": item["reason"]}
                    for item in gaps
                ]
                write_json(port_dir / "report.json", report)
                print(report["exit"]["message"], file=sys.stderr)
                return UNSUPPORTED_SEMANTICS.code
            distinct_profiles = {
                value_sha256(profile)
                for profile in request["representative_profiles"]
            }
            thresholds = request["correctness_thresholds"]
            if len(distinct_profiles) < 2:
                issue = {
                    "path": "representative_profiles",
                    "message": (
                        "canonical equivalence requires at least two distinct "
                        "representative input profiles"
                    ),
                }
            elif not all(
                is_number(thresholds.get(name))
                for name in ("absolute", "relative")
            ):
                issue = {
                    "path": "correctness_thresholds",
                    "message": (
                        "canonical equivalence requires absolute and relative "
                        "correctness thresholds"
                    ),
                }
            else:
                issue = None
            capture_path = port_dir / artifacts["private_capture"]["path"]
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
            if issue is None:
                issue = canonical_input_issue(inventory, capture)
            if issue is not None:
                gap = {
                    "schema_version": "1.0",
                    "port_id": request["port_id"],
                    "category": CORRECTNESS_FAILURE.category,
                    "contract": classification["contract"],
                    "source_inventory_sha256": classification[
                        "source_inventory_sha256"
                    ],
                    "gaps": [
                        {
                            "kind": "incomplete_canonicalization",
                            **issue,
                        }
                    ],
                }
                gap_path = port_dir / "private" / "canonicalization_gap_report.json"
                write_json(gap_path, gap)
                report["artifacts"]["gap_report"] = artifact_record(
                    port_dir,
                    gap_path,
                    request,
                    port_dir / artifacts["reference_environment"]["path"],
                    {
                        "source_inventory": inventory_path,
                        "capability_classification": classification_path,
                    },
                )
                report["canonicalization"] = {
                    "status": "failed",
                    "mode": (
                        "canonicalized"
                        if classification["summary"]["canonicalizable"]
                        else "direct"
                    ),
                    "cases": 0,
                    "comparisons": 0,
                    "failures": 1,
                }
                core.set_gate("canonical_equivalence", "blocked", {"failures": 1})
                report["stages"]["preflight"] = "blocked"
                report["exit"] = {
                    "code": CORRECTNESS_FAILURE.code,
                    "category": CORRECTNESS_FAILURE.category,
                    "message": issue["message"],
                }
                report["issues"] = [issue]
                write_json(port_dir / "report.json", report)
                print(issue["message"], file=sys.stderr)
                return CORRECTNESS_FAILURE.code
            if not classification["summary"]["canonicalizable"]:
                summary, canonical_artifacts = write_direct_canonical_evidence(
                    port_dir,
                    request,
                    inventory,
                    capture,
                    classification,
                    port_dir / artifacts["reference_environment"]["path"],
                    {
                        "source_inventory": inventory_path,
                        "reference_capture": capture_path,
                        "capability_classification": classification_path,
                    },
                )
                report["artifacts"].update(canonical_artifacts)
                report["canonicalization"] = {
                    "status": "passed",
                    "mode": "direct",
                    **summary,
                }
                core.set_gate("canonical_equivalence", "passed", summary)
                report["exit"]["message"] = "Preflight canonical trace passed"
            else:
                environment_path = (
                    port_dir / artifacts["reference_environment"]["path"]
                )
                (
                    canonical_outcome,
                    canonical_issues,
                    canonical_artifacts,
                    summary,
                    canonical_gaps,
                ) = run_canonical_verification(
                    port_dir,
                    request,
                    inventory_path,
                    classification_path,
                    environment_path,
                )
                report["artifacts"].update(canonical_artifacts)
                report["canonicalization"] = {
                    "status": (
                        "passed" if canonical_outcome == SUCCESS else "failed"
                    ),
                    "mode": "canonicalized",
                    **summary,
                }
                if canonical_outcome != SUCCESS:
                    core.set_gate("canonical_equivalence", "blocked", summary)
                    if canonical_gaps:
                        gap = {
                            "schema_version": "1.0",
                            "port_id": request["port_id"],
                            "category": CORRECTNESS_FAILURE.category,
                            "contract": classification["contract"],
                            "source_inventory_sha256": classification[
                                "source_inventory_sha256"
                            ],
                            "gaps": canonical_gaps,
                        }
                        gap_path = (
                            port_dir
                            / "private"
                            / "canonicalization_gap_report.json"
                        )
                        write_json(gap_path, gap)
                        gap_upstream = {
                            "source_inventory": inventory_path,
                            "capability_classification": classification_path,
                            **{
                                name: port_dir / artifact["path"]
                                for name, artifact in canonical_artifacts.items()
                            },
                        }
                        report["artifacts"]["gap_report"] = artifact_record(
                            port_dir,
                            gap_path,
                            request,
                            environment_path,
                            gap_upstream,
                        )
                    report["stages"]["preflight"] = (
                        "blocked"
                        if canonical_outcome == CORRECTNESS_FAILURE
                        else "failed"
                    )
                    report["exit"] = {
                        "code": canonical_outcome.code,
                        "category": canonical_outcome.category,
                        "message": canonical_issues[0]["message"],
                    }
                    report["issues"] = canonical_issues
                    write_json(port_dir / "report.json", report)
                    print(report["exit"]["message"], file=sys.stderr)
                    return canonical_outcome.code
                core.set_gate("canonical_equivalence", "passed", summary)
                report["exit"]["message"] = "Preflight canonical equivalence passed"
            trace_path = port_dir / "private" / "canonical_trace.json"
            try:
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                catalog_document = json.loads(
                    args.kernel_capabilities.read_text(encoding="utf-8")
                )
                coverage = analyze_kernel_coverage(
                    trace,
                    catalog_document["capabilities"],
                    request["requested_targets"],
                )
            except (OSError, json.JSONDecodeError, KeyError, KernelCoverageError) as error:
                issue = {
                    "path": "kernel_coverage",
                    "message": f"could not classify canonical computations: {error}",
                }
                core.set_gate("kernel_coverage", "blocked", {"unclassified": 1})
                report["stages"]["preflight"] = "blocked"
                core.finish(CORRECTNESS_FAILURE, issue["message"])
                report["issues"] = [issue]
                write_json(port_dir / "report.json", report)
                print(issue["message"], file=sys.stderr)
                return CORRECTNESS_FAILURE.code
            coverage_path = port_dir / "private" / "kernel_coverage.json"
            write_json(coverage_path, coverage)
            report["artifacts"]["kernel_coverage"] = artifact_record(
                port_dir,
                coverage_path,
                request,
                port_dir / artifacts["reference_environment"]["path"],
                {
                    "canonical_trace": trace_path,
                    "kernel_capabilities": args.kernel_capabilities,
                },
            )
            if coverage["status"] == "blocked":
                if coverage["kernel_gaps"]:
                    handoff = {
                        "schema_version": "1.0",
                        "port_id": request["port_id"],
                        "family": request["model_family"],
                        "requirements": coverage["kernel_gaps"],
                    }
                    handoff_path = port_dir / "private" / "kernel_gap_handoff.json"
                    write_json(handoff_path, handoff)
                    report["artifacts"]["kernel_gap_handoff"] = artifact_record(
                        port_dir,
                        handoff_path,
                        request,
                        port_dir / artifacts["reference_environment"]["path"],
                        {
                            "canonical_trace": trace_path,
                            "kernel_coverage": coverage_path,
                        },
                    )
                core.set_gate(
                    "kernel_coverage",
                    "blocked",
                    {"kernel_gaps": len(coverage["kernel_gaps"])},
                )
                report["stages"]["preflight"] = "blocked"
                if coverage["kernel_gaps"]:
                    core.finish(KERNEL_GAP, "required kernel capability is missing")
                else:
                    core.finish(
                        UNSUPPORTED_SEMANTICS,
                        "canonical computation is explicitly unsupported",
                    )
                write_json(port_dir / "report.json", report)
                print(report["exit"]["message"], file=sys.stderr)
                return report["exit"]["code"]
            core.set_gate(
                "kernel_coverage",
                "passed",
                {
                    "computations": len(coverage["classifications"]),
                    "optimization_opportunities": len(
                        coverage["optimization_opportunities"]
                    ),
                },
            )
            core.pass_stage("preflight")
            core.finish(SUCCESS, "Preflight passed")
        else:
            report["exit"] = {
                "code": outcome.code,
                "category": outcome.category,
                "message": inspection_issues[0]["message"],
            }
            report["issues"] = inspection_issues
            write_json(port_dir / "report.json", report)
            print(inspection_issues[0]["message"], file=sys.stderr)
            return outcome.code
    write_json(port_dir / "report.json", report)
    print(port_dir / "report.json")
    return SUCCESS.code


def show_report(args: argparse.Namespace) -> int:
    report_path = args.port_dir.resolve() / "report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"cannot read report: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


def bundle_port(args: argparse.Namespace) -> int:
    try:
        result = create_bundle(args.port_dir, args.output)
    except (BundleError, OSError) as error:
        print(f"cannot create Portable Run Bundle: {error}", file=sys.stderr)
        return INVALID_INPUT.code
    print(result)
    return SUCCESS.code


def merge_port_bundle(args: argparse.Namespace) -> int:
    try:
        result = merge_bundle(args.port_dir, args.bundle)
    except (BundleError, OSError) as error:
        print(f"cannot merge Portable Run Bundle: {error}", file=sys.stderr)
        return INVALID_INPUT.code
    print(result)
    return SUCCESS.code


def prepare_port_publication(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(args.publication.read_text(encoding="utf-8"))
        report_path = args.port_dir.resolve() / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        declared_family = report.get("request_declarations", {}).get("model_family")
        if payload.get("port_id") != report.get("port_id"):
            raise PublicationError("publication port_id does not match the Port report")
        if payload.get("family") != declared_family:
            raise PublicationError("publication family does not match the Port report")
        result = prepare_publication(
            args.repository, args.base_commit, payload, args.output
        )
        report["refactor_assessment"] = payload["refactor_assessment"]
        write_json(report_path, report)
    except (PublicationError, OSError, json.JSONDecodeError) as error:
        print(f"cannot prepare Port publication: {error}", file=sys.stderr)
        return INVALID_INPUT.code
    print(result)
    return SUCCESS.code


def cleanup_port(args: argparse.Namespace) -> int:
    port_dir = args.port_dir.resolve()
    if not port_dir.is_dir():
        print(f"Port directory does not exist: {port_dir}", file=sys.stderr)
        return MISSING_INPUT.code
    if not args.confirm:
        print(f"retained {port_dir}; cleanup is explicit and no files were removed")
        return SUCCESS.code
    request_path = port_dir / "request.json"
    if port_dir_is_unsafe(port_dir) or not request_path.is_file():
        print("refusing to remove a directory that is not a safe Port", file=sys.stderr)
        return INVALID_INPUT.code
    shutil.rmtree(port_dir)
    print(
        f"removed private Port directory {port_dir}; "
        "recovery requires a prior backup"
    )
    return SUCCESS.code


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="apxinf-port")
    subcommands = command.add_subparsers(dest="command", required=True)

    init = subcommands.add_parser("init", help="create a private Port request draft")
    init.add_argument("--family", required=True)
    init.add_argument("--source", type=Path)
    init.add_argument("--source-revision")
    init.add_argument("--checkpoint", type=Path)
    init.add_argument("--reference-entrypoint")
    init.add_argument("--dependency-lock")
    init.add_argument("--port-dir", type=Path, required=True)
    init.set_defaults(handler=initialize)

    run = subcommands.add_parser(
        "run", help="validate a request and run Intake and Preflight"
    )
    run.add_argument("--port-dir", type=Path, required=True)
    run.add_argument("--capability-contract", type=Path)
    run.add_argument(
        "--kernel-capabilities", type=Path, default=default_kernel_capabilities()
    )
    run.set_defaults(handler=run_port)

    resume = subcommands.add_parser(
        "resume", help="reconcile saved evidence and recover interrupted stages"
    )
    resume.add_argument("--port-dir", type=Path, required=True)
    resume.set_defaults(handler=resume_port)

    report = subcommands.add_parser("report", help="print the structured Port report")
    report.add_argument("--port-dir", type=Path, required=True)
    report.set_defaults(handler=show_report)

    bundle = subcommands.add_parser("bundle", help="create a local Portable Run Bundle")
    bundle.add_argument("--port-dir", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    bundle.set_defaults(handler=bundle_port)

    merge = subcommands.add_parser(
        "merge-bundle", help="validate and merge a Portable Run Bundle"
    )
    merge.add_argument("--port-dir", type=Path, required=True)
    merge.add_argument("--bundle", type=Path, required=True)
    merge.set_defaults(handler=merge_port_bundle)

    publication = subcommands.add_parser(
        "prepare-publication",
        help="validate and prepare local PR, support, and refactor artifacts",
    )
    publication.add_argument("--port-dir", type=Path, required=True)
    publication.add_argument("--repository", type=Path, required=True)
    publication.add_argument("--base-commit", required=True)
    publication.add_argument("--publication", type=Path, required=True)
    publication.add_argument("--output", type=Path, required=True)
    publication.set_defaults(handler=prepare_port_publication)

    cleanup = subcommands.add_parser(
        "cleanup", help="show the explicit retention policy for a Port"
    )
    cleanup.add_argument("--port-dir", type=Path, required=True)
    cleanup.add_argument(
        "--confirm",
        action="store_true",
        help="irreversibly remove the complete private Port",
    )
    cleanup.set_defaults(handler=cleanup_port)
    return command


def main() -> int:
    args = parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
