#!/usr/bin/env python3
"""Initialize Intake and inspect trusted source code for an ApxInf VLA Port."""

from __future__ import annotations

import argparse
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


REQUEST_SCHEMA_VERSION = "1.0"
REPORT_SCHEMA_VERSION = "1.0"
REFERENCE_ADAPTER_CONTRACT_VERSION = "1.0"
SUPPORTED_TUPLES = (
    ("thor", "bf16"),
    ("thor", "fp8"),
    ("orin", "bf16"),
    ("orin", "int8_w8a8"),
)
TARGETS = {"thor", "orin"}
PRECISIONS = {"bf16", "fp8", "int8_w8a8"}


@dataclass(frozen=True)
class IntakeOutcome:
    code: int
    category: str


SUCCESS = IntakeOutcome(0, "success")
MISSING_INPUT = IntakeOutcome(2, "missing_input")
INVALID_INPUT = IntakeOutcome(3, "invalid_input")
UNSUPPORTED_TARGET = IntakeOutcome(4, "unsupported_target")
ENVIRONMENT_FAILURE = IntakeOutcome(5, "environment_failure")
REFERENCE_LOAD_FAILURE = IntakeOutcome(6, "reference_load_failure")
REFERENCE_TRACE_FAILURE = IntakeOutcome(7, "reference_trace_failure")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def reference_adapter_template() -> Path:
    return repository_root() / "scripts" / "reference_adapter_template.py"


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def initialize(args: argparse.Namespace) -> int:
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


def tuple_states(requested: Any) -> list[dict[str, str]]:
    if not isinstance(requested, list):
        requested = []
    selected = {
        (item.get("target"), item.get("precision"))
        for item in requested
        if isinstance(item, dict)
        and isinstance(item.get("target"), str)
        and isinstance(item.get("precision"), str)
    }
    return [
        {
            "target": target,
            "precision": precision,
            "status": "requested" if (target, precision) in selected else "not_requested",
        }
        for target, precision in SUPPORTED_TUPLES
    ]


def successful_report(
    request: dict[str, Any], warnings: list[dict[str, str]]
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "port_id": request["port_id"],
        "request_schema_version": request["schema_version"],
        "stages": {
            "intake": "passed",
            "preflight": "not_started",
        },
        "exit": {
            "code": SUCCESS.code,
            "category": SUCCESS.category,
            "message": "Intake passed with warnings" if warnings else "Intake passed",
        },
        "request_declarations": {
            "source": request["source"],
            "checkpoint": request["checkpoint"],
            "reference": request["reference"],
            "representative_profiles": request["representative_profiles"],
            "requested_targets": request["requested_targets"],
            "correctness_thresholds": request["correctness_thresholds"],
            "tuning_budgets": request["tuning_budgets"],
            "environment": request.get("user_environment_declarations", {}),
        },
        "observed_environment": environment_facts(),
        "target_precisions": tuple_states(request["requested_targets"]),
        "issues": [],
        "warnings": warnings,
        "artifacts": {},
    }


def failed_report(
    request: dict[str, Any], outcome: IntakeOutcome, issues: list[dict[str, str]]
) -> dict[str, Any]:
    port_id = request.get("port_id")
    request_schema_version = request.get("schema_version")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "port_id": port_id if isinstance(port_id, str) else None,
        "request_schema_version": (
            request_schema_version if isinstance(request_schema_version, str) else None
        ),
        "stages": {
            "intake": "failed",
            "preflight": "not_started",
        },
        "exit": {
            "code": outcome.code,
            "category": outcome.category,
            "message": issues[0]["message"] if issues else "Intake failed",
        },
        "request_declarations": json_safe(
            {
                "source": request.get("source"),
                "checkpoint": request.get("checkpoint"),
                "reference": request.get("reference"),
                "representative_profiles": request.get("representative_profiles"),
                "requested_targets": request.get("requested_targets"),
                "correctness_thresholds": request.get("correctness_thresholds"),
                "tuning_budgets": request.get("tuning_budgets"),
                "environment": request.get("user_environment_declarations"),
            }
        ),
        "observed_environment": environment_facts(),
        "target_precisions": tuple_states(request.get("requested_targets", [])),
        "issues": issues,
        "warnings": [],
        "artifacts": {},
    }


def unsupported_target_issues(request: dict[str, Any]) -> list[dict[str, str]]:
    issues = []
    for index, item in enumerate(request.get("requested_targets", [])):
        pair = (item.get("target"), item.get("precision"))
        if None in pair:
            continue
        if pair not in SUPPORTED_TUPLES:
            if pair == ("orin", "fp8"):
                message = "orin does not support fp8; use bf16 or int8_w8a8"
            else:
                message = f"unsupported target/precision tuple: {pair[0]}/{pair[1]}"
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
        "source",
        "checkpoint",
        "reference",
        "representative_profiles",
        "requested_targets",
        "correctness_thresholds",
        "tuning_budgets",
        "user_environment_declarations",
    }
    required_fields = {"schema_version", "port_id"}
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
) -> dict[str, Any]:
    return {
        "path": path.relative_to(port_dir).as_posix(),
        "status": "fresh",
        "fingerprints": {
            "content_sha256": file_sha256(path),
            "tool_sha256": file_sha256(Path(__file__).resolve()),
            "source_sha256": request["source"]["sha256"],
            "checkpoint_sha256": request["checkpoint"]["sha256"],
            "environment_sha256": file_sha256(environment_path),
            "upstream_sha256": {
                "request": file_sha256(port_dir / "request.json"),
            },
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
    virtual_environment = environment_dir / "venv"
    environment_path = environment_dir / "environment.json"
    try:
        if not virtual_environment.is_dir():
            venv.EnvBuilder(with_pip=True).create(virtual_environment)
        python = virtual_environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        lock_text = lock_path.read_text(encoding="utf-8")
        if "http://" in lock_text.lower() or "https://" in lock_text.lower():
            raise RuntimeError("dependency lock contains a network URL")
        requirements = [
            line
            for line in lock_text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
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
        environment = {
            "schema_version": "1.0",
            "python": platform.python_version(),
            "dependency_lock": {
                "path": dependency_lock,
                "sha256": file_sha256(lock_path),
            },
            "isolation": {
                "kind": "venv",
                "system_site_packages": False,
            },
            "runtime_network_access": False,
            "network_enforcement": ["offline_environment", "python_socket_guard"],
        }
        write_json(environment_path, environment)
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
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


def run_intake(args: argparse.Namespace) -> int:
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

    report = successful_report(request, warnings)
    if request["reference"].get("entrypoint") is not None:
        outcome, inspection_issues, artifacts = run_reference_inspection(
            port_dir, request
        )
        report["artifacts"] = artifacts
        if outcome == SUCCESS:
            report["stages"]["preflight"] = "passed"
            report["exit"]["message"] = (
                "Intake and source inspection passed with warnings"
                if warnings
                else "Intake and source inspection passed"
            )
        else:
            report["stages"]["preflight"] = "failed"
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


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="apxinf-port")
    subcommands = command.add_subparsers(dest="command", required=True)

    init = subcommands.add_parser("init", help="create a private Port request draft")
    init.add_argument("--source", type=Path)
    init.add_argument("--source-revision")
    init.add_argument("--checkpoint", type=Path)
    init.add_argument("--reference-entrypoint")
    init.add_argument("--dependency-lock")
    init.add_argument("--port-dir", type=Path, required=True)
    init.set_defaults(handler=initialize)

    run = subcommands.add_parser("run", help="validate a request and run Intake")
    run.add_argument("--port-dir", type=Path, required=True)
    run.set_defaults(handler=run_intake)

    report = subcommands.add_parser("report", help="print the structured Port report")
    report.add_argument("--port-dir", type=Path, required=True)
    report.set_defaults(handler=show_report)
    return command


def main() -> int:
    args = parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
