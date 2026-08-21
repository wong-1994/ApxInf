#!/usr/bin/env python3
"""Initialize and run the read-only Intake stage of an ApxInf VLA Port."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUEST_SCHEMA_VERSION = "1.0"
REPORT_SCHEMA_VERSION = "1.0"
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


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def initialize(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    checkpoint = args.checkpoint.resolve()
    port_dir = args.port_dir.resolve()
    if not source.is_dir():
        print(f"source directory does not exist: {source}", file=sys.stderr)
        return 2
    if not checkpoint.is_file():
        print(f"checkpoint does not exist: {checkpoint}", file=sys.stderr)
        return 2
    if port_dir_is_unsafe(port_dir, source):
        print("port directory must be outside source checkouts", file=sys.stderr)
        return 3

    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "port_id": f"{source.name}-{args.source_revision[:8]}",
        "source": {
            "path": str(source),
            "revision": args.source_revision,
            "sha256": source_sha256(source),
        },
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
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


def successful_report(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "port_id": request["port_id"],
        "request_schema_version": request["schema_version"],
        "stages": {"intake": "passed", "preflight": "not_started"},
        "exit": {
            "code": SUCCESS.code,
            "category": SUCCESS.category,
            "message": "Intake passed",
        },
        "request_declarations": {
            "source": request["source"],
            "checkpoint": request["checkpoint"],
            "representative_profiles": request["representative_profiles"],
            "requested_targets": request["requested_targets"],
            "correctness_thresholds": request["correctness_thresholds"],
            "tuning_budgets": request["tuning_budgets"],
            "environment": request.get("user_environment_declarations", {}),
        },
        "observed_environment": environment_facts(),
        "target_precisions": tuple_states(request["requested_targets"]),
        "issues": [],
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
        "stages": {"intake": "failed", "preflight": "not_started"},
        "exit": {
            "code": outcome.code,
            "category": outcome.category,
            "message": issues[0]["message"] if issues else "Intake failed",
        },
        "request_declarations": json_safe(
            {
                "source": request.get("source"),
                "checkpoint": request.get("checkpoint"),
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
    }


def unsupported_target_issues(request: dict[str, Any]) -> list[dict[str, str]]:
    issues = []
    for index, item in enumerate(request.get("requested_targets", [])):
        pair = (item.get("target"), item.get("precision"))
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
    required_fields = {
        "schema_version",
        "port_id",
        "source",
        "checkpoint",
        "representative_profiles",
        "requested_targets",
        "correctness_thresholds",
        "tuning_budgets",
        "user_environment_declarations",
    }
    for field in sorted(required_fields - request.keys()):
        issues.append({"path": field, "message": "field is required by the request schema"})
    add_unknown_field_issues(
        issues,
        request,
        required_fields,
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
    for field in ("source", "checkpoint", "correctness_thresholds"):
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
            if not isinstance(goal, dict):
                issues.append({"path": f"{path}.latency_goal", "message": "must be an object"})
            else:
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


def missing_input_issues(request: dict[str, Any]) -> list[dict[str, str]]:
    issues = []
    required_values = (
        ("port_id", request.get("port_id")),
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
            issues.append({"path": path, "message": "required fact is missing"})

    profiles = request.get("representative_profiles", [])
    if not profiles:
        issues.append({"path": "representative_profiles", "message": "required fact is missing"})
    for index, profile in enumerate(profiles):
        for field in ("name", "inputs"):
            value = profile.get(field)
            if value is None or value == "" or value == {}:
                issues.append(
                    {
                        "path": f"representative_profiles[{index}].{field}",
                        "message": "required fact is missing",
                    }
                )

    requested = request.get("requested_targets", [])
    if not requested:
        issues.append({"path": "requested_targets", "message": "required fact is missing"})
    for index, item in enumerate(requested):
        for field in ("target", "precision"):
            if not item.get(field):
                issues.append(
                    {
                        "path": f"requested_targets[{index}].{field}",
                        "message": "required fact is missing",
                    }
                )
        goal = item.get("latency_goal", {})
        for field in ("p50_ms", "p95_ms"):
            if goal.get(field) is None:
                issues.append(
                    {
                        "path": f"requested_targets[{index}].latency_goal.{field}",
                        "message": "required fact is missing",
                    }
                )

    budgets = request.get("tuning_budgets", [])
    if not budgets:
        issues.append({"path": "tuning_budgets", "message": "required fact is missing"})
    for index, budget in enumerate(budgets):
        for field in ("target", "seconds"):
            if budget.get(field) is None:
                issues.append(
                    {
                        "path": f"tuning_budgets[{index}].{field}",
                        "message": "required fact is missing",
                    }
                )
    requested_targets = {
        item.get("target") for item in requested if item.get("target") is not None
    }
    budget_targets = {
        budget.get("target") for budget in budgets if budget.get("target") is not None
    }
    for target in sorted(requested_targets - budget_targets):
        issues.append(
            {
                "path": f"tuning_budgets[{target}]",
                "message": f"a tuning budget is required for requested target {target}",
            }
        )
    return issues


def provenance_issues(
    request: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    missing = []
    invalid = []
    source = Path(request["source"]["path"])
    checkpoint = Path(request["checkpoint"]["path"])
    if not source.is_dir():
        missing.append(
            {"path": "source.path", "message": "source directory no longer exists"}
        )
    elif source_sha256(source) != request["source"]["sha256"]:
        invalid.append(
            {
                "path": "source.sha256",
                "message": "source content no longer matches its pinned SHA-256",
            }
        )
    if not checkpoint.is_file():
        missing.append(
            {"path": "checkpoint.path", "message": "checkpoint no longer exists"}
        )
    elif file_sha256(checkpoint) != request["checkpoint"]["sha256"]:
        invalid.append(
            {
                "path": "checkpoint.sha256",
                "message": "checkpoint content no longer matches its pinned SHA-256",
            }
        )
    return missing, invalid


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

    source_path = request.get("source", {}).get("path")
    source = Path(source_path) if isinstance(source_path, str) and source_path else None
    if port_dir_is_unsafe(port_dir, source):
        print("port directory must be outside source checkouts", file=sys.stderr)
        return 3

    issues = missing_input_issues(request)
    if issues:
        report = failed_report(request, MISSING_INPUT, issues)
        write_json(port_dir / "report.json", report)
        print(issues[0]["message"], file=sys.stderr)
        return MISSING_INPUT.code

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

    report = successful_report(request)
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
    init.add_argument("--source", type=Path, required=True)
    init.add_argument("--source-revision", required=True)
    init.add_argument("--checkpoint", type=Path, required=True)
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
