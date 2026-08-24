"""Family-neutral export and budgeted tuning of physical GEMM workloads."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping


WORKLOAD_SCHEMA = "apxinf.tuning.gemm-workloads.v1"
TACTIC_SCHEMA = "apxinf.cuda.tuning.v1"
LOGICAL_PHASES = frozenset({"vision", "prefill", "decode", "action"})
DTYPES = frozenset({"f32", "f16", "bf16", "f8e4m3", "i8", "i32"})
OPS = frozenset({"bf16", "w8a8", "fp8_f16"})
LAYOUTS = frozenset({"row_major", "weight_output_major"})
SCALE_MODES = frozenset(
    {"none", "per_tensor", "dynamic_row_per_output_channel"}
)
EPILOGUES = frozenset({"none", "bias", "bias_gelu", "bias_residual"})


def _required(mapping: Mapping[str, Any], field: str, kind: type) -> Any:
    value = mapping.get(field)
    if not isinstance(value, kind) or isinstance(value, bool):
        raise ValueError(f"{field} must be a {kind.__name__}")
    if kind is str and not value:
        raise ValueError(f"{field} must not be empty")
    return value


def _positive_integer(mapping: Mapping[str, Any], field: str) -> int:
    value = _required(mapping, field, int)
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _nonnegative_number(mapping: Mapping[str, Any], field: str) -> float:
    value = mapping.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative number")
    return float(value)


def _target_fingerprint(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("target_fingerprint must be an object")
    result = {
        "device_name": _required(value, "device_name", str),
        "sm": _positive_integer(value, "sm"),
        "multiprocessor_count": _positive_integer(
            value, "multiprocessor_count"
        ),
        "kernel_build_id": _required(value, "kernel_build_id", str),
        "cuda_version": _required(value, "cuda_version", str),
    }
    libraries = value.get("library_versions")
    if not isinstance(libraries, Mapping) or not libraries:
        raise ValueError("library_versions must be a non-empty object")
    if "cublas" not in libraries:
        raise ValueError("library_versions must include cublas")
    if any(
        not isinstance(key, str)
        or not isinstance(version, str)
        or not version
        for key, version in libraries.items()
    ):
        raise ValueError("library_versions must contain non-empty string versions")
    result["library_versions"] = dict(sorted(libraries.items()))
    return result


@dataclass(frozen=True)
class WorkloadManifest:
    """Validated v1 manifest containing only physical GEMM workloads."""

    family: str
    profile: str
    target_fingerprint: dict[str, Any]
    workloads: tuple[dict[str, Any], ...]

    @classmethod
    def from_execution_plan(cls, plan: Mapping[str, Any]) -> "WorkloadManifest":
        """Normalize a Family Pack execution plan without knowing its config."""
        if not isinstance(plan, Mapping):
            raise ValueError("execution plan must be an object")
        if plan.get("tunable_object", "gemm") != "gemm":
            raise ValueError("v1 supports only GEMM workloads")
        family = _required(plan, "family", str)
        profile = _required(plan, "profile", str)
        target = _target_fingerprint(plan.get("target_fingerprint"))
        operations = plan.get("operations")
        if not isinstance(operations, list):
            raise ValueError("operations must be an array")
        workloads = tuple(
            _normalize_operation(operation, family, profile, target)
            for operation in operations
        )
        return cls(family, profile, target, workloads)

    @classmethod
    def from_dict(cls, manifest: Mapping[str, Any]) -> "WorkloadManifest":
        if not isinstance(manifest, Mapping):
            raise ValueError("workload manifest must be an object")
        if manifest.get("schema") != WORKLOAD_SCHEMA:
            raise ValueError(f"workload manifest schema must be {WORKLOAD_SCHEMA}")
        if manifest.get("tunable_object") != "gemm":
            raise ValueError("v1 supports only GEMM workloads")
        workloads = manifest.get("workloads")
        if not isinstance(workloads, list):
            raise ValueError("workloads must be an array")
        if not workloads:
            raise ValueError("workloads must not be empty")
        first = workloads[0]
        if not isinstance(first, Mapping):
            raise ValueError("each workload must be an object")
        family = _required(first, "family", str)
        profile = _required(first, "profile", str)
        target = _target_fingerprint(first.get("target_fingerprint"))
        operations = []
        for workload in workloads:
            if not isinstance(workload, Mapping):
                raise ValueError("each workload must be an object")
            if (
                workload.get("family") != family
                or workload.get("profile") != profile
            ):
                raise ValueError("one manifest must contain one family and profile")
            if _target_fingerprint(workload.get("target_fingerprint")) != target:
                raise ValueError("one manifest must contain one target fingerprint")
            operations.append(workload)
        return cls.from_execution_plan(
            {
                "family": family,
                "profile": profile,
                "target_fingerprint": target,
                "operations": operations,
            }
        )

    @classmethod
    def from_json_file(cls, path: Any) -> "WorkloadManifest":
        import json
        from pathlib import Path

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": WORKLOAD_SCHEMA,
            "tunable_object": "gemm",
            "workloads": deepcopy(list(self.workloads)),
        }


def _normalize_operation(
    operation: Any,
    family: str,
    profile: str,
    target: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(operation, Mapping):
        raise ValueError("each operation must be an object")
    phase = _required(operation, "logical_phase", str)
    if phase not in LOGICAL_PHASES:
        raise ValueError(f"logical_phase must be one of {sorted(LOGICAL_PHASES)}")
    op = _required(operation, "op", str)
    if op not in OPS:
        raise ValueError(f"unsupported GEMM op {op!r}")
    result = {
        "op": op,
        "m": _positive_integer(operation, "m"),
        "n": _positive_integer(operation, "n"),
        "k": _positive_integer(operation, "k"),
    }
    for field, choices in (
        ("activation_dtype", DTYPES),
        ("weight_dtype", DTYPES),
        ("output_dtype", DTYPES),
        ("layout", LAYOUTS),
        ("scale_mode", SCALE_MODES),
        ("epilogue", EPILOGUES),
    ):
        value = _required(operation, field, str)
        if value not in choices:
            raise ValueError(f"unsupported {field} {value!r}")
        result[field] = value
    result.update(
        {
            "workspace_limit": int(
                _nonnegative_number(operation, "workspace_limit")
            ),
            "family": family,
            "logical_phase": phase,
            "source_operation": _required(operation, "source_operation", str),
            "profile": profile,
            "repetitions": _positive_integer(operation, "repetitions"),
            "target_fingerprint": deepcopy(target),
            "estimated_milliseconds_saved": _nonnegative_number(
                operation, "estimated_milliseconds_saved"
            ),
            "best_current_milliseconds": _nonnegative_number(
                operation, "best_current_milliseconds"
            ),
        }
    )
    return result


def export_gemm_workloads(plan: Mapping[str, Any]) -> dict[str, Any]:
    return WorkloadManifest.from_execution_plan(plan).to_dict()


class GenericGemmTuner:
    """Budget coordinator which only consumes physical workload manifests."""

    def __init__(self, benchmark: Callable[[dict[str, Any]], Mapping[str, Any]]):
        self._benchmark = benchmark

    def tune(
        self,
        manifest: Mapping[str, Any],
        budgets: Mapping[tuple[str, str], float],
        install_and_verify: Callable[[dict[str, Any], str], bool],
    ) -> dict[str, Any]:
        parsed = WorkloadManifest.from_dict(manifest)
        budget_key = (parsed.target_fingerprint["device_name"], parsed.profile)
        budget = budgets.get(budget_key)
        if (
            not isinstance(budget, (int, float))
            or isinstance(budget, bool)
            or budget < 0
        ):
            raise ValueError(f"missing non-negative tuning budget for {budget_key!r}")
        ranked = sorted(
            parsed.workloads,
            key=lambda item: (
                -item["estimated_milliseconds_saved"] * item["repetitions"],
                item["source_operation"],
            ),
        )
        records = []
        measurements_by_operation = {}
        elapsed = 0.0
        observed_cost = 0.0
        for workload in ranked:
            if elapsed >= budget or (
                observed_cost and elapsed + observed_cost > budget
            ):
                break
            measurement = self._benchmark(deepcopy(workload))
            spent = _nonnegative_number(measurement, "seconds_spent")
            milliseconds = _nonnegative_number(measurement, "milliseconds")
            tactic = measurement.get("tactic")
            if not isinstance(tactic, Mapping):
                raise ValueError("benchmark tactic must be an object")
            backend = _required(tactic, "backend", str)
            tactic_id = _required(tactic, "id", int)
            elapsed += spent
            observed_cost = max(observed_cost, spent)
            record = _tactic_record(workload, backend, tactic_id, milliseconds)
            records.append(record)
            measurements_by_operation[workload["source_operation"]] = milliseconds
        target = parsed.target_fingerprint
        tactics = {
            "schema": TACTIC_SCHEMA,
            "kernel_build_id": target["kernel_build_id"],
            "device_name": target["device_name"],
            "sm": target["sm"],
            "multiprocessor_count": target["multiprocessor_count"],
            "cuda_version": target["cuda_version"],
            "cublas_version": target["library_versions"]["cublas"],
            "library_versions": deepcopy(target["library_versions"]),
            "records": records,
        }
        self.validate_tactics(tactics, parsed.target_fingerprint)
        if not install_and_verify(deepcopy(tactics), parsed.family):
            raise RuntimeError(
                "selected Family Pack failed complete inference correctness after tactic installation"
            )
        tuned_names = {item["source_operation"] for item in records}
        remaining = [
            {
                "source_operation": item["source_operation"],
                "logical_phase": item["logical_phase"],
                "estimated_milliseconds_saved_per_inference": item[
                    "estimated_milliseconds_saved"
                ]
                * item["repetitions"],
            }
            for item in ranked
            if item["source_operation"] not in tuned_names
        ]
        results = []
        for item in ranked:
            milliseconds = measurements_by_operation.get(
                item["source_operation"], item["best_current_milliseconds"]
            )
            results.append(
                {
                    "source_operation": item["source_operation"],
                    "logical_phase": item["logical_phase"],
                    "milliseconds": milliseconds,
                    "tuned": item["source_operation"] in tuned_names,
                    "milliseconds_saved_per_inference": max(
                        0.0, item["best_current_milliseconds"] - milliseconds
                    )
                    * item["repetitions"],
                }
            )
        return {
            "schema": "apxinf.tuning.report.v1",
            "family": parsed.family,
            "profile": parsed.profile,
            "target_fingerprint": deepcopy(parsed.target_fingerprint),
            "budget_seconds": float(budget),
            "seconds_spent": elapsed,
            "coverage": {"tuned": len(records), "total": len(ranked)},
            "best_current_results": results,
            "remaining_hotspots": remaining,
            "tactics": tactics,
            "complete_inference_correctness": True,
        }

    @staticmethod
    def validate_tactics(
        tactics: Mapping[str, Any], environment: Mapping[str, Any]
    ) -> None:
        if tactics.get("schema") != TACTIC_SCHEMA:
            raise ValueError("unsupported tactic schema")
        actual = _target_fingerprint(environment)
        header = {
            "device_name": tactics.get("device_name"),
            "sm": tactics.get("sm"),
            "multiprocessor_count": tactics.get("multiprocessor_count"),
            "kernel_build_id": tactics.get("kernel_build_id"),
            "cuda_version": tactics.get("cuda_version"),
            "library_versions": tactics.get("library_versions"),
        }
        if header != actual:
            raise ValueError("incompatible tactic environment")
        records = tactics.get("records")
        if not isinstance(records, list):
            raise ValueError("tactic records must be an array")
        for record in records:
            expected = {
                "device_name": record.get("device_name"),
                "sm": record.get("device", {}).get("sm"),
                "multiprocessor_count": record.get("device", {}).get(
                    "multiprocessor_count"
                ),
                "kernel_build_id": record.get("kernel_build_id"),
                "cuda_version": record.get("cuda_version"),
                "library_versions": record.get("library_versions"),
            }
            if expected != actual:
                raise ValueError("incompatible tactic environment")


def _tactic_record(
    workload: Mapping[str, Any], backend: str, tactic_id: int, milliseconds: float
) -> dict[str, Any]:
    target = workload["target_fingerprint"]
    return {
        "source_operation": workload["source_operation"],
        "logical_phase": workload["logical_phase"],
        "profile": workload["profile"],
        "key": {
            **{
                field: workload[field]
                for field in (
                    "op",
                    "m",
                    "n",
                    "k",
                    "activation_dtype",
                    "weight_dtype",
                    "output_dtype",
                    "layout",
                    "scale_mode",
                    "epilogue",
                    "workspace_limit",
                )
            },
            "device": {
                "sm": target["sm"],
                "multiprocessor_count": target["multiprocessor_count"],
            },
        },
        "tactic": {"backend": backend, "id": tactic_id},
        "milliseconds": milliseconds,
        "device_name": target["device_name"],
        "device": {
            "sm": target["sm"],
            "multiprocessor_count": target["multiprocessor_count"],
        },
        "kernel_build_id": target["kernel_build_id"],
        "cuda_version": target["cuda_version"],
        "library_versions": deepcopy(target["library_versions"]),
    }
