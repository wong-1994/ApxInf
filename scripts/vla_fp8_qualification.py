"""Machine-evaluated qualification of requested VLA Thor FP8 Ports."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from tuning_workloads import GenericGemmTuner


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REFERENCE_FIELDS = (
    "weights_sha256",
    "scales_sha256",
    "calibration_inputs_sha256",
)
_CORRECTNESS_OUTPUTS = ("normalized_actions", "deployable_actions")
_MAX_THRESHOLDS = {
    "stage": {"absolute": 0.1, "relative": 0.05},
    "normalized_actions": {"absolute": 0.05, "relative": 0.03},
    "deployable_actions": {"absolute": 0.02, "relative": 0.01},
}


def qualify_vla_port(
    request: Mapping[str, Any], tuple_evidence: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Qualify exactly the requested VLA target/precision tuples.

    Evidence is deliberately supplied by the Family Pack.  This coordinator
    understands tuple selection and Gates, while VLA owns its named stages and
    action outputs.
    """
    if request.get("family") != "vla":
        raise ValueError("qualification requires the VLA Family Pack")
    requested = _requested_pairs(request.get("requested_tuples"))
    if ("orin", "fp8") in requested:
        raise ValueError("Orin FP8 is unsupported and cannot be qualified")
    unknown = requested - {
        ("thor", "bf16"),
        ("thor", "fp8"),
        ("orin", "bf16"),
        ("orin", "int8_w8a8"),
    }
    if unknown:
        target, precision = sorted(unknown)[0]
        raise ValueError(f"unsupported target/precision tuple: {target}/{precision}")
    if ("thor", "fp8") in requested:
        _validate_fp8_reference(request.get("reference_evidence"))

    evidence = _index_evidence(tuple_evidence)
    extras = set(evidence) - requested
    if extras:
        target, precision = sorted(extras)[0]
        raise ValueError(f"evidence supplied for unrequested tuple: {target}/{precision}")

    gates: dict[str, dict[str, Any]] = {}
    validated: set[tuple[str, str]] = set()
    for pair in sorted(requested):
        item = evidence.get(pair)
        if item is None:
            gates[_gate_name(pair)] = {"status": "failed", "reason": "missing evidence"}
            continue
        if pair == ("thor", "fp8"):
            _validate_shared_fp8_contracts(item)
            passed, comparisons = _evaluate_fp8_correctness(request, item)
            passed = passed and _deployment_passed(item)
            gates["fp8_correctness"] = {
                "status": "passed" if passed else "failed",
                "comparisons": comparisons,
            }
            if passed:
                validated.add(pair)
        else:
            _validate_kernel_coverage(item)
            correctness_passed = (
                item.get("correctness", {}).get("status") == "passed"
            )
            if correctness_passed and _deployment_passed(item):
                validated.add(pair)

    fp8 = ("thor", "fp8")
    bf16 = ("thor", "bf16")
    if fp8 in requested and bf16 in requested:
        relative = _same_device_improvement(evidence.get(bf16), evidence.get(fp8))
        minimum = request.get("thresholds", {}).get("fp8", {}).get(
            "minimum_bf16_improvement"
        )
        if not _nonnegative(minimum) or minimum > 1:
            raise ValueError(
                "dual-precision qualification requires minimum_bf16_improvement"
            )
        gates["fp8_vs_bf16"] = {
            "status": "passed" if relative >= minimum else "failed",
            "improvement": relative,
            "minimum_improvement": float(minimum),
        }
        if relative < minimum:
            validated.discard(fp8)

    all_passed = validated == requested
    return {
        "status": "release_qualified" if all_passed else "correctness_failed",
        "gates": gates,
        "public_support": [
            {"family": "vla", "target": target, "precision": precision}
            for target, precision in sorted(validated)
        ],
    }


def _requested_pairs(value: Any) -> set[tuple[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("requested_tuples must be a non-empty array")
    pairs: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("each requested tuple must be an object")
        target, precision = item.get("target"), item.get("precision")
        if not isinstance(target, str) or not isinstance(precision, str):
            raise ValueError("requested tuple target and precision must be strings")
        pair = (target, precision)
        if pair in pairs:
            raise ValueError(f"duplicate requested tuple: {target}/{precision}")
        pairs.add(pair)
    return pairs


def _validate_fp8_reference(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("FP8 requires reference_evidence")
    for field in _REFERENCE_FIELDS:
        digest = value.get(field)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"FP8 reference_evidence requires {field}")


def _index_evidence(
    values: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result = {}
    for item in values:
        if not isinstance(item, Mapping):
            raise ValueError("tuple evidence must be an object")
        pair = (item.get("target"), item.get("precision"))
        if not all(isinstance(value, str) for value in pair):
            raise ValueError("tuple evidence target and precision must be strings")
        if pair in result:
            raise ValueError(f"duplicate evidence for {pair[0]}/{pair[1]}")
        result[pair] = item
    return result


def _validate_kernel_coverage(item: Mapping[str, Any]) -> None:
    coverage = item.get("kernel_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("tuple evidence requires shared kernel_coverage")
    if coverage.get("schema_version") != "1.0" or coverage.get("family") != "vla":
        raise ValueError("kernel coverage must use the shared family-attributed contract")
    if coverage.get("status") != "passed":
        raise ValueError("kernel coverage did not pass")


def _validate_shared_fp8_contracts(item: Mapping[str, Any]) -> None:
    _validate_kernel_coverage(item)
    tuning = item.get("tuning")
    if not isinstance(tuning, Mapping):
        raise ValueError("FP8 evidence requires a family-attributed tuning report")
    if (
        tuning.get("schema") != "apxinf.tuning.report.v1"
        or tuning.get("family") != "vla"
        or tuning.get("complete_inference_correctness") is not True
    ):
        raise ValueError("FP8 evidence requires the shared VLA tuning contract")
    fingerprint = item.get("target_fingerprint")
    if (
        not isinstance(fingerprint, Mapping)
        or "thor" not in str(fingerprint.get("device_name", "")).lower()
        or fingerprint.get("sm") != 110
    ):
        raise ValueError("FP8 evidence requires a validated Thor fingerprint")
    if fingerprint != tuning.get("target_fingerprint"):
        raise ValueError("tuning report was not validated on the tuple fingerprint")
    GenericGemmTuner.validate_tactics(tuning.get("tactics", {}), fingerprint)


def _evaluate_fp8_correctness(
    request: Mapping[str, Any], item: Mapping[str, Any]
) -> tuple[bool, list[dict[str, Any]]]:
    thresholds = request.get("thresholds", {}).get("fp8")
    actual = item.get("correctness")
    if not isinstance(thresholds, Mapping) or not isinstance(actual, Mapping):
        raise ValueError("FP8 requires thresholds and correctness evidence")
    stage_thresholds = thresholds.get("stages")
    stage_actual = actual.get("stages")
    if not isinstance(stage_thresholds, Mapping) or not stage_thresholds:
        raise ValueError("FP8 requires at least one VLA stage threshold")
    if not isinstance(stage_actual, Mapping):
        raise ValueError("FP8 requires VLA stage correctness evidence")
    comparisons = []
    for name, limit in stage_thresholds.items():
        comparisons.extend(
            _compare_metrics(
                f"stages.{name}",
                limit,
                stage_actual.get(name),
                _MAX_THRESHOLDS["stage"],
            )
        )
    for name in _CORRECTNESS_OUTPUTS:
        comparisons.extend(
            _compare_metrics(
                name, thresholds.get(name), actual.get(name), _MAX_THRESHOLDS[name]
            )
        )
    return all(item["passed"] for item in comparisons), comparisons


def _compare_metrics(
    name: str, limits: Any, observed: Any, maximums: Mapping[str, float]
) -> list[dict[str, Any]]:
    if not isinstance(limits, Mapping) or not isinstance(observed, Mapping):
        raise ValueError(f"FP8 requires {name} thresholds and evidence")
    result = []
    for metric, observed_name in (
        ("absolute", "max_absolute"),
        ("relative", "max_relative"),
    ):
        limit, actual = limits.get(metric), observed.get(observed_name)
        if not _nonnegative(limit) or not _nonnegative(actual):
            raise ValueError(f"FP8 {name}.{metric} must be a non-negative number")
        if limit > maximums[metric]:
            raise ValueError(f"FP8 {name}.{metric} exceeds the VLA FP8 maximum")
        result.append(
            {
                "output": name,
                "metric": metric,
                "observed": float(actual),
                "threshold": float(limit),
                "passed": actual <= limit,
            }
        )
    return result


def _same_device_improvement(bf16: Any, fp8: Any) -> float:
    if not isinstance(bf16, Mapping) or not isinstance(fp8, Mapping):
        raise ValueError("same-device BF16 and FP8 evidence is required")
    if bf16.get("target_fingerprint") != fp8.get("target_fingerprint"):
        raise ValueError("BF16 and FP8 comparison requires the same Thor fingerprint")
    baseline = bf16.get("performance", {}).get("control_step_p95_ms")
    candidate = fp8.get("performance", {}).get("control_step_p95_ms")
    if not _nonnegative(baseline) or baseline == 0 or not _nonnegative(candidate):
        raise ValueError("comparison requires positive BF16 and non-negative FP8 p95")
    return round((baseline - candidate) / baseline, 12)


def _nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
    )


def _deployment_passed(item: Mapping[str, Any]) -> bool:
    deployment = item.get("deployment")
    return isinstance(deployment, Mapping) and all(
        deployment.get(name) is True
        for name in ("inference", "policy_io", "serving")
    )


def _gate_name(pair: tuple[str, str]) -> str:
    return f"{pair[0]}_{pair[1]}_evidence"
