"""Family-neutral qualification state for requested Port tuples."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Protocol, Sequence


Pair = tuple[str, str]


class QualificationProfile(Protocol):
    family: str
    lower_precisions: frozenset[str]

    def validate_request(self, request: Mapping[str, Any], pair: Pair) -> None: ...
    def evaluate(
        self, request_tuple: Mapping[str, Any], evidence: Mapping[str, Any]
    ) -> dict[str, dict[str, Any]]: ...
    def deployment_passed(self, evidence: Mapping[str, Any]) -> bool: ...
    def comparison_metric(self, evidence: Mapping[str, Any]) -> float: ...


class QualificationEngine:
    """Compute common Port states while leaving metrics to a Family Pack."""

    def __init__(self, profile: QualificationProfile):
        self.profile = profile

    def qualify(
        self,
        request: Mapping[str, Any],
        tuple_evidence: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if request.get("family") != self.profile.family:
            raise ValueError(
                f"qualification requires the {self.profile.family} Family Pack"
            )
        requested = _requested(request.get("requested_tuples"))
        evidence = _evidence(tuple_evidence)
        extras = set(evidence) - set(requested)
        if extras:
            target, precision = sorted(extras)[0]
            raise ValueError(
                f"evidence supplied for unrequested tuple: {target}/{precision}"
            )
        waivers = _waivers(request.get("waivers", []), set(requested))
        representative = request.get("representative_real_inputs") is True
        diagnostics: list[dict[str, Any]] = []
        tuples: list[dict[str, Any]] = []

        for pair, declaration in requested.items():
            self.profile.validate_request(request, pair)
            item = evidence.get(pair)
            if item is None:
                tuples.append(_missing_tuple(pair))
                continue
            _validate_common_evidence(pair, item)
            gates = self.profile.evaluate(declaration, item)
            correctness = item.get("correctness", {}).get("status") == "passed"
            gates["correctness"] = {
                "status": "passed" if correctness else "failed"
            }
            environment = item.get("environment_conforming") is True
            gates["environment"] = {
                "status": "passed" if environment else "failed"
            }
            if not environment:
                diagnostics.append(
                    {
                        "kind": "nonconforming_environment",
                        "target": pair[0],
                        "precision": pair[1],
                        "environment": dict(item["environment"]),
                    }
                )
            fresh = item.get("fresh") is True
            gates["freshness"] = {"status": "passed" if fresh else "failed"}
            applied_waivers = _apply_waivers(gates, waivers.get(pair, []))
            deployment = self.profile.deployment_passed(item)
            performance = all(
                gate.get("status") == "passed"
                for name, gate in gates.items()
                if name not in {"correctness"}
            )
            releasable = (
                deployment
                and correctness
                and performance
                and representative
                and not applied_waivers
            )
            status = (
                "release_qualified"
                if releasable
                else "provisional"
                if deployment and correctness and performance and not representative
                else "qualification_failed"
                if deployment and not correctness
                else "performance_pending"
                if deployment
                else "incomplete"
            )
            tuples.append(
                {
                    "target": pair[0],
                    "precision": pair[1],
                    "status": status,
                    "deployment_complete": deployment,
                    "gates": gates,
                    "benchmark": dict(item["benchmark"]),
                    "environment": dict(item["environment"]),
                    "waivers": applied_waivers,
                }
            )

        self._add_relative_gates(request, requested, evidence, tuples)
        deployment_complete = all(item["deployment_complete"] for item in tuples)
        release_qualified = all(
            item["status"] == "release_qualified" for item in tuples
        )
        provisional = all(
            item["status"] in {"release_qualified", "provisional"} for item in tuples
        ) and not release_qualified
        qualification_failed = deployment_complete and any(
            item["status"] == "qualification_failed" for item in tuples
        )
        performance_pending = (
            deployment_complete and not release_qualified and not qualification_failed
        )
        status = (
            "release_qualified"
            if release_qualified
            else "provisional"
            if provisional
            else "qualification_failed"
            if qualification_failed
            else "performance_pending"
            if performance_pending
            else "incomplete"
        )
        return {
            "family": self.profile.family,
            "status": status,
            "deployment_complete": deployment_complete,
            "performance_pending": performance_pending,
            "release_qualified": release_qualified,
            "tuples": tuples,
            "diagnostics": diagnostics,
        }

    def _add_relative_gates(self, request, requested, evidence, tuples) -> None:
        by_pair = {(item["target"], item["precision"]): item for item in tuples}
        minimums = request.get("minimum_bf16_improvement", {})
        for target, precision in requested:
            if precision not in self.profile.lower_precisions:
                continue
            bf16 = (target, "bf16")
            if bf16 not in requested:
                continue
            minimum = minimums.get(precision)
            if not _number(minimum) or not 0 <= minimum <= 1:
                raise ValueError(
                    "dual-precision qualification requires "
                    f"{precision} BF16 improvement"
                )
            baseline, candidate = evidence.get(bf16), evidence.get((target, precision))
            if baseline is None or candidate is None:
                observed = None
                passed = False
            else:
                if baseline.get("environment") != candidate.get("environment"):
                    raise ValueError(
                        "BF16 comparison requires the same device environment"
                    )
                base = self.profile.comparison_metric(baseline)
                lower = self.profile.comparison_metric(candidate)
                if base <= 0:
                    raise ValueError("BF16 comparison metric must be positive")
                observed = round((base - lower) / base, 12)
                passed = observed >= minimum
            gate = {
                "status": "passed" if passed else "failed",
                "observed": observed,
                "minimum": float(minimum),
            }
            result = by_pair[(target, precision)]
            result["gates"]["bf16_improvement"] = gate
            if not passed and result["status"] == "release_qualified":
                result["status"] = "performance_pending"


def _requested(value: Any) -> dict[Pair, Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("requested_tuples must be a non-empty array")
    result = {}
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("each requested tuple must be an object")
        pair = (item.get("target"), item.get("precision"))
        if not all(isinstance(part, str) and part for part in pair):
            raise ValueError("requested tuple target and precision must be strings")
        if pair in result:
            raise ValueError(f"duplicate requested tuple: {pair[0]}/{pair[1]}")
        result[pair] = item
    return result


def _evidence(values: Sequence[Mapping[str, Any]]) -> dict[Pair, Mapping[str, Any]]:
    result = {}
    for item in values:
        if not isinstance(item, Mapping):
            raise ValueError("tuple evidence must be an object")
        pair = (item.get("target"), item.get("precision"))
        if not all(isinstance(part, str) for part in pair):
            raise ValueError("tuple evidence target and precision must be strings")
        if pair in result:
            raise ValueError(f"duplicate evidence for {pair[0]}/{pair[1]}")
        result[pair] = item
    return result


def _validate_common_evidence(pair: Pair, item: Mapping[str, Any]) -> None:
    benchmark = item.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise ValueError("tuple evidence requires benchmark")
    for field in (
        "warmup",
        "samples",
        "workspace_bytes",
        "peak_memory_bytes",
        "metrics",
    ):
        if field not in benchmark:
            raise ValueError(f"benchmark requires {field}")
    if not isinstance(benchmark["warmup"], int) or benchmark["warmup"] < 0:
        raise ValueError("benchmark warmup must be a non-negative integer")
    if not isinstance(benchmark["samples"], int) or benchmark["samples"] <= 0:
        raise ValueError("benchmark samples must be a positive integer")
    for field in ("workspace_bytes", "peak_memory_bytes"):
        if not isinstance(benchmark[field], int) or benchmark[field] < 0:
            raise ValueError(f"benchmark {field} must be a non-negative integer")
    if not isinstance(benchmark["metrics"], Mapping):
        raise ValueError("benchmark metrics must be an object")
    environment = item.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("tuple evidence requires environment")
    if pair[0] in {"thor", "orin"}:
        for field in (
            "power_mode",
            "clocks",
            "temperature_c",
            "device",
            "driver",
            "cuda",
            "libraries",
            "kernel_build",
        ):
            if field not in environment:
                raise ValueError(f"target environment requires {field}")


def _waivers(value: Any, requested: set[Pair]) -> dict[Pair, list[dict[str, Any]]]:
    if not isinstance(value, list):
        raise ValueError("waivers must be an array")
    result: dict[Pair, list[dict[str, Any]]] = {}
    for waiver in value:
        if not isinstance(waiver, Mapping):
            raise ValueError("waiver must be an object")
        pair = (waiver.get("target"), waiver.get("precision"))
        if pair not in requested:
            raise ValueError("waiver must be scoped to a requested tuple")
        if waiver.get("gate") not in {"correctness", "performance"}:
            raise ValueError("waiver gate must be correctness or performance")
        if (
            waiver.get("maintainer_approved") is not True
            or not waiver.get("approved_by")
        ):
            raise ValueError("waiver requires explicit maintainer approval")
        if not waiver.get("evidence"):
            raise ValueError("waiver requires evidence")
        try:
            expiry = date.fromisoformat(waiver.get("expires", ""))
        except (TypeError, ValueError) as error:
            raise ValueError("waiver requires an ISO expiration date") from error
        if expiry < date.today():
            raise ValueError("waiver has expired")
        result.setdefault(pair, []).append(dict(waiver))
    return result


def _apply_waivers(gates, waivers):
    applied = []
    for waiver in waivers:
        gate_name = waiver["gate"]
        names = (
            ["correctness"]
            if gate_name == "correctness"
            else [name for name in gates if name != "correctness"]
        )
        failed = [name for name in names if gates[name]["status"] == "failed"]
        for name in failed:
            gates[name] = {**gates[name], "status": "waived"}
        if failed:
            applied.append(waiver)
    return applied


def _missing_tuple(pair: Pair) -> dict[str, Any]:
    return {
        "target": pair[0],
        "precision": pair[1],
        "status": "incomplete",
        "deployment_complete": False,
        "gates": {"evidence": {"status": "failed", "reason": "missing evidence"}},
        "waivers": [],
    }


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
