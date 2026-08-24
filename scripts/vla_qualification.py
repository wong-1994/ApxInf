"""VLA performance profile for the common qualification engine."""

from __future__ import annotations

from typing import Any, Mapping


class VlaQualificationProfile:
    family = "vla"
    lower_precisions = frozenset({"fp8", "int8_w8a8"})
    supported_tuples = frozenset(
        {
            ("thor", "bf16"),
            ("thor", "fp8"),
            ("orin", "bf16"),
            ("orin", "int8_w8a8"),
        }
    )

    def validate_request(
        self, request: Mapping[str, Any], pair: tuple[str, str]
    ) -> None:
        if pair not in self.supported_tuples:
            raise ValueError(f"unsupported VLA tuple: {pair[0]}/{pair[1]}")

    def evaluate(
        self, request_tuple: Mapping[str, Any], evidence: Mapping[str, Any]
    ) -> dict[str, dict[str, Any]]:
        limits = request_tuple.get("performance_limits")
        if not isinstance(limits, Mapping):
            raise ValueError("requested VLA tuple requires performance_limits")
        benchmark = evidence["benchmark"]
        for field in ("observation_profile", "action_profile"):
            if not isinstance(benchmark.get(field), Mapping):
                raise ValueError(f"VLA benchmark requires {field}")
        metrics = benchmark["metrics"]
        gates = {}
        for name in ("control_step_p50_ms", "control_step_p95_ms"):
            limit = limits.get(name)
            observed = metrics.get(name)
            if not _positive(limit):
                raise ValueError(f"requested VLA tuple requires positive {name}")
            if not _nonnegative(observed):
                raise ValueError(f"VLA benchmark requires non-negative {name}")
            gates[name] = {
                "status": "passed" if observed <= limit else "failed",
                "observed": float(observed),
                "limit": float(limit),
            }
        return gates

    def deployment_passed(self, evidence: Mapping[str, Any]) -> bool:
        deployment = evidence.get("deployment")
        return isinstance(deployment, Mapping) and all(
            deployment.get(name) is True
            for name in ("inference", "policy_processing", "action_serving")
        )

    def comparison_metric(self, evidence: Mapping[str, Any]) -> float:
        value = evidence["benchmark"]["metrics"].get("control_step_p95_ms")
        if not _nonnegative(value):
            raise ValueError("VLA BF16 comparison requires control-step p95")
        return float(value)


def _positive(value: Any) -> bool:
    return _nonnegative(value) and value > 0


def _nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
    )


VLA_QUALIFICATION_PROFILE = VlaQualificationProfile()
