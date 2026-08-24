"""Family-neutral canonical-trace kernel coverage and handoff protocol."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


CLASSIFICATIONS = frozenset(
    {
        "existing_fused",
        "existing_primitive",
        "layout_only",
        "correct_fallback",
        "unsupported",
    }
)
REQUIRED_COMPUTATION_FIELDS = (
    "id",
    "operation",
    "semantics",
    "references",
    "dtype",
    "layout",
    "shapes",
    "tolerances",
    "golden_tensors",
    "frequency",
    "performance_impact",
    "expected_interface",
)


class KernelCoverageError(ValueError):
    """The trace or capability catalog cannot support a safe classification."""


def _require_computation(computation: Any, index: int) -> Mapping[str, Any]:
    if not isinstance(computation, dict):
        raise KernelCoverageError(f"computations[{index}] must be an object")
    for field in REQUIRED_COMPUTATION_FIELDS:
        value = computation.get(field)
        if value is None or value == "" or value == [] or value == {}:
            raise KernelCoverageError(
                f"computations[{index}] requires non-empty {field}"
            )
    return computation


def _capability_matches(computation: Mapping[str, Any], capability: Any) -> bool:
    return (
        isinstance(capability, dict)
        and capability.get("operation") == computation["operation"]
        and computation["dtype"] in capability.get("supported_dtypes", [])
        and computation["layout"] in capability.get("supported_layouts", [])
        and all(shape in capability.get("target_shapes", []) for shape in computation["shapes"])
        and capability.get("interface") == computation["expected_interface"]
    )


def _validate_returned_capability(
    computation: Mapping[str, Any], capability: Mapping[str, Any]
) -> None:
    if capability.get("provenance") != "kernel_workflow_return":
        return
    validation = capability.get("reference_validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise KernelCoverageError(
            f"returned capability {computation['operation']} must be revalidated "
            "against the original family references"
        )
    if validation.get("references") != computation["references"]:
        raise KernelCoverageError(
            f"returned capability {computation['operation']} was not revalidated "
            "against the original family references"
        )


def analyze_kernel_coverage(
    trace: Mapping[str, Any],
    capability_catalog: Sequence[Mapping[str, Any]],
    requested_targets: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Classify every required computation; never synthesize an approximation."""
    if trace.get("schema_version") != "1.0":
        raise KernelCoverageError("canonical trace schema_version must be 1.0")
    family = trace.get("family")
    if not isinstance(family, str) or not family:
        raise KernelCoverageError("canonical trace requires a family")
    computations = trace.get("computations")
    if not isinstance(computations, list) or not computations:
        raise KernelCoverageError("canonical trace requires computations")

    classifications = []
    gaps = []
    opportunities = []
    for index, raw_computation in enumerate(computations):
        computation = _require_computation(raw_computation, index)
        matches = [
            capability
            for capability in capability_catalog
            if _capability_matches(computation, capability)
        ]
        if len(matches) > 1:
            raise KernelCoverageError(
                f"computation {computation['id']} has ambiguous kernel capabilities"
            )
        capability = matches[0] if matches else None
        if capability is not None and capability.get("provenance") not in {
            "built_in",
            "kernel_workflow_return",
        }:
            raise KernelCoverageError(
                f"capability {computation['operation']} requires explicit provenance"
            )
        classification = (
            "missing_required_capability"
            if capability is None
            else capability.get("classification")
        )
        if classification != "missing_required_capability" and classification not in CLASSIFICATIONS:
            raise KernelCoverageError(
                f"computation {computation['id']} has an unrecognized classification"
            )
        if capability is not None:
            _validate_returned_capability(computation, capability)
        classified = {
            "computation_id": computation["id"],
            "operation": computation["operation"],
            "classification": classification,
            "semantics": computation["semantics"],
            "references": computation["references"],
        }
        classifications.append(classified)
        if classification == "missing_required_capability":
            gaps.append(
                {
                    "computation_id": computation["id"],
                    **{
                        field: computation[field]
                        for field in REQUIRED_COMPUTATION_FIELDS
                        if field not in {"id", "operation"}
                    },
                    "operation": computation["operation"],
                    "requested_targets": list(requested_targets),
                }
            )
        elif classification == "correct_fallback":
            opportunities.append(
                {
                    "computation_id": computation["id"],
                    "operation": computation["operation"],
                    "semantics": computation["semantics"],
                    "frequency": computation["frequency"],
                    "performance_impact": computation["performance_impact"],
                    "requested_targets": list(requested_targets),
                }
            )

    blocked = bool(gaps) or any(
        item["classification"] == "unsupported" for item in classifications
    )
    return {
        "schema_version": "1.0",
        "port_id": trace.get("port_id"),
        "family": family,
        "status": "blocked" if blocked else "passed",
        "classifications": classifications,
        "kernel_gaps": gaps,
        "optimization_opportunities": opportunities,
    }
