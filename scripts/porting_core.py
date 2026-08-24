"""Family-neutral contracts used by the model Porting Workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


HARDWARE_MATRIX = frozenset(
    {
        ("thor", "bf16"),
        ("thor", "fp8"),
        ("orin", "bf16"),
        ("orin", "int8_w8a8"),
    }
)
KNOWN_FAMILIES = frozenset({"llm", "vlm", "vla"})
WORKFLOW_ARTIFACT_ENVELOPE_VERSION = "1.0"


@dataclass(frozen=True)
class FamilyPack:
    """The narrow interface through which the Core obtains family semantics."""

    family: str
    contract_family: str
    default_contract_version: str
    supported_tuples: frozenset[tuple[str, str]]
    required_capabilities: frozenset[str]
    reference_contract: str
    canonicalization_contract: str
    verification_contract: str
    integration_contract: str
    serving_contract: str
    performance_contract: str
    payload_schema: Callable[[Path], str]

    def validate_payload(self, path: Path, payload: Any) -> str:
        if path.suffix != ".json":
            if not isinstance(payload, bytes) or not payload:
                raise ValueError("Workflow Artifact payload must not be empty")
            return self.payload_schema(path)
        if not isinstance(payload, dict):
            raise ValueError("Workflow Artifact payload must be a JSON object")
        declared_family = payload.get("family")
        if declared_family is not None and declared_family not in {
            self.family,
            self.contract_family,
        }:
            raise ValueError(
                f"artifact payload family {declared_family!r} does not match "
                f"selected {self.family!r} Family Pack"
            )
        return self.payload_schema(path)


def _vla_payload_schema(path: Path) -> str:
    schemas = {
        "source_inventory.json": "source-inventory-v1.schema.json",
        "reference_capture.json": "reference-capture-v1.schema.json",
        "capability_classification.json": "capability-classification-v1.schema.json",
        "canonical_trace.json": "canonical-trace-v1.schema.json",
        "canonical_equivalence.json": "canonical-equivalence-v1.schema.json",
        "canonicalization_gap_report.json": "canonicalization-gap-report-v1.schema.json",
        "gap_report.json": "gap-report-v1.schema.json",
    }
    if path.suffix == ".py":
        return "python-reference-adapter-v1"
    if path.suffix == ".txt":
        return "text-environment-lock-v1"
    return schemas.get(path.name, "workflow-artifact-payload-v1.schema.json")


VLA_FAMILY_PACK = FamilyPack(
    family="vla",
    contract_family="canonical_transformer_vla",
    default_contract_version="1.0",
    supported_tuples=HARDWARE_MATRIX,
    required_capabilities=frozenset(
        {
            "shape_profiles",
            "attention",
            "masks",
            "position_encodings",
            "normalization",
            "activations",
            "conditioning",
            "action_heads",
            "schedules",
            "control_flow",
        }
    ),
    reference_contract="vla-reference-adapter-1.0",
    canonicalization_contract="vla-canonicalization-1.0",
    verification_contract="vla-equivalence-1.0",
    integration_contract="observation-to-action",
    serving_contract="action-serving",
    performance_contract="vla-control-step-1.0",
    payload_schema=_vla_payload_schema,
)

FAMILY_PACKS = {VLA_FAMILY_PACK.family: VLA_FAMILY_PACK}


def select_family_pack(family: Any) -> FamilyPack:
    if family not in KNOWN_FAMILIES:
        raise ValueError("model_family must explicitly select llm, vlm, or vla")
    pack = FAMILY_PACKS.get(family)
    if pack is None:
        raise ValueError(f"{family} Family Pack is not registered")
    return pack


def validate_requested_tuple(pack: FamilyPack, target: str, precision: str) -> None:
    pair = (target, precision)
    if pair not in HARDWARE_MATRIX:
        if pair == ("orin", "fp8"):
            raise ValueError("orin does not support fp8; use bf16 or int8_w8a8")
        raise ValueError(f"unsupported target/precision tuple: {target}/{precision}")
    if pair not in pack.supported_tuples:
        raise ValueError(
            f"{pack.family} Family Pack does not support target/precision tuple: "
            f"{target}/{precision}"
        )
