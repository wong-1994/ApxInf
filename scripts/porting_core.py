"""Family-neutral contracts used by the model Porting Workflow."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


HARDWARE_MATRIX = frozenset(
    {
        ("thor", "bf16"),
        ("thor", "fp8"),
        ("orin", "bf16"),
        ("orin", "int8_w8a8"),
    }
)
HARDWARE_TUPLES = (
    ("thor", "bf16"),
    ("thor", "fp8"),
    ("orin", "bf16"),
    ("orin", "int8_w8a8"),
)
KNOWN_FAMILIES = frozenset({"llm", "vlm", "vla"})
WORKFLOW_ARTIFACT_ENVELOPE_VERSION = "1.0"
REPORT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class PortOutcome:
    code: int
    category: str


SUCCESS = PortOutcome(0, "success")
MISSING_INPUT = PortOutcome(2, "missing_input")
INVALID_INPUT = PortOutcome(3, "invalid_input")
UNSUPPORTED_TARGET = PortOutcome(4, "unsupported_target")
ENVIRONMENT_FAILURE = PortOutcome(5, "environment_failure")
REFERENCE_LOAD_FAILURE = PortOutcome(6, "reference_load_failure")
REFERENCE_TRACE_FAILURE = PortOutcome(7, "reference_trace_failure")
UNSUPPORTED_SEMANTICS = PortOutcome(8, "unsupported_semantics")
CORRECTNESS_FAILURE = PortOutcome(9, "correctness_failure")
KERNEL_GAP = PortOutcome(10, "kernel_gap")


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
    payload_schema: Callable[[Path, Any], str]
    report_defaults: Callable[[dict[str, Any]], dict[str, Any]]

    def validate_payload(self, path: Path, payload: Any) -> str:
        if path.suffix != ".json":
            if not isinstance(payload, bytes) or not payload:
                raise ValueError("Workflow Artifact payload must not be empty")
            return self.payload_schema(path, payload)
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
        return self.payload_schema(path, payload)


def _vla_payload_schema(path: Path, payload: Any) -> str:
    schemas = {
        "source_inventory.json": "reference-inventory-v1.schema.json",
        "environment.json": "reference-environment-v1.schema.json",
        "inspection.json": "reference-capture-v1.schema.json",
        "capability_classification.json": "capability-classification-v1.schema.json",
        "canonical_trace.json": "canonical-trace-v1.schema.json",
        "canonical_equivalence.json": "canonical-equivalence-v1.schema.json",
        "canonicalization_gap_report.json": "canonicalization-gap-report-v1.schema.json",
        "capability_gap_report.json": "capability-gap-report-v1.schema.json",
        "kernel_coverage.json": "kernel-coverage-v1.schema.json",
        "kernel_gap_handoff.json": "kernel-gap-handoff-v1.schema.json",
    }
    if path.suffix == ".py":
        return "python-reference-adapter-v1"
    if path.suffix == ".txt":
        return "text-environment-lock-v1"
    schema_name = schemas.get(path.name)
    if schema_name is None:
        raise ValueError(f"no VLA payload schema is registered for {path.name}")
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / schema_name
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    _validate_json_schema(payload, schema, path.name, schema, schema_path)
    return schema_name


def _json_type_matches(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }[expected]


def _resolve_schema_reference(
    reference: str,
    root_schema: Mapping[str, Any],
    schema_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    document, _, fragment = reference.partition("#")
    resolved_path = schema_path
    resolved_root = root_schema
    if document:
        resolved_path = schema_path.parent / document
        resolved_root = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved: Any = resolved_root
    if fragment:
        if not fragment.startswith("/"):
            raise ValueError(f"unsupported payload schema reference: {reference}")
        for token in fragment[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if not isinstance(resolved, dict) or token not in resolved:
                raise ValueError(f"unresolved payload schema reference: {reference}")
            resolved = resolved[token]
    if not isinstance(resolved, dict):
        raise ValueError(f"payload schema reference is not an object: {reference}")
    return resolved, resolved_root, resolved_path


def _validate_json_schema(
    value: Any,
    schema: Mapping[str, Any],
    path: str,
    root_schema: Mapping[str, Any] | None = None,
    schema_path: Path | None = None,
) -> None:
    root_schema = root_schema or schema
    schema_path = schema_path or Path(__file__)
    if "$ref" in schema:
        referenced, referenced_root, referenced_path = _resolve_schema_reference(
            schema["$ref"], root_schema, schema_path
        )
        _validate_json_schema(
            value, referenced, path, referenced_root, referenced_path
        )
    for item_schema in schema.get("allOf", []):
        _validate_json_schema(value, item_schema, path, root_schema, schema_path)
    condition = schema.get("if")
    if isinstance(condition, dict) and isinstance(schema.get("then"), dict):
        try:
            _validate_json_schema(value, condition, path, root_schema, schema_path)
        except ValueError:
            pass
        else:
            _validate_json_schema(
                value, schema["then"], path, root_schema, schema_path
            )
    if "const" in schema and value != schema["const"]:
        raise ValueError(
            f"{path} does not satisfy payload schema const {schema['const']!r}"
        )
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} does not satisfy payload schema enum")
    expected = schema.get("type")
    if expected is not None:
        choices = [expected] if isinstance(expected, str) else expected
        if not any(_json_type_matches(value, choice) for choice in choices):
            raise ValueError(
                f"{path} does not satisfy payload schema type {expected!r}"
            )
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path} does not satisfy payload schema minLength")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            raise ValueError(f"{path} does not satisfy payload schema pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} does not satisfy payload schema minimum")
    if isinstance(value, dict):
        required = set(schema.get("required", []))
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"{path} payload schema requires {', '.join(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = sorted(value.keys() - properties.keys())
            if unknown:
                raise ValueError(f"{path} payload schema rejects {', '.join(unknown)}")
        for name, item in value.items():
            item_schema = properties.get(name)
            if item_schema is None and isinstance(
                schema.get("additionalProperties"), dict
            ):
                item_schema = schema["additionalProperties"]
            if item_schema is not None:
                _validate_json_schema(
                    item, item_schema, f"{path}.{name}", root_schema, schema_path
                )
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path} does not satisfy payload schema minItems")
        if schema.get("uniqueItems"):
            serialized = [
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in value
            ]
            if len(serialized) != len(set(serialized)):
                raise ValueError(f"{path} does not satisfy payload schema uniqueItems")
        if isinstance(schema.get("items"), dict):
            for index, item in enumerate(value):
                _validate_json_schema(
                    item,
                    schema["items"],
                    f"{path}[{index}]",
                    root_schema,
                    schema_path,
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ArtifactStore:
    """Stores family-validated Workflow Artifact provenance envelopes."""

    def __init__(
        self,
        port_dir: Path,
        request: dict[str, Any],
        pack: FamilyPack,
        orchestrator: Path,
        reference_adapter: Path,
    ):
        self.port_dir = port_dir
        self.request = request
        self.pack = pack
        self.orchestrator = orchestrator
        self.reference_adapter = reference_adapter

    def record(
        self,
        path: Path,
        environment_path: Path,
        upstream: Mapping[str, Path] | None = None,
        stage: str = "preflight",
    ) -> dict[str, Any]:
        try:
            payload = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.suffix == ".json"
                else path.read_bytes()
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid Workflow Artifact payload: {error}") from error
        payload_schema = self.pack.validate_payload(path, payload)
        dependencies = {"request": _sha256(self.port_dir / "request.json")}
        dependencies.update(
            {name: _sha256(item) for name, item in (upstream or {}).items()}
        )
        declared = self.request.get("dependency_fingerprints", {})
        target_environments = declared.get("target_environment_sha256", {})
        return {
            "envelope_version": WORKFLOW_ARTIFACT_ENVELOPE_VERSION,
            "family": self.pack.family,
            "capability_contract_version": self.request["capability_contract_version"],
            "stage": stage,
            "payload_schema": payload_schema,
            "path": path.relative_to(self.port_dir).as_posix(),
            "dependency_paths": {
                "environment": str(environment_path),
                "orchestrator": str(self.orchestrator),
                "reference_adapter": str(self.reference_adapter),
                **{name: str(item) for name, item in (upstream or {}).items()},
            },
            "state": "current",
            "explanation": {"changed_dependencies": [], "upstream_stale": []},
            "fingerprints": {
                "content_sha256": _sha256(path),
                "tool_sha256": {
                    "orchestrator": _sha256(self.orchestrator),
                    "reference_adapter": _sha256(self.reference_adapter),
                },
                "source_sha256": declared.get(
                    "source_sha256", self.request["source"]["sha256"]
                ),
                "checkpoint_sha256": declared.get(
                    "checkpoint_sha256", self.request["checkpoint"]["sha256"]
                ),
                "apxinf_source_sha256": declared.get(
                    "apxinf_source_sha256", _sha256(self.orchestrator)
                ),
                "kernel_build_sha256": declared.get("kernel_build_sha256"),
                "environment_sha256": declared.get(
                    "environment_sha256", _sha256(environment_path)
                ),
                "capability_contract_sha256": declared.get(
                    "capability_contract_sha256",
                    _value_sha256(self.request["capability_contract_version"]),
                ),
                "documentation_sha256": declared.get("documentation_sha256"),
                "target_environment_sha256": target_environments,
                "upstream_sha256": dependencies,
            },
        }


def resume_report(
    previous_report: Mapping[str, Any],
    current_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconcile recorded evidence with freshly fingerprinted dependencies.

    Existing payloads are retained.  They become usable only when their complete
    recorded fingerprint set still matches and every named upstream artifact is
    current.
    """

    report = deepcopy(dict(previous_report))
    artifacts = report.setdefault("artifacts", {})
    stale: set[str] = set()
    changed_by_artifact: dict[str, list[str]] = {}
    for name, recorded in artifacts.items():
        current = current_artifacts.get(name)
        recorded_fingerprints = recorded.get("fingerprints", {})
        current_fingerprints = current.get("fingerprints", {}) if current else {}
        keys = sorted(set(recorded_fingerprints) | set(current_fingerprints))
        changed = [
            key
            for key in keys
            if recorded_fingerprints.get(key) != current_fingerprints.get(key)
        ]
        if current is None:
            changed = ["current_fingerprint_evidence_missing"]
        if changed:
            stale.add(name)
            changed_by_artifact[name] = changed

    propagated = True
    while propagated:
        propagated = False
        for name, artifact in artifacts.items():
            upstream = artifact.get("fingerprints", {}).get("upstream_sha256", {})
            if name not in stale and any(parent in stale for parent in upstream):
                stale.add(name)
                propagated = True

    for name, artifact in artifacts.items():
        upstream = artifact.get("fingerprints", {}).get("upstream_sha256", {})
        upstream_stale = sorted(parent for parent in upstream if parent in stale)
        artifact["state"] = "stale" if name in stale else "current"
        artifact["explanation"] = {
            "changed_dependencies": changed_by_artifact.get(name, []),
            "upstream_stale": upstream_stale,
        }

    interrupted = sorted(
        name for name, status in report.get("stages", {}).items() if status == "running"
    )
    for name in interrupted:
        report["stages"][name] = "not_started"
    stale_stages = {
        artifact.get("stage")
        for name, artifact in artifacts.items()
        if name in stale and artifact.get("stage")
    }
    for stage in stale_stages:
        if (
            stage not in interrupted
            and report.get("stages", {}).get(stage) == "passed"
        ):
            report["stages"][stage] = "stale"
    gate_artifacts = {
        "capability_contract": ["capability_classification"],
        "canonical_equivalence": ["canonical_trace", "canonical_equivalence"],
        "kernel_coverage": ["kernel_coverage"],
    }
    for gate_name, gate in report.get("gates", {}).items():
        evidence_names = gate.get("evidence", {}).get("artifacts", [])
        if not evidence_names:
            evidence_names = gate_artifacts.get(gate_name, [])
        if gate.get("status") == "running" or any(
            name in stale for name in evidence_names
        ):
            gate["status"] = "stale"
    report["resume"] = {
        "interrupted_stages": interrupted,
        "stale_artifacts": sorted(stale),
        "resumable": True,
        "explanation": (
            "interrupted stages reset to not_started; stale evidence was retained "
            "for diagnosis and excluded from Gates"
        ),
    }
    return report


def _tuple_states(requested: Any) -> list[dict[str, str]]:
    selected = (
        {
            (item.get("target"), item.get("precision"))
            for item in requested
            if isinstance(item, dict)
            and isinstance(item.get("target"), str)
            and isinstance(item.get("precision"), str)
        }
        if isinstance(requested, list)
        else set()
    )
    return [
        {
            "target": target,
            "precision": precision,
            "status": (
                "requested" if (target, precision) in selected else "not_requested"
            ),
        }
        for target, precision in HARDWARE_TUPLES
    ]


class PortingCore:
    """Family-neutral owner of Port lifecycle, Gates, exits, and artifacts."""

    def __init__(
        self,
        request: dict[str, Any],
        pack: FamilyPack,
        environment: dict[str, Any],
        warnings: list[dict[str, str]] | None = None,
    ):
        self.request = request
        self.pack = pack
        self.report = self._base_report(environment, warnings or [])

    def _base_report(
        self,
        environment: dict[str, Any],
        warnings: list[dict[str, str]],
    ) -> dict[str, Any]:
        request = self.request
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "port_id": request.get("port_id"),
            "request_schema_version": request.get("schema_version"),
            "stages": {"intake": "not_started", "preflight": "not_started"},
            "gates": {},
            "exit": {
                "code": SUCCESS.code,
                "category": SUCCESS.category,
                "message": "",
            },
            "request_declarations": {
                "model_family": request.get("model_family"),
                "source": request.get("source"),
                "checkpoint": request.get("checkpoint"),
                "reference": request.get("reference"),
                "capability_contract_version": request.get(
                    "capability_contract_version"
                ),
                "representative_profiles": request.get("representative_profiles"),
                "requested_targets": request.get("requested_targets"),
                "correctness_thresholds": request.get("correctness_thresholds"),
                "tuning_budgets": request.get("tuning_budgets"),
                "environment": request.get("user_environment_declarations", {}),
            },
            "observed_environment": environment,
            "target_precisions": _tuple_states(request.get("requested_targets", [])),
            "issues": [],
            "warnings": warnings,
            "artifacts": {},
        }
        report.update(self.pack.report_defaults(request))
        return report

    @classmethod
    def failed(
        cls,
        request: dict[str, Any],
        pack: FamilyPack,
        outcome: PortOutcome,
        issues: list[dict[str, str]],
        environment: dict[str, Any],
    ) -> "PortingCore":
        core = cls(request, pack, environment)
        core.report["stages"]["intake"] = "failed"
        core.report["issues"] = issues
        core.finish(outcome, issues[0]["message"] if issues else "Intake failed")
        return core

    def start_stage(self, stage: str) -> None:
        self.report["stages"][stage] = "running"

    def pass_stage(self, stage: str) -> None:
        self.report["stages"][stage] = "passed"

    def set_gate(
        self,
        name: str,
        status: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        self.report["gates"][name] = {"status": status, "evidence": dict(evidence or {})}

    def finish(self, outcome: PortOutcome, message: str) -> None:
        self.report["exit"] = {
            "code": outcome.code,
            "category": outcome.category,
            "message": message,
        }

    def add_artifacts(self, artifacts: Mapping[str, dict[str, Any]]) -> None:
        self.report["artifacts"].update(artifacts)


def _vla_report_defaults(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_inspection": {
            "status": "not_configured",
            "adapter_contract_version": None,
        },
        "capability_assessment": {
            "status": "not_configured",
            "contract_version": request.get("capability_contract_version"),
            "supported": 0,
            "canonicalizable": 0,
            "unsupported": 0,
        },
        "canonicalization": {
            "status": "not_started",
            "mode": None,
            "cases": 0,
            "comparisons": 0,
            "failures": 0,
        },
    }


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
    report_defaults=_vla_report_defaults,
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
