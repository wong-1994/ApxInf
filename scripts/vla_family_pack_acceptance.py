#!/usr/bin/env python3
"""Execute the public VLA Family Pack acceptance matrix."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SUPPORTED_TUPLES = frozenset({("thor", "bf16"), ("thor", "fp8"), ("orin", "bf16"), ("orin", "int8_w8a8")})
CONTRACTS = frozenset({"capability", "reference", "canonicalization", "verification", "integration", "serving", "benchmark"})
FORBIDDEN_PUBLIC_KINDS = frozenset({"external_source", "private_adapter", "checkpoint", "real_input", "planning_document", "transient_state"})

class AcceptanceError(ValueError):
    """The executable VLA acceptance matrix did not pass."""

@dataclass(frozen=True)
class AcceptanceCheck:
    name: str
    stages: tuple[str, ...]
    command: tuple[str, ...]

def acceptance_checks(python: str, cargo: str) -> tuple[AcceptanceCheck, ...]:
    """Return executable evidence checks; no lifecycle stage self-reports."""
    return (
        AcceptanceCheck("core_preflight", ("intake", "preflight"), (python, "-m", "unittest", "tests.test_apxinf_port.PortIntakeTest.test_vla_request_and_artifacts_pin_the_family_pack", "tests.test_apxinf_port.PortIntakeTest.test_valid_rewrite_proves_private_canonical_equivalence", "tests.test_apxinf_port.PortIntakeTest.test_unsupported_semantics_block_preflight_with_a_gap_report", "tests.test_apxinf_port.PortIntakeTest.test_true_kernel_gap_blocks_preflight_with_complete_handoff")),
        AcceptanceCheck("minimal_vla_runtime", ("maintained_implementation",), (cargo, "test", "-p", "apxinf-model", "--lib", "minimal_vla::tests")),
        AcceptanceCheck("minimal_vla_policy", ("policy_integration",), (python, "-m", "pytest", "-q", "python/apxinf/tests/test_minimal_vla.py")),
        AcceptanceCheck("pi05_replay", ("existing_vla_replay",), (cargo, "test", "-p", "apxinf-model", "--lib", "pi05::math::tests")),
        AcceptanceCheck("action_serving", ("serving",), (python, "-m", "unittest", "tests.test_pi05_openpi_websocket")),
        AcceptanceCheck("kernel_and_tuning", ("tuning", "kernel_gap", "optimization_opportunity"), (python, "-m", "unittest", "tests.test_kernel_coverage", "tests.test_tuning_workloads")),
        AcceptanceCheck("qualification", ("qualification", "requested_tuple_subset"), (python, "-m", "unittest", "tests.test_qualification", "tests.test_vla_fp8_qualification", "tests.test_vla_int8_qualification")),
        AcceptanceCheck("portable_bundle", ("bundling", "stale_resume"), (python, "-m", "unittest", "tests.test_portable_bundle", "tests.test_porting_core.PortingCoreTest.test_resume_marks_only_changed_dependencies_and_descendants_stale")),
        AcceptanceCheck("publication", ("pr_preparation", "publication_safety"), (python, "-m", "unittest", "tests.test_publication")),
        AcceptanceCheck("deterministic_export", ("deterministic_fixture",), (python, "-m", "unittest", "tests.test_export_minimal_vla")),
    )

def _pairs(value: Any, name: str) -> set[tuple[str, str]]:
    if not isinstance(value, list):
        raise AcceptanceError(f"{name} must be a list")
    pairs: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, Mapping) or not all(isinstance(item.get(field), str) for field in ("target", "precision")):
            raise AcceptanceError(f"{name} entries require target and precision")
        pair = (item["target"], item["precision"])
        if pair in pairs:
            raise AcceptanceError(f"{name} contains duplicate tuple {pair[0]}/{pair[1]}")
        pairs.add(pair)
    return pairs

def validate_manifest(manifest: Mapping[str, Any], repository: Path) -> set[tuple[str, str]]:
    if manifest.get("schema_version") != "1.0" or manifest.get("family") != "vla":
        raise AcceptanceError("VLA acceptance requires schema_version 1.0 and family vla")
    contracts = manifest.get("contracts")
    if not isinstance(contracts, Mapping):
        raise AcceptanceError("contracts must be an object")
    for name in sorted(CONTRACTS):
        path = contracts.get(name)
        if not isinstance(path, str) or not (repository / path).is_file():
            raise AcceptanceError(f"missing executable {name} contract")
    requested = _pairs(manifest.get("requested_tuples"), "requested_tuples")
    if not requested or not requested <= SUPPORTED_TUPLES:
        raise AcceptanceError("requested tuples must be a supported non-empty subset")
    public = manifest.get("public_artifacts")
    if not isinstance(public, list):
        raise AcceptanceError("public_artifacts must be a list")
    for item in public:
        if not isinstance(item, Mapping) or item.get("kind") in FORBIDDEN_PUBLIC_KINDS:
            raise AcceptanceError("public artifact contains forbidden material")
        path = item.get("path")
        if not isinstance(path, str) or not (repository / path).is_file():
            raise AcceptanceError("public artifact must name an existing maintained file")
    if manifest.get("next_production_model") is not None:
        raise AcceptanceError("acceptance must not name the next production VLA model")
    independent = manifest.get("independent_family_development")
    if not isinstance(independent, Mapping) or not all(independent.get(family) is True for family in ("llm", "vlm")):
        raise AcceptanceError("LLM and VLM Family Pack development must remain independent")
    return requested

def run_acceptance(manifest: Mapping[str, Any], repository: Path, *, python: str = sys.executable, cargo: str = "cargo", checks: Sequence[AcceptanceCheck] | None = None) -> dict[str, Any]:
    requested = validate_manifest(manifest, repository)
    selected = tuple(checks) if checks is not None else acceptance_checks(python, cargo)
    results: list[dict[str, Any]] = []
    stages: dict[str, str] = {}
    for check in selected:
        completed = subprocess.run(check.command, cwd=repository, text=True, capture_output=True)
        status = "passed" if completed.returncode == 0 else "failed"
        results.append({"name": check.name, "stages": list(check.stages), "command": list(check.command), "status": status})
        stages.update({stage: status for stage in check.stages})
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            raise AcceptanceError(f"{check.name} failed{suffix}")
    required = {stage for check in acceptance_checks(python, cargo) for stage in check.stages}
    missing = sorted(required - set(stages))
    if missing:
        raise AcceptanceError(f"acceptance evidence is missing stage {missing[0]}")
    return {"schema_version": "1.0", "status": "accepted", "family": "vla", "requested_tuples": [{"target": t, "precision": p} for t, p in sorted(requested)], "unrequested_tuples": [{"target": t, "precision": p} for t, p in sorted(SUPPORTED_TUPLES - requested)], "stages": stages, "checks": results}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_acceptance(json.loads(args.manifest.read_text(encoding="utf-8")), args.repository.resolve(), python=args.python, cargo=args.cargo)
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")

if __name__ == "__main__":
    main()
