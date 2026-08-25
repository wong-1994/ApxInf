#!/usr/bin/env python3
"""Execute the public VLA Family Pack acceptance matrix."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SUPPORTED_TUPLES = frozenset({("thor", "bf16"), ("thor", "fp8"), ("orin", "bf16"), ("orin", "int8_w8a8")})
CONTRACTS = frozenset({"capability", "reference", "canonicalization", "verification", "integration", "serving", "benchmark"})
PUBLIC_KINDS = frozenset({"maintained_source", "synthetic_fixture", "ci_configuration"})
THOR_BF16_L0_P95_LIMIT_MS = 200.0

class AcceptanceError(ValueError):
    """The executable VLA acceptance matrix did not pass."""

@dataclass(frozen=True)
class AcceptanceCheck:
    name: str
    stages: tuple[str, ...]
    command: tuple[str, ...]

def acceptance_checks(
    python: str, cargo: str, runtime_python: str, *, controlled_hardware: bool,
    environment_python: str | None = None, environment_pythonpath: str | None = None,
) -> tuple[AcceptanceCheck, ...]:
    """Return executable evidence checks; no lifecycle stage self-reports."""
    software = (
        AcceptanceCheck("synthetic_vla_lifecycle", ("intake", "preflight", "maintained_implementation", "policy_integration", "serving", "tuning", "qualification", "bundling", "pr_preparation"), (python, "scripts/vla_synthetic_lifecycle.py", "--python", python, "--cargo", cargo, "--repository", ".")),
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
    if not controlled_hardware:
        return software
    environment_command = (
        ("env", f"PYTHONPATH={environment_pythonpath}", environment_python or runtime_python, "scripts/thor_environment.py")
        if environment_pythonpath else
        (environment_python or runtime_python, "scripts/thor_environment.py")
    )
    return software + (
        AcceptanceCheck("thor_identity", ("controlled_hardware_identity",), environment_command),
        AcceptanceCheck("thor_bf16_performance", ("controlled_hardware_performance",), (runtime_python, "scripts/bench_pi05.py", "--random-weights", "--layer", "l0", "--precision", "bf16", "--device", "cuda:0", "--warmup", "1", "--samples", "3")),
    )

def _hardware_evidence(check: AcceptanceCheck, stdout: str) -> dict[str, Any] | None:
    if check.name == "synthetic_vla_lifecycle":
        try:
            evidence = json.loads(stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise AcceptanceError("synthetic lifecycle did not emit valid JSON") from error
        artifacts = evidence.get("artifacts")
        if evidence.get("port_id") != "synthetic-minimal-vla-v1" or not isinstance(artifacts, list):
            raise AcceptanceError("synthetic lifecycle used the wrong Port")
        expected_upstream = _digest_json({"port_id": evidence["port_id"], "model_type": "minimal_vla", "source_kind": "synthetic_external"})
        for index, artifact in enumerate(artifacts):
            if artifact.get("sequence") != index or artifact.get("upstream_sha256") != expected_upstream:
                raise AcceptanceError("synthetic lifecycle artifact chain is not causal")
            payload = dict(artifact)
            claimed = payload.pop("artifact_sha256", None)
            if claimed != _digest_json(payload):
                raise AcceptanceError("synthetic lifecycle artifact digest is invalid")
            expected_upstream = claimed
        return evidence
    if check.name == "thor_identity":
        try:
            identity = json.loads(stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise AcceptanceError("thor_identity did not emit valid JSON") from error
        required = {"driver", "cuda", "libraries", "kernel_build", "power_mode", "clocks_and_temperature"}
        if not required <= identity.keys() or any(not identity[field] for field in required):
            raise AcceptanceError("Thor environment evidence is incomplete")
        if not identity.get("available") or "thor" not in str(identity.get("device", "")).lower():
            raise AcceptanceError("controlled hardware is not NVIDIA Thor")
        if identity.get("capability") != [11, 0]:
            raise AcceptanceError("controlled Thor must report CUDA capability 11.0")
        return identity
    if check.name == "thor_bf16_performance":
        match = re.search(r"^L0_model\s+\S+\s+(\d+(?:\.\d+)?)", stdout, re.MULTILINE)
        if match is None:
            raise AcceptanceError("Thor benchmark did not emit an L0 p95 measurement")
        p95_ms = float(match.group(1))
        if p95_ms > THOR_BF16_L0_P95_LIMIT_MS:
            raise AcceptanceError(
                f"Thor BF16 L0 p95 {p95_ms:.2f} ms exceeds {THOR_BF16_L0_P95_LIMIT_MS:.2f} ms"
            )
        return {"p95_ms": p95_ms, "limit_ms": THOR_BF16_L0_P95_LIMIT_MS, "warmup": 1, "samples": 3}
    return None

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
        candidate = (repository / path).resolve() if isinstance(path, str) else None
        if candidate is None or not candidate.is_relative_to(repository) or not candidate.is_file():
            raise AcceptanceError(f"missing executable {name} contract")
    subject = manifest.get("acceptance_subject")
    if not isinstance(subject, Mapping) or subject != {
        "port_id": "synthetic-minimal-vla-v1",
        "model_type": "minimal_vla",
        "source_kind": "synthetic_external",
    }:
        raise AcceptanceError("acceptance_subject must identify the canonical synthetic minimal VLA")
    requested = _pairs(manifest.get("requested_tuples"), "requested_tuples")
    if not requested or not requested <= SUPPORTED_TUPLES:
        raise AcceptanceError("requested tuples must be a supported non-empty subset")
    public = manifest.get("public_artifacts")
    if not isinstance(public, list):
        raise AcceptanceError("public_artifacts must be a list")
    public_paths: set[str] = set()
    for item in public:
        if not isinstance(item, Mapping) or item.get("kind") not in PUBLIC_KINDS:
            raise AcceptanceError("public artifact kind is not publishable")
        path = item.get("path")
        candidate = (repository / path).resolve() if isinstance(path, str) else None
        if (
            candidate is None
            or not candidate.is_relative_to(repository)
            or not candidate.is_file()
        ):
            raise AcceptanceError("public artifact must name an existing maintained file")
        if path in public_paths:
            raise AcceptanceError("public artifact paths must be unique")
        public_paths.add(path)
    if manifest.get("next_production_model") is not None:
        raise AcceptanceError("acceptance must not name the next production VLA model")
    independent = manifest.get("independent_family_development")
    if not isinstance(independent, Mapping) or not all(independent.get(family) is True for family in ("llm", "vlm")):
        raise AcceptanceError("LLM and VLM Family Pack development must remain independent")
    return requested

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _pi05_core_replay(repository: Path, subject: Mapping[str, str], result: Mapping[str, Any]) -> dict[str, Any]:
    """Record unchanged PI0.5 mathematics as a shared-Core Workflow Artifact."""
    from porting_core import ArtifactStore, VLA_FAMILY_PACK
    with tempfile.TemporaryDirectory(prefix="apxinf-pi05-replay-") as temporary:
        port = Path(temporary)
        private = port / "private"
        private.mkdir()
        source_path = repository / "crates/apxinf-model/src/pi05/math.rs"
        checkpoint_marker = private / "random_weights.json"
        checkpoint_marker.write_text(json.dumps({"kind": "deterministic_random_weights", "seed": 0}), encoding="utf-8")
        request = {
            "schema_version": "1.0", "port_id": subject["port_id"], "model_family": "vla",
            "capability_contract_version": "1.0",
            "source": {"path": str(source_path), "sha256": _sha256(source_path)},
            "checkpoint": {"path": str(checkpoint_marker), "sha256": _sha256(checkpoint_marker)},
        }
        (port / "request.json").write_text(json.dumps(request), encoding="utf-8")
        environment = private / "environment.json"
        environment.write_text(json.dumps({"schema_version": "1.0", "platform": platform.platform(), "python": sys.version}), encoding="utf-8")
        adapter = private / "reference_adapter.py"
        adapter.write_text("# acceptance replay adapter\n", encoding="utf-8")
        payload = private / "pi05_replay.json"
        payload.write_text(json.dumps({
            "schema_version": "1.0", "family": "vla", "port_id": subject["port_id"],
            "model": "pi05", "mathematics": ["prompt_tokenization", "state_discretization", "reverse_time_euler_flow"],
            "check": result["name"], "status": result["status"],
            "evidence_sha256": _digest_json({"command": result["command"], "stdout": result["stdout_sha256"]}),
        }, sort_keys=True), encoding="utf-8")
        return ArtifactStore(port, request, VLA_FAMILY_PACK, repository / "scripts/apxinf_port.py", adapter).record(payload, environment, stage="existing_vla_replay")

def run_acceptance(
    manifest: Mapping[str, Any], repository: Path, *, python: str = sys.executable,
    cargo: str = "cargo", runtime_python: str | None = None,
    controlled_hardware: bool = False, environment_python: str | None = None,
    environment_pythonpath: str | None = None,
) -> dict[str, Any]:
    requested = validate_manifest(manifest, repository)
    selected = acceptance_checks(
        python, cargo, runtime_python or python,
        controlled_hardware=controlled_hardware, environment_python=environment_python,
        environment_pythonpath=environment_pythonpath,
    )
    results: list[dict[str, Any]] = []
    stages: dict[str, str] = {}
    for check in selected:
        environment = os.environ.copy()
        package_root = str(repository / "python" / "apxinf")
        environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
        completed = subprocess.run(
            check.command, cwd=repository, env=environment, text=True, capture_output=True
        )
        status = "passed" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            raise AcceptanceError(f"{check.name} failed{suffix}")
        evidence = _hardware_evidence(check, completed.stdout)
        result = {"name": check.name, "stages": list(check.stages), "command": list(check.command), "status": status, "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest()}
        if evidence is not None:
            result["evidence"] = evidence
        results.append(result)
        stages.update({stage: status for stage in check.stages})
    public_files = {
        item["path"]: _sha256(repository / item["path"])
        for item in manifest["public_artifacts"]
    }
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repository, text=True, capture_output=True, check=True
    ).stdout.strip()
    lifecycle = next(result["evidence"]["artifacts"] for result in results if result["name"] == "synthetic_vla_lifecycle")
    pi05_result = next(result for result in results if result["name"] == "pi05_replay")
    replay = _pi05_core_replay(repository, manifest["acceptance_subject"], pi05_result)
    status = "accepted" if controlled_hardware else "software-validated"
    return {"schema_version": "1.0", "status": status, "family": "vla", "acceptance_subject": dict(manifest["acceptance_subject"]), "requested_tuples": [{"target": t, "precision": p} for t, p in sorted(requested)], "unrequested_tuples": [{"target": t, "precision": p} for t, p in sorted(SUPPORTED_TUPLES - requested)], "stages": stages, "checks": results, "lifecycle_artifacts": lifecycle, "existing_vla_core_replay": replay, "provenance": {"git_commit": commit, "public_file_sha256": public_files, "python": python, "cargo": cargo, "runtime_python": runtime_python or python, "environment_python": environment_python or runtime_python or python, "controlled_hardware": controlled_hardware, "platform": platform.platform()}}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--runtime-python")
    parser.add_argument("--environment-python")
    parser.add_argument("--environment-pythonpath")
    parser.add_argument("--controlled-hardware", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_acceptance(json.loads(args.manifest.read_text(encoding="utf-8")), args.repository.resolve(), python=args.python, cargo=args.cargo, runtime_python=args.runtime_python, controlled_hardware=args.controlled_hardware, environment_python=args.environment_python, environment_pythonpath=args.environment_pythonpath)
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")

if __name__ == "__main__":
    main()
