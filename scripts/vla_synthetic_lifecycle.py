#!/usr/bin/env python3
"""Drive one synthetic VLA Port causally through every acceptance stage."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

PORT_ID = "synthetic-minimal-vla-v1"

def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--cargo", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    python, cargo, repository = args.python, args.cargo, args.repository.resolve()
    stages = (
        ("intake", (python, "-m", "unittest", "tests.test_apxinf_port.PortIntakeTest.test_vla_request_and_artifacts_pin_the_family_pack")),
        ("preflight", (python, "-m", "unittest", "tests.test_apxinf_port.PortIntakeTest.test_valid_rewrite_proves_private_canonical_equivalence")),
        ("maintained_implementation", (cargo, "test", "-p", "apxinf-model", "--lib", "minimal_vla::tests")),
        ("policy_integration", (python, "-m", "pytest", "-q", "python/apxinf/tests/test_minimal_vla.py")),
        ("serving", (python, "-m", "unittest", "tests.test_pi05_openpi_websocket")),
        ("tuning", (python, "-m", "unittest", "tests.test_tuning_workloads")),
        ("qualification", (python, "-m", "unittest", "tests.test_qualification")),
        ("bundling", (python, "-m", "unittest", "tests.test_portable_bundle")),
        ("pr_preparation", (python, "-m", "unittest", "tests.test_publication")),
    )
    artifacts: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="apxinf-vla-lifecycle-") as temporary:
        upstream_path = Path(temporary) / "subject.json"
        subject = {"port_id": PORT_ID, "model_type": "minimal_vla", "source_kind": "synthetic_external"}
        upstream_path.write_text(json.dumps(subject), encoding="utf-8")
        upstream_digest = digest(subject)
        environment = os.environ.copy()
        environment["APXINF_ACCEPTANCE_PORT_ID"] = PORT_ID
        for index, (stage, command) in enumerate(stages):
            consumed = json.loads(upstream_path.read_text(encoding="utf-8"))
            consumed_payload = dict(consumed)
            claimed_digest = consumed_payload.pop("artifact_sha256", None)
            consumed_digest = claimed_digest or digest(consumed_payload)
            if claimed_digest is not None and claimed_digest != digest(consumed_payload):
                raise RuntimeError(f"{stage} received a corrupt upstream artifact")
            if consumed_digest != upstream_digest:
                raise RuntimeError(f"{stage} received a stale upstream artifact")
            completed = subprocess.run(command, cwd=repository, env=environment, text=True, capture_output=True)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip().splitlines()
                raise RuntimeError(f"{stage} failed: {detail[-1] if detail else 'no detail'}")
            artifact = {
                "port_id": PORT_ID, "sequence": index, "stage": stage, "status": "passed",
                "upstream_sha256": consumed_digest, "command": list(command),
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            }
            artifact["artifact_sha256"] = digest(artifact)
            artifact_path = Path(temporary) / f"{index:02d}-{stage}.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            artifacts.append(artifact)
            upstream_path, upstream_digest = artifact_path, artifact["artifact_sha256"]
    print(json.dumps({"schema_version": "1.0", "port_id": PORT_ID, "artifacts": artifacts}, sort_keys=True))

if __name__ == "__main__":
    main()
