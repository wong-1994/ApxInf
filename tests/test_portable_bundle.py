import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMAND = ROOT / "scripts" / "apxinf_port.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PortableBundleTest(unittest.TestCase):
    def run_port(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(COMMAND), *args], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )

    def make_port(self, root: Path, target: str, environment: str) -> Path:
        port = root / f"port-{target}"
        private = port / "private"
        private.mkdir(parents=True)
        request = {
            "schema_version": "1.0", "port_id": "portable-vla-1",
            "model_family": "vla", "capability_contract_version": "1.0",
            "source": {"path": "/secret/source", "revision": "abc", "sha256": "1" * 64},
            "checkpoint": {"path": "/secret/model.ckpt", "sha256": "2" * 64},
            "requested_targets": [
                {"target": "thor", "precision": "bf16"},
                {"target": "orin", "precision": "bf16"},
            ],
        }
        (port / "request.json").write_text(json.dumps(request), encoding="utf-8")
        payload = private / f"qualification-{target}.json"
        payload.write_text(json.dumps({
            "family": "vla", "target": target, "precision": "bf16",
            "environment_fingerprint": environment, "status": "passed",
        }), encoding="utf-8")
        artifact = {
            "envelope_version": "1.0", "family": "vla",
            "capability_contract_version": "1.0", "stage": "qualification",
            "payload_schema": "qualification-v1", "path": f"private/{payload.name}",
            "dependency_paths": {"environment": "/private/machine/environment.json"},
            "state": "current", "explanation": {"changed_dependencies": [], "upstream_stale": []},
            "fingerprints": {
                "content_sha256": digest(payload), "tool_sha256": {"orchestrator": "3" * 64},
                "source_sha256": "1" * 64, "checkpoint_sha256": "2" * 64,
                "apxinf_source_sha256": "4" * 64, "kernel_build_sha256": "5" * 64,
                "environment_sha256": environment, "capability_contract_sha256": "6" * 64,
                "documentation_sha256": None,
                "target_environment_sha256": {f"{target}/bf16": environment},
                "upstream_sha256": {"request": digest(port / "request.json")},
            },
        }
        report = {"port_id": "portable-vla-1", "artifacts": {f"qualification_{target}": artifact}}
        (port / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (private / "reference_adapter.py").write_text("# private adapter\n", encoding="utf-8")
        (private / "real-input.json").write_text('{"token":"credential"}', encoding="utf-8")
        (port / "model.ckpt").write_bytes(b"checkpoint")
        return port

    def test_bundle_is_hashed_sanitized_and_merges_cross_machine_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            thor = self.make_port(root, "thor", "a" * 64)
            orin = self.make_port(root, "orin", "b" * 64)
            bundle = root / "thor-bundle"

            created = self.run_port("bundle", "--port-dir", str(thor), "--output", str(bundle))
            self.assertEqual(created.returncode, 0, created.stderr)
            manifest = json.loads((bundle / "manifest.json").read_text())
            self.assertEqual(manifest["port"]["id"], "portable-vla-1")
            self.assertIn("artifacts/qualification_thor.json", manifest["files"])
            self.assertIn("private/reference_adapter.py", manifest["files"])
            serialized = json.dumps(manifest)
            self.assertNotIn("/secret/", serialized)
            self.assertNotIn("/private/machine", serialized)
            self.assertFalse((bundle / "model.ckpt").exists())
            self.assertFalse((bundle / "private/real-input.json").exists())

            merged = self.run_port("merge-bundle", "--port-dir", str(orin), "--bundle", str(bundle))
            self.assertEqual(merged.returncode, 0, merged.stderr)
            report = json.loads((orin / "report.json").read_text())
            self.assertEqual(
                {(item["target"], item["precision"]) for item in report["portable_evidence"]},
                {("orin", "bf16"), ("thor", "bf16")},
            )

    def test_merge_rejects_tampering_stale_evidence_and_environment_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_port(root, "thor", "a" * 64)
            destination = self.make_port(root, "orin", "b" * 64)
            for mutation, message in (
                ("tamper", "content hash"), ("stale", "stale"), ("environment", "environment"),
            ):
                bundle = root / f"bundle-{mutation}"
                self.assertEqual(self.run_port("bundle", "--port-dir", str(source), "--output", str(bundle)).returncode, 0)
                manifest_path = bundle / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                if mutation == "tamper":
                    (bundle / "artifacts/qualification_thor.json").write_text("tampered")
                else:
                    envelope = manifest["artifacts"]["qualification_thor"]
                    if mutation == "stale":
                        envelope["state"] = "stale"
                    else:
                        envelope["fingerprints"]["target_environment_sha256"]["thor/bf16"] = "c" * 64
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                result = self.run_port("merge-bundle", "--port-dir", str(destination), "--bundle", str(bundle))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_cleanup_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port = self.make_port(Path(temporary), "thor", "a" * 64)
            result = self.run_port("cleanup", "--port-dir", str(port))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(port.exists())
            self.assertIn("retained", result.stdout)

    def test_merge_detects_conflicts_missing_dependencies_and_family_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_port(root, "thor", "a" * 64)
            for mutation, message in (
                ("conflict", "conflicts"),
                ("dependency", "missing dependencies"),
                ("family", "family-payload mismatch"),
            ):
                destination = self.make_port(root / mutation, "orin", "b" * 64)
                bundle = root / f"bundle-{mutation}"
                self.assertEqual(
                    self.run_port(
                        "bundle", "--port-dir", str(source), "--output", str(bundle)
                    ).returncode,
                    0,
                )
                manifest_path = bundle / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                envelope = manifest["artifacts"]["qualification_thor"]
                if mutation == "conflict":
                    report_path = destination / "report.json"
                    report = json.loads(report_path.read_text())
                    report["artifacts"]["qualification_thor"] = report["artifacts"][
                        "qualification_orin"
                    ]
                    report_path.write_text(json.dumps(report), encoding="utf-8")
                elif mutation == "dependency":
                    envelope["fingerprints"]["upstream_sha256"]["missing"] = "9" * 64
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                else:
                    envelope["family"] = "llm"
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                result = self.run_port(
                    "merge-bundle", "--port-dir", str(destination),
                    "--bundle", str(bundle),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
