import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from porting_core import (  # noqa: E402
    INVALID_INPUT,
    SUCCESS,
    ArtifactStore,
    PortingCore,
    VLA_FAMILY_PACK,
)


class PortingCoreTest(unittest.TestCase):
    def request(self) -> dict:
        return {
            "schema_version": "1.0",
            "port_id": "example-port",
            "model_family": "vla",
            "capability_contract_version": "1.0",
            "source": {"sha256": "1" * 64},
            "checkpoint": {"sha256": "2" * 64},
            "reference": {},
            "representative_profiles": [],
            "requested_targets": [{"target": "thor", "precision": "bf16"}],
            "correctness_thresholds": {},
            "tuning_budgets": [],
        }

    def test_core_owns_lifecycle_gates_exit_and_requested_tuples(self) -> None:
        core = PortingCore(self.request(), VLA_FAMILY_PACK, {"python": "test"})

        core.pass_stage("intake")
        core.start_stage("preflight")
        core.set_gate("capability", "passed", {"supported": 10})
        core.finish(SUCCESS, "Preflight passed")

        self.assertEqual(core.report["stages"], {"intake": "passed", "preflight": "running"})
        self.assertEqual(core.report["gates"]["capability"]["status"], "passed")
        self.assertEqual(core.report["exit"]["category"], "success")
        requested = [
            item
            for item in core.report["target_precisions"]
            if item["status"] == "requested"
        ]
        self.assertEqual(
            requested,
            [{"target": "thor", "precision": "bf16", "status": "requested"}],
        )

    def test_core_builds_family_neutral_failed_report(self) -> None:
        core = PortingCore.failed(
            {},
            VLA_FAMILY_PACK,
            INVALID_INPUT,
            [{"path": "request", "message": "bad"}],
            {},
        )

        self.assertEqual(core.report["stages"]["intake"], "failed")
        self.assertEqual(core.report["issues"][0]["message"], "bad")
        self.assertEqual(core.report["exit"]["code"], 3)

    def test_artifact_store_validates_payload_schema_and_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port_dir = Path(temporary)
            private = port_dir / "private"
            private.mkdir()
            request_path = port_dir / "request.json"
            request_path.write_text(json.dumps(self.request()), encoding="utf-8")
            adapter = private / "reference_adapter.py"
            adapter.write_text("pass\n", encoding="utf-8")
            environment = private / "environment.json"
            environment.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
            inventory = private / "source_inventory.json"
            inventory.write_text(json.dumps({"schema_version": "9.9"}), encoding="utf-8")
            store = ArtifactStore(
                port_dir,
                self.request(),
                VLA_FAMILY_PACK,
                ROOT / "scripts/apxinf_port.py",
                adapter,
            )

            with self.assertRaisesRegex(ValueError, "schema"):
                store.record(inventory, environment)
