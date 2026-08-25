import copy
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from vla_family_pack_acceptance import (  # noqa: E402
    AcceptanceError, acceptance_checks, run_acceptance, validate_manifest,
)

class VlaFamilyPackAcceptanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads((ROOT / "tests/fixtures/vla-family-pack-acceptance-v1.json").read_text())

    @mock.patch("vla_family_pack_acceptance.subprocess.run")
    def test_complete_matrix_executes_every_stage_and_contract(self, run) -> None:
        run.side_effect = lambda command, **kwargs: mock.Mock(
            returncode=0,
            stdout="deadbeef\n" if tuple(command) == ("git", "rev-parse", "HEAD") else "",
            stderr="",
        )
        result = run_acceptance(self.manifest, ROOT, python="python", cargo="cargo")
        expected = acceptance_checks("python", "cargo", "python", controlled_hardware=False)
        self.assertEqual(
            [call.args[0] for call in run.call_args_list[:len(expected)]],
            [c.command for c in expected],
        )
        self.assertIn(
            ("git", "rev-parse", "HEAD"),
            [call.args[0] for call in run.call_args_list],
        )
        self.assertEqual(result["status"], "software-validated")
        self.assertTrue(result["stages"])
        self.assertTrue(all(status == "passed" for status in result["stages"].values()))
        self.assertEqual(result["requested_tuples"], [{"target": "thor", "precision": "bf16"}])
        self.assertEqual(result["acceptance_subject"]["port_id"], "synthetic-minimal-vla-v1")

    @mock.patch("vla_family_pack_acceptance.subprocess.run")
    def test_command_failure_fails_closed_with_check_name(self, run) -> None:
        run.return_value = mock.Mock(returncode=1, stdout="", stderr="runtime mismatch")
        with self.assertRaisesRegex(AcceptanceError, "core_preflight.*runtime mismatch"):
            run_acceptance(self.manifest, ROOT)

    @mock.patch("vla_family_pack_acceptance.subprocess.run")
    def test_controlled_hardware_is_required_for_accepted_status(self, run) -> None:
        run.side_effect = lambda command, **kwargs: mock.Mock(
            returncode=0,
            stdout="deadbeef\n" if tuple(command) == ("git", "rev-parse", "HEAD") else "",
            stderr="",
        )
        result = run_acceptance(
            self.manifest, ROOT, python="python", cargo="cargo",
            runtime_python="runtime-python", controlled_hardware=True,
        )
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["stages"]["controlled_hardware_performance"], "passed")
        self.assertIn(
            "thor_bf16_performance", [check["name"] for check in result["checks"]]
        )

    def test_manifest_requires_contracts_requested_subset_and_public_safety(self) -> None:
        for contract in tuple(self.manifest["contracts"]):
            with self.subTest(contract=contract):
                manifest = copy.deepcopy(self.manifest)
                del manifest["contracts"][contract]
                with self.assertRaisesRegex(AcceptanceError, contract):
                    validate_manifest(manifest, ROOT)
        manifest = copy.deepcopy(self.manifest)
        manifest["requested_tuples"].append({"target": "orin", "precision": "fp8"})
        with self.assertRaisesRegex(AcceptanceError, "supported"):
            validate_manifest(manifest, ROOT)
        manifest = copy.deepcopy(self.manifest)
        manifest["public_artifacts"].append({"path": "model.safetensors", "kind": "checkpoint"})
        with self.assertRaisesRegex(AcceptanceError, "not publishable"):
            validate_manifest(manifest, ROOT)

        manifest = copy.deepcopy(self.manifest)
        manifest["public_artifacts"].append({"path": "../outside", "kind": "maintained_source"})
        with self.assertRaisesRegex(AcceptanceError, "existing maintained"):
            validate_manifest(manifest, ROOT)

        manifest = copy.deepcopy(self.manifest)
        manifest["acceptance_subject"]["model_type"] = "self-reported-model"
        with self.assertRaisesRegex(AcceptanceError, "canonical synthetic"):
            validate_manifest(manifest, ROOT)

    def test_acceptance_stays_model_neutral_and_does_not_gate_other_packs(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["next_production_model"] = "some-vla"
        with self.assertRaisesRegex(AcceptanceError, "next production"):
            validate_manifest(manifest, ROOT)
        manifest = copy.deepcopy(self.manifest)
        manifest["independent_family_development"]["llm"] = False
        with self.assertRaisesRegex(AcceptanceError, "independent"):
            validate_manifest(manifest, ROOT)

if __name__ == "__main__":
    unittest.main()
