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
        lifecycle = result["lifecycle_artifacts"]
        self.assertEqual({item["port_id"] for item in lifecycle}, {"synthetic-minimal-vla-v1"})
        self.assertEqual(lifecycle[0]["upstream_sha256"], __import__("hashlib").sha256(json.dumps(self.manifest["acceptance_subject"], sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        for previous, current in zip(lifecycle, lifecycle[1:]):
            self.assertEqual(current["upstream_sha256"], previous["artifact_sha256"])
        replay = result["existing_vla_core_replay"]
        self.assertEqual(replay["family"], "vla")
        self.assertEqual(replay["stage"], "existing_vla_replay")
        self.assertEqual(replay["payload_schema"], "vla-core-replay-v1.schema.json")

    @mock.patch("vla_family_pack_acceptance.subprocess.run")
    def test_command_failure_fails_closed_with_check_name(self, run) -> None:
        run.return_value = mock.Mock(returncode=1, stdout="", stderr="runtime mismatch")
        with self.assertRaisesRegex(AcceptanceError, "core_preflight.*runtime mismatch"):
            run_acceptance(self.manifest, ROOT)

    @mock.patch("vla_family_pack_acceptance.subprocess.run")
    def test_controlled_hardware_is_required_for_accepted_status(self, run) -> None:
        def completed(command, **kwargs):
            command = tuple(command)
            stdout = ""
            if command == ("git", "rev-parse", "HEAD"):
                stdout = "deadbeef\n"
            elif command[1:3] == ("-c", "import json,torch; p=torch.cuda.get_device_properties(0); print(json.dumps({'available':torch.cuda.is_available(),'name':p.name,'capability':[p.major,p.minor],'cuda':torch.version.cuda,'total_memory':p.total_memory}))"):
                stdout = '{"available": true, "name": "NVIDIA Thor", "capability": [11, 0], "cuda": "13.0", "total_memory": 1}\n'
            elif "scripts/bench_pi05.py" in command:
                stdout = "layer                 p50      p95\nL0_model           100.00   110.00\n"
            return mock.Mock(returncode=0, stdout=stdout, stderr="")
        run.side_effect = completed
        result = run_acceptance(
            self.manifest, ROOT, python="python", cargo="cargo",
            runtime_python="runtime-python", controlled_hardware=True,
        )
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["stages"]["controlled_hardware_performance"], "passed")
        self.assertEqual(result["stages"]["controlled_hardware_identity"], "passed")
        self.assertIn(
            "thor_bf16_performance", [check["name"] for check in result["checks"]]
        )
        performance = next(check for check in result["checks"] if check["name"] == "thor_bf16_performance")
        self.assertEqual(performance["evidence"], {"p95_ms": 110.0, "limit_ms": 200.0, "warmup": 1, "samples": 3})

    @mock.patch("vla_family_pack_acceptance.subprocess.run")
    def test_slow_thor_fails_the_machine_evaluated_gate(self, run) -> None:
        def completed(command, **kwargs):
            command = tuple(command)
            if command[1:2] == ("-c",):
                stdout = '{"available": true, "name": "NVIDIA Thor", "capability": [11, 0]}\n'
            elif "scripts/bench_pi05.py" in command:
                stdout = "L0_model           100.00   250.00\n"
            else:
                stdout = ""
            return mock.Mock(returncode=0, stdout=stdout, stderr="")
        run.side_effect = completed
        with self.assertRaisesRegex(AcceptanceError, "exceeds 200.00"):
            run_acceptance(self.manifest, ROOT, controlled_hardware=True)

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
