import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import apxinf_port  # noqa: E402
import reference_adapter_template  # noqa: E402
import kernel_coverage  # noqa: E402


class GrootWorkflowGapTest(unittest.TestCase):
    def test_internal_stochastic_input_may_be_observed_in_output(self) -> None:
        inventory = {"stochastic_inputs": [{"name": "action_noise"}]}
        capture = {
            "profiles": [
                {"profile": "a", "seed": 0, "inputs": {"x": 1}, "output": {"y": 2}},
                {"profile": "a", "seed": 1, "inputs": {"x": 1}, "output": {"y": 3}},
                {"profile": "b", "seed": 0, "inputs": {"x": 2}, "output": {"y": 4}},
                {"profile": "b", "seed": 1, "inputs": {"x": 2}, "output": {"y": 5}},
            ]
        }
        self.assertIsNone(apxinf_port.canonical_input_issue(inventory, capture))

    def test_kernel_capability_may_explicitly_cover_all_shapes(self) -> None:
        computation = {"operation": "aten.linear", "dtype": "bf16", "layout": "row_major", "shapes": [[41, 1536], [1536, 6144]], "expected_interface": "linear(input, weight, bias) -> output"}
        capability = {"operation": "aten.linear", "supported_dtypes": ["bf16"], "supported_layouts": ["row_major"], "target_shapes": ["*"], "interface": "linear(input, weight, bias) -> output"}
        self.assertTrue(kernel_coverage._capability_matches(computation, capability))

    def test_qwen3vl_non_overlapping_conv3d_has_exact_layout_capability(self) -> None:
        catalog = json.loads((ROOT / "contracts/kernel-capabilities-1.0.json").read_text())
        capability = next(item for item in catalog["capabilities"] if item["operation"] == "conv3d")
        self.assertEqual(capability["classification"], "layout_only")
        self.assertEqual(
            capability["target_shapes"],
            [[256, 3, 2, 16, 16], [1024, 3, 2, 16, 16]],
        )
        self.assertIn("bias", capability["interface"])

    def test_generic_sdpa_is_not_claimed_by_the_kernel_catalog(self) -> None:
        catalog = json.loads((ROOT / "contracts/kernel-capabilities-1.0.json").read_text())
        self.assertFalse(
            any(item["operation"] == "scaled_dot_product_attention" for item in catalog["capabilities"])
        )

    def groot_inventory(self) -> dict:
        facts = {
            "shape_profiles": ["finite"],
            "attention": ["scaled_dot_product"],
            "masks": ["causal_and_padding", "bidirectional"],
            "position_encodings": ["rotary", "learned_absolute", "sinusoidal"],
            "normalization": ["rms_norm", "layer_norm"],
            "activations": ["gelu", "silu"],
            "conditioning": ["vision_language_state", "timestep", "noise"],
            "action_heads": ["flow_matching"],
            "schedules": ["euler_flow_matching"],
            "control_flow": ["bounded_unrolled_loop"],
        }
        return {
            "input_schema": [{"profile": "libero", "schema": {}}],
            "normalization": {"model": "rms_norm"},
            "schedules": [{"kind": "euler_flow_matching"}],
            "dynamic_branches": [{"kind": "bounded_unrolled_loop"}],
            "capability_facts": facts,
            "operator_traces": [{"operation": "category_specific_mlp"}],
            "custom_operators": [{
                "name": "category_specific_mlp",
                "semantics": "embodiment-indexed linear weights",
            }],
        }

    def test_vla_2_contract_accepts_composite_groot_semantics(self) -> None:
        contract = apxinf_port.load_capability_contract(
            ROOT / "contracts/vla-capability-contract-2.0.json", "2.0"
        )
        classification = apxinf_port.classify_capabilities(
            self.groot_inventory(), contract
        )
        self.assertEqual(classification["summary"]["unsupported"], 0)
        observed = {
            item["capability"]: item["observed"]
            for item in classification["classifications"]
            if item["path"].startswith("capability_facts.")
        }
        self.assertEqual(observed["masks"], "bidirectional")
        custom = next(
            item for item in classification["classifications"]
            if item["capability"] == "custom_operators"
        )
        self.assertEqual(custom["classification"], "supported")

    def test_untraced_custom_operator_still_fails_closed(self) -> None:
        inventory = self.groot_inventory()
        inventory["operator_traces"] = []
        contract = apxinf_port.load_capability_contract(
            ROOT / "contracts/vla-capability-contract-2.0.json", "2.0"
        )
        classification = apxinf_port.classify_capabilities(inventory, contract)
        custom = next(
            item for item in classification["classifications"]
            if item["capability"] == "custom_operators"
        )
        self.assertEqual(custom["classification"], "unsupported")

    def test_checkpoint_directory_has_stable_content_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"one")
            (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"two")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}})
            )

            first = apxinf_port.path_sha256(checkpoint)
            self.assertEqual(first, apxinf_port.path_sha256(checkpoint))
            (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"changed")
            self.assertNotEqual(first, apxinf_port.path_sha256(checkpoint))

    def test_apxinf_source_identity_supports_git_archives(self) -> None:
        expected = "a" * 64
        failed_git = subprocess.CompletedProcess([], 128, b"", b"not a repository")
        with mock.patch.object(apxinf_port.subprocess, "run", return_value=failed_git), mock.patch.object(
            apxinf_port, "source_sha256", return_value=expected
        ) as fallback:
            self.assertEqual(apxinf_port.apxinf_source_sha256(), expected)
        fallback.assert_called_once_with(apxinf_port.repository_root())

    def test_environment_recorder_does_not_install_from_uv_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "uv.lock"
            lock.write_text(
                'version = 1\n[[package]]\nname = "torch"\n'
                'wheels = [{ url = "https://example.invalid/torch.whl", hash = "sha256:aa" }]\n'
            )
            output = root / "environment.json"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/record_reference_environment.py"), "--dependency-lock", str(lock), "--output", str(output)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(output.read_text())
            self.assertEqual(record["isolation"]["kind"], "agent_prepared")
            self.assertEqual(record["dependency_lock"]["sha256"], hashlib.sha256(lock.read_bytes()).hexdigest())

    def test_tensor_inventory_hashes_raw_bytes_without_float_or_tolist(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        value = torch.arange(32, dtype=torch.bfloat16)
        expected = hashlib.sha256(
            value.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        with mock.patch.object(torch.Tensor, "float", side_effect=AssertionError), mock.patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError
        ):
            record = reference_adapter_template.tensor_record("weight", value)
        self.assertEqual(record["data_sha256"], expected)
        self.assertEqual(record["dtype"], "torch.bfloat16")


if __name__ == "__main__":
    unittest.main()
