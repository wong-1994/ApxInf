import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vla_int8_qualification import qualify_vla_port  # noqa: E402


class VlaInt8QualificationTest(unittest.TestCase):
    def fingerprint(self):
        return {
            "device_name": "NVIDIA Jetson AGX Orin",
            "sm": 87,
            "multiprocessor_count": 16,
            "kernel_build_id": "orin-build-123",
            "cuda_version": "12.6",
            "library_versions": {"cublas": "12.6.4", "cudnn": "9.3"},
        }

    def request(self, *precisions):
        return {
            "family": "vla",
            "requested_tuples": [
                {"target": "orin", "precision": precision}
                for precision in precisions
            ],
            "reference_evidence": {
                "weights_sha256": "1" * 64,
                "scales_sha256": "2" * 64,
                "calibration_inputs_sha256": "3" * 64,
            },
            "thresholds": {
                "int8_w8a8": {
                    "stages": {
                        "action_expert": {"absolute": 0.15, "relative": 0.08}
                    },
                    "normalized_actions": {"absolute": 0.08, "relative": 0.05},
                    "deployable_actions": {"absolute": 0.04, "relative": 0.02},
                    "minimum_bf16_improvement": 0.1,
                }
            },
        }

    def evidence(self, *precisions):
        fingerprint = self.fingerprint()
        tuples = []
        for precision in precisions:
            item = {
                "target": "orin",
                "precision": precision,
                "target_fingerprint": fingerprint,
                "kernel_coverage": {
                    "schema_version": "1.0",
                    "family": "vla",
                    "status": "passed",
                },
                "correctness": {
                    "status": "passed",
                    "stages": {
                        "action_expert": {
                            "max_absolute": 0.14,
                            "max_relative": 0.07,
                        }
                    },
                    "normalized_actions": {
                        "max_absolute": 0.07,
                        "max_relative": 0.04,
                    },
                    "deployable_actions": {
                        "max_absolute": 0.03,
                        "max_relative": 0.015,
                    },
                },
                "deployment": {"inference": True, "policy_io": True, "serving": True},
                "performance": {"control_step_p95_ms": 100.0},
            }
            if precision == "int8_w8a8":
                item["tuning"] = {
                    "schema": "apxinf.tuning.report.v1",
                    "family": "vla",
                    "target_fingerprint": fingerprint,
                    "complete_inference_correctness": True,
                    "tactics": {
                        "schema": "apxinf.cuda.tuning.v1",
                        **fingerprint,
                        "records": [],
                    },
                }
            tuples.append(item)
        return tuples

    def test_int8_only_qualifies_without_inventing_bf16_or_relative_gate(self):
        result = qualify_vla_port(
            self.request("int8_w8a8"), self.evidence("int8_w8a8")
        )

        self.assertEqual(result["status"], "release_qualified")
        self.assertEqual(result["gates"]["int8_correctness"]["status"], "passed")
        self.assertNotIn("int8_vs_bf16", result["gates"])
        self.assertEqual(
            result["public_support"],
            [{"family": "vla", "target": "orin", "precision": "int8_w8a8"}],
        )

    def test_int8_intake_requires_weights_scales_and_calibration_evidence(self):
        request = self.request("int8_w8a8")
        del request["reference_evidence"]["scales_sha256"]

        with self.assertRaisesRegex(ValueError, "scales_sha256"):
            qualify_vla_port(request, self.evidence("int8_w8a8"))

    def test_int8_thresholds_are_machine_evaluated_for_stages_and_actions(self):
        evidence = self.evidence("int8_w8a8")
        evidence[0]["correctness"]["deployable_actions"]["max_absolute"] = 0.041

        result = qualify_vla_port(self.request("int8_w8a8"), evidence)

        self.assertEqual(result["status"], "correctness_failed")
        self.assertEqual(result["gates"]["int8_correctness"]["status"], "failed")
        self.assertEqual(result["public_support"], [])

    def test_int8_uses_family_attributed_kernel_and_tuning_contracts(self):
        evidence = self.evidence("int8_w8a8")
        evidence[0]["tuning"]["family"] = "llm"

        with self.assertRaisesRegex(ValueError, "shared VLA tuning contract"):
            qualify_vla_port(self.request("int8_w8a8"), evidence)

    def test_int8_tactics_must_match_the_validated_orin_fingerprint(self):
        evidence = self.evidence("int8_w8a8")
        evidence[0]["tuning"]["tactics"]["sm"] = 86

        with self.assertRaisesRegex(ValueError, "incompatible tactic environment"):
            qualify_vla_port(self.request("int8_w8a8"), evidence)

    def test_int8_rejects_a_non_orin_validation_fingerprint(self):
        evidence = self.evidence("int8_w8a8")
        evidence[0]["target_fingerprint"]["device_name"] = "NVIDIA Thor"
        evidence[0]["tuning"]["target_fingerprint"]["device_name"] = "NVIDIA Thor"
        evidence[0]["tuning"]["tactics"]["device_name"] = "NVIDIA Thor"

        with self.assertRaisesRegex(ValueError, "Orin fingerprint"):
            qualify_vla_port(self.request("int8_w8a8"), evidence)

    def test_int8_thresholds_cannot_lower_the_family_gate(self):
        request = self.request("int8_w8a8")
        request["thresholds"]["int8_w8a8"]["deployable_actions"]["absolute"] = 0.5

        with self.assertRaisesRegex(ValueError, "exceeds the VLA INT8 maximum"):
            qualify_vla_port(request, self.evidence("int8_w8a8"))

    def test_bf16_and_int8_add_same_device_improvement_gate(self):
        evidence = self.evidence("bf16", "int8_w8a8")
        evidence[0]["performance"]["control_step_p95_ms"] = 125.0

        result = qualify_vla_port(self.request("bf16", "int8_w8a8"), evidence)

        self.assertEqual(result["gates"]["int8_vs_bf16"]["status"], "passed")
        self.assertEqual(result["gates"]["int8_vs_bf16"]["improvement"], 0.2)
        self.assertEqual(len(result["public_support"]), 2)

    def test_fp8_is_never_qualified_or_advertised_on_orin(self):
        with self.assertRaisesRegex(ValueError, "Orin FP8"):
            qualify_vla_port(self.request("fp8"), [])

    def test_public_support_excludes_a_tuple_without_deployment_evidence(self):
        evidence = self.evidence("int8_w8a8")
        evidence[0]["deployment"]["serving"] = False

        result = qualify_vla_port(self.request("int8_w8a8"), evidence)

        self.assertEqual(result["status"], "correctness_failed")
        self.assertEqual(result["public_support"], [])


if __name__ == "__main__":
    unittest.main()
