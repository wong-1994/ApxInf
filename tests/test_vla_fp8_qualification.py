import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vla_fp8_qualification import qualify_vla_port  # noqa: E402


class VlaFp8QualificationTest(unittest.TestCase):
    def fingerprint(self):
        return {
            "device_name": "NVIDIA Thor",
            "sm": 110,
            "multiprocessor_count": 14,
            "kernel_build_id": "build-123",
            "cuda_version": "13.0",
            "library_versions": {"cublas": "13.0.0", "cudnn": "9.8"},
        }

    def request(self, *precisions):
        return {
            "family": "vla",
            "requested_tuples": [
                {"target": "thor", "precision": precision}
                for precision in precisions
            ],
            "reference_evidence": {
                "weights_sha256": "1" * 64,
                "scales_sha256": "2" * 64,
                "calibration_inputs_sha256": "3" * 64,
            },
            "thresholds": {
                "fp8": {
                    "stages": {"vision": {"absolute": 0.08, "relative": 0.04}},
                    "normalized_actions": {"absolute": 0.05, "relative": 0.03},
                    "deployable_actions": {"absolute": 0.02, "relative": 0.01},
                    "minimum_bf16_improvement": 0.1,
                }
            },
        }

    def evidence(self, *precisions):
        fingerprint = self.fingerprint()
        tuples = []
        for precision in precisions:
            item = {
                "target": "thor",
                "precision": precision,
                "target_fingerprint": fingerprint,
                "kernel_coverage": {
                    "schema_version": "1.0",
                    "family": "vla",
                    "status": "passed",
                },
                "correctness": {
                    "status": "passed",
                    "stages": {"vision": {"max_absolute": 0.07, "max_relative": 0.03}},
                    "normalized_actions": {"max_absolute": 0.04, "max_relative": 0.02},
                    "deployable_actions": {"max_absolute": 0.01, "max_relative": 0.009},
                },
                "deployment": {"inference": True, "policy_io": True, "serving": True},
                "performance": {"control_step_p95_ms": 12.0},
            }
            if precision == "fp8":
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

    def test_fp8_only_qualifies_without_inventing_bf16_or_relative_gate(self):
        result = qualify_vla_port(self.request("fp8"), self.evidence("fp8"))

        self.assertEqual(result["status"], "release_qualified")
        self.assertEqual(result["gates"]["fp8_correctness"]["status"], "passed")
        self.assertNotIn("bf16", str(result).lower().replace("no_bf16_relative_gate", ""))
        self.assertEqual(
            result["public_support"],
            [{"family": "vla", "target": "thor", "precision": "fp8"}],
        )

    def test_fp8_intake_requires_weights_scales_and_calibration_evidence(self):
        request = self.request("fp8")
        del request["reference_evidence"]["scales_sha256"]

        with self.assertRaisesRegex(ValueError, "scales_sha256"):
            qualify_vla_port(request, self.evidence("fp8"))

    def test_fp8_thresholds_are_machine_evaluated_for_stages_and_actions(self):
        evidence = self.evidence("fp8")
        evidence[0]["correctness"]["deployable_actions"]["max_absolute"] = 0.021

        result = qualify_vla_port(self.request("fp8"), evidence)

        self.assertEqual(result["status"], "correctness_failed")
        self.assertEqual(result["gates"]["fp8_correctness"]["status"], "failed")
        self.assertEqual(result["public_support"], [])

    def test_fp8_tactics_must_match_the_validated_thor_fingerprint(self):
        evidence = self.evidence("fp8")
        evidence[0]["tuning"]["tactics"]["sm"] = 100

        with self.assertRaisesRegex(ValueError, "incompatible tactic environment"):
            qualify_vla_port(self.request("fp8"), evidence)

    def test_fp8_rejects_a_non_thor_validation_fingerprint(self):
        evidence = self.evidence("fp8")
        evidence[0]["target_fingerprint"]["device_name"] = "NVIDIA Orin"
        evidence[0]["tuning"]["target_fingerprint"]["device_name"] = "NVIDIA Orin"
        evidence[0]["tuning"]["tactics"]["device_name"] = "NVIDIA Orin"

        with self.assertRaisesRegex(ValueError, "Thor fingerprint"):
            qualify_vla_port(self.request("fp8"), evidence)

    def test_fp8_thresholds_cannot_lower_the_family_gate(self):
        request = self.request("fp8")
        request["thresholds"]["fp8"]["deployable_actions"]["absolute"] = 0.5

        with self.assertRaisesRegex(ValueError, "exceeds the VLA FP8 maximum"):
            qualify_vla_port(request, self.evidence("fp8"))

    def test_bf16_and_fp8_add_same_device_improvement_gate(self):
        evidence = self.evidence("bf16", "fp8")
        evidence[0]["performance"]["control_step_p95_ms"] = 15.0

        result = qualify_vla_port(self.request("bf16", "fp8"), evidence)

        self.assertEqual(result["gates"]["fp8_vs_bf16"]["status"], "passed")
        self.assertEqual(result["gates"]["fp8_vs_bf16"]["improvement"], 0.2)
        self.assertEqual(len(result["public_support"]), 2)

    def test_orin_fp8_is_never_qualified_or_advertised(self):
        request = self.request("fp8")
        request["requested_tuples"][0]["target"] = "orin"

        with self.assertRaisesRegex(ValueError, "Orin FP8"):
            qualify_vla_port(request, [])

    def test_public_support_excludes_a_tuple_without_deployment_evidence(self):
        evidence = self.evidence("fp8")
        evidence[0]["deployment"]["serving"] = False

        result = qualify_vla_port(self.request("fp8"), evidence)

        self.assertEqual(result["status"], "correctness_failed")
        self.assertEqual(result["public_support"], [])


if __name__ == "__main__":
    unittest.main()
