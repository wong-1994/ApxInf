import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qualification import QualificationEngine  # noqa: E402
from vla_qualification import VLA_QUALIFICATION_PROFILE  # noqa: E402


class QualificationTest(unittest.TestCase):
    def request(self, *precisions):
        return {
            "family": "vla",
            "representative_real_inputs": True,
            "requested_tuples": [
                {
                    "target": "thor",
                    "precision": precision,
                    "performance_limits": {
                        "control_step_p50_ms": 12.0,
                        "control_step_p95_ms": 15.0,
                    },
                }
                for precision in precisions
            ],
            "minimum_bf16_improvement": {"fp8": 0.1},
            "waivers": [],
        }

    def evidence(self, *precisions):
        result = []
        for precision in precisions:
            result.append(
                {
                    "target": "thor",
                    "precision": precision,
                    "fresh": True,
                    "environment_conforming": True,
                    "environment": {
                        "power_mode": "MAXN",
                        "clocks": "locked",
                        "temperature_c": 48.0,
                        "device": "NVIDIA Thor",
                        "driver": "580.1",
                        "cuda": "13.0",
                        "libraries": {"cublas": "13.0"},
                        "kernel_build": "thor-build-1",
                    },
                    "correctness": {"status": "passed"},
                    "deployment": {
                        "inference": True,
                        "policy_processing": True,
                        "action_serving": True,
                    },
                    "benchmark": {
                        "warmup": 5,
                        "samples": 100,
                        "observation_profile": {"images": [3, 224, 224, 3]},
                        "action_profile": {"shape": [50, 32], "schedule_steps": 10},
                        "workspace_bytes": 1024,
                        "peak_memory_bytes": 2048,
                        "metrics": {
                            "control_step_p50_ms": 10.0,
                            "control_step_p95_ms": 12.0 if precision == "bf16" else 9.0,
                        },
                    },
                }
            )
        return result

    def qualify(self, request, evidence):
        return QualificationEngine(VLA_QUALIFICATION_PROFILE).qualify(
            request, evidence
        )

    def test_release_qualifies_only_requested_tuples_with_complete_vla_evidence(self):
        result = self.qualify(self.request("fp8"), self.evidence("fp8"))

        self.assertEqual(result["status"], "release_qualified")
        self.assertEqual(result["tuples"][0]["status"], "release_qualified")
        self.assertEqual(result["tuples"][0]["benchmark"]["warmup"], 5)
        self.assertNotIn("bf16_improvement", result["tuples"][0]["gates"])

    def test_dual_precision_adds_same_device_bf16_improvement(self):
        result = self.qualify(
            self.request("bf16", "fp8"), self.evidence("bf16", "fp8")
        )

        fp8 = next(item for item in result["tuples"] if item["precision"] == "fp8")
        self.assertEqual(fp8["gates"]["bf16_improvement"]["status"], "passed")
        self.assertEqual(fp8["gates"]["bf16_improvement"]["observed"], 0.25)

    def test_missing_real_inputs_is_provisional(self):
        request = self.request("fp8")
        request["representative_real_inputs"] = False

        result = self.qualify(request, self.evidence("fp8"))

        self.assertEqual(result["status"], "provisional")
        self.assertEqual(result["tuples"][0]["status"], "provisional")

    def test_nonconforming_environment_is_diagnostic_not_release_evidence(self):
        evidence = self.evidence("fp8")
        evidence[0]["environment_conforming"] = False

        result = self.qualify(self.request("fp8"), evidence)

        self.assertEqual(result["status"], "performance_pending")
        self.assertEqual(
            result["tuples"][0]["gates"]["environment"]["status"], "failed"
        )
        self.assertEqual(result["diagnostics"][0]["kind"], "nonconforming_environment")

    def test_deployment_and_performance_status_use_every_requested_tuple(self):
        evidence = self.evidence("bf16", "fp8")
        evidence[1]["deployment"]["action_serving"] = False
        evidence[0]["benchmark"]["metrics"]["control_step_p95_ms"] = 20.0

        result = self.qualify(self.request("bf16", "fp8"), evidence)

        self.assertFalse(result["deployment_complete"])
        self.assertFalse(result["performance_pending"])
        self.assertEqual(result["status"], "incomplete")

    def test_extra_evidence_cannot_affect_requested_tuple_status(self):
        evidence = self.evidence("fp8") + self.evidence("bf16")
        with self.assertRaisesRegex(ValueError, "unrequested tuple"):
            self.qualify(self.request("fp8"), evidence)

    def test_required_benchmark_and_environment_fields_are_enforced(self):
        evidence = self.evidence("fp8")
        del evidence[0]["benchmark"]["peak_memory_bytes"]
        with self.assertRaisesRegex(ValueError, "peak_memory_bytes"):
            self.qualify(self.request("fp8"), evidence)

        evidence = self.evidence("fp8")
        del evidence[0]["environment"]["driver"]
        with self.assertRaisesRegex(ValueError, "driver"):
            self.qualify(self.request("fp8"), evidence)

    def test_family_metrics_remain_opaque_to_common_engine(self):
        class TokenProfile:
            family = "llm"
            lower_precisions = frozenset()

            def validate_request(self, request, pair):
                return None

            def evaluate(self, request_tuple, evidence):
                observed = evidence["benchmark"]["metrics"]["tokens_per_second"]
                required = request_tuple["performance_limits"]["tokens_per_second"]
                return {
                    "tokens_per_second": {
                        "status": "passed" if observed >= required else "failed"
                    }
                }

            def deployment_passed(self, evidence):
                return True

            def comparison_metric(self, evidence):
                return 0.0

        request = {
            "family": "llm",
            "representative_real_inputs": True,
            "requested_tuples": [
                {
                    "target": "thor",
                    "precision": "bf16",
                    "performance_limits": {"tokens_per_second": 20},
                }
            ],
            "waivers": [],
        }
        evidence = self.evidence("bf16")
        evidence[0]["benchmark"]["metrics"] = {"tokens_per_second": 25}

        result = QualificationEngine(TokenProfile()).qualify(request, evidence)
        self.assertEqual(result["status"], "release_qualified")

    def test_waivers_require_maintainer_approval_and_never_release_qualify(self):
        request = self.request("fp8")
        request["waivers"] = [
            {
                "target": "thor",
                "precision": "fp8",
                "gate": "correctness",
                "evidence": "tracked deviation",
                "expires": "2999-12-01",
                "approved_by": "maintainer@example.com",
                "maintainer_approved": False,
            }
        ]
        with self.assertRaisesRegex(ValueError, "maintainer approval"):
            self.qualify(request, self.evidence("fp8"))

        request["waivers"][0]["maintainer_approved"] = True
        evidence = self.evidence("fp8")
        evidence[0]["correctness"]["status"] = "failed"
        result = self.qualify(request, evidence)
        self.assertEqual(result["status"], "qualification_failed")
        self.assertEqual(
            result["tuples"][0]["gates"]["correctness"]["status"], "waived"
        )


if __name__ == "__main__":
    unittest.main()
