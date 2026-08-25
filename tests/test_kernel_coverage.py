import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kernel_coverage import KernelCoverageError, analyze_kernel_coverage  # noqa: E402


class KernelCoverageTest(unittest.TestCase):
    def computation(self, name="projection", operation="linear") -> dict:
        return {
            "id": name,
            "operation": operation,
            "semantics": {"equation": "y = x W^T", "family_role": "action head"},
            "references": [{"case": "control-step/0", "tensor": "action_head.output"}],
            "dtype": "bf16",
            "layout": "row_major",
            "shapes": [[1, 4], [8, 4]],
            "tolerances": {"absolute": 0.001, "relative": 0.01},
            "golden_tensors": ["private/captures/action_head-output.json"],
            "frequency": {"calls_per_inference": 2},
            "performance_impact": {"estimated_latency_fraction": 0.4},
            "expected_interface": "linear(input, weight) -> output",
        }

    def trace(self, family="vla", computations=None) -> dict:
        if computations is None:
            computation = self.computation()
            if family != "vla":
                computation["semantics"]["family_role"] = "token logits"
            computations = [computation]
        return {
            "schema_version": "1.0",
            "family": family,
            "port_id": f"synthetic-{family}",
            "computations": computations,
        }

    def capability(self, classification="existing_primitive", **extra) -> dict:
        capability = {
            "operation": "linear",
            "provenance": "built_in",
            "classification": classification,
            "supported_dtypes": ["bf16"],
            "supported_layouts": ["row_major"],
            "target_shapes": [[1, 4], [8, 4]],
            "interface": "linear(input, weight) -> output",
        }
        capability.update(extra)
        return capability

    def test_vla_and_non_vla_traces_use_the_same_classification_protocol(self) -> None:
        catalog = [self.capability()]

        vla = analyze_kernel_coverage(
            self.trace("vla"), catalog, [{"target": "thor", "precision": "bf16"}]
        )
        llm = analyze_kernel_coverage(
            self.trace("llm"), catalog, [{"target": "thor", "precision": "bf16"}]
        )

        self.assertEqual(vla["status"], "passed")
        self.assertEqual(llm["status"], "passed")
        self.assertEqual(vla["classifications"][0]["classification"], "existing_primitive")
        self.assertEqual(llm["classifications"][0]["semantics"]["family_role"], "token logits")

    def test_every_supported_classification_is_explicit(self) -> None:
        for classification in (
            "existing_fused",
            "existing_primitive",
            "layout_only",
            "correct_fallback",
            "unsupported",
        ):
            with self.subTest(classification=classification):
                computation = self.computation()
                if classification in {"layout_only", "correct_fallback"}:
                    computation["operator_replay"] = {
                        "passed": True,
                        "references": computation["references"],
                        "comparisons": [{
                            "passed": True,
                            "max_absolute": 0.0005,
                            "max_relative": 0.005,
                            "max_tolerance_excess": -0.0005,
                        }],
                    }
                result = analyze_kernel_coverage(
                    self.trace(computations=[computation]),
                    [self.capability(classification)],
                    [{"target": "thor", "precision": "bf16"}],
                )
                self.assertEqual(
                    result["classifications"][0]["classification"], classification
                )

    def test_missing_capability_blocks_with_complete_kernel_handoff(self) -> None:
        result = analyze_kernel_coverage(
            self.trace(), [], [{"target": "thor", "precision": "bf16"}]
        )

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["classifications"][0]["classification"], "missing_required_capability")
        requirement = result["kernel_gaps"][0]
        for field in (
            "semantics", "references", "dtype", "layout", "shapes", "tolerances",
            "golden_tensors", "requested_targets", "frequency", "performance_impact",
            "expected_interface",
        ):
            self.assertIn(field, requirement)

    def test_fallback_is_an_optimization_opportunity_not_a_blocker(self) -> None:
        computation = self.computation()
        computation["operator_replay"] = {
            "passed": True,
            "references": computation["references"],
            "comparisons": [{
                "passed": True,
                "max_absolute": 0.0005,
                "max_relative": 0.005,
                "max_tolerance_excess": -0.0005,
            }],
        }
        result = analyze_kernel_coverage(
            self.trace(computations=[computation]),
            [self.capability("correct_fallback")],
            [{"target": "thor", "precision": "bf16"}],
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["kernel_gaps"], [])
        self.assertEqual(result["optimization_opportunities"][0]["computation_id"], "projection")

    def test_layout_or_fallback_without_operator_replay_becomes_a_kernel_gap(self) -> None:
        for classification in ("layout_only", "correct_fallback"):
            with self.subTest(classification=classification):
                result = analyze_kernel_coverage(
                    self.trace(),
                    [self.capability(classification)],
                    [{"target": "thor", "precision": "bf16"}],
                )
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(
                    result["classifications"][0]["classification"],
                    "missing_required_capability",
                )

    def test_replay_uses_combined_absolute_and_relative_tolerance(self) -> None:
        computation = self.computation()
        computation["operator_replay"] = {
            "passed": True,
            "references": computation["references"],
            "comparisons": [{
                "passed": True,
                "max_absolute": 0.015625,
                "max_relative": 0.12,
                "max_tolerance_excess": -0.004375,
            }],
        }
        result = analyze_kernel_coverage(
            self.trace(computations=[computation]),
            [self.capability("correct_fallback")],
            [{"target": "thor", "precision": "bf16"}],
        )
        self.assertEqual(result["status"], "passed")

    def test_returned_capability_must_be_revalidated_against_original_references(self) -> None:
        returned = self.capability(
            provenance="kernel_workflow_return",
            reference_validation={
                "passed": False,
                "references": [{"case": "control-step/0", "tensor": "action_head.output"}],
            },
        )

        with self.assertRaisesRegex(KernelCoverageError, "revalidated"):
            analyze_kernel_coverage(
                self.trace(), returned and [returned],
                [{"target": "thor", "precision": "bf16"}],
            )

    def test_unclassified_or_incomplete_computation_fails_closed(self) -> None:
        incomplete = self.computation()
        del incomplete["semantics"]
        with self.assertRaisesRegex(KernelCoverageError, "semantics"):
            analyze_kernel_coverage(
                self.trace(computations=[incomplete]), [],
                [{"target": "thor", "precision": "bf16"}],
            )


if __name__ == "__main__":
    unittest.main()
