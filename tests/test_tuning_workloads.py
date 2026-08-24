import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tuning_workloads import (  # noqa: E402
    GenericGemmTuner,
    WorkloadManifest,
    export_gemm_workloads,
)


TARGET = {
    "device_name": "NVIDIA Thor",
    "sm": 110,
    "multiprocessor_count": 14,
    "kernel_build_id": "build-123",
    "cuda_version": "13.0",
    "library_versions": {"cublas": "13.1", "cudnn": "9.8"},
}


def operation(name, phase, m, repetitions, estimated_ms):
    return {
        "source_operation": name,
        "logical_phase": phase,
        "op": "bf16",
        "m": m,
        "n": 256,
        "k": 128,
        "activation_dtype": "bf16",
        "weight_dtype": "bf16",
        "output_dtype": "bf16",
        "layout": "row_major",
        "scale_mode": "none",
        "epilogue": "none",
        "workspace_limit": 1048576,
        "repetitions": repetitions,
        "estimated_milliseconds_saved": estimated_ms,
        "best_current_milliseconds": estimated_ms,
    }


class WorkloadExportTest(unittest.TestCase):
    def test_family_plan_exports_complete_physical_gemm_workloads(self):
        manifest = export_gemm_workloads(
            {
                "family": "vla",
                "profile": "thor-two-view",
                "target_fingerprint": TARGET,
                "operations": [operation("vision.qkv", "vision", 512, 27, 0.4)],
            }
        )

        self.assertEqual(manifest["schema"], "apxinf.tuning.gemm-workloads.v1")
        self.assertEqual(manifest["tunable_object"], "gemm")
        workload = manifest["workloads"][0]
        self.assertEqual(workload["family"], "vla")
        self.assertEqual(workload["logical_phase"], "vision")
        self.assertEqual(workload["source_operation"], "vision.qkv")
        self.assertEqual(workload["profile"], "thor-two-view")
        self.assertEqual(workload["target_fingerprint"], TARGET)
        self.assertEqual(
            (workload["m"], workload["n"], workload["k"]), (512, 256, 128)
        )
        self.assertEqual(workload["repetitions"], 27)

    def test_vla_and_synthetic_generation_phases_use_the_same_interface(self):
        phases = []
        for family, profile, operations in (
            ("vla", "control", [operation("policy.action", "action", 10, 50, 0.2)]),
            (
                "llm",
                "prompt-128-output-2",
                [
                    operation("transformer.prefill", "prefill", 128, 1, 2.0),
                    operation("transformer.decode", "decode", 1, 2, 0.1),
                ],
            ),
        ):
            manifest = WorkloadManifest.from_execution_plan(
                {
                    "family": family,
                    "profile": profile,
                    "target_fingerprint": TARGET,
                    "operations": operations,
                }
            )
            phases.extend(
                item["logical_phase"]
                for item in manifest.to_dict()["workloads"]
            )
        self.assertEqual(phases, ["action", "prefill", "decode"])

    def test_existing_pi05_schedule_replays_as_physical_workloads(self):
        vision = operation("vision.patch", "vision", 512, 1, 0.4)
        vision.update(n=1152, k=588)
        action = operation("action.qkv", "action", 10, 180, 0.1)
        action.update(n=2560, k=1024)
        exported = export_gemm_workloads(
            {
                "family": "vla",
                "profile": "pi05-thor-two-view",
                "target_fingerprint": TARGET,
                "operations": [vision, action],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workloads.json"
            path.write_text(json.dumps(exported), encoding="utf-8")
            replayed = WorkloadManifest.from_json_file(path).to_dict()
        self.assertEqual(replayed, exported)
        self.assertEqual(
            [
                (item["source_operation"], item["m"], item["n"], item["k"])
                for item in replayed["workloads"]
            ],
            [
                ("vision.patch", 512, 1152, 588),
                ("action.qkv", 10, 2560, 1024),
            ],
        )

    def test_non_gemm_tunable_objects_are_rejected_in_v1(self):
        with self.assertRaisesRegex(ValueError, "v1 supports only GEMM"):
            export_gemm_workloads(
                {
                    "family": "llm",
                    "profile": "decode",
                    "target_fingerprint": TARGET,
                    "tunable_object": "attention",
                    "operations": [],
                }
            )


class GenericTunerTest(unittest.TestCase):
    def test_budget_prioritizes_total_saved_time_and_reports_hotspots(self):
        plan = {
            "family": "vlm",
            "profile": "image-chat",
            "target_fingerprint": TARGET,
            "operations": [
                operation("decode.frequent", "decode", 1, 100, 0.2),
                operation("vision.slow", "vision", 256, 1, 5.0),
                operation("prefill.medium", "prefill", 128, 2, 1.0),
            ],
        }
        manifest = export_gemm_workloads(plan)
        visited = []
        correctness = []

        def benchmark(workload):
            visited.append(workload["source_operation"])
            return {
                "seconds_spent": 0.6,
                "milliseconds": workload["best_current_milliseconds"] / 2,
                "tactic": {"backend": "vendor", "id": len(visited)},
            }

        report = GenericGemmTuner(benchmark).tune(
            manifest,
            budgets={("NVIDIA Thor", "image-chat"): 1.0},
            install_and_verify=lambda tactics, family: correctness.append(
                (tactics, family)
            )
            or True,
        )

        self.assertEqual(visited, ["decode.frequent"])
        self.assertEqual(report["coverage"]["tuned"], 1)
        self.assertEqual(report["coverage"]["total"], 3)
        self.assertEqual(
            report["best_current_results"][0]["source_operation"],
            "decode.frequent",
        )
        self.assertEqual(len(report["best_current_results"]), 3)
        self.assertEqual(
            [item["source_operation"] for item in report["remaining_hotspots"]],
            ["vision.slow", "prefill.medium"],
        )
        self.assertEqual(correctness[0][1], "vlm")
        self.assertTrue(report["complete_inference_correctness"])
        tactic = report["tactics"]["records"][0]
        self.assertEqual(report["tactics"]["schema"], "apxinf.cuda.tuning.v1")
        self.assertEqual(tactic["device"], {"sm": 110, "multiprocessor_count": 14})
        self.assertEqual(tactic["kernel_build_id"], "build-123")
        self.assertEqual(tactic["cuda_version"], "13.0")
        self.assertEqual(
            tactic["library_versions"],
            {"cublas": "13.1", "cudnn": "9.8"},
        )

    def test_tactic_installation_fails_for_an_incompatible_environment(self):
        manifest = export_gemm_workloads(
            {
                "family": "vla",
                "profile": "control",
                "target_fingerprint": TARGET,
                "operations": [operation("action.output", "action", 10, 1, 0.2)],
            }
        )
        tuner = GenericGemmTuner(
            lambda workload: {
                "seconds_spent": 0.1,
                "milliseconds": 0.1,
                "tactic": {"backend": "vendor", "id": 1},
            }
        )
        report = tuner.tune(
            manifest,
            budgets={("NVIDIA Thor", "control"): 1.0},
            install_and_verify=lambda tactics, family: True,
        )
        incompatible = dict(TARGET, cuda_version="13.1")
        with self.assertRaisesRegex(ValueError, "incompatible tactic environment"):
            GenericGemmTuner.validate_tactics(report["tactics"], incompatible)

    def test_correctness_failure_rejects_tuned_results(self):
        manifest = export_gemm_workloads(
            {
                "family": "llm",
                "profile": "decode",
                "target_fingerprint": TARGET,
                "operations": [operation("decode.qkv", "decode", 1, 1, 0.2)],
            }
        )
        tuner = GenericGemmTuner(
            lambda workload: {
                "seconds_spent": 0.1,
                "milliseconds": 0.1,
                "tactic": {"backend": "vendor", "id": 1},
            }
        )
        with self.assertRaisesRegex(RuntimeError, "complete inference correctness"):
            tuner.tune(
                manifest,
                budgets={("NVIDIA Thor", "decode"): 1.0},
                install_and_verify=lambda tactics, family: False,
            )


if __name__ == "__main__":
    unittest.main()
