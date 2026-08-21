import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMAND = ROOT / "scripts" / "apxinf_port.py"


class PortIntakeTest(unittest.TestCase):
    def run_port(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(COMMAND), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def initialize_port(self, root: Path) -> Path:
        source = root / "trusted-source"
        checkpoint = root / "model.ckpt"
        port = root / "private-port"
        source.mkdir()
        (source / "model.py").write_text("class Model: pass\n", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint bytes")

        result = self.run_port(
            "init",
            "--source",
            str(source),
            "--source-revision",
            "0123456789abcdef",
            "--checkpoint",
            str(checkpoint),
            "--port-dir",
            str(port),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return port

    def complete_request(
        self, port: Path, target: str = "thor", precision: str = "bf16"
    ) -> None:
        request_path = port / "request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["representative_profiles"] = [
            {
                "name": "control-step",
                "inputs": {"camera": [1, 224, 224, 3], "tokens": [1, 32]},
            }
        ]
        request["requested_targets"] = [
            {
                "target": target,
                "precision": precision,
                "latency_goal": {"p50_ms": 80.0, "p95_ms": 90.0},
            }
        ]
        request["correctness_thresholds"] = {
            "absolute": 0.001,
            "relative": 0.01,
        }
        request["tuning_budgets"] = [{"target": target, "seconds": 300}]
        request["user_environment_declarations"] = {"power_mode": "MAXN"}
        request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")

    def test_valid_subset_marks_only_selected_tuple_as_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            port = self.initialize_port(root)
            source_file = root / "trusted-source" / "model.py"
            source_before = source_file.read_bytes()
            self.complete_request(port)

            result = self.run_port("run", "--port-dir", str(port))
            self.assertEqual(result.returncode, 0, result.stderr)

            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["stages"]["intake"], "passed")
            self.assertEqual(report["exit"]["category"], "success")
            self.assertEqual(
                report["target_precisions"],
                [
                    {"target": "thor", "precision": "bf16", "status": "requested"},
                    {"target": "thor", "precision": "fp8", "status": "not_requested"},
                    {"target": "orin", "precision": "bf16", "status": "not_requested"},
                    {
                        "target": "orin",
                        "precision": "int8_w8a8",
                        "status": "not_requested",
                    },
                ],
            )
            self.assertIsInstance(report["observed_environment"]["os"], str)
            self.assertIsInstance(report["observed_environment"]["arch"], str)
            self.assertEqual(
                report["request_declarations"]["environment"],
                {"power_mode": "MAXN"},
            )
            self.assertNotIn("power_mode", report["observed_environment"])
            self.assertEqual(source_file.read_bytes(), source_before)

    def test_initialization_records_provenance_and_all_request_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            port = self.initialize_port(root)
            request = json.loads((port / "request.json").read_text(encoding="utf-8"))

            self.assertEqual(request["schema_version"], "1.0")
            self.assertEqual(request["source"]["revision"], "0123456789abcdef")
            self.assertEqual(len(request["source"]["sha256"]), 64)
            self.assertEqual(
                request["checkpoint"]["sha256"],
                hashlib.sha256(b"checkpoint bytes").hexdigest(),
            )
            self.assertEqual(
                request["representative_profiles"],
                [{"name": None, "inputs": {}}],
            )
            self.assertEqual(
                request["requested_targets"],
                [
                    {
                        "target": None,
                        "precision": None,
                        "latency_goal": {"p50_ms": None, "p95_ms": None},
                    }
                ],
            )
            self.assertEqual(
                request["correctness_thresholds"],
                {"absolute": None, "relative": None},
            )
            self.assertEqual(
                request["tuning_budgets"],
                [{"target": None, "seconds": None}],
            )

    def test_initialization_refuses_to_write_into_the_source_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "trusted-source"
            checkpoint = root / "model.ckpt"
            port = source / "port-state"
            source.mkdir()
            (source / "model.py").write_text("class Model: pass\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint bytes")

            result = self.run_port(
                "init",
                "--source",
                str(source),
                "--source-revision",
                "0123456789abcdef",
                "--checkpoint",
                str(checkpoint),
                "--port-dir",
                str(port),
            )

            self.assertEqual(result.returncode, 3)
            self.assertFalse(port.exists())
            self.assertEqual(
                (source / "model.py").read_text(encoding="utf-8"),
                "class Model: pass\n",
            )

    def test_run_refuses_to_write_into_the_source_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            port = self.initialize_port(root)
            self.complete_request(port)
            unsafe_port = root / "trusted-source" / "port-state"
            unsafe_port.mkdir()
            shutil.copy(port / "request.json", unsafe_port / "request.json")

            result = self.run_port("run", "--port-dir", str(unsafe_port))

            self.assertEqual(result.returncode, 3)
            self.assertFalse((unsafe_port / "report.json").exists())

    def test_intake_rejects_checkpoint_that_changed_after_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            port = self.initialize_port(root)
            self.complete_request(port)
            (root / "model.ckpt").write_bytes(b"replacement checkpoint")

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 3)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["exit"]["category"], "invalid_input")
            self.assertEqual(report["issues"][0]["path"], "checkpoint.sha256")

    def test_intake_rejects_source_that_changed_after_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            port = self.initialize_port(root)
            self.complete_request(port)
            (root / "trusted-source" / "model.py").write_text(
                "class ChangedModel: pass\n", encoding="utf-8"
            )

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 3)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["exit"]["category"], "invalid_input")
            self.assertEqual(report["issues"][0]["path"], "source.sha256")

    def test_intake_reports_missing_source_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            port = self.initialize_port(root)
            self.complete_request(port)
            shutil.rmtree(root / "trusted-source")

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 2)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["exit"]["category"], "missing_input")
            self.assertEqual(report["issues"][0]["path"], "source.path")

    def test_missing_tuning_budget_warns_without_blocking_intake(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port = self.initialize_port(Path(temporary))
            self.complete_request(port, target="orin", precision="bf16")
            request_path = port / "request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["tuning_budgets"] = [{"target": "thor", "seconds": 300}]
            request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 0)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["exit"]["category"], "success")
            self.assertIn(
                "tuning_budgets[orin]",
                {warning["path"] for warning in report["warnings"]},
            )

    def test_missing_performance_goals_do_not_block_requested_port(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port = self.initialize_port(Path(temporary))
            self.complete_request(port)
            request_path = port / "request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["requested_targets"][0]["latency_goal"] = {
                "p50_ms": None,
                "p95_ms": None,
            }
            request["tuning_budgets"] = []
            request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["stages"]["intake"], "passed")
            self.assertEqual(report["exit"]["category"], "success")
            self.assertIn("warnings", report)
            warning_paths = {warning["path"] for warning in report["warnings"]}
            self.assertIn("requested_targets[0].latency_goal.p50_ms", warning_paths)
            self.assertIn("requested_targets[0].latency_goal.p95_ms", warning_paths)
            self.assertIn("tuning_budgets", warning_paths)
            self.assertEqual(report["target_precisions"][0]["status"], "requested")

    def test_intake_rejects_non_finite_numbers_and_keeps_report_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port = self.initialize_port(Path(temporary))
            self.complete_request(port)
            request_path = port / "request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["requested_targets"][0]["latency_goal"]["p50_ms"] = math.nan
            request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 3)
            report_text = (port / "report.json").read_text(encoding="utf-8")

            def reject_non_json_number(value: str) -> None:
                raise ValueError(f"non-JSON number: {value}")

            report = json.loads(report_text, parse_constant=reject_non_json_number)
            self.assertEqual(report["exit"]["category"], "invalid_input")

    def test_unhashable_enum_values_produce_an_invalid_input_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port = self.initialize_port(Path(temporary))
            self.complete_request(port)
            request_path = port / "request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["requested_targets"][0]["target"] = []
            request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 3)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["exit"]["category"], "invalid_input")
            self.assertEqual(report["issues"][0]["path"], "requested_targets[0].target")

    def test_non_finite_user_declaration_still_writes_a_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port = self.initialize_port(Path(temporary))
            self.complete_request(port)
            request_path = port / "request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["user_environment_declarations"] = {"temperature": 1.0}
            request_text = json.dumps(request, indent=2) + "\n"
            request_path.write_text(
                request_text.replace('"temperature": 1.0', '"temperature": 1e999'),
                encoding="utf-8",
            )

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 3)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["exit"]["category"], "invalid_input")

    def test_all_supported_target_precision_tuples_pass_intake(self) -> None:
        supported = (
            ("thor", "bf16"),
            ("thor", "fp8"),
            ("orin", "bf16"),
            ("orin", "int8_w8a8"),
        )
        for target, precision in supported:
            with self.subTest(target=target, precision=precision):
                with tempfile.TemporaryDirectory() as temporary:
                    port = self.initialize_port(Path(temporary))
                    self.complete_request(port, target=target, precision=precision)
                    result = self.run_port("run", "--port-dir", str(port))
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_report_command_prints_the_last_structured_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port = self.initialize_port(Path(temporary))
            self.complete_request(port)
            self.assertEqual(
                self.run_port("run", "--port-dir", str(port)).returncode,
                0,
            )

            result = self.run_port("report", "--port-dir", str(port))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(result.stdout),
                json.loads((port / "report.json").read_text(encoding="utf-8")),
            )

    def test_orin_fp8_is_rejected_with_a_structured_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port = self.initialize_port(Path(temporary))
            self.complete_request(port, target="orin", precision="fp8")

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 4)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["stages"]["intake"], "failed")
            self.assertEqual(report["exit"]["category"], "unsupported_target")
            self.assertEqual(
                report["issues"],
                [
                    {
                        "path": "requested_targets[0]",
                        "message": "orin does not support fp8; use bf16 or int8_w8a8",
                    }
                ],
            )

    def test_missing_facts_pass_intake_with_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port = self.initialize_port(Path(temporary))

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 0)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["stages"]["intake"], "passed")
            self.assertEqual(report["exit"]["category"], "success")
            self.assertEqual(report["issues"], [])
            self.assertEqual(
                {warning["path"] for warning in report["warnings"]},
                {
                    "representative_profiles[0].name",
                    "representative_profiles[0].inputs",
                    "requested_targets[0].target",
                    "requested_targets[0].precision",
                    "requested_targets[0].latency_goal.p50_ms",
                    "requested_targets[0].latency_goal.p95_ms",
                    "correctness_thresholds.absolute",
                    "correctness_thresholds.relative",
                    "tuning_budgets[0].target",
                    "tuning_budgets[0].seconds",
                },
            )

    def test_initialization_allows_source_checkpoint_and_revision_to_be_omitted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port = Path(temporary) / "private-port"

            result = self.run_port("init", "--port-dir", str(port))

            self.assertEqual(result.returncode, 0, result.stderr)
            request = json.loads((port / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(
                request["source"],
                {"path": None, "revision": None, "sha256": None},
            )
            self.assertEqual(
                request["checkpoint"],
                {"path": None, "sha256": None},
            )
            run = self.run_port("run", "--port-dir", str(port))
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_malformed_request_still_writes_invalid_input_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port = self.initialize_port(Path(temporary))
            (port / "request.json").write_text("{not json\n", encoding="utf-8")

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 3)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["stages"]["intake"], "failed")
            self.assertEqual(report["exit"]["category"], "invalid_input")
            self.assertEqual(report["issues"][0]["path"], "request.json")

    def test_request_schema_rejects_wrong_types_and_latency_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port = self.initialize_port(Path(temporary))
            self.complete_request(port)
            request_path = port / "request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["schema_version"] = 2
            request["port_id"] = 42
            request["requested_targets"][0]["latency_goal"] = {
                "p50_ms": 90.0,
                "p95_ms": 80.0,
            }
            request["correctness_thresholds"]["absolute"] = "close enough"
            request["unexpected"] = True
            request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 3)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["exit"]["category"], "invalid_input")
            self.assertIsNone(report["port_id"])
            self.assertIsNone(report["request_schema_version"])
            self.assertEqual(
                {issue["path"] for issue in report["issues"]},
                {
                    "schema_version",
                    "port_id",
                    "requested_targets[0].latency_goal",
                    "correctness_thresholds.absolute",
                    "$.unexpected",
                },
            )


if __name__ == "__main__":
    unittest.main()
