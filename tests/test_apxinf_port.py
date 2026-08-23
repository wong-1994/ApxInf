import copy
import hashlib
import importlib.util
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

    def initialize_inspectable_port(
        self, root: Path, *, failure=None, lock_text: str = ""
    ) -> tuple[Path, Path]:
        source = root / "trusted-source"
        checkpoint = root / "model.ckpt"
        port = root / "private-port"
        source.mkdir()
        package = source / "reference_pkg"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (source / "requirements.lock").write_text(lock_text, encoding="utf-8")
        (package / "model_support.py").write_text(
            "MODEL_VALUE = 3.0\n", encoding="utf-8"
        )
        (package / "reference_impl.py").write_text(
            """
FAILURE = __FAILURE__

import sys
from .model_support import MODEL_VALUE


class Storage:
    def data_ptr(self):
        return id(self)


class Tensor:
    def __init__(self, shape, value, storage=None, offset=0):
        self.shape = shape
        self.dtype = "float32"
        self.value = value
        self.requires_grad = False
        self._storage = storage or Storage()
        self._offset = offset

    def data_ptr(self):
        return self._storage.data_ptr() + self._offset

    def untyped_storage(self):
        return self._storage

    def storage_offset(self):
        return self._offset

    def stride(self):
        return tuple(reversed(range(1, len(self.shape) + 1)))

    def detach(self):
        return self

    def float(self):
        self.dtype = "float32"
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class Block:
    pass


class Model:
    def __init__(self):
        self.shared = Tensor((2, 2), MODEL_VALUE)
        self.shared_view = Tensor(
            (1, 2), [[MODEL_VALUE, MODEL_VALUE]], self.shared._storage, offset=2
        )
        self.shared_alias = Tensor(
            (2, 2), MODEL_VALUE, self.shared._storage, offset=0
        )
        self.position = Tensor((4,), 1.0)

    def named_modules(self):
        return [("", self), ("encoder", Block())]

    def named_parameters(self, remove_duplicate=False):
        return [
            ("encoder.weight", self.shared),
            ("action_head.weight", self.shared),
            ("decoder.weight_alias", self.shared_alias),
            ("encoder.weight_view", self.shared_view),
        ]

    def named_buffers(self, remove_duplicate=False):
        return [("position_ids", self.position)]


def load(checkpoint_path):
    assert sys.prefix != sys.base_prefix
    if FAILURE == "load":
        raise RuntimeError("synthetic load failed")
    assert checkpoint_path.endswith("model.ckpt")
    return Model()


def preprocess(profile):
    assert profile["name"] == "control-step"
    return {
        "tokens": Tensor((1, 2), [[1.0, 2.0]]),
        "noise": Tensor((1, 2), [[0.25, -0.25]]),
    }


def infer(model, inputs):
    if FAILURE == "network":
        import socket
        socket.create_connection(("example.com", 443))
    assert model.shared.value == 3.0
    return {"actions": Tensor((1, 2), [[inputs["noise"].value[0][0], 2.0]])}


def capture_intermediates(model, inputs):
    if FAILURE == "trace":
        raise RuntimeError("synthetic trace failed")
    return {
        "encoder.output": Tensor(
            (1, 2), [[model.shared.value, inputs["tokens"].value[0][0]]]
        )
    }


def postprocess(output):
    return {
        "actions": Tensor(
            (1, 2), [[output["actions"].value[0][0] * 2.0, 4.0]]
        )
    }


def describe():
    description = {
        "operator_traces": [{"name": "action_head", "operator": "aten.linear"}],
        "preprocessing": {"images": "uint8_to_float32"},
        "tokenization": {"kind": "synthetic_ids"},
        "normalization": {"actions": "q01_q99"},
        "stochastic_inputs": [{"name": "noise", "distribution": "normal"}],
        "schedules": [{"name": "flow", "steps": 2}],
        "custom_operators": [],
        "dynamic_branches": [],
        "capability_facts": {
            "shape_profiles": ["finite"],
            "attention": ["scaled_dot_product"],
            "masks": ["causal_and_padding"],
            "position_encodings": ["rotary"],
            "normalization": ["rms_norm"],
            "activations": ["gelu"],
            "conditioning": ["vision_language_state"],
            "action_heads": ["flow_matching"],
            "schedules": ["euler_flow_matching"],
            "control_flow": ["static"],
        },
    }
    if FAILURE == "invalid_inventory":
        description["preprocessing"] = []
    elif FAILURE == "canonicalizable_attention":
        description["capability_facts"]["attention"] = [
            "separate_qkv_scaled_dot_product"
        ]
    elif FAILURE == "unsupported_attention":
        description["capability_facts"]["attention"] = ["linear_attention"]
    elif FAILURE == "unknown_masks":
        del description["capability_facts"]["masks"]
    elif FAILURE == "missing_capability_facts":
        del description["capability_facts"]
    elif FAILURE == "contradictory_normalization":
        description["capability_facts"]["normalization"] = [
            "rms_norm",
            "layer_norm",
        ]
    elif FAILURE == "unexplained_control_flow":
        description["dynamic_branches"] = [{"name": "data_dependent_router"}]
    return description


if FAILURE == "missing_description":
    del describe
""".replace("__FAILURE__", repr(failure)).lstrip(),
            encoding="utf-8",
        )
        checkpoint.write_bytes(b"checkpoint bytes")
        result = self.run_port(
            "init",
            "--source",
            str(source),
            "--source-revision",
            "0123456789abcdef",
            "--checkpoint",
            str(checkpoint),
            "--reference-entrypoint",
            "reference_pkg/reference_impl.py",
            "--dependency-lock",
            "requirements.lock",
            "--port-dir",
            str(port),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.complete_request(port)
        return port, source

    def test_run_inspects_trusted_source_through_private_reference_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            port, source = self.initialize_inspectable_port(root)
            source_digest = hashlib.sha256(
                (source / "reference_pkg/reference_impl.py").read_bytes()
            ).hexdigest()

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["stages"]["intake"], "passed")
            self.assertEqual(report["stages"]["preflight"], "passed")
            self.assertEqual(report["reference_inspection"]["status"], "passed")
            self.assertEqual(
                report["capability_assessment"],
                {
                    "status": "passed",
                    "contract_version": "1.0",
                    "supported": 10,
                    "canonicalizable": 0,
                    "unsupported": 0,
                },
            )
            self.assertEqual(report["exit"]["category"], "success")

            artifacts = report["artifacts"]
            adapter_path = port / artifacts["reference_adapter"]["path"]
            inventory_path = port / artifacts["source_inventory"]["path"]
            environment_path = port / artifacts["reference_environment"]["path"]
            capture_path = port / artifacts["private_capture"]["path"]
            classification_path = port / artifacts["capability_classification"]["path"]
            for artifact_path in (
                adapter_path,
                inventory_path,
                environment_path,
                capture_path,
                classification_path,
            ):
                self.assertTrue(artifact_path.is_file(), artifact_path)
            for artifact in artifacts.values():
                self.assertEqual(
                    set(artifact["fingerprints"]),
                    {
                        "content_sha256",
                        "tool_sha256",
                        "source_sha256",
                        "checkpoint_sha256",
                        "environment_sha256",
                        "upstream_sha256",
                    },
                )
                self.assertEqual(len(artifact["fingerprints"]["content_sha256"]), 64)
                self.assertEqual(
                    set(artifact["fingerprints"]["tool_sha256"]),
                    {"orchestrator", "reference_adapter"},
                )

            adapter_spec = importlib.util.spec_from_file_location(
                "generated_reference_adapter", adapter_path
            )
            self.assertIsNotNone(adapter_spec)
            self.assertIsNotNone(adapter_spec.loader)
            adapter_module = importlib.util.module_from_spec(adapter_spec)
            adapter_spec.loader.exec_module(adapter_module)
            for method in (
                "load",
                "preprocess",
                "infer",
                "capture_intermediates",
                "postprocess",
            ):
                self.assertTrue(hasattr(adapter_module.ReferenceAdapter, method))

            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            self.assertEqual(inventory["schema_version"], "1.0")
            self.assertEqual(inventory["adapter_contract_version"], "1.0")
            self.assertEqual(inventory["source"]["revision"], "0123456789abcdef")
            self.assertEqual(
                inventory["source"]["sha256"],
                json.loads((port / "request.json").read_text())["source"]["sha256"],
            )
            self.assertNotEqual(inventory["source"]["sha256"], source_digest)
            self.assertEqual(
                inventory["checkpoint"]["sha256"],
                hashlib.sha256(b"checkpoint bytes").hexdigest(),
            )
            self.assertEqual(
                [module["name"] for module in inventory["modules"]],
                ["", "encoder"],
            )
            self.assertEqual(len(inventory["parameters"]), 4)
            self.assertEqual(len(inventory["buffers"]), 1)
            self.assertEqual(
                {parameter["name"] for parameter in inventory["parameters"]},
                {
                    "encoder.weight",
                    "action_head.weight",
                    "decoder.weight_alias",
                    "encoder.weight_view",
                },
            )
            parameter_shapes = {
                parameter["name"]: parameter["shape"]
                for parameter in inventory["parameters"]
            }
            self.assertEqual(parameter_shapes["encoder.weight"], [2, 2])
            self.assertEqual(parameter_shapes["encoder.weight_view"], [1, 2])
            self.assertTrue(
                all(
                    parameter["dtype"] == "float32"
                    for parameter in inventory["parameters"]
                )
            )
            self.assertEqual(inventory["buffers"][0]["shape"], [4])
            self.assertEqual(inventory["buffers"][0]["dtype"], "float32")
            self.assertEqual(
                inventory["tied_weights"],
                [["action_head.weight", "encoder.weight"]],
            )
            self.assertEqual(
                inventory["aliases"],
                [[
                    "action_head.weight",
                    "decoder.weight_alias",
                    "encoder.weight",
                    "encoder.weight_view",
                ]],
            )
            self.assertEqual(
                inventory["operator_traces"][0]["operator"], "aten.linear"
            )
            for field in (
                "input_schema",
                "output_schema",
                "intermediate_schema",
                "preprocessing",
                "tokenization",
                "normalization",
                "stochastic_inputs",
                "schedules",
                "custom_operators",
                "dynamic_branches",
                "capability_facts",
            ):
                self.assertIn(field, inventory)

            classification = json.loads(
                classification_path.read_text(encoding="utf-8")
            )
            self.assertEqual(classification["schema_version"], "1.0")
            self.assertEqual(classification["contract"]["version"], "1.0")
            self.assertEqual(
                {item["classification"] for item in classification["classifications"]},
                {"supported"},
            )
            self.assertEqual(len(classification["classifications"]), 10)
            self.assertEqual(
                set(classification["dependency_fingerprints"]),
                set(inventory["capability_facts"]),
            )

            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            self.assertEqual(environment["schema_version"], "1.0")
            self.assertFalse(environment["runtime_network_access"])
            self.assertEqual(
                environment["isolation"],
                {
                    "kind": "venv",
                    "environment_id": environment["isolation"]["environment_id"],
                    "system_site_packages": False,
                },
            )
            self.assertEqual(
                environment["dependency_lock"]["sha256"],
                hashlib.sha256(b"").hexdigest(),
            )
            self.assertIn("private", capture_path.relative_to(port).parts)
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
            captured_actions = capture["profiles"][0]["postprocessed"]["actions"]
            self.assertEqual(captured_actions["dtype"], "f32")
            self.assertEqual(captured_actions["source_dtype"], "float32")
            self.assertEqual(captured_actions["shape"], [1, 2])
            self.assertEqual(captured_actions["data"], [[0.5, 4.0]])
            for schema_name, artifact in (
                ("reference-inventory-v1.schema.json", inventory),
                ("reference-environment-v1.schema.json", environment),
                ("reference-capture-v1.schema.json", capture),
            ):
                schema = json.loads(
                    (ROOT / "schemas" / schema_name).read_text(encoding="utf-8")
                )
                self.assertEqual(schema["properties"]["schema_version"]["const"], "1.0")
                self.assertEqual(set(schema["required"]) - artifact.keys(), set())

    def test_locked_environment_failure_is_reported_without_executing_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            port, source = self.initialize_inspectable_port(
                root,
                lock_text=(
                    "package-that-does-not-exist==1.0 "
                    "--hash=sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
                ),
            )
            source_before = {
                path.relative_to(source): path.read_bytes()
                for path in source.rglob("*")
                if path.is_file()
            }

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 5)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["stages"]["intake"], "passed")
            self.assertEqual(report["stages"]["preflight"], "not_started")
            self.assertEqual(report["reference_inspection"]["status"], "failed")
            self.assertEqual(report["exit"]["category"], "environment_failure")
            self.assertEqual(report["issues"][0]["path"], "reference.environment")
            self.assertEqual(
                source_before,
                {
                    path.relative_to(source): path.read_bytes()
                    for path in source.rglob("*")
                    if path.is_file()
                },
            )

    def test_each_inspection_uses_a_fresh_locked_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port, _ = self.initialize_inspectable_port(Path(temporary))

            first = self.run_port("run", "--port-dir", str(port))
            self.assertEqual(first.returncode, 0, first.stderr)
            first_environment = json.loads(
                (port / "private/reference_environment/environment.json").read_text(
                    encoding="utf-8"
                )
            )

            second = self.run_port("run", "--port-dir", str(port))
            self.assertEqual(second.returncode, 0, second.stderr)
            second_environment = json.loads(
                (port / "private/reference_environment/environment.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertNotEqual(
                first_environment["isolation"]["environment_id"],
                second_environment["isolation"]["environment_id"],
            )
            self.assertIsInstance(
                second_environment["installed_distributions"], list
            )
            self.assertTrue(
                all(
                    set(distribution) == {"name", "version"}
                    for distribution in second_environment["installed_distributions"]
                )
            )

    def test_reference_load_failure_has_a_distinct_structured_category(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port, _ = self.initialize_inspectable_port(
                Path(temporary), failure="load"
            )

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 6)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["exit"]["category"], "reference_load_failure")
            self.assertEqual(report["issues"][0]["path"], "reference.load")
            self.assertEqual(report["stages"]["preflight"], "not_started")
            self.assertEqual(report["reference_inspection"]["status"], "failed")

    def test_reference_trace_failure_has_a_distinct_structured_category(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port, _ = self.initialize_inspectable_port(
                Path(temporary), failure="trace"
            )

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 7)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["exit"]["category"], "reference_trace_failure")
            self.assertEqual(report["issues"][0]["path"], "reference.trace")
            self.assertIn("synthetic trace failed", report["issues"][0]["message"])

    def test_reference_runtime_network_is_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port, _ = self.initialize_inspectable_port(
                Path(temporary), failure="network"
            )

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 7)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["exit"]["category"], "reference_trace_failure")
            self.assertIn(
                "network access is disabled", report["issues"][0]["message"]
            )

    def test_schema_invalid_source_inventory_is_a_trace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port, _ = self.initialize_inspectable_port(
                Path(temporary), failure="invalid_inventory"
            )

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 7)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["exit"]["category"], "reference_trace_failure")
            self.assertIn("preprocessing", report["issues"][0]["message"])

    def test_missing_semantic_inventory_is_a_trace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port, _ = self.initialize_inspectable_port(
                Path(temporary), failure="missing_description"
            )

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 7)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["exit"]["category"], "reference_trace_failure")
            self.assertIn("describe", report["issues"][0]["message"])

    def test_preflight_reports_declared_canonicalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port, _ = self.initialize_inspectable_port(
                Path(temporary), failure="canonicalizable_attention"
            )

            result = self.run_port("run", "--port-dir", str(port))

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["stages"]["preflight"], "passed")
            self.assertEqual(report["capability_assessment"]["canonicalizable"], 1)
            classification_path = (
                port / report["artifacts"]["capability_classification"]["path"]
            )
            classifications = json.loads(
                classification_path.read_text(encoding="utf-8")
            )["classifications"]
            attention = next(
                item for item in classifications if item["capability"] == "attention"
            )
            self.assertEqual(attention["classification"], "canonicalizable")
            self.assertEqual(attention["canonical"], "scaled_dot_product")

    def test_unsupported_semantics_block_preflight_with_a_gap_report(self) -> None:
        cases = {
            "unsupported_attention": "capability_facts.attention[0]",
            "unknown_masks": "capability_facts.masks",
            "missing_capability_facts": "capability_facts.attention",
            "contradictory_normalization": "capability_facts.normalization",
            "unexplained_control_flow": "dynamic_branches[0]",
        }
        for failure, expected_path in cases.items():
            with self.subTest(failure=failure):
                with tempfile.TemporaryDirectory() as temporary:
                    port, _ = self.initialize_inspectable_port(
                        Path(temporary), failure=failure
                    )

                    result = self.run_port("run", "--port-dir", str(port))

                    self.assertEqual(result.returncode, 8)
                    report = json.loads(
                        (port / "report.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(report["stages"]["intake"], "passed")
                    self.assertEqual(report["stages"]["preflight"], "blocked")
                    self.assertEqual(
                        report["exit"]["category"], "unsupported_semantics"
                    )
                    self.assertEqual(
                        report["capability_assessment"]["status"], "blocked"
                    )
                    gap_path = port / report["artifacts"]["gap_report"]["path"]
                    gap = json.loads(gap_path.read_text(encoding="utf-8"))
                    self.assertEqual(gap["schema_version"], "1.0")
                    self.assertEqual(gap["category"], "unsupported_semantics")
                    self.assertIn(expected_path, {item["path"] for item in gap["gaps"]})

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
            self.assertEqual(request["capability_contract_version"], "1.0")
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

    def test_published_contract_declares_model_neutral_vla_semantics(self) -> None:
        contract = json.loads(
            (ROOT / "contracts/vla-capability-contract-1.0.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(contract["contract_version"], "1.0")
        self.assertEqual(contract["family"], "canonical_transformer_vla")
        self.assertEqual(
            set(contract["capabilities"]),
            {
                "shape_profiles",
                "attention",
                "masks",
                "position_encodings",
                "normalization",
                "activations",
                "conditioning",
                "action_heads",
                "schedules",
                "control_flow",
            },
        )
        self.assertNotIn("models", contract)
        serialized = json.dumps(contract).lower()
        for model_name in ("pi0", "pi05", "openvla", "deepseek"):
            self.assertNotIn(model_name, serialized)

        schema = json.loads(
            (ROOT / "schemas/vla-capability-contract-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["required"]) - contract.keys(), set())

    def test_additive_contract_update_preserves_unaffected_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            port, _ = self.initialize_inspectable_port(root)
            initial = self.run_port("run", "--port-dir", str(port))
            self.assertEqual(initial.returncode, 0, initial.stderr)
            initial_report = json.loads(
                (port / "report.json").read_text(encoding="utf-8")
            )
            initial_classification = json.loads(
                (
                    port
                    / initial_report["artifacts"]["capability_classification"]["path"]
                ).read_text(encoding="utf-8")
            )

            contract = json.loads(
                (ROOT / "contracts/vla-capability-contract-1.0.json").read_text(
                    encoding="utf-8"
                )
            )
            contract["contract_version"] = "1.1"
            contract["revision"] = {
                "kind": "additive",
                "previous_version": "1.0",
                "changes": [
                    {"capability": "attention", "kind": "additive"},
                    {"capability": "gripper_control", "kind": "additive"}
                ],
            }
            contract["capabilities"]["attention"]["supported"].append(
                "flash_equivalent_scaled_dot_product"
            )
            contract["capabilities"]["gripper_control"] = {
                "required": False,
                "cardinality": "exactly_one",
                "supported": ["parallel_jaw"],
                "canonicalizable": {},
            }
            contract_path = root / "capability-contract-1.1.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            request_path = port / "request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["capability_contract_version"] = "1.1"
            request_path.write_text(json.dumps(request), encoding="utf-8")

            updated = self.run_port(
                "run",
                "--port-dir",
                str(port),
                "--capability-contract",
                str(contract_path),
            )

            self.assertEqual(updated.returncode, 0, updated.stderr)
            updated_report = json.loads(
                (port / "report.json").read_text(encoding="utf-8")
            )
            updated_classification = json.loads(
                (
                    port
                    / updated_report["artifacts"]["capability_classification"]["path"]
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                updated_report["capability_assessment"]["contract_version"], "1.1"
            )
            initial_fingerprints = initial_classification["dependency_fingerprints"]
            updated_fingerprints = updated_classification["dependency_fingerprints"]
            self.assertEqual(set(initial_fingerprints), set(updated_fingerprints))
            self.assertEqual(
                {
                    capability
                    for capability in initial_fingerprints
                    if initial_fingerprints[capability]
                    != updated_fingerprints[capability]
                },
                {"attention"},
            )
            self.assertNotIn(
                "gripper_control", updated_classification["dependency_fingerprints"]
            )

    def test_breaking_contract_update_requires_a_new_major_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            port, _ = self.initialize_inspectable_port(root)
            contract = json.loads(
                (ROOT / "contracts/vla-capability-contract-1.0.json").read_text(
                    encoding="utf-8"
                )
            )
            contract["contract_version"] = "2.0"
            contract["revision"] = {
                "kind": "breaking",
                "previous_version": "1.0",
                "changes": [{"capability": "attention", "kind": "changed"}],
            }
            contract["capabilities"]["attention"]["supported"] = [
                "linear_attention"
            ]
            contract["capabilities"]["attention"]["canonicalizable"] = {}
            contract_path = root / "capability-contract-2.0.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            request_path = port / "request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["capability_contract_version"] = "2.0"
            request_path.write_text(json.dumps(request), encoding="utf-8")

            result = self.run_port(
                "run",
                "--port-dir",
                str(port),
                "--capability-contract",
                str(contract_path),
            )

            self.assertEqual(result.returncode, 8)
            report = json.loads((port / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["capability_assessment"]["contract_version"], "2.0")
            self.assertEqual(report["exit"]["category"], "unsupported_semantics")

            request["capability_contract_version"] = "1.0"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            mismatched = self.run_port(
                "run",
                "--port-dir",
                str(port),
                "--capability-contract",
                str(contract_path),
            )
            self.assertEqual(mismatched.returncode, 3)
            mismatched_report = json.loads(
                (port / "report.json").read_text(encoding="utf-8")
            )
            self.assertIn(
                "exact request pin", mismatched_report["issues"][0]["message"]
            )

            invalid_minor = copy.deepcopy(contract)
            invalid_minor["contract_version"] = "1.1"
            invalid_minor["revision"]["kind"] = "additive"
            invalid_path = root / "invalid-capability-contract-1.1.json"
            invalid_path.write_text(json.dumps(invalid_minor), encoding="utf-8")
            request["capability_contract_version"] = "1.1"
            request_path.write_text(json.dumps(request), encoding="utf-8")

            invalid = self.run_port(
                "run",
                "--port-dir",
                str(port),
                "--capability-contract",
                str(invalid_path),
            )
            self.assertEqual(invalid.returncode, 3)
            invalid_report = json.loads(
                (port / "report.json").read_text(encoding="utf-8")
            )
            self.assertIn("additive", invalid_report["issues"][0]["message"])

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
