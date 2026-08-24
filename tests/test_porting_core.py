import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from porting_core import (  # noqa: E402
    INVALID_INPUT,
    SUCCESS,
    ArtifactStore,
    PortingCore,
    VLA_FAMILY_PACK,
    resume_report,
)


class PortingCoreTest(unittest.TestCase):
    def request(self) -> dict:
        return {
            "schema_version": "1.0",
            "port_id": "example-port",
            "model_family": "vla",
            "capability_contract_version": "1.0",
            "source": {"sha256": "1" * 64},
            "checkpoint": {"sha256": "2" * 64},
            "reference": {},
            "representative_profiles": [],
            "requested_targets": [{"target": "thor", "precision": "bf16"}],
            "correctness_thresholds": {},
            "tuning_budgets": [],
        }

    def test_core_owns_lifecycle_gates_exit_and_requested_tuples(self) -> None:
        core = PortingCore(self.request(), VLA_FAMILY_PACK, {"python": "test"})

        core.pass_stage("intake")
        core.start_stage("preflight")
        core.set_gate("capability", "passed", {"supported": 10})
        core.finish(SUCCESS, "Preflight passed")

        self.assertEqual(core.report["stages"], {"intake": "passed", "preflight": "running"})
        self.assertEqual(core.report["gates"]["capability"]["status"], "passed")
        self.assertEqual(core.report["exit"]["category"], "success")
        self.assertEqual(core.report["refactor_assessment"]["status"], "none")
        requested = [
            item
            for item in core.report["target_precisions"]
            if item["status"] == "requested"
        ]
        self.assertEqual(
            requested,
            [{"target": "thor", "precision": "bf16", "status": "requested"}],
        )

    def test_core_builds_family_neutral_failed_report(self) -> None:
        core = PortingCore.failed(
            {},
            VLA_FAMILY_PACK,
            INVALID_INPUT,
            [{"path": "request", "message": "bad"}],
            {},
        )

        self.assertEqual(core.report["stages"]["intake"], "failed")
        self.assertEqual(core.report["issues"][0]["message"], "bad")
        self.assertEqual(core.report["exit"]["code"], 3)

    def test_artifact_store_validates_payload_schema_and_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            port_dir = Path(temporary)
            private = port_dir / "private"
            private.mkdir()
            request_path = port_dir / "request.json"
            request_path.write_text(json.dumps(self.request()), encoding="utf-8")
            adapter = private / "reference_adapter.py"
            adapter.write_text("pass\n", encoding="utf-8")
            environment = private / "environment.json"
            environment.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
            inventory = private / "source_inventory.json"
            inventory.write_text(json.dumps({"schema_version": "9.9"}), encoding="utf-8")
            store = ArtifactStore(
                port_dir,
                self.request(),
                VLA_FAMILY_PACK,
                ROOT / "scripts/apxinf_port.py",
                adapter,
            )

            with self.assertRaisesRegex(ValueError, "schema"):
                store.record(inventory, environment)

    def test_resume_marks_only_changed_dependencies_and_descendants_stale(self) -> None:
        artifacts = {
            "common": self.envelope("vla", "common", content="a"),
            "vla": self.envelope(
                "vla", "vla", content="b", upstream={"common": "a"}
            ),
            "llm": self.envelope("llm", "llm", content="c"),
        }
        current = json.loads(json.dumps(artifacts))
        current["common"]["fingerprints"]["documentation_sha256"] = "9" * 64

        resumed = resume_report(
            {
                "stages": {"intake": "passed", "preflight": "passed"},
                "gates": {
                    "common": {
                        "status": "passed",
                        "evidence": {"artifacts": ["common"]},
                    },
                    "vla": {"status": "passed", "evidence": {"artifacts": ["vla"]}},
                    "llm": {"status": "passed", "evidence": {"artifacts": ["llm"]}},
                },
                "artifacts": artifacts,
            },
            current,
        )

        self.assertEqual(resumed["artifacts"]["common"]["state"], "stale")
        self.assertEqual(resumed["artifacts"]["vla"]["state"], "stale")
        self.assertEqual(resumed["artifacts"]["llm"]["state"], "current")
        self.assertEqual(resumed["gates"]["common"]["status"], "stale")
        self.assertEqual(resumed["gates"]["vla"]["status"], "stale")
        self.assertEqual(resumed["gates"]["llm"]["status"], "passed")
        self.assertIn(
            "documentation_sha256",
            resumed["artifacts"]["common"]["explanation"][
                "changed_dependencies"
            ],
        )

    def test_resume_recovers_interrupted_stage_without_trusting_existing_files(
        self,
    ) -> None:
        recorded = self.envelope("vlm", "capture", content="a")
        current = json.loads(json.dumps(recorded))
        current["fingerprints"]["content_sha256"] = "f" * 64
        resumed = resume_report(
            {
                "stages": {"intake": "passed", "preflight": "running"},
                "gates": {
                    "capture": {
                        "status": "running",
                        "evidence": {"artifacts": ["capture"]},
                    }
                },
                "artifacts": {"capture": recorded},
            },
            {"capture": current},
        )

        self.assertEqual(resumed["stages"]["preflight"], "not_started")
        self.assertEqual(resumed["gates"]["capture"]["status"], "stale")
        self.assertEqual(resumed["resume"]["interrupted_stages"], ["preflight"])
        self.assertEqual(resumed["artifacts"]["capture"]["state"], "stale")

    def test_resume_recognizes_each_dependency_class_without_cross_invalidation(
        self,
    ) -> None:
        dependency_keys = (
            "checkpoint_sha256",
            "source_sha256",
            "apxinf_source_sha256",
            "kernel_build_sha256",
            "environment_sha256",
            "capability_contract_sha256",
            "documentation_sha256",
            "target_environment_sha256",
        )
        for key in dependency_keys:
            with self.subTest(key=key):
                artifacts = {
                    "affected": self.envelope("llm", "affected", content="a"),
                    "unrelated": self.envelope("vlm", "unrelated", content="b"),
                }
                current = json.loads(json.dumps(artifacts))
                current["affected"]["fingerprints"][key] = {"changed": "value"}
                resumed = resume_report(
                    {"stages": {}, "gates": {}, "artifacts": artifacts}, current
                )
                self.assertEqual(
                    resumed["artifacts"]["affected"]["state"], "stale"
                )
                self.assertEqual(
                    resumed["artifacts"]["unrelated"]["state"], "current"
                )

    def envelope(self, family: str, path: str, *, content: str, upstream=None) -> dict:
        return {
            "envelope_version": "1.0",
            "family": family,
            "capability_contract_version": "1.0",
            "stage": "preflight",
            "payload_schema": f"synthetic-{family}-v1",
            "path": f"private/{path}.json",
            "dependency_paths": {},
            "state": "current",
            "explanation": {"changed_dependencies": [], "upstream_stale": []},
            "fingerprints": {
                "content_sha256": content,
                "tool_sha256": {"runner": "1" * 64},
                "source_sha256": "2" * 64,
                "checkpoint_sha256": "3" * 64,
                "apxinf_source_sha256": "4" * 64,
                "kernel_build_sha256": "5" * 64,
                "environment_sha256": "6" * 64,
                "capability_contract_sha256": "7" * 64,
                "documentation_sha256": "8" * 64,
                "target_environment_sha256": {},
                "upstream_sha256": upstream or {},
            },
        }

    def test_family_payload_validation_follows_references_and_constraints(self) -> None:
        inventory = {
            "schema_version": "1.0",
            "adapter_contract_version": "1.0",
            "source": {
                "revision": "main",
                "sha256": "1" * 64,
                "entrypoint": "model.py",
            },
            "checkpoint": {"sha256": "2" * 64},
            "modules": [{"name": "model", "type": ""}],
            "parameters": [],
            "buffers": [],
            "aliases": [],
            "tied_weights": [],
            "operator_traces": [],
            "input_schema": [],
            "output_schema": [],
            "intermediate_schema": [],
            "preprocessing": {},
            "tokenization": {},
            "normalization": {},
            "stochastic_inputs": [],
            "schedules": [],
            "custom_operators": [],
            "dynamic_branches": [],
            "capability_facts": {},
        }

        with self.assertRaisesRegex(ValueError, "minLength"):
            VLA_FAMILY_PACK.validate_payload(
                Path("source_inventory.json"), inventory
            )
