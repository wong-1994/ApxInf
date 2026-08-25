import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from publication import PublicationError, prepare_publication  # noqa: E402


class PublicationTest(unittest.TestCase):
    def prepare(self, repo: Path, base: str, payload: dict, output: Path) -> Path:
        return prepare_publication(
            repo,
            base,
            payload,
            output,
            requested_tuples={("thor", "bf16"), ("orin", "int8_w8a8")},
            qualified_tuples={("thor", "bf16")},
        )

    def make_repo(self, root: Path) -> tuple[Path, str]:
        repo = root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=repo, check=True
        )
        (repo / "model.rs").write_text("maintained source\n", encoding="utf-8")
        subprocess.run(["git", "add", "model.rs"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        subprocess.run(["git", "switch", "-qc", "port/example"], cwd=repo, check=True)
        (repo / "model.rs").write_text("reviewable maintained source\n", encoding="utf-8")
        subprocess.run(["git", "add", "model.rs"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "port model"], cwd=repo, check=True)
        return repo, base

    def payload(self, family: str, debt: bool = False) -> dict:
        return {
            "schema_version": "1.0",
            "port_id": f"{family}-example",
            "family": family,
            "base_branch": "main",
            "public_files": [
                {
                    "path": "model.rs",
                    "kind": "maintained_source",
                    "redistribution_approved": True,
                }
            ],
            "supported_tuples": [
                {"target": "thor", "precision": "bf16", "evidence": "a" * 64},
                {"target": "orin", "precision": "int8_w8a8", "evidence": "b" * 64},
            ],
            "qualification": [
                {
                    "target": "thor",
                    "precision": "bf16",
                    "status": "release_qualified",
                    "evidence": "a" * 64,
                },
                {
                    "target": "orin",
                    "precision": "int8_w8a8",
                    "status": "performance_pending",
                    "evidence": "b" * 64,
                },
            ],
            "refactor_assessment": (
                {
                    "status": "deferred",
                    "title": "Share cache layout",
                    "evidence": ["model.rs:1"],
                    "proposal": "Extract after another family needs it.",
                }
                if debt else {"status": "none", "summary": "No shared debt found."}
            ),
        }

    def test_core_policy_prepares_family_metadata_and_explicit_none_assessment(self) -> None:
        for family in ("llm", "vlm", "vla"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                repo, base = self.make_repo(root)
                output = self.prepare(repo, base, self.payload(family), root / "out")
                support = json.loads((output / "support-metadata.json").read_text())
                self.assertEqual(support["family"], family)
                self.assertEqual(
                    support["supported_tuples"],
                    [{"target": "thor", "precision": "bf16"}],
                )
                description = (output / "pull-request.md").read_text()
                self.assertIn("Deferred Refactors\n\nNone", description)
                self.assertFalse((output / "refactor-issue.md").exists())

    def test_concrete_debt_emits_issue_and_required_pr_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, base = self.make_repo(root)
            output = self.prepare(
                repo, base, self.payload("vla", True), root / "out"
            )
            self.assertIn("Share cache layout", (output / "refactor-issue.md").read_text())
            self.assertIn("Deferred Refactors", (output / "pull-request.md").read_text())

    def test_rejects_dirty_or_non_dedicated_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, base = self.make_repo(root)
            (repo / "uncommitted.txt").write_text("mine", encoding="utf-8")
            with self.assertRaisesRegex(PublicationError, "clean"):
                self.prepare(repo, base, self.payload("vla"), root / "out")

    def test_rejects_private_sensitive_unapproved_and_oversized_material(self) -> None:
        forbidden = {
            "reference_adapter.py": "reference adapter",
            "checkpoint.safetensors": "checkpoint",
            "real_input.json": "real input",
            "credentials.env": "TOKEN=secret-value",
            "fixture.json": "redistribution_approved=false",
            "original_source.py": "print('upstream implementation')",
        }
        for name, content in forbidden.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                repo, base = self.make_repo(root)
                path = repo / name
                path.write_text(content, encoding="utf-8")
                subprocess.run(["git", "add", name], cwd=repo, check=True)
                subprocess.run(["git", "commit", "-qm", "bad material"], cwd=repo, check=True)
                with self.assertRaises(PublicationError):
                    self.prepare(repo, base, self.payload("vla"), root / "out")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, base = self.make_repo(root)
            path = repo / "large.bin"
            path.write_bytes(b"x" * (1024 * 1024 + 1))
            subprocess.run(["git", "add", "large.bin"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "large material"], cwd=repo, check=True)
            payload = self.payload("vla")
            payload["public_files"].append(
                {
                    "path": "large.bin",
                    "kind": "synthetic_fixture",
                    "redistribution_approved": True,
                }
            )
            with self.assertRaisesRegex(PublicationError, "oversized"):
                self.prepare(repo, base, payload, root / "out")

    def test_remote_publication_requires_explicit_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, base = self.make_repo(root)
            payload = self.payload("vla") | {"remote_actions": ["push"]}
            with self.assertRaisesRegex(PublicationError, "authorization"):
                self.prepare(repo, base, payload, root / "out")

    def test_support_metadata_requires_report_backed_requested_and_qualified_tuples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, base = self.make_repo(root)
            output = prepare_publication(
                repo,
                base,
                self.payload("vla"),
                root / "out",
                requested_tuples={("thor", "bf16")},
                qualified_tuples=set(),
            )
            support = json.loads((output / "support-metadata.json").read_text())
            self.assertEqual(support["supported_tuples"], [])

    def test_rejects_deleted_undeclared_files_and_false_base_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, base = self.make_repo(root)
            subprocess.run(["git", "rm", "model.rs"], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-qm", "delete model"], cwd=repo, check=True)
            undeclared = self.payload("vla")
            undeclared["public_files"][0]["path"] = "different.rs"
            with self.assertRaisesRegex(PublicationError, "exactly match"):
                self.prepare(repo, base, undeclared, root / "out")
            subprocess.run(["git", "branch", "-m", "main-work"], cwd=repo, check=True)
            with self.assertRaisesRegex(PublicationError, "dedicated"):
                self.prepare(repo, base, self.payload("vla"), root / "out")

    def test_rejects_private_key_blocks_and_malformed_refactor_debt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, base = self.make_repo(root)
            (repo / "model.rs").write_text(
                "-----BEGIN OPENSSH PRIVATE KEY-----\nsecret\n", encoding="utf-8"
            )
            subprocess.run(["git", "add", "model.rs"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "bad secret"], cwd=repo, check=True)
            with self.assertRaisesRegex(PublicationError, "sensitive"):
                self.prepare(repo, base, self.payload("vla"), root / "out")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, base = self.make_repo(root)
            (repo / "model.rs").write_text(
                "token = 'ghp_abcdefghijklmnopqrstuvwxyz123456'\n", encoding="utf-8"
            )
            subprocess.run(["git", "add", "model.rs"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "bad token"], cwd=repo, check=True)
            with self.assertRaisesRegex(PublicationError, "sensitive"):
                self.prepare(repo, base, self.payload("vla"), root / "out")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, base = self.make_repo(root)
            payload = self.payload("vla")
            payload["refactor_assessment"] = {
                "status": "deferred", "title": "Debt", "evidence": "model.rs:1", "proposal": "Fix later"
            }
            with self.assertRaisesRegex(PublicationError, "evidence strings"):
                self.prepare(repo, base, payload, root / "out")


if __name__ == "__main__":
    unittest.main()
