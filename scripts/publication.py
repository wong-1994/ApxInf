"""Fail-closed preparation of public artifacts for a completed model Port."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


MAX_PUBLIC_FILE_BYTES = 1024 * 1024
FAMILIES = frozenset({"llm", "vlm", "vla"})
REMOTE_ACTIONS = frozenset({"push", "create_pr", "create_issue", "link_issue"})
FORBIDDEN_NAME_PARTS = (
    "reference_adapter",
    "canonical_adapter",
    "checkpoint",
    "original_source",
    "source_model",
    "upstream_source",
    "real_input",
    "real-input",
    "credential",
    "private",
)
FORBIDDEN_SUFFIXES = frozenset(
    {".ckpt", ".env", ".onnx", ".pt", ".pth", ".safetensors"}
)
SENSITIVE_CONTENT = re.compile(
    rb"(?i)(api[_-]?key|access[_-]?key(?:_id)?|access[_-]?token|secret(?:_access_key)?|password|private[_-]?key)\s*[:=]\s*[^\s,}]+"
)
PRIVATE_KEY_BLOCK = re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----")


class PublicationError(ValueError):
    """A Port cannot safely be prepared for publication."""


def _git(repo: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise PublicationError(f"Git safety check failed: {detail.strip()}") from error
    return result.stdout.strip()


def _require_safe_git_state(
    repo: Path, base_commit: str, base_branch: str
) -> dict[Path, str]:
    if not repo.is_dir() or _git(repo, "rev-parse", "--show-toplevel") != str(
        repo.resolve()
    ):
        raise PublicationError("repository must be the dedicated worktree root")
    _git(repo, "rev-parse", "--verify", f"{base_commit}^{{commit}}")
    if _git(repo, "status", "--porcelain"):
        raise PublicationError("publication preparation requires a clean worktree")
    branch = _git(repo, "branch", "--show-current")
    if not branch or branch == base_branch:
        raise PublicationError("publication requires a dedicated non-base branch")
    base_ref = f"refs/heads/{base_branch}"
    _git(repo, "rev-parse", "--verify", base_ref)
    merge_base = _git(repo, "merge-base", base_ref, "HEAD")
    if merge_base != _git(repo, "rev-parse", base_commit):
        raise PublicationError("base_commit must be the merge base of the dedicated branch")
    commits = _git(repo, "rev-list", "--count", f"{base_commit}..HEAD")
    if commits == "0":
        raise PublicationError("publication requires at least one local stage commit")
    names = _git(repo, "diff", "--name-status", f"{base_commit}...HEAD")
    changed: dict[Path, str] = {}
    for line in names.splitlines():
        fields = line.split("\t")
        status = fields[0][0]
        name = fields[-1]
        changed[repo / name] = status
    return changed


def _check_public_files(
    repo: Path, paths: Mapping[Path, str], declarations: Mapping[str, Mapping[str, Any]]
) -> None:
    for path, status in paths.items():
        relative = path.relative_to(repo).as_posix()
        declaration = declarations.get(relative)
        if declaration is None:
            raise PublicationError(f"publication candidate is undeclared: {relative}")
        if declaration.get("kind") not in {
            "maintained_source",
            "synthetic_fixture",
            "support_metadata",
        }:
            raise PublicationError(
                f"publication rejects original or private material: {relative}"
            )
        if declaration.get("redistribution_approved") is not True:
            raise PublicationError(f"publication lacks redistribution approval: {relative}")
        lowered = relative.lower()
        if (
            any(part in lowered for part in FORBIDDEN_NAME_PARTS)
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
        ):
            raise PublicationError(
                f"publication rejects private or original material: {relative}"
            )
        if status == "D":
            continue
        if not path.is_file():
            raise PublicationError(f"publication candidate is not a regular file: {relative}")
        if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
            raise PublicationError(f"publication rejects oversized artifact: {relative}")
        content = path.read_bytes()
        if SENSITIVE_CONTENT.search(content) or PRIVATE_KEY_BLOCK.search(content):
            raise PublicationError(f"publication rejects sensitive content: {relative}")
        if b"redistribution_approved=false" in content.replace(b" ", b"").lower():
            raise PublicationError(f"publication lacks redistribution approval: {relative}")


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "1.0":
        raise PublicationError("unsupported publication schema_version")
    if payload.get("family") not in FAMILIES:
        raise PublicationError("family must be llm, vlm, or vla")
    for field in ("port_id", "base_branch"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise PublicationError(f"{field} must be a non-empty string")
    actions = payload.get("remote_actions", [])
    if not isinstance(actions, list) or any(
        action not in REMOTE_ACTIONS for action in actions
    ):
        raise PublicationError("remote_actions contains an unsupported action")
    if actions and payload.get("publication_authorized") is not True:
        raise PublicationError("remote actions require explicit publication authorization")
    assessment = payload.get("refactor_assessment")
    if not isinstance(assessment, dict) or assessment.get("status") not in {
        "none",
        "deferred",
    }:
        raise PublicationError("every Port requires a none or deferred refactor assessment")
    if assessment["status"] == "none" and (
        not isinstance(assessment.get("summary"), str)
        or not assessment["summary"].strip()
    ):
        raise PublicationError("a none refactor assessment requires a summary")
    if assessment["status"] == "deferred":
        if not all(
            isinstance(assessment.get(field), str) and assessment[field].strip()
            for field in ("title", "proposal")
        ) or (
            not isinstance(assessment.get("evidence"), list)
            or not assessment["evidence"]
            or any(
                not isinstance(item, str) or not item.strip()
                for item in assessment["evidence"]
            )
        ):
            raise PublicationError("deferred refactor debt requires string title/proposal and evidence strings")
    files = payload.get("public_files")
    if not isinstance(files, list) or not files:
        raise PublicationError("public_files must declare every publication candidate")
    paths = [item.get("path") for item in files if isinstance(item, dict)]
    if len(paths) != len(files) or any(
        not isinstance(path, str) or not path for path in paths
    ):
        raise PublicationError("every public_files entry requires a path")
    if len(paths) != len(set(paths)):
        raise PublicationError("public_files paths must be unique")


def _supported_tuples(
    payload: Mapping[str, Any],
    requested_tuples: set[tuple[str, str]],
    qualified_tuples: set[tuple[str, str]],
) -> list[dict[str, str]]:
    requested = {
        (item.get("target"), item.get("precision"))
        for item in payload.get("supported_tuples", [])
        if isinstance(item, dict)
        and all(isinstance(item.get(field), str) for field in ("target", "precision"))
    }
    return [
        {"target": target, "precision": precision}
        for target, precision in sorted(requested & requested_tuples & qualified_tuples)
        if isinstance(target, str) and isinstance(precision, str)
    ]


def _write(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def prepare_publication(
    repository: Path,
    base_commit: str,
    payload: Mapping[str, Any],
    output_dir: Path,
    *,
    requested_tuples: set[tuple[str, str]],
    qualified_tuples: set[tuple[str, str]],
) -> Path:
    """Validate a Port and prepare local review text without remote side effects."""

    repository = repository.resolve()
    output_dir = output_dir.resolve()
    _validate_payload(payload)
    changed = _require_safe_git_state(
        repository, base_commit, str(payload["base_branch"])
    )
    declarations = {item["path"]: item for item in payload["public_files"]}
    changed_names = {path.relative_to(repository).as_posix() for path in changed}
    if set(declarations) != changed_names:
        raise PublicationError("public_files must exactly match the fixed-point diff")
    _check_public_files(repository, changed, declarations)
    if output_dir.is_relative_to(repository):
        raise PublicationError(
            "prepared publication artifacts must remain outside the repository"
        )
    if output_dir.exists():
        raise PublicationError("publication output already exists")
    output_dir.mkdir(parents=True)

    tuples = _supported_tuples(payload, requested_tuples, qualified_tuples)
    support = {
        "schema_version": "1.0",
        "port_id": payload["port_id"],
        "family": payload["family"],
        "supported_tuples": tuples,
    }
    _write(
        output_dir / "support-metadata.json",
        json.dumps(support, indent=2, sort_keys=True),
    )

    assessment = payload["refactor_assessment"]
    if assessment["status"] == "none":
        deferred_summary = "None — " + assessment["summary"]
    else:
        deferred_summary = f"- {assessment['title']}: {assessment['proposal']}"
        evidence = "\n".join(f"- {item}" for item in assessment["evidence"])
        _write(
            output_dir / "refactor-issue.md",
            f"# {assessment['title']}\n\n## Evidence\n\n{evidence}\n\n"
            f"## Proposed follow-up\n\n{assessment['proposal']}\n\n"
            "## Scope\n\nImplement as a separate change; no refactor is included in this Port.",
        )
    tuple_lines = "\n".join(
        f"- `{item['target']}/{item['precision']}`" for item in tuples
    ) or "- None"
    _write(
        output_dir / "pull-request.md",
        f"# Port {payload['port_id']}\n\nFamily: `{payload['family']}`\n\n"
        f"## Qualified support\n\n{tuple_lines}\n\n"
        f"## Deferred Refactors\n\n{deferred_summary}\n\n"
        "## Publication\n\nPrepared locally. Push, PR creation, issue creation, "
        "and linking require explicit authorization.",
    )
    return output_dir
