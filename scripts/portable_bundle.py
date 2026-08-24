"""Create and merge provenance-bound Portable Run Bundles."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


BUNDLE_SCHEMA_VERSION = "1.0"
PRIVATE_ADAPTERS = ("private/reference_adapter.py", "private/canonical_adapter.py")
SENSITIVE_ARTIFACT_SCHEMAS = frozenset({"reference-capture-v1.schema.json"})


class BundleError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read {description}: {error}") from error
    if not isinstance(value, dict):
        raise BundleError(f"{description} must be a JSON object")
    return value


def _safe_relative(value: Any) -> Path:
    if not isinstance(value, str):
        raise BundleError("artifact path must be a string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise BundleError(f"unsafe artifact path: {value!r}")
    return Path(*pure.parts)


def _safe_file(root: Path, value: Any) -> Path:
    path = (root / _safe_relative(value)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise BundleError(f"artifact path escapes its root: {value!r}")
    return path


def _portable_request(request: Mapping[str, Any]) -> dict[str, Any]:
    portable = copy.deepcopy(dict(request))
    for field in ("source", "checkpoint"):
        declaration = portable.get(field)
        if isinstance(declaration, dict):
            declaration["path"] = None
    reference = portable.get("reference")
    if isinstance(reference, dict):
        reference.pop("entrypoint", None)
        reference.pop("dependency_lock", None)
    portable.pop("dependency_fingerprints", None)
    return _sanitize_paths_and_credentials(portable)


def _sanitize_paths_and_credentials(value: Any, key: str = "") -> Any:
    normalized = key.lower()
    sensitive_markers = (
        "credential",
        "password",
        "secret",
        "api_key",
        "access_token",
        "auth_token",
    )
    if any(marker in normalized for marker in sensitive_markers):
        return None
    if isinstance(value, dict):
        return {
            name: _sanitize_paths_and_credentials(item, str(name))
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_paths_and_credentials(item, key) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return None
    return value


def _portable_envelope(envelope: Mapping[str, Any], bundle_path: str) -> dict[str, Any]:
    portable = copy.deepcopy(dict(envelope))
    portable["path"] = bundle_path
    portable["dependency_paths"] = {}
    return portable


def _contains_private_material(value: Any, key: str = "") -> bool:
    sensitive_key = any(
        marker in key.lower()
        for marker in (
            "credential",
            "password",
            "secret",
            "api_key",
            "access_token",
            "auth_token",
        )
    )
    if sensitive_key and value not in (None, "", [], {}):
        return True
    if isinstance(value, dict):
        return any(
            _contains_private_material(item, str(name))
            for name, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_private_material(item, key) for item in value)
    return isinstance(value, str) and Path(value).is_absolute()


def _artifact_must_be_omitted(envelope: Mapping[str, Any], source: Path) -> bool:
    if envelope.get("payload_schema") in SENSITIVE_ARTIFACT_SCHEMAS:
        return True
    if source.suffix.lower() in {".ckpt", ".safetensors", ".pt", ".pth"}:
        return True
    if source.suffix == ".json":
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            return _contains_private_material(payload)
        except (OSError, json.JSONDecodeError):
            return False
    return False


def _copy_file(
    source: Path, destination: Path, files: dict[str, str], relative: str
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    files[relative] = _sha256(destination)


def create_bundle(port_dir: Path, output: Path) -> Path:
    port_dir = port_dir.resolve()
    output = output.resolve()
    if output.exists():
        raise BundleError(f"bundle output already exists: {output}")
    request_path = port_dir / "request.json"
    request = _load_json(request_path, "Port request")
    report = _load_json(port_dir / "report.json", "Port report")
    family = request.get("model_family")
    contract = request.get("capability_contract_version")
    port_id = request.get("port_id")
    if not all(isinstance(item, str) and item for item in (family, contract, port_id)):
        raise BundleError("Port request lacks portable identity provenance")

    output.mkdir(parents=True)
    files: dict[str, str] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    portable_request = _portable_request(request)
    _write_json(output / "request.json", portable_request)
    files["request.json"] = _sha256(output / "request.json")

    report_artifacts = report.get("artifacts")
    if not isinstance(report_artifacts, dict):
        raise BundleError("Port report artifacts must be an object")
    for name, envelope_value in sorted(report_artifacts.items()):
        if not isinstance(name, str) or not isinstance(envelope_value, dict):
            raise BundleError("report contains an invalid artifact envelope")
        if envelope_value.get("family") != family:
            raise BundleError(f"artifact {name} has a family-payload mismatch")
        source = _safe_file(port_dir, envelope_value.get("path"))
        if not source.is_file():
            raise BundleError(f"artifact {name} payload is missing")
        expected = envelope_value.get("fingerprints", {}).get("content_sha256")
        if expected != _sha256(source):
            raise BundleError(
                f"artifact {name} content hash does not match its provenance"
            )
        if _artifact_must_be_omitted(envelope_value, source):
            continue
        suffix = source.suffix or ".bin"
        relative = f"artifacts/{name}{suffix}"
        _copy_file(source, output / relative, files, relative)
        artifacts[name] = _portable_envelope(envelope_value, relative)

    for relative in PRIVATE_ADAPTERS:
        source = port_dir / relative
        if source.is_file():
            _copy_file(source, output / relative, files, relative)

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "port": {
            "id": port_id,
            "family": family,
            "capability_contract_version": contract,
            "source_revision": request.get("source", {}).get("revision"),
            "source_sha256": request.get("source", {}).get("sha256"),
            "checkpoint_sha256": request.get("checkpoint", {}).get("sha256"),
            "request_sha256": _sha256(request_path),
        },
        "requested_tuples": request.get("requested_targets", []),
        "artifacts": artifacts,
        "files": dict(sorted(files.items())),
        "privacy": {
            "contains_private_adapters": any(
                relative in files for relative in PRIVATE_ADAPTERS
            ),
            "publishable": False,
            "excluded": ["credentials", "absolute_paths", "checkpoints", "real_inputs"],
        },
    }
    _write_json(output / "manifest.json", manifest)
    return output


def _tuple_key(item: Mapping[str, Any]) -> str | None:
    target = item.get("target")
    precision = item.get("precision")
    if isinstance(target, str) and isinstance(precision, str):
        return f"{target}/{precision}"
    return None


def _artifact_tuple(
    name: str, envelope: Mapping[str, Any], payload_path: Path
) -> dict[str, str] | None:
    targets = envelope.get("fingerprints", {}).get("target_environment_sha256", {})
    payload: dict[str, Any] | None = None
    if payload_path.suffix == ".json":
        payload = _load_json(payload_path, f"artifact {name}")
        if payload.get("family") not in (None, envelope.get("family")):
            raise BundleError(f"artifact {name} has a family-payload mismatch")
    target_specific = envelope.get("stage") in {
        "tuning",
        "performance",
        "qualification",
    }
    target_specific = target_specific or (
        isinstance(payload, dict)
        and any(key in payload for key in ("tactics", "performance"))
    )
    if not isinstance(targets, dict) or not targets:
        if target_specific:
            raise BundleError(
                f"target evidence {name} lacks an environment fingerprint"
            )
        return None
    if len(targets) != 1:
        raise BundleError(f"target evidence {name} must bind exactly one environment")
    key, environment = next(iter(targets.items()))
    if not isinstance(key, str) or "/" not in key or not isinstance(environment, str):
        raise BundleError(f"target evidence {name} has invalid environment provenance")
    target, precision = key.split("/", 1)
    if payload is not None:
        for field, expected in (("target", target), ("precision", precision)):
            if payload.get(field) not in (None, expected):
                raise BundleError(f"artifact {name} has mismatched {field} evidence")
        declared_environment = payload.get("environment_fingerprint")
        if declared_environment not in (None, environment):
            raise BundleError(f"artifact {name} has mismatched environment evidence")
    return {
        "artifact": name,
        "target": target,
        "precision": precision,
        "environment_sha256": environment,
    }


def merge_bundle(port_dir: Path, bundle: Path) -> Path:
    port_dir = port_dir.resolve()
    bundle = bundle.resolve()
    manifest = _load_json(bundle / "manifest.json", "bundle manifest")
    request = _load_json(port_dir / "request.json", "destination Port request")
    report_path = port_dir / "report.json"
    report = _load_json(report_path, "destination Port report")
    identity = manifest.get("port", {})
    if not isinstance(identity, dict):
        raise BundleError("bundle Port provenance is invalid")
    expected_identity = {
        "id": request.get("port_id"),
        "family": request.get("model_family"),
        "capability_contract_version": request.get("capability_contract_version"),
        "source_revision": request.get("source", {}).get("revision"),
        "source_sha256": request.get("source", {}).get("sha256"),
        "checkpoint_sha256": request.get("checkpoint", {}).get("sha256"),
    }
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise BundleError(f"bundle Port provenance conflicts at {field}")

    files = manifest.get("files")
    artifacts = manifest.get("artifacts")
    if not isinstance(files, dict) or not isinstance(artifacts, dict):
        raise BundleError("bundle manifest is incomplete")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise BundleError("bundle file manifest is invalid")
        path = _safe_file(bundle, relative)
        if not path.is_file() or _sha256(path) != expected:
            raise BundleError(f"bundle content hash mismatch: {relative}")

    destination_artifacts = report.setdefault("artifacts", {})
    if not isinstance(destination_artifacts, dict):
        raise BundleError("destination Port artifacts must be an object")
    available = set(destination_artifacts) | set(artifacts)
    requested = {
        key for item in request.get("requested_targets", [])
        if isinstance(item, dict) and (key := _tuple_key(item)) is not None
    }
    evidence = {
        (item.get("artifact"), item.get("target"), item.get("precision")): item
        for item in report.get("portable_evidence", []) if isinstance(item, dict)
    }
    for existing_name, existing_envelope in destination_artifacts.items():
        if (
            not isinstance(existing_envelope, dict)
            or existing_envelope.get("state") != "current"
        ):
            continue
        existing_path = _safe_file(port_dir, existing_envelope.get("path"))
        if not existing_path.is_file():
            continue
        existing_tuple = _artifact_tuple(
            existing_name, existing_envelope, existing_path
        )
        if existing_tuple:
            evidence[
                (existing_name, existing_tuple["target"], existing_tuple["precision"])
            ] = existing_tuple
    imports: list[tuple[str, dict[str, Any], Path, dict[str, str] | None]] = []
    for name, envelope_value in sorted(artifacts.items()):
        if not isinstance(envelope_value, dict):
            raise BundleError(f"artifact {name} envelope is invalid")
        if envelope_value.get("family") != request.get("model_family"):
            raise BundleError(f"artifact {name} has a family-payload mismatch")
        if envelope_value.get("state") != "current":
            raise BundleError(f"artifact {name} is stale and cannot be merged")
        fingerprints = envelope_value.get("fingerprints")
        if not isinstance(fingerprints, dict):
            raise BundleError(f"artifact {name} fingerprints are invalid")
        upstream = fingerprints.get("upstream_sha256", {})
        if not isinstance(upstream, dict):
            raise BundleError(f"artifact {name} dependencies are invalid")
        missing = sorted(
            key for key in upstream if key != "request" and key not in available
        )
        if missing:
            raise BundleError(
                f"artifact {name} has missing dependencies: {', '.join(missing)}"
            )
        for dependency, expected_hash in upstream.items():
            if dependency == "request":
                continue
            parent = artifacts.get(dependency) or destination_artifacts.get(dependency)
            actual_hash = parent.get("fingerprints", {}).get("content_sha256")
            if actual_hash != expected_hash:
                raise BundleError(
                    f"artifact {name} dependency {dependency} was not revalidated"
                )
        relative = envelope_value.get("path")
        payload_path = _safe_file(bundle, relative)
        expected = envelope_value.get("fingerprints", {}).get("content_sha256")
        if expected != _sha256(payload_path):
            raise BundleError(
                f"artifact {name} content hash conflicts with its envelope"
            )
        tuple_evidence = _artifact_tuple(name, envelope_value, payload_path)
        if (
            tuple_evidence
            and f"{tuple_evidence['target']}/{tuple_evidence['precision']}"
            not in requested
        ):
            raise BundleError(
                f"artifact {name} is for an unrequested target/precision tuple"
            )
        existing = destination_artifacts.get(name)
        if (
            existing
            and existing.get("fingerprints", {}).get("content_sha256") != expected
        ):
            raise BundleError(f"artifact {name} conflicts with destination evidence")
        imports.append((name, envelope_value, payload_path, tuple_evidence))

    request_sha = identity.get("request_sha256")
    if not isinstance(request_sha, str) or len(request_sha) != 64:
        raise BundleError("bundle request provenance is invalid")
    imported_root = port_dir / "portable" / request_sha[:16]
    for name, envelope_value, payload_path, tuple_evidence in imports:
        destination = imported_root / payload_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(payload_path, destination)
        imported = copy.deepcopy(envelope_value)
        imported["path"] = destination.relative_to(port_dir).as_posix()
        imported["dependency_paths"] = {}
        destination_artifacts[name] = imported
        if tuple_evidence:
            evidence[
                (name, tuple_evidence["target"], tuple_evidence["precision"])
            ] = tuple_evidence
    report["portable_evidence"] = sorted(
        evidence.values(),
        key=lambda item: (item["target"], item["precision"], item["artifact"]),
    )
    _write_json(report_path, report)
    return report_path
