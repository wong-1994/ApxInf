#!/usr/bin/env python3
"""Run a trusted source implementation through the ApxInf reference contract.

This generic adapter is copied into private Port state. It imports source code by
an explicitly declared path; it never copies that source into ApxInf.
"""

from __future__ import annotations

import argparse
import importlib
import json
import socket
import sys
import traceback
from pathlib import Path
from typing import Any


ADAPTER_CONTRACT_VERSION = "1.0"
SEMANTIC_FIELDS = (
    "operator_traces",
    "preprocessing",
    "tokenization",
    "normalization",
    "stochastic_inputs",
    "schedules",
    "custom_operators",
    "dynamic_branches",
)
OBJECT_SEMANTIC_FIELDS = {"preprocessing", "tokenization", "normalization"}


class ReferenceAdapter:
    """Stable private contract around a source-specific reference module."""

    def __init__(self, source_root: Path, entrypoint: str) -> None:
        source_root = source_root.resolve()
        entrypoint_path = (source_root / entrypoint).resolve()
        if not entrypoint_path.is_relative_to(source_root):
            raise ValueError("reference entrypoint escapes the trusted source root")
        relative_entrypoint = entrypoint_path.relative_to(source_root)
        if relative_entrypoint.suffix != ".py":
            raise ValueError("reference entrypoint must be a Python module")
        module_parts = relative_entrypoint.with_suffix("").parts
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        if not module_parts or any(not part.isidentifier() for part in module_parts):
            raise ValueError("reference entrypoint must use importable module names")
        sys.path.insert(0, str(source_root))
        self._module = importlib.import_module(".".join(module_parts))

    def load(self, checkpoint_path: str) -> Any:
        return self._call("load", checkpoint_path)

    def preprocess(self, profile: dict[str, Any]) -> Any:
        return self._call("preprocess", profile)

    def infer(self, model: Any, inputs: Any) -> Any:
        return self._call("infer", model, inputs)

    def capture_intermediates(self, model: Any, inputs: Any) -> Any:
        return self._call("capture_intermediates", model, inputs)

    def postprocess(self, output: Any) -> Any:
        return self._call("postprocess", output)

    def describe(self) -> dict[str, Any]:
        description = self._call("describe")
        if not isinstance(description, dict):
            raise TypeError("describe() must return an object")
        return description

    def _call(self, name: str, *args: Any) -> Any:
        function = getattr(self._module, name, None)
        if not callable(function):
            raise AttributeError(f"reference entrypoint must define callable {name}()")
        return function(*args)


def disable_runtime_network() -> None:
    """Block accidental Python socket use by trusted reference code."""

    original_socket = socket.socket

    class OfflineSocket(original_socket):
        def connect(self, *args: Any, **kwargs: Any) -> Any:
            raise PermissionError("reference runtime network access is disabled")

        def connect_ex(self, *args: Any, **kwargs: Any) -> Any:
            raise PermissionError("reference runtime network access is disabled")

        def sendto(self, *args: Any, **kwargs: Any) -> Any:
            raise PermissionError("reference runtime network access is disabled")

    def offline(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("reference runtime network access is disabled")

    socket.socket = OfflineSocket
    socket.create_connection = offline
    socket.getaddrinfo = offline
    socket.gethostbyaddr = offline
    socket.gethostbyname = offline
    socket.gethostbyname_ex = offline


def tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return [int(dimension) for dimension in shape]
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        dimensions = []
        current = value
        while isinstance(current, (list, tuple)):
            dimensions.append(len(current))
            current = current[0] if current else None
        return dimensions
    return None


def value_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {
                str(key): value_schema(item) for key, item in sorted(value.items())
            },
        }
    shape = tensor_shape(value)
    if shape is not None:
        dtype = getattr(value, "dtype", None)
        return {
            "type": "tensor" if hasattr(value, "dtype") else "array",
            "shape": shape,
            "dtype": str(dtype) if dtype is not None else type(value).__name__,
        }
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def json_capture(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): json_capture(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_capture(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        source_dtype = str(value.dtype)
        converted = value.detach() if callable(getattr(value, "detach", None)) else value
        converted = (
            converted.float()
            if callable(getattr(converted, "float", None))
            else converted
        )
        converted = (
            converted.cpu() if callable(getattr(converted, "cpu", None)) else converted
        )
        converted_dtype = str(getattr(converted, "dtype", ""))
        if converted_dtype not in {"f32", "float32", "torch.float32"}:
            raise TypeError(
                f"cannot capture {source_dtype} tensor as unambiguous f32 data"
            )
        tolist = getattr(converted, "tolist", None)
        if not callable(tolist):
            raise TypeError("tensor capture requires tolist() after f32 conversion")
        return {
            "dtype": "f32",
            "source_dtype": source_dtype,
            "shape": tensor_shape(converted) or [],
            "data": json_capture(tolist()),
        }
    return {"schema": value_schema(value), "repr": repr(value)}


def named_values(model: Any, method_name: str) -> list[tuple[str, Any]]:
    method = getattr(model, method_name, None)
    if not callable(method):
        return []
    try:
        values = method(remove_duplicate=False)
    except TypeError:
        values = method()
    return sorted(
        [(str(name), value) for name, value in values], key=lambda item: item[0]
    )


def storage_key(value: Any) -> tuple[str, int]:
    untyped_storage = getattr(value, "untyped_storage", None)
    if callable(untyped_storage):
        storage = untyped_storage()
        data_ptr = getattr(storage, "data_ptr", None)
        if callable(data_ptr):
            return ("storage", int(data_ptr()))
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        return ("data", int(data_ptr()))
    return ("object", id(value))


def tensor_record(name: str, value: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": tensor_shape(value) or [],
        "dtype": str(getattr(value, "dtype", type(value).__name__)),
        "requires_grad": bool(getattr(value, "requires_grad", False)),
    }


def alias_groups(values: list[tuple[str, Any]]) -> list[list[str]]:
    groups: dict[tuple[str, int], list[str]] = {}
    for name, value in values:
        groups.setdefault(storage_key(value), []).append(name)
    return sorted(
        [sorted(names) for names in groups.values() if len(names) > 1],
        key=lambda names: names[0],
    )


def tied_weight_groups(parameters: list[tuple[str, Any]]) -> list[list[str]]:
    groups: dict[int, list[str]] = {}
    for name, value in parameters:
        groups.setdefault(id(value), []).append(name)
    return sorted(
        [sorted(names) for names in groups.values() if len(names) > 1],
        key=lambda names: names[0],
    )


def module_records(model: Any) -> list[dict[str, str]]:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return [{"name": "", "type": type(model).__qualname__}]
    return sorted(
        [
            {"name": str(name), "type": type(module).__qualname__}
            for name, module in named_modules()
        ],
        key=lambda item: item["name"],
    )


def validate_description(description: dict[str, Any]) -> None:
    missing = sorted(set(SEMANTIC_FIELDS) - description.keys())
    if missing:
        raise ValueError(f"describe() is missing semantic fields: {', '.join(missing)}")
    for field in SEMANTIC_FIELDS:
        expected = dict if field in OBJECT_SEMANTIC_FIELDS else list
        if not isinstance(description[field], expected):
            raise TypeError(f"describe().{field} must be a {expected.__name__}")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def inspect_source(args: argparse.Namespace) -> None:
    disable_runtime_network()
    try:
        adapter = ReferenceAdapter(args.source_root, args.entrypoint)
        model = adapter.load(str(args.checkpoint))
    except Exception as error:
        write_json(
            args.result,
            {
                "status": "load_failure",
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        return

    try:
        profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
        captures = []
        input_schemas = []
        output_schemas = []
        intermediate_schemas = []
        for profile in profiles:
            inputs = adapter.preprocess(profile)
            output = adapter.infer(model, inputs)
            intermediates = adapter.capture_intermediates(model, inputs)
            postprocessed = adapter.postprocess(output)
            captures.append(
                {
                    "profile": profile.get("name"),
                    "inputs": json_capture(inputs),
                    "output": json_capture(output),
                    "intermediates": json_capture(intermediates),
                    "postprocessed": json_capture(postprocessed),
                }
            )
            input_schemas.append(
                {"profile": profile.get("name"), "schema": value_schema(inputs)}
            )
            output_schemas.append(
                {
                    "profile": profile.get("name"),
                    "raw": value_schema(output),
                    "postprocessed": value_schema(postprocessed),
                }
            )
            intermediate_schemas.append(
                {"profile": profile.get("name"), "schema": value_schema(intermediates)}
            )

        parameters = named_values(model, "named_parameters")
        buffers = named_values(model, "named_buffers")
        description = adapter.describe()
        validate_description(description)
        inventory = {
            "schema_version": "1.0",
            "adapter_contract_version": ADAPTER_CONTRACT_VERSION,
            "source": {
                "revision": args.source_revision,
                "sha256": args.source_sha256,
                "entrypoint": args.entrypoint,
            },
            "checkpoint": {"sha256": args.checkpoint_sha256},
            "modules": module_records(model),
            "parameters": [tensor_record(name, value) for name, value in parameters],
            "buffers": [tensor_record(name, value) for name, value in buffers],
            "aliases": alias_groups(parameters + buffers),
            "tied_weights": tied_weight_groups(parameters),
            "input_schema": input_schemas,
            "output_schema": output_schemas,
            "intermediate_schema": intermediate_schemas,
        }
        for field in SEMANTIC_FIELDS:
            inventory[field] = description.get(
                field,
                {} if field in OBJECT_SEMANTIC_FIELDS else [],
            )
        write_json(args.capture, {"schema_version": "1.0", "profiles": captures})
        write_json(args.inventory, inventory)
        write_json(args.result, {"status": "success"})
    except Exception as error:
        write_json(
            args.result,
            {
                "status": "trace_failure",
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("--source-root", type=Path, required=True)
    command.add_argument("--entrypoint", required=True)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--profiles", type=Path, required=True)
    command.add_argument("--inventory", type=Path, required=True)
    command.add_argument("--capture", type=Path, required=True)
    command.add_argument("--result", type=Path, required=True)
    command.add_argument("--source-revision", required=True)
    command.add_argument("--source-sha256", required=True)
    command.add_argument("--checkpoint-sha256", required=True)
    return command


if __name__ == "__main__":
    inspect_source(parser().parse_args())
