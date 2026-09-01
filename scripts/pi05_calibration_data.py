"""Data-source adapters for PI0.5 calibration Observations.

The calibration module consumes ApxInf Observations.  This module is the
optional outer seam that translates storage formats into that contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import pathlib
from typing import Any

import numpy as np
from PIL import Image


def _decode_npz_value(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return np.ascontiguousarray(array)


def load_npz_observations(
    paths: Sequence[pathlib.Path],
) -> tuple[Mapping[str, object], ...]:
    observations = []
    for path in paths:
        with np.load(path, allow_pickle=False) as sample:
            observations.append(
                {name: _decode_npz_value(sample[name]) for name in sample.files}
            )
    return tuple(observations)


def _load_rgb(path: pathlib.Path, *, field: str, line_number: int) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, ValueError) as error:
        raise ValueError(
            f"manifest line {line_number}: cannot load {field} image {path}"
        ) from error


def load_observation_manifest(
    path: pathlib.Path,
    *,
    image_keys: Sequence[str],
    prompt_key: str,
    state_key: str,
) -> tuple[Mapping[str, object], ...]:
    """Load JSONL rows whose fields mirror the public Observation contract.

    Image fields contain paths relative to the manifest (or absolute paths),
    the prompt is a string, and optional state is an inline numeric array.
    """
    observations: list[Mapping[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"manifest line {line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number}: expected a JSON object")

            missing = [key for key in (*image_keys, prompt_key) if key not in row]
            if missing:
                raise ValueError(
                    f"manifest line {line_number}: missing Observation field(s): {missing}"
                )
            prompt = row[prompt_key]
            if not isinstance(prompt, str):
                raise ValueError(
                    f"manifest line {line_number}: {prompt_key} must be a string"
                )

            observation: dict[str, Any] = {prompt_key: prompt}
            for key in image_keys:
                image_value = row[key]
                if not isinstance(image_value, str):
                    raise ValueError(
                        f"manifest line {line_number}: {key} must be an image path"
                    )
                image_path = pathlib.Path(image_value).expanduser()
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                observation[key] = _load_rgb(
                    image_path, field=key, line_number=line_number
                )

            if state_key in row:
                try:
                    state = np.asarray(row[state_key], dtype=np.float32)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"manifest line {line_number}: {state_key} must be numeric"
                    ) from error
                if state.ndim != 1 or not np.all(np.isfinite(state)):
                    raise ValueError(
                        f"manifest line {line_number}: {state_key} must be a finite 1D array"
                    )
                observation[state_key] = np.ascontiguousarray(state)
            observations.append(observation)

    if not observations:
        raise ValueError(f"calibration manifest has no observations: {path}")
    return tuple(observations)


def task_stratified_indices(
    task_indices: Sequence[object], *, sample_count: int, seed: int
) -> list[int]:
    """Choose deterministic, balanced frame indices across every dataset task."""
    if sample_count < 1:
        raise ValueError("calibration sample count must be positive")
    if sample_count > len(task_indices):
        raise ValueError(
            f"requested {sample_count} calibration samples from {len(task_indices)} frames"
        )
    groups: dict[object, list[int]] = {}
    for index, raw_task in enumerate(task_indices):
        task = raw_task.item() if hasattr(raw_task, "item") else raw_task
        groups.setdefault(task, []).append(index)
    if sample_count < len(groups):
        raise ValueError(
            f"--samples={sample_count} cannot cover all {len(groups)} dataset tasks; "
            "pass a deployment-specific dataset split or increase --samples"
        )

    rng = np.random.default_rng(seed)
    tasks = sorted(groups, key=lambda value: (type(value).__name__, repr(value)))
    queues = {task: list(rng.permutation(groups[task])) for task in tasks}
    selected: list[int] = []
    while len(selected) < sample_count:
        progressed = False
        for task in tasks:
            if queues[task] and len(selected) < sample_count:
                selected.append(int(queues[task].pop()))
                progressed = True
        if not progressed:
            break
    return selected


def _open_lerobot_dataset(repo_id: str, root: pathlib.Path | None):
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as error:
        raise ImportError(
            "the optional --dataset adapter requires LeRobot; install it or use "
            "--manifest/--input-dir"
        ) from error
    options = {"repo_id": repo_id, "return_uint8": True}
    if root is not None:
        options["root"] = root
    try:
        return LeRobotDataset(**options)
    except TypeError as error:
        if "return_uint8" not in str(error):
            raise
        options.pop("return_uint8")
        return LeRobotDataset(**options)


def _task_indices(dataset) -> list[object] | None:
    try:
        values = dataset.hf_dataset["task_index"]
    except (AttributeError, KeyError, TypeError):
        return None
    return [value.item() if hasattr(value, "item") else value for value in values]


def load_lerobot_observations(
    repo_id: str,
    *,
    root: pathlib.Path | None,
    image_keys: Sequence[str],
    sample_count: int | None,
    seed: int,
) -> tuple[Mapping[str, object], ...]:
    """Optional LeRobot adapter; calibration itself has no LeRobot dependency."""
    from apxinf.adapters.lerobot import observation_to_apxinf

    dataset = _open_lerobot_dataset(repo_id, root)
    tasks = _task_indices(dataset)
    if tasks is None:
        if sample_count is None:
            raise ValueError(
                "LeRobot dataset has no task_index metadata; pass --samples explicitly"
            )
        tasks = [0] * len(dataset)
    selected_count = sample_count if sample_count is not None else len(set(tasks))
    indices = task_stratified_indices(tasks, sample_count=selected_count, seed=seed)
    return tuple(
        observation_to_apxinf(dataset[index], image_keys=image_keys)
        for index in indices
    )
