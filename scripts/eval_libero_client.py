#!/usr/bin/env python3
"""Client-only, resumable LIBERO accuracy evaluation over websocket.

This self-contained script creates LIBERO tasks, builds observations, connects
an unmodified OpenPI websocket client to an already-running policy server,
executes action chunks, measures task success, and writes a resumable JSONL
ledger plus aggregate summary.

It never loads a checkpoint, imports ``apxinf``, selects a CUDA device, or starts
a server. The server must be launched separately.

    python scripts/eval_libero_client.py \\
        --host <server-host> --port 8000 --precision bf16 \\
        --suite libero_10 --tasks all --trials-per-task 10 \\
        --results-jsonl out/libero.jsonl \\
        --summary-json out/libero.summary.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import sys
import time
import traceback
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# --- rollout protocol constants (OpenPI's public PI0.5 LIBERO configuration) ---
LIBERO_ACTION_DIM = 7
MAX_STEPS = 520
WAIT_STEPS = 10
REPLAN_STEPS = 5
IMAGE_SIZE = 224

#: The five LIBERO task suites ``--suite all`` expands to, in a stable order.
ALL_SUITES = (
    "libero_10",
    "libero_90",
    "libero_spatial",
    "libero_object",
    "libero_goal",
)

LedgerKey = Tuple[str, int, int]  # (suite, task_id, trial_id)


# --- LIBERO harness (inlined; was scripts/libero_harness.py) ------------------


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / denominator).astype(np.float32)


def resize_images(base: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    # LIBERO camera arrays are upside down relative to the training convention.
    def resize_with_pad(image: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(image[::-1, ::-1])
        height, width = image.shape[:2]
        ratio = max(width / IMAGE_SIZE, height / IMAGE_SIZE)
        resized_height = int(height / ratio)
        resized_width = int(width / ratio)
        resized = np.asarray(
            Image.fromarray(image).resize(
                (resized_width, resized_height), resample=Image.Resampling.BILINEAR
            )
        )
        canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        offset_y = (IMAGE_SIZE - resized_height) // 2
        offset_x = (IMAGE_SIZE - resized_width) // 2
        canvas[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
        ] = resized
        return canvas

    return np.stack([resize_with_pad(base), resize_with_pad(wrist)], axis=0)


def completed_runs(path: pathlib.Path, precision: str) -> dict[LedgerKey, dict]:
    """Load the ``status == "completed"`` rows from a resumable ledger.

    Keyed by ``(suite, task_id, trial_id)`` so one ledger can hold several suites
    without ``task_id=0`` colliding across them. The precision guard rejects a
    ledger written at a different precision than the one requested now.
    """
    result: dict[LedgerKey, dict] = {}
    if not path.exists():
        return result
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        item_precision = item.get("precision", "fp8")
        if item_precision != precision:
            raise ValueError(
                f"ledger precision is {item_precision!r}, requested {precision!r} "
                f"at line {line_number}"
            )
        if item.get("status") == "completed":
            key: LedgerKey = (
                str(item["suite"]),
                int(item["task_id"]),
                int(item["trial_id"]),
            )
            if key in result:
                raise ValueError(f"duplicate completed run {key} at line {line_number}")
            result[key] = item
    return result


def append_record(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_summary(
    path: pathlib.Path,
    ledger: dict[LedgerKey, dict],
    expected_keys: set[LedgerKey],
    precision: str,
    transport: str,
) -> None:
    """Write the aggregate summary, grouped per-suite then per-task."""
    per_suite: dict[str, dict] = {}
    for suite in sorted({key[0] for key in expected_keys}):
        per_task = {}
        for task_id in sorted({t for (s, t, _) in expected_keys if s == suite}):
            rows = [
                row
                for (s, t, _), row in ledger.items()
                if s == suite and t == task_id
            ]
            successes = sum(bool(row["success"]) for row in rows)
            per_task[str(task_id)] = {
                "completed": len(rows),
                "successes": successes,
                "success_rate": (successes / len(rows) if rows else None),
            }
        suite_rows = [row for (s, _, _), row in ledger.items() if s == suite]
        suite_successes = sum(bool(row["success"]) for row in suite_rows)
        per_suite[suite] = {
            "completed": len(suite_rows),
            "successes": suite_successes,
            "success_rate": (
                suite_successes / len(suite_rows) if suite_rows else None
            ),
            "per_task": per_task,
        }
    rows = list(ledger.values())
    document = {
        "schema": "apxinf.libero-eval.v2",
        "suites": sorted({key[0] for key in expected_keys}),
        "transport": transport,
        "precision": precision,
        "expected_runs": len(expected_keys),
        "completed_runs": len(rows),
        "missing_runs": [
            {"suite": suite, "task_id": task, "trial_id": trial}
            for suite, task, trial in sorted(expected_keys - set(ledger))
        ],
        "successes": sum(bool(row["success"]) for row in rows),
        "success_rate": (
            sum(bool(row["success"]) for row in rows) / len(rows) if rows else None
        ),
        "per_suite": per_suite,
        "timing": _aggregate_timing(rows),
        "updated_unix_seconds": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _aggregate_timing(rows: list[dict]) -> dict:
    """Roll the per-episode latency split up to totals + per-inference means.

    Each ``replan`` is one inference call, so per-call means divide a segment's
    summed seconds by the total replan count. Segments are the four
    non-overlapping parts recorded by :func:`run_episode`: ``preprocess``
    (client), ``model`` (pure bare-model), ``server_processor`` (server/policy
    pre/post pipeline) and ``websocket_transport`` (send/recv/serialize).
    ``inference`` is the client's full round-trip wall clock, reported for
    cross-check, not summed into the split. Rows missing a segment contribute 0.
    """
    completed = [row for row in rows if row.get("status") == "completed"]
    total_replans = sum(int(row.get("replans", 0)) for row in completed)
    segments = (
        "preprocess_seconds",
        "model_seconds",
        "server_processor_seconds",
        "websocket_transport_seconds",
        "inference_seconds",
    )
    totals = {
        name: sum(float(row.get(name, 0.0)) for row in completed) for name in segments
    }
    per_call_ms = {
        name.replace("_seconds", "_ms"): (
            (value / total_replans * 1000.0) if total_replans else None
        )
        for name, value in totals.items()
    }
    return {
        "episodes": len(completed),
        "total_inference_calls": total_replans,
        "total_seconds": {name: round(value, 6) for name, value in totals.items()},
        "per_call_ms": {
            name: (round(value, 4) if value is not None else None)
            for name, value in per_call_ms.items()
        },
    }


def make_env(task, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(seed)
    return env


# --- websocket client ---------------------------------------------------------


def _observation(base, wrist, state, prompt) -> dict:
    """Build the OpenPI LIBERO observation sent over the wire."""
    return {
        "observation/image": base,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
    }


class LiberoWebsocketClient:
    """Reach an OpenPI-compatible server through the unmodified ``openpi_client``."""

    def __init__(self, host: str, port: int, expected_precision: str) -> None:
        from openpi_client import websocket_client_policy

        # WebsocketClientPolicy honours the ambient proxy; exempt the target host
        # so a loopback / LAN server is reached directly (the old evaluator did
        # the same).
        for variable in ("NO_PROXY", "no_proxy"):
            entries = [item for item in os.environ.get(variable, "").split(",") if item]
            if host not in entries:
                entries.append(host)
            os.environ[variable] = ",".join(entries)
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.metadata = self._client.get_server_metadata()
        actual_precision = self.metadata.get("precision")
        if actual_precision != expected_precision:
            self.close()
            raise RuntimeError(
                f"server precision is {actual_precision!r}, expected {expected_precision!r}"
            )
        try:
            validate_libero_server_metadata(self.metadata)
        except Exception:
            self.close()
            raise

    def infer(self, base, wrist, state, prompt) -> Tuple[np.ndarray, dict]:
        response = self._client.infer(_observation(base, wrist, state, prompt))
        actions = np.asarray(response["actions"], dtype=np.float32)
        # Only OpenPI-contract keys ride the wire; ``policy_timing`` /
        # ``server_timing`` are the server's tolerated diagnostic namespaces.
        #   policy_timing.infer_ms  = pure model (bare-model infer_rgb)
        #   policy_timing.policy_ms = whole policy (pre + model + post)
        #   server_timing.infer_ms  = event-loop wall clock of policy.infer
        # Missing keys degrade gracefully to zero / to the coarser field.
        policy_timing = response.get("policy_timing", {}) or {}
        server_timing = response.get("server_timing", {}) or {}
        model_ms = float(policy_timing.get("infer_ms", 0.0))
        server_total_ms = float(policy_timing.get("policy_ms", model_ms))
        server_compute_ms = float(server_timing.get("infer_ms", server_total_ms))
        return actions, {
            "model_seconds": model_ms / 1000.0,
            # Server compute the client can see beyond the pure model = the
            # server-side processor pipeline (event-loop wall clock minus model).
            "server_processor_seconds": max(0.0, server_compute_ms - model_ms) / 1000.0,
        }

    def close(self) -> None:
        connection = getattr(self._client, "_ws", None)
        if connection is not None:
            connection.close()


def validate_libero_server_metadata(metadata: dict) -> None:
    """Reject a published server contract that cannot run LIBERO correctly.

    Third-party OpenPI servers do not necessarily publish every ApxInf metadata
    field, so absent fields remain compatible.  Once a server publishes a field,
    however, a mismatch is actionable and must fail before an expensive rollout.
    """
    expected = {
        "protocol": "openpi.websocket_policy",
        "robot": "franka_libero",
        "image_keys": ["observation/image", "observation/wrist_image"],
        "state_key": "observation/state",
        "action_dim": LIBERO_ACTION_DIM,
    }
    mismatches = []
    for key, expected_value in expected.items():
        if key not in metadata:
            continue
        actual_value = metadata[key]
        if key == "image_keys" and not isinstance(actual_value, list):
            try:
                actual_value = list(actual_value)
            except TypeError:
                pass
        if actual_value != expected_value:
            mismatches.append(f"{key}={actual_value!r} (expected {expected_value!r})")
    if mismatches:
        raise RuntimeError(
            "server wire contract is not LIBERO-compatible: " + "; ".join(mismatches)
        )


# --- rollout ------------------------------------------------------------------


def run_episode(
    env,
    initial_state: np.ndarray,
    suite: str,
    task_id: int,
    trial_id: int,
    prompt: str,
    client: LiberoWebsocketClient,
    transport: str,
    seed: int,
) -> dict:
    episode_started = time.perf_counter()
    env.reset()
    observation = env.set_init_state(initial_state)
    dummy_action = [0.0] * 6 + [-1.0]
    for _ in range(WAIT_STEPS):
        observation, _, _, _ = env.step(dummy_action)

    action_plan: collections.deque[np.ndarray] = collections.deque()
    success = False
    action_steps = 0
    replans = 0
    # Timing is split into four non-overlapping segments so the delivery report
    # can attribute latency:
    #   preprocess  = client-side camera resize + state assembly (this process)
    #   model       = pure bare-model infer_rgb (reported by the server)
    #   server_proc = server/policy pre/post processor pipeline
    #   transport   = round_trip - server compute
    # round_trip (== inference_seconds) is the wall clock around client.infer and
    # equals transport + server_proc + model up to measurement noise.
    preprocess_seconds = 0.0
    inference_seconds = 0.0
    model_seconds = 0.0
    server_processor_seconds = 0.0
    transport_seconds = 0.0
    first_action_checksum = None

    while action_steps < MAX_STEPS:
        if not action_plan:
            preprocess_started = time.perf_counter()
            images = resize_images(
                observation["agentview_image"],
                observation["robot0_eye_in_hand_image"],
            )
            state = np.concatenate(
                (
                    observation["robot0_eef_pos"],
                    quat_to_axis_angle(observation["robot0_eef_quat"]),
                    observation["robot0_gripper_qpos"],
                )
            ).astype(np.float32, copy=False)
            preprocess_seconds += time.perf_counter() - preprocess_started

            request_started = time.perf_counter()
            actions, timing = client.infer(images[0], images[1], state, prompt)
            round_trip_seconds = time.perf_counter() - request_started
            inference_seconds += round_trip_seconds
            # The action horizon is a checkpoint property (metadata ``action_horizon``),
            # not a fixed 10: the public OpenPI pi0.5 LIBERO config emits H=10, but the
            # native ``pi05_libero_base`` checkpoints emit H=50. The rollout only
            # consumes ``REPLAN_STEPS`` actions per chunk, so any horizon >=
            # REPLAN_STEPS is valid; we only require the correct action width.
            if (
                actions.ndim != 2
                or actions.shape[1] != LIBERO_ACTION_DIM
                or actions.shape[0] < REPLAN_STEPS
            ):
                raise ValueError(
                    f"expected actions (>= {REPLAN_STEPS}, {LIBERO_ACTION_DIM}), "
                    f"got {actions.shape}"
                )
            if not np.isfinite(actions).all():
                raise FloatingPointError("server returned non-finite actions")

            segment_model = float(timing.get("model_seconds", 0.0))
            segment_processor = float(timing.get("server_processor_seconds", 0.0))
            model_seconds += segment_model
            server_processor_seconds += segment_processor
            transport_seconds += max(
                0.0, round_trip_seconds - segment_model - segment_processor
            )
            if first_action_checksum is None:
                first_action_checksum = float(np.abs(actions).sum())
            action_plan.extend(actions[:REPLAN_STEPS])
            replans += 1

        action = action_plan.popleft()
        observation, _, done, _ = env.step(action.tolist())
        action_steps += 1
        if done:
            success = True
            break

    return {
        "status": "completed",
        "suite": suite,
        "task_id": task_id,
        "trial_id": trial_id,
        "prompt": prompt,
        "success": success,
        "action_steps": action_steps,
        "replans": replans,
        "preprocess_seconds": preprocess_seconds,
        "inference_seconds": inference_seconds,
        "model_seconds": model_seconds,
        "server_processor_seconds": server_processor_seconds,
        "websocket_transport_seconds": transport_seconds,
        "elapsed_seconds": time.perf_counter() - episode_started,
        "first_action_abs_checksum": first_action_checksum,
        "transport": transport,
        "image_input": "openpi_uint8_hwc",
        "seed": seed,
    }


# --- argument parsing + suite/task resolution ---------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--precision", choices=("fp8", "bf16", "int8"), required=True)
    parser.add_argument("--suite", default="libero_10", choices=(*ALL_SUITES, "all"))
    parser.add_argument(
        "--tasks",
        default="all",
        help="'all' (default) or a comma list of task ids, applied within each suite",
    )
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--results-jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--summary-json", required=True, type=pathlib.Path)
    parser.add_argument("--host", default="127.0.0.1", help="policy server host")
    parser.add_argument("--port", type=int, default=8000, help="policy server port")
    args = parser.parse_args(argv)
    if args.trials_per_task <= 0 or args.trials_per_task > 50:
        parser.error("--trials-per-task must be in 1..=50")
    return args


def resolve_suites(name: str) -> list[str]:
    return list(ALL_SUITES) if name == "all" else [name]


def resolve_task_ids(spec: str, n_tasks: int, suite: str) -> list[int]:
    """Task ids for one suite: ``all`` -> ``0..n_tasks-1``; else a validated list."""
    if spec.strip() == "all":
        return list(range(n_tasks))
    task_ids = [int(value) for value in spec.split(",") if value.strip()]
    if sorted(set(task_ids)) != sorted(task_ids):
        raise ValueError(f"--tasks has duplicate ids: {task_ids}")
    out_of_range = [t for t in task_ids if not 0 <= t < n_tasks]
    if out_of_range:
        raise ValueError(
            f"--tasks {out_of_range} out of range for suite {suite!r} (0..{n_tasks - 1})"
        )
    return task_ids


# --- main ---------------------------------------------------------------------


def run_evaluation(args: argparse.Namespace) -> None:
    transport = "openpi_websocket"

    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    # Resolve every (suite -> instance, task_ids) up front so the expected-key set
    # and the rollout iterate the exact same scope.
    suites: dict[str, object] = {}
    task_ids_by_suite: dict[str, list[int]] = {}
    for name in resolve_suites(args.suite):
        suite = benchmark_dict[name]()
        suites[name] = suite
        task_ids_by_suite[name] = resolve_task_ids(args.tasks, suite.n_tasks, name)

    expected_keys: set[LedgerKey] = {
        (name, task_id, trial_id)
        for name, task_ids in task_ids_by_suite.items()
        for task_id in task_ids
        for trial_id in range(args.trials_per_task)
    }
    ledger = completed_runs(args.results_jsonl, args.precision)
    unexpected = set(ledger) - expected_keys
    if unexpected:
        raise ValueError(
            f"ledger contains runs outside requested scope: {sorted(unexpected)}"
        )
    write_summary(args.summary_json, ledger, expected_keys, args.precision, transport)

    client = LiberoWebsocketClient(args.host, args.port, args.precision)
    print(f"client=websocket server_metadata={client.metadata}", flush=True)
    try:
        for name, suite in suites.items():
            for task_id in task_ids_by_suite[name]:
                task = suite.get_task(task_id)
                prompt = str(task.language)
                pending = [
                    trial_id
                    for trial_id in range(args.trials_per_task)
                    if (name, task_id, trial_id) not in ledger
                ]
                if not pending:
                    print(f"{name} task {task_id}: already complete", flush=True)
                    continue
                print(f"{name} task {task_id}: pending trials {pending}", flush=True)
                initial_states = suite.get_task_init_states(task_id)
                env = make_env(task, args.seed)
                try:
                    for trial_id in pending:
                        for attempt in range(1, args.max_attempts + 1):
                            try:
                                record = run_episode(
                                    env,
                                    initial_states[trial_id],
                                    name,
                                    task_id,
                                    trial_id,
                                    prompt,
                                    client,
                                    transport,
                                    args.seed,
                                )
                                record["attempt"] = attempt
                                record["precision"] = args.precision
                                append_record(args.results_jsonl, record)
                                ledger[(name, task_id, trial_id)] = record
                                write_summary(
                                    args.summary_json, ledger, expected_keys,
                                    args.precision, transport,
                                )
                                print(
                                    f"{name} task={task_id} trial={trial_id} "
                                    f"success={record['success']} steps={record['action_steps']} "
                                    f"replans={record['replans']} "
                                    f"completed={len(ledger)}/{len(expected_keys)}",
                                    flush=True,
                                )
                                break
                            except Exception as error:
                                failure = {
                                    "status": "technical_error",
                                    "suite": name,
                                    "task_id": task_id,
                                    "trial_id": trial_id,
                                    "attempt": attempt,
                                    "precision": args.precision,
                                    "transport": transport,
                                    "error": repr(error),
                                    "traceback": traceback.format_exc(),
                                    "time_unix_seconds": time.time(),
                                }
                                append_record(args.results_jsonl, failure)
                                print(
                                    f"{name} task={task_id} trial={trial_id} attempt={attempt} "
                                    f"ERROR: {error}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                if attempt == args.max_attempts:
                                    raise
                finally:
                    env.close()
    finally:
        client.close()

    write_summary(args.summary_json, ledger, expected_keys, args.precision, transport)
    missing = expected_keys - set(ledger)
    if missing:
        raise RuntimeError(f"evaluation incomplete; missing {sorted(missing)}")
    successes = sum(bool(record["success"]) for record in ledger.values())
    suites_label = ",".join(suites)
    print(
        f"LIBERO [{suites_label}] complete: {successes}/{len(expected_keys)} successes",
        flush=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_evaluation(parse_args(argv))


if __name__ == "__main__":
    main()
