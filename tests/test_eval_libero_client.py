from __future__ import annotations

import contextlib
import io
import pathlib
import sys
import types
import unittest
from unittest import mock

import numpy as np


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from eval_libero_client import (  # noqa: E402
    LiberoWebsocketClient,
    parse_args,
    run_episode,
    validate_libero_server_metadata,
)


class ClientOnlyCliTest(unittest.TestCase):
    def test_is_self_contained_instead_of_importing_another_script(self) -> None:
        source = (REPOSITORY_ROOT / "scripts" / "eval_libero_client.py").read_text()
        self.assertNotIn("from eval_libero import", source)
        self.assertIn("WebsocketClientPolicy", source)
        self.assertIn("def run_episode", source)
        self.assertIn('"success_rate"', source)

    def test_exposes_only_remote_server_selection(self) -> None:
        args = parse_args(
            [
                "--precision",
                "bf16",
                "--host",
                "inference.example",
                "--port",
                "9000",
                "--results-jsonl",
                "results.jsonl",
                "--summary-json",
                "summary.json",
            ]
        )
        self.assertEqual(args.host, "inference.example")
        self.assertEqual(args.port, 9000)
        self.assertFalse(hasattr(args, "backend"))
        self.assertFalse(hasattr(args, "model_dir"))
        self.assertFalse(hasattr(args, "device"))

    def test_rejects_model_loading_flags(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--precision",
                        "bf16",
                        "--model-dir",
                        "/checkpoint",
                        "--results-jsonl",
                        "results.jsonl",
                        "--summary-json",
                        "summary.json",
                    ]
                )


class LiberoServerContractTest(unittest.TestCase):
    def test_starts_the_openpi_websocket_client(self) -> None:
        class FakeSocket:
            closed = False

            def close(self) -> None:
                self.closed = True

        class FakePolicy:
            def __init__(self, host: str, port: int) -> None:
                self.endpoint = (host, port)
                self._ws = FakeSocket()

            def get_server_metadata(self) -> dict:
                return {
                    "precision": "bf16",
                    "robot": "franka_libero",
                    "image_keys": [
                        "observation/image",
                        "observation/wrist_image",
                    ],
                    "state_key": "observation/state",
                    "action_dim": 7,
                }

        package = types.ModuleType("openpi_client")
        package.websocket_client_policy = types.SimpleNamespace(
            WebsocketClientPolicy=FakePolicy
        )
        with mock.patch.dict(sys.modules, {"openpi_client": package}):
            client = LiberoWebsocketClient("inference.example", 9000, "bf16")

        self.assertEqual(client._client.endpoint, ("inference.example", 9000))
        client.close()
        self.assertTrue(client._client._ws.closed)

    def test_accepts_the_franka_libero_contract(self) -> None:
        validate_libero_server_metadata(
            {
                "protocol": "openpi.websocket_policy",
                "robot": "franka_libero",
                "image_keys": ["observation/image", "observation/wrist_image"],
                "state_key": "observation/state",
                "action_dim": 7,
            }
        )

    def test_rejects_a_different_embodiment_before_rollout(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "robot='unitree_g1'"):
            validate_libero_server_metadata(
                {"robot": "unitree_g1", "action_dim": 16}
            )


class AccuracyRolloutTest(unittest.TestCase):
    class FakeEnv:
        def __init__(self) -> None:
            self.steps = 0
            self.observation = {
                "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
                "robot0_eef_pos": np.zeros(3, dtype=np.float32),
                "robot0_eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float32),
                "robot0_gripper_qpos": np.zeros(1, dtype=np.float32),
            }

        def reset(self) -> None:
            self.steps = 0

        def set_init_state(self, _state):
            return self.observation

        def step(self, _action):
            self.steps += 1
            return self.observation, 0.0, self.steps > 10, {}

    class FakeClient:
        def infer(self, _base, _wrist, _state, _prompt):
            return np.zeros((5, 7), dtype=np.float32), {}

    def test_records_success_from_the_libero_rollout(self) -> None:
        record = run_episode(
            self.FakeEnv(),
            np.zeros(1),
            "libero_10",
            0,
            0,
            "pick up the object",
            self.FakeClient(),
            "openpi_websocket",
            7,
        )
        self.assertTrue(record["success"])
        self.assertEqual(record["action_steps"], 1)
        self.assertEqual(record["replans"], 1)


if __name__ == "__main__":
    unittest.main()
