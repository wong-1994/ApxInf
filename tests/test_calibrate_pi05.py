import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import calibrate_pi05


class CalibratePi05Test(unittest.TestCase):
    def test_unified_workflow_maps_observation_to_manifest(self):
        import apxinf

        class Model:
            image_size = 4
            action_horizon = 2
            action_dim = 3

            @staticmethod
            def _calibration_plan():
                return ["vision.patch_input"]

        class Policy:
            model = Model()
            image_keys = ("observation/image",)
            prompt_key = "prompt"
            state_key = "observation/state"
            discrete_state = False

            def calibrate_observation(self, observation, *, noise):
                self.last_observation = observation
                self.last_noise = noise
                return {"vision.patch_input": 4.0}

            def close(self):
                pass

        policy = Policy()
        captured_options = {}

        class Pi05Policy:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                captured_options.update(kwargs)
                return policy

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"weights")
            sample = root / "sample.npz"
            np.savez(
                sample,
                **{
                    "observation/image": np.zeros((4, 4, 3), np.uint8),
                    "prompt": np.asarray("move"),
                },
            )
            output = root / "profile.json"
            with mock.patch.object(apxinf, "Pi05Policy", Pi05Policy):
                calibrate_pi05.main(
                    [
                        "--model-dir",
                        str(root),
                        "--input",
                        str(sample),
                        "--output",
                        str(output),
                        "--data-id",
                        "dataset:test-v1",
                        "--image-key",
                        "observation/image",
                        "--num-views",
                        "1",
                        "--prompt-key",
                        "prompt",
                        "--action-horizon",
                        "2",
                        "--seed",
                        "9",
                    ]
                )

            document = json.loads(output.read_text())
            self.assertEqual(document["calibration_data"]["identity"], "dataset:test-v1")
            self.assertEqual(document["observed_sites"], ["vision.patch_input"])
            self.assertEqual(policy.last_observation["prompt"], "move")
            self.assertEqual(policy.last_noise.shape, (2, 3))
            self.assertEqual(captured_options["image_keys"], ("observation/image",))
            self.assertEqual(captured_options["num_views"], 1)
            self.assertEqual(captured_options["action_horizon"], 2)

    def test_requires_exactly_one_input_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            for extra in ([], ["--zero-fixture", "--input", str(root / "x.npz")]):
                args = calibrate_pi05.parse_args(["--model-dir", str(root), *extra])
                with self.assertRaisesRegex(ValueError, "--input"):
                    calibrate_pi05.validate_args(args)

    def test_representative_input_rejects_synthetic_data_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            sample = root / "sample.npz"
            np.savez(sample, prompt=np.asarray("move"))
            args = calibrate_pi05.parse_args(
                [
                    "--model-dir",
                    str(root),
                    "--input",
                    str(sample),
                    "--data-id",
                    "synthetic:zero-observation-v1",
                ]
            )
            with self.assertRaisesRegex(ValueError, "cannot use a synthetic"):
                calibrate_pi05.validate_args(args)

    def test_rejects_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "calibration.json").write_text("existing")
            args = calibrate_pi05.parse_args(
                ["--model-dir", str(root), "--zero-fixture"]
            )
            with self.assertRaisesRegex(ValueError, "--force"):
                calibrate_pi05.validate_args(args)

    def test_write_profile_exclusively_protects_against_overwrite_races(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "calibration.json"
            calibrate_pi05.write_profile(output, {"version": 1}, force=False)
            with self.assertRaisesRegex(ValueError, "--force"):
                calibrate_pi05.write_profile(output, {"version": 2}, force=False)
            self.assertEqual(output.read_text(), '{\n  "version": 1\n}\n')

    def test_loads_business_observation_without_preprocessed_tensors(self):
        class Policy:
            image_keys = ("observation/image", "observation/wrist_image")
            prompt_key = "prompt"
            state_key = "observation/state"
            discrete_state = False

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "sample.npz"
            np.savez(
                path,
                **{
                    "observation/image": np.zeros((4, 5, 3), np.uint8),
                    "observation/wrist_image": np.ones((3, 2, 3), np.uint8),
                    "prompt": np.asarray("move the block"),
                },
            )
            args = calibrate_pi05.parse_args(
                ["--model-dir", directory, "--input", str(path)]
            )
            observation = next(iter(calibrate_pi05.load_observations(args, Policy())))
            self.assertEqual(observation["prompt"], "move the block")
            self.assertNotIn("rgb", observation)
            self.assertNotIn("token_ids", observation)

    def test_deterministic_noise_is_stable_per_sample(self):
        class Model:
            action_horizon = 2
            action_dim = 3

        class Policy:
            model = Model()

        first = calibrate_pi05.deterministic_noise(Policy(), 17, 2)
        second = calibrate_pi05.deterministic_noise(Policy(), 17, 2)
        other = calibrate_pi05.deterministic_noise(Policy(), 17, 3)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))

    def test_source_revision_fails_closed_outside_git(self):
        with mock.patch.object(
            calibrate_pi05.subprocess,
            "check_output",
            side_effect=OSError("git unavailable"),
        ):
            with self.assertRaisesRegex(ValueError, "--source-revision"):
                calibrate_pi05.source_revision()
        with self.assertRaisesRegex(ValueError, "real commit"):
            calibrate_pi05.source_revision("unknown")
        self.assertEqual(calibrate_pi05.source_revision("release-1.2.3"), "release-1.2.3")

    def test_aggregation_is_order_independent_and_rejects_non_finite_values(self):
        forward = {}
        reverse = {}
        for records in ({"a": 2.0, "b": 3.0}, {"a": 4.0}):
            calibrate_pi05.merge_records(forward, records)
        for records in ({"a": 4.0}, {"a": 2.0, "b": 3.0}):
            calibrate_pi05.merge_records(reverse, records)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward, {"a": 4.0, "b": 3.0})
        with self.assertRaisesRegex(ValueError, "invalid amax"):
            calibrate_pi05.merge_records({}, {"a": float("nan")})

    def test_manifest_is_complete_and_self_describing(self):
        document = calibrate_pi05.calibration_document(
            {"vision.patch_input": 4.0},
            margin=1.25,
            sample_count=2,
            bootstrap=False,
            required_sites=["vision.patch_input"],
            checkpoint="sha256:abc",
            data_identity="dataset:libero-v1",
            seed=7,
            device="cuda:0",
        )
        self.assertEqual(document["schema"], calibrate_pi05.SCHEMA)
        self.assertEqual(document["model"], {"family": "pi05", "checkpoint": "sha256:abc"})
        self.assertEqual(document["quantization"]["format"], "e4m3fn")
        self.assertEqual(document["quantization"]["statistic"], "absmax")
        self.assertEqual(document["calibration_data"]["sample_count"], 2)
        self.assertEqual(document["seed_policy"]["base_seed"], 7)
        self.assertEqual(document["observed_sites"], ["vision.patch_input"])
        self.assertAlmostEqual(document["scales"]["vision.patch_input"]["scale"], 5 / 448)
        self.assertIn("source_revision", document)
        self.assertIn("device", document)

    def test_manifest_rejects_missing_and_unknown_sites(self):
        with self.assertRaisesRegex(ValueError, "missing=.*b.*unknown=.*c"):
            calibrate_pi05.calibration_document(
                {"a": 1.0, "c": 2.0},
                margin=1.0,
                sample_count=1,
                bootstrap=False,
                required_sites=["a", "b"],
                checkpoint="sha256:abc",
                data_identity="dataset:test",
                seed=0,
                device="cuda:0",
            )

    def test_bootstrap_is_unambiguously_non_production(self):
        document = calibrate_pi05.calibration_document(
            {"vision.patch_input": 0.0, "action.input": 0.0},
            margin=2.35,
            sample_count=1,
            bootstrap=True,
            required_sites=["vision.patch_input", "action.input"],
            checkpoint="sha256:abc",
            data_identity="synthetic:zero-observation-v1",
            seed=0,
            device="cuda:0",
        )
        self.assertFalse(document["calibration_data"]["production"])
        self.assertEqual(document["calibration_data"]["kind"], "synthetic-zero-fixture")
        self.assertAlmostEqual(document["scales"]["vision.patch_input"]["amax"], 1 / 2.35)
        self.assertAlmostEqual(document["scales"]["action.input"]["amax"], 5 / 2.35)
        self.assertAlmostEqual(document["scales"]["vision.patch_input"]["scale"], 1 / 448)
        self.assertAlmostEqual(document["scales"]["action.input"]["scale"], 5 / 448)


if __name__ == "__main__":
    unittest.main()
