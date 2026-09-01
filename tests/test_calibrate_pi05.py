import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from apxinf.calibration import CalibrationPlan
from scripts import calibrate_pi05


class CalibratePi05Test(unittest.TestCase):
    def test_task_stratified_indices_cover_tasks_before_repeating(self):
        indices = calibrate_pi05.task_stratified_indices(
            [0, 0, 0, 1, 1, 2], sample_count=5, seed=7
        )

        self.assertEqual(len(indices), 5)
        selected_tasks = [
            [0, 0, 0, 1, 1, 2][index]
            for index in indices
        ]
        self.assertEqual(set(selected_tasks[:3]), {0, 1, 2})
        self.assertLessEqual(max(selected_tasks.count(task) for task in set(selected_tasks)), 2)

    def test_dataset_mode_loads_lerobot_records_without_npz_export(self):
        class MetadataColumn:
            def __init__(self):
                self.values = [0, 0, 1, 1]

            def __getitem__(self, name):
                if name != "task_index":
                    raise KeyError(name)
                return self.values

        class Dataset:
            repo_id = "company/libero"
            revision = "v3.0"
            hf_dataset = MetadataColumn()

            def __len__(self):
                return 4

            def __getitem__(self, index):
                return {
                    "observation.images.base_0_rgb": np.full(
                        (3, 4, 3), index, dtype=np.uint8
                    ),
                    "observation.images.left_wrist_0_rgb": np.full(
                        (3, 4, 3), index + 10, dtype=np.uint8
                    ),
                    "observation.state": np.arange(8, dtype=np.float32),
                    "task": f"task {index // 2}",
                }

        class Policy:
            image_keys = ("observation/image", "observation/wrist_image")

        args = calibrate_pi05.parse_args(
            [
                "--model-dir",
                "/model",
                "--dataset",
                "company/libero",
                "--samples",
                "2",
            ]
        )
        with mock.patch.object(
            calibrate_pi05, "_open_lerobot_dataset", return_value=Dataset()
        ):
            observations, identity = calibrate_pi05.resolve_observations(args, Policy())

        self.assertEqual(len(observations), 2)
        self.assertEqual({observation["prompt"] for observation in observations}, {"task 0", "task 1"})
        self.assertTrue(identity.startswith("sha256:"))
        self.assertEqual(
            set(observations[0]),
            {
                "observation/image",
                "observation/wrist_image",
                "observation/state",
                "prompt",
            },
        )

    def test_dataset_without_task_metadata_requires_explicit_sample_count(self):
        class Dataset:
            def __len__(self):
                return 4

        class Policy:
            image_keys = ("observation/image",)

        args = calibrate_pi05.parse_args(
            ["--model-dir", "/model", "--dataset", "company/unlabeled"]
        )
        with mock.patch.object(
            calibrate_pi05, "_open_lerobot_dataset", return_value=Dataset()
        ):
            with self.assertRaisesRegex(ValueError, "no task_index.*--samples"):
                calibrate_pi05.resolve_observations(args, Policy())

    def test_input_directory_expands_npz_files_in_stable_order(self):
        class Policy:
            image_keys = ("observation/image",)
            prompt_key = "prompt"
            state_key = "observation/state"
            discrete_state = False

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for name, prompt in (("sample-010.npz", "ten"), ("sample-002.npz", "two")):
                np.savez(
                    root / name,
                    **{
                        "observation/image": np.zeros((2, 2, 3), np.uint8),
                        "prompt": np.asarray(prompt),
                    },
                )
            args = calibrate_pi05.parse_args(
                ["--model-dir", directory, "--input-dir", directory]
            )
            observations, identity = calibrate_pi05.resolve_observations(args, Policy())

        self.assertEqual([item["prompt"] for item in observations], ["two", "ten"])
        self.assertTrue(identity.startswith("sha256:"))

    def test_calibration_job_consumes_dataset_source_end_to_end(self):
        class Dataset:
            hf_dataset = {"task_index": [0, 1]}

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return {
                    "observation.images.base": np.full(
                        (2, 2, 3), index, dtype=np.uint8
                    ),
                    "task": f"task {index}",
                }

        class Model:
            image_size = 2
            action_horizon = 2
            action_dim = 3

        class Policy:
            model = Model()
            image_keys = ("observation/image",)
            prompt_key = "prompt"
            state_key = "observation/state"
            discrete_state = False

            def calibration_plan(self):
                return CalibrationPlan.runtime_validated_sites(
                    model_family="pi05",
                    sites=("vision.patch_input",),
                    schema=calibrate_pi05.SCHEMA,
                    seed_algorithm="numpy-pcg64-seed-sequence-v1",
                )

            def collect_calibration(self, observation, context):
                return {"vision.patch_input": float(context.sample_index + 1)}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            output = root / "profile.json"
            args = calibrate_pi05.parse_args(
                [
                    "--model-dir",
                    str(root),
                    "--dataset",
                    "company/libero",
                    "--output",
                    str(output),
                    "--source-revision",
                    "test-revision",
                ]
            )
            with mock.patch.object(
                calibrate_pi05, "_open_lerobot_dataset", return_value=Dataset()
            ):
                result = calibrate_pi05.Pi05CalibrationJob(
                    args, policy_factory=lambda *_args, **_kwargs: Policy()
                ).run()

            document = json.loads(output.read_text())

        self.assertEqual(result.output, output)
        self.assertEqual(document["calibration_data"]["sample_count"], 2)
        self.assertTrue(document["calibration_data"]["identity"].startswith("sha256:"))
        self.assertEqual(document["scales"]["vision.patch_input"]["amax"], 2.0)

    def test_calibration_job_maps_observation_to_manifest(self):
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

            def calibration_plan(self):
                return CalibrationPlan.runtime_validated_sites(
                    model_family="pi05",
                    sites=("vision.patch_input",),
                    schema=calibrate_pi05.SCHEMA,
                    seed_algorithm="numpy-pcg64-seed-sequence-v1",
                )

            def collect_calibration(self, observation, context):
                self.last_observation = observation
                self.last_noise = calibrate_pi05.deterministic_noise(
                    self, context.seed, context.sample_index
                )
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
            self.assertEqual(document["plan"], {"sites": ["vision.patch_input"]})
            self.assertEqual(
                document["seed_policy"]["algorithm"],
                "numpy-pcg64-seed-sequence-v1",
            )
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
