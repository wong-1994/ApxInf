"""Public OpenVLA observation and action semantics without a GPU/checkpoint."""

import unittest
import numpy as np
from apxinf.policies.impls.openvla import OpenVlaPolicy
from apxinf.policies.registry import get_policy


class Tokenizer:
    def encode(self, text):
        self.text = text
        return [10, 20]


class Runner:
    def _infer_patches(self, pixels, ids):
        self.pixels, self.ids = pixels, ids
        return np.array([[31999, 31872, 31745, 31744, 31999, 31872, 31744]], np.float32)


def make_policy(runner=None):
    config = dict(
        model_type="openvla",
        n_action_bins=256,
        norm_stats={
            "bridge_orig": {
                "action": dict(q01=[-2] * 7, q99=[4] * 7, mask=[True] * 6 + [False])
            }
        },
    )
    processor = dict(
        image_resize_strategy="resize-naive",
        input_sizes=[[3, 224, 224]] * 2,
        interpolations=["bicubic"] * 2,
        tvf_normalize_params=[dict(mean=[0.5] * 3, std=[0.5] * 3)] * 2,
    )
    return OpenVlaPolicy(
        runner or Runner(),
        tokenizer=Tokenizer(),
        config=config,
        processor_config=processor,
    )


class OpenVlaPolicyTests(unittest.TestCase):
    def test_registered_observation_to_action(self):
        self.assertIs(get_policy("openvla"), OpenVlaPolicy)
        p = make_policy()
        out = p.infer(dict(image=np.zeros((240, 320, 3), np.uint8), prompt="Pick UP"))
        self.assertEqual(
            p.tokenizer.text, "In: What action should the robot take to pick up?\nOut:"
        )
        np.testing.assert_array_equal(p.model_runner.ids, [1, 10, 20, 29871])
        self.assertEqual(p.model_runner.pixels.shape, (6, 50176))
        np.testing.assert_array_equal(
            p.model_runner.pixels, -np.ones((6, 50176), np.float32)
        )
        self.assertEqual(out["actions"].shape, (1, 7))
        self.assertAlmostEqual(float(out["actions"][0, 0]), -2 + 3 / 255, places=6)
        self.assertAlmostEqual(float(out["actions"][0, 6]), 254 / 255, places=6)

    def test_eos_uses_reference_final_seven_sequence_tokens(self):
        p = make_policy()
        _, tokens = p.decode(
            [31800, 2, 31801, 31802, 31803, 31804, 31805],
            [1, 11, 12, 13, 14, 15, 29871],
        )
        np.testing.assert_array_equal(tokens, [12, 13, 14, 15, 29871, 31800, 2])

    def test_rejects_wrong_image_and_fractional_tokens(self):
        p = make_policy()
        with self.assertRaises(ValueError):
            p.preprocess(dict(image=np.zeros((224, 224, 3), np.float32), prompt="test"))
        with self.assertRaises(ValueError):
            p.decode([1.5] * 7, [1, 29871])
        with self.assertRaises(ValueError):
            p.infer({}, noise=np.zeros(7))


if __name__ == "__main__":
    unittest.main()
