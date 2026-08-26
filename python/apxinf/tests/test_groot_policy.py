"""Offline tests for the public GR00T N1.7 policy boundary."""

import numpy as np

from apxinf import GrootPolicy


class _MockGrootModel:
    action_horizon = 40
    action_dim = 132

    def infer_patches(self, patches, tokens, state, embodiment_id, noise):
        assert patches.shape == (256, 1536)
        assert tokens.dtype == np.uint32
        assert state.shape == (1, 132)
        assert embodiment_id == 2
        return np.asarray(noise, dtype=np.float32)


def test_canonical_processor_output_reaches_native_model():
    policy = GrootPolicy(_MockGrootModel(), embodiment="libero_sim", embodiment_id=2)
    noise = np.arange(40 * 132, dtype=np.float32).reshape(40, 132)
    result = policy.infer(
        {
            "pixel_values": np.zeros((256, 1536), dtype=np.float32),
            "input_ids": np.array([[151652, 151655, 151653]], dtype=np.int64),
            "state": np.zeros((1, 1, 132), dtype=np.float32),
        },
        noise=noise,
    )

    np.testing.assert_array_equal(result["actions"], noise)
    assert result["metadata"]["model_type"] == "Gr00tN1d7"
    assert result["metadata"]["embodiment_id"] == 2
