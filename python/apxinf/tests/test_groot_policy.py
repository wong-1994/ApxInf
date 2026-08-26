import numpy as np
import pytest

from apxinf.policies.impls.groot import GrootPolicy


class MockGroot:
    action_horizon = 40
    action_dim = 132
    def infer_groot(self, *args):
        self.args = args
        return np.zeros((40, 132), dtype=np.float32)


def canonical():
    return {
        "pixel_values": np.zeros((4, 1536), dtype=np.float32),
        "image_grid_thw": np.array([[1, 2, 2]], dtype=np.uint32),
        "token_ids": np.array([1, 151655], dtype=np.uint32),
        "attention_mask": np.ones(2, dtype=np.uint8),
        "image_mask": np.array([0, 1], dtype=np.uint8),
        "state": np.zeros((1, 132), dtype=np.float32),
        "embodiment_id": 0,
    }


def test_groot_policy_canonical_path():
    model = MockGroot()
    policy = GrootPolicy(model)
    result = policy.infer(canonical(), noise=np.zeros((40, 132), dtype=np.float32))
    assert result["actions"].shape == (40, 132)
    assert policy.metadata["flow_steps"] == 4
    assert model.args[6] == 0


def test_groot_policy_rejects_incomplete_processor_output():
    obs = canonical(); del obs["state"]
    with pytest.raises(KeyError, match="state"):
        GrootPolicy(MockGroot()).infer(obs)
