import json

import numpy as np


def test_minimal_vla_is_registered():
    from apxinf import MinimalVlaPolicy
    from apxinf.policies.registry import get_policy

    assert get_policy("minimal_vla") is MinimalVlaPolicy


def test_auto_policy_uses_existing_processing_and_generic_dispatch(tmp_path):
    from apxinf import AutoPolicy, MinimalVlaPolicy

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "minimal_vla"}))

    class Model:
        action_horizon = 1
        action_dim = 2
        num_views = 1
        image_size = 1
        max_token_len = 1

        def infer_rgb(self, rgb, layout, token_ids, noise):
            assert rgb.shape == (1, 1, 1, 3)
            assert layout == "nhwc"
            return np.array([[0.25, -0.5]], dtype=np.float32)

    policy = AutoPolicy.from_pretrained(tmp_path, model=Model())
    assert isinstance(policy, MinimalVlaPolicy)
    result = policy.infer({"observation/image": np.zeros((1, 1, 3), dtype=np.uint8), "prompt": "move"})
    np.testing.assert_array_equal(result["normalized_actions"], [[0.25, -0.5]])
    np.testing.assert_array_equal(result["actions"], [[0.25, -0.5]])
    assert result["metadata"]["model_type"] == "minimal_vla"
