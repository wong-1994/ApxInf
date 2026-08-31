from __future__ import annotations

import json

import numpy as np


class _FakeModel:
    action_horizon = 10
    action_dim = 26
    num_views = 2

    def __init__(self):
        self.call = None

    def _infer_patches(self, patches, token_ids, noise=None, action_mask=None):
        self.call = (patches, token_ids, noise, action_mask)
        return np.zeros((10, 26), dtype=np.float32)


class _FakeProcessor:
    image_keys = ("observation/image", "observation/wrist_image")

    def __call__(self, observation):
        return (
            np.zeros((648, 1176), dtype=np.float32),
            np.arange(313, dtype=np.uint32),
            np.ones((10, 26), dtype=np.float32),
        )


class _ArrayLike:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.value


class _FakeImageProcessor:
    patch_size = 14
    merge_size = 2
    min_pixels = 56 * 56
    max_pixels = 14 * 14 * 4 * 1280

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        assert [image.shape for image in images] == [(252, 252, 3)] * 2
        return {
            "image_grid_thw": _ArrayLike([[1, 18, 18], [1, 18, 18]]),
            "pixel_values": _ArrayLike(np.zeros((648, 1176), np.float32)),
        }


class _FakeTokenizer:
    image_pad_token_id = 99

    def __init__(self):
        self.prompt = None

    def __call__(self, prompt, **kwargs):
        self.prompt = prompt
        # 141 text + 2 image placeholders + 10 action tokens. Expansion adds
        # 80 tokens per image, yielding the real 313-token fixture shape.
        return {"input_ids": [1] * 141 + [99, 99] + [2] * 10}


def test_registry_exports_walloss():
    from apxinf import WallossPolicy
    from apxinf.policies import available_policies, get_policy

    assert "walloss" in available_policies()
    assert "wall_oss_05" in available_policies()
    assert get_policy("walloss") is WallossPolicy


def test_autopolicy_detects_walloss_checkpoint_signature(tmp_path, monkeypatch):
    from apxinf import AutoPolicy, WallossPolicy

    document = {
        "model_type": "qwen2_5_vl",
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "experts": [{}, {}],
        "action_hidden_size": 1024,
        "noise_scheduler": {},
    }
    (tmp_path / "config.json").write_text(json.dumps(document))
    monkeypatch.setattr(
        WallossPolicy, "from_pretrained", classmethod(lambda cls, model_dir, **kwargs: (model_dir, kwargs))
    )
    model_dir, kwargs = AutoPolicy.from_pretrained(tmp_path, precision="bf16")
    assert model_dir == tmp_path
    assert kwargs == {"precision": "bf16"}


def test_policy_calls_patch_contract_and_unnormalizes():
    from apxinf import WallossPolicy

    model = _FakeModel()
    minimum = np.arange(26, dtype=np.float32)
    delta = np.full(26, 2.0, dtype=np.float32)
    policy = WallossPolicy(
        model,
        _FakeProcessor(),
        action_min=minimum,
        action_delta=delta,
        action_dim=7,
    )
    noise = np.zeros((10, 26), dtype=np.float32)
    result = policy.infer({}, noise=noise)

    patches, token_ids, passed_noise, action_mask = model.call
    assert patches.shape == (648, 1176)
    assert token_ids.shape == (313,)
    np.testing.assert_array_equal(passed_noise, noise)
    assert action_mask.shape == (10, 26)
    np.testing.assert_array_equal(result["actions"][0], minimum[:7] + 1.0)
    assert result["actions"].shape == (10, 7)
    assert result["timing"]["total_ms"] >= result["timing"]["model_ms"]


def test_processor_builds_fixed_walloss_contract():
    from apxinf.policies.impls.walloss import _WallossProcessor

    processor = _WallossProcessor.__new__(_WallossProcessor)
    processor.image_keys = ("observation/image", "observation/wrist_image")
    processor.camera_names = ("face_view", "right_wrist_view")
    processor.action_horizon = 10
    processor.action_dim = 26
    processor.tokenizer = _FakeTokenizer()
    processor.image_processor = _FakeImageProcessor()
    processor.image_pad_token_id = 99
    processor.propri_min = np.full(26, -1.0, np.float32)
    processor.propri_delta = np.full(26, 2.0, np.float32)
    processor.factor = 28
    processor.min_pixels = 56 * 56
    processor.max_pixels = 14 * 14 * 4 * 1280

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    patches, tokens, mask = processor(
        {
            "observation/image": image,
            "observation/wrist_image": image,
            "observation/state": np.zeros(7, np.float32),
            "prompt": "move",
        }
    )
    assert patches.shape == (648, 1176)
    assert tokens.shape == (313,)
    assert mask.shape == (10, 26)
    np.testing.assert_array_equal(mask[:, :7], 1.0)
    np.testing.assert_array_equal(mask[:, 7:], 0.0)
    assert "front view" in processor.tokenizer.prompt
    assert "right wrist view" in processor.tokenizer.prompt
    assert "Proprioception: " + " ".join(["128"] * 7) in processor.tokenizer.prompt
