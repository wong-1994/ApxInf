from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str):
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLES))


class _PolicyContractOnly:
    metadata = {"image_keys": ["front", "wrist"]}
    action_dim = 7


def test_autopolicy_example_uses_public_metadata_contract():
    example = _load_example("autopolicy_infer")
    observation = example.synthetic_observation_for(_PolicyContractOnly())

    assert set(observation) == {"front", "wrist", "observation/state", "prompt"}
    assert observation["front"].shape == (256, 256, 3)
    assert observation["front"].dtype == np.uint8
    assert observation["observation/state"].shape == (7,)


@pytest.mark.parametrize("image_keys", [None, [], ["front", ""]])
def test_autopolicy_example_rejects_missing_or_invalid_image_contract(image_keys):
    example = _load_example("autopolicy_infer")

    class BadPolicy:
        metadata = {} if image_keys is None else {"image_keys": image_keys}
        action_dim = 7

    with pytest.raises(ValueError, match="image_keys"):
        example.synthetic_observation_for(BadPolicy())


def test_common_policy_options_accept_a_json_object():
    common = _load_example("_common")

    assert common.json_object('{"norm_key":"x2_normal","image_keys":["front","wrist"]}') == {
        "norm_key": "x2_normal",
        "image_keys": ["front", "wrist"],
    }


def test_common_policy_options_keep_model_knobs_and_protect_generic_flags():
    common = _load_example("_common")

    options = common.policy_kwargs(
        {
            "norm_key": "x2_normal",
            "device": "wrong",
            "metadata": {"deployment": "lab", "protocol": "wrong"},
        },
        device="cuda:1",
        precision="fp8",
        action_dim=7,
        metadata={"protocol": "openpi.websocket_policy"},
    )

    assert options == {
        "norm_key": "x2_normal",
        "device": "cuda:1",
        "precision": "fp8",
        "action_dim": 7,
        "metadata": {"deployment": "lab", "protocol": "openpi.websocket_policy"},
    }


@pytest.mark.parametrize("value", ["[]", '"walloss"', "not-json"])
def test_common_policy_options_reject_non_objects(value):
    common = _load_example("_common")

    with pytest.raises(argparse.ArgumentTypeError):
        common.json_object(value)
