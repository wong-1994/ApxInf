"""Exact exported-weight through binding correctness gate."""

import os

import numpy as np
import pytest

import apxinf_py


def test_exported_minimal_vla_stages_and_normalized_action():
    checkpoint = os.environ.get("APXINF_MINIMAL_VLA_MODEL_DIR")
    if not checkpoint:
        pytest.skip("set APXINF_MINIMAL_VLA_MODEL_DIR to an exported fixture")
    model = apxinf_py.Model.load("minimal_vla", checkpoint, device="cpu", precision="bf16")
    action = model.infer_rgb(
        np.array([[[[255, 0, 0]]]], dtype=np.uint8),
        "nhwc",
        np.array([1], dtype=np.uint32),
        np.array([[0.25, 0.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(action, [[1.0, 0.25]])
