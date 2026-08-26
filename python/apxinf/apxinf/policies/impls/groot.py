"""GR00T N1.7 policy over canonical NVIDIA processor outputs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ..registry import register_policy

__all__ = ["GrootPolicy"]

_REQUIRED = ("pixel_values", "image_grid_thw", "token_ids", "attention_mask",
             "image_mask", "state", "embodiment_id")


def _single(value, dtype, *, rank):
    array = np.asarray(value, dtype=dtype)
    if array.ndim == rank + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != rank:
        raise ValueError(f"GrootPolicy: expected rank {rank} single-sample input, got {array.shape}")
    return np.ascontiguousarray(array)


@register_policy("Gr00tN1d7")
@register_policy("gr00t_n1d7")
@register_policy("groot")
class GrootPolicy:
    """Canonical GR00T N1.7 processor output -> deployable action chunk."""

    def __init__(self, model, *, metadata: Optional[Mapping[str, Any]] = None):
        self.model = model
        self.metadata = {
            "model_type": "Gr00tN1d7",
            "action_horizon": int(model.action_horizon),
            "action_dim": int(model.action_dim),
            "flow_steps": 4,
            "processor_contract": "nvidia-gr00t-n1d7-canonical",
            **(dict(metadata) if metadata else {}),
        }

    @classmethod
    def from_pretrained(cls, model_dir, *, model=None, device="cuda:0",
                        precision="bf16", metadata=None, **_):
        if precision != "bf16":
            raise ValueError("GrootPolicy supports the validated BF16 path only")
        if model is None:
            import apxinf_py
            model = apxinf_py.Model.load(
                "Gr00tN1d7", str(Path(model_dir)), device=device, precision=precision
            )
        return cls(model, metadata=metadata)

    @property
    def action_horizon(self):
        return int(self.metadata["action_horizon"])

    @property
    def action_dim(self):
        return int(self.metadata["action_dim"])

    def infer(self, observation: Mapping[str, Any], *, noise=None):
        missing = [key for key in _REQUIRED if key not in observation]
        if missing:
            raise KeyError(f"GrootPolicy: missing canonical processor fields {missing}")
        started = time.perf_counter()
        actions = self.model.infer_groot(
            _single(observation["pixel_values"], np.float32, rank=2),
            _single(observation["image_grid_thw"], np.uint32, rank=2),
            _single(observation["token_ids"], np.uint32, rank=1),
            _single(observation["attention_mask"], np.uint8, rank=1),
            _single(observation["image_mask"], np.uint8, rank=1),
            _single(observation["state"], np.float32, rank=2),
            int(observation["embodiment_id"]),
            None if noise is None else _single(noise, np.float32, rank=2),
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.action_horizon, self.action_dim)
        if actions.shape != expected:
            raise RuntimeError(f"GrootPolicy: model returned {actions.shape}, expected {expected}")
        return {"actions": actions, "timing": {"model_ms": elapsed, "total_ms": elapsed}}

    def close(self):
        close = getattr(self.model, "close", None)
        if callable(close):
            close()
