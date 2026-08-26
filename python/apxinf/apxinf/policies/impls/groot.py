"""NVIDIA GR00T N1.7 policy backed by the native ApxInf BF16 runtime."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ..registry import register_policy

__all__ = ["GrootPolicy"]


@register_policy("gr00tn1d7")
class GrootPolicy:
    """Observation-to-action policy for the official ``Gr00tN1d7`` contract.

    The maintained dependency-free boundary accepts canonical processor output
    (``pixel_values``, ``input_ids``, normalized padded ``state``). When NVIDIA's
    public processor package is installed, :meth:`from_pretrained` also wires it
    so raw modality dictionaries can be passed directly.
    """

    def __init__(self, model, *, embodiment: str, embodiment_id: int, processor=None):
        self.model = model
        self.embodiment = embodiment
        self.embodiment_id = int(embodiment_id)
        self.processor = processor
        self.metadata = {
            "model_type": "Gr00tN1d7", "precision": "bf16",
            "embodiment": embodiment, "embodiment_id": self.embodiment_id,
            "action_horizon": int(model.action_horizon), "model_action_dim": int(model.action_dim),
        }

    @classmethod
    def from_pretrained(cls, model_dir, *, embodiment: str, device="cuda:0",
                        precision="bf16", seed=0, processor=None):
        model_dir = Path(model_dir)
        import json
        mapping = json.loads((model_dir / "embodiment_id.json").read_text())
        if embodiment not in mapping:
            raise ValueError(f"unknown GR00T embodiment {embodiment!r}")
        if model_dir.name == "models--nvidia--GR00T-N1.7-LIBERO":
            raise ValueError("refs-only GR00T-N1.7-LIBERO cache is not a usable checkpoint snapshot")
        import apxinf_py
        model = apxinf_py.GrootModel.load(
            str(model_dir), device=device, precision=precision, sampling_seed=int(seed)
        )
        if processor is None:
            try:
                from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
                processor = Gr00tN1d7Processor.from_pretrained(str(model_dir))
            except ImportError:
                processor = None
        return cls(model, embodiment=embodiment, embodiment_id=mapping[embodiment], processor=processor)

    def infer(self, observation: Mapping[str, Any], *, noise: Optional[np.ndarray] = None) -> dict:
        started = time.perf_counter()
        raw_observation = observation
        if {"pixel_values", "input_ids", "state"}.issubset(observation):
            inputs = observation
        elif self.processor is not None:
            from gr00t.data.embodiment_tags import EmbodimentTag
            inputs = self.processor.process_observation(dict(observation), EmbodimentTag(self.embodiment))
        else:
            raise ValueError(
                "GrootPolicy requires canonical pixel_values/input_ids/state, or the official "
                "NVIDIA GR00T processor package for raw observations"
            )
        patches = np.ascontiguousarray(np.asarray(inputs["pixel_values"]), dtype=np.float32)
        tokens = np.ascontiguousarray(np.asarray(inputs["input_ids"]).reshape(-1), dtype=np.uint32)
        state = np.ascontiguousarray(np.asarray(inputs["state"]).reshape(1, -1), dtype=np.float32)
        selected_noise = None if noise is None else np.ascontiguousarray(noise, dtype=np.float32)
        model_started = time.perf_counter()
        normalized = np.asarray(self.model.infer_patches(
            patches, tokens, state, self.embodiment_id, selected_noise
        ), dtype=np.float32)
        model_ms = (time.perf_counter() - model_started) * 1000.0
        actions: Any = normalized
        if self.processor is not None and not {"pixel_values", "input_ids", "state"}.issubset(raw_observation):
            from gr00t.data.embodiment_tags import EmbodimentTag
            state_dict = {k: v for k, v in raw_observation.items() if k.startswith("state.")}
            actions = self.processor.unapply(normalized[None], EmbodimentTag(self.embodiment), state_dict)
        return {"actions": actions, "normalized_actions": normalized, "token_ids": tokens,
                "noise": selected_noise, "timing": {"model_ms": model_ms,
                "total_ms": (time.perf_counter() - started) * 1000.0}, "metadata": self.metadata}

    __call__ = infer

    def close(self):
        close = getattr(self.model, "close", None)
        if callable(close): close()
