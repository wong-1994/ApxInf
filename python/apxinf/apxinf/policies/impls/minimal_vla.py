"""Policy integration for the maintained BF16-only minimal VLA fixture."""

from __future__ import annotations

from pathlib import Path

from .pi05 import Pi05Policy
from ..registry import register_policy

__all__ = ["MinimalVlaPolicy"]


@register_policy("minimal_vla")
class MinimalVlaPolicy(Pi05Policy):
    """Reuse the standard VLA preprocessing/postprocessing protocol."""

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        model=None,
        checkpoint=None,
        device="cuda:0",
        precision="bf16",
        seed=0,
        image_keys=("observation/image",),
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"MinimalVlaPolicy does not accept {sorted(kwargs)}")
        if model is None:
            import apxinf_py

            model_dir = Path(model_dir)
            model = apxinf_py.Model.load(
                "minimal_vla",
                str(checkpoint or model_dir / "model.safetensors"),
                device=device,
                precision=precision,
            )
        policy = cls.from_random(model, seed=seed, image_keys=image_keys, warn=False)
        policy.metadata["model_type"] = "minimal_vla"
        policy.metadata["precision"] = "bf16"
        return policy
