"""Original OpenVLA policy: host observation preparation and action decoding.

The model core runs exclusively in ApxInf's native ModelRunner. No PyTorch or
remote checkpoint code is imported by this policy.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from ..registry import register_policy
from ...processors.transforms import lookup_key


@register_policy("openvla")
class OpenVlaPolicy:
    def __init__(
        self,
        model_runner,
        *,
        tokenizer,
        config,
        processor_config,
        unnorm_key="bridge_orig",
        image_key="image",
        prompt_key="prompt",
    ):
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.image_key, self.prompt_key = image_key, prompt_key
        if config.get("model_type") != "openvla" or config.get("n_action_bins") != 256:
            raise ValueError(
                "OpenVLA requires the original 256-bin action configuration"
            )
        if processor_config.get(
            "image_resize_strategy"
        ) != "resize-naive" or processor_config.get("input_sizes") != [
            [3, 224, 224],
            [3, 224, 224],
        ]:
            raise ValueError(
                "OpenVLA requires the original fused 224px image processor"
            )
        self._normalization = processor_config["tvf_normalize_params"]
        if len(self._normalization) != 2 or processor_config.get("interpolations") != [
            "bicubic",
            "bicubic",
        ]:
            raise ValueError("OpenVLA requires two bicubic RGB transforms")
        stats = config["norm_stats"][unnorm_key]["action"]
        self._lo = np.asarray(stats["q01"], dtype=np.float64)
        self._hi = np.asarray(stats["q99"], dtype=np.float64)
        self._mask = np.asarray(stats.get("mask", [True] * 7), dtype=bool)
        if any(x.shape != (7,) for x in (self._lo, self._hi, self._mask)):
            raise ValueError("OpenVLA runtime supports seven-dimensional actions")
        edges = np.linspace(-1, 1, 256)
        self._centers = (edges[:-1] + edges[1:]) / 2
        self.metadata = dict(
            model_type="openvla",
            precision="bf16",
            action_horizon=1,
            action_dim=7,
            image_keys=[image_key],
            prompt_key=prompt_key,
            unnorm_key=unnorm_key,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        model_runner=None,
        device="cuda:0",
        precision="bf16",
        unnorm_key="bridge_orig",
        image_key="image",
        prompt_key="prompt",
    ):
        if precision not in ("auto", "bf16"):
            raise ValueError("OpenVLA supports BF16 only")
        root = Path(model_dir)
        import apxinf_py

        tokenizer = apxinf_py.HfTokenizer.from_file(str(root / "tokenizer.json"))
        config = json.loads((root / "config.json").read_text())
        processor = json.loads((root / "preprocessor_config.json").read_text())
        if model_runner is None:
            model_runner = apxinf_py.ModelRunner.load(
                "openvla", str(root), device=device, precision=precision
            )
        return cls(
            model_runner,
            tokenizer=tokenizer,
            config=config,
            processor_config=processor,
            unnorm_key=unnorm_key,
            image_key=image_key,
            prompt_key=prompt_key,
        )

    @property
    def action_dim(self):
        return 7

    @property
    def action_horizon(self):
        return 1

    def reset(self):
        pass

    def preprocess(self, observation):
        from PIL import Image

        image = lookup_key(observation, self.image_key)
        if not isinstance(image, Image.Image):
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("OpenVLA image must be HWC RGB uint8")
            image = Image.fromarray(image)
        if image.mode != "RGB":
            raise ValueError("OpenVLA image must be RGB")
        rgb = np.asarray(
            image.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32
        ).transpose(2, 0, 1) / np.float32(255)
        pixels = []
        for p in self._normalization:
            mean = np.asarray(p["mean"], np.float32).reshape(3, 1, 1)
            std = np.asarray(p["std"], np.float32).reshape(3, 1, 1)
            if not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("invalid normalization scale")
            pixels.append((rgb - mean) / std)
        pixels = np.ascontiguousarray(
            np.concatenate(pixels).reshape(6, 50176), np.float32
        )
        instruction = lookup_key(observation, self.prompt_key)
        if not isinstance(instruction, str):
            raise TypeError("OpenVLA instruction must be text")
        prompt = (
            f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        )
        # The native tokenizer omits special tokens; original OpenVLA adds BOS.
        ids = [1] + self.tokenizer.encode(prompt)
        if not ids or ids[-1] != 29871:
            ids.append(29871)
        return pixels, np.asarray(ids, dtype=np.uint32)

    def decode(self, candidates, prompt_ids):
        tokens = np.asarray(candidates).reshape(-1)
        if (
            tokens.shape != (7,)
            or not np.isfinite(tokens).all()
            or np.any(tokens != np.floor(tokens))
            or np.any((tokens < 0) | (tokens >= 32064))
        ):
            raise ValueError(
                "OpenVLA runner must return seven valid integral token IDs"
            )
        tokens = tokens.astype(np.int64)
        eos = np.flatnonzero(tokens == 2)
        if eos.size:
            # Match reference generate's early stop and predict_action's [-7:] slice.
            tokens = np.concatenate([prompt_ids, tokens[: eos[0] + 1]])[-7:].astype(
                np.int64
            )
        normalized = self._centers[np.clip(32000 - tokens - 1, 0, 254)]
        actions = np.where(
            self._mask,
            0.5 * (normalized + 1) * (self._hi - self._lo) + self._lo,
            normalized,
        )
        return np.ascontiguousarray(actions[None], np.float32), tokens

    def infer(self, observation, *, noise=None):
        if noise is not None:
            raise ValueError(
                "OpenVLA uses deterministic greedy action tokens, not continuous noise"
            )
        start = time.perf_counter()
        pixels, ids = self.preprocess(observation)
        model_start = time.perf_counter()
        output = self.model_runner._infer_patches(pixels, ids)
        model_ms = (time.perf_counter() - model_start) * 1000
        actions, tokens = self.decode(output, ids)
        return dict(
            actions=actions,
            token_ids=tokens,
            timing=dict(
                model_ms=model_ms, total_ms=(time.perf_counter() - start) * 1000
            ),
        )
