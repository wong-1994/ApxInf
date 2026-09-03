"""Checkpoint-directory layout: which format, and where is each file.

One place that answers "what shape is this checkpoint" for all three callers who
need to know — the policy loader, the preflight checks, and the offline auditor —
so a directory convention is written down once rather than guessed three times.

    from apxinf.checkpoints import detect_checkpoint, require_norm_stats

    layout = detect_checkpoint(model_dir)
    normalization = layout.normalization
    if normalization is None:
        require_norm_stats(layout)
    config_json = layout.config_json_text() # -> apxinf_py.Model.load(config_json=...)

Run ``python -m apxinf.checkpoints <dir>`` to see the same answer as a report.
Nothing here loads weights, imports torch, or touches the network.
"""

from __future__ import annotations

from .descriptor import NormalizationPlan, TokenizerSpec, TransformSpec

from .layout import (
    FORMATS,
    LEROBOT,
    NORM_STATS_NAME,
    OPENPI_PYTORCH,
    TOKENIZER_NAMES,
    CheckpointError,
    CheckpointLayout,
    detect_checkpoint,
    has_layout_metadata,
    require_norm_stats,
    resolve_tokenizer,
)
from .metadata import (
    MetadataError,
    read_metadata_pt,
    repack_structure,
    train_config_facts,
)

__all__ = [
    "CheckpointError",
    "CheckpointLayout",
    "FORMATS",
    "LEROBOT",
    "MetadataError",
    "NORM_STATS_NAME",
    "NormalizationPlan",
    "OPENPI_PYTORCH",
    "TOKENIZER_NAMES",
    "TokenizerSpec",
    "TransformSpec",
    "detect_checkpoint",
    "has_layout_metadata",
    "read_metadata_pt",
    "repack_structure",
    "require_norm_stats",
    "resolve_tokenizer",
    "train_config_facts",
]
