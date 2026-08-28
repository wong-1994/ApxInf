"""Unitree G1 adapter: wrap the G1 steps around whatever policy the checkpoint is.

This module is the *robot* half of the robot/model split. It knows the G1 body —
16 DoF laid out ``[L-arm 7, L-gripper 1, R-arm 7, R-gripper 1]``, three cameras,
delta joint actions — and it knows nothing about the model serving it. It names
no model class, no step inside the model's chain, and no checkpoint layout:
:class:`~apxinf.policies.auto.AutoPolicy` decides which policy the directory
holds, and :meth:`~apxinf.policies.base.ComposablePolicy.with_adapter` wraps the
G1 steps around that policy's own pre/post chain.

It knows nothing about the *dataset* either: the wire keys arrive as required
arguments from a :class:`~apxinf.conventions.Convention`, so the same body serves
a re-recorded G1 without an edit here.

The nesting is the entire coupling, and it is an ordering rule, not a naming one:

    decode G1 state  ->  [ whatever the model does ]  ->  delta->absolute -> 32->16

which is openpi's ``data_transforms`` sitting outside its ``model_transforms``.
Everything else — how the model tokenizes, whether it discretizes state, what its
steps are called — stays the model's business, so a G1 checkpoint of a different
architecture serves through this same adapter unchanged.

Only the primary serving path (3 cameras, ``adapt_to_pi=True``, no fixed-hand
override) is wired — that is what the shipped ``pi05_UnitreeG1_groundwire`` config
uses; the integrator's ``_NoLeftCam`` / ``fixed_hand`` variants are training-data
cleaning knobs, not serving paths, so they are intentionally omitted.

.. note::
   With a stand-in checkpoint (no real G1 weights / norm_stats) this adapter
   validates *plumbing and shape* (G1 obs -> ``[action_horizon, 16]``), not G1
   action values. Numeric parity needs the integrator's real gripper limits
   **and** the G1 checkpoint/norm_stats; the same adapter code then runs
   unchanged.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..policies.auto import AutoPolicy
from ..policies.base import ComposablePolicy, Policy
from ..processors.base import StepSpec
from ..processors.robots.unitree_g1 import (
    G1_ROBOT_DIM,
    UnitreeG1AbsoluteActions,
    UnitreeG1DecodeState,
    UnitreeG1EncodeActions,
)

__all__ = ["build_unitree_g1_policy"]


def build_unitree_g1_policy(
    model_dir,
    *,
    state_key: str,
    image_keys: Sequence[str],
    use_delta_joint_actions: bool = True,
    adapt_to_pi: bool = True,
    discrete_state: bool = True,
    action_dim: Optional[int] = None,
    metadata: Optional[dict] = None,
    **load_kwargs: Any,
) -> Policy:
    """Load ``model_dir`` and wrap the Unitree G1 pre/post adapter around it.

    The checkpoint is loaded through :class:`~apxinf.policies.auto.AutoPolicy`, so
    the model type comes from ``config.json`` (or an explicit ``model_type=``)
    rather than from this file. It is loaded at **full model width**
    (``action_dim=None``) because the delta->absolute step must see the whole
    action before ``UnitreeG1EncodeActions`` truncates it to 16. The model's own
    unnormalizer therefore comes from the checkpoint's ``norm_stats["actions"]``
    at its native width, so a real deployment needs G1 norm_stats at least ``16``
    wide; pass ``unnormalizer=`` through to inject one instead (e.g. a full-width
    identity for a shape/plumbing run on a stand-in checkpoint).

    The resulting chain is the model's own, wrapped:

    * **input**  — ``[g1_decode_state?, <the model's input steps>]``
    * **output** — ``[<the model's output steps>, g1_absolute?, g1_encode?]``

    ``adapt_to_pi=False`` drops the decode/encode conventions (raw robot space);
    ``use_delta_joint_actions=False`` drops the delta->absolute step (the model
    already emits absolute actions). ``action_dim`` is the **deployable** width
    reported to clients: it defaults to :data:`G1_ROBOT_DIM` when ``adapt_to_pi``
    wires the truncating encode step, and otherwise to whatever the model's own
    chain emits — a policy must not advertise a width no step produces.

    Extra ``load_kwargs`` reach the concrete policy's ``from_pretrained``
    (``device`` / ``precision`` / ``checkpoint`` / ``model_type`` / a pre-built
    ``model=`` handle / an injected ``unnormalizer=``, ...).

    ``state_key`` and ``image_keys`` are **required and have no defaults**. They
    are a recording convention, not a fact about this body: the G1 client's
    dialect lives in :data:`apxinf.conventions.UNITREE_G1`, and defaulting to it
    here would rebuild the robot↔dataset coupling one layer up. Pass
    ``**vars(...)`` of a convention, or let
    :func:`~apxinf.robots.presets.build_robot_policy` do it from the preset
    table::

        from apxinf.conventions import UNITREE_G1 as G1_KEYS
        policy = build_unitree_g1_policy(
            ckpt, state_key=G1_KEYS.state_key, image_keys=G1_KEYS.image_keys
        )

    Serving with ``discrete_state=False`` silently drops state, which also makes
    ``use_delta_joint_actions`` a no-op (a delta cannot be resolved without the
    current joint positions).
    """
    base = AutoPolicy.from_pretrained(
        model_dir,
        image_keys=tuple(image_keys),
        action_dim=None,  # keep full model width; the g1_encode step truncates to 16
        state_key=state_key,
        prompt_key=prompt_key,
        discrete_state=discrete_state,
        **load_kwargs,
    )
    if not isinstance(base, ComposablePolicy):
        raise TypeError(
            f"the Unitree G1 adapter has to run its steps around the model's, which "
            f"{type(base).__name__} (loaded from {model_dir}) does not allow: it has no "
            "with_adapter(). Give that policy class a ComposablePolicy.with_adapter "
            "rather than teaching this adapter about its internals."
        )

    before: List[StepSpec] = []
    if adapt_to_pi:
        # Ahead of the model's own steps, so both the (optional) state
        # discretization and the delta->absolute output step read decoded state.
        before.append(("g1_decode_state", UnitreeG1DecodeState(state_key)))

    after: List[StepSpec] = []
    if use_delta_joint_actions:
        after.append(("g1_absolute", UnitreeG1AbsoluteActions(state_key)))
    if adapt_to_pi:
        after.append(("g1_encode", UnitreeG1EncodeActions()))

    if action_dim is not None:
        width: Optional[int] = int(action_dim)
    elif adapt_to_pi:
        width = G1_ROBOT_DIM  # g1_encode does the truncation
    else:
        width = None  # no truncating step is wired; keep the model chain's width

    return base.with_adapter(
        before=before,
        after=after,
        action_dim=width,
        metadata={"robot": "unitree_g1", **(metadata or {})},
    )
