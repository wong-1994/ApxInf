"""The contracts shared across the frontend: :class:`Policy` and :class:`BareModel`.

Both are structural :class:`typing.Protocol` types — nothing has to inherit
them; any object of the right shape satisfies them. They are the stable anchor
points other layers code against:

* :class:`BareModel` — the model-side contract (L1 bare inference) a policy
  consumes, i.e. the subset of a ``apxinf_py.Model`` handle it relies on.
* :class:`Policy` — the L2 contract (``obs dict -> result dict``) every
  model-specific policy satisfies (``Pi05Policy`` today; e.g. a future
  ``GrootPolicy``). Downstream consumers (the websocket server, the
  :class:`~apxinf.policies.auto.AutoPolicy` registry, a lerobot adaptor) code
  against this, never a concrete class.
* :class:`ComposablePolicy` — the narrow extra capability a policy needs before
  an *outer* layer (a robot adapter) can wrap steps around its chain. Optional:
  a policy is a perfectly good :class:`Policy` without it.

It exists to pin these contracts *now*, before a second model lands, so the
layout is set for whoever adds one. They intentionally stay tiny: extract richer
shared structure when a second model actually teaches us what is common, not by
guessing from one example.

The ``Policy.infer`` result is a plain dict. Two keys are guaranteed across all
policies: ``actions`` (deployable, unnormalized-domain ``float32``
``[action_horizon, action_dim]``) and ``timing`` (at least ``model_ms`` and
``total_ms``). Policies may add model-specific keys (``Pi05Policy`` also returns
``normalized_actions`` / ``token_ids`` / ``noise``); consumers must not rely on
anything beyond the two guaranteed ones.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = ["Policy", "BareModel", "ComposablePolicy"]


@runtime_checkable
class BareModel(Protocol):
    """The subset of a ``apxinf_py.Model`` handle an L2 policy relies on (L1).

    Only L1 ``infer_rgb`` is part of the contract. The L0 patches path exists on
    the binding but is internal (exposed privately as ``_infer_patches``) and is
    deliberately not surfaced here.
    """

    action_horizon: int
    action_dim: int
    num_views: int
    image_size: int

    def infer_rgb(
        self,
        rgb_u8: np.ndarray,
        layout: str,
        token_ids: np.ndarray,
        noise: np.ndarray | None = None,
    ) -> np.ndarray: ...


@runtime_checkable
class Policy(Protocol):
    """Structural contract for an L2 policy: ``obs dict -> result dict``."""

    #: Static description of the policy (model_type, shapes, keys, ...).
    metadata: Mapping[str, Any]

    @property
    def action_dim(self) -> int:
        """Width of one deployable action vector (post-processing output)."""
        ...

    @property
    def action_horizon(self) -> int:
        """Number of actions in one predicted chunk."""
        ...

    def infer(
        self, observation: Mapping[str, Any], *, noise: np.ndarray | None = None
    ) -> Dict[str, Any]:
        """Map a raw observation dict (+ prompt) to a result dict.

        ``noise`` optionally supplies an exact continuous initial latent. A
        policy whose model samples such a latent internally uses that path when
        the argument is absent.

        Guarantees the ``actions`` and ``timing`` keys (see module docstring).
        """
        ...

    def close(self) -> None:
        """Release any underlying model resources."""
        ...


@runtime_checkable
class ComposablePolicy(Protocol):
    """A :class:`Policy` an outer layer can wrap without knowing its internals.

    This is the seam between the **robot** layer and the **model** layer. A robot
    adapter (``apxinf.robots.unitree_g1``) has to run its own steps around the
    model's — decode the wire state before anything model-specific reads it, turn
    the model's delta actions into absolute joint targets after unnormalization.
    That is an *ordering* requirement and nothing more: outside, in both
    directions, exactly the way openpi's ``data_transforms`` sit outside its
    ``model_transforms``.

    Without this method the only way to express it is to reach into the concrete
    policy — import ``Pi05Policy``, address its steps by name
    (``insert_before("tokenize", ...)``), and rebuild it. That makes every robot
    adapter depend on one model class and on that class's private step names.
    :meth:`with_adapter` states the ordering instead, so the robot layer imports
    no model and the model layer knows about no robot.

    Only this one method is promoted, and only because real code calls it. The
    rest of the composition surface (``input_pipeline``, ``output_pipeline``,
    ``model``) stays private to the concrete class until a second model shows
    what is genuinely shared.
    """

    def with_adapter(
        self,
        *,
        before: Sequence[Any] = (),
        after: Sequence[Any] = (),
        action_dim: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "Policy":
        """Return a policy running ``before`` ahead of, and ``after`` behind, its own chain.

        Each entry is a ``(name, step)`` pair (a
        :data:`~apxinf.processors.base.StepSpec`). ``action_dim`` declares the
        deployable width the wrapped chain now emits, since an appended step may
        change it; ``metadata`` is merged over the inherited description. The
        returned policy shares the underlying model handle — do not close both.
        """
        ...
