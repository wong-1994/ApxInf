"""Robot presets: the wire contract of one embodiment, as a named table entry.

OpenPI keeps this information in a Python registry of ``TrainConfig``s: which
``DataTransformFn`` pair runs, and therefore which **wire keys** the client must
send. ``serve_policy.py --policy.config pi05_UnitreeG1_...`` selects an entry;
the client cannot negotiate it and must match by hand.

This module is the same idea with the same shape, so switching embodiments is a
launch flag on both sides rather than a code edit on ours:

    openpi:  serve_policy.py --policy.config pi05_UnitreeG1_groundwire
    apxinf:  pi05_openpi_websocket_server.py --robot unitree_g1

**Why a table and not a default.** ``Pi05Policy`` used to default ``image_keys``
to ``("observation/image", "observation/wrist_image")`` — LIBERO's wire contract,
living in the model layer as *the* default for every checkpoint. A G1 checkpoint
served without extra arguments therefore ran LIBERO's keys, dropped state, and
skipped the G1 delta→absolute and 32→16 steps — every symptom of a "model
accuracy problem" with nothing in the logs. The policy now names its cameras
after its own :data:`~apxinf.policies.base.VIEW_SLOTS` when the caller names
none, so no dataset's convention can pass for a model default; a preset makes the
embodiment an explicit, named choice.

**Why slot names.** ``image_keys`` is order-significant: entry ``i`` is stacked
into model view slot ``i``, which the checkpoint trained as openpi's
``base_0_rgb`` / ``left_wrist_0_rgb`` / ``right_wrist_0_rgb``. A tuple written in
the wrong order still stacks, still has the right shape, and silently feeds the
wrong camera to each slot. Pairing every wire key with the slot it fills makes
that order reviewable instead of positional. The slot vocabulary itself is a
*model* fact, so it is imported from :mod:`apxinf.policies.base` rather than
restated here.

**Naming rule: ``<arm>_<convention>``.** The arm alone does not determine the
contract — LIBERO and DROID are both Franka Panda, yet LIBERO sends
``observation/image`` with a 7-dim EEF-delta action while DROID sends
``observation/exterior_image_1_left`` with a different action space. So a preset
is named for the arm *and* the dataset convention whose keys it implements:
``franka_libero``, not ``libero`` (a benchmark, not a robot) and not ``franka``
(ambiguous). Single-embodiment robots that own their convention need no suffix —
``unitree_g1``.

**Two halves, one flag.** That naming rule is an admission, so the dataclasses
follow it: an :class:`Embodiment` is the body (camera count, action width, which
pre/post steps its actions need) and a :class:`Convention` is a dataset's
recording dialect (the wire keys, and whether state was recorded into the
prompt). Neither knows the other; they vary independently, which is the point —
the same Franka under DROID's keys is a second :class:`Convention`, not a second
body, and re-recording the G1 changes no :class:`Embodiment` field.

A :class:`RobotPreset` names one *pairing*, and only the pairing is deployable.
``--robot`` stays a single flag over that registry rather than becoming
``--robot`` + ``--convention``: separate flags would let an operator spell a
combination nobody ever recorded, and a body served under a convention it was not
recorded on is exactly the silent mismatch this whole mechanism exists to
prevent. The pairing is validated when the module loads — a convention naming
three cameras cannot be attached to a two-camera body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from ..policies.auto import AutoPolicy
from ..policies.base import VIEW_SLOTS, Policy
from ..processors.robots.unitree_g1 import G1_CAMERAS, G1_STATE_KEY
from .unitree_g1 import build_unitree_g1_policy

__all__ = [
    "VIEW_SLOTS",
    "Embodiment",
    "Convention",
    "RobotPreset",
    "ROBOT_PRESETS",
    "ROBOT_ALIASES",
    "available_robots",
    "get_robot_preset",
    "build_robot_policy",
]


def _build_generic(model_dir, **kwargs) -> Policy:
    """Default builder: the stock policy, no robot-specific pre/post steps."""
    return AutoPolicy.from_pretrained(model_dir, **kwargs)


@dataclass(frozen=True)
class Embodiment:
    """A robot body: how many cameras it has, and what its actions mean.

    Everything here is a fact about hardware. It survives a change of dataset:
    re-record the G1 under a different key convention and this is unchanged.
    Nothing here is a string that goes on the network.
    """

    #: Robot name, used in the served metadata and in preset names.
    name: str
    #: Cameras this body carries; must equal the checkpoint's ``num_views``.
    num_cameras: int
    #: Deployable action width. ``None`` keeps the model's full vector — correct
    #: when a robot output step does the truncation itself (G1's 32→16 encode).
    action_dim: Optional[int] = None
    #: Factory that loads the checkpoint and wires this robot's pre/post steps.
    builder: Callable[..., Policy] = _build_generic
    #: Extra keyword arguments the builder always receives.
    builder_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 1 <= self.num_cameras <= len(VIEW_SLOTS):
            raise ValueError(
                f"Embodiment {self.name!r}: num_cameras={self.num_cameras} is outside "
                f"1..{len(VIEW_SLOTS)}, the view slots {VIEW_SLOTS} a checkpoint fills"
            )

    @property
    def has_robot_steps(self) -> bool:
        """Whether this body needs pre/post arithmetic beyond naming its keys.

        ``False`` for a body the generic builder serves, whose entire contract is
        its wire keys and action width. A caller that *cannot* run :attr:`builder`
        — the checkpoint-free synthetic path — uses this to say what it dropped
        rather than publishing the preset name unqualified.
        """
        return self.builder is not _build_generic


@dataclass(frozen=True)
class Convention:
    """A dataset's recording convention: what the client calls things.

    Everything here is a string on the network, plus the one routing decision
    that follows from how the data was recorded. It survives a change of robot:
    LIBERO's keys describe a Franka today and would describe any arm recorded the
    same way. Nothing here knows an action width or a camera count.
    """

    #: Convention name (a dataset or an integrator's dialect).
    name: str
    #: Camera wire keys, in model view-slot order. Entry *i* fills slot *i*.
    image_keys: Tuple[str, ...]
    #: Wire key of the proprioceptive state vector.
    state_key: str
    #: Wire key of the task instruction.
    prompt_key: str = "prompt"
    #: Optional policy override for state string encoding. ``None`` leaves the
    #: choice to the checkpoint's concrete policy; state representation is a
    #: model semantic, not solely a recording-convention property.
    discrete_state: Optional[bool] = None

    def __post_init__(self) -> None:
        if not self.image_keys:
            raise ValueError(f"Convention {self.name!r}: at least one camera key is required")
        if len(self.image_keys) > len(VIEW_SLOTS):
            raise ValueError(
                f"Convention {self.name!r}: {len(self.image_keys)} camera keys exceeds "
                f"pi05's {len(VIEW_SLOTS)} view slots {VIEW_SLOTS}"
            )
        if len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError(
                f"Convention {self.name!r}: duplicate camera wire keys {list(self.image_keys)}"
            )

    @property
    def num_cameras(self) -> int:
        """Cameras this convention names — cross-checked against the body's."""
        return len(self.image_keys)

    @property
    def slots(self) -> Tuple[Tuple[str, str], ...]:
        """``(view slot, wire key)`` pairs — the pairing, for logs and review."""
        return tuple(zip(VIEW_SLOTS, self.image_keys))


@dataclass(frozen=True)
class RobotPreset:
    """One deployable pairing: a body recorded under a convention.

    The two halves are separable because they vary independently — the same
    Franka arm under LIBERO's and DROID's keys is one :class:`Embodiment` and two
    :class:`Convention`\\ s — but they are validated *together*, because only the
    pair is deployable: a convention naming three cameras cannot serve a two-camera
    body, and nothing else in the system will notice if it tries.

    This stays one ``--robot`` flag. Splitting the flag too would let an operator
    spell combinations that were never recorded, which is the silent mismatch the
    flag exists to prevent; the registry below decides which pairings are real.
    """

    #: Registry name, used as ``--robot <name>``. Spelled ``<arm>_<convention>``
    #: when the arm is shared, bare when the body owns its convention.
    name: str
    #: The body being served.
    embodiment: Embodiment
    #: The key convention its client speaks.
    convention: Convention
    #: One-line description for ``--help`` and the served metadata.
    summary: str = ""

    def __post_init__(self) -> None:
        if self.convention.num_cameras != self.embodiment.num_cameras:
            raise ValueError(
                f"RobotPreset {self.name!r}: convention {self.convention.name!r} names "
                f"{self.convention.num_cameras} cameras but embodiment "
                f"{self.embodiment.name!r} has {self.embodiment.num_cameras}; a "
                "convention can only be paired with a body it was recorded on"
            )

    # --- the pairing's flat contract ---------------------------------------
    #
    # Delegating properties rather than a rename: this is what the server, the
    # metadata and the tests read, and which half a field came from is not their
    # business. They also keep every existing caller working unchanged.

    @property
    def image_keys(self) -> Tuple[str, ...]:
        """Camera wire keys in model slot order — what ``ImageStack`` consumes."""
        return self.convention.image_keys

    @property
    def state_key(self) -> str:
        return self.convention.state_key

    @property
    def prompt_key(self) -> str:
        return self.convention.prompt_key

    @property
    def discrete_state(self) -> Optional[bool]:
        return self.convention.discrete_state

    @property
    def slots(self) -> Tuple[Tuple[str, str], ...]:
        """``(view slot, wire key)`` pairs in model slot order."""
        return self.convention.slots

    @property
    def action_dim(self) -> Optional[int]:
        return self.embodiment.action_dim

    @property
    def builder(self) -> Callable[..., Policy]:
        return self.embodiment.builder

    @property
    def builder_kwargs(self) -> Mapping[str, Any]:
        return self.embodiment.builder_kwargs

    @property
    def num_views(self) -> int:
        """Cameras this preset sends; must equal the checkpoint's ``num_views``."""
        return self.embodiment.num_cameras

    @property
    def has_robot_steps(self) -> bool:
        """Whether this preset wires robot-specific pre/post steps."""
        return self.embodiment.has_robot_steps

    def synthetic_gaps(
        self, *, discrete_state: bool, served_action_dim: int
    ) -> Tuple[str, ...]:
        """What a checkpoint-free (random-weights) server cannot honour here.

        The synthetic path reproduces this preset's wire keys and view count
        exactly and nothing else: its tokenizer emits a fixed token stream and
        never reads state, and :attr:`builder` never runs. Each returned string
        names one gap, for a startup warning — a synthetic server publishing this
        preset's name unqualified would be the silent embodiment mismatch
        ``--robot`` exists to prevent.
        """
        gaps = []
        if discrete_state:
            gaps.append(
                "discrete_state=True — the synthetic tokenizer ignores state, so the "
                "served metadata says discrete_state=False"
            )
        if self.has_robot_steps:
            gap = (
                f"its robot pre/post steps — {self.builder.__name__} never runs, so no "
                "state decode, delta->absolute or action re-encode is wired"
            )
            if self.action_dim is None:
                # action_dim=None means an output step owns the truncation (G1's
                # 32->16 encode). Without that step the full model vector ships.
                gap += (
                    f"; the served action_dim is the model's full {served_action_dim}, "
                    "not the width that skipped step would emit"
                )
            gaps.append(gap)
        return tuple(gaps)

    def describe(self) -> str:
        """Human-readable slot→wire-key mapping, for ``--help`` and startup logs."""
        rendered = ", ".join(f"{slot}={key}" for slot, key in self.slots)
        return f"{self.name}: {rendered}; state={self.state_key}"


#: --- bodies -----------------------------------------------------------------

#: Franka Emika Panda, 7-DoF arm + parallel gripper, 2-camera rig. Its whole port
#: is a table row: the checkpoint already emits absolute actions at the deployable
#: width, so no robot pre/post step is needed and the generic builder serves it.
FRANKA = Embodiment(name="franka", num_cameras=2, action_dim=7)

#: Unitree G1 humanoid: dual-arm + 2 dexterous hands, 16 DoF laid out
#: ``[L-arm 7, L-gripper 1, R-arm 7, R-gripper 1]``, 3 cameras. ``action_dim``
#: stays ``None`` because ``UnitreeG1EncodeActions`` does the 32→16 truncation
#: after delta→absolute has seen the full-width action.
UNITREE_G1_BODY = Embodiment(
    name="unitree_g1",
    num_cameras=3,
    action_dim=None,
    builder=build_unitree_g1_policy,
    builder_kwargs={"use_delta_joint_actions": True, "adapt_to_pi": True},
)

#: --- key conventions ---------------------------------------------------------

#: LIBERO's recording convention (the 7-DoF sim benchmark). Mirrors openpi
#: ``LiberoInputs``: flat ``observation/...`` keys. State encoding is
#: checkpoint-specific: PI0.5 drops it by default, while WallOSS discretizes it.
LIBERO_KEYS = Convention(
    name="libero",
    image_keys=("observation/image", "observation/wrist_image"),
    state_key="observation/state",
    discrete_state=None,
)

#: The ``pi05_UnitreeG1`` fine-tune's convention, from its
#: ``unitreeG1Inputs``/``Outputs``: cameras nested one level under ``images``,
#: a flat ``state``, discretized into the prompt.
UNITREE_G1_KEYS = Convention(
    name="unitree_g1",
    image_keys=tuple(G1_CAMERAS),
    state_key=G1_STATE_KEY,
    discrete_state=True,
)

#: --- deployable pairings -----------------------------------------------------

#: Franka Panda under LIBERO's keys: 2 cameras, 7-dim action.
FRANKA_LIBERO = RobotPreset(
    name="franka_libero",
    embodiment=FRANKA,
    convention=LIBERO_KEYS,
    summary="Franka Panda, LIBERO keys: 2 cameras, 7-dim action (6 EEF deltas + gripper)",
)

#: Unitree G1 under its own convention — a single-embodiment robot that owns its
#: keys, so body and convention share a name and the preset needs no suffix.
UNITREE_G1 = RobotPreset(
    name="unitree_g1",
    embodiment=UNITREE_G1_BODY,
    convention=UNITREE_G1_KEYS,
    summary="Unitree G1: 3 cameras, 16-DoF state, delta joint actions, 32->16 encode",
)

#: Name -> preset. Add an embodiment here once its steps and factory exist; that
#: is the whole registration step (openpi's ``training/config.py`` equivalent).
ROBOT_PRESETS: Dict[str, RobotPreset] = {p.name: p for p in (FRANKA_LIBERO, UNITREE_G1)}

#: Accepted spellings that are not canonical names. ``libero`` names a benchmark
#: rather than a robot, but it is what the deployed launch commands and docs say,
#: so it keeps resolving instead of failing a running deployment on a rename.
ROBOT_ALIASES: Dict[str, str] = {"libero": "franka_libero"}


def available_robots(*, include_aliases: bool = False) -> Tuple[str, ...]:
    """Registered preset names, for ``--robot`` choices and error messages."""
    names = tuple(ROBOT_PRESETS)
    return names + tuple(ROBOT_ALIASES) if include_aliases else names


def get_robot_preset(name: str) -> RobotPreset:
    """Look up a preset by canonical name or alias, with a listing in the error."""
    canonical = ROBOT_ALIASES.get(name, name)
    try:
        return ROBOT_PRESETS[canonical]
    except KeyError:
        raise KeyError(
            f"unknown robot preset {name!r}; known: {list(available_robots())}"
            f" (aliases: {list(ROBOT_ALIASES)})"
        ) from None


def build_robot_policy(
    robot: str,
    model_dir,
    *,
    image_keys: Optional[Sequence[str]] = None,
    state_key: Optional[str] = None,
    prompt_key: Optional[str] = None,
    action_dim: Optional[int] = None,
    discrete_state: Optional[bool] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Policy:
    """Load ``model_dir`` under the named preset, with per-argument overrides.

    Each override defaults to the preset's value; passing one replaces just that
    field. ``image_keys``, ``state_key``, and ``prompt_key`` exist because a
    deployed client may already speak a fixed dialect — they let a server match
    an installed robot stack without editing the preset or touching the client.

    The resulting wire contract is published in the policy ``metadata``
    (``robot`` / ``robot_steps`` / ``image_keys`` / ``state_key`` /
    ``discrete_state``), which the server pushes on connect, so a client can
    assert it rather than guess. ``robot_steps`` says whether this robot's
    pre/post steps are actually wired, so a server that serves the preset's keys
    without its arithmetic cannot pass for the real thing.
    """
    preset = get_robot_preset(robot)
    keys = tuple(image_keys) if image_keys is not None else preset.image_keys
    state = state_key if state_key is not None else preset.state_key
    prompt = prompt_key if prompt_key is not None else preset.prompt_key
    discrete = preset.discrete_state if discrete_state is None else bool(discrete_state)
    width = preset.action_dim if action_dim is None else action_dim
    policy_kwargs = {
        "image_keys": keys,
        "state_key": state,
        "prompt_key": prompt,
        "action_dim": width,
        "metadata": {
            "robot": preset.name,
            "robot_steps": preset.has_robot_steps,
            "robot_slots": [list(pair) for pair in zip(VIEW_SLOTS, keys)],
            **(dict(metadata) if metadata else {}),
        },
        **dict(preset.builder_kwargs),
        **kwargs,
    }
    if discrete is not None:
        policy_kwargs["discrete_state"] = discrete
    return preset.builder(model_dir, **policy_kwargs)
