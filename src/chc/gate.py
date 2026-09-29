"""A deployment gate that can be read at any time: per zone, a candidate policy runs in shadow of
the baseline until the evidence says deploy, hold, experiment or roll back.

The baseline acts, and each decision is logged with its propensity. After every batch -- every
fifteen minutes, or every decision -- :meth:`DeploymentGate.update` returns a :data:`Verdict` per
zone:

* ``"deploy"``: the candidate beats the baseline by more than ``delta``;
* ``"hold"``: the candidate is worse than the baseline by more than ``delta_harm`` and retires, or
  the zone's channel drifted and its evidence starts again;
* ``"experiment"``: shadow evidence would come too slowly, so the zone logs under a mixture of the
  two policies, which bounds both weights;
* ``"rollback"``: a deployed candidate turned out worse than the baseline, or its channel drifted;
* ``"shadow"``: not yet.

Each verdict is an e-process crossing a threshold: a mixture over constant bets of
``prod(1 + lam Y_t)``, with ``Y_t`` a decision's weighted contrast less the margin, scaled so that
``Y_t >= -1``. Only the baseline's weight needs a bound, and the logging design supplies it
(Waudby-Smith et al. 2022, arXiv:2210.10768). Across zones, e-BH over e-values that freeze when
their zone stops gathering keeps the false-deploy rate at every read (Wang, Dandapanthula and
Ramdas 2025, arXiv:2502.08539).

What the guarantee needs, and the gate cannot check:

* **Propensities logged at decision time.** A behaviour law fitted to the logs afterwards, as
  :func:`chc.offpolicy.fit_behavior_policy` does, took a lab gate's type-I error from 0.013 to
  1.000. The gate takes the logged propensity and refuses a batch whose propensities are not those
  of the policy it asked to log, but it cannot tell a logged propensity from a refitted one. A
  :class:`DecisionLog` is the record that carries it: read back from storage, it refuses a record
  that does not say which version it is, or lacks a field, rather than filling one in.
* **No spillover between zones**: each zone's null holds given every zone's past.
* **No carryover**: a decision's reward does not depend on earlier decisions. On a plant with
  memory, per-decision increments certify the myopic contrast; in the lab they deployed, with
  probability 1.000, a candidate that was +0.029 better myopically and -0.010 worse in the long run.
  :func:`chc.evaluation.evaluate_plan` estimates a plan's value on such a plant.
* **Rewards in** ``[0, 1]``.

A zone's channel is watched through per-decision drift e-values. :func:`channel_drift_evalues`
reads them off the Gaussian dither a :class:`DecisionLog` records, and :class:`DriftAlarm` turns
them into an alarm outside the gate as well. Once the channel has moved, :func:`channel_move`
re-reads it off the same dither, and :meth:`ChannelMove.price` puts the move in a plan's own cost.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

from chc.plan import DecisionWeight

_log = logging.getLogger(__name__)

Verdict = Literal["deploy", "shadow", "experiment", "hold", "rollback"]
GateMode = Literal["shadow", "experiment", "deployed", "retired"]

_Array = NDArray[np.float64]

_BETS = 0.9 * 2.0 ** -np.arange(12)  # constant bets, mixed with equal weight
_LOG_BETS = math.log(_BETS.size)
_MATCH = 1e-6  # relative gap allowed between a logged propensity and the mode's
_DRIFT_BETS = 2.0 ** -np.arange(8)  # |theta|, in residual scales per dither scale
_DITHER_LEVEL = 1e-9  # a dither this far from its stated scale is a slip, not a draw


def _vector(value: ArrayLike, name: str) -> _Array:
    array = np.array(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a vector, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} is not finite")
    array.setflags(write=False)
    return array


def _evalues(value: ArrayLike, name: str, size: int | None = None) -> _Array:
    """E-values as one row per decision and one column per detector."""
    evalues = np.array(value, dtype=np.float64)
    if evalues.ndim == 1:
        evalues = evalues[:, None]
    rows = "decisions" if size is None else size
    if (
        evalues.ndim != 2
        or evalues.shape[1] == 0
        or (size is not None and evalues.shape[0] != size)
    ):
        raise ValueError(
            f"{name} must have shape ({rows},) or ({rows}, detectors), got {evalues.shape}"
        )
    if not (np.all(np.isfinite(evalues)) and np.all(evalues >= 0.0)):
        raise ValueError(f"{name} e-values must be finite and non-negative")
    evalues.setflags(write=False)
    return evalues


@dataclass(frozen=True)
class GateConfig:
    """What the gate tests, and at what level.

    ``delta`` is the improvement margin: a DEPLOY claims ``V_new - V_base > delta``. ``delta_harm``
    is the harm margin: a HOLD for harm or a ROLLBACK claims ``V_base - V_new > delta_harm``.
    ``alpha`` bounds the expected share of wrong DEPLOY decisions across zones, ``alpha_harm`` each
    HOLD for harm and each ROLLBACK, and ``drift_arl`` is the average number of decisions between
    false drift alarms on an unchanged channel. ``rho`` is the candidate's share of the logging
    mixture in EXPERIMENT.

    ``min_effect``, the smallest improvement beyond ``delta`` worth deploying, and ``horizon``, the
    decisions per zone the gate may spend, decide when to switch to EXPERIMENT, and no guarantee
    depends on them.

    Raises:
        ValueError: on a margin outside ``[0, 1)``, a level or ``rho`` outside ``(0, 1)``, a
            ``min_effect`` that is not positive, a ``horizon`` below 1, or a ``drift_arl`` not
            above 1.
    """

    delta: float
    delta_harm: float
    min_effect: float
    horizon: int
    alpha: float = 0.05
    alpha_harm: float = 0.05
    drift_arl: float = 10_000.0
    rho: float = 0.5

    def __post_init__(self) -> None:
        for name in ("delta", "delta_harm"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1), got {value}")
        for name in ("alpha", "alpha_harm", "rho"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1), got {value}")
        if not (math.isfinite(self.min_effect) and self.min_effect > 0.0):
            raise ValueError(f"min_effect must be positive, got {self.min_effect}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be at least 1 decision, got {self.horizon}")
        if not self.drift_arl > 1.0:
            raise ValueError(f"drift_arl must exceed 1, got {self.drift_arl}")


@dataclass(frozen=True)
class ZonePlan:
    """What is known about a zone's candidate before any data, used only to decide when to switch
    to EXPERIMENT.

    ``chi2`` is the chi-square divergence of the candidate's action law from the baseline's, per
    decision: the variance of the shadow weight. ``tv`` is their total variation distance. Compute
    both from the two policies, never from logged weights: when the candidate's action law is wider
    than the baseline's by a factor of ``sqrt(2)`` or more, ``chi2`` is infinite, an estimate from
    weights is finite, and a tail index estimated from them reads far heavier than the true one.
    For Gaussian actions ``N(m_new, s_new^2)`` against ``N(m_base, s_base^2)``::

        1 + chi2 = s_base^2 / (s_new sqrt(2 s_base^2 - s_new^2))
                   * exp((m_new - m_base)^2 / (2 s_base^2 - s_new^2)),

    infinite when ``s_new^2 >= 2 s_base^2``.

    Raises:
        ValueError: on a negative or NaN ``chi2``, or a ``tv`` outside ``[0, 1]``.
    """

    chi2: float
    tv: float

    def __post_init__(self) -> None:
        if not self.chi2 >= 0.0:
            raise ValueError(f"chi2 must be non-negative, got {self.chi2}")
        if not 0.0 <= self.tv <= 1.0:
            raise ValueError(f"tv must lie in [0, 1], got {self.tv}")


def _column(records: list[Mapping[str, object]], key: str) -> list[object]:
    missing = [i for i, record in enumerate(records) if key not in record]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(records)} records have no {key!r} (the first is record"
            f" {missing[0]}); a decision log is read as it was written, never filled in"
        )
    return [record[key] for record in records]


@dataclass(frozen=True)
class DecisionLog:
    """What each decision recorded when it was taken, one entry per decision.

    ``action`` is the action as applied, one row per decision for a vector action. ``propensity``
    is the probability, or density, of drawing it, computed by the policy that drew it when it drew
    it: the number the gate's guarantee is stated on. ``saturated`` flags a decision whose action
    was clipped, so that what was applied is not what was drawn. ``dither``, optional, is the
    Gaussian perturbation drawn for the policy's action, as drawn: on a clipped decision, the draw,
    not what the clip left of it.

    Stored, a decision is a record with ``decision_log_version`` beside those fields.
    :meth:`from_records` reads version :attr:`VERSION` and refuses a record with no version, or
    with another, rather than guess what an older log meant. What a new version would require of
    stored logs is in ``docs/adr/0015-what-a-logged-decision-records.md``.

    Raises:
        ValueError: on entries whose number differs between the fields, a non-finite entry, a
            propensity that is not positive, a ``saturated`` that is not boolean, or a ``dither``
            whose shape is not the action's.
    """

    action: NDArray[np.float64]
    propensity: NDArray[np.float64]
    saturated: NDArray[np.bool_]
    dither: NDArray[np.float64] | None = None

    VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        propensity = _vector(self.propensity, "propensity")
        size = propensity.shape[0]
        if size and not np.min(propensity) > 0.0:
            raise ValueError(
                f"a logged propensity must be positive, got {np.min(propensity)}: the action could"
                " not have been drawn"
            )
        action = np.array(self.action, dtype=np.float64)
        if action.ndim not in (1, 2) or action.shape[0] != size:
            raise ValueError(
                f"action must have shape ({size},) or ({size}, actions), got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("action is not finite")
        saturated = np.array(self.saturated)
        if saturated.size == 0:
            saturated = saturated.astype(np.bool_)
        if saturated.dtype != np.bool_ or saturated.shape != (size,):
            raise ValueError(
                f"saturated must hold {size} booleans, got {saturated.dtype} of shape"
                f" {saturated.shape}"
            )
        for name, array in (
            ("propensity", propensity),
            ("action", action),
            ("saturated", saturated),
        ):
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.dither is not None:
            dither = np.array(self.dither, dtype=np.float64)
            if dither.shape != action.shape:
                raise ValueError(f"dither has shape {dither.shape}, the action {action.shape}")
            if not np.all(np.isfinite(dither)):
                raise ValueError("dither is not finite")
            dither.setflags(write=False)
            object.__setattr__(self, "dither", dither)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]]) -> DecisionLog:
        """The log that stored records hold, such as JSON lines or a table's rows.

        Every record needs ``decision_log_version`` equal to :attr:`VERSION`, and ``action``,
        ``propensity`` and ``saturated``, a boolean. ``dither`` is on every record or on none. Keys
        the log does not define, such as a timestamp, the reward or the caller's own ``version``,
        are left to the caller.

        Raises:
            ValueError: on a record with no version or with another, a missing field, a
                ``saturated`` that is not a boolean, or a ``dither`` on some records and not
                others; and on what the constructor refuses.
        """
        rows = list(records)
        versions = _column(rows, "decision_log_version")
        other = sorted(
            {
                repr(v)
                for v in versions
                if isinstance(v, bool | np.bool_)
                or not isinstance(v, int | np.integer)
                or v != cls.VERSION
            }
        )
        if other:
            raise ValueError(
                f"records of decision_log_version {', '.join(other)}; this version of chc reads"
                f" version {cls.VERSION} only"
            )
        saturated = _column(rows, "saturated")
        if not all(isinstance(flag, bool | np.bool_) for flag in saturated):
            raise ValueError("saturated must be true or false on every record")
        with_dither = sum(row.get("dither") is not None for row in rows)
        if 0 < with_dither < len(rows):
            raise ValueError(
                f"{with_dither} of {len(rows)} records carry a dither: a log records the dither of"
                " every decision or of none"
            )
        return cls(
            action=np.array(_column(rows, "action"), dtype=np.float64),
            propensity=np.array(_column(rows, "propensity"), dtype=np.float64),
            saturated=np.array(saturated, dtype=np.bool_),
            dither=np.array(_column(rows, "dither"), dtype=np.float64) if with_dither else None,
        )

    def to_records(self) -> list[dict[str, object]]:
        """One record per decision, versioned and ready for JSON, which :meth:`from_records`
        reads back to the bit."""
        records: list[dict[str, object]] = []
        for i in range(self.propensity.shape[0]):
            record: dict[str, object] = {
                "decision_log_version": self.VERSION,
                "action": self.action[i].tolist(),
                "propensity": float(self.propensity[i]),
                "saturated": bool(self.saturated[i]),
            }
            if self.dither is not None:
                record["dither"] = self.dither[i].tolist()
            records.append(record)
        return records

    def dither_draws(self) -> NDArray[np.float64]:
        """The dither of every decision, for a reader that needs each action to carry its whole
        draw, as :func:`channel_move` does: on a clipped decision, what the product of the residual
        and the draw reads is scaled by the chance that the draw was not clipped.

        Raises:
            ValueError: on a log with no dither, or with a clipped decision.
        """
        if self.dither is None:
            raise ValueError("the log records no dither")
        if self.saturated.any():
            raise ValueError(
                f"{int(self.saturated.sum())} of {self.saturated.size} decisions were clipped (the"
                f" first is decision {int(np.argmax(self.saturated))}), and a clipped decision's"
                " action does not carry the whole of its draw"
            )
        return self.dither


@dataclass(frozen=True)
class ZoneBatch:
    """One zone's decisions since the gate was last read, one entry per decision.

    ``candidate`` and ``baseline`` are the two policies' propensities of the action taken -- a
    probability for a discrete action, a density for a continuous one -- and ``logged`` is the
    propensity recorded when the action was drawn, under the policy the zone's mode asked for;
    :meth:`from_log` reads it off a :class:`DecisionLog`. ``drift``, optional, holds per-decision
    e-values for "the channel is unchanged", one column per detector, each with expectation at most
    1 given the past while the channel holds: :func:`channel_drift_evalues` reads them off the
    log's dither.

    Raises:
        ValueError: on arrays of different lengths, a non-finite entry, a reward outside ``[0, 1]``,
            a negative propensity, a logged propensity that is not positive, or a negative e-value.
    """

    reward: NDArray[np.float64]
    candidate: NDArray[np.float64]
    baseline: NDArray[np.float64]
    logged: NDArray[np.float64]
    drift: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        reward = _vector(self.reward, "reward")
        size = reward.shape[0]
        if size and not (np.min(reward) >= 0.0 and np.max(reward) <= 1.0):
            raise ValueError(
                f"rewards must lie in [0, 1], got [{np.min(reward)}, {np.max(reward)}]"
            )
        object.__setattr__(self, "reward", reward)
        for name in ("candidate", "baseline", "logged"):
            array = _vector(getattr(self, name), name)
            if array.shape[0] != size:
                raise ValueError(f"{name} has {array.shape[0]} entries for {size} rewards")
            if size and np.min(array) < 0.0:
                raise ValueError(f"{name} holds a negative propensity, {np.min(array)}")
            object.__setattr__(self, name, array)
        if size and np.min(self.logged) <= 0.0:
            raise ValueError("a logged propensity is 0: the action could not have been drawn")
        if self.drift is not None:
            object.__setattr__(self, "drift", _evalues(self.drift, "drift", size))

    @classmethod
    def from_log(
        cls,
        log: DecisionLog,
        *,
        reward: NDArray[np.float64],
        candidate: NDArray[np.float64],
        baseline: NDArray[np.float64],
        drift: NDArray[np.float64] | None = None,
    ) -> ZoneBatch:
        """The batch whose logged propensities are the ones ``log`` recorded, decision by decision.

        Raises:
            ValueError: on what the constructor refuses.
        """
        return cls(reward, candidate, baseline, log.propensity, drift)


def _entries(value: ArrayLike, shape: tuple[int, ...], name: str) -> _Array:
    array = np.array(value, dtype=np.float64)
    if array.ndim and array.shape != shape:
        raise ValueError(f"{name} must be a scalar or of shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} is not finite")
    return np.broadcast_to(array, shape)


def _residual_rows(residual: ArrayLike, size: int) -> _Array:
    """The residual as one row per decision and one column per state."""
    rows = np.array(residual, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[:, None]
    if rows.ndim != 2 or rows.shape[0] != size:
        raise ValueError(
            f"residual must have shape ({size},) or ({size}, states), one row per decision, got"
            f" {rows.shape}"
        )
    if not np.all(np.isfinite(rows)):
        raise ValueError("residual is not finite")
    return rows


def _standardised_dither(dither: _Array, sigma: _Array) -> _Array:
    """The dither over its stated scale, refused when a two-sided chi-square test rejects the draws
    as ``N(0, 1)`` there."""
    xi = dither / sigma
    size = xi.shape[0]
    if size:
        squares = np.sum(xi * xi, axis=0)
        tail = 2.0 * np.minimum(special.chdtr(size, squares), special.chdtrc(size, squares))
        if np.any(tail < _DITHER_LEVEL):
            j = int(np.argmin(tail))
            raise ValueError(
                f"the dither of action {j}, over its dither_scale {sigma[j]:g}, has mean square"
                f" {squares[j] / size:.4g} over {size} decisions, where N(0, 1) draws give 1"
                f" (two-sided chi-square p = {tail[j]:.2g}): drawn at another scale than stated,"
                " or a variance passed for a standard deviation"
            )
    return xi


def channel_drift_evalues(
    log: DecisionLog,
    residual: ArrayLike,
    *,
    dither_scale: ArrayLike,
    radius: ArrayLike,
    residual_scale: ArrayLike,
) -> NDArray[np.float64]:
    """Per-decision e-values for "every entry of the one-step channel lies within ``radius`` of the
    model's", read off the logged dither: a :class:`ZoneBatch`'s ``drift``, or what a
    :class:`DriftAlarm` runs on.

    ``residual`` holds, for each decision in ``log``, the next state less the model's one-step
    prediction at the action as applied, one column per state. ``dither_scale`` is the standard
    deviation each action's dither was drawn with. ``radius`` bounds, entry by entry, how far the
    plant's one-step channel ``d x_(t+1) / d u_t`` may lie from the model's while nothing has
    moved, in the residual's units per unit of action. ``residual_scale``, one per state and fixed
    before the data, such as the model's one-step noise, is the unit the bets are placed in. A
    scalar stands for every state or action.

    For state ``i`` and action ``j``, the dither is standardised, ``xi = dither_j /
    dither_scale_j``, and the residual is moved to the radius's edge and scaled,
    ``r = (residual_i - radius_ij u_j) / residual_scale_i`` against a channel that grew and
    ``(residual_i + radius_ij u_j) / residual_scale_i`` against one that shrank. The e-value at a
    bet ``theta`` is ``exp(theta r xi - theta^2 r^2 / 2)``, with ``theta`` among ``1, 1/2, ...,
    1/128``, negated against shrinkage. On a plant whose one-step map is ``g(x) + B u`` plus noise,
    ``r = c + k xi``, where ``k`` is how far ``B_ij`` lies past that edge, times
    ``dither_scale_j / residual_scale_i``, and ``c`` is everything else. When ``xi`` is ``N(0, 1)``
    and independent of ``c``, ``E[exp(theta r xi - theta^2 r^2 / 2) | c] = 1 / |1 - theta k|``,
    which is at most 1 on each side the entry has not crossed, whatever the model's error in ``g``,
    the noise's law or the policy. Row ``t`` of the result, reshaped to ``(states, actions, 2,
    8)``, holds decision ``t``'s e-values by state, action, growth then shrinkage, and bet.

    A clipped decision is read the same way: the bet reads the draw, and the residual moves with
    the action as applied. With a box cutting the draw to ``h = clip(xi, -m1, m2)`` in dither
    units, ``r = c + k h``, and the mean given ``c`` is ``1 - (Phi(u2) - Phi(u1)) (1 - 1 / s)``,
    with ``s = 1 - theta k``, ``u1 = -s m1 - theta c`` and ``u2 = s m2 - theta c``. That is at
    most 1 on each side the entry has not crossed, exactly 1 on the radius's edge whatever the
    clip, and at least 1 on the side it has (``validation/dither_drift_evalue.mac`` STEP 8). On the
    sides not crossed, the mean stays at most 1 for an action applied as any nondecreasing function
    of its own draw, such as a saturating actuator. The draw is what the log must carry: the dither
    as applied, put in its place, is not an e-value, and on the edge, with the nominal action on a
    bound, its mean reaches 1.15.

    On the lab's plant, whose model has the drift wrong and whose noise is Laplace, a
    :class:`DriftAlarm` on these e-values ran at least 3.2 and 5.6 times its target on an unchanged
    channel, at ``10^3`` and ``10^4``, and caught a channel at 1.4 instead of 1.07 in 259 and 456
    decisions on average: 1.56 and 1.37 times an oracle that knew which entry moved, which way and
    at what rate (``scripts/bench_drift.py``, 300 paths). With the actions boxed to ``+-0.5``, so
    that a quarter of the decisions clipped, it ran at least 2.8 and 4.6 times its target and caught
    the move in 438 and 749 decisions, 1.69 and 1.64 times the unboxed delays. With the clipped
    decisions' e-values set to 0 instead, which is also valid, it caught the move on none of the
    paths within 3000 decisions.

    What the guarantee needs, and the function cannot check:

    * **A plant affine in the action over one step.** The model's error in ``g`` and the noise
      must not depend on the dither drawn at that step. A response nonlinear in the action, such
      as ``xi^2 - 1`` in the next state, is uncorrelated with the dither and still breaks the
      identity: in the lab, a running product of such e-values passed 20 on 27% of paths, where
      Ville's inequality allows 5%. A plant integrated over a step has a one-step channel that
      carries the drift's Jacobian too, to first order in the step.
    * **A dither drawn as stated and logged as drawn**: each action's from
      ``N(0, dither_scale_j^2)``, independently of the other actions' and of the past, and applied
      whole or cut by a box on that action alone, fixed before the draw. A projection that couples
      the actions, such as a budget shared across them, moves one action's residual with another's
      draw, and is not covered. With a fifth more variance than stated, the e-value's mean at
      ``theta r = 1`` is ``exp(0.1) = 1.105``.
    * **A radius that covers the identification error.** The e-values hold while every entry lies
      inside it. On the same plant, a channel 0.07 from the model's, watched with no radius,
      alarmed after 0.36 of the average run length the alarm was set for. A standard error, such as
      :attr:`chc.dynamics_id.CausalDynamicsFit.channel_error`, is a scale, not a radius.

    Raises:
        ValueError: on a log with no dither; on a residual without one row per decision, a
            non-finite entry, a scale that is not positive, a negative radius, or a scale or radius
            that is neither a scalar nor of its full shape; and on a dither whose draws, over
            ``dither_scale``, a two-sided chi-square test rejects at ``1e-9``: a slip in units, or
            a variance passed for a standard deviation.
    """
    if log.dither is None:
        raise ValueError("the log records no dither")
    dither = log.dither
    size = dither.shape[0]
    action = log.action if log.action.ndim == 2 else log.action[:, None]
    xi = dither if dither.ndim == 2 else dither[:, None]
    r = _residual_rows(residual, size)
    states, actions = r.shape[1], action.shape[1]
    sigma = _entries(dither_scale, (actions,), "dither_scale")
    scale = _entries(residual_scale, (states,), "residual_scale")
    edge = _entries(radius, (states, actions), "radius")
    for name, array in (("dither_scale", sigma), ("residual_scale", scale)):
        if not np.all(array > 0.0):
            raise ValueError(f"{name} must be positive, got {np.min(array)}")
    if not np.all(edge >= 0.0):
        raise ValueError(f"radius must be non-negative, got {np.min(edge)}")
    xi = _standardised_dither(xi, sigma)
    side = np.array([1.0, -1.0])
    moved = r[:, :, None, None] - side * edge[None, :, :, None] * action[:, None, :, None]
    bet = (moved / scale[None, :, None, None])[..., None] * (side[:, None] * _DRIFT_BETS)
    log_e = bet * xi[:, None, :, None, None] - bet * bet / 2.0
    return np.exp(log_e).reshape(size, states * actions * side.size * _DRIFT_BETS.size)


class DriftAlarm:
    """Shiryaev-Roberts over e-values for "the channel is unchanged": an alarm that sounds, on an
    unchanged channel, after ``arl`` decisions on average or later.

    Each detector's statistic runs ``R_t = (R_(t-1) + 1) e_t`` from ``R_0 = 0``, and the alarm
    sounds when their average reaches ``arl``. While every e-value has expectation at most 1 given
    the past, the average less ``t`` is a supermartingale, whatever the dependence between the
    detectors, so the average run length is at least ``arl`` (Shin, Ramdas and Rinaldo 2024,
    arXiv:2203.03532). After an alarm the statistic starts again at the next call, and the call's
    e-values after the alarming decision are dropped: they were computed against the model the
    alarm rejected. :class:`DeploymentGate` runs one per zone.

    Raises:
        ValueError: on an ``arl`` not above 1.
    """

    def __init__(self, arl: float) -> None:
        if not arl > 1.0:
            raise ValueError(f"arl must exceed 1, got {arl}")
        self.arl = float(arl)
        self._running: _Array | None = None

    @property
    def detectors(self) -> int | None:
        """How many detectors the running statistic follows: ``None`` before the first e-value and
        after an alarm, when any number may start."""
        return None if self._running is None else self._running.shape[0]

    @property
    def statistic(self) -> float:
        """The detectors' average statistic, 0 before the first e-value and after an alarm."""
        return 0.0 if self._running is None else float(self._running.mean())

    def update(self, evalues: ArrayLike) -> bool:
        """Add e-values, one row per decision and one column per detector, and say whether the
        alarm sounded.

        Raises:
            ValueError: on e-values that are not finite and non-negative, or whose number of
                detectors is not the running statistic's.
        """
        rows = _evalues(evalues, "evalues")
        if self._running is not None and rows.shape[1] != self._running.shape[0]:
            raise ValueError(
                f"{rows.shape[1]} drift detectors, where the running statistic has"
                f" {self._running.shape[0]}"
            )
        if rows.shape[0] == 0:
            return False
        running = np.zeros(rows.shape[1]) if self._running is None else self._running
        for t, row in enumerate(rows):
            running = (running + 1.0) * row
            if running.mean() >= self.arl:
                self._running = None
                _log.warning(
                    "channel drift alarm: the detectors' average Shiryaev-Roberts statistic"
                    " reached %.4g >= %g at decision %d of %d in this update; %d later e-values"
                    " were dropped",
                    running.mean(),
                    self.arl,
                    t,
                    rows.shape[0],
                    rows.shape[0] - t - 1,
                    extra={
                        "chc_event": "drift_alarm",
                        "statistic": float(running.mean()),
                        "arl": self.arl,
                        "decision": t,
                        "detectors": rows.shape[1],
                    },
                )
                return True
        self._running = running
        return False


@dataclass(frozen=True)
class MovePrice:
    """What a move of the one-step channel costs a plan, against the plan that knew it, in
    expectation. *Experimental.*

    See :meth:`ChannelMove.price`.
    """

    # Keeping the plan: d' W d / 2, estimated without bias, so negative when the estimate's noise
    # outweighs the move rather than clipped, which would bias it up.
    keep: float
    # keep's standard error: a scale, not coverage
    keep_error: float
    # A plan re-solved on the estimate: tr(W S) / 2, what the estimate's error costs it on average
    replan: float


@dataclass(frozen=True)
class ChannelMove:
    """How far the plant's one-step channel lies from the model's, read off a log's dither.
    *Experimental.*

    See :func:`channel_move`.
    """

    # (states, actions): the plant's one-step channel less the model's, per unit of action
    estimate: NDArray[np.float64]
    # (states * actions, states * actions): the estimate's covariance, entry (i, j) of the channel
    # at index i * actions + j, the order chc.plan.DecisionWeight reads it in
    covariance: NDArray[np.float64]
    effective_size: float  # (sum w)^2 / sum w^2: how many equally weighted decisions it is worth

    def price(self, weight: DecisionWeight) -> MovePrice:
        """What the move costs the plan ``weight`` was read off, kept or re-solved on the estimate.

        With ``W`` the plan's :meth:`chc.plan.CausalPlan.decision_weight`, ``d`` the move and ``S``
        the covariance, keeping the plan loses ``d' W d / 2`` to the plan that knew ``d``, and a
        plan re-solved on the estimate ``dh`` loses ``(dh - d)' W (dh - d) / 2``, ``tr(W S) / 2`` on
        average. Since ``E[dh' W dh] = d' W d + tr(W S)``, keeping is priced at
        ``(dh' W dh - tr(W S)) / 2``, and its standard error is ``sqrt(4 d' W S W d + 2 tr((W S)^2))
        / 2``, with ``d' W S W d`` read as ``dh' W S W dh - tr((W S)^2)``, for the same reason, and
        floored at 0: ``dh`` put in for ``d`` reads its variance three times too large on a channel
        that has not moved (``validation/dither_channel_move.mac`` STEPs 4-5,
        ``proofs/dither_channel_move.v``).

        **Two expectations, not a decision rule.** Re-planning pays in expectation when
        ``d' W d > tr(W S)``, but comparing the two prices of one log selects on its error: the
        logs whose estimate reads a large move are the ones whose estimate errs most along ``W``,
        and they are the ones re-planned. On the budgeted plan of
        ``scripts/bench_channel_move.py``, 200 logs of each size, re-planning when the keep price
        beat the re-plan price lost 0.149 where always keeping lost 0.095, at 30 decisions, and
        0.100 and 0.043 where always re-planning lost 0.080 and 0.024, at 100 and 400. Re-planning
        on the estimate shrunk by ``1 - tr(W S) / dh' W dh``, or choosing on one half of the log
        and re-planning on the other, did no better than the better of the two throughout. Re-read
        the move on the decisions logged after the choice, such as those after a
        :class:`DriftAlarm` sounds, and the re-plan price is the re-solved plan's.

        **How far they reach.** Both are second order in the move, and hold as far as ``W`` does:
        while the plan's active set stays as it is, and for a move small against the channel (see
        :meth:`chc.plan.CausalPlan.decision_weight`). With a move of 10-15% of the channel's
        entries, the kept plan lost 0.095 where the keep price at the true move reads 0.107. A
        plan re-solved on the estimate from 30 decisions lost 0.18 where the re-plan price read
        0.37, its box capping how far it moved; from 400 and 1600, 0.024 and 0.0060 against 0.026
        and 0.0066.

        Raises:
            ValueError: on a weight whose channel is not the move's shape.
        """
        if tuple(weight.channel_shape) != self.estimate.shape:
            raise ValueError(
                f"the weight reads a channel of shape {tuple(weight.channel_shape)}, the move is"
                f" of shape {self.estimate.shape}"
            )
        w = np.asarray(weight.matrix, dtype=np.float64)
        d = self.estimate.ravel()
        ws = w @ self.covariance
        trace, trace_squared = float(np.trace(ws)), float(np.trace(ws @ ws))
        leverage = max(float(d @ ws @ w @ d) - trace_squared, 0.0)
        return MovePrice(
            keep=(float(d @ w @ d) - trace) / 2.0,
            keep_error=math.sqrt(4.0 * leverage + 2.0 * trace_squared) / 2.0,
            replan=trace / 2.0,
        )


def channel_move(
    log: DecisionLog,
    residual: ArrayLike,
    *,
    dither_scale: ArrayLike,
    forgetting: float = 1.0,
) -> ChannelMove:
    """How far the plant's one-step channel lies from the model's, read off the logged dither, with
    the older decisions forgotten at a constant rate. *Experimental.*

    ``residual`` and ``dither_scale`` are :func:`channel_drift_evalues`': per decision, the next
    state less the model's one-step prediction at the action as applied, and the standard deviation
    each action's dither was drawn with. With ``xi = dither_j / dither_scale_j``, decision ``t``
    reads entry ``(i, j)`` as the product ``z = residual_i xi_j / dither_scale_j``. On a plant whose
    one-step map is ``g(x) + B u`` plus noise, ``residual_i = c + sum_k d_ik dither_k``, with ``d``
    the plant's channel less the model's and ``c`` everything else: the model's error in ``g``, the
    policy's action through ``d``, and the noise. When the dither is ``N(0, dither_scale^2)`` and
    independent of ``c``, ``E[z] = d_ij`` whatever ``c`` holds, and ``c`` adds to the product's
    variance rather than to its mean (``validation/dither_channel_move.mac`` STEPs 1-2).

    The estimate is the products' mean under the weights ``forgetting^k``, ``k`` the decisions
    since, so the last decision weighs 1. Re-read as decisions arrive, it follows a channel that
    moves, and is worth ``(1 + forgetting) / (1 - forgetting)`` equally weighted decisions once the
    log is long (STEP 3); ``1``, the default, weighs every decision alike. The covariance is the
    products' own, weighted the same way and scaled so that it is unbiased when they share one:
    ``E[sum_t w_t^2 (z_t - dh)(z_t - dh)'] = sum_t w_t^2 [(sum_(s != t) w_s)^2 + sum_(s != t)
    w_s^2] Var(z) / (sum w)^2``, which is ``(n - 1) Var(z)`` for equal weights. The products'
    errors are uncorrelated across decisions, since each dither is drawn after everything before
    it, so it holds for noise that moves with the state as well; unequal variances leave a bias of
    order ``1 / effective_size`` under forgetting, and none under equal weights.

    :meth:`ChannelMove.price` reads the move in the cost of a plan: what keeping it costs, and what
    re-solving it on the estimate would.

    On the drift monitor's plant (``scripts/bench_channel_move.py``, 2000 paths), whose model has
    the drift wrong and whose noise is Laplace, the estimate of a move of 0.07 lay within 0.75 of
    its standard errors of it over 500 and 2000 decisions, at dithers of 0.3 and 1.0; its spread
    was 0.99-1.03 of its standard error, and ``estimate +- 1.96`` of them covered the move on
    0.943-0.955 of the paths. With the channel at 1.4 for the last 1000 of 4000 decisions, the
    intervals covered what each estimate follows on 0.960 of the paths without forgetting and
    0.951 forgetting at 0.995.

    What the estimate needs, and the function cannot check: :func:`channel_drift_evalues`'
    contract. A plant affine in the action over one step, since a response nonlinear in the action
    reads into the product; a dither drawn as stated and applied as logged; and, for the move to be
    one number per entry, a channel that does not move with the state over the log: the estimate
    is the one-step channel's average over the decisions, weighted as above, and a plant integrated
    over a step has a channel that carries the drift's Jacobian, to first order in the step.

    Raises:
        ValueError: on a log with no dither or with a clipped decision (see
            :meth:`DecisionLog.dither_draws`), fewer than two decisions, a residual without one row
            per decision or not finite, a ``dither_scale`` that is not positive or is neither a
            scalar nor one per action, a dither the chi-square test in :func:`channel_drift_evalues`
            rejects, a ``forgetting`` outside ``(0, 1]``, or one so close to 0 that a single
            decision carries all the weight.
    """
    dither = log.dither_draws()
    size = dither.shape[0]
    xi = dither if dither.ndim == 2 else dither[:, None]
    r = _residual_rows(residual, size)
    if size < 2:
        raise ValueError(f"{size} decisions: the estimate's covariance needs two at least")
    if not 0.0 < forgetting <= 1.0:
        raise ValueError(f"forgetting must lie in (0, 1], got {forgetting}")
    states, actions = r.shape[1], xi.shape[1]
    sigma = _entries(dither_scale, (actions,), "dither_scale")
    if not np.all(sigma > 0.0):
        raise ValueError(f"dither_scale must be positive, got {np.min(sigma)}")
    xi = _standardised_dither(xi, sigma)
    products = (r[:, :, None] * (xi / sigma)[:, None, :]).reshape(size, states * actions)
    w = forgetting ** np.arange(size - 1, -1, -1, dtype=np.float64)
    total, squares = float(w.sum()), w * w
    estimate = w @ products / total
    # Sums over the other decisions, as sums of positive terms: ``total - w_t`` cancels to nothing
    # when one weight carries almost all of it.
    others = _exclusive_sums(w)
    spread = float(np.sum(squares * (others * others + _exclusive_sums(squares))))
    if not spread > 0.0:
        raise ValueError(
            f"forgetting at {forgetting:g} leaves all the weight on the last decision: nothing is"
            " left to estimate the covariance from"
        )
    centred = products - estimate
    covariance = (centred.T * squares) @ centred * (float(squares.sum()) / spread)
    for array in (estimate, covariance):
        array.setflags(write=False)
    return ChannelMove(
        estimate=estimate.reshape(states, actions),
        covariance=covariance,
        effective_size=total * total / float(squares.sum()),
    )


def _exclusive_sums(values: _Array) -> _Array:
    """``sum_(s != t) values_s`` for each ``t``, from the two sides, with nothing subtracted."""
    before = np.concatenate([[0.0], np.cumsum(values)[:-1]])
    after = np.concatenate([np.cumsum(values[::-1])[:-1][::-1], [0.0]])
    return before + after


def _growth(mean: float, variance: float) -> float:
    """The growth rate one fixed bet guarantees against every law on ``[-1, oo)`` with this mean
    and variance: ``sup_lam E log(1 + lam Y)`` at the two-point law, which is the worst case (lab
    AG2; the minimax property is proved by a quadratic minorant of ``log(1 + lam y)``)."""
    if mean <= 0.0 or math.isinf(variance):
        return 0.0
    if variance == 0.0:
        return math.log1p(mean)
    p = variance / (variance + (1 + mean) ** 2)
    return p * math.log(variance / (variance + mean * (1 + mean))) + (1 - p) * math.log1p(mean)


def _ebh(log_e: _Array, alpha: float) -> NDArray[np.bool_]:
    """e-BH on log e-values: each ``e_k >= K / (alpha r*)``, where
    ``r* = max{r : e_(r) >= K / (alpha r)}`` and ``e_(r)`` is the ``r``-th largest."""
    k = log_e.size
    ranks = np.arange(1, k + 1)
    ok = np.sort(log_e)[::-1] >= np.log(k / (alpha * ranks))
    if not ok.any():
        return np.zeros(k, dtype=bool)
    r = int(np.max(ranks[ok]))
    return log_e >= math.log(k / (alpha * r))


def _epoch_weight(epoch: int) -> float:
    return 0.9 if epoch == 1 else 0.1 / ((epoch - 1) * epoch)


class _Evidence:
    """A mixture over evidence epochs of mixtures over constant bets: one test supermartingale.

    Epoch ``k`` carries weight ``0.9`` for ``k = 1`` and ``0.1 / ((k - 1) k)`` after, which sum to
    1, and an epoch not yet opened counts at its initial value 1. A drift alarm zeroes the current
    epoch -- a nonnegative supermartingale may always drop to zero -- and opens the next, so
    evidence from before a change never supports a decision after it.
    """

    def __init__(self) -> None:
        self.epoch = 1
        self.log_wealth = np.zeros(_BETS.size)

    def add(self, increments: _Array) -> None:
        self.log_wealth += np.log1p(np.multiply.outer(increments, _BETS)).sum(axis=0)

    def restart(self) -> None:
        self.epoch += 1
        self.log_wealth[:] = 0.0

    @property
    def log_mixture(self) -> float:
        return float(functools.reduce(np.logaddexp, self.log_wealth)) - _LOG_BETS

    @property
    def log_value(self) -> float:
        unopened = 0.1 / self.epoch  # the weights of every later epoch, summed
        return float(
            np.logaddexp(math.log(_epoch_weight(self.epoch)) + self.log_mixture, math.log(unopened))
        )


@dataclass
class _Zone:
    plan: ZonePlan
    drift: DriftAlarm
    mode: GateMode = "shadow"
    improvement: _Evidence = field(default_factory=_Evidence)
    harm: _Evidence = field(default_factory=_Evidence)
    rollback: _Evidence = field(default_factory=_Evidence)
    decisions: int = 0


class DeploymentGate:
    """Shadow evidence per zone that ends in DEPLOY, HOLD, EXPERIMENT or ROLLBACK, read any time.

    Whenever the gate is read -- after every batch, or at a time chosen by looking at the data --
    and under the conditions in :mod:`chc.gate`:

    * the expected share of wrong DEPLOY decisions among those made so far is at most ``alpha``,
      under any dependence across zones; a DEPLOY is wrong when the candidate does not beat the
      baseline by more than ``delta``;
    * each HOLD for harm, and each ROLLBACK on evidence, is wrong with probability at most
      ``alpha_harm``;
    * with drift e-values supplied, an unchanged channel alarms at most once per ``drift_arl``
      decisions on average.

    The rules, in order, for each zone at each read:

    1. a drift alarm: HOLD, and the zone's evidence opens a new epoch; ROLLBACK if deployed;
    2. selected by e-BH over every zone's e-value: DEPLOY;
    3. the harm e-value reaches ``1 / alpha_harm``: HOLD, and the zone retires;
    4. in shadow, a verdict at ``min_effect`` would take longer than the ``horizon`` has left, and
       the mixture would gather evidence faster: EXPERIMENT;
    5. otherwise the zone stays where it is, in shadow or in the experiment.

    A deployed zone is watched by a rollback e-process on the baseline's weight. A retired zone
    reads HOLD from then on; a new candidate is a new hypothesis, for a new gate. :meth:`mode` says
    how to log a zone's next batch: under the baseline in ``"shadow"``, under
    ``(1 - rho) baseline + rho candidate`` in ``"experiment"``, and under the candidate once
    ``"deployed"``. The gate keeps its evidence in memory, and one caller updates it at a time.

    Raises:
        ValueError: on no zones.
    """

    def __init__(self, plans: Mapping[str, ZonePlan], config: GateConfig) -> None:
        if not plans:
            raise ValueError("the gate needs at least one zone")
        self.config = config
        self._zones = {
            name: _Zone(plan, DriftAlarm(config.drift_arl)) for name, plan in plans.items()
        }

    def mode(self, zone: str) -> GateMode:
        return self._zones[zone].mode

    def update(self, batches: Mapping[str, ZoneBatch]) -> dict[str, Verdict]:
        """Add each zone's batch to its evidence and return every zone's verdict.

        A zone without a batch keeps its evidence and still gets a verdict, since e-BH reads every
        zone's; a retired zone's batch is ignored. Every batch is checked before any is added.

        Raises:
            KeyError: on a zone the gate has no plan for.
            ValueError: on a batch whose logged propensities are not the mode's, or whose drift
                e-values change their number of detectors mid-run.
        """
        cfg = self.config
        unknown = sorted(set(batches) - set(self._zones))
        if unknown:
            raise KeyError(f"the gate has no plan for zones {unknown}")
        live = {z: b for z, b in batches.items() if self._zones[z].mode != "retired"}
        for z, b in live.items():
            self._check(z, b)
        alarms = {
            z: b.drift is not None and self._zones[z].drift.update(b.drift) for z, b in live.items()
        }
        for z, b in live.items():
            zone = self._zones[z]
            zone.decisions += b.reward.shape[0]
            w_new, w_base, r = b.candidate / b.logged, b.baseline / b.logged, b.reward
            if zone.mode == "deployed":
                zone.rollback.add(((w_base - w_new) * r - cfg.delta_harm) / (1 + cfg.delta_harm))
                continue
            bound = 1.0 if zone.mode == "shadow" else 1.0 / (1.0 - cfg.rho)
            diff = w_new - w_base
            zone.improvement.add((diff * r - cfg.delta) / (bound + cfg.delta))
            zone.harm.add((diff * (1 - r) - cfg.delta_harm) / (bound + cfg.delta_harm))
            if alarms[z]:
                zone.improvement.restart()
                zone.harm.restart()

        # A zone's improvement evidence stops at DEPLOY or retirement, so e-BH reads stopped
        # e-values, and the deployed set is self-consistent at every read.
        log_e = np.array([zone.improvement.log_value for zone in self._zones.values()])
        selected = dict(zip(self._zones, _ebh(log_e, cfg.alpha), strict=True))
        verdicts: dict[str, Verdict] = {}
        for z, zone in self._zones.items():
            before, alarm = zone.mode, alarms.get(z, False)
            verdicts[z] = self._rule(zone, alarm, bool(selected[z]))
            if zone.mode != before or alarm:
                _log.info(
                    "gate verdict",
                    extra={
                        "chc_event": "deployment_gate",
                        "zone": z,
                        "verdict": verdicts[z],
                        "mode": zone.mode,
                        "drift_alarm": alarm,
                        "log_evidence": zone.improvement.log_value,
                        "log_harm": zone.harm.log_value,
                        "log_rollback": zone.rollback.log_value,
                        "epoch": zone.improvement.epoch,
                        "decisions": zone.decisions,
                    },
                )
        return verdicts

    def _check(self, name: str, batch: ZoneBatch) -> None:
        zone, rho = self._zones[name], self.config.rho
        if zone.mode == "shadow":
            expected, policy = batch.baseline, "the baseline"
        elif zone.mode == "deployed":
            expected, policy = batch.candidate, "the candidate"
        else:
            expected = (1.0 - rho) * batch.baseline + rho * batch.candidate
            policy = f"the mixture {1.0 - rho:g} baseline + {rho:g} candidate"
        gap = np.abs(batch.logged - expected) > _MATCH * expected
        if gap.any():
            worst = float(np.max(np.abs(batch.logged - expected) / np.maximum(expected, 1e-300)))
            raise ValueError(
                f"zone {name!r} is in {zone.mode} and must log under {policy}, but {int(gap.sum())}"
                f" of {gap.size} logged propensities differ from it (largest relative gap"
                f" {worst:.3g}, tolerance {_MATCH:g}); a propensity fitted after the fact is not"
                " a logged one"
            )
        detectors = zone.drift.detectors
        if batch.drift is not None and detectors is not None and batch.drift.shape[1] != detectors:
            raise ValueError(
                f"zone {name!r}: {batch.drift.shape[1]} drift detectors, where the running"
                f" statistic has {detectors}"
            )

    def _rule(self, zone: _Zone, alarm: bool, selected: bool) -> Verdict:
        threshold = math.log(1.0 / self.config.alpha_harm)
        if zone.mode == "retired":
            return "hold"
        if zone.mode == "deployed":
            if alarm or zone.rollback.log_value >= threshold:
                zone.mode = "retired"
                return "rollback"
            return "deploy"
        if alarm:
            return "hold"
        if selected:
            zone.mode = "deployed"
            return "deploy"
        if zone.harm.log_value >= threshold:
            zone.mode = "retired"
            return "hold"
        if zone.mode == "shadow" and self._experiment_is_faster(zone):
            zone.mode = "experiment"
            return "experiment"
        return "shadow" if zone.mode == "shadow" else "experiment"

    def _experiment_is_faster(self, zone: _Zone) -> bool:
        """Shadow cannot reach a lone e-BH selection at ``min_effect`` in the horizon left, and the
        mixture, which bounds both weights, would grow evidence faster. Both rates are the two-point
        bound, with the variance ``chi2`` in shadow and ``2 tv / min(rho, 1 - rho)`` under the
        mixture (lab AG2). A rule on the budget alone sent 99.8 % of slow nulls to EXPERIMENT."""
        cfg = self.config
        bound = 1.0 / (1.0 - cfg.rho)
        shadow = _growth(cfg.min_effect / (1 + cfg.delta), zone.plan.chi2 / (1 + cfg.delta) ** 2)
        spread = 2.0 * zone.plan.tv / min(cfg.rho, 1.0 - cfg.rho)
        mixture = _growth(cfg.min_effect / (bound + cfg.delta), spread / (bound + cfg.delta) ** 2)
        weight = _epoch_weight(zone.improvement.epoch)
        need = (
            math.log(_BETS.size * len(self._zones) / (cfg.alpha * weight))
            - zone.improvement.log_mixture
        )
        return mixture > shadow and max(need, 0.0) > shadow * (cfg.horizon - zone.decisions)
