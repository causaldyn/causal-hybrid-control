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
* ``"shadow"``: not yet, or, with a futility stop, no longer in an experiment whose candidate is
  not worth ``min_effect``.

Each verdict is an e-process crossing a threshold: a mixture over constant bets of
``prod(1 + lam Y_t)``, with ``Y_t`` a decision's weighted contrast less the margin, scaled so that
``Y_t >= -1``. Only the baseline's weight needs a bound, and the logging design supplies it
(Waudby-Smith et al. 2024, ACM/IMS Journal of Data Science 1(3), doi:10.1145/3643693). Across
zones, e-BH over e-values that freeze when their zone stops gathering keeps the false-deploy rate
at every read (Wang, Dandapanthula and Ramdas 2025, arXiv:2502.08539).

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
  On such a plant, run the gate in blocks (below), or estimate a plan's value with
  :func:`chc.evaluation.evaluate_plan`. Without blocks the gate tests for carryover of order up to
  ``GateConfig.carryover_lags``, 5 by default: the logged actions are randomised, so a reward that
  correlates with the score ``(candidate - baseline) / logged`` of a decision ``k`` back reads that
  decision. The check reads within each of the zone's modes, shadow, experiment and deployed, and
  not across a switch between them. It logs one warning per zone, ``gate_carryover``, naming the
  lag and the block design, and moves no verdict. Over 200 paths of 6000 decisions, an action that
  pays 0.4 now and costs 0.6 one, three or five decisions on was warned of on every path, at a
  median of 150 decisions, and deployed on every path as before; with no late cost, or none at
  all, or a reward common to both actions that persists as an AR(0.95), it warned on 0.040 to
  0.070 of them, and on 0.020 to 0.045 with a candidate that always takes the action and is
  deployed. A carryover past the lags read is not seen: at seven decisions on, 0.035. A
  candidate that takes one action for certain logs no randomised action once deployed, so the
  check reads only the decisions before: a cost of 0.3 one or three decisions on was warned of
  on 0.67 and 0.70 of 100 paths.
* **Rewards in** ``[0, 1]``.
* **A contrast that holds still.** A DEPLOY rejects the null that every decision's contrast, given
  the past, is at most ``delta``: it says the candidate beat the baseline over the decisions read,
  not that it still does. A candidate +0.06 better for its first 3000 decisions and -0.06 worse for
  the next 3000, even on average, was deployed in the first half on 200 of 200 paths, and the
  rollback e-process had caught the turn on 3 of them by the end.

**Plants with memory: the gate in blocks** (*experimental*). With ``GateConfig(block_length=L,
burn_in=b)``, a zone's evidence comes from an experiment on policies, not on decisions: each block
of ``L`` periods draws one policy, the candidate with probability ``rho``, runs it for the whole
block, and logs the probability of that draw. A :class:`BlockBatch` carries the blocks. The gate
drops the first ``b`` periods of each block, averages the reward over the last ``r = L - b``, and
weighs the block by the inverse of its logged probability. One block is one increment of the same
e-processes, and e-BH reads them as before. The inference is design-based: the only randomness it
uses is the gate's own draw, so the dependence between one zone's blocks does not matter (Bojinov
and Shephard 2019; Ham, Bojinov, Lindon and Tingley 2026). What a DEPLOY then claims is that the
candidate's mean reward over the read periods beats the baseline's by more than ``delta``, block by
block, given everything before the block.

* **Carryover of order** ``p <= b`` (a reward depends on the decisions of the last ``p`` periods
  only; Bojinov and Shephard's ``m``-period causal impact): the read periods see only their own
  block's policy, so the contrast is the sustained one, that of running the candidate rather than
  the baseline for good.
* **Geometric memory**: a plant whose expected reward ``t`` periods after a policy takes over lies
  within ``C a^t`` of that policy's long-run value ``V``, whatever the state it took over in. The
  block's contrast then lies within ``2 C a^b (1 - a^r) / (r (1 - a))``, of order ``a^b``, of
  ``V_new - V_base`` (``validation/gate_block_contrast.mac``), and a DEPLOY claims only
  ``V_new - V_base > delta`` less that bias: ``delta`` must cover it. On the lab's plant above
  (``a = 0.9``), with ``r = 10``, the block's contrast is +0.014 from a start at 0.2, the
  stationary mean under ``rho = 0.5``, and +0.062 from one at -1.1: the wrong sign. With
  ``b = 40`` it lies between -0.0115 and -0.0069 for every start in ``[-1.1, 1.5]``, against
  -0.0096 in the long run, and the gate in blocks deployed the candidate on 5 of 1000 paths of
  20 000 periods, where without blocks it deployed on every path.
* The gate cannot test ``p`` or ``a`` on its own data.
* A block is one increment however long it is, and its burned periods read nothing. Under
  carryover of order 1, a candidate +0.3 better sustained was deployed after a median of 562
  periods in blocks of 2 with 1 burned, 978 in blocks of 5 with 1 burned, and 1288 in blocks of 5
  with 4 burned.

A zone's channel is watched through per-decision drift e-values. :func:`channel_drift_evalues`
reads them off the Gaussian dither a :class:`DecisionLog` records, and :class:`DriftAlarm` turns
them into an alarm outside the gate as well. Once the channel has moved, :func:`channel_move`
re-reads it off the same dither, and :meth:`ChannelMove.price` puts the move in a plan's own cost.

A deployed model's one-step innovations are watched too (*experimental*): :class:`MissRateAlarm`
alarms when they leave a split-conformal radius more often than its calibration allows, and
:class:`OnlineRadius` moves a radius per horizon so that each horizon's long-run miss rate stays
near its target on any sequence.
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

from chc.frames import _real_numbers
from chc.plan import DecisionWeight
from chc.uncertainty import _as_written, _conformal_rank

_log = logging.getLogger(__name__)

Verdict = Literal["deploy", "shadow", "experiment", "hold", "rollback"]
GateMode = Literal["shadow", "experiment", "deployed", "retired"]

_Array = NDArray[np.float64]

_BETS = 0.9 * 2.0 ** -np.arange(12)  # constant bets, mixed with equal weight
_LOG_BETS = math.log(_BETS.size)
_MATCH = 1e-6  # relative gap allowed between a logged propensity and the mode's
_DRIFT_BETS = 2.0 ** -np.arange(8)  # |theta|, in residual scales per dither scale
_DITHER_LEVEL = 1e-9  # a dither this far from its stated scale is a slip, not a draw
_CARRYOVER_FIRST = 100  # decisions before the carryover check first reads, then at every doubling
_CARRYOVER_LEVEL = 0.05  # a zone's chance of a false carryover warning over its whole run


def _vector(value: ArrayLike, name: str) -> _Array:
    array = np.array(_real_numbers(value, name), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a vector, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} is not finite")
    array.setflags(write=False)
    return array


def _evalues(value: ArrayLike, name: str, size: int | None = None) -> _Array:
    """E-values as one row per decision and one column per detector."""
    evalues = np.array(_real_numbers(value, name), dtype=np.float64)
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
    ``alpha`` bounds the expected share of wrong DEPLOY decisions across zones, a zone rolled back
    since its DEPLOY still counted, ``alpha_harm`` each HOLD for harm and each ROLLBACK, and
    ``drift_arl`` is the average number of decisions between false drift alarms on an unchanged
    channel. ``rho`` is the candidate's share of the logging mixture in EXPERIMENT.

    ``min_effect``, the smallest improvement beyond ``delta`` worth deploying, and ``horizon``, the
    decisions per zone the gate may spend, decide when to switch to EXPERIMENT, and no guarantee
    depends on them.

    ``alpha_futility`` (*experimental*), unset by default, adds a futility stop: an experiment
    whose evidence says the candidate does not beat the baseline by ``delta + min_effect`` returns
    to shadow, and the zone no longer switches to EXPERIMENT until its channel drifts. Each stop is
    wrong, stopping a candidate that does beat the baseline by more than ``delta + min_effect``,
    with probability at most ``alpha_futility``.

    ``block_length`` (*experimental*), unset by default, runs the gate in blocks of that many
    periods, for a plant with memory (see :mod:`chc.gate`), and ``burn_in``, the periods at the
    start of each block that are not read, must be at least the plant's carryover order. In blocks,
    ``rho`` is the probability that a block runs the candidate, and ``horizon`` and the zones'
    :class:`ZonePlan` are not read.

    ``carryover_lags``, 5 by default, is how many lags the carryover check reads when the gate is
    not in blocks (see :mod:`chc.gate`); 0 turns it off. The check logs a warning and changes no
    verdict.

    Raises:
        ValueError: on a margin outside ``[0, 1)``, a level or ``rho`` outside ``(0, 1)``, a
            ``min_effect`` that is not positive, a ``horizon`` below 1, a ``drift_arl`` not above
            1, an ``alpha_futility`` with ``delta + min_effect`` not below 1, which no reward
            in ``[0, 1]`` can beat, a ``block_length`` below 1, a ``burn_in`` that is negative,
            leaves no period of a block to read, or is set without a ``block_length``, or a
            negative ``carryover_lags``.
        TypeError: on a ``horizon``, ``block_length``, ``burn_in`` or ``carryover_lags`` that is
            not a whole number: a float, or a bool.
    """

    delta: float
    delta_harm: float
    min_effect: float
    horizon: int
    alpha: float = 0.05
    alpha_harm: float = 0.05
    drift_arl: float = 10_000.0
    rho: float = 0.5
    alpha_futility: float | None = None
    block_length: int | None = None
    burn_in: int = 0
    carryover_lags: int = 5

    def __post_init__(self) -> None:
        for name in ("horizon", "burn_in", "carryover_lags") + (
            ("block_length",) if self.block_length is not None else ()
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | np.integer):
                raise TypeError(f"{name} must be a whole number, got {value!r}")
        for name in ("delta", "delta_harm"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1), got {value}")
        levels = ("alpha", "alpha_harm", "rho") + (
            ("alpha_futility",) if self.alpha_futility is not None else ()
        )
        for name in levels:
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1), got {value}")
        if not (math.isfinite(self.min_effect) and self.min_effect > 0.0):
            raise ValueError(f"min_effect must be positive, got {self.min_effect}")
        if self.alpha_futility is not None and not self.delta + self.min_effect < 1.0:
            raise ValueError(
                f"a futility stop tests whether the candidate beats the baseline by delta +"
                f" min_effect = {self.delta + self.min_effect:g}, which no reward in [0, 1] can"
            )
        if self.horizon < 1:
            raise ValueError(f"horizon must be at least 1 decision, got {self.horizon}")
        if not self.drift_arl > 1.0:
            raise ValueError(f"drift_arl must exceed 1, got {self.drift_arl}")
        if self.carryover_lags < 0:
            raise ValueError(f"carryover_lags must be at least 0, got {self.carryover_lags}")
        if self.block_length is None:
            if self.burn_in != 0:
                raise ValueError(
                    f"burn_in = {self.burn_in} without a block_length: a burn-in is the start of a"
                    " block"
                )
        elif self.block_length < 1:
            raise ValueError(f"block_length must be at least 1 period, got {self.block_length}")
        elif not 0 <= self.burn_in < self.block_length:
            raise ValueError(
                f"burn_in must lie in [0, block_length) = [0, {self.block_length}), so that a"
                f" block has a period to read, got {self.burn_in}"
            )


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


def _record_numbers(records: list[Mapping[str, object]], key: str) -> _Array:
    """Field ``key`` of every record as float64, where it holds real numbers: a record's row is its
    position among ``records``."""
    return np.array(_real_numbers(_column(records, key), key), dtype=np.float64)


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
        action = np.array(_real_numbers(self.action, "action"), dtype=np.float64)
        if action.ndim not in (1, 2) or action.shape[0] != size:
            raise ValueError(
                f"action must have shape ({size},) or ({size}, actions), got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("action is not finite")
        saturated = np.array(_real_numbers(self.saturated, "saturated"))
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
            dither = np.array(_real_numbers(self.dither, "dither"), dtype=np.float64)
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
            action=_record_numbers(rows, "action"),
            propensity=_record_numbers(rows, "propensity"),
            saturated=np.array(saturated, dtype=np.bool_),
            dither=_record_numbers(rows, "dither") if with_dither else None,
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
        with np.errstate(over="ignore"):
            weights = np.concatenate([self.candidate, self.baseline]) / np.tile(self.logged, 2)
        if not np.isfinite(weights).all():
            # an infinite weight times a reward of 0 is nan: the e-value keeps it until its epoch
            # restarts, and e-BH sorts a nan first, which lowers the bar the other zones meet
            raise ValueError(
                f"a logged propensity of {np.min(self.logged)} makes an importance weight overflow"
            )
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


@dataclass(frozen=True)
class BlockBatch:
    """One zone's blocks since the gate was last read, for a gate in blocks. *Experimental.*

    ``reward`` holds one row per block and one column per period of it, burn-in included.
    ``candidate`` says whether the block ran the candidate, and ``logged`` is the probability with
    which the block's draw chose the policy it ran, recorded at the draw: ``rho`` or ``1 - rho`` in
    an experiment, and 1 when the zone's mode left no choice. ``drift``, optional, is a
    :class:`ZoneBatch`'s, with one row per period, block by block.

    Raises:
        ValueError: on rewards that are not one row per block of at least one period, a non-finite
            entry, a reward outside ``[0, 1]``, a ``candidate`` that is not one boolean per block,
            a ``logged`` that is not one probability in ``(0, 1]`` per block, or a negative
            e-value.
    """

    reward: NDArray[np.float64]
    candidate: NDArray[np.bool_]
    logged: NDArray[np.float64]
    drift: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        reward = np.array(_real_numbers(self.reward, "reward"), dtype=np.float64)
        if reward.ndim != 2 or reward.shape[1] == 0:
            raise ValueError(
                f"reward must have shape (blocks, periods), at least one period, got {reward.shape}"
            )
        if not np.all(np.isfinite(reward)):
            raise ValueError("reward is not finite")
        if reward.size and not (np.min(reward) >= 0.0 and np.max(reward) <= 1.0):
            raise ValueError(
                f"rewards must lie in [0, 1], got [{np.min(reward)}, {np.max(reward)}]"
            )
        blocks = reward.shape[0]
        candidate = np.array(_real_numbers(self.candidate, "candidate"))
        if candidate.size == 0:
            candidate = candidate.astype(np.bool_)
        if candidate.dtype != np.bool_ or candidate.shape != (blocks,):
            raise ValueError(
                f"candidate must hold {blocks} booleans, got {candidate.dtype} of shape"
                f" {candidate.shape}"
            )
        logged = _vector(self.logged, "logged")
        if logged.shape != (blocks,):
            raise ValueError(f"logged has {logged.shape[0]} entries for {blocks} blocks")
        if blocks and not (np.min(logged) > 0.0 and np.max(logged) <= 1.0):
            raise ValueError(
                f"a logged probability must lie in (0, 1], got [{np.min(logged)}, {np.max(logged)}]"
            )
        for name, array in (("reward", reward), ("candidate", candidate), ("logged", logged)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.drift is not None:
            object.__setattr__(self, "drift", _evalues(self.drift, "drift", reward.size))


def _per_block(batch: BlockBatch, burn_in: int) -> ZoneBatch:
    """One decision per block: the policy drawn is the action, so the candidate's propensity is 1
    on its own blocks and 0 on the baseline's, and the reward is the read periods' mean."""
    ran = batch.candidate.astype(np.float64)
    return ZoneBatch(batch.reward[:, burn_in:].mean(axis=1), ran, 1.0 - ran, batch.logged)


def _entries(value: ArrayLike, shape: tuple[int, ...], name: str) -> _Array:
    array = np.array(_real_numbers(value, name), dtype=np.float64)
    if array.ndim and array.shape != shape:
        raise ValueError(f"{name} must be a scalar or of shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} is not finite")
    return np.broadcast_to(array, shape)


def _residual_rows(residual: ArrayLike, size: int) -> _Array:
    """The residual as one row per decision and one column per state."""
    rows = np.array(_real_numbers(residual, "residual"), dtype=np.float64)
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


def _state_rows(residuals: ArrayLike, name: str, states: int | None) -> _Array:
    """Residuals as one row per step and one column per state."""
    rows = np.array(_real_numbers(residuals, name), dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[:, None]
    if rows.ndim != 2 or (states is not None and rows.shape[1] != states):
        raise ValueError(
            f"{name} must have shape (steps,) or (steps, {states or 'states'}), got {rows.shape}"
        )
    if not np.all(np.isfinite(rows)):
        raise ValueError(f"{name} is not finite")
    return rows


def _log_miss_mixture(misses: NDArray[np.int64], steps: NDArray[np.int64], bound: float) -> _Array:
    """The log of ``int_bound^1 (p / bound)^S ((1 - p) / (1 - bound))^F dp / (1 - bound)``, the
    likelihood ratios of a miss rate ``p`` against ``bound`` mixed over ``p`` uniform on
    ``[bound, 1]``, after ``S`` misses in ``S + F`` steps: ``B(S + 1, F + 1)`` times the upper
    regularised incomplete beta at ``bound``, over ``bound^S (1 - bound)^(F + 1)``."""
    s = misses.astype(np.float64)
    f = (steps - misses).astype(np.float64)
    with np.errstate(divide="ignore"):  # a tail below the smallest float: the e-value is 0
        tail = np.log(special.betaincc(s + 1.0, f + 1.0, bound))
    return (
        special.betaln(s + 1.0, f + 1.0)
        + tail
        - s * math.log(bound)
        - (f + 1.0) * math.log1p(-bound)
    )


class MissRateAlarm:
    """An alarm that sounds once a deployed model's one-step innovations leave a split-conformal
    radius more often than its calibration allows: on a plant the model still describes, it ever
    sounds with probability at most ``delta``.

    ``calibration`` holds one-step residuals, the next state less the model's prediction, of
    consecutive steps held out from the fit, one column per state. Each residual is whitened against
    the one before, ``w_t = r_t - persistence r_(t-1)``, and scored by its Euclidean norm, in the
    states' units: scale them first where those differ. ``radius`` is the ``k``-th smallest of the
    ``n`` scores, ``k = ceil((n + 1)(1 - alpha))``, split conformal's radius. Given the calibration,
    the share of later innovations outside it is a fixed ``p``, and over calibrations ``p`` follows
    ``Beta(n + 1 - k, k)`` (Vovk 2012; Lin and Bansal 2024): ``bound``, its upper ``delta / 2``
    quantile, exceeds ``p`` with probability at least ``1 - delta / 2``.

    :meth:`update` whitens each deployed residual against the one before it and counts the
    innovations outside ``radius``. After ``S`` misses in ``t`` innovations, ``statistic`` is the
    likelihood ratio of a miss rate ``q`` against ``bound``, mixed over ``q`` uniform on
    ``[bound, 1]``, in closed form through the incomplete beta function. While each innovation
    misses with probability at most ``bound`` given the past, it is a nonnegative supermartingale
    from 1, so by Ville's inequality it ever reaches ``2 / delta`` with probability at most
    ``delta / 2``, and the alarm sounds there. The two halves add up to ``delta``: Podkopaev and
    Ramdas's test for a harmful shift (ICLR 2022, Proposition 1), with the source risk's bound read
    off the calibration's exact law and no tolerance. After an alarm the counts start again at the
    next call, and the call's later residuals are dropped.

    On a plant whose model is exact and whose noise is AR(1) at 0.8, with 300 calibration
    innovations, ``alpha = delta = 0.1`` and 1000 paths of 1000 steps read every 100, the alarm
    sounded on 0.016 of the paths at the noise's persistence, on 0.020 at the lag-1 correlation
    fitted on each calibration log, and on 0.209 unwhitened. With the noise's scale 1.5 times the
    calibration's, it sounded on every one of 200 paths.

    What the guarantee needs, and the alarm cannot check:

    * **Innovations independent of the past and exchangeable with the calibration's**: the model
      right, so a residual is the plant's noise, and the noise AR(1) at rate ``persistence``,
      such as :attr:`chc.decision.DecisionCertificate.tube_persistence` or a fit's lag-1
      correlation. Without whitening, misses cluster: a miss makes the next one likelier, and the
      supermartingale fails.
    * **A persistence fixed before the deployed data**; one estimated on the calibration log
      leaves a correlation of the order of its error.

    Raises:
        ValueError: on ``alpha`` or ``delta`` not in ``(0, 1)``, a persistence not in ``(-1, 1)``,
            residuals that are not finite or not one row per step, and fewer calibration
            innovations than ``(1 - alpha) / alpha``, below which no radius carries the coverage.
    """

    def __init__(
        self, calibration: ArrayLike, *, persistence: float, alpha: float, delta: float
    ) -> None:
        for name, level in (("alpha", alpha), ("delta", delta)):
            if not 0.0 < level < 1.0:
                raise ValueError(f"{name} must lie in (0, 1), got {level}")
        if not -1.0 < persistence < 1.0:
            raise ValueError(f"persistence must lie in (-1, 1), got {persistence}")
        rows = _state_rows(calibration, "calibration", None)
        self.persistence = float(persistence)
        self.delta = float(delta)
        scores = np.linalg.norm(rows[1:] - self.persistence * rows[:-1], axis=1)
        n = scores.size
        k = _conformal_rank(n, alpha)
        if k > n:
            least = math.ceil((1 - _as_written(alpha)) / _as_written(alpha))
            raise ValueError(
                f"{n} calibration innovations cannot give a radius covering {1.0 - alpha:.6g}: it"
                f" is the {k}-th smallest score, so it needs at least {least} of them, that is"
                f" {least + 1} residuals"
            )
        self.radius = float(np.sort(scores)[k - 1])
        self.bound = float(special.betaincinv(n + 1 - k, k, 1.0 - self.delta / 2.0))
        self._states = rows.shape[1]
        self._last: _Array | None = None
        self._misses = 0
        self._steps = 0

    @property
    def misses(self) -> int:
        """Innovations outside ``radius`` since the start or the last alarm."""
        return self._misses

    @property
    def steps(self) -> int:
        """Innovations read since the start or the last alarm."""
        return self._steps

    @property
    def statistic(self) -> float:
        """The mixture's e-value: 1 before any innovation and after an alarm."""
        log_e = _log_miss_mixture(np.array([self._misses]), np.array([self._steps]), self.bound)
        return float(np.exp(log_e[0]))

    def update(self, residuals: ArrayLike) -> bool:
        """Add one-step residuals, one row per consecutive step, and say whether the alarm
        sounded. The first residual after the start or an alarm only whitens the next.

        Raises:
            ValueError: on residuals that are not finite, or not one column per calibration state.
        """
        rows = _state_rows(residuals, "residuals", self._states)
        if rows.shape[0] == 0:
            return False
        lagged = rows if self._last is None else np.vstack([self._last, rows])
        innovations = lagged[1:] - self.persistence * lagged[:-1]
        missed = np.linalg.norm(innovations, axis=1) > self.radius
        misses = self._misses + np.cumsum(missed)
        steps = self._steps + np.arange(1, missed.size + 1)
        crossed = np.flatnonzero(
            _log_miss_mixture(misses, steps, self.bound) >= math.log(2.0 / self.delta)
        )
        if crossed.size == 0:
            self._last = rows[-1]
            if missed.size:
                self._misses, self._steps = int(misses[-1]), int(steps[-1])
            return False
        at = int(crossed[0])
        self._last, self._misses, self._steps = None, 0, 0
        _log.warning(
            "miss-rate alarm: %d of %d innovations outside the radius %.4g, where at most a share"
            " %.4g may miss; %d later residuals were dropped",
            misses[at],
            steps[at],
            self.radius,
            self.bound,
            missed.size - at - 1,
            extra={
                "chc_event": "miss_rate_alarm",
                "misses": int(misses[at]),
                "steps": int(steps[at]),
                "radius": self.radius,
                "bound": self.bound,
            },
        )
        return True


class OnlineRadius:
    """A radius per horizon, updated online so that, over any sequence of scores, each horizon's
    long-run miss rate comes within a stated bound of ``alpha`` (Angelopoulos, Barber and Bates,
    ICML 2024).

    Horizon ``h`` (``1`` to ``H``, one per entry of ``initial``) has a radius. Each period issues
    ``radius``, one per horizon, and closes with :meth:`update`, which takes one score per
    horizon: for horizon ``h``, the score of the prediction issued ``h - 1`` periods before this
    one, ``nan`` where there is none. A score is any number in ``[0, bound]`` the caller computes,
    such as the error's norm over the tube's scale, and the prediction misses when its score
    exceeds the radius it was issued with. Each scored horizon's radius then moves by
    ``eta_n (miss - alpha)``, with ``eta_n = step n^(-decay)`` at its ``n``-th score: up after a
    miss, down otherwise.

    On any sequence of scores in ``[0, bound]``, chosen even by an adversary who sees the radii,
    such as a plan steering its own states, the miss rate of horizon ``h`` after ``n`` scores
    lies within ``miss_bound = (bound + h step) / (eta_n n)`` of ``alpha``, which shrinks as
    ``n^(decay - 1)``. For ``h = 1`` it is their Theorem 1. For ``h > 1``, up to ``h - 1`` updates
    are in flight while a prediction waits for its score, which widens their Lemma 1's range of the
    radius from ``bound + eta_1`` to ``bound + h eta_1``; the proof is otherwise theirs. On every
    one of the ``2^12`` sequences of misses an adversary can force in 12 periods, at a constant
    step of 1, horizon 3 came within 0.9 of that bound and past the one without ``h``. The bound
    holds for each horizon on its own, and on no other sequence than the one scored; it says
    nothing of coverage at any one period. Their step is ``decay = 0.6``.

    Raises:
        ValueError: on an empty or non-finite ``initial`` or one outside ``[0, bound]``, ``alpha``
            not in ``(0, 1)``, a ``bound`` or ``step`` that is not positive and finite, or a
            ``decay`` not in ``[0, 1)``.
    """

    def __init__(
        self,
        initial: ArrayLike,
        *,
        alpha: float,
        bound: float,
        step: float,
        decay: float = 0.6,
    ) -> None:
        radius = np.array(_vector(initial, "initial"))
        if radius.size == 0:
            raise ValueError("initial must hold one radius per horizon, got none")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        for name, value in (("bound", bound), ("step", step)):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {value}")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must lie in [0, 1), got {decay}")
        if not np.all((radius >= 0.0) & (radius <= bound)):
            raise ValueError(f"initial radii must lie in [0, bound = {bound}], got {radius}")
        self.alpha, self.bound, self.step, self.decay = (
            float(alpha),
            float(bound),
            float(step),
            float(decay),
        )
        self._radius = radius
        self._issued: list[_Array] = []
        self._periods = 0
        self._scored = np.zeros(radius.size, dtype=np.int64)
        self._misses = np.zeros(radius.size, dtype=np.int64)

    @property
    def radius(self) -> _Array:
        """The radius each horizon issues at the next period."""
        radius = self._radius.copy()
        radius.setflags(write=False)
        return radius

    @property
    def scored(self) -> NDArray[np.int64]:
        """How many predictions each horizon has scored."""
        return self._scored.copy()

    @property
    def miss_rate(self) -> _Array:
        """Each horizon's share of scored predictions that missed, ``nan`` before its first."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return self._misses / self._scored

    @property
    def miss_bound(self) -> _Array:
        """``(bound + h step) / (eta_n n)``: how far each horizon's miss rate may lie from
        ``alpha``, ``inf`` before its first score."""
        horizon = np.arange(1, self._radius.size + 1)
        n = self._scored.astype(np.float64)
        with np.errstate(divide="ignore"):
            return (self.bound + horizon * self.step) / (self.step * n ** (1.0 - self.decay))

    def update(self, scores: ArrayLike) -> NDArray[np.bool_]:
        """Close a period: score each horizon's prediction that resolved in it, ``nan`` for
        none, against the radius it was issued with, and move the scored horizons' radii. Returns
        which horizons missed.

        Raises:
            ValueError: on scores that are not one per horizon, a score outside ``[0, bound]``, or
                a score for horizon ``h`` before ``h`` periods have closed, when no prediction for
                it has been issued.
        """
        given = np.array(_real_numbers(scores, "scores"), dtype=np.float64)
        horizons = self._radius.size
        if given.shape != (horizons,):
            raise ValueError(
                f"scores must hold one entry per horizon, shape ({horizons},), got {given.shape}"
            )
        scored = ~np.isnan(given)
        if not np.all((given[scored] >= 0.0) & (given[scored] <= self.bound)):
            raise ValueError(f"scores must lie in [0, bound = {self.bound}], got {given}")
        early = np.flatnonzero(scored & (np.arange(1, horizons + 1) > self._periods + 1))
        if early.size:
            raise ValueError(
                f"horizon {early[0] + 1} scored at period {self._periods + 1}, before any"
                " prediction for it was issued"
            )
        self._periods += 1
        self._issued = [*self._issued, self._radius.copy()][-horizons:]
        missed = np.zeros(horizons, dtype=bool)
        for h in np.flatnonzero(scored):
            issued = self._issued[-(h + 1)][h]
            missed[h] = given[h] > issued
            self._scored[h] += 1
            self._misses[h] += missed[h]
            eta = self.step * float(self._scored[h]) ** -self.decay
            self._radius[h] += eta * (float(missed[h]) - self.alpha)
        return missed


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


class _Carryover:
    """A score test of a reward on the actions ``1 .. lags`` decisions back.

    A decision's score ``s = (candidate - baseline) / logged`` is fixed by the logged law of its
    mode: given the past, it does not depend on any later reward unless that reward reads the
    action. Where no reward depends on an earlier action, the score of decision ``t - k`` is
    uncorrelated with the reward at ``t``. The test reads that covariance within each mode, with
    the score and the reward centred on the mode's own means, and drops a pair whose two decisions
    were logged in different modes: a mode's scores need not have mean zero (a deterministic
    candidate, once deployed, scores ``1 - baseline`` on every decision), and its rewards have their
    own mean, so a covariance pooled across modes reads the switch as carryover. Each lag's sum of
    ``(s_(t-k) - s_bar) (r_t - r_bar)`` over the modes, over the square root of the summed squares,
    is near a standard normal. The test reads after ``_CARRYOVER_FIRST`` pairs and at every
    doubling after, the ``m``-th read at level ``_CARRYOVER_LEVEL / (m (m + 1))`` split over the
    lags, so a zone's false warnings over its whole run stay within ``_CARRYOVER_LEVEL`` as far as
    the normal law holds. It warns once."""

    _MODES: ClassVar[tuple[GateMode, ...]] = ("shadow", "experiment", "deployed")

    def __init__(self, lags: int) -> None:
        self.lags = lags
        self.scores = np.zeros(0)  # the last ``lags`` scores, oldest first
        self.modes = np.zeros(0, dtype=np.int64)  # and the modes they were logged in
        # per mode and lag, sum s^i r^j at [i, j], s and r taken from the mode's first decision so
        # that a constant score sums to zero exactly
        self.sums = np.zeros((len(self._MODES), 3, 3, lags))
        self.origin = np.full((len(self._MODES), 2), np.nan)
        self.next_read = _CARRYOVER_FIRST
        self.reads = 0
        self.warned = False

    def add(
        self, scores: _Array, rewards: _Array, mode: GateMode
    ) -> tuple[int, float, float] | None:
        """Add a batch logged in ``mode``, and return the lag, the statistic and the p-value of the
        most significant lag where this read rejects; ``None`` otherwise."""
        code = self._MODES.index(mode)
        if rewards.size and np.isnan(self.origin[code, 0]):
            self.origin[code] = scores[0], rewards[0]
        joined = np.concatenate([self.scores, scores])
        modes = np.concatenate([self.modes, np.full(scores.size, code)])
        held = self.scores.size
        powers = np.arange(3)
        for k in range(1, self.lags + 1):
            first = max(0, k - held)
            if first >= rewards.size:
                continue
            ends = slice(held + first - k, held + rewards.size - k)
            same = modes[ends] == code
            s = joined[ends][same] - self.origin[code, 0]
            r = rewards[first:][same] - self.origin[code, 1]
            self.sums[code, :, :, k - 1] += np.power.outer(s, powers).T @ np.power.outer(r, powers)
        self.scores = joined[-self.lags :]
        self.modes = modes[-self.lags :]
        pairs = self.sums[:, 0, 0, 0].sum()
        if self.warned or pairs < self.next_read:
            return None
        while self.next_read <= pairs:
            self.next_read *= 2
        self.reads += 1
        m = self.sums
        count = m[:, 0, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            s_bar = np.where(count > 0, m[:, 1, 0] / count, 0.0)
            r_bar = np.where(count > 0, m[:, 0, 1] / count, 0.0)
        covariance = (m[:, 1, 1] - count * s_bar * r_bar).sum(axis=0)
        # sum (s - s_bar)^2 (r - r_bar)^2, expanded in the raw sums
        weights = np.array([1.0, -2.0, 1.0])
        s_terms = weights[:, None, None] * s_bar[None] ** np.array([0, 1, 2])[:, None, None]
        r_terms = weights[:, None, None] * r_bar[None] ** np.array([0, 1, 2])[:, None, None]
        variance = np.zeros(self.lags)
        for i in range(3):
            for j in range(3):
                variance += (s_terms[i] * r_terms[j] * m[:, 2 - i, 2 - j]).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(variance > 0.0, covariance / np.sqrt(variance), 0.0)
        p = 2.0 * special.ndtr(-np.abs(z))
        lag = int(np.argmin(p))
        if p[lag] >= _CARRYOVER_LEVEL / (self.reads * (self.reads + 1) * self.lags):
            return None
        self.warned = True
        return lag + 1, float(z[lag]), float(p[lag])


@dataclass
class _Zone:
    plan: ZonePlan
    drift: DriftAlarm
    mode: GateMode = "shadow"
    improvement: _Evidence = field(default_factory=_Evidence)
    harm: _Evidence = field(default_factory=_Evidence)
    rollback: _Evidence = field(default_factory=_Evidence)
    futility: _Evidence = field(default_factory=_Evidence)
    futile: bool = False  # the futility e-value crossed in this epoch; a crossing is a rejection
    decisions: int = 0
    carryover: _Carryover | None = None


class DeploymentGate:
    """Shadow evidence per zone that ends in DEPLOY, HOLD, EXPERIMENT or ROLLBACK, read any time.

    Whenever the gate is read -- after every batch, or at a time chosen by looking at the data --
    and under the conditions in :mod:`chc.gate`:

    * the expected share of wrong DEPLOY decisions among those made so far is at most ``alpha``,
      under any dependence across zones; a DEPLOY is wrong when the candidate does not beat the
      baseline by more than ``delta`` over the decisions read, which says nothing of a contrast that
      turns after them. A zone deployed and later rolled back still counts among the DEPLOYs,
      its evidence frozen at the DEPLOY: the share among the zones still deployed at a read is not
      what is bounded;
    * each HOLD for harm, and each ROLLBACK on evidence, is wrong with probability at most
      ``alpha_harm``;
    * with drift e-values supplied, an unchanged channel alarms at most once per ``drift_arl``
      decisions on average;
    * with ``alpha_futility``, each futility stop is wrong with probability at most
      ``alpha_futility``; a stop is wrong when the candidate beats the baseline by more than
      ``delta + min_effect``.

    The rules, in order, for each zone at each read:

    1. a drift alarm: HOLD, and the zone's evidence opens a new epoch; ROLLBACK if deployed;
    2. selected by e-BH over every zone's e-value: DEPLOY;
    3. the harm e-value reaches ``1 / alpha_harm``: HOLD, and the zone retires;
    4. with ``alpha_futility``, the futility e-value has reached ``1 / alpha_futility`` in this
       epoch: SHADOW, back from an experiment, and the zone does not switch to EXPERIMENT again;
    5. in shadow, a verdict at ``min_effect`` would take longer than the ``horizon`` has left, and
       the mixture would gather evidence faster: EXPERIMENT;
    6. otherwise the zone stays where it is, in shadow or in the experiment.

    A deployed zone is watched by a rollback e-process on the baseline's weight. A retired zone
    reads HOLD from then on; a new candidate is a new hypothesis, for a new gate. :meth:`mode` says
    how to log a zone's next batch: under the baseline in ``"shadow"``, under
    ``(1 - rho) baseline + rho candidate`` in ``"experiment"``, and under the candidate once
    ``"deployed"``. The gate keeps its evidence in memory, and one caller updates it at a time.

    In blocks (``GateConfig.block_length``), each zone is fed a :class:`BlockBatch` and starts in
    ``"experiment"``, where each block runs the candidate with probability ``rho``. ``"shadow"``,
    after a futility stop, runs the baseline's blocks, and ``"deployed"`` the candidate's. Shadow
    blocks are no evidence for the candidate, so rule 5 sends a shadow zone back to the experiment
    as soon as it may enter one, and a deployed zone's rollback evidence has no baseline block to
    read: only a drift alarm rolls it back.

    Raises:
        ValueError: on no zones.
    """

    def __init__(self, plans: Mapping[str, ZonePlan], config: GateConfig) -> None:
        if not plans:
            raise ValueError("the gate needs at least one zone")
        self.config = config
        start: GateMode = "shadow" if config.block_length is None else "experiment"
        lags = config.carryover_lags if config.block_length is None else 0
        self._zones = {
            name: _Zone(
                plan,
                DriftAlarm(config.drift_arl),
                start,
                carryover=_Carryover(lags) if lags else None,
            )
            for name, plan in plans.items()
        }

    def mode(self, zone: str) -> GateMode:
        return self._zones[zone].mode

    def update(self, batches: Mapping[str, ZoneBatch | BlockBatch]) -> dict[str, Verdict]:
        """Add each zone's batch to its evidence and return every zone's verdict.

        A zone without a batch keeps its evidence and still gets a verdict, since e-BH reads every
        zone's; a retired zone's batch is ignored. Every batch is checked before any is added. A
        gate in blocks reads each block as one decision, whose action is the policy drawn.

        Raises:
            KeyError: on a zone the gate has no plan for.
            TypeError: on a :class:`BlockBatch` for a gate not in blocks, or a :class:`ZoneBatch`
                for one that is.
            ValueError: on a batch whose logged propensities are not the mode's, blocks whose
                length is not ``block_length``, or drift e-values that change their number of
                detectors mid-run.
        """
        cfg = self.config
        unknown = sorted(set(batches) - set(self._zones))
        if unknown:
            raise KeyError(f"the gate has no plan for zones {unknown}")
        given = {z: b for z, b in batches.items() if self._zones[z].mode != "retired"}
        live = {z: self._decisions(z, b) for z, b in given.items()}
        for z, b in live.items():
            self._check(z, b, given[z].drift)
        alarms = {
            z: b.drift is not None and self._zones[z].drift.update(b.drift)
            for z, b in given.items()
        }
        for z, b in live.items():
            zone = self._zones[z]
            zone.decisions += given[z].reward.size
            w_new, w_base, r = b.candidate / b.logged, b.baseline / b.logged, b.reward
            if zone.carryover is not None:
                self._check_carryover(z, zone, zone.carryover.add(w_new - w_base, r, zone.mode))
            if zone.mode == "deployed":
                zone.rollback.add(((w_base - w_new) * r - cfg.delta_harm) / (1 + cfg.delta_harm))
                continue
            bound = 1.0 if zone.mode == "shadow" else 1.0 / (1.0 - cfg.rho)
            diff = w_new - w_base
            zone.improvement.add((diff * r - cfg.delta) / (bound + cfg.delta))
            zone.harm.add((diff * (1 - r) - cfg.delta_harm) / (bound + cfg.delta_harm))
            if cfg.alpha_futility is not None:
                # The harm increment at the margin -(delta + min_effect): E[diff (1 - r)] is
                # V_base - V_new, so this bets against V_new - V_base >= delta + min_effect.
                worth = cfg.delta + cfg.min_effect
                zone.futility.add((diff * (1 - r) + worth) / (bound - worth))
            if alarms[z]:
                zone.improvement.restart()
                zone.harm.restart()
                zone.futility.restart()
                zone.futile = False

        # A zone's improvement evidence stops at DEPLOY or retirement, so e-BH reads stopped
        # e-values, and the deployed set is self-consistent at every read.
        log_e = np.array([zone.improvement.log_value for zone in self._zones.values()])
        selected = dict(zip(self._zones, _ebh(log_e, cfg.alpha), strict=True))
        verdicts: dict[str, Verdict] = {}
        for z, zone in self._zones.items():
            before, alarm, futile = zone.mode, alarms.get(z, False), zone.futile
            verdicts[z] = self._rule(zone, alarm, bool(selected[z]))
            if zone.mode != before or alarm or zone.futile != futile:
                _log.info(
                    "gate verdict",
                    extra={
                        "chc_event": "deployment_gate",
                        "zone": z,
                        "verdict": verdicts[z],
                        "mode": zone.mode,
                        "drift_alarm": alarm,
                        "futile": zone.futile,
                        "log_evidence": zone.improvement.log_value,
                        "log_harm": zone.harm.log_value,
                        "log_futility": zone.futility.log_value,
                        "log_rollback": zone.rollback.log_value,
                        "epoch": zone.improvement.epoch,
                        "decisions": zone.decisions,
                    },
                )
        return verdicts

    def _check_carryover(
        self, name: str, zone: _Zone, found: tuple[int, float, float] | None
    ) -> None:
        if found is None:
            return
        lag, statistic, p_value = found
        _log.warning(
            "zone %r: the reward reads the action %d decisions back (z = %.2f, p = %.2g), so the"
            " gate's per-decision evidence certifies the myopic contrast, not that of running the"
            " candidate for good; run the gate in blocks, GateConfig(block_length=L, burn_in=b),"
            " with b at least the carryover's order",
            name,
            lag,
            statistic,
            p_value,
            extra={
                "chc_event": "gate_carryover",
                "zone": name,
                "lag": lag,
                "statistic": statistic,
                "p_value": p_value,
                "decisions": zone.decisions,
            },
        )

    def _decisions(self, name: str, batch: ZoneBatch | BlockBatch) -> ZoneBatch:
        """The batch as decisions: as given, or, in blocks, one per block."""
        length = self.config.block_length
        if length is None:
            if not isinstance(batch, ZoneBatch):
                raise TypeError(
                    f"zone {name!r}: a gate not in blocks reads a ZoneBatch, got"
                    f" {type(batch).__name__}"
                )
            return batch
        if not isinstance(batch, BlockBatch):
            raise TypeError(
                f"zone {name!r}: a gate in blocks of {length} periods reads a BlockBatch, got"
                f" {type(batch).__name__}"
            )
        if batch.reward.shape[1] != length:
            raise ValueError(
                f"zone {name!r}: blocks of {batch.reward.shape[1]} periods, where the gate's are"
                f" {length}"
            )
        return _per_block(batch, self.config.burn_in)

    def _check(self, name: str, batch: ZoneBatch, drift: _Array | None) -> None:
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
        if drift is not None and detectors is not None and drift.shape[1] != detectors:
            raise ValueError(
                f"zone {name!r}: {drift.shape[1]} drift detectors, where the running"
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
        if self._futile(zone):
            zone.mode = "shadow"
            return "shadow"
        if zone.mode == "shadow" and (
            self.config.block_length is not None or self._experiment_is_faster(zone)
        ):
            zone.mode = "experiment"
            return "experiment"
        return "shadow" if zone.mode == "shadow" else "experiment"

    def _futile(self, zone: _Zone) -> bool:
        """The futility e-value has reached ``1 / alpha_futility`` in this epoch. The flag stays up
        after the e-value falls back: a crossing rejects, and an experiment that re-entered on the
        way down would stop again."""
        level = self.config.alpha_futility
        if level is not None and zone.futility.log_value >= math.log(1.0 / level):
            zone.futile = True
        return zone.futile

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
