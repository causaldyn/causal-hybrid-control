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

_log = logging.getLogger(__name__)

Verdict = Literal["deploy", "shadow", "experiment", "hold", "rollback"]
GateMode = Literal["shadow", "experiment", "deployed", "retired"]

_Array = NDArray[np.float64]

_BETS = 0.9 * 2.0 ** -np.arange(12)  # constant bets, mixed with equal weight
_LOG_BETS = math.log(_BETS.size)
_MATCH = 1e-6  # relative gap allowed between a logged propensity and the mode's


def _vector(value: ArrayLike, name: str) -> _Array:
    array = np.array(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a vector, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} is not finite")
    array.setflags(write=False)
    return array


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
    Gaussian perturbation added to the policy's action, as applied.

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
        """The dither of every decision, for a reader that needs each to be the Gaussian draw.

        Raises:
            ValueError: on a log with no dither, or with a clipped decision, whose dither as
                applied is not the draw.
        """
        if self.dither is None:
            raise ValueError("the log records no dither")
        if self.saturated.any():
            raise ValueError(
                f"{int(self.saturated.sum())} of {self.saturated.size} decisions were clipped (the"
                f" first is decision {int(np.argmax(self.saturated))}), and a clipped decision's"
                " dither is not the Gaussian draw"
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
    1 given the past while the channel holds.

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
            drift = np.array(self.drift, dtype=np.float64)
            if drift.ndim == 1:
                drift = drift[:, None]
            if drift.ndim != 2 or drift.shape[0] != size or drift.shape[1] == 0:
                raise ValueError(
                    f"drift must have shape ({size},) or ({size}, detectors), got {drift.shape}"
                )
            if not (np.all(np.isfinite(drift)) and np.all(drift >= 0.0)):
                raise ValueError("drift e-values must be finite and non-negative")
            drift.setflags(write=False)
            object.__setattr__(self, "drift", drift)

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
    mode: GateMode = "shadow"
    improvement: _Evidence = field(default_factory=_Evidence)
    harm: _Evidence = field(default_factory=_Evidence)
    rollback: _Evidence = field(default_factory=_Evidence)
    drift: _Array | None = None  # one Shiryaev-Roberts statistic per detector
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
        self._zones = {name: _Zone(plan) for name, plan in plans.items()}

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
        alarms = {z: self._drift_alarm(self._zones[z], b.drift) for z, b in live.items()}
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
        if (
            batch.drift is not None
            and zone.drift is not None
            and batch.drift.shape[1] != zone.drift.shape[0]
        ):
            raise ValueError(
                f"zone {name!r}: {batch.drift.shape[1]} drift detectors, where the running"
                f" statistic has {zone.drift.shape[0]}"
            )

    def _drift_alarm(self, zone: _Zone, drift: _Array | None) -> bool:
        """Shiryaev-Roberts over e-values, ``R_t = (R_{t-1} + 1) e_t`` per detector, averaged; an
        alarm at the average reaching ``drift_arl`` keeps the average run length on an unchanged
        channel at least that."""
        if drift is None or drift.shape[0] == 0:
            return False
        if zone.drift is None:
            zone.drift = np.zeros(drift.shape[1])
        for row in drift:
            zone.drift = (zone.drift + 1.0) * row
            if zone.drift.mean() >= self.config.drift_arl:
                zone.drift = None
                return True
        return False

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
