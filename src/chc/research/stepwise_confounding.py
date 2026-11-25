"""A plan's value under unobserved confounding that acts one step at a time, and the confounding
that overturns it.

Logged transitions ``(s, a, r, s')`` of a plant with a discrete state and a discrete action identify
the plan's value only where nothing unobserved moves both the logger's action and the transition.
Where an unobserved ``U_t`` does, and ``U_t`` is *memoryless*, independent of the past given the
state ``S_t`` (Bruns-Smith and Zhou, "Robust fitted-Q-evaluation and iteration under sequentially
exogenous unobserved confounders", arXiv 2302.00662, Definition 1), the plant seen by a plan that
does not read ``U_t`` is still a Markov decision process on the observed state, and each step's
transition law under the plan is the logged one reweighted by ``w = e(a|s) / e(a|s, U)``, with
``E[w | s, a] = 1`` (their Theorem 1).

**The sensitivity level is CHC's ``Gamma``.** The model bounding ``w`` is Tan's (2006) marginal
sensitivity model on the odds of the logged action, the one
:func:`chc.uncertainty.msm_worst_case_mean` reads with ``propensity=`` and
:func:`chc.uncertainty.benchmark_gamma` calibrates: the odds of ``a`` given ``(s, U)`` lie within
a factor ``Gamma`` of its odds given ``s``. Bruns-Smith and Zhou call the same number ``Lambda``
(their Assumption 3): ``Gamma`` here is their ``Lambda``, and their
``tau = Lambda / (1 + Lambda)`` is ``Gamma / (Gamma + 1)``. The weight then lies in
``[alpha, beta] = [e + (1 - e) / Gamma, e + (1 - e) Gamma]``, ``e = e(a|s)`` the logger's
propensity, read off the log as the share of the state's transitions that took ``a``.
``weight_box="density"`` takes the box ``[1/Gamma, Gamma]`` instead, the continuous-dose model
:func:`~chc.uncertainty.msm_worst_case_mean` reads without a propensity (Jesson et al. 2022; the
paper's Appendix 4.1); on a discrete action it is the wider of the two, the limit ``e -> 0``.

**The closed form** (``validation/stepwise_confounding.mac``). The least ``E[w Y]`` over weights
in ``[alpha, beta]`` with mean 1 puts ``beta`` on the lowest outcomes, a share
``(1 - alpha) / (beta - alpha) = 1 / (Gamma + 1)`` of them whatever ``e``, and ``alpha`` on the
rest: ``lower(Y) = alpha E[Y] + (1 - alpha) LTM(Y)``, and the largest is
``upper(Y) = alpha E[Y] + (1 - alpha) UTM(Y)``. ``LTM`` is the mean of the lowest
``1 / (Gamma + 1)`` of the law, below its quantile at ``1 - tau``, and ``UTM`` that of the
highest, above its quantile at ``tau``: the paper's Proposition 5. At ``Gamma = 1`` both are
``E[Y]``. Read on the empirical law of a cell, the share falls between two transitions and the
tail takes the fraction of the next one, as :func:`~chc.uncertainty.msm_worst_case_mean` does;
the bound is then that function's own at the cell's propensity, the knapsack the tests compare it
with.

**The robust fitted-Q evaluation.** Over the plan's horizon ``H``, from ``V_H = 0`` back to step
0, each state ``s`` the plan reaches at step ``t`` reads the cell of its action
``a = plan[t, s]``: ``Y = r + V_{t+1}(s')`` over the cell's logged transitions, and
``V_t(s) = lower(Y)`` (``upper`` for the upper value). On a discrete state the fitted ``Q`` is the
cell's mean and the fitted quantile the cell's own, so the fit is the saturated one and nothing is
smoothed. The value is ``V_0`` averaged over the initial law.

The bound holds under the model: the true value lies in ``[lower, upper]`` at any ``Gamma`` at or
above the confounding's own, since each step's operator is monotone in ``V`` and each step's true
law is one of the reweightings it ranges over. It is sharp for each step's cell where the logger
has two actions (the paper's Proposition 4 and note 5). Two things can make the interval wider than
the model allows. With more actions the cells of one state are bounded apart, the paper's
rectangular relaxation. And the cells are pooled over the log's periods, a time-homogeneous plant,
whose one transition law each step's worst case may read differently.

**The premise is checked, not assumed.** Where ``U`` persists, as an unobserved state the plant
carries from one period to the next, the observed process is not Markov, and this evaluation has no
guarantee. The log's transitions are labelled by ``"unit"`` and ``"period"``, and the innovations of
each transition, its reward and its next state less their cell's means, are read for a lag-1
autocorrelation within units: :func:`chc.dynamics_id.persistence_check`'s statistic and test, on
these innovations instead of a fitted channel's moment residual. Under memoryless confounding the
innovation of a step is uncorrelated with the step before it; where a persistent ``U`` moves the
reward or the next state, they move together. A rejection at ``persistence_alpha`` raises
:class:`PersistentConfounding`. A clean check does not prove the premise: the test reads one lag
and a correlation, where the paper's test is of conditional independence. Where a persistent ``U``
is suspected, Bruns-Smith and Zhou's ``k``-order memoryless model adds the ``k - 1`` previous states
and actions to the state, and the check then reads the augmented log.

**The breakdown.** :func:`stepwise_breakdown` bisects for ``Gamma*``, the least ``Gamma`` at which
the plan's lower value falls below a comparator: a threshold, or the nominal value of a baseline
plan on the same log. The lower value falls with ``Gamma``, continuously, toward its limit at
``Gamma -> inf``, ``e E[Y] + (1 - e) min Y`` per cell (``min Y`` under the density box), so
``Gamma*`` is one crossing, and ``inf`` where the limit stays at or above the comparator: no
confounding of this kind can overturn the plan. As every ``Gamma`` in CHC, ``Gamma*`` is a level of
the sensitivity model, not a measured amount of confounding.

Scope. Discrete states and actions, integer codes from 0; a low-dimensional continuous state must be
binned first, and the premise then concerns the bins. A plan is deterministic: a fixed action
sequence over the horizon, or an action per step and state. Rewards are rewards, larger is better;
a cost enters negated. The bounds are plug-in: they are the sharp bounds on the log's empirical law
and carry none of its sampling error, which a cell of few transitions reads in full: its tails are
then a few points. :attr:`StepwiseRobustValue.least_count` reports the fewest any cell read. The
premise's check holds one float a transition and state, so a log of many states and transitions
takes their product in memory. The paper's orthogonalised estimator, its confidence bounds, and
the infinite-horizon estimator of Bennett and Kallus (NeurIPS 2024) are not built.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chc.dynamics_id import PersistenceCheck, _no_step_apart, _persistence
from chc.frames import _real_numbers
from chc.uncertainty import _top_tail_mean

__all__ = [
    "PersistentConfounding",
    "StepwiseBreakdown",
    "StepwiseRobustValue",
    "WeightBox",
    "stepwise_breakdown",
    "stepwise_robust_value",
]

WeightBox = Literal["odds", "density"]
"""The box the weight ``e(a|s) / e(a|s, U)`` lies in at ``Gamma``: Tan's odds box at the logged
propensity, or the density box ``[1/Gamma, Gamma]``."""

_PERSISTENCE_ALPHA = 0.05
_BISECTION_TOLERANCE = 1e-6  # relative, on Gamma
_DOUBLINGS = 1000  # a bound on the search for a level below the comparator, never reached
_KEYS = ("state", "action", "reward", "next_state", "unit", "period")


class PersistentConfounding(ValueError):
    """The log's innovations persist within units: the confounding may not be memoryless, and the
    evaluation is refused. ``check`` is the reading that refused it."""

    def __init__(self, check: PersistenceCheck, alpha: float) -> None:
        super().__init__(
            "the log's transition innovations persist within units (lag-1 autocorrelation "
            f"{check.correlation:+.3f}, p = {check.p_value:.3g} <= {alpha:g} over {check.pairs} "
            f"pairs of {check.units} units): an unobserved confounder may persist, so the observed "
            "process need not be Markov and the per-step bound has no guarantee. Add the previous "
            "state and action to the state and read the augmented log"
        )
        self.check = check


@dataclass(frozen=True)
class StepwiseRobustValue:
    """A plan's value bounded under memoryless confounding at ``gamma``, by
    :func:`stepwise_robust_value`. *Research.*"""

    gamma: float
    lower: float
    nominal: float  # the plain fitted evaluation, ``gamma = 1``
    upper: float
    weight_box: WeightBox
    least_count: int  # the fewest logged transitions in a cell the plan reads
    persistence: PersistenceCheck  # the premise's check, which did not reject


@dataclass(frozen=True)
class StepwiseBreakdown:
    """The least sensitivity level at which a plan's lower value falls below a comparator, by
    :func:`stepwise_breakdown`. *Research.*"""

    # the bisection's upper end, a level at which the lower value was read below the comparator,
    # within a relative 1e-6 of ``gamma_held``; 1 where the nominal value is below it already, inf
    # where the limit at an infinite level is not
    gamma_star: float
    gamma_held: float  # the bisection's lower end, a level at which it was not; nan at 1, inf alike
    comparator: float
    nominal: float  # the plan's value at ``gamma = 1``
    limit: float  # its lower value as ``gamma -> inf``
    weight_box: WeightBox
    persistence: PersistenceCheck


@dataclass(frozen=True)
class _Log:
    """A tabular log, by cell: each ``(state, action)``'s rewards and next states in log order,
    the logged propensity of each, the plan as a table, and the states it reaches at each step."""

    rewards: dict[tuple[int, int], NDArray[np.float64]]
    next_states: dict[tuple[int, int], NDArray[np.intp]]
    propensity: NDArray[np.float64]  # (states, actions)
    plan: NDArray[np.intp]  # (horizon, states)
    reached: tuple[NDArray[np.intp], ...]  # one per step
    initial: NDArray[np.float64]  # (states,)
    persistence: PersistenceCheck


def _codes(value: ArrayLike, name: str) -> NDArray[np.intp]:
    """``value`` as non-negative integer codes, one dimension."""
    array = np.asarray(_real_numbers(value, name))
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} is one code a transition, a non-empty vector; got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
        raise ValueError(f"{name} are codes, integers from 0; got dtype {array.dtype}")
    return array.astype(np.intp)


def _plan_table(plan: ArrayLike, states: int) -> NDArray[np.intp]:
    """``plan`` as ``(horizon, states)`` action codes: a sequence ``(H,)`` holds its action in
    every state; a table ``(H, S)`` must cover every state the log names."""
    table = np.asarray(_real_numbers(plan, "plan"))
    if table.ndim not in (1, 2) or table.shape[0] == 0:
        raise ValueError(f"a plan is an action sequence (H,) or a table (H, S); got {table.shape}")
    if not np.issubdtype(table.dtype, np.integer) or np.any(table < 0):
        raise ValueError(f"a plan's actions are codes, integers from 0; got dtype {table.dtype}")
    if table.ndim == 1:
        return np.repeat(table.astype(np.intp)[:, None], states, axis=1)
    if table.shape[1] < states:
        raise ValueError(
            f"a plan's table covers {table.shape[1]} states; the log and the plans name {states}"
        )
    return table.astype(np.intp)


def _initial_law(
    initial: int | ArrayLike | None, states: int, state: NDArray[np.intp], units, periods
) -> NDArray[np.float64]:
    """The initial law: a point mass at a state code, a law over the states, or by default the
    law of each unit's state in its earliest period."""
    if initial is None:
        codes = np.unique(units, return_inverse=True)[1].reshape(-1)
        order = np.lexsort((periods, codes))
        first = order[np.r_[True, codes[order[1:]] != codes[order[:-1]]]]
        return np.bincount(state[first], minlength=states) / first.size
    law = np.asarray(_real_numbers(initial, "initial"))
    if law.ndim == 0:
        if not np.issubdtype(law.dtype, np.integer) or not 0 <= int(law) < states:
            raise ValueError(f"an initial state is a code in [0, {states}); got {initial!r}")
        return np.eye(states)[int(law)]
    law = law.astype(np.float64)
    if (
        law.shape != (states,)
        or not np.all(np.isfinite(law))
        or np.any(law < 0.0)
        or abs(float(law.sum()) - 1.0) > 1e-9
    ):
        raise ValueError(f"an initial law is ({states},), non-negative, summing to 1")
    return law


def _innovations(
    state: NDArray[np.intp],
    action: NDArray[np.intp],
    reward: NDArray[np.float64],
    next_state: NDArray[np.intp],
    states: int,
    actions: int,
) -> NDArray[np.float64]:
    """Each transition's reward and next state, one-hot, less its cell's means: ``(N, 1 + S)``."""
    cell = state * actions + action
    count = np.bincount(cell, minlength=states * actions).astype(np.float64)
    outcome = np.column_stack([reward, np.eye(states)[next_state]])
    means = (
        np.stack(
            [np.bincount(cell, weights=column, minlength=states * actions) for column in outcome.T],
            axis=1,
        )
        / np.maximum(count, 1.0)[:, None]
    )
    return outcome - means[cell]


def _parse(
    logs: Mapping[str, ArrayLike],
    plan: ArrayLike,
    initial: int | ArrayLike | None,
    persistence_alpha: float,
    weight_box: str,
    baseline: ArrayLike | None = None,
) -> tuple[_Log, _Log | None]:
    """The log by cell for ``plan`` (and ``baseline``), after the premise's check, refused where
    the plan reaches a state whose action the log never took there."""
    missing = [key for key in _KEYS if key not in logs]
    if missing:
        raise KeyError(f"logs need {list(_KEYS)}; missing {missing}")
    if weight_box not in ("odds", "density"):
        raise ValueError(f"weight_box is 'odds' or 'density'; got {weight_box!r}")
    if not 0.0 <= persistence_alpha < 1.0:
        raise ValueError(f"persistence_alpha is a level in [0, 1); got {persistence_alpha}")
    state, action = _codes(logs["state"], "state"), _codes(logs["action"], "action")
    next_state = _codes(logs["next_state"], "next_state")
    reward = np.asarray(_real_numbers(logs["reward"], "reward"), dtype=np.float64)
    units, periods = np.asarray(logs["unit"]).reshape(-1), np.asarray(logs["period"]).reshape(-1)
    n = state.size
    if any(v.shape != (n,) for v in (action, next_state, reward, units, periods)):
        raise ValueError(f"state, action, reward, next_state, unit and period are {n} each")
    if not np.all(np.isfinite(reward)):
        raise ValueError("reward must be finite")
    plans = [p for p in (plan, baseline) if p is not None]
    widths = [np.shape(p)[1] for p in plans if np.ndim(p) == 2]
    states = max([int(max(state.max(), next_state.max())) + 1, *widths])
    tables = [_plan_table(p, states) for p in plans]
    actions = int(max(action.max(), *(t.max() for t in tables))) + 1
    if not np.issubdtype(periods.dtype, np.integer):
        raise ValueError(f"periods are whole numbers of steps; got dtype {periods.dtype}")
    check = _persistence(
        _innovations(state, action, reward, next_state, states, actions), units, periods, False
    )
    if check.pairs == 0 and np.unique(periods).size >= 2:
        raise ValueError(_no_step_apart(units, periods))
    if check.p_value <= persistence_alpha:
        raise PersistentConfounding(check, persistence_alpha)
    law = _initial_law(initial, states, state, units, periods)
    counts = np.zeros((states, actions))
    np.add.at(counts, (state, action), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    propensity = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    cells = state * actions + action
    order = np.argsort(cells, kind="stable")
    bounds = np.searchsorted(cells[order], np.arange(states * actions + 1))
    rewards, nexts = {}, {}
    for k in np.flatnonzero(np.diff(bounds)):
        rows = order[bounds[k] : bounds[k + 1]]
        rewards[divmod(int(k), actions)] = reward[rows]
        nexts[divmod(int(k), actions)] = next_state[rows]
    logs_ = tuple(
        _reach(table, law, rewards, nexts, propensity, weight_box, check) for table in tables
    )
    return logs_[0], (logs_[1] if len(logs_) > 1 else None)


def _reach(
    table: NDArray[np.intp],
    law: NDArray[np.float64],
    rewards: dict[tuple[int, int], NDArray[np.float64]],
    nexts: dict[tuple[int, int], NDArray[np.intp]],
    propensity: NDArray[np.float64],
    weight_box: str,
    check: PersistenceCheck,
) -> _Log:
    """The states ``table`` reaches at each step from the initial law's support, refused at the
    first whose action the log never took there, or, on the odds box, always took there."""
    reached = []
    now = np.flatnonzero(law > 0.0)
    for t in range(table.shape[0]):
        reached.append(now)
        following: set[int] = set()
        for s in now:
            cell = (int(s), int(table[t, s]))
            if cell not in rewards:
                raise ValueError(
                    f"the plan takes action {cell[1]} in state {cell[0]} at step {t}, which it "
                    "reaches, and the log never took that action there: the value is not identified"
                )
            if weight_box == "odds" and propensity[cell] == 1.0:
                raise ValueError(
                    f"the log took action {cell[1]} in every transition from state {cell[0]}, "
                    f"which the plan reaches at step {t}: Tan's model needs a propensity in "
                    "(0, 1), and a logged one of 1 would read no confounding; use "
                    "weight_box='density'"
                )
            following.update(int(x) for x in np.unique(nexts[cell]))
        now = np.array(sorted(following), dtype=np.intp)
    return _Log(rewards, nexts, propensity, table, tuple(reached), law, check)


def _cell_bound(y: NDArray[np.float64], gamma: float, e: float, side: int, box: str) -> float:
    """``alpha mean(y) + (1 - alpha) tail``: the sharp bound on ``E[w y]`` over the weight box at
    ``gamma``, its upper tail where ``side`` is 1 and its lower where -1."""
    mean = float(np.mean(y))
    if gamma == 1.0:
        return mean
    floor = e if box == "odds" else 0.0
    alpha = floor + (1.0 - floor) / gamma
    if math.isinf(gamma):
        tail = float(np.max(y)) if side > 0 else float(np.min(y))
    else:
        tail = side * _top_tail_mean(side * y, 1.0 / (gamma + 1.0))
    return alpha * mean + (1.0 - alpha) * tail


def _value(log: _Log, gamma: float, side: int, box: str) -> float:
    """The plan's value from the initial law, each step's cell bounded at ``gamma`` on ``side``."""
    horizon, states = log.plan.shape
    following = np.zeros(states)
    for t in range(horizon - 1, -1, -1):
        current = np.full(states, np.nan)
        for s in log.reached[t]:
            cell = (int(s), int(log.plan[t, s]))
            y = log.rewards[cell] + following[log.next_states[cell]]
            current[s] = _cell_bound(y, gamma, float(log.propensity[cell]), side, box)
        following = current
    support = log.reached[0]
    return float(log.initial[support] @ following[support])


def _gamma(gamma: float) -> float:
    if not 1.0 <= gamma < math.inf:
        raise ValueError(f"the MSM sensitivity Gamma must be >= 1 and finite, got {gamma}")
    return float(gamma)


def stepwise_robust_value(
    logs: Mapping[str, ArrayLike],
    plan: ArrayLike,
    gamma: float,
    *,
    initial: int | ArrayLike | None = None,
    weight_box: WeightBox = "odds",
    persistence_alpha: float = _PERSISTENCE_ALPHA,
) -> StepwiseRobustValue:
    """The lower and upper values of ``plan`` under memoryless confounding at ``gamma``.
    *Research.*

    ``logs`` holds one entry a transition under each of ``"state"``, ``"action"``,
    ``"reward"``, ``"next_state"``, ``"unit"`` and ``"period"``: states and actions as integer
    codes from 0, the reward as a real number, the unit as any label and the period as the whole
    number of steps the transition starts in. ``plan`` is a sequence of action codes ``(H,)``, one
    a step, or a table ``(H, S)``, one a step and state; ``H`` is the horizon. ``initial`` is a
    state code or a law over the states, by default the law of each unit's state in its earliest
    period. ``gamma`` is the sensitivity level, CHC's MSM ``Gamma`` and Bruns-Smith and Zhou's
    ``Lambda`` (the module says how the two meet), on ``weight_box``: ``"odds"``, Tan's model at
    the logged propensity, or ``"density"``, the box ``[1/Gamma, Gamma]``.

    The premise, that the confounding is memoryless, is checked first, as the module says: where
    the log's innovations persist within units at ``persistence_alpha``, the evaluation is refused.
    At ``gamma = 1`` the three values are the plain fitted evaluation, each cell's mean, exactly.

    Raises:
        PersistentConfounding: where the premise's check rejects at ``persistence_alpha``.
        KeyError: on logs without one of the six entries.
        ValueError: on ``gamma`` below 1 or not finite; entries of other lengths, codes that are
            not integers from 0, a reward not finite, periods not whole numbers or a unit with two
            transitions starting in one period; a plan of
            another shape or whose table covers fewer states than the log names; an initial state
            or law that is not one; a log of two periods or more on which no unit has two
            transitions one period apart, which leaves the premise unread; a state the plan
            reaches whose action the log never took there; and on the odds box, one whose action
            the log always took there, whose propensity of 1 would read no confounding.
    """
    gamma = _gamma(gamma)
    log, _ = _parse(logs, plan, initial, persistence_alpha, weight_box)
    least = min(
        log.rewards[(int(s), int(log.plan[t, s]))].size
        for t in range(log.plan.shape[0])
        for s in log.reached[t]
    )
    return StepwiseRobustValue(
        gamma=gamma,
        lower=_value(log, gamma, -1, weight_box),
        nominal=_value(log, 1.0, -1, weight_box),
        upper=_value(log, gamma, 1, weight_box),
        weight_box=weight_box,
        least_count=int(least),
        persistence=log.persistence,
    )


def stepwise_breakdown(
    logs: Mapping[str, ArrayLike],
    plan: ArrayLike,
    *,
    threshold: float | None = None,
    baseline: ArrayLike | None = None,
    initial: int | ArrayLike | None = None,
    weight_box: WeightBox = "odds",
    persistence_alpha: float = _PERSISTENCE_ALPHA,
) -> StepwiseBreakdown:
    """``Gamma*``, the least sensitivity level at which ``plan``'s lower value falls below
    ``threshold``, or below ``baseline``'s nominal value on the same log. *Research.*

    ``logs``, ``plan``, ``initial``, ``weight_box`` and ``persistence_alpha`` are
    :func:`stepwise_robust_value`'s; ``baseline`` is a plan over the same horizon, evaluated at
    ``Gamma = 1`` from the same initial law. Exactly one of ``threshold`` and ``baseline`` is given.

    The lower value is continuous and non-increasing in ``Gamma``, so ``Gamma*`` is the crossing
    of the comparator: a search doubling ``Gamma`` from 1 brackets it, and a bisection in
    ``log Gamma`` narrows the bracket to a relative ``1e-6``. ``gamma_star`` is its upper end, a
    level at which the lower value was read below the comparator, and ``gamma_held`` its lower
    end, at which it was not. ``gamma_star`` is 1 where the nominal value is below the comparator
    already, and ``inf`` where the lower value's limit as ``Gamma -> inf``,
    ``e E[Y] + (1 - e) min Y`` per cell on the odds box and ``min Y`` on the density box, is not.

    Raises:
        PersistentConfounding: as :func:`stepwise_robust_value`.
        KeyError: as :func:`stepwise_robust_value`.
        ValueError: where neither or both of ``threshold`` and ``baseline`` are given, the
            threshold is not finite, the baseline's horizon is not the plan's, and as
            :func:`stepwise_robust_value`, for the baseline as for the plan.
    """
    if (threshold is None) == (baseline is None):
        raise ValueError("give exactly one comparator: threshold or baseline")
    log, base = _parse(logs, plan, initial, persistence_alpha, weight_box, baseline)
    comparator = _comparator(log, base, threshold, weight_box)
    nominal = _value(log, 1.0, -1, weight_box)
    limit = _value(log, math.inf, -1, weight_box)
    if nominal < comparator:
        star, held = 1.0, math.nan
    elif limit >= comparator:
        star, held = math.inf, math.inf
    else:
        star, held = _crossing(log, comparator, weight_box)
    return StepwiseBreakdown(star, held, comparator, nominal, limit, weight_box, log.persistence)


def _comparator(log: _Log, base: _Log | None, threshold: float | None, box: str) -> float:
    """The threshold, or the baseline's nominal value."""
    if threshold is not None:
        comparator = float(_real_numbers(threshold, "threshold"))
        if not math.isfinite(comparator):
            raise ValueError(f"the threshold must be finite, got {comparator}")
        return comparator
    if base is None:
        raise ValueError("give exactly one comparator: threshold or baseline")
    if base.plan.shape[0] != log.plan.shape[0]:
        raise ValueError(
            f"the baseline's horizon is {base.plan.shape[0]}, the plan's {log.plan.shape[0]}"
        )
    return _value(base, 1.0, -1, box)


def _crossing(log: _Log, comparator: float, box: str) -> tuple[float, float]:
    """``(star, held)``: levels at which the lower value reads below ``comparator`` and at or above
    it, a relative ``1e-6`` apart, where it is at or above at 1 and below in the limit."""
    held, star = 1.0, 2.0
    for _ in range(_DOUBLINGS):
        if _value(log, star, -1, box) < comparator:
            break
        held, star = star, 2.0 * star
    else:
        raise RuntimeError(f"no level up to 2^{_DOUBLINGS} reads below the comparator")
    while star > held * (1.0 + _BISECTION_TOLERANCE):
        middle = math.sqrt(held * star)
        if _value(log, middle, -1, box) < comparator:
            star = middle
        else:
            held = middle
    return star, held
