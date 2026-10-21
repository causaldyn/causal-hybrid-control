"""Certainty-equivalence suboptimality for LQ control -- CHC's regret guarantee (LQ special case).

For a linear-quadratic problem the certainty-equivalent controller (solve the LQR for an *estimated*
model, then apply that gain to the true plant) has a suboptimality gap that is **quadratic** in the
model error: ``J(K_hat) - J* <= O(||[dA, dB]||^2)`` in the small-error regime, *provided* the
estimated controller stabilises the true plant. This LOCAL quadratic bound is Mania, Tu & Recht
(2019), which improved the earlier LINEAR-in-error robust-synthesis bound of Dean, Mania, Matni,
Recht & Tu (2018); it is a small-error suboptimality bound, not a global equality. This is the
analysable special case of CHC's pessimism story -- small model error costs almost nothing, but the
penalty grows with error, which is exactly what the calibrated uncertainty penalty
(:mod:`chc.uncertainty`) is there to price in offline.

A NumPy/scipy analysis tool (like :mod:`chc.did` / :mod:`chc.scm`), independent of the JAX ``x64``
flag. The infinite-horizon discrete LQR is solved via the DARE; a controller's true-plant cost via
the discrete Lyapunov equation.

The certificates that each back one research result -- the regret scaling and its constants,
orthogonal, multi-channel and transfer control, the information and Van Trees floors, interference
and fold designs, the exact ratio moments and the confounding-robust controllers -- moved to
:mod:`chc.research.regret` in 0.19.0. What stays here is the LQ solve and the certainty-equivalence
gap, the exploration certificates and the capped exploration policy, and the minimax-regret
actions, moves and LQ policies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.optimize import brentq
from scipy.special import erf, ndtr, psi

from chc import _moved
from chc._action_units import action_units

Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]


def dlqr(a: Matrix, b: Matrix, q: Matrix, r: Matrix) -> tuple[Matrix, Matrix]:
    """Infinite-horizon discrete LQR: optimal gain ``K`` and cost-to-go ``P`` (via the DARE)."""
    scale = action_units(r)
    b_unit, r_unit = b * scale, scale[:, None] * r * scale
    p = solve_discrete_are(a, b_unit, q, r_unit)
    k = scale[:, None] * np.linalg.solve(r_unit + b_unit.T @ p @ b_unit, b_unit.T @ p @ a)
    return k, p


def closed_loop_cost(a: Matrix, b: Matrix, k: Matrix, q: Matrix, r: Matrix, x0: Vector) -> float:
    """Infinite-horizon LQ cost of applying gain ``k`` to plant ``(a, b)`` from ``x0``.

    ``x0' P x0`` where ``P`` solves the discrete Lyapunov equation for the closed loop; ``+inf`` if
    ``k`` fails to stabilise ``(a, b)`` (a destabilising controller has unbounded cost).
    """
    a_cl = a - b @ k
    if np.max(np.abs(np.linalg.eigvals(a_cl))) >= 1.0:
        return float("inf")
    p = solve_discrete_lyapunov(a_cl.T, q + k.T @ r @ k)
    return float(x0 @ p @ x0)


def certainty_equivalence_gap(
    a: Matrix, b: Matrix, q: Matrix, r: Matrix, a_hat: Matrix, b_hat: Matrix, x0: Vector
) -> float:
    """Suboptimality ``J(K_hat) - J*`` of the certainty-equivalent controller on the true plant.

    ``K_hat`` is the LQR-optimal gain for the estimated model ``(a_hat, b_hat)``; the gap is its
    true-plant cost minus the cost of the true-optimal gain. Zero iff the estimate induces the
    optimal gain; otherwise non-negative (the optimum is optimal).
    """
    k_hat, _ = dlqr(a_hat, b_hat, q, r)
    k_star, _ = dlqr(a, b, q, r)
    return closed_loop_cost(a, b, k_hat, q, r, x0) - closed_loop_cost(a, b, k_star, q, r, x0)


@dataclass(frozen=True)
class OptimalExplorationCurve:
    """Explore-exploit for causal control: excess(v) = A*v + B/v is minimised at v* = sqrt(B/A)."""

    exploration_grid: Vector  # injected exploration variance v
    total_cost: Vector  # A*v + C*Var(b_hat): explore cost + estimation-driven control regret
    explore_cost: Vector  # A*v, A = b^2 + rr (the control-cost curvature)
    estimation_cost: Vector  # C*Var(b_hat) ~ B/v, B = C*sigma^2/n (the Cramer-Rao floor, Result 10)
    total_cost_confounded: Vector  # same curve when confounding steals signal (B -> B/kappa)
    vstar_theory: float  # sqrt(B/A): the AM-GM optimum
    vstar_empirical: float  # argmin of the simulated total cost
    vstar_confounded_theory: float  # sqrt(B/(kappa*A)) > vstar_theory
    vstar_confounded_empirical: float  # argmin under confounding (>= the experimental one)
    floor_theory: float  # 2*sqrt(A*B): the irreducible minimal excess cost


def optimal_exploration_certificate(
    *,
    b: float = 0.3,
    rr: float = 0.2,
    xt: float = 1.0,
    sigma: float = 1.0,
    n: int = 30,
    kappa_conf: float = 0.4,
    grid_lo: float = 0.02,
    grid_hi: float = 3.0,
    n_grid: int = 15,
    n_seeds: int = 1000,
) -> OptimalExplorationCurve:
    """Optimal exploration for causal control -- the actionable DUAL of the Cramer-Rao lower bound
    (Result 10; derived in ``validation/optimal_exploration.mac``, proved in
    ``proofs/optimal_exploration.v``). Injecting exploration variance ``v`` buys identifying
    information (shrinking the estimation floor ``B/v``, ``B=C*sigma^2/n``) but is itself an action
    error costing ``A*v`` control (``A=b^2+rr``, by completing-the-square). The total excess
    ``A*v + B/v`` is minimal at ``v* = sqrt(B/A)``, irreducible cost ``2*sqrt(A*B)`` (AM-GM). The
    optimum is interior -- pure exploitation is never optimal -- and confounding raises ``B``
    (here via a fraction ``kappa`` of identifying signal), lifting BOTH ``v*`` and the floor. The
    estimation term is Monte-Carlo; the ``A*v`` term is the proven CE cost of the injected variance.

    SCOPE of ``B/v``: it is the large-``n`` form. The estimation arm fits least squares on ``n``
    Gaussian rounds, whose mean squared error is ``sigma^2/((n-2)*v)`` exactly, an inverse
    chi-square mean (``validation/optimal_exploration.mac``). So the simulated cost is
    ``n/(n-2)`` times ``B/v``, 7% above it at the default ``n = 30``, and its optimum lies
    ``sqrt(n/(n-2))`` times ``vstar_theory`` out, 1.035 at ``n = 30``, inside one step of the
    default grid (a ratio of 1.43). The arm's mean is infinite for ``n <= 2`` and its variance for
    ``n <= 4``: there its Monte Carlo average does not settle.
    """
    coeff = xt**2 * (rr - b**2) ** 2 / (rr + b**2) ** 3
    a_curv = b**2 + rr
    grid = np.geomspace(grid_lo, grid_hi, n_grid)

    def estimation_arm(noise: float) -> Vector:
        est = np.zeros(grid.size)
        for i, v in enumerate(grid):
            sq = np.empty(n_seeds)
            for s in range(n_seeds):
                rng = np.random.default_rng(7919 * s + int(1e6 * v))
                u = np.sqrt(v) * rng.standard_normal(n)
                y = b * u + noise * rng.standard_normal(n)
                sq[s] = (np.dot(u, y) / np.dot(u, u) - b) ** 2
            est[i] = coeff * float(np.mean(sq))
        return est

    est_exp = estimation_arm(sigma)
    est_conf = estimation_arm(sigma / np.sqrt(kappa_conf))  # less identifying signal <=> more noise
    explore = a_curv * grid
    total_exp = explore + est_exp
    total_conf = explore + est_conf
    b_floor = coeff * sigma**2 / n
    vstar = float(np.sqrt(b_floor / a_curv))
    vstar_conf = float(np.sqrt(b_floor / (kappa_conf * a_curv)))
    return OptimalExplorationCurve(
        grid,
        total_exp,
        explore,
        est_exp,
        total_conf,
        vstar,
        float(grid[int(np.argmin(total_exp))]),
        vstar_conf,
        float(grid[int(np.argmin(total_conf))]),
        float(2.0 * np.sqrt(a_curv * b_floor)),
    )


@dataclass(frozen=True)
class AdaptiveExplorationCurve:
    """Adaptive exploration: a tapering schedule reaches the van-Trees sqrt(T) rate."""

    horizons: Vector  # horizon T grid
    adaptive_regret: Vector  # cumulative regret of the tapering schedule (v_t ~ 1/sqrt(t)): sqrt(T)
    greedy_regret: Vector  # cumulative regret with no exploration: ~ T (linear)
    static_regret: Vector  # cumulative regret of a constant-v schedule (Result 11 static): ~ T
    # 2*sqrt(A*K*T/eta) - A*m0/eta, K = C*sigma^2: the proved sequence lower bound, which the
    # information m0 held before the first round lowers
    lower_bound: Vector
    # log-log slopes of adaptive (~0.5) and greedy (~1) regret vs T; nan at rr = b^2, where the
    # decision does not move with b and every schedule's regret is zero
    adaptive_slope: float
    greedy_slope: float
    schedule: Vector  # the tapering exploration schedule v_t (decreasing)
    adaptive_over_bound: float  # adaptive / van-Trees bound at the largest T (>= 1, ~ constant)


def adaptive_exploration_certificate(
    *,
    b: float = 1.0,
    rr: float = 0.5,
    xt: float = 1.0,
    sigma: float = 0.5,
    m0: float = 1.0,
    eta: float = 1.0,
    static_v: float = 0.05,
    horizons: Sequence[int] = (50, 150, 500, 1500, 5000),
    schedule_horizon: int = 400,
) -> AdaptiveExplorationCurve:
    """CONTRIBUTION 3 -- ADAPTIVE information-exploration duality (derived in
    ``validation/adaptive_exploration.mac``, proved in ``proofs/adaptive_exploration.v``). At round
    ``t`` the controller injects ``v_t`` that raises accumulated Fisher information
    ``m_t = m0 + eta*sum_{s<t} v_s`` for FUTURE rounds, where ``eta`` in (0,1] is the identifying
    information PER UNIT exploration (``eta=1`` identified, ``eta->0`` confounded). The per-round
    regret is ``A*v_t + K/m_t``, ``K/m_t`` the van-Trees (Bayesian Cramer-Rao) floor -- valid for
    adaptive data, unlike ordinary Cramer-Rao (algebraic core in ``proofs/van_trees.v``). Its
    numerator is ``K = C*sigma^2`` with ``C = A*(du*/db)^2``, the convention
    ``validation/adaptive_exploration.mac`` states in its first line: the observation noise enters
    it, so ``sigma`` moves both the bound and the schedule scale. The RATE-OPTIMAL schedule
    ``v_t = kappa/sqrt(t)`` achieves ``Theta(sqrt(T))``, matching the ``Theta(sqrt(T))`` sequence
    lower bound ``>= 2*sqrt(A*K*T/eta) - A*m0/eta`` (Rocq ``reduced_objective_lower_bound`` with
    ``a=A/eta``, where the Rocq ``C`` is this ``K``, proves its one-variable core
    ``a*x + C*T/x >= 2*sqrt(a*C*T)``; the reduction from the sequence is not in Rocq), NOT a
    corollary of the per-round floor.
    The leading term scales as ``1/sqrt(eta)``, where ``eta`` is the INJECTED-EXPLORATION efficiency
    (information per unit exploration variance, distinct from an observational residual fraction):
    smaller ``eta`` provably raises the leading term ``2*sqrt(A*K*T/eta)`` (Rocq
    ``lower_efficiency_raises_sequence_floor``); the whole floor rises only while
    ``eta > A*m0^2/(K*T)``, below which its ``-A*m0/eta`` term wins --
    this ``1/sqrt(eta)`` is identification-efficiency-specific (attenuation: noncompliance,
    dilution, partial observability, interference -- NOT necessarily confounding; ``eta=1`` for
    clean directly-observed randomisation), the causal content the generic bound lacked. Greedy is
    ``Theta(T)``; the static ``v*`` of Result 11 over-explores. SCOPE: the ``t^{-1/2}`` schedule,
    ``sqrt(T)`` rate and van-Trees ``sqrt(T)`` lower bounds are KNOWN in adaptive LQR
    (Ziemann-Sandberg; Wagenmaker et al.); the FULL minimax constant is no longer open --
    :func:`minimax_exploration_certificate` proves it over ALL policies and shows this schedule pays
    a factor sqrt(2) over that floor. Conditions: ``m0=O(1)``; exploitation adds no identifying
    information; exploration information is linear in ``v_t`` (else greedy need not be
    ``Theta(T)``). The myopic ``max(0, sqrt(K/A)-m_t)`` rule is a DIFFERENT (one-shot) object.
    """
    a_curv = b * b + rr
    c_curv = xt**2 * (rr - b**2) ** 2 / (rr + b**2) ** 3  # C = A*(du*/db)^2
    k_vt = c_curv * sigma**2  # K = C*sigma^2: the van-Trees floor NUMERATOR, not C alone
    kappa = np.sqrt(k_vt / (a_curv * eta))  # tapering scale (eta-aware): v_t = kappa / sqrt(t)

    def cumulative(horizon: int, sched: np.ndarray) -> float:
        m = m0 + eta * np.concatenate([[0.0], np.cumsum(sched)[:-1]])  # info BEFORE round t
        return float(np.sum(a_curv * sched + k_vt / m))

    hs = np.asarray(horizons, dtype=np.float64)
    adaptive = np.zeros(hs.size)
    greedy = np.zeros(hs.size)
    static = np.zeros(hs.size)
    for i, horizon in enumerate(horizons):
        t = np.arange(1, horizon + 1)
        adaptive[i] = cumulative(horizon, kappa / np.sqrt(t))
        greedy[i] = cumulative(horizon, np.zeros(horizon))
        static[i] = cumulative(horizon, np.full(horizon, static_v))
    # 2*sqrt(A*K*T/eta) - A*m0/eta (Rocq reduced_objective_lower_bound): without the offset a
    # single burst of exploration in round 1 beat it, by up to A*m0/eta
    lb = 2.0 * np.sqrt(a_curv * k_vt * hs / eta) - a_curv * m0 / eta

    def slope(regret: np.ndarray) -> float:
        if not np.all(regret > 0.0):
            return float("nan")
        return float(np.polyfit(np.log(hs), np.log(regret), 1)[0])

    adaptive_slope, greedy_slope = slope(adaptive), slope(greedy)
    sched = kappa / np.sqrt(np.arange(1, schedule_horizon + 1, dtype=np.float64))
    return AdaptiveExplorationCurve(
        hs,
        adaptive,
        greedy,
        static,
        lb,
        adaptive_slope,
        greedy_slope,
        sched,
        float(adaptive[-1] / lb[-1]),
    )


@dataclass(frozen=True)
class MinimaxExplorationCurve:
    """The sequential minimax floor: no policy beats it, a front-loaded design attains it."""

    horizons: Vector  # horizon T grid
    floor: Vector  # c_causal*sqrt(T) - (A*sigma^2/eta)*I0, the local-minimax lower bound
    burst_regret: Vector  # front-loaded design: whole budget in round 1 -- attains the floor
    taper_regret: Vector  # the 1/sqrt(t) schedule at its own optimal scale -- sqrt(2) above
    greedy_regret: Vector  # no exploration: linear in T
    constant_regret: Vector  # a fixed per-round v: linear in T
    c_causal: float  # 2*A*|du*/db|*sigma/sqrt(eta), the explicit minimax constant
    min_policy_ratio: float  # min regret/floor over policies and horizons; < 1 falsifies the floor
    burst_over_floor: (
        float  # burst/floor at the largest T (-> 1: the constant is SHARP, not a rate)
    )
    taper_over_floor: float  # taper/floor at the largest T (-> sqrt(2): tapering costs a constant)
    eta_slope: float  # log-log slope of c_causal vs eta (-> -0.5: the 1/sqrt(eta) causal scaling)


@dataclass(frozen=True)
class CappedExplorationPolicy:
    """The optimal exploration schedule under a per-round action cap -- Result 56, extended.

    For a constant cap the policy is three numbers: explore for ``block_rounds`` rounds, at ``cap``
    in all but the last and at ``last_round`` in that one, then stop. It is not an approximation --
    the objective is convex and depends on the schedule only through its prefix sums, so
    front-loading as hard as the box allows is globally optimal, and the exchange argument leaves
    at most one round below its cap.

    When the cap VARIES or a total ``budget`` is imposed, the invariant is not the block length but
    :attr:`exploration_mass`: the first-order condition sees the cap only through the mass already
    delivered, so the optimal mass is cap-free and the schedule decides only how many rounds it
    takes to deliver it (``validation/capped_exploration_schedule.mac``).
    """

    cap: float  # the constant cap; for a schedule, the mean over the block that was chosen
    block_rounds: int
    cost: float
    uncapped_floor: float
    taper_cost: float
    excess: float  # cost - uncapped_floor: additive, and logarithmic in T
    predicted_excess: float  # (K/(2 c cap)) ln T, the derived leading form of that excess
    exploration_mass: float  # sum of the caps actually used -- what the stopping rule is about
    predicted_mass: float  # the self-consistent stopping mass; it does not know the cap SCHEDULE
    budget_binds: bool  # True when the budget, not the caps, is what stopped the block
    # the exploration in the block's last round: its cap, or less where the optimum fills that
    # round only partly -- the one unsaturated round the exchange argument allows
    last_round: float


def _capped_block_cost(
    rounds: NDArray[np.float64],
    *,
    cap: float,
    curvature: float,
    numerator: float,
    info_rate: float,
    prior_info: float,
    horizon: int,
) -> NDArray[np.float64]:
    """Cost of exploring at ``cap`` for ``rounds`` rounds and then stopping.

    The estimation term inside the block is a harmonic sum with step ``info_rate*cap``, which is
    a difference of digammas -- so the whole curve is O(1) per candidate length and the search
    below never has to materialise a schedule.
    """
    offset = prior_info / (info_rate * cap)
    inside = (psi(offset + rounds) - psi(offset)) / (info_rate * cap)
    after = (horizon - rounds) / (prior_info + info_rate * cap * rounds)
    return curvature * cap * rounds + numerator * (inside + after)


def _scheduled_block_costs(
    caps: NDArray[np.float64],
    *,
    curvature: float,
    numerator: float,
    info_rate: float,
    prior_info: float,
) -> NDArray[np.float64]:
    """Exact cost of stopping after each round under a greedy fill of ``caps``; index 0 = never.

    A varying cap makes the prefix sums arbitrary, so the digamma shortcut of
    :func:`_capped_block_cost` -- which needs the information to grow by a constant step -- does
    not apply and the harmonic sum has to be materialised. That is O(T) once against O(1) per
    candidate, which is why the two paths are kept apart rather than unified on the slower one.
    """
    mass = np.concatenate([[0.0], np.cumsum(caps)])
    info = prior_info + info_rate * mass
    inside = np.concatenate([[0.0], np.cumsum(numerator / info[:-1])])
    remaining = caps.size - np.arange(caps.size + 1, dtype=np.float64)
    return curvature * mass + inside + numerator * remaining / info


def _self_consistent_mass(
    caps: NDArray[np.float64],
    *,
    horizon: int,
    curvature: float,
    numerator: float,
    info_rate: float,
    prior_info: float,
) -> float:
    """The stopping mass written against the REMAINING horizon rather than the whole one.

    The first-order condition of ``validation/capped_exploration_schedule.mac`` STEP 2 is
    ``A (I0 + c S)^2 = K c (T - n)`` with ``n`` the round on which the greedy fill delivers ``S``
    -- so the target mass and the round that reaches it are defined together, and only the prefix
    sums of the caps connect them. Result 56's ``sqrt(K T/(A c)) - I0/c`` is this at ``n << T``;
    it runs 25% high as soon as the caps open late, which a schedule is free to do. At a CONSTANT
    cap the gap to it is not asymptotic at all but the exact constant ``K/(2 A c cap)``, derived
    in :func:`capped_exploration_policy`.

    The mass ``S(n)`` falls with ``n`` and the round that reaches it rises with the mass, so
    ``reach(S(n)) - n`` is strictly decreasing: the balance round is the smallest ``n`` whose mass
    the fill reaches by round ``n``, found by bisection. It is the fixed point where one exists and
    otherwise the lower side of the round boundary the balance straddles. Iterating the map is not
    enough: it converges only while its slope is below one, and a tight cap or a prefix of zero
    caps makes the fixed point repelling (``T = 41`` at ``cap = 0.01`` read a mass of 0 against the
    balance's ``0.167``).
    """
    prefix = np.cumsum(caps)

    def mass_for(rounds: int) -> float:
        reach = numerator * max(horizon - rounds, 0) / (curvature * info_rate)
        return max(float(np.sqrt(reach)) - prior_info / info_rate, 0.0)

    lo, hi = 1, horizon  # at n = T the mass is 0, which round 1 reaches
    while lo < hi:
        mid = (lo + hi) // 2
        if int(np.searchsorted(prefix, mass_for(mid))) + 1 > mid:
            lo = mid + 1
        else:
            hi = mid
    return mass_for(lo)


def _greedy_fill(caps: NDArray[np.float64], budget: float | None) -> NDArray[np.float64]:
    """Saturate the caps from round 1 until the budget runs out -- Result 56's exchange argument
    says an earlier round is strictly cheaper, and it never mentions what the caps are."""
    if budget is None:
        return caps
    return np.diff(np.concatenate([[0.0], np.minimum(np.cumsum(caps), budget)]))


def capped_exploration_policy(
    *,
    horizon: int,
    cap: float | Sequence[float],
    budget: float | None = None,
    b: float = 1.0,
    rr: float = 0.5,
    xt: float = 1.0,
    sigma: float = 0.7,
    i0: float = 1.0,
    eta: float = 0.6,
) -> CappedExplorationPolicy:
    """The matching causal policy under a per-round action cap -- Result 56.

    :func:`minimax_exploration_certificate` showed that a front-loaded BURST attains the minimax
    floor while a ``1/sqrt(t)`` taper sits ``sqrt(2)`` above it, and conjectured that a per-round
    action cap -- which every real actuator has -- is what would make the taper the right shape.
    It is not. The regret

        ``F(v) = A sum_t v_t + K sum_t 1/(I0 + c S_{t-1})``,  ``S`` the prefix sum,
        ``c = eta/sigma^2``

    depends on the schedule only through its prefix sums and is strictly decreasing in each of
    them, so moving exploration from a later round to an earlier one at equal budget STRICTLY
    lowers it (``proofs/capped_exploration.v``, ``earlier_is_strictly_cheaper``) -- an argument
    that never mentions the cap. The Hessian is a sum of rank-one PSD terms, so the objective is
    convex and the exchange argument gives a GLOBAL optimum: saturate the cap on a prefix, fill at
    most one more round partly, then stop. A clipped burst, not a taper. The partial round is
    worth ``6.8e-4`` of ``3.078`` at ``T = 50``, ``cap = 0.3`` and ``3e-6`` at ``T = 1e5``,
    ``cap = 0.03``; a whole block alone can be worth less than not exploring at all -- at
    ``T = 10``, ``cap = 3`` one round at the cap costs ``4.72`` against ``0.74`` for none.

    The price of the cap is ADDITIVE and LOGARITHMIC, not a constant factor. The optimal block
    length is ``n* = sqrt(K T/(A c))/cap`` TO LEADING ORDER -- exactly it is ``S*/cap``, smaller
    by the constant derived below -- and at it the leading term is ``2 sqrt(A K T/c)``,
    which with ``K = A (du*/db)^2`` and ``c = eta/sigma^2`` is EXACTLY the uncapped constant
    ``c_causal sqrt(T)`` (``capped_leading_term_is_the_uncapped_floor``). What is left over is the
    harmonic sum, ``(K/(2 c cap)) ln T + O(1)`` -- so the ratio to the uncapped floor tends to 1
    for every fixed cap, while the taper's ``sqrt(2)`` does not. A tighter actuator explores for
    LONGER, not more gently.

    ``block_rounds`` is found by an exact search over the convex block-cost curve, which digamma
    makes O(1) per candidate, and the last round's amount in closed form on the two rounds next to
    the best whole block; ``cost`` is the true cost of that schedule, the convex program's minimum
    to rounding. ``taper_cost`` is the
    clipped ``1/sqrt(t)`` schedule at its own optimal scale, for comparison -- at ``T = 1e5`` and
    ``cap = 0.03`` the block costs ``196.11`` against the taper's ``266.81``, a ``36%`` gap that
    does not close.

    ``predicted_excess`` carries only the LEADING log term; its ``O(1)`` is not modelled, so read
    it as a slope rather than a level. Per decade of ``T`` at ``cap = 0.03`` it predicts ``2.32``
    against a measured ``2.20`` then ``2.29`` -- converging, which is the claim.

    A VARYING cap, or a total ``budget`` on top of it, was Result 56's own open item: the exchange
    argument still says "as early as feasible", but ``n*`` is then not a single number. It turns out
    ``n*`` was never the invariant. The first-order condition is

        ``dF/dn = cap(n) [ A - K c (T - n) / (I0 + c S(n))^2 ]``

    (``validation/capped_exploration_schedule.mac`` STEP 2b, residual 0), whose bracket contains the
    cap nowhere -- only the mass ``S(n)`` delivered so far. So the optimal exploration MASS solves

        ``A (I0 + c S)^2 = K c (T - n)``,  with ``n`` the round the prefix sum delivers ``S``

    and is free of the cap LEVEL while the round that reaches it is not. It is not free of the
    horizon: ``n`` sits inside its own condition, so ``predicted_mass`` is a fixed point rather
    than a closed form, and Result 56's ``sqrt(K T/(A c)) - I0/c`` is the ``n << T`` limit -- 25%
    high as soon as the caps open late.

    Under a CONSTANT cap that fixed point has a closed form, and it says exactly what Result 56
    drops. The stopping round is ``n = S/cap``, so the remaining horizon is itself a function of
    the mass and the condition becomes a QUADRATIC in ``w = I0 + c S``,

        ``A w^2 + (K/cap) w = K c T + K I0/cap``

    whose Result 56 counterpart is the same balance with the horizon held at the full ``T``,
    ``A w0^2 = K c T``. Subtracting the two kills ``T`` entirely and leaves
    ``A (w0^2 - w^2) = K c S/cap``, so the cap-free form OVER-states the mass at every finite
    horizon -- never under-states it -- by ``(w0 - w)/c <= K/(2 A c cap)``, a CONSTANT rather than
    a vanishing remainder (``proofs/capped_exploration_schedule.v`` section (H)). At the defaults
    and ``cap = 0.01`` the ceiling is ``2.016``, approached strictly from below as
    ``1.834, 1.959, 1.998, 2.011, 2.015`` over ``T = 1e4 .. 1e8``, the remainder decaying like
    ``(K + 4 A I0 cap) sqrt(K/(A c)) / (8 A c cap^2 sqrt(T))`` (STEP 7 of
    ``validation/capped_exploration_schedule.mac``, confirmed to 60 digits by
    ``validation/capped_exploration_o1.gp``). It is the TIGHT actuator the omission hurts: at
    ``T = 4000`` the over-statement is ``0.5%`` of the mass at ``cap = 0.3`` and ``17%`` at
    ``cap = 0.01``.

    A budget does not interact with the caps either: below the
    root the objective is strictly decreasing in the mass, so a binding budget is spent in full, as
    early as the caps allow. The two collapse to

        deliver mass ``min(budget, S*)``, greedily -- ``block_rounds`` is where the prefix sum of
        the caps first reaches it.

    ``cap`` therefore takes either a scalar or a per-round schedule of length ``horizon``. What does
    NOT become cap-free is the COST: delivering the same mass earlier is strictly cheaper, so two
    schedules with the same prefix mass at ``n*`` still cost differently. The stopping rule is the
    invariant, not the value. A prefix of ZERO caps is the sharp case -- it adds exactly ``m K/I0``
    and hands the rest of the horizon to the same problem, so a dead actuator has a price and not a
    distortion.

    Scope: as in :func:`minimax_exploration_certificate` -- the van Trees score identity for a
    functional under an adaptive design and Fisher-information additivity are cited, not
    formalised, and exploitation is assumed to contribute no identifying information.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one round")
    caps = (
        np.full(horizon, float(cap))
        if isinstance(cap, int | float)
        else np.asarray(cap, dtype=np.float64)
    )
    if caps.shape != (horizon,):
        raise ValueError(
            f"a cap schedule needs one entry per round: got {caps.shape} for horizon {horizon}"
        )
    if np.isnan(caps).any() or (budget is not None and np.isnan(budget)):
        raise ValueError("the caps and the budget must be numbers, not nan")
    if caps.min() < 0.0:
        raise ValueError("a cap bounds the action, so it cannot be negative")
    if caps.max() <= 0.0:
        raise ValueError("cap must be positive: a zero cap forbids exploration entirely")
    if budget is not None and budget <= 0.0:
        raise ValueError("budget must be positive: a zero budget forbids exploration entirely")
    curvature = b * b + rr
    sensitivity = -xt * (rr - b * b) / (rr + b * b) ** 2
    numerator = curvature * sensitivity * sensitivity
    info_rate = eta / sigma**2
    c_causal = 2.0 * curvature * abs(sensitivity) * sigma / np.sqrt(eta)
    floor = c_causal * np.sqrt(horizon) - curvature * i0 / info_rate

    predicted_mass = _self_consistent_mass(
        caps,
        horizon=horizon,
        curvature=curvature,
        numerator=numerator,
        info_rate=info_rate,
        prior_info=i0,
    )
    fill = _greedy_fill(caps, budget)
    constant = budget is None and float(caps.min()) == float(caps.max())

    if constant:
        # The information grows by a constant step, so the harmonic sum is a difference of
        # digammas and the search costs O(1) per candidate -- which is what lets the certificate
        # run to T = 1e7. This path is Result 56's and is left numerically untouched.
        level = float(caps[0])

        def cost_of(rounds: NDArray[np.float64]) -> NDArray[np.float64]:
            return _capped_block_cost(
                rounds,
                cap=level,
                curvature=curvature,
                numerator=numerator,
                info_rate=info_rate,
                prior_info=i0,
                horizon=horizon,
            )

        def block(rounds: int) -> tuple[float, float]:
            return level * rounds, float(cost_of(np.array([float(rounds)]))[0])

        # Convex in the block length, so the first length whose next round no longer pays is the
        # best, found by bisection on the sign of that step from zero rounds -- a cap above the
        # whole optimal mass may not be worth one round. A geometric bracket with a sweep of 64
        # rounds around it missed the minimum wherever the bracket's spacing was wider: 90932
        # rounds against 95108 at T = 1e6, cap = 0.002, 0.60 dearer.
        lo, hi = 0, horizon
        while lo < hi:
            mid = (lo + hi) // 2
            step = cost_of(np.array([mid + 1.0, float(mid)]))
            if step[0] < step[1]:
                lo = mid + 1
            else:
                hi = mid
        best = lo
        crumb = 0.0
    else:
        curve = _scheduled_block_costs(
            fill,
            curvature=curvature,
            numerator=numerator,
            info_rate=info_rate,
            prior_info=i0,
        )
        # Once the budget is spent the curve is EXACTLY flat -- a round with no exploration adds
        # one term to the estimation sum and removes one from the exploitation tail at the same
        # information -- so argmin lands wherever float noise puts it. Trim to the last round that
        # actually explores, which is what "explore for n rounds then stop" means. The threshold is
        # not cosmetic: a prefix sum that reaches the budget carries O(eps*T) of rounding, so the
        # round after the last real one is filled with a few ulps rather than a clean zero.
        best = int(np.argmin(curve))
        crumb = np.finfo(np.float64).eps * caps.size * float(np.sum(fill[:best]))
        used = np.flatnonzero(fill[:best] > crumb)
        best = int(used[-1]) + 1 if used.size else 0
        delivered = np.concatenate([[0.0], np.cumsum(fill)])

        def block(rounds: int) -> tuple[float, float]:
            return float(delivered[rounds]), float(curve[rounds])

    # The objective is convex in the mass the greedy fill delivers (each prefix sum is concave in
    # it), so its minimum lies within one exploring round of the best whole block, and on that
    # round the amount to explore is a closed form: A x + K (T - n - 1)/(w + c x) is least at
    # x = sqrt(K (T - n - 1)/(A c)) - w/c. A whole block alone missed this one partial round.
    saturated, partial = best, 0.0
    mass, cost = block(best)
    ahead = np.flatnonzero(fill[best:] > crumb)  # the next round that can explore, past idle ones
    for rounds in (best - 1, best + int(ahead[0]) if ahead.size else horizon):
        if not 0 <= rounds < horizon:
            continue
        base, base_cost = block(rounds)
        info = i0 + info_rate * base
        rest = horizon - rounds - 1
        reach = float(np.sqrt(numerator * rest / (curvature * info_rate))) - info / info_rate
        amount = min(max(reach, 0.0), float(fill[rounds]))
        if amount <= crumb:
            continue
        total = (
            base_cost
            + curvature * amount
            + numerator * rest * (1.0 / (info + info_rate * amount) - 1.0 / info)
        )
        if total < cost:
            saturated, partial, mass, cost = rounds, amount, base + amount, total
    best = saturated + 1 if partial > 0.0 else saturated
    last_round = partial if partial > 0.0 else float(fill[best - 1]) if best > 0 else 0.0
    effective = mass / best if best > 0 and not constant else float(caps[0])

    scale = sigma * np.sqrt(numerator / (2.0 * curvature * eta))
    steps = np.arange(1, horizon + 1, dtype=np.float64)
    taper = _greedy_fill(np.minimum(scale / np.sqrt(steps), caps), budget)
    info = i0 + info_rate * np.concatenate([[0.0], np.cumsum(taper)[:-1]])
    taper_cost = float(np.sum(curvature * taper + numerator / info))

    return CappedExplorationPolicy(
        cap=effective,
        block_rounds=best,
        cost=cost,
        uncapped_floor=float(floor),
        taper_cost=taper_cost,
        excess=cost - float(floor),
        predicted_excess=float(numerator * np.log(horizon) / (2.0 * info_rate * effective)),
        exploration_mass=mass,
        predicted_mass=predicted_mass,
        budget_binds=budget is not None and mass >= budget - 1e-9 * max(1.0, budget),
        last_round=last_round,
    )


def minimax_exploration_certificate(
    *,
    b: float = 1.0,
    rr: float = 0.5,
    xt: float = 1.0,
    sigma: float = 0.7,
    i0: float = 1.0,
    eta: float = 0.6,
    constant_v: float = 0.01,
    horizons: Sequence[int] = (10**3, 10**4, 10**5, 10**6, 10**7),
    eta_grid: Sequence[float] = (0.1, 0.2, 0.4, 0.7, 1.0),
) -> MinimaxExplorationCurve:
    """CONTRIBUTION 3 -- the FULL LOCAL-MINIMAX sequential lower bound with an EXPLICIT constant
    (derived in ``validation/minimax_exploration.mac``, proved in ``proofs/minimax_exploration.v``).

    :func:`adaptive_exploration_certificate` gave a *sequence* bound: over exploration schedules,
    inside an assumed per-round decomposition, with an opaque floor numerator ``K``. This closes
    both gaps. Splitting any policy's action into its conditional mean and conditional variance is
    an identity (Rocq ``action_variance_decomposition``), so the ``inf`` runs over ALL policies, not
    schedules; and van Trees on the *functional* ``u*(theta)`` identifies the numerator as
    ``K = A*(du*/db)^2*sigma^2``. The result is

        ``inf_pi sup_theta E R_T >= c_causal*sqrt(T) - O(1)``,
        ``c_causal = 2*A*|du*/db|*sigma/sqrt(eta)``,

    with ``A`` the local cost curvature in the action, ``du*/db`` the decision sensitivity of the
    oracle action, ``sigma^2`` the noise, and ``eta`` in (0,1] the identifying information per unit
    injected exploration variance. The ``sup`` is over a shrinking neighbourhood of radius
    ``H*T^(-1/4)`` -- *not* the usual ``T^(-1/2)``, since the optimiser sits at order ``sqrt(T)``
    and a ``T^(-1/2)`` prior would swamp the data information -- with the limits taken ``T -> inf``
    first, then ``H -> inf``, as in the clustered result.

    Two things this certificate shows that the rate alone does not. The floor is **attained**:
    the crude step (every denominator at the total budget) is an equality exactly when information
    is raised immediately, so a front-loaded burst hits it and ``burst_over_floor -> 1`` -- the
    constant is sharp. And **tapering costs a constant factor**: the ``1/sqrt(t)`` schedule, even at
    its own optimal scale, sits at ``sqrt(2)`` times the floor (the two constants are Maxima's;
    Rocq ``taper_gap_is_sqrt_two`` checks that their squared ratio is 2). The
    burst is optimal *in this model*, where exploration cost is linear in the injected variance and
    unbounded per round. A per-round action cap does NOT rescue the taper -- Result 56 shows the
    capped optimum is a cap-saturating BLOCK, still front-loaded, whose price over this floor is
    an additive logarithm; see :func:`capped_exploration_policy`.

    SCOPE, as elsewhere in this line: Rocq proves the algebra. The van Trees score identity for a
    functional under an adaptive design, and Fisher-information additivity along the trajectory, are
    cited (Gill-Levit 1995; Gassiat-Stoltz 2024), not formalised. The model assumes exploitation
    contributes no identifying information -- that is what ``eta`` encodes.
    """
    a_curv = b * b + rr
    gp = -xt * (rr - b * b) / (rr + b * b) ** 2  # du*/db for u*(b) = -b*xt/(b^2+rr)
    k_id = a_curv * gp * gp  # regret numerator A*(du*/db)^2; the van-Trees floor carries sigma^2
    c_causal = 2.0 * a_curv * abs(gp) * sigma / np.sqrt(eta)

    def cumulative(sched: np.ndarray) -> float:
        info = i0 + (eta / sigma**2) * np.concatenate([[0.0], np.cumsum(sched)[:-1]])
        return float(np.sum(a_curv * sched + k_id / info))

    hs = np.asarray(horizons, dtype=np.float64)
    floor = c_causal * np.sqrt(hs) - (a_curv * sigma**2 / eta) * i0
    kappa = sigma * np.sqrt(k_id / (2.0 * a_curv * eta))  # optimal scale for the 1/sqrt(t) family
    burst = np.zeros(hs.size)
    taper = np.zeros(hs.size)
    greedy = np.zeros(hs.size)
    constant = np.zeros(hs.size)
    for i, horizon in enumerate(horizons):
        t = np.arange(1, horizon + 1, dtype=np.float64)
        x_star = abs(gp) * np.sqrt(horizon * eta) / sigma  # optimal final information level
        one_shot = np.zeros(horizon)
        one_shot[0] = max(0.0, (sigma**2 / eta) * (x_star - i0))
        burst[i] = cumulative(one_shot)
        taper[i] = cumulative(kappa / np.sqrt(t))
        greedy[i] = cumulative(np.zeros(horizon))
        constant[i] = cumulative(np.full(horizon, constant_v))

    ratios = np.concatenate([burst / floor, taper / floor, greedy / floor, constant / floor])
    etas = np.asarray(eta_grid, dtype=np.float64)
    consts = 2.0 * a_curv * abs(gp) * sigma / np.sqrt(etas)
    return MinimaxExplorationCurve(
        hs,
        floor,
        burst,
        taper,
        greedy,
        constant,
        float(c_causal),
        float(np.min(ratios)),
        float(burst[-1] / floor[-1]),
        float(taper[-1] / floor[-1]),
        float(np.polyfit(np.log(etas), np.log(consts), 1)[0]),
    )


# --- Result 58: the minimax LQ controller over the identified effect interval ---


MinimaxBranch = Literal["lower", "upper", "equalise", "zero", "interior"]
"""Which candidate the robust action landed on. ``lower`` / ``upper`` are the certainty-equivalent
actions for an interval *endpoint*, ``equalise`` the action that makes both endpoints equally bad
(``b_hat*u = target`` for the cost, the ``sqrt(D)``-weighted mean of the endpoint actions for the
regret), ``zero`` doing nothing -- the answer when the sign is unidentified -- and ``interior`` a
regret whose worst effect lies inside the interval."""

MinimaxCriterion = Literal["cost", "regret"]


@dataclass(frozen=True)
class MinimaxAction:
    """The robust action, its worst case over the interval, and which candidate bound.

    ``worst_case`` is the worst of whatever was minimised: the cost, or the regret.
    """

    action: float
    worst_case: float
    binding: MinimaxBranch


def minimax_action(
    target: float,
    b_lo: float,
    b_hi: float,
    effort: float,
    curvature: float = 1.0,
    *,
    criterion: MinimaxCriterion = "cost",
) -> MinimaxAction:
    """Exact ``argmin_u max_{b in [b_lo, b_hi]} curvature*(b*u - target)^2 + effort*u^2``.

    §33 showed by counterexample that certainty equivalence is *not* minimax for the LQ loss and
    left the robust controller unbuilt; this is it, in closed form rather than by search
    (``validation/minimax_lq.mac``, ``proofs/minimax_lq.v``).

    Two facts make it closed-form. The stage cost is convex in ``b`` with curvature
    ``2*curvature*u^2 >= 0``, so the inner maximum is attained at an **endpoint** and the outer
    problem is the minimisation of a max of two convex quadratics. Those two branches cross at
    exactly two points, ``u = 0`` and the equalising action ``u = target/b_hat`` -- *both* of them,
    which is why the candidate set is enumerated rather than reasoned about by the sign of ``u``.
    A convex piecewise quadratic attains its minimum at a branch minimiser or at a kink, so
    evaluating the four candidates is exact, not a heuristic.

    On an interval of one sign the choice is decided by ``curvature*|b_0|*(b_hi - b_lo)/2 <=
    effort``, ``b_0`` the endpoint nearest zero -- ``b_lo`` on a positive interval, ``b_hi`` on a
    negative one, for a target of either sign: below it the answer is the CE action for that
    *pessimistic* endpoint, above it the equalising action. ``proofs/minimax_lq.v`` proves the rule
    for ``0 <= u`` and ``b_hi u <= target``; the rest is the symmetry ``(b, u) -> (-b, -u)``, and an
    interval containing zero is left to the enumeration. And the correction
    does NOT have a fixed sign -- the robust action is smaller than CE at ``b_hat`` only when
    ``curvature*b_lo*b_hat < effort``, and larger otherwise. "Be robust, act less" is a statement
    about expensive effort, not about pessimism.

    ``criterion="regret"`` minimises the worst *regret* instead: the cost over what the action
    that knew ``b`` would have paid, ``D(b) (u - u*(b))^2`` with ``D(b) = curvature*b^2 + effort``
    and ``u*(b) = curvature*b*target/D(b)``. The two answers differ: on the five intervals of its
    test the minimax-cost action carries 4.1-6.2x the minimax regret. The regret's worst
    effect is an endpoint unless ``sqrt(D) (u - u*)`` turns inside the interval, which happens
    exactly where ``u*b*D(b) - target*effort`` changes sign on it -- that root test decides the
    case, and a monotone ``u*`` does not: on ``b in [1, 10]`` with unit weights the endpoint answer
    is 20% low. With no turn, the answer is the endpoint equaliser, ``(sqrt(D_lo)*u*_lo +
    sqrt(D_hi)*u*_hi) / (sqrt(D_lo) + sqrt(D_hi))``; with one, the worst regret is evaluated at
    the endpoints and the turn, is convex in ``u``, and is minimised over the range of ``u*`` by
    golden section.
    """
    for name, value in (("target", target), ("b_lo", b_lo), ("b_hi", b_hi)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    if b_hi < b_lo:
        raise ValueError(f"empty identified interval: b_lo={b_lo} > b_hi={b_hi}")
    if not 0.0 < effort < np.inf:
        raise ValueError(f"effort must be positive and finite to bound the action; got {effort}")
    if not 0.0 <= curvature < np.inf:
        raise ValueError(
            f"curvature is a cost-to-go coefficient and cannot be negative, infinite or nan: "
            f"{curvature}"
        )
    if criterion == "regret":
        return _minimax_regret_action(target, b_lo, b_hi, effort, curvature)
    if criterion != "cost":
        raise ValueError(f"criterion must be 'cost' or 'regret', got {criterion!r}")

    def branch(b: float, u: float) -> float:
        # products, not ``**``: a float power past the float range raises where these read inf
        miss = b * u - target
        return curvature * miss * miss + effort * u * u

    def worst(u: float) -> float:
        return max(branch(b_lo, u), branch(b_hi, u))

    def endpoint_action(b: float) -> float:
        return curvature * b * target / (curvature * b**2 + effort)

    midpoint = 0.5 * (b_lo + b_hi)
    candidates: list[tuple[float, MinimaxBranch]] = [
        (endpoint_action(b_lo), "lower"),
        (endpoint_action(b_hi), "upper"),
        (0.0, "zero"),
    ]
    if midpoint != 0.0:
        candidates.append((target / midpoint, "equalise"))
    action, binding = min(candidates, key=lambda c: worst(c[0]))
    return MinimaxAction(action=action, worst_case=worst(action), binding=binding)


def _minimax_regret_action(
    target: float, b_lo: float, b_hi: float, effort: float, curvature: float
) -> MinimaxAction:
    """``minimax_action(..., criterion="regret")``; see there for the two cases."""

    def spread(b: float) -> float:
        return curvature * b**2 + effort

    def best(b: float) -> float:
        return curvature * b * target / spread(b)

    def regret(u: float, b: float) -> float:
        gap = u - best(b)
        return spread(b) * gap * gap

    def turn(u: float) -> float | None:
        """Where ``sqrt(D) (u - u*)`` turns inside the interval; it turns at most once."""

        def slope(b: float) -> float:
            return u * b * spread(b) - target * effort

        if slope(b_lo) * slope(b_hi) >= 0.0:
            return None
        return float(brentq(slope, b_lo, b_hi, xtol=1e-15, rtol=4.0 * np.finfo(float).eps))

    def worst(u: float) -> float:
        inside = turn(u)
        edges = max(regret(u, b_lo), regret(u, b_hi))
        return edges if inside is None else max(edges, regret(u, inside))

    root_lo, root_hi = np.sqrt(spread(b_lo)), np.sqrt(spread(b_hi))
    equaliser = float((root_lo * best(b_lo) + root_hi * best(b_hi)) / (root_lo + root_hi))
    if turn(equaliser) is None:
        return MinimaxAction(
            action=equaliser, worst_case=float(worst(equaliser)), binding="equalise"
        )

    # ``u*`` peaks where ``curvature*b^2 = effort``; past the range of ``u*`` every regret grows.
    reach = [best(b_lo), best(b_hi)]
    peak = np.sqrt(effort / curvature)
    reach += [best(b) for b in (peak, -peak) if b_lo < b < b_hi]
    lo, hi = min(reach), max(reach)
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    worst_left, worst_right = worst(left), worst(right)
    while hi - lo > 4.0 * np.finfo(float).eps * max(1.0, abs(lo), abs(hi)):
        if worst_left <= worst_right:
            hi, right, worst_right = right, left, worst_left
            left = hi - ratio * (hi - lo)
            worst_left = worst(left)
        else:
            lo, left, worst_left = left, right, worst_right
            right = lo + ratio * (hi - lo)
            worst_right = worst(right)
    action = float(0.5 * (lo + hi))
    return MinimaxAction(action=action, worst_case=float(worst(action)), binding="interior")


MoveRule = Literal["sign", "linear"]
"""Which minimax-regret rule :func:`minimax_regret_move` took: ``sign`` makes the whole move where
its estimate is not negative and holds otherwise; ``linear`` makes a share of it that rises
linearly from 0 to 1 across ``[-threshold, threshold]``."""


@dataclass(frozen=True)
class MinimaxRegretMove:
    """How much of a move to make where its value is known only to an interval around a noisy
    estimate, and what that costs at worst. *Experimental.*

    See :func:`minimax_regret_move`.
    """

    # The share of the move to make, in [0, 1]: 1 the whole move, 0 holding. A share in between is
    # the probability of making the move, or, where the move's value is linear in its size, the
    # fraction of it to make, which pays the same in expectation.
    share: float
    rule: MoveRule
    threshold: float  # rho: the half-width of the band the share rises across; 0 under ``sign``
    halfwidth: float  # k = (gamma - 1)/(gamma + 1) * cvar_gap * size: how far the value may sit
    worst_regret: float  # the rule's worst regret over every value the data allow: the minimax one
    sign_regret: float  # the sign rule's worst regret, which is ``worst_regret`` under ``sign``
    # The largest Gamma at which this estimate still makes the whole move, sampling noise included:
    # inf where no Gamma takes it away, nan where no Gamma makes it, as where the estimate is
    # negative. With no noise it is the flip (G + d)/(G - d) of the identified set alone; noise
    # moves it up.
    full_move_gamma: float


def minimax_regret_move(
    estimate: float,
    standard_error: float,
    gamma: float,
    cvar_gap: float = 1.0,
    size: float = 1.0,
) -> MinimaxRegretMove:
    """How much of a move to make when its value is linear in the channel, its estimate is noisy,
    and confounding up to ``gamma`` may bias it: the minimax-regret rule of Montiel Olea, Qiu and
    Stoye (Review of Economic Studies 2026; arXiv 2312.17623 v3, Section 4.2). *Experimental.*

    The move's estimated value is ``estimate ~ N(beta, standard_error^2)``, the standard error its
    influence's, and its true value lies in ``[beta - k, beta + k]``, ``k = (gamma - 1)/(gamma + 1)
    * cvar_gap * size``: the marginal sensitivity model's half-width, with the gap a share of
    ``size``, the channel's size on the move's scale, as ADR 0085 sizes it; ``size = 1`` reads
    ``cvar_gap`` in the move's own units. A rule makes the move with probability ``d``. Where the
    move's value is ``u > 0`` its regret is ``u (1 - E d)``, and where ``u < 0`` it is ``-u E d``.

    * ``k <= sqrt(pi/2) standard_error``: the sign rule, the whole move where the estimate is not
      negative, is minimax regret, and essentially the only rule that is. Its worst regret is
      ``max over beta of (beta + k) Phi(-beta / standard_error)``.
    * Past it every minimax-regret rule randomises, the minimax regret is ``k / 2``, and the rule
      that randomises least makes ``clip((estimate + rho) / (2 rho), 0, 1)`` of the move, with
      ``rho = k (1 - 2 Phi(-rho / standard_error))``, the one root in ``(0, k]``. The sign rule
      loses more there: 1.0509 against 1.000 at ``k = 2``, ``standard_error = 1``, and 3.371
      against 2.5 at ``k = 5`` (``validation/minimax_regret_move.mac``).

    :attr:`MinimaxRegretMove.full_move_gamma` is the largest ``gamma`` at which the estimate still
    makes the whole move. The share is 1 while ``rho <= estimate``, and ``rho`` grows with ``k``,
    so the move is whole up to ``k = estimate / erf(estimate / (standard_error sqrt 2))``, which
    exceeds ``estimate``. So the move stays whole past the flip ``(G + d)/(G - d)`` of the
    noiseless identified set, where the set already holds values below zero: the noise is a regret
    the confounding's hedge cannot buy back, and below ``sqrt(pi/2) standard_error`` the sign rule
    moves whole on any estimate that is not negative. The Gamma falls to the flip as the error
    shrinks.

    For linear welfare only: a value linear in the move, an estimate Gaussian about its mean, and a
    half-width that does not depend on the data. A cost quadratic in the action, as
    :func:`minimax_action`'s, has another minimax rule, which this one does not approximate.

    Raises:
        ValueError: on an estimate that is not finite, a standard error that is negative or not
            finite, a ``gamma`` below 1 or not finite, or a ``cvar_gap`` or ``size`` that is
            negative or not finite.
    """
    if not np.isfinite(estimate):
        raise ValueError(f"estimate must be finite, got {estimate}")
    if not 0.0 <= standard_error < np.inf:
        raise ValueError(f"standard_error must be nonnegative and finite, got {standard_error}")
    if not 1.0 <= gamma < np.inf:
        raise ValueError(f"gamma is a sensitivity level, at least 1 and finite, got {gamma}")
    for name, value in (("cvar_gap", cvar_gap), ("size", size)):
        if not 0.0 <= value < np.inf:
            raise ValueError(f"{name} must be nonnegative and finite, got {value}")
    gap = float(cvar_gap) * float(size)
    k = (float(gamma) - 1.0) / (float(gamma) + 1.0) * gap
    sigma = float(standard_error)
    sign_regret = _sign_rule_regret(k, sigma)
    full = _full_move_gamma(float(estimate), sigma, gap)
    # the threshold read as the inverse's bracket reads it, so the two agree at the boundary
    if k == 0.0 if sigma == 0.0 else k / sigma <= _whole_move(0.0):
        return MinimaxRegretMove(
            share=1.0 if estimate >= 0.0 else 0.0,
            rule="sign",
            threshold=0.0,
            halfwidth=k,
            worst_regret=sign_regret,
            sign_regret=sign_regret,
            full_move_gamma=full,
        )
    rho = k if sigma == 0.0 else sigma * _whole_move_inverse(k / sigma)
    return MinimaxRegretMove(
        share=float(np.clip((estimate + rho) / (2.0 * rho), 0.0, 1.0)),
        rule="linear",
        threshold=rho,
        halfwidth=k,
        worst_regret=k / 2.0,
        sign_regret=sign_regret,
        full_move_gamma=full,
    )


def _whole_move(t: float) -> float:
    """``t / erf(t / sqrt 2)``: the half-width, in standard errors, at which the least-randomising
    rule's threshold is ``t`` standard errors. Rising from ``sqrt(pi/2)`` at ``t = 0``."""
    return float(np.sqrt(np.pi / 2.0)) if t == 0.0 else t / float(erf(t / np.sqrt(2.0)))


def _whole_move_inverse(scale: float) -> float:
    """The threshold, in standard errors, at a half-width of ``scale > sqrt(pi/2)`` of them: the
    root of :func:`_whole_move`, which rises from ``sqrt(pi/2)`` and is at least ``t``, so the root
    lies in ``(0, scale]``. Read through ``t / erf``, not ``t - scale erf``, whose two terms cancel
    near the threshold."""
    return float(
        brentq(
            lambda t: _whole_move(t) - scale,
            0.0,
            scale,
            xtol=float(np.finfo(float).tiny),
            rtol=4.0 * np.finfo(float).eps,
        )
    )


def _sign_rule_regret(k: float, sigma: float) -> float:
    """The sign rule's worst regret, ``max over beta of (beta + k) Phi(-beta / sigma)``, the larger
    branch by symmetry. In ``z = beta / sigma`` the product is log-concave on ``z > -k / sigma``, so
    its slope ``Phi(-z) - (z + k / sigma) phi(z)`` changes sign once: it is ``Phi(k / sigma)``,
    positive, at ``z = -k / sigma``, and negative at ``z = 1``, where ``Phi(-1) < phi(1)``."""
    if sigma == 0.0:
        return k  # the supremum as beta rises to 0 from below, where the rule holds
    scale = k / sigma

    def slope(z: float) -> float:
        return float(ndtr(-z)) - (z + scale) * float(np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi))

    z = float(brentq(slope, -scale, 1.0, xtol=1e-15, rtol=4.0 * np.finfo(float).eps))
    return float(sigma * (z + scale) * ndtr(-z))


def _full_move_gamma(estimate: float, sigma: float, gap: float) -> float:
    """The largest Gamma at which ``estimate`` makes the whole move: the half-width it stays whole
    up to, ``sigma`` times :func:`_whole_move` of it in standard errors, inverted through
    ``(gamma - 1)/(gamma + 1) gap``."""
    if estimate < 0.0:
        return float("nan")
    if sigma == 0.0:
        # with no noise the threshold is k itself, and the share 1/2 at an estimate of 0 once k > 0
        if estimate == 0.0:
            return float("inf") if gap == 0.0 else 1.0
        reach = estimate
    else:
        # the threshold is at least the estimate, t / erf >= t; once erf rounds to 1,
        # sigma (estimate / sigma) can land an ulp below it, and the Gamma below the flip
        reach = max(estimate, sigma * _whole_move(estimate / sigma))
    if reach >= gap:
        return float("inf")
    return float((gap + reach) / (gap - reach))


@dataclass(frozen=True)
class MinimaxLQPolicy:
    """Robust finite-horizon feedback over a partially identified effect, beside its CE rival."""

    gains: NDArray[np.float64]  # u_k = -gains[k]*x_k
    value: float  # worst-case cost coefficient: J <= value*x0^2
    binding: tuple[MinimaxBranch, ...]
    ce_gains: NDArray[np.float64]
    ce_value: float  # the CE policy's worst case over the SAME interval


def _worst_case_of_gains(
    gains: NDArray[np.float64],
    state_gain: float,
    b_lo: float,
    b_hi: float,
    state_cost: float,
    effort: float,
    terminal_cost: float,
) -> float:
    """Worst case of a FIXED linear policy against a per-step adversarial effect."""
    p = terminal_cost
    for gain in reversed(gains.tolist()):
        s = -gain
        step = max((state_gain + b_lo * s) ** 2, (state_gain + b_hi * s) ** 2)
        p = state_cost + effort * s**2 + p * step
    return p


def minimax_lq_policy(
    state_gain: float,
    b_lo: float,
    b_hi: float,
    state_cost: float,
    effort: float,
    terminal_cost: float,
    horizon: int,
) -> MinimaxLQPolicy:
    """Robust Riccati recursion for ``x' = a*x + b*u`` with ``b`` only known to lie in an interval.

    The Bellman step is homogeneous of degree two in ``(x, u)``, so substituting ``u = s*x`` turns
    it into exactly the static problem of :func:`minimax_action` with ``target = -a`` and
    ``curvature = p_{k+1}`` (identity (8) of ``validation/minimax_lq.mac``). The robust policy is
    therefore linear feedback with a closed form per step, not a minimax search, and the value
    function stays quadratic.

    The adversary re-picks ``b`` at **every step**, which is what makes dynamic programming apply
    and is normally a strict relaxation of a constant unknown effect. Here it is not, and that is a
    result rather than a caveat: identity (9) shows the robust action always leaves the endpoint
    nearest zero (weakly) worst, decided by the same kink rule that picked the branch, so the
    per-step optimum is realised by that constant effect -- ``b_lo`` on a positive interval,
    ``b_hi`` on a negative one (``a = 1``, ``b`` in ``[-2, -1]``, ``q = r = 1``, ``T = 2``: value
    1.5, which ``b_hi`` costs and ``b_lo`` does not, 1.25), and both endpoints on an interval
    containing zero. The value is the constant-effect worst case rather than an upper bound on it,
    and :func:`minimax_lq_certificate` measures that instead of assuming it.

    HONEST SCOPE. Scalar state and scalar effect: the interval is the §32 sensitivity interval for
    one control channel, and the multivariate lift is not this function.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least one step; got {horizon}")
    for name, value in (
        ("state_gain", state_gain),
        ("state_cost", state_cost),
        ("terminal_cost", terminal_cost),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    b_hat = 0.5 * (b_lo + b_hi)

    p = terminal_cost
    gains: list[float] = []
    binding: list[MinimaxBranch] = []
    for _ in range(horizon):
        step = minimax_action(-state_gain, b_lo, b_hi, effort, curvature=p)
        gains.append(-step.action)
        binding.append(step.binding)
        p = state_cost + step.worst_case

    p_ce = terminal_cost
    ce_gains: list[float] = []
    for _ in range(horizon):
        gain = b_hat * p_ce * state_gain / (effort + b_hat**2 * p_ce)
        ce_gains.append(gain)
        p_ce = state_cost + state_gain**2 * p_ce - state_gain * p_ce * b_hat * gain

    robust = np.asarray(gains[::-1], dtype=float)
    equivalent = np.asarray(ce_gains[::-1], dtype=float)
    return MinimaxLQPolicy(
        gains=robust,
        value=p,
        binding=tuple(binding[::-1]),
        ce_gains=equivalent,
        ce_value=_worst_case_of_gains(
            equivalent, state_gain, b_lo, b_hi, state_cost, effort, terminal_cost
        ),
    )


def _closed_loop_cost(
    gains: NDArray[np.float64],
    state_gain: float,
    effect: float,
    state_cost: float,
    effort: float,
    terminal_cost: float,
) -> float:
    """Realised cost of ``u_k = -gains[k]*x_k`` from ``x_0 = 1`` under one CONSTANT true effect."""
    x, total = 1.0, 0.0
    for gain in gains.tolist():
        u = -gain * x
        total += state_cost * x**2 + effort * u**2
        x = state_gain * x + effect * u
    return total + terminal_cost * x**2


@dataclass(frozen=True)
class MinimaxLQCertificate:
    """Evidence that the closed form is the minimiser, beats CE, and how much the DP relaxes."""

    static_action: float
    static_value: float
    static_ce_worst: float
    grid_action: float  # brute-force argmin over a fine grid -- must reproduce static_action
    halfwidths: tuple[float, ...]
    values: tuple[float, ...]
    ce_values: tuple[float, ...]
    constant_effect_worst: float  # worst cost of the robust policy at a CONSTANT unknown effect
    worst_constant_effect: float  # the effect attaining it -- b_lo, by identity (9)
    relaxation_ratio: float  # value / constant_effect_worst; 1 means the per-step adversary is free
    lower_endpoint_margin: float  # min over steps of the normalised (lower - upper) endpoint gap
    beats_ce: bool
    monotone: bool
    ok: bool


def minimax_lq_certificate(
    state_gain: float = 1.0,
    b_hat: float = 1.0,
    state_cost: float = 1.0,
    effort: float = 1.0,
    terminal_cost: float = 1.0,
    horizon: int = 5,
    halfwidths: Sequence[float] = (0.0, 0.1, 0.25, 0.5),
    grid: int = 200_001,
) -> MinimaxLQCertificate:
    """Check the closed form against brute force, against CE, and against its own DP relaxation.

    Three things, and the third is the one that could embarrass the entry. (i) On §33's own
    counterexample the closed form must return the action a fine grid search finds, and must beat
    the CE action's worst case -- §33 reported 0.4 against 0.5 and 0.8 against 0.8125. (ii) Over a
    sweep of interval half-widths the robust worst case must stay at or below CE's and grow with the
    width. (iii) The robust policy is derived against an adversary that re-picks the effect every
    step, which is normally a strict relaxation; ``relaxation_ratio`` measures the price of it
    against a grid of *constant* effects, and ``lower_endpoint_margin`` measures the structural
    reason the price is zero -- the lower endpoint is the worst one at every step.
    """
    b_lo, b_hi = b_hat - halfwidths[-1], b_hat + halfwidths[-1]

    static = minimax_action(1.0, 0.5, 1.5, 1.0)  # §33's instance, fixed: it is the published one
    span = np.linspace(-1.0, 1.0, grid)
    worst_on_grid = np.maximum((0.5 * span - 1.0) ** 2, (1.5 * span - 1.0) ** 2) + span**2
    grid_action = float(span[int(np.argmin(worst_on_grid))])
    ce_worst = max((0.5 * 0.5 - 1.0) ** 2, (1.5 * 0.5 - 1.0) ** 2) + 0.5**2

    policies = [
        minimax_lq_policy(
            state_gain, b_hat - d, b_hat + d, state_cost, effort, terminal_cost, horizon
        )
        for d in halfwidths
    ]
    values = tuple(p.value for p in policies)
    ce_values = tuple(p.ce_value for p in policies)

    widest = policies[-1]
    effects = np.linspace(b_lo, b_hi, 401)
    costs = [
        _closed_loop_cost(widest.gains, state_gain, float(b), state_cost, effort, terminal_cost)
        for b in effects
    ]
    constant_worst = max(costs)
    worst_effect = float(effects[int(np.argmax(costs))])

    actions = -widest.gains
    lower = (state_gain + b_lo * actions) ** 2
    upper = (state_gain + b_hi * actions) ** 2
    margin = float(np.min(lower - upper) / max(float(np.max(lower)), 1e-30))

    beats_ce = all(v <= c + 1e-12 for v, c in zip(values, ce_values, strict=True))
    monotone = all(values[i] <= values[i + 1] + 1e-12 for i in range(len(values) - 1))
    return MinimaxLQCertificate(
        static_action=static.action,
        static_value=static.worst_case,
        static_ce_worst=ce_worst,
        grid_action=grid_action,
        halfwidths=tuple(halfwidths),
        values=values,
        ce_values=ce_values,
        constant_effect_worst=constant_worst,
        worst_constant_effect=worst_effect,
        relaxation_ratio=widest.value / constant_worst,
        lower_endpoint_margin=margin,
        beats_ce=beats_ce,
        monotone=monotone,
        ok=(
            abs(static.action - grid_action) < 1e-4
            and static.worst_case < ce_worst
            and beats_ce
            and monotone
            and abs(widest.value / constant_worst - 1.0) < 1e-9
            and margin > -1e-9
            and abs(worst_effect - b_lo) < 1e-9
        ),
    )


# The research half of this module moved to chc.research.regret in 0.19.0. Each of its names
# is served from here with a DeprecationWarning until 0.20.0 (docs/adr/0117).
_MOVED = frozenset(
    {
        "asymmetric_control_improvement",
        "bandit_causal_certificate",
        "BanditCausalCurve",
        "causal_vs_predictive_certificate",
        "CausalControlCurve",
        "ce_explicit_constant_certificate",
        "CeExplicitConstantCurve",
        "certainty_equivalence_control",
        "cluster_fold_leakage_certificate",
        "clustered_lower_bound_certificate",
        "ClusteredLowerBoundCurve",
        "ClusterFoldLeakageCurve",
        "composition_transfer_certificate",
        "CompositionTransferCurve",
        "confounded_turnpike_certificate",
        "ConfoundedTurnpikeCurve",
        "confounding_regret_floor_certificate",
        "confounding_robust_control",
        "confounding_robust_control_benchmark",
        "confounding_robust_control_certificate",
        "confounding_robust_lq_regret",
        "confounding_robust_lq_regret_certificate",
        "confounding_robust_lq_regret_matrix",
        "confounding_robust_tracking_benchmark",
        "confounding_robust_tracking_loop",
        "ConfoundingRegretFloorCurve",
        "ConfoundingRobustControlCurve",
        "ConfoundingRobustLQRegretCurve",
        "conjugate_time_certificate",
        "ConjugateTimeCurve",
        "constrained_ce_regret_certificate",
        "ConstrainedRegretCurve",
        "cross_cluster_mixing_certificate",
        "CrossClusterMixingCurve",
        "delayed_network_certificate",
        "DelayedNetworkCurve",
        "doubly_robust_control_certificate",
        "DoublyRobustCurve",
        "dynamic_causal_regret_certificate",
        "DynamicCausalCurve",
        "DynamicConfoundingCurve",
        "end_to_end_c2_certificate",
        "EndToEndC2Curve",
        "ensemble_control_certificate",
        "EnsembleControlCurve",
        "exact_matrix_ratio_moment",
        "exact_ratio_moment",
        "exposure_map_certificate",
        "ExposureMapCurve",
        "finite_horizon_pl_certificate",
        "FiniteHorizonPLCurve",
        "fold_exactness_certificate",
        "fold_heuristic_certificate",
        "FoldDesign",
        "FoldExactnessCurve",
        "FoldHeuristicCurve",
        "highprob_regret_certificate",
        "HighProbRegretCurve",
        "hinf_robust_regret_certificate",
        "HInfRobustCurve",
        "information_lower_bound_certificate",
        "InformationLowerBoundCurve",
        "interference_convexity_certificate",
        "interference_orthogonal_certificate",
        "interference_regret_certificate",
        "InterferenceConvexityCurve",
        "InterferenceOrthogonalCurve",
        "lq_regret_sensitivity",
        "MarketplaceControlCurve",
        "matrix_ratio_certificate",
        "MatrixRatioAccuracy",
        "MatrixRatioStatus",
        "multichannel_control_certificate",
        "MultiChannelControlCurve",
        "multivariate_action_floor",
        "multivariate_interference_certificate",
        "multivariate_transfer_certificate",
        "multivariate_van_trees_certificate",
        "MultivariateActionFloor",
        "MultivariateInterferenceCurve",
        "MultivariateTransferCurve",
        "MultivariateVanTreesCurve",
        "nonlinear_regret_certificate",
        "NonlinearRegretCurve",
        "optimal_fold_partition",
        "orthogonal_control_certificate",
        "OrthogonalControlCurve",
        "panel_estimator_certificate",
        "PanelEstimatorGate",
        "partial_id_control_certificate",
        "PartialIdControlCurve",
        "pessimism_variance_certificate",
        "PessimismCurve",
        "regret_scaling",
        "RegretCurve",
        "space_time_fold_certificate",
        "SpaceTimeFoldCurve",
        "strong_convexity_regret_bound",
        "transportability_regret_certificate",
        "TransportabilityCurve",
        "van_trees_certificate",
        "VanTreesCurve",
        "worst_case_asymmetric_loss",
    }
)

if TYPE_CHECKING:
    # isort: off
    from chc.research.regret import (
        asymmetric_control_improvement as asymmetric_control_improvement,
        bandit_causal_certificate as bandit_causal_certificate,
        BanditCausalCurve as BanditCausalCurve,
        causal_vs_predictive_certificate as causal_vs_predictive_certificate,
        CausalControlCurve as CausalControlCurve,
        ce_explicit_constant_certificate as ce_explicit_constant_certificate,
        CeExplicitConstantCurve as CeExplicitConstantCurve,
        certainty_equivalence_control as certainty_equivalence_control,
        cluster_fold_leakage_certificate as cluster_fold_leakage_certificate,
        clustered_lower_bound_certificate as clustered_lower_bound_certificate,
        ClusteredLowerBoundCurve as ClusteredLowerBoundCurve,
        ClusterFoldLeakageCurve as ClusterFoldLeakageCurve,
        composition_transfer_certificate as composition_transfer_certificate,
        CompositionTransferCurve as CompositionTransferCurve,
        confounded_turnpike_certificate as confounded_turnpike_certificate,
        ConfoundedTurnpikeCurve as ConfoundedTurnpikeCurve,
        confounding_regret_floor_certificate as confounding_regret_floor_certificate,
        confounding_robust_control as confounding_robust_control,
        confounding_robust_control_benchmark as confounding_robust_control_benchmark,
        confounding_robust_control_certificate as confounding_robust_control_certificate,
        confounding_robust_lq_regret as confounding_robust_lq_regret,
        confounding_robust_lq_regret_certificate as confounding_robust_lq_regret_certificate,
        confounding_robust_lq_regret_matrix as confounding_robust_lq_regret_matrix,
        confounding_robust_tracking_benchmark as confounding_robust_tracking_benchmark,
        confounding_robust_tracking_loop as confounding_robust_tracking_loop,
        ConfoundingRegretFloorCurve as ConfoundingRegretFloorCurve,
        ConfoundingRobustControlCurve as ConfoundingRobustControlCurve,
        ConfoundingRobustLQRegretCurve as ConfoundingRobustLQRegretCurve,
        conjugate_time_certificate as conjugate_time_certificate,
        ConjugateTimeCurve as ConjugateTimeCurve,
        constrained_ce_regret_certificate as constrained_ce_regret_certificate,
        ConstrainedRegretCurve as ConstrainedRegretCurve,
        cross_cluster_mixing_certificate as cross_cluster_mixing_certificate,
        CrossClusterMixingCurve as CrossClusterMixingCurve,
        delayed_network_certificate as delayed_network_certificate,
        DelayedNetworkCurve as DelayedNetworkCurve,
        doubly_robust_control_certificate as doubly_robust_control_certificate,
        DoublyRobustCurve as DoublyRobustCurve,
        dynamic_causal_regret_certificate as dynamic_causal_regret_certificate,
        DynamicCausalCurve as DynamicCausalCurve,
        DynamicConfoundingCurve as DynamicConfoundingCurve,
        end_to_end_c2_certificate as end_to_end_c2_certificate,
        EndToEndC2Curve as EndToEndC2Curve,
        ensemble_control_certificate as ensemble_control_certificate,
        EnsembleControlCurve as EnsembleControlCurve,
        exact_matrix_ratio_moment as exact_matrix_ratio_moment,
        exact_ratio_moment as exact_ratio_moment,
        exposure_map_certificate as exposure_map_certificate,
        ExposureMapCurve as ExposureMapCurve,
        finite_horizon_pl_certificate as finite_horizon_pl_certificate,
        FiniteHorizonPLCurve as FiniteHorizonPLCurve,
        fold_exactness_certificate as fold_exactness_certificate,
        fold_heuristic_certificate as fold_heuristic_certificate,
        FoldDesign as FoldDesign,
        FoldExactnessCurve as FoldExactnessCurve,
        FoldHeuristicCurve as FoldHeuristicCurve,
        highprob_regret_certificate as highprob_regret_certificate,
        HighProbRegretCurve as HighProbRegretCurve,
        hinf_robust_regret_certificate as hinf_robust_regret_certificate,
        HInfRobustCurve as HInfRobustCurve,
        information_lower_bound_certificate as information_lower_bound_certificate,
        InformationLowerBoundCurve as InformationLowerBoundCurve,
        interference_convexity_certificate as interference_convexity_certificate,
        interference_orthogonal_certificate as interference_orthogonal_certificate,
        interference_regret_certificate as interference_regret_certificate,
        InterferenceConvexityCurve as InterferenceConvexityCurve,
        InterferenceOrthogonalCurve as InterferenceOrthogonalCurve,
        lq_regret_sensitivity as lq_regret_sensitivity,
        MarketplaceControlCurve as MarketplaceControlCurve,
        matrix_ratio_certificate as matrix_ratio_certificate,
        MatrixRatioAccuracy as MatrixRatioAccuracy,
        MatrixRatioStatus as MatrixRatioStatus,
        multichannel_control_certificate as multichannel_control_certificate,
        MultiChannelControlCurve as MultiChannelControlCurve,
        multivariate_action_floor as multivariate_action_floor,
        multivariate_interference_certificate as multivariate_interference_certificate,
        multivariate_transfer_certificate as multivariate_transfer_certificate,
        multivariate_van_trees_certificate as multivariate_van_trees_certificate,
        MultivariateActionFloor as MultivariateActionFloor,
        MultivariateInterferenceCurve as MultivariateInterferenceCurve,
        MultivariateTransferCurve as MultivariateTransferCurve,
        MultivariateVanTreesCurve as MultivariateVanTreesCurve,
        nonlinear_regret_certificate as nonlinear_regret_certificate,
        NonlinearRegretCurve as NonlinearRegretCurve,
        optimal_fold_partition as optimal_fold_partition,
        orthogonal_control_certificate as orthogonal_control_certificate,
        OrthogonalControlCurve as OrthogonalControlCurve,
        panel_estimator_certificate as panel_estimator_certificate,
        PanelEstimatorGate as PanelEstimatorGate,
        partial_id_control_certificate as partial_id_control_certificate,
        PartialIdControlCurve as PartialIdControlCurve,
        pessimism_variance_certificate as pessimism_variance_certificate,
        PessimismCurve as PessimismCurve,
        regret_scaling as regret_scaling,
        RegretCurve as RegretCurve,
        space_time_fold_certificate as space_time_fold_certificate,
        SpaceTimeFoldCurve as SpaceTimeFoldCurve,
        strong_convexity_regret_bound as strong_convexity_regret_bound,
        transportability_regret_certificate as transportability_regret_certificate,
        TransportabilityCurve as TransportabilityCurve,
        van_trees_certificate as van_trees_certificate,
        VanTreesCurve as VanTreesCurve,
        worst_case_asymmetric_loss as worst_case_asymmetric_loss,
    )
    # isort: on
else:
    __getattr__ = _moved.serve_names(__name__, "chc.research.regret", _MOVED)

# The names this module keeps, then the moved ones: listed, so that a star-import reads
# each moved name through __getattr__, and warns, and dir() shows them.
__all__ = [
    "AdaptiveExplorationCurve",
    "CappedExplorationPolicy",
    "Matrix",
    "MinimaxAction",
    "MinimaxBranch",
    "MinimaxCriterion",
    "MinimaxExplorationCurve",
    "MinimaxLQCertificate",
    "MinimaxLQPolicy",
    "MinimaxRegretMove",
    "MoveRule",
    "OptimalExplorationCurve",
    "Vector",
    "adaptive_exploration_certificate",
    "capped_exploration_policy",
    "certainty_equivalence_gap",
    "closed_loop_cost",
    "dlqr",
    "minimax_action",
    "minimax_exploration_certificate",
    "minimax_lq_certificate",
    "minimax_lq_policy",
    "minimax_regret_move",
    "optimal_exploration_certificate",
    *sorted(_MOVED),
]


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
