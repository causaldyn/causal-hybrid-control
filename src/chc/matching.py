"""Kantorovich optimal transport for marketplace matching -- dispatch plan + dual surge prices.

The discrete, marketplace-native sibling of the continuum :mod:`chc.transport`: driver->rider
dispatch is Kantorovich's transportation problem (1939; Nobel 1975) -- move drivers/zone
(``supply``) to riders/zone (``demand``) at least travel ``cost``. The **dual potentials are the
market-clearing prices**: the demand dual ``g`` is the surge signal (higher where demand outstrips
supply), free as the dual of the same optimisation -- which no dispatch heuristic gives. Solved by
entropic (log-domain, differentiable) Sinkhorn so pricing is optimisable; ``eps -> 0`` recovers the
exact LP (Octave `glpk` cross-check: Kantorovich-Rubinstein gap = 0). NumPy baselines, JAX OT core.

An experiment that treats some of the rows moves the prices every row faces, so the treated-minus-
control difference an A/B test reads is not what treating every row would do.
:func:`shadow_price_effect` reads that global effect off the rows' rents in the experiment's own
matching instead, and :func:`shadow_price_interval` says how far it is determined at ``eps = 0``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike
from scipy import sparse, special
from scipy.optimize import linprog

from chc.games import project_simplex

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SinkhornResult:
    """A solved entropic OT: plan, dual potentials (surge prices), cost, duality gap, and how far
    the plan is from the marginals it was asked to meet."""

    plan: Array  # (m, n) transport / matching plan
    potentials_f: Array  # (m,) supply-side dual potentials
    potentials_g: Array  # (n,) demand-side dual potentials = market-clearing (surge) prices
    transport_cost: float  # <plan, cost>
    duality_gap: float  # |cost - dual|; -> 0 as eps -> 0 (Kantorovich-Rubinstein)
    # (||P 1 - a||_1 + ||P' 1 - b||_1) / sum(a). The loop ends on the demand update, so the columns
    # hold to rounding and this is the supply side's miss. The duality gap is no substitute: it
    # carries the entropic term, so it stays positive at convergence.
    marginal_residual: float


def sinkhorn(
    cost: Array,
    supply: Array,
    demand: Array,
    *,
    eps: float = 0.05,
    iters: int = 1000,
    tol: float = 1e-4,
) -> SinkhornResult:
    """Entropic (log-domain, stable) Kantorovich OT; returns the plan and dual potentials (surge).

    ``cost`` is ``(m, n)`` dispatch cost; ``supply``/``demand`` are the marginals (equal totals).
    Recovers the exact transportation LP as ``eps -> 0``; the demand potentials ``g`` are the
    market-clearing surge prices.

    The iteration count is fixed, so that the solve stays a ``lax.scan`` and differentiable, and a
    fixed count is a guess: at small ``eps`` the residual can sit on a plateau for thousands of
    iterations before it falls. On ``MarketplaceMatching.synthetic_city(seed=3)`` at its own
    ``eps = 0.02`` it is still 2.2e-3 after 2000 iterations and 1.7e-15 after 16000 (float64).
    :attr:`SinkhornResult.marginal_residual` says whether the guess was good, and a residual over
    ``tol`` is logged as a warning, because potentials read off an unconverged plan are not the
    market-clearing prices.
    """
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    cost = jnp.asarray(cost)
    a, b = jnp.asarray(supply), jnp.asarray(demand)
    log_a, log_b = jnp.log(a), jnp.log(b)

    def step(carry: tuple[Array, Array], _: Array) -> tuple[tuple[Array, Array], None]:
        f, g = carry
        f = eps * (log_a - logsumexp((g[None, :] - cost) / eps, axis=1))
        g = eps * (log_b - logsumexp((f[:, None] - cost) / eps, axis=0))
        return (f, g), None

    (f, g), _ = jax.lax.scan(step, (jnp.zeros_like(a), jnp.zeros_like(b)), None, length=iters)
    plan = jnp.exp((f[:, None] + g[None, :] - cost) / eps)
    transport = float(jnp.sum(plan * cost))
    dual = float(jnp.dot(f, a) + jnp.dot(g, b))
    miss = jnp.sum(jnp.abs(plan.sum(axis=1) - a)) + jnp.sum(jnp.abs(plan.sum(axis=0) - b))
    residual = float(miss / jnp.sum(a))
    if not residual <= tol:
        _log.warning(
            "sinkhorn stopped %.2e of the mass away from its marginals after %d iterations; "
            "raise iters or eps before reading prices off this plan",
            residual,
            iters,
            extra={
                "chc_event": "sinkhorn",
                "marginal_residual": residual,
                "eps": eps,
                "iters": iters,
                "tol": tol,
            },
        )
    return SinkhornResult(plan, f, g, transport, abs(transport - dual), residual)


@dataclass(frozen=True)
class ShadowPriceEffect:
    """What treating every randomised row would do to a matching's welfare, read off one experiment.

    ``effect`` is the treated-minus-control difference of the rows' rents, and ``naive`` the same
    difference of the surplus each row's matches earn it, which is what an A/B test reads. Both are
    totals over the randomised mass. ``standard_error`` is the effect's over the assignment.

    ``curvature`` estimates ``-f''(p)``, how concave welfare is in the treated share ``p``, from the
    same matching, less the noise the assignment puts into it; on a market of fixed types split into
    treated and control copies, which has no such noise, it reads low. ``second_order_bias`` is
    ``(1/2 - p) * curvature``: what the effect misses the global effect by, to second order in the
    treatment. ``nu_hat`` is how far the treatment would move the prices between no row treated and
    every row, to first order, in units of ``eps``. Above 1 the market is near the LP limit, where
    the bias is first order; ``second_order_bias`` is then ``None``, and ``reason`` says why.
    """

    effect: float
    standard_error: float
    naive: float
    treated_share: float
    curvature: float
    nu_hat: float
    marginal_residual: float
    second_order_bias: float | None
    reason: str  # why second_order_bias is None; empty when it is not


def _experiment_strata(
    cost: np.ndarray,
    supply: np.ndarray,
    demand: np.ndarray,
    treated: ArrayLike,
    randomised: ArrayLike | None,
    strata: ArrayLike | None,
) -> list[tuple[float, float, np.ndarray, np.ndarray]]:
    """Per stratum: its randomised mass, its treated mass, and each arm's weights on the rows,
    ``supply`` normalised within the stratum's arm and zero outside it. Without ``strata`` the
    experiment is one stratum."""
    if cost.ndim != 2 or supply.shape != cost.shape[:1] or demand.shape != cost.shape[1:]:
        raise ValueError(
            f"cost must be (m, n) with supply (m,) and demand (n,); got {cost.shape}, "
            f"{supply.shape} and {demand.shape}"
        )
    if not (np.all(supply > 0.0) and np.all(demand > 0.0)):
        raise ValueError("supply and demand must be positive")
    if abs(supply.sum() - demand.sum()) > 1e-9 * supply.sum():
        raise ValueError(
            f"supply and demand must have equal totals; got {supply.sum()} and {demand.sum()}"
        )
    rows = cost.shape[0]
    in_experiment = np.ones(rows, dtype=bool) if randomised is None else np.asarray(randomised)
    assignment = np.asarray(treated)
    for name, mask in (("treated", assignment), ("randomised", in_experiment)):
        if mask.dtype != bool or mask.shape != (rows,):
            raise ValueError(f"{name} must be one bool per row, shape ({rows},)")
    labels = np.zeros(rows, dtype=int) if strata is None else np.asarray(strata)
    if labels.shape != (rows,):
        raise ValueError(f"strata must be one label per row, shape ({rows},)")
    blocks = []
    for label in np.unique(labels[in_experiment]):
        member = in_experiment & (labels == label)
        arms = member & assignment, member & ~assignment
        if min(np.count_nonzero(arm) for arm in arms) < 2:
            where = "" if strata is None else f" in stratum {label}"
            raise ValueError(
                f"the experiment needs at least two treated rows and two control rows{where}"
            )
        weights = [np.where(arm, supply, 0.0) / supply[arm].sum() for arm in arms]
        blocks.append((float(supply[member].sum()), float(supply[arms[0]].sum()), *weights))
    if not blocks:
        raise ValueError("the experiment needs at least two treated rows and two control rows")
    return blocks


def _contrast_covariance(
    values: np.ndarray, blocks: list[tuple[float, float, np.ndarray, np.ndarray]]
) -> np.ndarray:
    """The covariance, over the assignment, of the post-stratified treated-minus-control
    difference of the rows' ``values`` (m, k): each arm's weighted spread about its own mean, as
    for independent coins, weighted by the square of its stratum's mass."""
    covariance = np.zeros((values.shape[1], values.shape[1]))
    for mass, _, *arms in blocks:
        for weights in arms:
            count = np.count_nonzero(weights)
            spread = weights[:, None] * (values - weights @ values)
            covariance += mass**2 * (spread.T @ spread) * count / (count - 1)
    return covariance


def shadow_price_effect(
    cost: ArrayLike,
    supply: ArrayLike,
    demand: ArrayLike,
    treated: ArrayLike,
    *,
    eps: float,
    randomised: ArrayLike | None = None,
    strata: ArrayLike | None = None,
    iters: int = 1000,
    tol: float = 1e-6,
) -> ShadowPriceEffect:
    """The global effect of a treatment on a matching's welfare, from one experiment on its rows.

    Experimental: it may change or be withdrawn in any release.

    ``cost``, ``supply`` and ``demand`` are the market the experiment runs, as :func:`sinkhorn`
    takes them, with the treatment already in the treated rows' costs. ``treated`` marks those rows,
    and ``randomised`` the rows the experiment assigned at all: a row outside it, such as an idle
    pool of supply, is priced but not counted. The rows are the randomised units; to randomise the
    other side, pass the transposed market. Welfare is ``-(<P, cost> + eps KL(P | supply x
    demand))`` at the optimal plan ``P``, and the effect is its change when every randomised row is
    treated. ``strata``, one label per row fixed before the assignment, such as the rows' types,
    post-stratify the experiment: each stratum's treated-minus-control difference is weighted by the
    stratum's mass, so the mix of each stratum the coins dealt either arm no longer moves the
    effect, and the standard error, the curvature and ``nu_hat`` are read the same way. Every
    stratum needs two rows in each arm, and ``second_order_bias`` takes every row to have had the
    same chance of treatment.

    Each row's rent is the derivative of welfare in its own mass, so the treated-minus-control
    difference of rents is the derivative ``f'(p)`` of welfare in the treated share. By concavity
    ``f'(1) <= f(1) - f(0) <= f'(0)``, and ``f'(p)`` misses the global effect by
    ``(1/2 - p) (-f''(p))`` to second order in the treatment, where the naive difference misses it
    at first order by the displacement it prices. ``-f''(p) = eps g' C^+ g`` is read off the same
    plan, with ``g`` the treated-minus-control difference of the rows' plans and ``C`` the
    Jacobian of their demand in the prices. Design at ``p = 1/2``, where the second-order term
    vanishes whatever the curvature; averaging two experiments at ``p = 1/2 -+ 1/(2 sqrt 3)`` leaves
    a bias of fifth order.

    Measured over 400 assignments at treated shares 0.2 and 0.5, on six markets (80 units against
    80 others at three balances of supply and demand, and 80, 320 or 1280 units against three
    zones), the effect's spread was 0.90--1.06 of its ``standard_error``, and 0.39--0.89 of the
    naive difference's. The curvature from one experiment is noisy: its spread over assignments
    was 2.1--4.2 times its value on 80 and 320 units against zones, 0.8--1.1 times on 1280, and
    1.3--5.8 times on 80 units against 80 others, its mean within 1.7 standard errors of the value
    throughout. At ``p = 0.2`` the bias it prices was 0.4--7% of the effect's standard error on
    those markets, so it matters where the standard error is small. It does not price the part of
    the bias that shrinks with the number of rows: under 0.3% of the standard error on five of the
    markets, but 2--3% where demand is short on 80 units against 80 others, which at ``p = 1/2`` is
    most of the bias there.

    Post-stratified by the rows' four types, those too small for two rows an arm merged before the
    assignment, the effect's spread on the same markets was 0.52--0.78 of the plain effect's, and
    0.85--1.04 of its standard error. Fix the strata before the assignment, each large enough that
    an arm short of two of its rows is unlikely: merging the strata an assignment left short into
    another after it moved the effect by up to a quarter of the global effect on average, and
    falling back to the plain difference on such assignments moves it about as much.

    Near the LP limit the second-order law fails, and the bias saturates at the LP's first-order
    bias instead. Over twenty markets swept towards that limit, the law gave the bias to within 10%
    in nine cases of ten where ``nu_hat`` was below 0.5, and within 33% below 1; past 1 the median
    miss was 24--70%. ``nu_hat`` above 1 leaves ``second_order_bias`` at ``None`` and logs why, and
    so does a solve whose marginal residual is over ``tol``. Read off one experiment, ``nu_hat``
    carries the assignment's noise, which raises it on average, so where an arm has few rows it
    withholds the claim far from the limit too: in 1--7% of the experiments on 80 units against
    three zones, in none on 320 or 1280, and in 4--100% on 80 units against 80 others, whose 80
    prices it reads one by one. Post-stratified, it withheld the claim in at most 1% of the
    experiments on 80 units against three zones, and nearly as often as before on 80 units against
    80 others. A float32 solve moved every number by about 1e-7 relative against float64 on those
    markets, at ``eps = 0.3``.

    Raises:
        ValueError: on ``eps <= 0`` (see :func:`shadow_price_interval`), shapes that do not agree,
            masses that are not positive or do not balance, a ``treated`` or ``randomised`` that is
            not one bool per row, ``strata`` that are not one label per row, or fewer than two rows
            in either arm of the experiment or of a stratum.
    """
    if eps <= 0.0:
        raise ValueError(
            "eps must be positive; at eps = 0 the rents are not unique, see shadow_price_interval"
        )
    costs, masses, capacities = (np.asarray(v, dtype=float) for v in (cost, supply, demand))
    blocks = _experiment_strata(costs, masses, capacities, treated, randomised, strata)
    total = sum(block[0] for block in blocks)
    share = sum(block[1] for block in blocks) / total
    contrast = sum(mass * (on - off) for mass, _, on, off in blocks)

    result = sinkhorn(
        jnp.asarray(costs),
        jnp.asarray(masses),
        jnp.asarray(capacities),
        eps=eps,
        iters=iters,
        tol=tol,
    )
    potentials = np.asarray(result.potentials_g, dtype=float)
    # Rents and plans are read from the column potentials alone: exact for them even where the row
    # potentials lag half a step, and defined for a row of any mass.
    logits = (potentials[None, :] - costs) / eps
    normaliser = special.logsumexp(logits, axis=1)
    rent = eps * normaliser
    plan = np.exp(logits - normaliser[:, None])  # each row's matches per unit of its mass
    surplus = rent + plan @ (eps * np.log(capacities) - potentials)

    displacement = contrast @ plan
    jacobian = np.diag(masses @ plan) - (plan * masses[:, None]).T @ plan
    # The last column's price is the gauge: C is singular along the constant vector.
    noise = _contrast_covariance(plan, blocks)[:-1, :-1]
    solved = np.linalg.lstsq(
        jacobian[:-1, :-1], np.column_stack([displacement[:-1], noise]), rcond=None
    )[0]
    price_speed = solved[:, 0]
    curvature = eps * float(displacement[:-1] @ price_speed - np.trace(solved[:, 1:]))
    nu_hat = float(np.ptp(np.append(price_speed, 0.0)))

    reason = ""
    if not result.marginal_residual <= tol:
        reason = (
            f"the solve stopped {result.marginal_residual:.1e} of the mass away from its "
            "marginals; raise iters"
        )
    elif nu_hat > 1.0:
        reason = (
            f"the prices move {nu_hat:.2f} eps across the treated share by this experiment's "
            "reading: near the LP limit, where the bias is first order, or too few rows in an arm "
            "to read them"
        )
    if reason:
        _log.warning(
            "shadow_price_effect makes no second-order claim: %s",
            reason,
            extra={
                "chc_event": "shadow_price_effect",
                "nu_hat": nu_hat,
                "marginal_residual": result.marginal_residual,
                "eps": eps,
            },
        )
    return ShadowPriceEffect(
        effect=float(contrast @ rent),
        standard_error=float(np.sqrt(_contrast_covariance(rent[:, None], blocks)[0, 0])),
        naive=float(contrast @ surplus),
        treated_share=share,
        curvature=curvature,
        nu_hat=nu_hat,
        marginal_residual=result.marginal_residual,
        second_order_bias=None if reason else (0.5 - share) * curvature,
        reason=reason,
    )


def shadow_price_interval(
    cost: ArrayLike,
    supply: ArrayLike,
    demand: ArrayLike,
    treated: ArrayLike,
    *,
    randomised: ArrayLike | None = None,
    strata: ArrayLike | None = None,
) -> tuple[float, float]:
    """The effect :func:`shadow_price_effect` reads, at ``eps = 0``: an interval, not a point.

    Experimental: it may change or be withdrawn in any release.

    The exact matching LP has many optimal duals wherever it is degenerate, and every one of them
    is a supergradient of welfare in the treated share. So the treated-minus-control difference of
    the rows' rents ranges over ``[f'(p+), f'(p-)]`` across the optimal dual face, and this returns
    that range, from two LPs over the face (HiGHS). A solver's own dual is one end of it. At a
    breakpoint of welfare in the treated share the interval is wide; between breakpoints it is a
    point, and it is the effect's limit as ``eps -> 0``. Its bias against the global effect is
    first order in the treatment: the LP has no second-order regime. Arguments and refusals are
    :func:`shadow_price_effect`'s.

    Raises:
        ValueError: as :func:`shadow_price_effect`.
        RuntimeError: when HiGHS does not solve an LP a balanced market always has a solution to.
    """
    costs, masses, capacities = (np.asarray(v, dtype=float) for v in (cost, supply, demand))
    blocks = _experiment_strata(costs, masses, capacities, treated, randomised, strata)
    contrast = sum(mass * (on - off) for mass, _, on, off in blocks)
    rows, columns = costs.shape
    marginals = sparse.vstack(
        [
            sparse.kron(sparse.eye(rows), np.ones((1, columns))),
            sparse.kron(np.ones((1, rows)), sparse.eye(columns)),
        ]
    ).tocsr()
    primal = linprog(
        costs.ravel(),
        A_eq=marginals,
        b_eq=np.concatenate([masses, capacities]),
        bounds=(0.0, None),
        method="highs",
    )
    if primal.status != 0:
        raise RuntimeError(f"HiGHS did not solve the matching LP: {primal.message}")
    # The face: f_i + g_j <= cost_ij, at the optimal value; the rents are -f. The contrast sums to
    # zero, so the gauge, the last column's g, moves neither end; fixing it keeps a line out of the
    # face, which put the ends 45 times closer to an independent solve's.
    gauge = np.zeros(rows + columns)
    gauge[-1] = 1.0
    ends = []
    for sense in (1.0, -1.0):
        face = linprog(
            np.concatenate([-sense * contrast, np.zeros(columns)]),
            A_ub=marginals.T,
            b_ub=costs.ravel(),
            A_eq=np.vstack([np.concatenate([masses, capacities]), gauge]),
            b_eq=[primal.fun, 0.0],
            bounds=(None, None),
            method="highs",
        )
        if face.status != 0:
            raise RuntimeError(f"HiGHS did not solve the dual face: {face.message}")
        ends.append(sense * face.fun)
    return float(ends[0]), float(ends[1])


def _nearest_local(cost: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> tuple[float, float]:
    """Naive: each rider zone served only by its single nearest driver zone; rest is stranded."""
    a, b = supply.copy(), demand.copy()
    served_cost, served = 0.0, 0.0
    for j in np.argsort(-b):  # high-demand zones first
        i = int(np.argmin(cost[:, j]))
        take = min(a[i], b[j])
        served_cost += take * cost[i, j]
        a[i] -= take
        served += take
    return served_cost, served / float(demand.sum())


def _nearest_reroute(cost: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> float:
    """Myopic but complete: each rider zone takes nearest available supply, rerouting to fill."""
    a, b = supply.copy(), demand.copy()
    total = 0.0
    for j in np.argsort(-b):
        for i in np.argsort(cost[:, j]):
            if b[j] <= 0.0:
                break
            take = min(a[i], b[j])
            total += take * cost[i, j]
            a[i] -= take
            b[j] -= take
    return total


@dataclass(frozen=True)
class MarketplaceMatching:
    """Zones with driver ``supply``, rider ``demand``, and ``cost`` (travel) between them."""

    supply: Array  # (Z,) drivers per zone
    demand: Array  # (Z,) riders per zone (equal total to supply)
    cost: Array  # (Z, Z) zone-to-zone dispatch cost (travel distance)
    eps: float = 0.02
    iters: int = 2000

    @classmethod
    def synthetic_city(cls, n_zones: int = 8, seed: int = 0) -> MarketplaceMatching:
        """Drivers cluster centrally, riders spread to the suburbs -- a supply-demand mismatch."""
        rng = np.random.default_rng(seed)
        positions = rng.uniform(-2.0, 2.0, (n_zones, 2))
        radius = np.linalg.norm(positions, axis=1)
        supply = np.exp(-radius)  # drivers concentrate near the centre (small radius)
        demand = 0.3 + radius  # riders concentrate in the periphery
        supply *= demand.sum() / supply.sum()  # balance totals
        cost = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        return cls(jnp.asarray(supply), jnp.asarray(demand), jnp.asarray(cost))

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.asarray(self.cost), np.asarray(self.supply), np.asarray(self.demand)

    def optimal(self) -> SinkhornResult:
        """The Kantorovich OT dispatch (100% served at min travel) + the dual surge prices."""
        return sinkhorn(self.cost, self.supply, self.demand, eps=self.eps, iters=self.iters)

    def surge_prices(self) -> Array:
        """Market-clearing surge = the demand-side dual potentials (high in undersupplied zones)."""
        return self.optimal().potentials_g

    def nearest_local(self) -> tuple[float, float]:
        """Naive local dispatch: ``(served_cost, coverage)`` -- strands demand it cannot reach."""
        return _nearest_local(*self._arrays())

    def nearest_reroute_cost(self) -> float:
        """Myopic complete dispatch cost (100% served, but locally greedy -- costlier than OT)."""
        return _nearest_reroute(*self._arrays())

    def surge_rebalance(self, step: float = 0.5) -> tuple[float, float]:
        """Apply surge as a driver incentive; drivers best-respond toward high-price zones.

        Returns supply-demand imbalance ``(before, after)``: the surge prices, projected onto the
        supply simplex (a best-response via :func:`chc.games.project_simplex`), pull drivers toward
        high-price zones -- surge as an *intervention* whose equilibrium effect the dual predicts.
        """
        g = self.surge_prices()
        s, d = jnp.asarray(self.supply), jnp.asarray(self.demand)
        before = float(jnp.sum(jnp.abs(s - d)))
        s_new = project_simplex(s + step * (g - jnp.mean(g)), float(jnp.sum(s)))
        return before, float(jnp.sum(jnp.abs(s_new - d)))


def marketplace_report(matching: MarketplaceMatching) -> str:
    """Kantorovich OT dispatch vs two naive failure modes, plus the surge/equilibrium win."""
    opt = matching.optimal()
    _, coverage = matching.nearest_local()
    reroute = matching.nearest_reroute_cost()
    before, after = matching.surge_rebalance()
    total = float(jnp.sum(matching.demand))
    saved = (reroute - opt.transport_cost) / reroute * 100
    stranded = (1.0 - coverage) * total
    return "\n".join(
        [
            f"Kantorovich OT: 100% served at min cost {opt.transport_cost:.2f} + surge prices.",
            f"naive local-only dispatch strands {stranded:.1f} of {total:.1f} riders "
            f"({(1.0 - coverage) * 100:.0f}%) where demand exceeds local supply.",
            f"naive reroute (100% served): cost {reroute:.2f} -> OT is {saved:.1f}% cheaper.",
            f"surge -> driver reallocation cuts imbalance {before:.2f} -> {after:.2f} "
            f"({(before - after) / before * 100:.0f}%).",
            f"Kantorovich-Rubinstein duality gap = {opt.duality_gap:.2e} (surge = free dual).",
        ]
    )
