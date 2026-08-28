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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]

_DEFAULT_ERRORS = (0.04, 0.02, 0.01, 0.005, 0.0025)


@dataclass(frozen=True)
class RegretCurve:
    """Empirical certificate: median CE suboptimality vs model error, and its log-log slope."""

    errors: Vector  # median realised model-error magnitude ||[dA, dB]|| at each swept level
    gaps: Vector  # median certainty-equivalence suboptimality gap at each level
    exponent: float  # fitted log-log slope of gap vs error (~2.0 => quadratic suboptimality)


def dlqr(a: Matrix, b: Matrix, q: Matrix, r: Matrix) -> tuple[Matrix, Matrix]:
    """Infinite-horizon discrete LQR: optimal gain ``K`` and cost-to-go ``P`` (via the DARE)."""
    p = solve_discrete_are(a, b, q, r)
    k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
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


def regret_scaling(
    a: Matrix,
    b: Matrix,
    q: Matrix,
    r: Matrix,
    x0: Vector,
    *,
    errors: Sequence[float] = _DEFAULT_ERRORS,
    n_samples: int = 400,
    seed: int = 0,
) -> RegretCurve:
    """Certificate that CE suboptimality is quadratic in model error (slope ~2 in the small limit).

    At each target magnitude ``eps`` draws ``n_samples`` Gaussian model perturbations ``(dA, dB)``
    scaled by ``eps``, records the :func:`certainty_equivalence_gap`, and fits the log-log slope of
    gap vs realised error over all samples. Perturbations that make the estimate unstabilisable are
    skipped. Theory (Mania, Tu & Recht 2019, the local quadratic CE bound) predicts exponent 2.
    """
    rng = np.random.default_rng(seed)
    median_errors: list[float] = []
    median_gaps: list[float] = []
    log_err: list[float] = []
    log_gap: list[float] = []
    for eps in errors:
        errs, gaps = [], []
        for _ in range(n_samples):
            d_a, d_b = eps * rng.normal(size=a.shape), eps * rng.normal(size=b.shape)
            try:
                gap = certainty_equivalence_gap(a, b, q, r, a + d_a, b + d_b, x0)
            except (np.linalg.LinAlgError, ValueError):
                continue  # estimate not stabilisable: no certainty-equivalent gain exists
            err = float(np.sqrt(np.sum(d_a**2) + np.sum(d_b**2)))
            if np.isfinite(gap) and gap > 0.0:
                errs.append(err)
                gaps.append(gap)
        if errs:
            median_errors.append(float(np.median(errs)))
            median_gaps.append(float(np.median(gaps)))
            log_err.extend(np.log(errs))
            log_gap.extend(np.log(gaps))
    exponent = float(np.polyfit(log_err, log_gap, 1)[0]) if log_err else float("nan")
    return RegretCurve(np.array(median_errors), np.array(median_gaps), exponent)


@dataclass(frozen=True)
class OrthogonalControlCurve:
    """Double-debiasing certificate: control regret vs nuisance error, plug-in vs orthogonal."""

    errors: Vector  # nuisance-estimation error levels eps
    single_regret: Vector  # median control regret from a single-residualisation (plug-in) effect
    orthogonal_regret: Vector  # median control regret from the Neyman-orthogonal (DML) effect
    single_exponent: float  # fitted log-log slope ~2 (plug-in regret ~ eps^2)
    orthogonal_exponent: float  # fitted log-log slope ~4 (orthogonal regret ~ eps^4)


def orthogonal_control_certificate(
    *,
    b_true: float = 2.0,
    alpha: float = 1.0,
    gamma: float = 1.5,
    target: float = 1.0,
    effort: float = 0.5,
    errors: Sequence[float] = (0.2, 0.1, 0.05, 0.025, 0.0125),
    n: int = 400_000,
    n_seeds: int = 6,
    noise: float = 0.05,
) -> OrthogonalControlCurve:
    """The DOUBLE debiasing of orthogonal certainty-equivalence control -- a novel result derived in
    ``validation/orthogonal_control.mac`` and proved in ``proofs/orthogonal_control.v``.

    A confounder ``z`` drives both the action (``u = alpha z + noise``) and the outcome
    (``y = b_true u + gamma z + noise``). A controller hits ``target`` using its estimate of the
    causal effect ``b = dy/du``, scored by regret on the true plant. Under nuisance error ``eps``,
    a **single-residualisation** (non-orthogonal) effect is ``O(eps)``-biased, so its control regret
    is ``O(eps^2)``; the **orthogonal Double ML** effect is ``O(eps^2)``-biased, so -- through the
    same certainty-equivalence quadraticity -- its regret is ``O(eps^4)``. The estimator's
    orthogonality compounds with the quadratic regret map: two debiasings, statistics and control.
    Fits the two log-log slopes (~2 vs ~4).
    """
    opt = target * b_true / (b_true**2 + effort)
    opt_cost = (b_true * opt - target) ** 2 + effort * opt**2

    def regret(b_hat: float) -> float:
        u = target * b_hat / (b_hat**2 + effort)  # action optimal for the estimated effect
        return (b_true * u - target) ** 2 + effort * u**2 - opt_cost

    err = np.asarray(errors, dtype=np.float64)
    single = np.zeros((n_seeds, err.size))
    orth = np.zeros((n_seeds, err.size))
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        for j, eps in enumerate(err):
            z = rng.standard_normal(n)
            u = alpha * z + rng.standard_normal(n)
            y = b_true * u + gamma * z + noise * rng.standard_normal(n)
            m_y = (b_true * alpha + gamma) * z + eps * z  # misspecified outcome nuisance
            m_u = alpha * z + eps * z  # misspecified treatment nuisance
            u_res = u - m_u
            b_single = float(np.dot(u_res, y) / np.dot(u_res, u))  # single residualisation
            b_orth = float(np.dot(u_res, y - m_y) / np.dot(u_res, u_res))  # orthogonal DML
            single[s, j], orth[s, j] = regret(b_single), regret(b_orth)
    single_med, orth_med = np.median(single, axis=0), np.median(orth, axis=0)
    log_e = np.log(err)
    return OrthogonalControlCurve(
        err,
        single_med,
        orth_med,
        float(np.polyfit(log_e, np.log(np.abs(single_med)), 1)[0]),
        float(np.polyfit(log_e, np.log(np.abs(orth_med)), 1)[0]),
    )


@dataclass(frozen=True)
class PessimismCurve:
    """Scalar one-step: uncertainty regularizer equals the effect variance (not general)."""

    variances: Vector  # effect-estimate variance s^2 grid
    optimal_rho: Vector  # empirical expected-regret-minimising pessimism (~ s^2)
    ce_regret: Vector  # expected regret of the certainty-equivalent control (rho = 0)
    pessimistic_regret: Vector  # expected regret of the pessimistic control at rho = s^2


def pessimism_variance_certificate(
    *,
    b0: float = 1.0,
    rr: float = 1.0,
    target: float = 1.0,
    variances: Sequence[float] = (0.05, 0.1, 0.2, 0.4),
    n: int = 200_000,
    rho_max: float = 1.2,
    rho_points: int = 61,
    seed: int = 0,
) -> PessimismCurve:
    """Certificate that pessimism belongs in the optimality condition, calibrated to the uncertainty
    (derived in ``validation/pessimistic_optimality.mac``, proved in
    ``proofs/pessimistic_optimality.v``). The estimated effect is ``b0``; the true effect is
    ``b0 + e`` with ``e`` of variance ``s^2``. The pessimistic control ``target b0/(b0^2+rr+rho)``
    adds an effective-effort ``rho`` (distrust) to the stationarity condition. Sweeping ``rho`` per
    ``s^2``, the expected-regret-minimising ``rho*`` tracks ``s^2`` (optimal pessimism = estimate
    variance -- no tuning), and the pessimistic control's expected regret is below the
    certainty-equivalent (``rho = 0``) one.
    """
    rng = np.random.default_rng(seed)
    rho_grid = np.linspace(0.0, rho_max, rho_points)
    s2 = np.asarray(variances, dtype=np.float64)
    opt_rho = np.zeros(s2.size)
    ce_reg = np.zeros(s2.size)
    pess_reg = np.zeros(s2.size)
    for i, var in enumerate(s2):
        b = b0 + np.sqrt(var) * rng.standard_normal(n)  # true effect = estimate + uncertainty
        u_opt = target * b / (b**2 + rr)  # oracle action for the realised true effect

        def exp_regret(rho: float, b: Vector = b, u_opt: Vector = u_opt) -> float:
            return float(np.mean((b**2 + rr) * (target * b0 / (b0**2 + rr + rho) - u_opt) ** 2))

        expected = np.array([exp_regret(rho) for rho in rho_grid])
        opt_rho[i] = float(rho_grid[int(np.argmin(expected))])
        ce_reg[i] = float(expected[0])  # rho = 0 (certainty equivalence)
        u_pess = target * b0 / (b0**2 + rr + var)  # rho = s^2 (the theorem's optimum)
        pess_reg[i] = float(np.mean((b**2 + rr) * (u_pess - u_opt) ** 2))
    return PessimismCurve(s2, opt_rho, ce_reg, pess_reg)


@dataclass(frozen=True)
class InformationLowerBoundCurve:
    """CR lower bound on control regret; confounding lowers information, raising the floor."""

    sample_sizes: Vector  # n grid
    experimental_regret: Vector  # realised regret of the efficient (full-information) controller
    cramer_rao_floor: Vector  # the CR lower bound C*sigma^2/(n*V_exp)
    confounded_floor: Vector  # the (higher) CR floor under confounding, C*sigma^2/(n*V_conf)
    rate_slope: float  # log-log slope of experimental_regret vs n (~ -1: the optimal 1/n rate)
    floor_ratio: float  # confounded / experimental floor (> 1: confounding costs information)


def information_lower_bound_certificate(
    *,
    b: float = 1.0,
    rr: float = 0.5,
    xt: float = 1.0,
    sigma: float = 0.5,
    alpha: float = 1.0,
    sample_sizes: Sequence[int] = (200, 500, 1000, 2000, 4000),
    n_seeds: int = 300,
) -> InformationLowerBoundCurve:
    """Information-theoretic lower bound for UNBIASED effect estimators (Cramer-Rao;
    derived in ``validation/information_lower_bound.mac``, proved in
    ``proofs/information_lower_bound.v``). The EXACT regret map is
    ``R = (b^2+rr)*(u*(bhat)-u*(b))^2`` (nonlinear in b); to first order in the effect error this is
    ``C*(bhat-b)^2``, and Cramer-Rao on the (unbiased) estimator gives the local floor
    ``E[regret] >= C*sigma^2/(n*V_id)``, ``V_id`` the identifying variance. Scope caveats: a *local*
    (delta-method) bound for unbiased estimators, NOT a global minimax lower bound (a full version
    needs local-asymptotic-minimax / van Trees for the estimand ``u*(b)``); and at the knife edge
    ``rr = b^2`` the coefficient ``C`` vanishes, so the leading order is higher. The ``1/n`` rate
    matches the online upper bound (Result 7) in this scalar model; confounding reduces ``V_id``
    (``V_exp`` -> residual ``V_conf``), raising the floor.
    """
    coeff = xt**2 * (rr - b**2) ** 2 / (rr + b**2) ** 3
    v_exp = alpha**2 + 1.0  # total action variance (Var(alpha*z + nu), Vz = Vnu = 1)
    v_conf = 1.0  # residual identifying variance Var(u|z) = Var(nu) under confounding
    ns = np.asarray(sample_sizes, dtype=np.float64)
    exp_regret = np.zeros(ns.size)
    for i, nf in enumerate(sample_sizes):
        n = int(nf)
        biases = np.empty(n_seeds)
        for s in range(n_seeds):
            rng = np.random.default_rng(1000 * s + n)
            u = np.sqrt(v_exp) * rng.standard_normal(n)  # randomised: all variance identifies
            y = b * u + sigma * rng.standard_normal(n)
            biases[s] = np.dot(u, y) / np.dot(u, u) - b
        exp_regret[i] = coeff * float(np.mean(biases**2))
    cr_floor = coeff * sigma**2 / (ns * v_exp)
    conf_floor = coeff * sigma**2 / (ns * v_conf)
    slope = float(np.polyfit(np.log(ns), np.log(exp_regret), 1)[0])
    return InformationLowerBoundCurve(ns, exp_regret, cr_floor, conf_floor, slope, v_exp / v_conf)


@dataclass(frozen=True)
class HighProbRegretCurve:
    """Finite-sample high-probability UPPER bound on scalar CE regret (w.p. >= 1-delta)."""

    deltas: Vector  # confidence levels delta (band holds with probability >= 1-delta)
    highprob_bands: Vector  # C*2*sigma^2*log(2/delta)/(n*V_exp): the finite-sample band
    empirical_coverage: Vector  # MC fraction of trials with regret <= band (should be >= 1-delta)
    confounded_bands: Vector  # the (wider) band under confounding (residual V_conf < V_exp)
    band_over_floor: Vector  # band / CR-floor = 2*log(2/delta): the log(1/delta) confidence price
    cr_floor: float  # the Result 10 in-expectation floor C*sigma^2/(n*V_exp)


def highprob_regret_certificate(
    *,
    b: float = 1.0,
    rr: float = 0.5,
    xt: float = 1.0,
    sigma: float = 0.5,
    n: int = 500,
    v_exp: float = 2.0,
    v_conf: float = 1.0,
    deltas: Sequence[float] = (0.5, 0.25, 0.1, 0.05, 0.01),
    n_trials: int = 4000,
) -> HighProbRegretCurve:
    """Finite-sample high-probability UPPER bound on scalar CE regret (derived in
    ``validation/highprob_regret.mac``, proved in ``proofs/highprob_regret.v``). NOTE: this is NOT a
    "high-probability version" of the Result-10 Cramer-Rao bound -- CR is an *expected LOWER* bound
    on estimator variance (hence a regret floor), this is a *high-probability UPPER* bound on
    realised regret; the two point in opposite directions and are complementary. The effect
    estimator concentrates (sub-Gaussian): ``|b_hat-b| <= r(delta)`` w.p. ``>= 1-delta``,
    ``r(delta)^2 = 2*sigma^2*log(2/delta)/(n*V_id)``. Composing with the LOCAL quadratic
    ``regret = C*(b_hat-b)^2`` gives, w.p. ``>= 1-delta``, ``regret <= 2*log(2/delta) * band-scale``
    -- a ``log(1/delta)`` multiple of ``C*sigma^2/(n*V_id)``; confounding (smaller ``V_id``) widens
    it. A fully rigorous upper bound replaces ``C`` by the Lipschitz constant of ``u*`` on the
    confidence region (``regret = (b^2+rr)(u*(b_hat)-u*(b))^2 <= (b^2+rr) L^2 r(delta)^2``); locally
    ``L^2 -> C``. Concentration checked by Monte-Carlo coverage; the implication proved in Rocq.
    """
    coeff = xt**2 * (rr - b**2) ** 2 / (rr + b**2) ** 3
    ds = np.asarray(deltas, dtype=np.float64)
    logs = np.log(2.0 / ds)
    floor = coeff * sigma**2 / (n * v_exp)
    bands = coeff * 2.0 * sigma**2 * logs / (n * v_exp)
    conf_bands = coeff * 2.0 * sigma**2 * logs / (n * v_conf)
    coverage = np.zeros(ds.size)
    regrets = np.empty(n_trials)
    for t in range(n_trials):
        rng = np.random.default_rng(4241 * t + n)
        u = np.sqrt(v_exp) * rng.standard_normal(n)
        y = b * u + sigma * rng.standard_normal(n)
        regrets[t] = coeff * (np.dot(u, y) / np.dot(u, u) - b) ** 2
    for i, band in enumerate(bands):
        coverage[i] = float(np.mean(regrets <= band))
    return HighProbRegretCurve(ds, bands, coverage, conf_bands, bands / floor, float(floor))


@dataclass(frozen=True)
class TransportabilityCurve:
    """Deployment regret = transportable part (zero) + W1 residual (quadratic in W1)."""

    w1_distances: Vector  # W1(P, P') distance between source and target domains
    transportable_regret: Vector  # regret when the effect is recoverable on target (b_tgt = b_src)
    nontransport_regret: Vector  # CE-quadratic regret C*Lip^2*d^2 when b_tgt = b_src + Lip*d
    exact_regret: Vector  # the simulated cost-gap (cross-checks the quadratic-in-W1 scaling)
    wdro_bound: float  # C*Lip^2*eps^2 for the W-DRO radius eps: covers all d <= eps
    nontransport_slope: float  # log-log slope of the CE-quadratic regret vs W1 (= 2)
    exact_slope: float  # log-log slope of the simulated regret vs W1 (~ 2)


def transportability_regret_certificate(
    *,
    b_src: float = 1.0,
    rr: float = 0.5,
    xt: float = 1.0,
    lip: float = 1.0,
    d_lo: float = 0.005,
    d_hi: float = 0.15,
    n_d: int = 12,
    wdro_radius: float = 0.15,
) -> TransportabilityCurve:
    """Transportability regret -- a controller trained on source domain P, deployed on target P'
    (Bareinboim, Causal AI, Ch 9). CONCEPTUAL proposition (not a self-contained theorem): the bounds
    below hold UNDER the stated assumption, they do not follow from Kantorovich-Rubinstein alone.
    Derived in ``validation/transportability_regret.mac``, proved in
    ``proofs/transportability_regret.v``. Deployment regret ``C*(b_src-b_tgt)^2`` splits into: (A) a
    TRANSPORTABLE part -- if the effect is recoverable on target (``b_tgt=b_src``) regret is ZERO
    at ANY distributional distance; (B) a non-transportable residual ``C*Lip^2*d^2`` -- this
    REQUIRES the identified effect functional ``b(.)`` to admit a Lipschitz representation
    (e.g. ``b(P)=E_P[phi]``, ``phi`` Lipschitz), so ``|b(P')-b(P)| <= Lip*W1(P,P')``; then the
    regret is quadratic in ``d = W1(P,P')``, the distance ``chc.uncertainty.WassersteinPenalty``
    penalises; (C) a W-DRO controller for radius ``eps`` covers this residual when ``d <= eps`` --
    conditional on the DRO ambiguity set being the W1-ball over the law that ``b`` is Lipschitz in.
    The robust radius is a transportability budget.
    """
    coeff = xt**2 * (rr - b_src**2) ** 2 / (rr + b_src**2) ** 3
    ds = np.geomspace(d_lo, d_hi, n_d)
    transportable = np.zeros(n_d)  # b_tgt = b_src: recoverable, zero regret at any distance
    nontransport = coeff * lip**2 * ds**2  # b_tgt = b_src + Lip*d: CE-quadratic regret

    def exact(b_tgt: float) -> float:
        u_src = b_src * xt / (b_src**2 + rr)  # source-optimal control
        u_tgt = b_tgt * xt / (b_tgt**2 + rr)  # target-optimal control
        c_src = (b_tgt * u_src - xt) ** 2 + rr * u_src**2  # deploy source control on target
        c_tgt = (b_tgt * u_tgt - xt) ** 2 + rr * u_tgt**2  # target-optimal cost
        return c_src - c_tgt

    exact_reg = np.array([exact(b_src + lip * d) for d in ds])
    slope = float(np.polyfit(np.log(ds), np.log(nontransport), 1)[0])
    exact_slope = float(np.polyfit(np.log(ds), np.log(exact_reg), 1)[0])
    return TransportabilityCurve(
        ds,
        transportable,
        nontransport,
        exact_reg,
        float(coeff * lip**2 * wdro_radius**2),
        slope,
        exact_slope,
    )


@dataclass(frozen=True)
class EnsembleControlCurve:
    """One control over a heterogeneous population pays a curvature-weighted Var(u*) floor."""

    spreads: Vector  # heterogeneity level (spread of the effect across the population)
    ensemble_floor: Vector  # R(u_ens): the irreducible regret of one control, ~ spread^2
    naive_mean_regret: Vector  # regret of the naive u*(mean effect) control: >= the ensemble floor
    floor_slope: float  # log-log slope of the floor vs heterogeneity (~2: quadratic)
    homogeneous_floor: float  # floor at the smallest spread (~0: one control serves a uniform pop)
    naive_excess_max: float  # max(naive - ensemble): the value of curvature-weighting over the mean


def ensemble_control_certificate(
    *,
    b0: float = 1.0,
    rr: float = 0.5,
    xt: float = 1.0,
    n_ctx: int = 21,
    spread_lo: float = 0.02,
    spread_hi: float = 0.5,
    n_spread: int = 12,
) -> EnsembleControlCurve:
    """The ENSEMBLE (heterogeneity) control regret floor -- one control over a heterogeneous causal
    population (Li & Khaneja 2006 ensemble control; derived in ``validation/ensemble_control.mac``,
    proved in ``proofs/ensemble_control.v``). When the effect ``b`` varies across contexts (CATE
    heterogeneity), a single control ``u`` serves all. Even with perfect per-context knowledge, one
    control pays an irreducible regret = the curvature-weighted VARIANCE of the per-context optimal
    actions, ``R(u_ens) = W*Var_w(u*)``, quadratic in the heterogeneity and zero for a homogeneous
    population. The ensemble-optimal control is the curvature-weighted mean of per-context optima;
    it WEAKLY beats the naive ``u*(mean effect)`` control (they coincide when curvature is uniform
    across contexts), and strictly beats it under heterogeneous curvature.
    """
    spreads = np.geomspace(spread_lo, spread_hi, n_spread)
    floor = np.zeros(n_spread)
    naive = np.zeros(n_spread)
    for i, s in enumerate(spreads):
        bs = np.linspace(b0 - s, b0 + s, n_ctx)  # heterogeneous effects across the population
        ks = bs**2 + rr  # per-context curvature
        us = bs * xt / ks  # per-context optimal action
        w = ks / n_ctx  # curvature weight (uniform population weight 1/n)
        u_ens = float(np.sum(w * us) / np.sum(w))  # ensemble-optimal: curvature-weighted mean
        floor[i] = float(np.sum(w * (u_ens - us) ** 2))
        u_naive = b0 * xt / (b0**2 + rr)  # naive: optimal for the mean effect b0
        naive[i] = float(np.sum(w * (u_naive - us) ** 2))
    slope = float(np.polyfit(np.log(spreads), np.log(floor), 1)[0])
    return EnsembleControlCurve(
        spreads,
        floor,
        naive,
        slope,
        float(floor[0]),
        float(np.max(naive - floor)),
    )


@dataclass(frozen=True)
class CompositionTransferCurve:
    """Order-p effect estimator -> order-2p control regret: the control map doubles the order."""

    deltas: Vector  # nuisance error delta
    orders: Vector  # estimator orders p (1 = plug-in, 2 = orthogonal/DML, 3 = higher-order)
    regrets: Vector  # exact regret per (order, delta): shape (len(orders), len(deltas))
    slopes: Vector  # measured log-log slope of regret vs delta, per order (~ 2p)
    expected_slopes: Vector  # 2*p: the predicted order-doubling


def composition_transfer_certificate(
    *,
    b: float = 1.0,
    rr: float = 0.5,
    xt: float = 1.0,
    orders: Sequence[int] = (1, 2, 3),
    delta_lo: float = 0.01,
    delta_hi: float = 0.2,
    n_delta: int = 12,
) -> CompositionTransferCurve:
    """The general orthogonal-to-control order-transfer LEMMA -- the general form of Result 0 (which
    only stated p=1 and p=2). Derived in ``validation/composition_transfer.mac``, proved in
    ``proofs/composition_transfer.v``. If the effect estimator has error of order ``delta^p`` (``p``
    the orthogonality order: 1 plug-in, 2 Neyman-orthogonal/DML, higher for higher-order), the
    certainty-equivalence control regret is order ``delta^(2p)`` -- the control map DOUBLES the
    estimator's order for EVERY ``p``, because regret is quadratic in the ACTION error (exact map
    ``(b^2+rr)*(u*(bhat)-u*(b))^2``, not a linearisation) and ``u*`` is Lipschitz in the effect.
    HONEST SCOPE: an immediate COMPOSITION (Lipschitz action-map + quadratic regret), not a deep
    theorem -- the p->2p arithmetic restates known facts (4th-order-under-orthogonality:
    Foster-Syrgkanis Orthogonal Statistical Learning; quadratic LQR CE gap: Mania-Tu-Recht). Its
    value is unifying them for control. The genuine open theorem is the full END-TO-END statement
    (cross-fit causal estimator -> dynamics-error rate -> stabilising controller -> finite-sample
    control regret with explicit constants), of which this is one link. Plug-in (``p=1 -> 2``) and
    DML (``p=2 -> 4``) are instances.
    """

    def u_star(bv: float) -> float:
        return bv * xt / (bv * bv + rr)

    curv = b * b + rr
    deltas = np.geomspace(delta_lo, delta_hi, n_delta)
    ps = np.asarray(orders, dtype=np.float64)
    regrets = np.zeros((len(orders), n_delta))
    slopes = np.zeros(len(orders))
    for i, p in enumerate(orders):
        # an order-p estimator: effect error delta^p; use the EXACT regret map (not C*(bhat-b)^2)
        regrets[i] = np.array([curv * (u_star(b + d**p) - u_star(b)) ** 2 for d in deltas])
        slopes[i] = float(np.polyfit(np.log(deltas), np.log(regrets[i]), 1)[0])
    return CompositionTransferCurve(deltas, ps, regrets, slopes, 2.0 * ps)


@dataclass(frozen=True)
class MultivariateTransferCurve:
    """Transfer theorem in MULTIVARIATE LQ: effect-matrix error delta^p -> LQ regret delta^(2p)."""

    deltas: Vector  # effect-matrix (input matrix B) error scale delta
    regret_order1: Vector  # LQ regret with ||dB|| ~ delta^1 (plug-in): ~ delta^2
    regret_order2: Vector  # LQ regret with ||dB|| ~ delta^2 (orthogonal/DML): ~ delta^4
    slope_order1: float  # log-log slope of regret_order1 vs delta (~2)
    slope_order2: float  # log-log slope of regret_order2 vs delta (~4)


def multivariate_transfer_certificate(
    *,
    delta_lo: float = 0.005,
    delta_hi: float = 0.1,
    n_delta: int = 10,
) -> MultivariateTransferCurve:
    """The transfer theorem (Result 18 / Contribution 1) in the MULTIVARIATE, DYNAMIC LQ setting --
    addressing the "needs multivariate/dynamic" gap. On a stable 2-state/1-input LQ plant the LOCAL
    quadratic certainty-equivalence suboptimality (Mania, Tu & Recht 2019, small-error/stabilising;
    ``certainty_equivalence_gap``) is quadratic in the effect-matrix (input matrix ``B``) error;
    composing an order-``p`` estimator (``||dB|| ~ delta^p``) gives regret ``~ delta^(2p)`` -- the
    SAME order-doubling as scalar Result 18, now for matrices. Rocq core shared (``regret_order_2p``
    is abstract in the error norm). This lifts the order-transfer to matrices; it is NOT the full
    finite-sample end-to-end theorem (which additionally needs nuisance rates + a stability margin).
    """
    a_mat = np.array([[1.0, 0.1], [0.0, 0.95]])
    b_mat = np.array([[0.5], [1.0]])
    q_mat = np.eye(2)
    r_mat = np.array([[0.5]])
    x0 = np.array([1.0, 0.5])
    db_dir = np.array([[1.0], [-0.5]])
    db_dir = db_dir / np.linalg.norm(db_dir)  # unit-norm effect-matrix perturbation direction
    ds = np.geomspace(delta_lo, delta_hi, n_delta)
    reg1 = np.array(
        [
            certainty_equivalence_gap(a_mat, b_mat, q_mat, r_mat, a_mat, b_mat + d * db_dir, x0)
            for d in ds
        ]
    )
    reg2 = np.array(
        [
            certainty_equivalence_gap(a_mat, b_mat, q_mat, r_mat, a_mat, b_mat + d**2 * db_dir, x0)
            for d in ds
        ]
    )
    slope1 = float(np.polyfit(np.log(ds), np.log(reg1), 1)[0])
    slope2 = float(np.polyfit(np.log(ds), np.log(reg2), 1)[0])
    return MultivariateTransferCurve(ds, reg1, reg2, slope1, slope2)


@dataclass(frozen=True)
class MultiChannelControlCurve:
    """Debias EVERY interference channel: regret order set by the least-orthogonalised one."""

    deltas: Vector  # nuisance error delta
    half_orth_regret: Vector  # direct debiased, spillover plug-in: regret ~ delta^2 (bottleneck)
    full_orth_regret: Vector  # both channels debiased: regret ~ delta^4
    half_slope: float  # log-log slope of half-orth regret vs delta (~2)
    full_slope: float  # log-log slope of full-orth regret vs delta (~4)
    cluster_counts: Vector  # number of network clusters G swept (cluster-robustness arm)
    estimation_sd: Vector  # sd of the total-effect estimate across seeds, per G (~ 1/sqrt(G))
    cluster_se_slope: float  # log-log slope of estimation_sd vs G (~ -0.5: effective n = clusters)


def _dml_two_channel(
    z: np.ndarray,
    u: np.ndarray,
    g: np.ndarray,
    y: np.ndarray,
    delta: float,
    orth_spillover: bool,
    fold: np.ndarray,
) -> float:
    """Cross-fit Robinson (partially-linear) DML for two channels (direct u, spillover g). Nuisances
    E[.|z] are fit by OLS on the held-out fold with an added ``delta*z`` systematic error. The
    spillover uses the ORTHOGONAL moment (residualise BOTH treatment & outcome -> O(delta^2) bias)
    when ``orth_spillover``, else PLUG-IN (residualise treatment but NOT outcome -> O(delta)
    bias). The direct channel is always orthogonal. Returns b_d + b_s.
    """

    def resid(t: np.ndarray) -> np.ndarray:
        r = t.astype(np.float64).copy()
        for f in (0, 1):
            tr, te = fold != f, fold == f
            slope, intercept = np.polyfit(z[tr], t[tr], 1)  # cross-fit nuisance E[t|z]
            r[te] = t[te] - ((slope + delta) * z[te] + intercept)  # + delta*z nuisance error
        return r

    u_t, g_t, y_t = resid(u), resid(g), resid(y)
    beta, *_ = np.linalg.lstsq(np.column_stack([u_t, g_t]), y_t, rcond=None)  # joint Robinson
    b_d_hat = float(beta[0])  # direct: always orthogonal, O(delta^2)
    if orth_spillover:
        b_s_hat = float(beta[1])  # spillover orthogonal (outcome residualised): O(delta^2)
    else:
        y1 = y - b_d_hat * u  # direct-adjusted RAW outcome (NOT residualised on z)
        b_s_hat = float(np.dot(g_t, y1) / np.dot(g_t, g_t))  # plug-in: O(delta) bias
    return b_d_hat + b_s_hat


def multichannel_control_certificate(
    *,
    b_d: float = 1.0,
    b_s: float = 0.6,
    rr: float = 0.5,
    xt: float = 1.0,
    alpha_u: float = 1.0,
    alpha_g: float = 0.8,
    gamma: float = 1.0,
    n_clusters: int = 40,
    cluster_size: int = 10,
    tau: float = 0.5,
    noise: float = 0.5,
    deltas: Sequence[float] = (0.02, 0.04, 0.07, 0.12, 0.2, 0.3),
    cluster_grid: Sequence[int] = (10, 20, 40, 80, 160),
    n_seeds: int = 60,
) -> MultiChannelControlCurve:
    """CONTRIBUTION 2 -- MULTI-CHANNEL orthogonal control on a network (the serious form of 4x0;
    derived in ``validation/multichannel_control.mac``; proved: ``multichannel_control.v``).
    The control-relevant total effect of a uniform action is ``B = b_d + b_s`` (direct + spillover):
    two interference channels, each with its own Neyman-orthogonal moment. The NOVELTY is the
    downstream control-regret consequence of incomplete orthogonalisation (DML *under interference*
    is not itself new -- Munro-Xu-Wager, Wager-Xu): orthogonalising ONLY the direct channel caps
    the control regret at ``O(delta^2)`` (the un-orthogonalised spillover bottleneck), while
    orthogonalising BOTH reaches ``O(delta^4)`` -- regret order is ``2*min`` over channel orders.
    ONLY this order-bottleneck is proved in Rocq; the per-channel cross-fit RATES that would make
    ``||b_j-b_j hat|| ~ delta^{p_j}`` rigorous are demonstrated EMPIRICALLY here, not proved.
    Cluster-robust: with roughly balanced, weakly-dependent clusters the effective sample size is
    the number of clusters ``G``, so the total effect concentrates at ``1/sqrt(G)``; this depends
    on the cluster count/size/balance and cross-cluster independence, and is not a universal rate.
    """
    b_total = b_d + b_s
    coeff = xt**2 * (rr - b_total**2) ** 2 / (rr + b_total**2) ** 3
    ds = np.asarray(deltas, dtype=np.float64)
    half = np.zeros(ds.size)
    full = np.zeros(ds.size)

    def simulate(rng: np.random.Generator, gclust: int) -> tuple[np.ndarray, ...]:
        cid = np.repeat(np.arange(gclust), cluster_size)
        n = cid.size
        z = rng.standard_normal(n)
        a = (tau * rng.standard_normal(gclust))[cid]  # cluster random effect (within-cluster dep.)
        u = alpha_u * z + 0.7 * rng.standard_normal(n)
        g = alpha_g * z + 0.7 * rng.standard_normal(n)  # confounded spillover exposure
        y = b_d * u + b_s * g + gamma * z + a + noise * rng.standard_normal(n)
        return z, u, g, y, np.mod(np.arange(n), 2)

    for i, delta in enumerate(ds):
        err_half = np.empty(n_seeds)
        err_full = np.empty(n_seeds)
        for s in range(n_seeds):
            rng = np.random.default_rng(6151 * s + int(1e5 * delta))
            z, u, g, y, fold = simulate(rng, n_clusters)
            err_half[s] = _dml_two_channel(z, u, g, y, delta, False, fold) - b_total
            err_full[s] = _dml_two_channel(z, u, g, y, delta, True, fold) - b_total
        # systematic bias^2 (mean over seeds isolates the delta-order bias from the sampling
        # variance, which is measured separately in the cluster-robustness arm below)
        half[i] = coeff * float(np.mean(err_half)) ** 2
        full[i] = coeff * float(np.mean(err_full)) ** 2
    half_slope = float(np.polyfit(np.log(ds), np.log(half), 1)[0])
    full_slope = float(np.polyfit(np.log(ds), np.log(full), 1)[0])

    gs = np.asarray(cluster_grid, dtype=np.float64)
    est_sd = np.zeros(gs.size)
    for i, gclust in enumerate(cluster_grid):
        ests = np.empty(n_seeds)
        for s in range(n_seeds):
            rng = np.random.default_rng(9377 * s + gclust)
            z, u, g, y, fold = simulate(rng, gclust)
            ests[s] = _dml_two_channel(z, u, g, y, 0.05, True, fold)
        est_sd[i] = float(np.std(ests))
    se_slope = float(np.polyfit(np.log(gs), np.log(est_sd), 1)[0])
    return MultiChannelControlCurve(ds, half, full, half_slope, full_slope, gs, est_sd, se_slope)


@dataclass(frozen=True)
class MultivariateInterferenceCurve:
    """Multi-channel bottleneck in MULTIVARIATE LQ: debias every channel or lose the order."""

    deltas: Vector  # nuisance error delta
    half_orth_regret: Vector  # direct channel debiased, spillover plug-in: LQ regret ~ delta^2
    full_orth_regret: Vector  # both channels debiased: LQ regret ~ delta^4
    half_slope: float  # log-log slope of the half-orth LQ regret (~2)
    full_slope: float  # log-log slope of the full-orth LQ regret (~4)


def multivariate_interference_certificate(
    *,
    delta_lo: float = 0.01,
    delta_hi: float = 0.15,
    n_delta: int = 10,
) -> MultivariateInterferenceCurve:
    """Contribution 2 in the MULTIVARIATE, DYNAMIC LQ setting -- composing multi-channel bottleneck
    (Result 19) with the multivariate LQ gap (Result 21). The total effect is an input MATRIX
    ``B = B_d + B_s`` (two interference channels, each a direction of ``B``). Using the LOCAL
    quadratic LQ suboptimality (Mania, Tu & Recht 2019, small-error + stabilising;
    ``certainty_equivalence_gap``), quadratic in the total input-matrix error: orthogonalising ONLY
    direct (spillover plug-in, ``O(delta)``) caps LQ regret at ``O(delta^2)`` -- spillover is the
    bottleneck; orthogonalising BOTH (each ``O(delta^2)``) reaches ``O(delta^4)``. The scalar
    order-composition is proved in ``proofs/multichannel_control.v``; shown numerically here in
    the multivariate LQ regret. The network cross-fit estimator rates that would make
    ``||dB_j|| ~ delta^{p_j}`` rigorous are NOT proved.
    """
    a_mat = np.array([[1.0, 0.1], [0.0, 0.95]])
    b_mat = np.array([[0.5], [1.0]])
    q_mat = np.eye(2)
    r_mat = np.array([[0.5]])
    x0 = np.array([1.0, 0.5])
    dir_d = np.array([[1.0], [0.0]])  # direct-channel direction of the input matrix B
    dir_s = np.array([[0.0], [1.0]])  # spillover-channel direction
    ds = np.geomspace(delta_lo, delta_hi, n_delta)
    half = np.array(
        [
            certainty_equivalence_gap(
                a_mat, b_mat, q_mat, r_mat, a_mat, b_mat + d**2 * dir_d + d * dir_s, x0
            )
            for d in ds
        ]
    )  # direct O(delta^2), spillover plug-in O(delta)
    full = np.array(
        [
            certainty_equivalence_gap(
                a_mat, b_mat, q_mat, r_mat, a_mat, b_mat + d**2 * (dir_d + dir_s), x0
            )
            for d in ds
        ]
    )  # both channels O(delta^2)
    half_slope = float(np.polyfit(np.log(ds), np.log(half), 1)[0])
    full_slope = float(np.polyfit(np.log(ds), np.log(full), 1)[0])
    return MultivariateInterferenceCurve(ds, half, full, half_slope, full_slope)


@dataclass(frozen=True)
class EndToEndC2Curve:
    """C2 END-TO-END on a clustered LQ network: regret ~ 1/G (sampling floor) + (sum delta^p)^2."""

    g_grid: Vector  # cluster counts G (delta-sweep held at delta_small)
    regret_vs_g: Vector  # mean multivariate-LQ regret from REAL cross-fit DML: ~ 1/G
    g_slope: float  # log-log slope of regret vs G (~ -1: the statistical 1/G floor)
    deltas: Vector  # nuisance error delta (deterministic bias-order sweep, sampling fixed)
    half_regret: Vector  # half-orth (spillover plug-in) LQ regret: ~ delta^2 (bottleneck)
    full_regret: Vector  # full-orth LQ regret: ~ delta^4
    half_slope: float  # ~ 2
    full_slope: float  # ~ 4
    floor_g: float  # regret at the largest G (small delta): the irreducible sampling floor


def end_to_end_c2_certificate(
    *,
    b_d: float = 1.0,
    b_s: float = 0.6,
    alpha_u: float = 1.0,
    alpha_g: float = 0.8,
    gamma: float = 1.0,
    cluster_size: int = 10,
    tau: float = 0.5,
    noise: float = 0.5,
    delta_small: float = 0.002,
    g_grid: Sequence[int] = (10, 20, 40, 80, 160),
    deltas: Sequence[float] = (0.02, 0.04, 0.07, 0.12, 0.2),
    n_seeds: int = 60,
) -> EndToEndC2Curve:
    """CONTRIBUTION 2, END-TO-END: ``multichannel causal estimation -> bottleneck rate -> dynamic
    control regret`` on a clustered network with a multivariate LQ plant (derived in
    ``validation/c2_end_to_end.mac``, control-side reduction proved in ``proofs/c2_end_to_end.v``).
    The cross-fit two-channel Robinson-DML total-effect error is ``||B_hat - B|| = O_p(G^{-1/2} +
    delta_d^{p_d} + delta_s^{p_s})`` (cluster sampling + two nuisance remainders); through the MTR
    local quadratic gap the regret is ``R = O_p[G^{-1} + (delta_d^{p_d} + delta_s^{p_s})^2]``. Two
    regimes are shown:
    (1) a **G-sweep** with REAL cross-fit DML at tiny ``delta`` (sampling-dominated, ``R ~ 1/G``);
    (2) a deterministic **delta-sweep** (half-orth ``~ delta^2``, full-orth ``~ delta^4``).
    Rocq proves the composition (the ``s + e_d + e_s`` reduction); the per-channel cluster cross-fit
    RATES are statistical assumptions (Chernozhukov; Robinson; Hays & Raghavan), not proved.
    """
    b_total = b_d + b_s
    a_mat = np.array([[1.0, 0.1], [0.0, 0.95]])
    b_mat = np.array([[0.5], [1.0]])
    q_mat = np.eye(2)
    r_mat = np.array([[0.5]])
    x0 = np.array([1.0, 0.5])
    dir_d = np.array([[1.0], [0.0]])  # direct-channel direction of the input matrix B
    dir_s = np.array([[0.0], [1.0]])  # spillover-channel direction
    unit = (dir_d - 0.5 * dir_s) / np.linalg.norm(dir_d - 0.5 * dir_s)  # scalar-error -> matrix map

    def simulate(rng: np.random.Generator, gclust: int) -> tuple[np.ndarray, ...]:
        cid = np.repeat(np.arange(gclust), cluster_size)
        n = cid.size
        z = rng.standard_normal(n)
        a = (tau * rng.standard_normal(gclust))[cid]  # within-cluster dependence
        u = alpha_u * z + 0.7 * rng.standard_normal(n)
        g = alpha_g * z + 0.7 * rng.standard_normal(n)
        y = b_d * u + b_s * g + gamma * z + a + noise * rng.standard_normal(n)
        return z, u, g, y, np.mod(np.arange(n), 2)

    def lq_regret(effect_err: float) -> float:
        return certainty_equivalence_gap(
            a_mat, b_mat, q_mat, r_mat, a_mat, b_mat + effect_err * unit, x0
        )

    # (1) G-sweep: real cross-fit DML (full-orth, tiny delta) -> sampling-dominated regret ~ 1/G
    gs = np.asarray(g_grid, dtype=np.float64)
    reg_g = np.zeros(gs.size)
    for i, gc in enumerate(g_grid):
        errs = np.empty(n_seeds)
        for s in range(n_seeds):
            z, u, g, y, fold = simulate(np.random.default_rng(9001 * s + gc), gc)
            errs[s] = _dml_two_channel(z, u, g, y, delta_small, True, fold) - b_total
        reg_g[i] = float(np.mean([lq_regret(e) for e in errs]))  # mean regret ~ Var(err) ~ 1/G
    g_slope = float(np.polyfit(np.log(gs), np.log(reg_g), 1)[0])

    # (2) delta-sweep: deterministic bias order (sampling fixed) -> half ~ delta^2, full ~ delta^4
    ds = np.asarray(deltas, dtype=np.float64)
    half = np.array(
        [
            certainty_equivalence_gap(
                a_mat, b_mat, q_mat, r_mat, a_mat, b_mat + d**2 * dir_d + d * dir_s, x0
            )
            for d in ds
        ]
    )
    full = np.array(
        [
            certainty_equivalence_gap(
                a_mat, b_mat, q_mat, r_mat, a_mat, b_mat + d**2 * (dir_d + dir_s), x0
            )
            for d in ds
        ]
    )
    half_slope = float(np.polyfit(np.log(ds), np.log(half), 1)[0])
    full_slope = float(np.polyfit(np.log(ds), np.log(full), 1)[0])
    return EndToEndC2Curve(
        gs, reg_g, g_slope, ds, half, full, half_slope, full_slope, float(reg_g[-1])
    )


@dataclass(frozen=True)
class ClusteredLowerBoundCurve:
    """Clustered van-Trees LOWER bound: G*regret -> c0 > 0, so the 1/G rate is TIGHT (two-sided)."""

    g_grid: Vector  # cluster counts G
    mean_regret: Vector  # mean multivariate-LQ regret from real cross-fit DML at tiny delta
    g_times_regret: Vector  # G * mean_regret: bounded below by a positive c0 (irreducible)
    plateau_slope: float  # log-log slope of G*regret vs G (~0: flat => regret ~ 1/G on BOTH sides)
    c0_estimate: float  # largest-G plateau value (the kappa0/Ic constant)
    floor_positive: float  # min over G of G*regret (> 0: the uniform positive lower bound)


def clustered_lower_bound_certificate(
    *,
    b_d: float = 1.0,
    b_s: float = 0.6,
    alpha_u: float = 1.0,
    alpha_g: float = 0.8,
    gamma: float = 1.0,
    cluster_size: int = 10,
    tau: float = 0.5,
    noise: float = 0.5,
    delta_small: float = 0.002,
    g_grid: Sequence[int] = (20, 40, 80, 160, 320),
    n_seeds: int = 80,
) -> ClusteredLowerBoundCurve:
    """CONTRIBUTION 2, the LOWER bound (proofs/clustered_van_trees.v): the ``1/G`` sampling regret
    is IRREDUCIBLE, not just an upper bound. By clustered van Trees the effective Fisher info is
    ``I0 + G*Ic``, so ``E[(Bhat-B)^2] >= 1/(I0+G*Ic)``; composing with the lower-Lipschitz regret
    map ``R >= kappa0*(Bhat-B)^2`` gives ``G*E[R] >= kappa0/(I0+Ic) > 0`` for all ``G``, increasing
    toward ``kappa0/Ic`` (Maxima limit). This certificate confirms it: with REAL cross-fit DML (tiny
    ``delta``, sampling-dominated), ``G*regret`` is roughly CONSTANT and bounded below by a positive
    ``c0`` -- so regret ``~ 1/G`` on BOTH sides (tight rate), not ``o(1/G)``. Contrast
    ``end_to_end_c2_certificate``, whose G-sweep showed only the ``~1/G`` upper trend.
    """
    b_total = b_d + b_s
    a_mat = np.array([[1.0, 0.1], [0.0, 0.95]])
    b_mat = np.array([[0.5], [1.0]])
    q_mat = np.eye(2)
    r_mat = np.array([[0.5]])
    x0 = np.array([1.0, 0.5])
    unit = np.array([[1.0], [-0.5]])
    unit = unit / np.linalg.norm(unit)

    def simulate(rng: np.random.Generator, gclust: int) -> tuple[np.ndarray, ...]:
        cid = np.repeat(np.arange(gclust), cluster_size)
        n = cid.size
        z = rng.standard_normal(n)
        a = (tau * rng.standard_normal(gclust))[cid]
        u = alpha_u * z + 0.7 * rng.standard_normal(n)
        g = alpha_g * z + 0.7 * rng.standard_normal(n)
        y = b_d * u + b_s * g + gamma * z + a + noise * rng.standard_normal(n)
        return z, u, g, y, np.mod(np.arange(n), 2)

    def lq_regret(err: float) -> float:
        return certainty_equivalence_gap(a_mat, b_mat, q_mat, r_mat, a_mat, b_mat + err * unit, x0)

    gs = np.asarray(g_grid, dtype=np.float64)
    reg = np.zeros(gs.size)
    for i, gc in enumerate(g_grid):
        errs = np.empty(n_seeds)
        for s in range(n_seeds):
            z, u, g, y, fold = simulate(np.random.default_rng(4127 * s + gc), gc)
            errs[s] = _dml_two_channel(z, u, g, y, delta_small, True, fold) - b_total
        reg[i] = float(np.mean([lq_regret(e) for e in errs]))
    g_times = gs * reg
    plateau_slope = float(np.polyfit(np.log(gs), np.log(g_times), 1)[0])
    return ClusteredLowerBoundCurve(
        gs, reg, g_times, plateau_slope, float(g_times[-1]), float(np.min(g_times))
    )


@dataclass(frozen=True)
class ExposureMapCurve:
    """Exposure-map C2: three channels (direct, spillover, exposure-map W); r_W squared iff orth."""

    deltas: Vector  # nuisance error delta
    full_regret: Vector  # all three channels orthogonalised (each O(delta^2)): LQ regret ~ delta^4
    wbottleneck_regret: (
        Vector  # exposure map W left plug-in (r_W ~ delta): the bottleneck ~ delta^2
    )
    full_slope: float  # ~ 4
    wbottleneck_slope: float  # ~ 2 (an un-orthogonalised W dominates the regret order)


def exposure_map_certificate(
    *,
    delta_lo: float = 0.01,
    delta_hi: float = 0.15,
    n_delta: int = 10,
) -> ExposureMapCurve:
    """CONTRIBUTION 2, exposure-map generalisation (derived in ``validation/exposure_map_c2.mac``,
    proved in ``proofs/exposure_map_c2.v``). The marketplace network plant
    ``x_{t+1} = A x_t + (B_d + B_s W) u_t + eps`` has effective effect ``B_eff(W) = B_d + B_s W``;
    estimating ``(B_d, B_s, W)`` gives THREE channels ``r_d, r_s, r_W`` (the last from the exposure
    map ``W``). Through the MTR gap, ``R = O_p[G^{-1} + (r_d + r_s + r_W)^2]``. Shown
    deterministically on a 3-state/1-input LQ plant (three orthogonal input-matrix directions):
    orthogonalising ALL three (each ``O(delta^2)``) gives ``~ delta^4``; leaving the exposure map
    ``W`` plug-in (``r_W ~ delta``) makes it the **bottleneck**, ``~ delta^2`` -- the load-bearing
    point (Hays & Raghavan) that ``r_W`` enters squared only if ``W`` is orthogonalised/cross-fit.
    """
    a_mat = np.array([[1.0, 0.1, 0.0], [0.0, 0.95, 0.1], [0.0, 0.0, 0.9]])
    b_mat = np.array([[0.5], [1.0], [0.7]])
    q_mat = np.eye(3)
    r_mat = np.array([[0.5]])
    x0 = np.array([1.0, 0.5, 0.3])
    dir_d = np.array([[1.0], [0.0], [0.0]])  # direct-effect channel
    dir_s = np.array([[0.0], [1.0], [0.0]])  # spillover-coefficient channel
    dir_w = np.array([[0.0], [0.0], [1.0]])  # exposure-map (W) channel
    ds = np.geomspace(delta_lo, delta_hi, n_delta)
    full = np.array(
        [
            certainty_equivalence_gap(
                a_mat, b_mat, q_mat, r_mat, a_mat, b_mat + d**2 * (dir_d + dir_s + dir_w), x0
            )
            for d in ds
        ]
    )  # all three channels O(delta^2)
    wbn = np.array(
        [
            certainty_equivalence_gap(
                a_mat, b_mat, q_mat, r_mat, a_mat, b_mat + d**2 * (dir_d + dir_s) + d * dir_w, x0
            )
            for d in ds
        ]
    )  # exposure map W plug-in: r_W ~ delta is the bottleneck
    full_slope = float(np.polyfit(np.log(ds), np.log(full), 1)[0])
    wbn_slope = float(np.polyfit(np.log(ds), np.log(wbn), 1)[0])
    return ExposureMapCurve(ds, full, wbn, full_slope, wbn_slope)


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
    lower_bound: Vector  # 2*sqrt(A*K*T/eta), K = C*sigma^2: the proved sequence lower bound
    adaptive_slope: float  # log-log slope of adaptive regret vs T (~0.5)
    greedy_slope: float  # log-log slope of greedy regret vs T (~1)
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
    ``a=A/eta``, where the Rocq ``C`` is this ``K``), NOT a corollary of the per-round floor.
    The leading term scales as ``1/sqrt(eta)``, where ``eta`` is the INJECTED-EXPLORATION efficiency
    (information per unit exploration variance, distinct from an observational residual fraction):
    smaller ``eta`` provably raises the floor (Rocq ``lower_efficiency_raises_sequence_floor``) --
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
    lb = 2.0 * np.sqrt(a_curv * k_vt * hs / eta)  # 2*sqrt(A*K*T/eta): 1/sqrt(eta) causal factor
    adaptive_slope = float(np.polyfit(np.log(hs), np.log(adaptive), 1)[0])
    greedy_slope = float(np.polyfit(np.log(hs), np.log(greedy), 1)[0])
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
    its own optimal scale, sits at ``sqrt(2)`` times the floor (Rocq ``taper_gap_is_sqrt_two``). The
    burst is optimal *in this model*, where exploration cost is linear in the injected variance and
    unbounded per round; a per-round action cap is what makes a taper the right shape.

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


@dataclass(frozen=True)
class VanTreesCurve:
    """Van Trees: Bayes MSE hits the floor 1/(I_prior+n*I_data); confounding lifts the floor."""

    sample_sizes: Vector  # number of observations n
    empirical_mse: Vector  # Monte-Carlo Bayes risk of the posterior-mean estimator
    van_trees_bound: Vector  # 1/(I_prior + n*I_data): the van-Trees lower bound
    confounded_bound: Vector  # 1/(I_prior + n*Vid*I_data): the (higher) floor under confounding
    tight_ratio: float  # empirical MSE / van-Trees bound at the largest n (~1: tight for Gaussian)


def van_trees_certificate(
    *,
    tau: float = 2.0,
    sigma: float = 1.0,
    v_id: float = 0.4,
    sample_sizes: Sequence[int] = (5, 10, 20, 50, 100, 200),
    n_trials: int = 6000,
) -> VanTreesCurve:
    """The FORMAL van Trees (Bayesian Cramer-Rao) inequality that Result 20 assumed (derived in
    ``validation/van_trees.mac``, proved in ``proofs/van_trees.v``). For a Gaussian conjugate model
    (prior ``N(0,tau^2)``, ``n`` obs ``x = theta + N(0,sigma^2)``) Bayes risk of ANY estimator is
    at least ``1/(I_prior + n*I_data)``, ``I_prior = 1/tau^2``, ``I_data = 1/sigma^2`` -- and the
    posterior-mean estimator HITS it (van Trees is tight for the Gaussian). Confounding reduces the
    per-obs identifying information (fraction ``v_id < 1``), lifting the floor -- the Bayesian /
    sequential analogue of the CR floor of Results 10/12 and the ``C/m_t`` term of Result 20.
    """
    j_prior = 1.0 / tau**2
    i_data = 1.0 / sigma**2
    ns = np.asarray(sample_sizes, dtype=np.float64)
    bound = 1.0 / (j_prior + ns * i_data)
    conf_bound = 1.0 / (j_prior + ns * v_id * i_data)
    mse = np.zeros(ns.size)
    for k, n_obs in enumerate(sample_sizes):
        errs = np.empty(n_trials)
        for t in range(n_trials):
            rng = np.random.default_rng(8263 * t + n_obs)
            theta = tau * rng.standard_normal()  # draw from the prior
            x = theta + sigma * rng.standard_normal(n_obs)  # observations
            post_mean = np.sum(x) / (1.0 / tau**2 + n_obs / sigma**2)  # Bayes estimator (mu0 = 0)
            errs[t] = (post_mean - theta) ** 2
        mse[k] = float(np.mean(errs))
    return VanTreesCurve(ns, mse, bound, conf_bound, float(mse[-1] / bound[-1]))


@dataclass(frozen=True)
class HInfRobustCurve:
    """H-inf robustness = pessimistic contraction, calibratable to ~ the variance-optimal action."""

    gamma_grid: Vector  # adversary energy budget gamma (robustness level)
    robust_gain: Vector  # u_rob(gamma): the cautious control; -> u_ce as gamma -> inf
    inflation_at_uce: Vector  # robust value at fixed u_ce: >= nominal, antitone in gamma (Rocq B,C)
    expected_regret: Vector  # regret of u_rob(gamma) under effect variance s^2 (U-shaped in gamma)
    u_ce: float  # certainty-equivalence gain b*xt/(b^2+rr)
    u_pess_star: float  # variance-optimal gain b*xt/(b^2+rr+s^2) (Result 2, scalar one-step)
    nominal_at_uce: float  # nominal cost at u_ce (the gamma -> inf limit of inflation_at_uce)
    gamma_star: float  # argmin of the expected regret
    gain_at_gamma_star: float  # robust gain at gamma*: ~ u_pess_star (approx, not an identity)
    expected_regret_min: float  # ~ 0: the robust optimum reaches the variance-pessimism optimum
    ce_expected_regret: float  # regret of the (under-cautious) CE control; > expected_regret_min


def hinf_robust_regret_certificate(
    *,
    b: float = 1.0,
    rr: float = 0.5,
    xt: float = 1.0,
    s: float = 0.6,
    gamma_lo: float = 0.72,
    gamma_hi: float = 8.0,
    n_gamma: int = 40,
    n_u: int = 4000,
) -> HInfRobustCurve:
    """H-infinity / differential-game robust control (Geering 2007, Ch 4). Derived in
    ``validation/hinf_robust_regret.mac``, proved in ``proofs/hinf_robust_regret.v``. Confounding
    uncertainty in the effect ``b`` is an adversary perturbing the gain, penalised by budget
    ``gamma^2``; the robust controller solves ``min_u max_w``. This is a *correspondence*, not an
    identity: the worst-case objective ``gamma^2*e^2/(gamma^2-u^2)+rr*u^2`` and the expected
    quadratic objective ``(b*u-xt)^2+(rr+s^2)u^2`` (Result 2, optimum ``u_pess=b*xt/(b^2+rr+s^2)``)
    are different functions. Robustness INDUCES a pessimistic contraction (``u_rob -> u_ce`` as
    ``gamma`` grows), and the regret-optimal ``gamma*`` can be CALIBRATED so ``u_rob(gamma*)``
    approximates ``u_pess`` (0.5415 vs 0.538) -- game-theoretic robustness and statistical pessimism
    as two roads to a similar cautious control, not a proven equality.
    """
    s2 = s * s
    u_ce = b * xt / (b * b + rr)
    u_pess = b * xt / (b * b + rr + s2)  # variance-optimal control (Result 2)
    nominal_uce = (b * u_ce - xt) ** 2 + rr * u_ce * u_ce
    oracle = (b * u_pess - xt) ** 2 + (rr + s2) * u_pess * u_pess  # min expected cost, Var(b)=s^2
    gammas = np.geomspace(gamma_lo, gamma_hi, n_gamma)
    robust_gain = np.zeros(n_gamma)
    inflation = np.zeros(n_gamma)
    exp_regret = np.zeros(n_gamma)
    for i, g in enumerate(gammas):
        g2 = g * g
        us = np.linspace(0.0, 0.999 * g, n_u)  # feasible region |u| < gamma
        j_rob = g2 * (b * us - xt) ** 2 / (g2 - us * us) + rr * us * us
        u_r = float(us[int(np.argmin(j_rob))])
        robust_gain[i] = u_r
        inflation[i] = g2 * (b * u_ce - xt) ** 2 / (g2 - u_ce * u_ce) + rr * u_ce * u_ce
        exp_regret[i] = (b * u_r - xt) ** 2 + (rr + s2) * u_r * u_r - oracle
    k = int(np.argmin(exp_regret))
    return HInfRobustCurve(
        gammas,
        robust_gain,
        inflation,
        exp_regret,
        float(u_ce),
        float(u_pess),
        float(nominal_uce),
        float(gammas[k]),
        float(robust_gain[k]),
        float(exp_regret[k]),
        float((b * u_ce - xt) ** 2 + (rr + s2) * u_ce * u_ce - oracle),
    )


@dataclass(frozen=True)
class ConstrainedRegretCurve:
    """Constrained CE regret is piecewise-quadratic; frozen (zero) on the active side."""

    deltas: Vector  # |effect-estimate error| from the activation threshold
    regret_inactive: Vector  # regret when the estimate is inactive-side: quadratic in delta
    regret_active: Vector  # regret when the estimate is active-side: ~0 (control frozen at umax)
    regret_unconstrained: Vector  # unconstrained regret at the same estimates (for the <= check)
    threshold: float  # activation threshold b_t where u*(b_t) = umax
    inactive_slope: float  # log-log slope of regret vs delta on the inactive side (~2: quadratic)
    active_regret_max: float  # max regret on the active side (~0: curvature collapses)
    max_constrained_ratio: float  # max(constrained / unconstrained) <= 1 (clipping non-expansive)
    pessimism_budget: float  # regret a naive controller pays if truth active but underestimated


def constrained_ce_regret_certificate(
    *,
    xt: float = 1.0,
    rr: float = 1.0,
    umax: float = 0.45,
    delta_lo: float = 0.005,
    delta_hi: float = 0.15,
    n_delta: int = 12,
    active_offset: float = 0.1,
) -> ConstrainedRegretCurve:
    """Constrained certainty-equivalence regret is PIECEWISE-QUADRATIC (Gros & Diehl 2022, Ch 16).
    Derived: ``validation/constrained_ce_regret.mac``; proved: ``proofs/constrained_ce_regret.v``.
    A budget/safety cap ``u <= umax`` clips the control to ``u_opt(b) = min(u*(b), umax)``. With the
    true effect at the activation threshold ``b_t`` (``u*(b_t) = umax``): on the INACTIVE side the
    regret is quadratic in the effect-estimate error; on the ACTIVE side the control freezes at
    ``umax`` so the regret is ZERO -- the curvature collapses, the source of the kink. Clipping is
    non-expansive: constrained regret never exceeds unconstrained. Consequence (ties to Result 9
    partial-ID and the marketplace budget): when the effect interval straddles ``b_t`` the
    pessimism budget must cover the active-set transition -- a naive controller that assumes the
    constraint inactive pays that gap.
    """

    def u_star(bv: float) -> float:
        return bv * xt / (bv * bv + rr)

    def u_opt(bv: float) -> float:
        return min(u_star(bv), umax)

    b_t = (xt - np.sqrt(xt * xt - 4.0 * rr * umax * umax)) / (2.0 * umax)  # lower activation root
    curv = b_t * b_t + rr  # (b^2+rr): the CE-regret curvature prefactor
    deltas = np.geomspace(delta_lo, delta_hi, n_delta)
    reg_in = np.array([curv * (u_opt(b_t - d) - u_opt(b_t)) ** 2 for d in deltas])
    reg_act = np.array([curv * (u_opt(b_t + d) - u_opt(b_t)) ** 2 for d in deltas])
    reg_unc = np.array([curv * (u_star(b_t - d) - u_star(b_t)) ** 2 for d in deltas])
    slope = float(np.polyfit(np.log(deltas), np.log(reg_in), 1)[0])
    ratio_in = reg_in / np.where(reg_unc > 0, reg_unc, 1.0)
    b0 = b_t + active_offset  # true effect in the active region (optimal control = umax)
    naive_regret = np.array([(b0 * b0 + rr) * (u_star(b_t - d) - umax) ** 2 for d in deltas])
    return ConstrainedRegretCurve(
        deltas,
        reg_in,
        reg_act,
        reg_unc,
        float(b_t),
        slope,
        float(reg_act.max()),
        float(ratio_in.max()),
        float(naive_regret.max()),
    )


@dataclass(frozen=True)
class ConfoundedTurnpikeCurve:
    """Confounded control converges to the wrong turnpike; discounted regret stays finite."""

    horizons: Vector  # horizon T
    undiscounted_regret: Vector  # T * per_step: linear, unbounded (sharpens Result 1d)
    discounted_regret: Vector  # sum g^t * per_step = per_step*(1-g^T)/(1-g): bounded
    discounted_bound: float  # per_step/(1-g): the finite discounted-regret cap
    turnpike_offset_formula: float  # xref*beta/(b+beta): the analytic turnpike gap
    turnpike_offset_simulated: float  # xref - x_conf from the loop (matches the formula)
    per_step_regret: float  # q*offset^2: paid every step forever by the confounded controller
    undiscounted_slope: float  # slope of undiscounted regret vs horizon (~ per_step)


def confounded_turnpike_certificate(
    *,
    b: float = 1.0,
    beta: float = 0.3,
    xref: float = 1.0,
    q: float = 1.0,
    g: float = 0.9,
    horizon: int = 200,
) -> ConfoundedTurnpikeCurve:
    """The CONFOUNDED TURNPIKE GAP (Weber 2011, Sec 3.5: turnpike + current-value Hamiltonian).
    Derived in ``validation/confounded_turnpike.mac``, proved in ``proofs/confounded_turnpike.v``. A
    confounded controller using the biased effect ``b_obs = b + beta`` (OVB) converges to the WRONG
    turnpike ``x_conf = b*xref/(b+beta)`` and pays the offset every step. This upgrades Result 1d
    (dynamic horizon regret, previously hand-argued) to a proper turnpike argument, and adds a NEW
    discounted-regret certificate: the undiscounted cumulative regret ``T*c`` is unbounded, but the
    discounted sum ``sum g^t c = c*(1-g^T)/(1-g)`` stays below ``c/(1-g)`` -- finite.
    """
    b_obs = b + beta
    u_conf = xref / b_obs  # confounded control (believes the effect is b_obs)
    x_conf = b * u_conf  # the biased turnpike the loop settles at
    offset_sim = xref - x_conf
    offset_formula = xref * beta / b_obs
    per_step = q * offset_sim**2  # tracking cost paid every step at the biased turnpike
    horizons = np.arange(1, horizon + 1, dtype=np.float64)
    per_step_costs = np.full(horizon, per_step)  # constant control => constant per-step cost
    undiscounted = np.cumsum(per_step_costs)  # = T * per_step (linear)
    disc_weights = g ** np.arange(horizon)
    discounted = np.cumsum(disc_weights * per_step_costs)  # = per_step*(1-g^T)/(1-g)
    slope = float(np.polyfit(horizons, undiscounted, 1)[0])
    return ConfoundedTurnpikeCurve(
        horizons,
        undiscounted,
        discounted,
        float(per_step / (1.0 - g)),
        float(offset_formula),
        float(offset_sim),
        float(per_step),
        slope,
    )


@dataclass(frozen=True)
class PartialIdControlCurve:
    """Partial-ID control: worst-case regret ~ Delta^2; action sign robust iff Delta < |b|."""

    half_widths: Vector  # partial-ID / confounding-budget interval half-widths Delta
    ce_worst_regret: Vector  # worst-case regret of the certainty-equivalent action
    robust_worst_regret: Vector  # worst-case regret of the minimax-robust action (<= the CE one)
    sign_id_threshold: float  # = |b_hat|: the sign-identification threshold (interval excludes 0)
    sign_identified: Vector  # per-Delta: is the action direction robust (interval excludes 0)?


def partial_id_control_certificate(
    *,
    b_hat: float = 1.0,
    rr: float = 0.5,
    xt: float = 1.0,
    half_widths: Sequence[float] = (0.2, 0.5, 0.8, 1.0, 1.5),
    u_points: int = 401,
    b_points: int = 101,
) -> PartialIdControlCurve:
    """Control under PARTIAL IDENTIFICATION: the sign-identification threshold (derived in
    ``validation/partial_id_control.mac``, proved in ``proofs/partial_id_control.v``; grounded in
    Manski partial-ID / Rosenbaum-VanderWeele-Ding sensitivity). When the effect is only interval-
    identified, ``b in [b_hat - Delta, b_hat + Delta]``: (1) the certainty-equivalent action's
    worst-case regret grows like ``Delta^2``, and the minimax action reduces it; (2) the optimal
    action has the sign of ``b``, so its **direction** is identified iff the interval excludes 0 --
    iff ``Delta < |b_hat|``. The critical width ``Delta* = |b_hat|`` is the **sign-identification
    threshold** (a directional identification margin; NOT the sensitivity E-value, whose name it
    deliberately avoids).
    """

    def u_star(b: np.ndarray) -> np.ndarray:
        return xt * b / (b**2 + rr)

    u_ce = float(u_star(np.array(b_hat)))
    span = abs(u_ce) + 0.5
    u_grid = np.linspace(u_ce - span, u_ce + span, u_points)
    widths = np.asarray(half_widths, dtype=np.float64)
    ce_wc = np.zeros(widths.size)
    robust_wc = np.zeros(widths.size)
    sign_ok = np.zeros(widths.size, dtype=bool)
    for i, delta in enumerate(widths):
        b_grid = np.linspace(b_hat - delta, b_hat + delta, b_points)
        u_opt_b = u_star(b_grid)  # per-effect optimum
        curv = b_grid**2 + rr  # regret(u, b) = curv * (u - u_opt(b))^2
        ce_wc[i] = float(np.max(curv * (u_ce - u_opt_b) ** 2))  # CE action's worst case
        reg = curv[None, :] * (u_grid[:, None] - u_opt_b[None, :]) ** 2  # (u, b)
        robust_wc[i] = float(np.min(np.max(reg, axis=1)))  # minimax over the action
        sign_ok[i] = bool((b_grid > 0).all() if b_hat > 0 else (b_grid < 0).all())
    return PartialIdControlCurve(widths, ce_wc, robust_wc, abs(b_hat), sign_ok)


@dataclass(frozen=True)
class DoublyRobustCurve:
    """The AIPW control effect is doubly robust: regret vanishes if either model is correct."""

    errors: Vector  # delta grid (both nuisances misspecified by delta together)
    dr_regret_both: Vector  # AIPW control regret when both nuisances err (~ (delta^2)^2 = delta^4)
    dr_slope: float  # log-log slope of dr_regret_both (~4: product-quartic)
    dr_outcome_ok: float  # AIPW regret with the outcome model correct, propensity wrong (-> 0)
    dr_propensity_ok: float  # AIPW regret with the propensity model correct, outcome wrong (-> 0)
    outcome_reg_fails: float  # outcome-regression regret when its outcome model is wrong (> 0)
    ipw_fails: float  # IPW regret when its propensity model is wrong (> 0)


def _aipw_effect(
    rng: np.random.Generator, n: int, theta: float, dmu: float, de: float, kind: str
) -> float:
    """Estimate a binary-intervention effect from n samples with outcome error ``dmu`` and
    propensity error ``de``; ``kind`` is outcome-regression, IPW, or the doubly-robust AIPW.
    """
    x = rng.standard_normal(n)
    e = 1.0 / (1.0 + np.exp(-0.8 * x))  # true propensity
    d = (rng.random(n) < e).astype(float)
    mu0 = 0.5 * x
    mu1 = mu0 + theta  # true outcome means; ATE = theta
    y = np.where(d > 0.5, mu1, mu0) + 0.3 * rng.standard_normal(n)
    mu1_hat = mu1 + dmu  # misspecified outcome model
    e_hat = e + de * (0.5 - e)  # misspecified propensity (shrink toward 0.5; bounded in (0,1))
    if kind == "outcome":
        return float(np.mean(mu1_hat - mu0))
    if kind == "ipw":
        return float(np.mean(d * y / e_hat - (1.0 - d) * y / (1.0 - e_hat)))
    aug = d * (y - mu1_hat) / e_hat - (1.0 - d) * (y - mu0) / (1.0 - e_hat)
    return float(np.mean((mu1_hat - mu0) + aug))  # AIPW / doubly robust


def doubly_robust_control_certificate(
    *,
    theta: float = 1.0,
    n: int = 100_000,
    n_seeds: int = 8,
    errors: Sequence[float] = (0.4, 0.2, 0.1, 0.05),
    seed: int = 0,
) -> DoublyRobustCurve:
    """The doubly-robust version of result 0 (derived in ``validation/doubly_robust.mac``, proved in
    ``proofs/doubly_robust.v``). For a binary intervention the AIPW estimator's bias is the PRODUCT
    of the outcome-model error and the propensity error, ``dmu*de/(e+de)`` -- so the regret is
    ``O((dmu*de)^2)`` and vanishes if EITHER nuisance model is correct (double robustness), unlike
    outcome-regression (needs the outcome model) or IPW (needs the propensity model).
    """

    def regret(dmu: float, de: float, kind: str) -> float:
        # systematic bias isolated from sampling noise by averaging the signed error over seeds;
        # control regret = kappa*(bias)^2, kappa = 1
        biases = [
            _aipw_effect(np.random.default_rng(1000 * s + 1), n, theta, dmu, de, kind) - theta
            for s in range(n_seeds)
        ]
        return float(np.mean(biases)) ** 2

    err = np.asarray(errors, dtype=np.float64)
    both = np.array([regret(d, d, "aipw") for d in err])  # both nuisances err by delta
    slope = float(np.polyfit(np.log(err), np.log(both), 1)[0])
    return DoublyRobustCurve(
        err,
        both,
        slope,
        regret(0.0, 0.4, "aipw"),  # outcome correct -> AIPW ~ 0
        regret(0.4, 0.0, "aipw"),  # propensity correct -> AIPW ~ 0
        regret(0.4, 0.0, "outcome"),  # outcome model wrong -> outcome-regression fails
        regret(0.0, 0.4, "ipw"),  # propensity model wrong -> IPW fails
    )


@dataclass(frozen=True)
class BanditCausalCurve:
    """Scalar persistently-identified model: deconfounded O(log T) vs confounded Theta(T) regret."""

    rounds: Vector  # cumulative-horizon checkpoints T
    deconfounded_regret: Vector  # cumulative online regret, de-confounded estimator (~ log T)
    confounded_regret: Vector  # cumulative online regret, confounded estimator (~ linear T)
    deconfounded_doubling: float  # cum(T)/cum(T/2) for de-confounded (-> 1: sublinear)
    confounded_doubling: float  # cum(T)/cum(T/2) for confounded (-> 2: linear)


def bandit_causal_certificate(
    *,
    b_true: float = 1.0,
    rr: float = 0.5,
    xt: float = 1.0,
    alpha: float = 1.0,
    gamma: float = 1.0,
    noise: float = 0.5,
    n_rounds: int = 1500,
    n_seeds: int = 12,
    ridge: float = 1e-3,
) -> BanditCausalCurve:
    """The bandit / adaptive-control version (derived in ``validation/bandit_causal.mac``, proved in
    ``proofs/bandit_causal.v``): learn the causal effect ONLINE while controlling. Per round the
    control regret is the certainty-equivalence coefficient ``C`` times the squared estimation err.
    A **de-confounded** online estimator (backdoor on ``z``) is consistent, ``err^2 ~ sigma^2/t``,
    so cumulative regret grows like ``log T`` (sublinear); a **confounded** one has err^2 -> beta^2
    (systematic), a per-round floor, so cumulative regret is linear in ``T``. The doubling
    ratio ``cum(T)/cum(T/2)`` separates them: ``-> 1`` (log) vs ``-> 2`` (linear). Scope: the
    ``O(log T)`` rate is specific to this scalar, persistently-identified setting -- NOT universal
    for adaptive control, where uninformative systems admit ``sqrt(T)`` lower bounds
    (Simchowitz-Foster; Ziemann et al.). The content is the deconfounded-vs-confounded separation.
    """
    coeff = xt**2 * (rr - b_true**2) ** 2 / (rr + b_true**2) ** 3  # per-round regret / error^2
    half = n_rounds // 2  # last two checkpoints are T/2 and T, an exact horizon doubling
    grid = np.geomspace(20, half, 10).astype(int)
    checkpoints = np.unique(np.concatenate([grid, [half, n_rounds]]))
    deconf = np.zeros((n_seeds, checkpoints.size))
    conf = np.zeros((n_seeds, checkpoints.size))
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        gram = ridge * np.eye(3)  # running Gram of features [1, u, z]
        xy = np.zeros(3)  # running [1,u,z] . y
        cum_d, cum_c, j = 0.0, 0.0, 0
        for t in range(1, n_rounds + 1):
            z = rng.standard_normal()
            u = alpha * z + rng.standard_normal()  # confounded excitation
            y = b_true * u + gamma * z + noise * rng.standard_normal()
            feat = np.array([1.0, u, z])
            gram += np.outer(feat, feat)
            xy += feat * y
            b_deconf = np.linalg.solve(gram, xy)[1]  # controls z (backdoor): consistent
            b_conf = np.linalg.solve(gram[:2, :2], xy[:2])[1]  # ignores z: confounded
            cum_d += coeff * (b_deconf - b_true) ** 2
            cum_c += coeff * (b_conf - b_true) ** 2
            if j < checkpoints.size and t == checkpoints[j]:
                deconf[s, j], conf[s, j] = cum_d, cum_c
                j += 1
    d_med, c_med = np.median(deconf, axis=0), np.median(conf, axis=0)
    d_double = float(d_med[-1] / d_med[-2])
    c_double = float(c_med[-1] / c_med[-2])
    return BanditCausalCurve(checkpoints.astype(float), d_med, c_med, d_double, c_double)


@dataclass(frozen=True)
class FiniteHorizonPLCurve:
    """PL self-certifying bound valid over a full finite horizon at mu = lambda_min(Hessian)."""

    horizons: Vector  # horizon lengths T
    mu_min: Vector  # lambda_min of the horizon-objective Hessian (the strong-convexity constant)
    bound_slack: Vector  # min over samples of (PL bound - true regret) -- non-negative = valid
    worst_mode_ratio: Vector  # PL bound / regret along the min-curvature eigenvector (~1, tight)


def finite_horizon_pl_certificate(
    *,
    a: float = 0.9,
    b: float = 1.0,
    q: float = 1.0,
    r: float = 0.5,
    qf: float = 5.0,
    x0: float = 1.0,
    horizons: Sequence[int] = (3, 5, 8, 13, 21),
    n_samples: int = 200,
    seed: int = 0,
) -> FiniteHorizonPLCurve:
    """Finite-horizon (multivariate) version of the PL self-certifying regret bound (result #3),
    without the steady-state / setpoint simplification of the dynamic result 1d. For the *full*
    finite-horizon LQ objective ``J(U)``, ``U = (u_0,...,u_{T-1})``, the regret admits the bound
    ``grad^2 / (2*lambda_min(H))`` (``H`` the Hessian) -- self-certified from the achieved gradient
    over the whole trajectory, no optimum needed. Justified per-eigenmode by
    ``proofs/nonlinear_regret.v`` (`pl_mode_bound`): the min-curvature mode is tight, stiffer modes
    slack. Verifies the bound is valid (upper-bounds the true regret) for random controls and tight
    along the min-curvature eigenvector, across horizons.
    """
    rng = np.random.default_rng(seed)
    hs = np.asarray(horizons, dtype=np.float64)
    mu_min = np.zeros(hs.size)
    slack = np.zeros(hs.size)
    wmr = np.zeros(hs.size)
    for i, tf in enumerate(horizons):
        t = int(tf)
        idx = np.arange(t)
        lag = idx[:, None] - idx[None, :]  # row - col
        g_map = b * np.where(lag >= 0, a ** np.abs(lag), 0.0)  # controllability map (lower-tri)
        free = a ** (idx + 1) * x0  # free response of x_{row+1}
        q_bar = np.diag([q] * (t - 1) + [qf])
        hess = 2 * (g_map.T @ q_bar @ g_map + r * np.eye(t))
        lin = 2 * g_map.T @ q_bar @ free
        u_star = np.linalg.solve(hess, -lin)
        evals, evecs = np.linalg.eigh(hess)
        mu = float(evals[0])
        mu_min[i] = mu

        def cost(
            u: np.ndarray, gm: np.ndarray = g_map, fr: np.ndarray = free, qb: np.ndarray = q_bar
        ) -> float:
            x = fr + gm @ u
            return float(x @ qb @ x + r * (u @ u))

        j_star = cost(u_star)
        worst = np.inf
        for _ in range(n_samples):
            u = u_star + rng.standard_normal(t)
            grad = hess @ (u - u_star)
            worst = min(worst, grad @ grad / (2 * mu) - (cost(u) - j_star))
        slack[i] = float(worst)
        u = u_star + 0.3 * evecs[:, 0]  # perturb along the min-curvature eigenvector
        grad = hess @ (u - u_star)
        wmr[i] = float((grad @ grad / (2 * mu)) / (cost(u) - j_star))
    return FiniteHorizonPLCurve(hs, mu_min, slack, wmr)


@dataclass(frozen=True)
class InterferenceConvexityCurve:
    """Cannibalising interference raises the convexity, so the self-certifying PL bound tightens."""

    cannibalisation: Vector  # kappa_int grid (interference / congestion curvature)
    mu_eff: Vector  # effective strong-convexity mu + kappa_int
    aware_bound: Vector  # interference-aware PL bound grad^2/(2*mu_eff) -- exact for the quadratic
    blind_bound: Vector  # interference-blind PL bound grad^2/(2*mu) -- over-states, loose
    true_regret: Vector  # actual J(u) - J*


def interference_convexity_certificate(
    *,
    mu: float = 1.0,
    a: float = 1.0,
    u_eval: float = 1.5,
    cannibalisations: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
) -> InterferenceConvexityCurve:
    """Strong convexity under interference (combines the PL bound of ``nonlinear_regret`` with the
    interference of §A; derived in ``validation/interference_convexity.mac``, proved in
    ``proofs/interference_convexity.v``). Marketplace interference cannibalises: the benefit of
    incentivising saturates, adding convexity, so the effective strong-convexity rises to
    ``mu + kappa_int``. Since the PL self-certifying bound ``grad^2/(2 mu)`` is antitone in the
    convexity, cannibalising interference makes the certificate **tighter** -- a curse-and-blessing
    duality (interference hurts identification but helps the control certificate). The
    interference-aware bound (using ``mu + kappa_int``) is exact for the quadratic; the blind one
    (using ``mu``) over-states the regret by the growing ratio ``(mu + kappa_int)/mu``.
    """
    k = np.asarray(cannibalisations, dtype=np.float64)
    mu_eff = mu + k
    d = u_eval - a / mu_eff  # deviation of the fixed action from the (interference-shifted) optimum
    grad = mu_eff * d
    aware = grad**2 / (2 * mu_eff)  # interference-aware bound (exact for the quadratic)
    blind = grad**2 / (2 * mu)  # interference-blind bound (over-states, loosens with kappa)
    return InterferenceConvexityCurve(k, mu_eff, aware, blind, 0.5 * mu_eff * d**2)


@dataclass(frozen=True)
class InterferenceOrthogonalCurve:
    """Under interference you must debias BOTH channels: a half measure stays O(eps^2)."""

    errors: Vector  # nuisance-estimation error levels eps
    plugin_exponent: float  # ~2: plug-in both the direct and the spillover effect
    half_orthogonal_exponent: float  # ~2: orthogonalise the direct channel only (spillover wins)
    full_orthogonal_exponent: float  # ~4: orthogonalise both direct and spillover channels


def _channel_effect(
    rng: np.random.Generator,
    n: int,
    eps: float,
    b: float,
    alpha: float,
    gamma: float,
    noise: float,
    *,
    orthogonal: bool,
) -> float:
    """Estimate one confounded channel's effect; single residualisation is O(eps)-biased in the
    nuisance error, double residualisation (orthogonal DML) is O(eps^2)-biased.
    """
    z = rng.standard_normal(n)
    u = alpha * z + rng.standard_normal(n)
    y = b * u + gamma * z + noise * rng.standard_normal(n)
    m_y = (b * alpha + gamma) * z + eps * z
    m_u = alpha * z + eps * z
    u_res = u - m_u
    if orthogonal:
        return float(np.dot(u_res, y - m_y) / np.dot(u_res, u_res))
    return float(np.dot(u_res, y) / np.dot(u_res, u))


def interference_orthogonal_certificate(
    *,
    b_direct: float = 2.0,
    b_int: float = 1.5,
    rr: float = 0.5,
    target: float = 1.0,
    errors: Sequence[float] = (0.2, 0.1, 0.05, 0.025, 0.0125),
    n: int = 300_000,
    n_seeds: int = 6,
    alpha: float = 1.0,
    gamma: float = 1.5,
    noise: float = 0.05,
) -> InterferenceOrthogonalCurve:
    """Interference x orthogonality (combines ``proofs/interference_regret.v`` and
    ``proofs/orthogonal_control.v``; derived in ``validation/interference_orthogonal.mac``). Under
    interference the control effect is ``B = b_direct + b_interference`` (direct + spillover), so
    control regret is quadratic in the total estimation error. Double ML makes a channel's error
    ``O(eps^2)``. The non-obvious consequence: you must orthogonalise **both** channels -- debiasing
    only the direct effect leaves the spillover error at ``O(eps)``, which then dominates the regret
    at ``O(eps^2)``; only double orthogonalisation reaches ``O(eps^4)``. Fits the three slopes
    (~2, ~2, ~4).
    """
    b_total = b_direct + b_int

    def cost(u: float) -> float:
        return (b_total * u - target) ** 2 + rr * u**2

    opt_cost = cost(target * b_total / (b_total**2 + rr))

    def regret(b_hat: float) -> float:
        return cost(target * b_hat / (b_hat**2 + rr)) - opt_cost

    err = np.asarray(errors, dtype=np.float64)
    arms = {"plugin": (False, False), "half": (True, False), "full": (True, True)}
    slopes: dict[str, float] = {}
    for name, (orth_d, orth_i) in arms.items():
        med = np.zeros(err.size)
        for j, eps in enumerate(err):
            regs = []
            for s in range(n_seeds):
                rng = np.random.default_rng(1000 * s + j)
                bd = _channel_effect(rng, n, eps, b_direct, alpha, gamma, noise, orthogonal=orth_d)
                bi = _channel_effect(rng, n, eps, b_int, alpha, gamma, noise, orthogonal=orth_i)
                regs.append(regret(bd + bi))
            med[j] = float(np.median(regs))
        slopes[name] = float(np.polyfit(np.log(err), np.log(np.abs(med)), 1)[0])
    return InterferenceOrthogonalCurve(err, slopes["plugin"], slopes["half"], slopes["full"])


@dataclass(frozen=True)
class DynamicCausalCurve:
    """Confounded tracking regret grows linearly with the horizon; causal control plateaus."""

    horizons: Vector  # swept horizon lengths T
    predictive_regret: Vector  # cumulative predictive-vs-causal tracking regret (grows ~ linearly)
    causal_cost: Vector  # cumulative causal tracking cost (bounded: transient only)
    growth_slope: float  # fitted linear slope of predictive_regret vs T (= per-step floor)


def dynamic_causal_regret_certificate(
    *,
    a: float = 0.7,
    b: float = 1.0,
    beta: float = 0.5,
    x_ref: float = 1.0,
    x0: float = 0.0,
    q: float = 1.0,
    horizons: Sequence[int] = (10, 20, 40, 80, 160),
) -> DynamicCausalCurve:
    """The DYNAMIC confounding theorem (proofs/dynamic_causal_mpc.v, derived in
    ``validation/dynamic_causal_mpc.mac``): a tracking controller ``x' = a x + b u`` using a
    confounded effect estimate ``b_obs = b + beta`` settles at a persistent steady-state offset and
    pays a per-step floor ``q*offset^2`` every step, so its cumulative regret grows *linearly in the
    horizon T* (unbounded), while the causal controller (effect ``b``) tracks exactly and its
    cumulative cost is bounded (the transient only). The fitted growth slope equals the per-step
    floor ``q*(x_ref*beta/b_obs)^2``.
    """
    b_obs = b + beta

    def cumulative_cost(b_hat: float, horizon: int) -> float:
        u_ss = (1.0 - a) * x_ref / b_hat  # feedforward to hold x_ref if the effect were b_hat
        x, cost = x0, 0.0
        for _ in range(horizon):
            cost += q * (x - x_ref) ** 2
            x = a * x + b * u_ss  # true plant
        return cost

    hs = np.asarray(horizons, dtype=np.float64)
    predictive = np.array([cumulative_cost(b_obs, int(t)) for t in hs])
    causal = np.array([cumulative_cost(b, int(t)) for t in hs])
    regret = predictive - causal
    slope = float(np.polyfit(hs, regret, 1)[0])
    return DynamicCausalCurve(hs, regret, causal, slope)


def strong_convexity_regret_bound(grad_norm_sq: float, mu: float) -> float:
    """Self-certifying regret upper bound ``||grad J(u)||^2 / (2 mu)`` for a ``mu``-strongly-convex
    cost -- GLOBAL (beyond the local linearisation), computed from the achieved gradient alone,
    without knowing the optimum. Proved in ``proofs/nonlinear_regret.v`` (the Polyak-Lojasiewicz /
    strong-convexity bound, control-first): ``2*mu*(J(u) - J*) <= ||grad J(u)||^2``.
    """
    return grad_norm_sq / (2.0 * mu)


@dataclass(frozen=True)
class NonlinearRegretCurve:
    """A strong-convexity bound stays valid globally where a linearised estimate under-states."""

    control: Vector  # action grid u (distance from the optimum at 0)
    true_regret: Vector  # J(u) - J*
    pl_bound: Vector  # the self-certifying strong-convexity bound grad^2/(2 mu)
    linearized_estimate: Vector  # a fixed-Hessian local estimate -- under-states away from optimum


def nonlinear_regret_certificate(
    *, mu: float = 1.0, kappa: float = 0.5, u_max: float = 2.0, points: int = 25
) -> NonlinearRegretCurve:
    """On the genuinely nonlinear cost ``J(u) = mu/2 u^2 + kappa u^4`` (minimum at ``u = 0``) the
    strong-convexity bound ``grad^2/(2 mu)`` upper-bounds the true regret *everywhere* (beyond the
    linearisation), while the fixed-Hessian local estimate ``mu/2 u^2`` under-states it away from
    the optimum -- an unsafe certificate exactly where a valid one is needed. This is a control
    specialization of the STANDARD strong-convexity / Polyak-Lojasiewicz certificate, not a new PL
    result. Verifies ``proofs/nonlinear_regret.v`` and ``validation/nonlinear_regret.mac``.
    """
    u = np.linspace(0.0, u_max, points)
    true_regret = mu / 2 * u**2 + kappa * u**4
    grad = mu * u + 4 * kappa * u**3
    return NonlinearRegretCurve(u, true_regret, grad**2 / (2 * mu), mu / 2 * u**2)


@dataclass(frozen=True)
class CausalControlCurve:
    """Predictive control plateaus at a confounding floor; causal control reaches the oracle."""

    sample_sizes: Vector  # swept dataset sizes n
    predictive_regret: Vector  # median regret of the observational (confounded) controller
    causal_regret: Vector  # median regret of the interventional (de-confounded) controller
    predictive_floor: float  # analytic confounding floor (b^2+rr)*(u*(b+beta)-u*(b))^2, fixed in n


def causal_vs_predictive_certificate(
    *,
    b_true: float = 1.0,
    alpha: float = 1.0,
    gamma: float = 1.0,
    target: float = 1.0,
    effort: float = 0.5,
    sample_sizes: Sequence[int] = (500, 2000, 8000, 32000, 128000),
    n_seeds: int = 8,
    noise: float = 0.3,
) -> CausalControlCurve:
    """Empirical proof that predictive control is asymptotically wrong under confounding, while the
    causal controller is consistent (the theorem in ``proofs/causal_mpc.v``, derived in
    ``validation/causal_mpc.mac``; the hardened notebook-01 headline).

    A confounder ``z`` drives both action and outcome (``u = alpha z + noise``,
    ``y = b_true u + gamma z + noise``). The controller hits ``target`` from its estimate of the
    control effect ``b``. The **predictive** controller regresses ``y`` on ``u`` (observational, so
    confounded): as ``n -> inf`` it converges to ``b + beta`` (a fixed omitted-variable bias),
    and its control regret converges to a positive floor that does **not** vanish. The
    **causal** controller partials ``z`` out (backdoor), so its estimate converges to ``b_true`` and
    its regret ``-> 0``. Only causal identification closes the gap.
    """

    def u_star(b: float) -> float:
        return target * b / (b**2 + effort)

    def cost(u: float, b: float) -> float:
        return (b * u - target) ** 2 + effort * u**2

    u_opt = u_star(b_true)

    def regret(b_hat: float) -> float:
        return cost(u_star(b_hat), b_true) - cost(u_opt, b_true)

    beta = gamma * alpha / (alpha**2 + 1.0)  # population OVB with Var(z)=Var(noise_u)=1
    floor = regret(b_true + beta)

    sizes = np.asarray(sample_sizes)
    pred = np.zeros((n_seeds, sizes.size))
    caus = np.zeros((n_seeds, sizes.size))
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        for j, n in enumerate(sizes):
            z = rng.standard_normal(n)
            u = alpha * z + rng.standard_normal(n)
            y = b_true * u + gamma * z + noise * rng.standard_normal(n)
            b_pred = float(np.dot(u, y) / np.dot(u, u))  # observational: confounded
            zc = z - z.mean()
            u_res = u - (np.dot(u, zc) / np.dot(zc, zc)) * zc  # partial z out of u (backdoor)
            y_res = y - (np.dot(y, zc) / np.dot(zc, zc)) * zc
            b_caus = float(np.dot(u_res, y_res) / np.dot(u_res, u_res))
            pred[s, j], caus[s, j] = regret(b_pred), regret(b_caus)
    return CausalControlCurve(sizes, np.median(pred, axis=0), np.median(caus, axis=0), float(floor))


def interference_regret_certificate(
    a: Matrix,
    b: Matrix,
    q: Matrix,
    r: Matrix,
    x0: Vector,
    *,
    interference_ratio: float = 1.0,
    errors: Sequence[float] = _DEFAULT_ERRORS,
    n_samples: int = 400,
    seed: int = 0,
) -> RegretCurve:
    """Interference-aware regret certificate: CE suboptimality is quadratic in the *total* error
    ``eid + eint`` -- the sum of an identification error ``eid`` of the autonomous dynamics (``dA``)
    and an interference / exposure-map error ``eint`` of the actuation channel (``dB``).

    This is the empirical twin of the machine-checked bound in ``proofs/interference_regret.v``
    (``regret <= (kappa C^2 / 2) (eid + eint)^2``) and the plans/20 §A theorem: the interference
    error enters *additively* inside the square, so ignoring it under-states the regret. Sets
    ``eint = interference_ratio * eid``; the recorded error is the additive ``||dA|| + ||dB||`` (not
    the Euclidean norm), so the fitted slope tests the quadratic-in-total law directly (~2).
    """
    rng = np.random.default_rng(seed)
    median_errors: list[float] = []
    median_gaps: list[float] = []
    log_err: list[float] = []
    log_gap: list[float] = []
    for eps in errors:
        eid, eint = eps, interference_ratio * eps
        errs, gaps = [], []
        for _ in range(n_samples):
            d_a = eid * rng.normal(size=a.shape)  # identification error (autonomous dynamics)
            d_b = eint * rng.normal(size=b.shape)  # interference / exposure-map error (actuation)
            try:
                gap = certainty_equivalence_gap(a, b, q, r, a + d_a, b + d_b, x0)
            except (np.linalg.LinAlgError, ValueError):
                continue
            total = float(np.sqrt(np.sum(d_a**2)) + np.sqrt(np.sum(d_b**2)))  # additive eid + eint
            if np.isfinite(gap) and gap > 0.0 and total > 0.0:
                errs.append(total)
                gaps.append(gap)
        if errs:
            median_errors.append(float(np.median(errs)))
            median_gaps.append(float(np.median(gaps)))
            log_err.extend(np.log(errs))
            log_gap.extend(np.log(gaps))
    exponent = float(np.polyfit(log_err, log_gap, 1)[0]) if log_err else float("nan")
    return RegretCurve(np.array(median_errors), np.array(median_gaps), exponent)


# --- Result 33: confounding-robust LQ regret (bounded density ratio -> CVaR radius -> floor) ---


def _lq_static_optimum(effect: float, effort: float, target: float) -> float:
    """Optimal static control ``u*(b) = b*target/(b^2 + r)`` for ``(b*u - target)^2 + r*u^2``."""
    return effect * target / (effect**2 + effort)


def lq_regret_sensitivity(effect: float, effort: float, target: float) -> float:
    """Second-order regret sensitivity ``L_reg = target^2*(r-b^2)^2/(b^2+r)^3`` of the LQ toy.

    The leading coefficient of the certainty-equivalence regret in the effect error: applying the
    optimum for a wrong effect ``b_hat = b + delta`` costs ``L_reg*delta^2 + O(delta^3)`` on the
    true plant (order-doubling -- linear effect error, quadratic control regret). Verified against
    the exact regret in ``validation/confounding_lq_regret.mac`` / the certificate below.
    Nonnegative; zero at the degenerate ``r = b^2`` (where ``u*`` is locally flat in the effect).
    """
    return target**2 * (effort - effect**2) ** 2 / (effect**2 + effort) ** 3


def confounding_robust_lq_regret(
    regret_sensitivity: float, stat_error: float, confounding_halfwidth: float
) -> float:
    """Confounding-robust LQ regret bound ``L_reg*(eps_stat + Delta)^2`` (Rocq ``cr_regret``).

    ``Delta`` is the confounding half-width on the effect estimate -- the §32 inflation
    (:func:`chc.uncertainty.confounding_robust_inflation`), an *irreducible* bias floor that does
    not vanish with sample size. Pushed through the order-doubling regret map it becomes a
    control-regret floor ``L_reg*Delta^2``: the LOCAL LQ control regret is SECOND-order in the
    confounding-induced effect bias (Rocq ``floor_below_linear``: ``L_reg*Delta^2<=L_reg*Delta`` on
    ``[0,1]``) -- a local-toy statement, not a general robustness claim. ``Delta=0`` (``Gamma=1``,
    point identification) recovers the statistical regret ``L_reg*eps^2``.
    """
    return regret_sensitivity * (stat_error + confounding_halfwidth) ** 2


def confounding_robust_lq_regret_matrix(
    curvature: Matrix, stat_error: Matrix, confounding: Matrix
) -> float:
    """Matrix confounding-robust LQ regret ``tr(E^T H E)``, ``E = stat + confounding``.

    The Frobenius lift of the scalar bound via the §21 matrix order-doubling (Rocq
    ``regret_order_2p``): for a positive-semidefinite LQ regret curvature ``H`` and effect-error
    matrix ``E``, the regret is the quadratic form ``tr(E^T H E) = O(||E||_F^2)``. The confounding
    floor (``stat = 0``) is ``tr(Delta^T H Delta)`` -- SECOND order in the Frobenius norm of the
    matrix confounding half-width, the same second-order robustness as the scalar §33. Reduces to
    :func:`confounding_robust_lq_regret` in the ``1x1`` case.
    """
    e = np.asarray(stat_error, dtype=np.float64) + np.asarray(confounding, dtype=np.float64)
    h = np.asarray(curvature, dtype=np.float64)
    return float(np.trace(e.T @ h @ e))


@dataclass(frozen=True)
class ConfoundingRobustLQRegretCurve:
    """Evidence the confounding regret bound order-doubles effect error and floors at Delta^2."""

    gammas: Vector  # swept §32 sensitivity levels Gamma
    regret_bounds: Vector  # R(Gamma) = L_reg*(eps + Delta(Gamma))^2
    statistical: float  # R(Gamma=1) = L_reg*eps^2 (no confounding penalty)
    floor_quadratic_ratio: float  # floor(2*Delta)/floor(Delta): ~4.0 => quadratic in the width
    order_doubling_ratio: float  # exact regret / (L_reg*delta^2) at small delta: ~1.0
    ok: bool


def confounding_robust_lq_regret_certificate(
    effect: float = 1.3,
    effort: float = 0.4,
    target: float = 1.0,
    stat_error: float = 0.02,
    cvar_gap: float = 0.5,
    gammas: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 5.0),
) -> ConfoundingRobustLQRegretCurve:
    """Confirm ``L_reg`` matches the exact toy regret and the Gamma sweep floors quadratically.

    Grounds the analytic ``lq_regret_sensitivity`` against the exact static-LQ regret
    ``kappa*(u*(b_hat) - u*(b))^2`` at a small perturbation (ratio -> 1 proves the coefficient),
    then maps each ``Gamma`` to its confounding half-width via the §32 inflation and checks the
    bound is monotone, recovers the statistical regret at ``Gamma=1``, and floors at ``Delta^2``.
    """
    from chc.uncertainty import (
        confounding_robust_inflation,  # §32 half-width; local to keep JAX out
    )

    l_reg = lq_regret_sensitivity(effect, effort, target)

    # order-doubling: exact regret vs the quadratic coefficient at a small effect perturbation
    delta = 1e-4
    kappa = effect**2 + effort
    u_hat = _lq_static_optimum(effect + delta, effort, target)
    u_star = _lq_static_optimum(effect, effort, target)
    exact = kappa * (u_hat - u_star) ** 2
    order_doubling_ratio = exact / (l_reg * delta**2)

    halfwidths = [confounding_robust_inflation(cvar_gap, 0.0, g) for g in gammas]
    bounds = [confounding_robust_lq_regret(l_reg, stat_error, d) for d in halfwidths]
    statistical = confounding_robust_lq_regret(l_reg, stat_error, 0.0)

    base = confounding_robust_lq_regret(l_reg, 0.0, cvar_gap)  # pure-confounding floor at width=gap
    doubled = confounding_robust_lq_regret(l_reg, 0.0, 2.0 * cvar_gap)
    floor_ratio = doubled / base

    monotone = all(bounds[i] <= bounds[i + 1] + 1e-15 for i in range(len(bounds) - 1))
    ok = (
        abs(order_doubling_ratio - 1.0) < 1e-3  # L_reg is the true leading coefficient
        and abs(bounds[0] - statistical) < 1e-15  # Gamma=1 recovers the statistical regret
        and monotone
        and abs(floor_ratio - 4.0) < 1e-9  # floor is quadratic in the confounding half-width
    )
    return ConfoundingRobustLQRegretCurve(
        gammas=np.array(gammas),
        regret_bounds=np.array(bounds),
        statistical=statistical,
        floor_quadratic_ratio=float(floor_ratio),
        order_doubling_ratio=float(order_doubling_ratio),
        ok=ok,
    )


# --- Result 35: minimax confounding-robust controller under asymmetric loss ---


def certainty_equivalence_control(effect_estimate: float, target: float) -> float:
    """CE control ``u = target/b_hat`` -- centres the outcome ``b*u`` at target under ``b_hat``."""
    return target / effect_estimate


def confounding_robust_control(
    effect_estimate: float,
    halfwidth: float,
    target: float,
    overshoot_penalty: float,
    undershoot_penalty: float,
) -> float:
    """Minimax control over the §32 effect interval ``[b_hat +/- halfwidth]`` under asymmetric loss.

    Loss ``alpha*(y-target)_+ + beta*(target-y)_+`` on the outcome ``y = b*u``; the controller does
    not know ``b`` in the confounding interval. The minimax ``u`` balances the two weighted worst
    tails (``validation/confounding_robust_control.mac``): ``u = (alpha+beta)*target /
    (alpha*(b_hat+D) + beta*(b_hat-D))``. Equals ``u_CE/(1 + kappa*D)`` with
    ``kappa = (alpha-beta)/((alpha+beta)*b_hat)`` -- the pessimism radius ``D`` SHIFTS the gain, a
    sign DICHOTOMY (Rocq): ``a>=b`` -> ``kappa>=0`` -> ``u_rob<=u_CE``, conservative
    (``shift_factor_nonneg`` / ``robust_gain_conservative``); ``a<=b`` -> ``kappa<=0`` ->
    ``u_rob>=u_CE``, aggressive (``shift_factor_nonpos`` / ``robust_gain_aggressive``). CE when
    symmetric (``a=b`` -> ``kappa=0``). Here ``a=overshoot_penalty``, ``b=undershoot_penalty``.

    SCOPE: a minimax STATIC tracking toy, not a general robust controller -- assumes
    ``b_hat > halfwidth > 0`` (the effect interval has an identified sign, so the denominator is
    positive), ``target > 0``, ``alpha, beta > 0``, and NO effort/control penalty (pure outcome
    tracking). The dynamic/effort-penalised case is out of scope.
    """
    return (
        (overshoot_penalty + undershoot_penalty)
        * target
        / (
            overshoot_penalty * (effect_estimate + halfwidth)
            + undershoot_penalty * (effect_estimate - halfwidth)
        )
    )


def worst_case_asymmetric_loss(
    control: float,
    effect_estimate: float,
    halfwidth: float,
    target: float,
    overshoot_penalty: float,
    undershoot_penalty: float,
) -> float:
    """Worst-case asymmetric loss of applying ``control`` over the §32 effect interval.

    For ``u > 0`` the outcome is monotone in ``b``, so the worst overshoot is at ``b_hat+D`` and the
    worst undershoot at ``b_hat-D``; the worst-case loss is the max of the two weighted tails.
    """
    overshoot = max(0.0, (effect_estimate + halfwidth) * control - target)
    undershoot = max(0.0, target - (effect_estimate - halfwidth) * control)
    return max(overshoot_penalty * overshoot, undershoot_penalty * undershoot)


def asymmetric_control_improvement(
    effect_estimate: float,
    halfwidth: float,
    target: float,
    overshoot_penalty: float,
    undershoot_penalty: float,
) -> float:
    """Piecewise worst-case-loss improvement ``W_CE - W_rob`` of the §35 robust controller over CE.

    ``W_rob = 2*a*b*target*D/Q`` (branch-free, ``Q = (a+b)*b_hat + (a-b)*D``), but
    ``W_CE = max(a,b)*target*D/b_hat``, so the improvement is PIECEWISE (reviewer-8): the
    OVERSHOOT-dominant branch ``a*target*D*(a-b)*(b_hat+D)/(b_hat*Q)`` for ``a >= b`` and the
    UNDERSHOOT-dominant branch ``b*target*D*(b-a)*(b_hat-D)/(b_hat*Q)`` for ``b >= a`` (Rocq
    ``rob_gap_*`` / ``rob_gap_under_*``). Both are ``>= 0`` for ``b_hat > D > 0`` and ``> 0`` iff
    ``a != b`` and ``D > 0`` -- the old single formula only covered ``a >= b`` (missing Result 37's
    ``b = 4*a`` regime). Here ``a = overshoot_penalty``, ``b = undershoot_penalty``.
    """
    a, b, bhat, d = overshoot_penalty, undershoot_penalty, effect_estimate, halfwidth
    q = (a + b) * bhat + (a - b) * d
    if a >= b:
        return a * target * d * (a - b) * (bhat + d) / (bhat * q)
    return b * target * d * (b - a) * (bhat - d) / (bhat * q)


@dataclass(frozen=True)
class ConfoundingRobustControlCurve:
    """Evidence the pessimism radius shifts the gain and strictly beats CE under asymmetric loss."""

    u_certainty_equivalence: float
    u_robust: float  # shifted by the pessimism radius (< u_CE when overshoot is costlier)
    worst_case_loss_ce: float
    worst_case_loss_robust: float  # < CE: pessimism strictly helps under asymmetry
    numeric_argmin: float  # grid-search argmin of the worst-case loss (matches u_robust)
    symmetric_equals_ce: bool  # alpha=beta -> robust control == target/b_hat, THIS loss's CE action
    ok: bool


def confounding_robust_control_certificate(
    effect_estimate: float = 1.3,
    halfwidth: float = 0.25,
    target: float = 1.0,
    overshoot_penalty: float = 3.0,
    undershoot_penalty: float = 1.0,
) -> ConfoundingRobustControlCurve:
    """Confirm the closed-form minimax control is the numeric argmin and strictly beats CE.

    Grid-searches the worst-case loss over ``u`` (the analytic ``u_robust`` should be its argmin),
    then checks the robust control is shifted below CE and its worst-case loss is strictly smaller,
    while a symmetric loss reproduces CE exactly.
    """
    u_ce = certainty_equivalence_control(effect_estimate, target)
    u_rob = confounding_robust_control(
        effect_estimate, halfwidth, target, overshoot_penalty, undershoot_penalty
    )
    w_ce = worst_case_asymmetric_loss(
        u_ce, effect_estimate, halfwidth, target, overshoot_penalty, undershoot_penalty
    )
    w_rob = worst_case_asymmetric_loss(
        u_rob, effect_estimate, halfwidth, target, overshoot_penalty, undershoot_penalty
    )

    grid = np.linspace(0.5 * u_rob, 1.5 * u_ce, 40_001)
    losses = [
        worst_case_asymmetric_loss(
            float(u), effect_estimate, halfwidth, target, overshoot_penalty, undershoot_penalty
        )
        for u in grid
    ]
    numeric_argmin = float(grid[int(np.argmin(losses))])

    u_sym = confounding_robust_control(effect_estimate, halfwidth, target, 1.0, 1.0)
    symmetric_equals_ce = abs(u_sym - u_ce) < 1e-12

    ok = (
        abs(u_rob - numeric_argmin) < 1e-3  # the closed form is the minimax control
        and u_rob < u_ce  # overshoot costlier -> shift the gain down
        and w_rob < w_ce  # pessimism strictly reduces the worst-case loss
        and symmetric_equals_ce
    )
    return ConfoundingRobustControlCurve(
        u_certainty_equivalence=u_ce,
        u_robust=u_rob,
        worst_case_loss_ce=w_ce,
        worst_case_loss_robust=w_rob,
        numeric_argmin=numeric_argmin,
        symmetric_equals_ce=symmetric_equals_ce,
        ok=ok,
    )


# --- Result 36: empirical confounding-regret floor (the §33 Delta^2 floor on synthetic data) ---


@dataclass(frozen=True)
class ConfoundingRegretFloorCurve:
    """Empirical certificate: control regret vs confounding bias, and its log-log slope (~2)."""

    biases: Vector  # realised confounding bias |b_hat - b_true| at each confounding level
    regrets: Vector  # realised control regret of the CE controller on the true plant
    exponent: float  # fitted log-log slope of regret vs bias (~2.0 => the Delta^2 floor)
    analytic_ratio: float  # empirical regret / (L_reg * bias^2) at the cleanest level (~1.0)
    ok: bool


def confounding_regret_floor_certificate(
    b_true: float = 1.3,
    effort: float = 0.4,
    target: float = 1.0,
    confounding_levels: Sequence[float] = (0.02, 0.04, 0.06, 0.08, 0.10),
    action_noise: float = 0.6,
    n: int = 200_000,
    seed: int = 0,
) -> ConfoundingRegretFloorCurve:
    """Show the §33 ``Delta^2`` regret floor emerges from real confounded data, not just algebra.

    A confounder ``z`` drives both the action (``u = z + noise``) and the outcome
    (``y = b_true*u + gamma*z + noise``). Naive OLS of ``y`` on ``u`` returns an effect biased by
    the confounding ``gamma`` -- an *irreducible* ``Delta`` that does NOT vanish with ``n``. The CE
    controller uses that biased effect on the true plant, and its measured regret scales as
    ``Delta^2`` (log-log slope ~2), matching the analytic ``L_reg*Delta^2`` -- the floor.
    """
    rng = np.random.default_rng(seed)
    kappa = b_true**2 + effort
    l_reg = lq_regret_sensitivity(b_true, effort, target)
    biases: list[float] = []
    regrets: list[float] = []
    for gamma in confounding_levels:
        z = rng.standard_normal(n)
        u = z + action_noise * rng.standard_normal(n)  # confounder drives the action
        y = b_true * u + gamma * z + action_noise * rng.standard_normal(n)  # ...and the outcome
        b_hat = float(np.cov(u, y)[0, 1] / np.var(u))  # naive OLS slope: biased by the confounding
        bias = abs(b_hat - b_true)
        regret = (
            kappa
            * (
                _lq_static_optimum(b_hat, effort, target)
                - _lq_static_optimum(b_true, effort, target)
            )
            ** 2
        )
        biases.append(bias)
        regrets.append(regret)

    exponent = float(np.polyfit(np.log(biases), np.log(regrets), 1)[0])
    analytic_ratio = regrets[-1] / (
        l_reg * biases[-1] ** 2
    )  # cleanest (largest, least relative noise)
    ok = abs(exponent - 2.0) < 0.15 and abs(analytic_ratio - 1.0) < 0.1
    return ConfoundingRegretFloorCurve(
        biases=np.array(biases),
        regrets=np.array(regrets),
        exponent=exponent,
        analytic_ratio=float(analytic_ratio),
        ok=ok,
    )


# --- Result 37: grounding §35 on a synthetic marketplace task (estimate -> control pipeline) ---


def _asymmetric_cost(
    outcome: float, target: float, overshoot_penalty: float, undershoot_penalty: float
) -> float:
    """Realised asymmetric business cost of a completion ``outcome`` vs the service ``target``."""
    return overshoot_penalty * max(0.0, outcome - target) + undershoot_penalty * max(
        0.0, target - outcome
    )


def _confounded_effect_estimate(
    b_true: float,
    confounding: float,
    incentive_demand_corr: float,
    action_noise: float,
    n: int,
    rng: np.random.Generator,
) -> float:
    """Naive OLS incentive->completions slope on confounded observational logs (biased by demand).

    A demand shock ``z`` drives the historical incentive (``u = corr*z + noise``, past policy raises
    incentive on busy periods) AND completions (``y = b_true*u + confounding*z + noise``).
    Regressing ``y`` on ``u`` ignoring ``z`` returns an effect biased upward by the confounding.
    """
    z = rng.standard_normal(n)
    u = incentive_demand_corr * z + action_noise * rng.standard_normal(n)
    y = b_true * u + confounding * z + action_noise * rng.standard_normal(n)
    return float(np.cov(u, y)[0, 1] / np.var(u))


@dataclass(frozen=True)
class MarketplaceControlCurve:
    """§35 robust controller vs CE on realised marketplace cost across a confounding grid."""

    confounding_levels: Vector  # swept true (unknown) confounding strengths
    ce_costs: Vector  # mean realised asymmetric cost of the CE controller at each level
    robust_costs: Vector  # ...of the §35 confounding-robust controller
    ce_worst_case: float  # max CE cost over the sweep (its downside)
    robust_worst_case: (
        float  # max robust cost over the sweep (<< CE: pessimism bounds the downside)
    )
    savings_at_target_pct: float  # cost reduction at the realistic target confounding level
    unconfounded_premium_pct: float  # extra cost at zero confounding, normalised by the CE
    # DOWNSIDE (ce_worst_case), not by the unconfounded CE cost -- against that near-zero
    # baseline the same gap is an order of magnitude larger. Both fields are here; divide
    # by whichever the claim needs and say which.
    ok: bool


def confounding_robust_control_benchmark(
    b_true: float = 2.0,
    target: float = 1.0,
    overshoot_penalty: float = 1.0,
    undershoot_penalty: float = 4.0,
    sensitivity_gamma: float = 2.5,
    target_confounding: float = 0.8,
    confounding_levels: Sequence[float] = (0.0, 0.4, 0.8, 1.2),
    incentive_demand_corr: float = 1.0,
    action_noise: float = 0.6,
    n_markets: int = 300,
    n_periods: int = 400,
    seed: int = 0,
) -> MarketplaceControlCurve:
    """Ground §35: does the confounding-robust controller beat CE on a full marketplace pipeline?

    Synthetic OBSERVATIONAL confounded marketplace (not a randomised switchback -- the action
    follows the demand confounder z, so this is confounded logging, not an experiment): the naive
    effect estimate is biased, so the
    CE controller (trusting it) under-incentivises and misses completions -- expensive when churn
    (``undershoot_penalty``) dominates budget waste (``overshoot_penalty``). The §35 controller uses
    an assumed sensitivity ``Gamma`` and a half-width ``D``. **Calibration (named):** §32 gives
    ``D = (Gamma-1)/(Gamma+1)*(CVaR_up - CVaR_lo)``; for THIS synthetic benchmark we calibrate the
    sensitivity gap to the estimate, ``CVaR_up - CVaR_lo := b_hat``, so ``D = (Gamma-1)/(Gamma+1)*
    b_hat`` -- a benchmark assumption, not a general consequence of §32.

    Reports realised cost across a confounding sweep: the robust controller bounds the WORST-CASE
    cost (pessimism) and wins **beyond a problem-dependent confounding threshold**, while paying a
    measurable premium **near zero confounding** (its conservatism then costs more than it saves) --
    the honest robustness trade-off, not a universal win.
    """
    from chc.uncertainty import (
        confounding_robust_inflation,  # §32 half-width; local to keep JAX out
    )

    rng = np.random.default_rng(seed)
    ce_costs: list[float] = []
    robust_costs: list[float] = []
    for conf in confounding_levels:
        ce_acc: list[float] = []
        rob_acc: list[float] = []
        for _ in range(n_markets):
            b_hat = _confounded_effect_estimate(
                b_true, conf, incentive_demand_corr, action_noise, n_periods, rng
            )
            halfwidth = confounding_robust_inflation(b_hat, 0.0, sensitivity_gamma)
            u_ce = certainty_equivalence_control(b_hat, target)
            u_rob = confounding_robust_control(
                b_hat, halfwidth, target, overshoot_penalty, undershoot_penalty
            )
            ce_acc.append(
                _asymmetric_cost(b_true * u_ce, target, overshoot_penalty, undershoot_penalty)
            )
            rob_acc.append(
                _asymmetric_cost(b_true * u_rob, target, overshoot_penalty, undershoot_penalty)
            )
        ce_costs.append(float(np.mean(ce_acc)))
        robust_costs.append(float(np.mean(rob_acc)))

    levels = list(confounding_levels)
    ce_worst = max(ce_costs)
    robust_worst = max(robust_costs)
    ti = levels.index(target_confounding)
    savings = 100.0 * (1.0 - robust_costs[ti] / ce_costs[ti]) if ce_costs[ti] > 0 else 0.0
    zi = levels.index(0.0)
    premium = 100.0 * (robust_costs[zi] - ce_costs[zi]) / ce_worst if ce_worst > 0 else 0.0
    return MarketplaceControlCurve(
        confounding_levels=np.array(levels),
        ce_costs=np.array(ce_costs),
        robust_costs=np.array(robust_costs),
        ce_worst_case=ce_worst,
        robust_worst_case=robust_worst,
        savings_at_target_pct=savings,
        unconfounded_premium_pct=premium,
        ok=(robust_worst < ce_worst and savings > 0.0),
    )


# --- Result 38: DYNAMIC grounding of §35 -- a real closed-loop controller on a confounded plant ---


def confounding_robust_tracking_loop(
    a: float,
    b_true: float,
    b_hat: float,
    halfwidth: float,
    x_target: float,
    overshoot_penalty: float,
    undershoot_penalty: float,
    x0: float = 0.0,
    n_steps: int = 30,
    noise_scale: float = 0.05,
    seed: int = 0,
) -> tuple[Vector, Vector, float]:
    """Roll the confounded scalar plant ``x' = a*x + b_true*u + noise`` under the §35 controller.

    Lifts the STATIC §35 minimax into a receding-horizon CLOSED LOOP (the reviewers' recurring
    critique): each step aims the one-step target ``tau_t = x_target - a*x_t`` with
    :func:`confounding_robust_control` trusting the biased offline ``b_hat`` while the TRUE plant
    uses ``b_true`` (the model/plant split of :func:`chc.flagship.closed_loop`, in NumPy to keep
    this module x64-flag-independent). ``halfwidth = 0`` recovers certainty-equivalence exactly (the
    §35 formula collapses to ``tau/b_hat``); ``halfwidth > 0`` (the §32 radius) is the robust
    controller. Returns the realised trajectory, the applied controls, and the accumulated
    ASYMMETRIC stage cost ``Sum_t [alpha*(x_t - x_target)_+ + beta*(x_target - x_t)_+]``.
    """
    rng = np.random.default_rng(seed)
    x = x0
    xs = [x0]
    us: list[float] = []
    cost = 0.0
    for _ in range(n_steps):
        tau = x_target - a * x  # the one-step target change the controller must produce via b*u
        u = confounding_robust_control(b_hat, halfwidth, tau, overshoot_penalty, undershoot_penalty)
        x = a * x + b_true * u + noise_scale * float(rng.standard_normal())
        cost += _asymmetric_cost(x, x_target, overshoot_penalty, undershoot_penalty)
        xs.append(x)
        us.append(u)
    return np.array(xs), np.array(us), cost


@dataclass(frozen=True)
class DynamicConfoundingCurve:
    """§35 robust vs CE on realised CLOSED-LOOP cost across a confounding grid (the dynamic §37)."""

    confounding_levels: Vector  # swept true (unknown) confounding strengths
    ce_costs: Vector  # mean realised closed-loop asymmetric cost of the CE controller
    robust_costs: Vector  # ...of the §35 confounding-robust controller
    ce_worst_case: float  # max CE cost over the sweep
    robust_worst_case: float  # max robust cost over the sweep (<= CE: pessimism bounds downside)
    savings_at_target_pct: float  # closed-loop cost reduction at the realistic confounding level
    unconfounded_premium_pct: float  # extra cost at zero confounding, normalised by the CE
    # DOWNSIDE (ce_worst_case), not by the unconfounded CE cost -- against that near-zero
    # baseline the same gap is an order of magnitude larger. Both fields are here; divide
    # by whichever the claim needs and say which.
    ok: bool


def confounding_robust_tracking_benchmark(
    a: float = 0.6,
    b_true: float = 2.0,
    x_target: float = 1.0,
    overshoot_penalty: float = 1.0,
    undershoot_penalty: float = 4.0,
    sensitivity_gamma: float = 2.5,
    target_confounding: float = 0.8,
    confounding_levels: Sequence[float] = (0.0, 0.4, 0.8, 1.2),
    incentive_demand_corr: float = 1.0,
    action_noise: float = 0.6,
    n_steps: int = 30,
    noise_scale: float = 0.05,
    n_markets: int = 200,
    n_periods: int = 400,
    seed: int = 0,
) -> DynamicConfoundingCurve:
    """The DYNAMIC §37: does the confounding-robust controller beat CE in CLOSED LOOP over time?

    Same honest story as :func:`confounding_robust_control_benchmark`, now closed-loop over
    ``n_steps`` on the confounded plant ``x' = a*x + b_true*u + noise``: a demand confounder biases
    the offline ``b_hat``, the CE controller under-actuates and undershoots ``x_target`` (expensive
    when churn = ``undershoot_penalty`` dominates budget waste), the §35 controller uses the §32
    radius (calibrated ``D = (Gamma-1)/(Gamma+1)*b_hat``, a benchmark assumption) to hedge. Reports
    mean realised closed-loop cost across a confounding sweep: robust bounds the worst case and wins
    beyond a problem-dependent threshold, paying a bounded premium near zero confounding.
    """
    from chc.uncertainty import (
        confounding_robust_inflation,  # §32 half-width; local to keep JAX out
    )

    rng = np.random.default_rng(seed)
    ce_costs: list[float] = []
    robust_costs: list[float] = []
    for conf in confounding_levels:
        ce_acc: list[float] = []
        rob_acc: list[float] = []
        for market in range(n_markets):
            b_hat = _confounded_effect_estimate(
                b_true, conf, incentive_demand_corr, action_noise, n_periods, rng
            )
            halfwidth = confounding_robust_inflation(b_hat, 0.0, sensitivity_gamma)
            # same plant-noise seed for CE and robust so the comparison is paired
            _, _, ce_cost = confounding_robust_tracking_loop(
                a,
                b_true,
                b_hat,
                0.0,
                x_target,
                overshoot_penalty,
                undershoot_penalty,
                n_steps=n_steps,
                noise_scale=noise_scale,
                seed=market,
            )
            _, _, rob_cost = confounding_robust_tracking_loop(
                a,
                b_true,
                b_hat,
                halfwidth,
                x_target,
                overshoot_penalty,
                undershoot_penalty,
                n_steps=n_steps,
                noise_scale=noise_scale,
                seed=market,
            )
            ce_acc.append(ce_cost)
            rob_acc.append(rob_cost)
        ce_costs.append(float(np.mean(ce_acc)))
        robust_costs.append(float(np.mean(rob_acc)))

    levels = list(confounding_levels)
    ce_worst = max(ce_costs)
    robust_worst = max(robust_costs)
    ti = levels.index(target_confounding)
    savings = 100.0 * (1.0 - robust_costs[ti] / ce_costs[ti]) if ce_costs[ti] > 0 else 0.0
    zi = levels.index(0.0)
    premium = 100.0 * (robust_costs[zi] - ce_costs[zi]) / ce_worst if ce_worst > 0 else 0.0
    return DynamicConfoundingCurve(
        confounding_levels=np.array(levels),
        ce_costs=np.array(ce_costs),
        robust_costs=np.array(robust_costs),
        ce_worst_case=ce_worst,
        robust_worst_case=robust_worst,
        savings_at_target_pct=savings,
        unconfounded_premium_pct=premium,
        ok=(robust_worst < ce_worst and savings > 0.0),
    )
