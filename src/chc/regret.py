"""Certainty-equivalence suboptimality for LQ control -- CHC's regret guarantee (LQ special case).

For a linear-quadratic problem the certainty-equivalent controller (solve the LQR for an *estimated*
model, then apply that gain to the true plant) has a suboptimality gap that is **quadratic** in the
model error: ``J(K_hat) - J* = O(||[dA, dB]||^2)`` in the small-error regime (Dean-Mania-Tu-Recht-
Matni, 2018/2020). This is the analysable special case of CHC's pessimism story -- small model error
costs almost nothing, but the penalty grows with error, which is exactly what the calibrated
uncertainty penalty (:mod:`chc.uncertainty`) is there to price in offline.

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
    skipped. Theory (Dean et al.) predicts an exponent of 2.
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
    """Optimal optimality-condition pessimism equals the effect-estimate variance; beats greedy."""

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
class PartialIdControlCurve:
    """Partial-ID control: worst-case regret ~ Delta^2; action sign robust iff Delta < |b|."""

    half_widths: Vector  # partial-ID / confounding-budget interval half-widths Delta
    ce_worst_regret: Vector  # worst-case regret of the certainty-equivalent action
    robust_worst_regret: Vector  # worst-case regret of the minimax-robust action (<= the CE one)
    control_evalue: float  # = |b_hat|: the half-width where the action direction is unidentified
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
    """Control under PARTIAL IDENTIFICATION and the control E-value (derived in
    ``validation/partial_id_control.mac``, proved in ``proofs/partial_id_control.v``; grounded in
    Bareinboim's partial-ID and the sensitivity literature). When the effect is only interval-
    identified, ``b in [b_hat - Delta, b_hat + Delta]``: (1) the certainty-equivalent action's
    worst-case regret grows like ``Delta^2``, and the minimax action reduces it; (2) the optimal
    action has the sign of ``b``, so its **direction** is identified iff the interval excludes 0 --
    iff ``Delta < |b_hat|``. The critical width ``Delta* = |b_hat|`` is the **control E-value**.
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
        biases = [_aipw_effect(np.random.default_rng(1000 * s + 1), n, theta, dmu, de, kind) - theta
                  for s in range(n_seeds)]
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
    """Online causal control has O(log T) cumulative regret; confounded control has Theta(T)."""

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
    ratio ``cum(T)/cum(T/2)`` separates them: ``-> 1`` (log) vs ``-> 2`` (linear).
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
    the optimum -- an unsafe certificate exactly where a valid one is needed. Verifies
    ``proofs/nonlinear_regret.v`` and ``validation/nonlinear_regret.mac`` numerically.
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
