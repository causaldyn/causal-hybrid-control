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
