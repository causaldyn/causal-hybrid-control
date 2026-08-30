"""Offline optimal control over a box: projected gradient, and a bound-constrained quasi-Newton.

Both minimise the same Bolza objective from the same discrete adjoint
(:func:`chc.adjoint.control_gradient_adjoint`); they differ only in how they use it.
:func:`projected_gradient_control` takes backtracked steepest-descent steps, which is monotone
and dependency-free but first-order: on an ill-conditioned instance it exhausts its iteration
budget far from stationarity. :func:`lbfgs_box_control` hands the same gradient to SciPy's
L-BFGS-B, which curves the step with a limited-memory secant approximation and handles the box
natively. :func:`nlp_solver_certificate` measures the gap rather than asserting it in prose.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray
from scipy.optimize import minimize

from chc.adjoint import control_gradient_adjoint
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import DampedOscillator, Dynamics, HybridDynamics
from chc.residual import MLPResidual, ZeroResidual


def project_box(us: Array, lo: float, hi: float) -> Array:
    """Euclidean projection onto the box ``[lo, hi]`` (elementwise clip)."""
    return jnp.clip(us, lo, hi)


def projected_gradient_control(
    dyn: Dynamics,
    x0: Array,
    us0: Array,
    dt: float,
    cost: QuadraticCost,
    u_lo: float,
    u_hi: float,
    steps: int = 200,
    lr0: float = 0.2,
    tol: float = 1e-9,
) -> tuple[Array, Array]:
    """Minimise ``J`` over the control sequence subject to box constraints.

    Uses the discrete adjoint (:func:`chc.adjoint.control_gradient_adjoint`) for the gradient and
    backtracking line search, so every accepted step strictly decreases the cost. Returns the
    optimised controls and the cost history (length ``1 + accepted steps``).
    """
    us = project_box(us0, u_lo, u_hi)
    current = total_cost(dyn, x0, us, dt, cost)
    history = [float(current)]

    for _ in range(steps):
        grad = control_gradient_adjoint(dyn, x0, us, dt, cost)
        lr = lr0
        improved = False
        candidate = us
        candidate_cost = current
        for _ls in range(40):
            candidate = project_box(us - lr * grad, u_lo, u_hi)
            candidate_cost = total_cost(dyn, x0, candidate, dt, cost)
            if candidate_cost < current - tol:
                improved = True
                break
            lr *= 0.5
        if not improved:
            break
        us, current = candidate, candidate_cost
        history.append(float(current))

    return us, jnp.asarray(history)


def lbfgs_box_control(
    dyn: Dynamics,
    x0: Array,
    us0: Array,
    dt: float,
    cost: QuadraticCost,
    u_lo: float,
    u_hi: float,
    steps: int = 300,
) -> tuple[Array, Array]:
    """Minimise ``J`` over the control sequence subject to box constraints, by L-BFGS-B.

    Same objective, same discrete-adjoint gradient and the same ``(controls, cost history)`` return
    as :func:`projected_gradient_control` -- a drop-in whose step is curved by a limited-memory
    secant approximation instead of scaled by a backtracked constant. ``steps`` caps L-BFGS-B
    iterations, not gradient evaluations, and it converges well inside the default on the instances
    :func:`nlp_solver_certificate` sweeps.

    SciPy is the trust boundary: the objective and gradient cross it as float64 NumPy and the answer
    is cast back to ``us0``'s dtype, so a float32 caller is not silently promoted.
    """
    shape = us0.shape
    size = int(np.prod(shape))
    history = [float(total_cost(dyn, x0, project_box(us0, u_lo, u_hi), dt, cost))]
    seen: dict[bytes, float] = {}

    def objective(flat: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        us = jnp.asarray(flat, dtype=us0.dtype).reshape(shape)
        value = float(total_cost(dyn, x0, us, dt, cost))
        seen[flat.tobytes()] = value
        gradient = control_gradient_adjoint(dyn, x0, us, dt, cost)
        return value, np.asarray(gradient, dtype=np.float64).ravel()

    def record(xk: NDArray[np.float64]) -> None:
        history.append(seen.get(xk.tobytes(), history[-1]))

    result = minimize(
        objective,
        np.asarray(project_box(us0, u_lo, u_hi), dtype=np.float64).ravel(),
        jac=True,
        method="L-BFGS-B",
        bounds=[(u_lo, u_hi)] * size,
        callback=record,
        options={"maxiter": steps},
    )
    optimised = jnp.asarray(result.x, dtype=us0.dtype).reshape(shape)
    return project_box(optimised, u_lo, u_hi), jnp.asarray(history)


def box_stationarity(
    dyn: Dynamics, x0: Array, us: Array, dt: float, cost: QuadraticCost, u_lo: float, u_hi: float
) -> float:
    """First-order optimality residual ``||u - P_box(u - grad J)||`` -- zero exactly at a KKT point.

    The convergence measure that does not depend on knowing the optimum: it is the natural
    stationarity map of a box-constrained problem, so it separates "the solver stopped" from "the
    solver arrived" without a reference solution and without a wall clock.
    """
    gradient = control_gradient_adjoint(dyn, x0, us, dt, cost)
    return float(jnp.linalg.norm(us - project_box(us - gradient, u_lo, u_hi)))


@dataclass(frozen=True)
class SolverComparison:
    """One instance of the box-constrained OC problem solved both ways."""

    label: str
    control_weight: float
    projected_gradient_cost: float
    lbfgs_cost: float
    projected_gradient_stationarity: float
    lbfgs_stationarity: float
    relative_gap: float


@dataclass(frozen=True)
class NLPSolverCertificate:
    """Evidence for where the first-order solver suffices and where it stops short."""

    comparisons: tuple[SolverComparison, ...]
    worst_relative_gap: float
    best_relative_gap: float
    worst_lbfgs_stationarity: float
    box_respected: bool
    ok: bool


def nlp_solver_certificate(
    horizon: int = 40, dt: float = 0.1, pg_steps: int = 150, seed: int = 3
) -> NLPSolverCertificate:
    """Sweep the control weight and confirm both halves of the claim about the first-order solver.

    The instances differ only in ``R``, which sets the conditioning of the reduced Hessian: a large
    control weight makes the objective well conditioned and steepest descent adequate, a small one
    makes it ill conditioned and the fixed step budget nowhere near enough. Asserting *both* is the
    point -- the certificate fails if the well-conditioned instance develops a gap (the comparison
    would be measuring something other than conditioning) and equally if the ill-conditioned one
    stops showing one (the reframing would no longer be justified).
    """
    x0 = jnp.array([1.0, 0.0])
    u_lo, u_hi = -5.0, 5.0
    known = DampedOscillator(omega=1.0, zeta=0.1)
    instances = (
        ("ill-conditioned, linear", 0.001, ZeroResidual(out_dim=2)),
        (
            "ill-conditioned, learned residual",
            0.01,
            MLPResidual(2, 1, 2, 16, 2, key=jax.random.key(seed)),
        ),
        (
            "well-conditioned, learned residual",
            0.1,
            MLPResidual(2, 1, 2, 16, 2, key=jax.random.key(seed + 8)),
        ),
    )

    comparisons: list[SolverComparison] = []
    box_ok = True
    for label, control_weight, residual in instances:
        cost = QuadraticCost(
            Q=jnp.diag(jnp.array([1.0, 0.0])),
            R=jnp.array([[control_weight]]),
            Qf=jnp.diag(jnp.array([10.0, 1.0])),
            x_target=jnp.zeros(2),
        )
        dyn = HybridDynamics(known=known, residual=residual)
        us0 = jnp.zeros((horizon, 1))
        pg_us, pg_history = projected_gradient_control(
            dyn, x0, us0, dt, cost, u_lo, u_hi, steps=pg_steps
        )
        qn_us, qn_history = lbfgs_box_control(dyn, x0, us0, dt, cost, u_lo, u_hi)
        box_ok = box_ok and bool((jnp.abs(qn_us) <= u_hi + 1e-9).all())
        pg_cost, qn_cost = float(pg_history[-1]), float(qn_history[-1])
        comparisons.append(
            SolverComparison(
                label=label,
                control_weight=control_weight,
                projected_gradient_cost=pg_cost,
                lbfgs_cost=qn_cost,
                projected_gradient_stationarity=box_stationarity(
                    dyn, x0, pg_us, dt, cost, u_lo, u_hi
                ),
                lbfgs_stationarity=box_stationarity(dyn, x0, qn_us, dt, cost, u_lo, u_hi),
                relative_gap=(pg_cost - qn_cost) / abs(qn_cost),
            )
        )

    gaps = [comparison.relative_gap for comparison in comparisons]
    worst_stationarity = max(comparison.lbfgs_stationarity for comparison in comparisons)
    return NLPSolverCertificate(
        comparisons=tuple(comparisons),
        worst_relative_gap=max(gaps),
        best_relative_gap=min(gaps),
        worst_lbfgs_stationarity=worst_stationarity,
        box_respected=box_ok,
        ok=(
            box_ok
            and worst_stationarity < 1e-3  # the quasi-Newton arm actually converged
            and min(gaps) > -1e-9  # and is never worse than the first-order one
            and max(gaps) > 0.05  # ill-conditioned: the first-order budget is not enough
            and min(gaps) < 0.005  # well-conditioned: it is, so the gap is about conditioning
        ),
    )
