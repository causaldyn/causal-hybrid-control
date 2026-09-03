"""Offline optimal control over a box: projected gradient, and a bound-constrained quasi-Newton.

Both minimise the same Bolza objective from the same discrete adjoint
(:func:`chc.adjoint.control_gradient_adjoint`); they differ only in how they use it.
:func:`projected_gradient_control` takes backtracked steepest-descent steps, which is monotone and
dependency-free but first-order, so an ill-conditioned instance needs *many* of them -- thousands,
not hundreds. Its ``steps`` is therefore a cap rather than a bill: the descent runs inside one
compiled program that stops the moment the line search fails, so an unused step costs nothing and
the default is loose enough for the stopping rule to decide. :func:`lbfgs_box_control` hands the
same gradient to SciPy's L-BFGS-B, which curves the step with a limited-memory secant
approximation and reaches stationarity in tens of iterations rather than thousands -- but crosses
the Python boundary on each one, so it cannot be compiled and is the reference rather than the
workhorse. :func:`nlp_solver_certificate` measures the gap rather than asserting it in prose.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
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

_MAX_BACKTRACK = 40

Bound = float | Array
"""One side of the action box: a scalar shared by every lever, or a per-lever array.

A real actuator set is rarely a cube -- a budget and a discount move in different units and over
different ranges -- so a single scalar pair forces the caller to widen every lever to the loosest
one, which is a larger feasible set than the plant has.
"""


def broadcast_box(bound: Bound, shape: tuple[int, ...], name: str, dtype: Any) -> Array:
    """Expand a scalar or per-lever bound to the full ``(horizon, m)`` action shape.

    Accepts a scalar, a ``(m,)`` per-lever vector, or an already-full ``(horizon, m)`` schedule.
    A 1-D array is read as **per-lever**, never as per-step: for ``m == 1`` a ``(horizon,)`` array
    would broadcast along the lever axis and silently constrain the wrong axis, so time-varying
    bounds have to be spelled out in two dimensions.

    ``dtype`` is the *actions'* dtype, and it is required rather than defaulted: the box is
    validated in float64 but returned in the caller's precision, because a float64 box clipped
    against float32 actions promotes the answer and silently changes what the caller asked for.

    Raises:
        ValueError: on a shape that is neither, or on an inverted box. ``jnp.clip`` with
            ``lo > hi`` returns ``hi`` everywhere without complaint, which is a wrong answer
            rather than a failure, so the ordering is checked here where it can still be reported.
    """
    values = np.asarray(bound, dtype=np.float64)
    if values.ndim == 1 and values.shape != shape[1:]:
        raise ValueError(
            f"{name} has shape {values.shape}; a 1-D bound is per-lever and must be "
            f"{shape[1:]}. For a bound that varies over time pass the full {shape}."
        )
    if values.ndim > 2 or (values.ndim == 2 and values.shape != shape):
        raise ValueError(
            f"{name} has shape {values.shape}; expected a scalar, {shape[1:]} or {shape}"
        )
    return jnp.asarray(np.broadcast_to(values, shape), dtype=dtype)


def check_box(lo: Array, hi: Array) -> None:
    """Reject an inverted or empty box before ``jnp.clip`` turns it into a silent answer."""
    bad = np.asarray(lo) > np.asarray(hi)
    if bool(bad.any()):
        first = tuple(int(i) for i in np.argwhere(bad)[0])
        raise ValueError(
            f"empty action box at index {first}: u_lo {float(np.asarray(lo)[first])} > "
            f"u_hi {float(np.asarray(hi)[first])}"
        )


def project_box(us: Array, lo: Bound, hi: Bound) -> Array:
    """Euclidean projection onto the box ``[lo, hi]`` (elementwise clip).

    ``lo`` and ``hi`` may be scalars or per-lever arrays; both broadcast against ``us``.
    """
    return jnp.clip(us, lo, hi)


def _backtrack(
    us: Array,
    current: Array,
    grad: Array,
    u_lo: Array,
    u_hi: Array,
    lr0: float,
    tol: float,
    value_of: Callable[[Array], Array],
) -> tuple[Array, Array, Array]:
    """Halve the step until it strictly decreases ``value_of``; report whether one ever did."""

    def searching(state: tuple[Array, Array, Array, Array, Array]) -> Array:
        trial, _, _, _, accepted = state
        return jnp.logical_and(trial < _MAX_BACKTRACK, jnp.logical_not(accepted))

    def halve(
        state: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        trial, lr, _, _, _ = state
        # `.astype(us.dtype)` is load-bearing, not defensive. The actions carry the working
        # precision, but the gradient does not: `control_gradient_adjoint` differentiates a cost
        # whose Q/R/Qf and x_target are whatever `jnp.array` produced, which under
        # `jax_enable_x64` is float64 even when the actions are float32. The subtraction then
        # promotes, the candidate re-enters the carry one dtype wider than it left, and
        # `lax.while_loop` rejects the body outright. Casting here -- at the point that decides
        # what the carry holds -- keeps the whole descent in the caller's precision.
        candidate = jnp.clip(us - lr * grad, u_lo, u_hi).astype(us.dtype)
        value = value_of(candidate)
        return trial + 1, lr * 0.5, candidate, value, value < current - tol

    _, _, candidate, value, accepted = jax.lax.while_loop(
        searching,
        halve,
        (jnp.asarray(0), jnp.asarray(lr0, dtype=us.dtype), us, current, jnp.asarray(False)),
    )
    return jnp.where(accepted, candidate, us), jnp.where(accepted, value, current), accepted


@eqx.filter_jit
def _projected_gradient_loop(
    dyn: Dynamics,
    x0: Array,
    us0: Array,
    dt: float,
    cost: QuadraticCost,
    u_lo: Array,
    u_hi: Array,
    steps: int,
    lr0: float,
    tol: float,
) -> tuple[Array, Array, Array]:
    """The whole descent as one XLA program, outer ``while_loop`` and inner ``while_loop``.

    Module level rather than a closure inside the caller, because ``filter_jit`` caches on the
    wrapped function object: a wrapper rebuilt per call is recompiled per call and never amortises.

    The outer loop is a ``while_loop``, not a ``scan``, so ``steps`` is a *cap* rather than a bill:
    the descent stops the moment the line search fails, exactly where the Python ``break`` did, and
    an unused step costs nothing at all. That is what lets the default budget be loose enough for
    the stopping rule -- not the caller's guess -- to decide when the solve is finished. The cost
    history is written into a preallocated buffer, so a real early exit keeps the ``1 + accepted
    steps`` return shape.
    """
    us = project_box(us0, u_lo, u_hi)
    initial = total_cost(dyn, x0, us, dt, cost)
    values = jnp.zeros((steps + 1,), dtype=initial.dtype).at[0].set(initial)

    def descending(carry: tuple[Array, Array, Array, Array, Array]) -> Array:
        taken, _, _, _, alive = carry
        return jnp.logical_and(taken < steps, alive)

    def descend(
        carry: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        taken, us, current, values, _ = carry
        grad = control_gradient_adjoint(dyn, x0, us, dt, cost)
        us, current, accepted = _backtrack(
            us,
            current,
            grad,
            u_lo,
            u_hi,
            lr0,
            tol,
            lambda candidate: total_cost(dyn, x0, candidate, dt, cost),
        )
        # On rejection ``taken`` does not advance and the write lands back on its own slot, so the
        # buffer holds exactly the accepted prefix whichever way the step went.
        taken = jnp.where(accepted, taken + 1, taken)
        return taken, us, current, values.at[taken].set(current), accepted

    taken, optimised, _, values, _ = jax.lax.while_loop(
        descending, descend, (jnp.asarray(0), us, initial, values, jnp.asarray(True))
    )
    return optimised, values, taken


def projected_gradient_control(
    dyn: Dynamics,
    x0: Array,
    us0: Array,
    dt: float,
    cost: QuadraticCost,
    u_lo: Bound,
    u_hi: Bound,
    steps: int = 10_000,
    lr0: float = 0.2,
    tol: float = 1e-9,
) -> tuple[Array, Array]:
    """Minimise ``J`` over the control sequence subject to box constraints.

    Uses the discrete adjoint (:func:`chc.adjoint.control_gradient_adjoint`) for the gradient and
    backtracking line search, so every accepted step strictly decreases the cost. Returns the
    optimised controls and the cost history (length ``1 + accepted steps``).

    The descent runs inside a single compiled program (:func:`_projected_gradient_scan`); only the
    trim to the accepted prefix happens on the host, so the return shape stays what a Python loop
    would have produced.

    ``u_lo`` and ``u_hi`` are a scalar shared by every lever, a per-lever ``(m,)`` array, or a full
    ``(horizon, m)`` schedule. They are normalised to the action shape here and enter the compiled
    program as *arrays*, so a caller sweeping boxes compiles once rather than once per box value.
    ``dt``, ``steps`` and the line-search scalars stay static to the compilation, the same
    convention :func:`chc.cost.total_cost` already uses for ``dt``.
    """
    lo = broadcast_box(u_lo, us0.shape, "u_lo", us0.dtype)
    hi = broadcast_box(u_hi, us0.shape, "u_hi", us0.dtype)
    check_box(lo, hi)
    optimised, values, taken = _projected_gradient_loop(
        dyn, x0, us0, dt, cost, lo, hi, steps, lr0, tol
    )
    return optimised, jnp.asarray(np.asarray(values)[: int(taken) + 1].tolist())


def lbfgs_box_control(
    dyn: Dynamics,
    x0: Array,
    us0: Array,
    dt: float,
    cost: QuadraticCost,
    u_lo: Bound,
    u_hi: Bound,
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
    lo = broadcast_box(u_lo, shape, "u_lo", us0.dtype)
    hi = broadcast_box(u_hi, shape, "u_hi", us0.dtype)
    check_box(lo, hi)
    history = [float(total_cost(dyn, x0, project_box(us0, lo, hi), dt, cost))]
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
        np.asarray(project_box(us0, lo, hi), dtype=np.float64).ravel(),
        jac=True,
        method="L-BFGS-B",
        # Per element, not per problem: L-BFGS-B keeps one bound pair per coordinate, which is
        # what makes a per-lever box expressible here at all.
        bounds=list(
            zip(
                np.asarray(lo, dtype=np.float64).ravel(),
                np.asarray(hi, dtype=np.float64).ravel(),
                strict=True,
            )
        ),
        callback=record,
        options={"maxiter": steps},
    )
    optimised = jnp.asarray(result.x, dtype=us0.dtype).reshape(shape)
    return project_box(optimised, lo, hi), jnp.asarray(history)


def box_stationarity(
    dyn: Dynamics, x0: Array, us: Array, dt: float, cost: QuadraticCost, u_lo: Bound, u_hi: Bound
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
    least_stationarity_ratio: float  # min over instances of PG stationarity / L-BFGS-B's
    box_respected: bool
    ok: bool


def nlp_solver_certificate(
    horizon: int = 40, dt: float = 0.1, pg_steps: int = 150, seed: int = 3
) -> NLPSolverCertificate:
    """Sweep the control weight and confirm both halves of the claim about the first-order solver.

    The instances differ only in ``R``, which sets the conditioning of the reduced Hessian: a large
    control weight makes the objective well conditioned and steepest descent adequate, a small one
    makes it ill conditioned and a short step budget nowhere near enough. Asserting *both* is the
    point -- the certificate fails if the well-conditioned instance develops a gap (the comparison
    would be measuring something other than conditioning) and equally if the ill-conditioned one
    stops showing one (the reframing would no longer be justified).

    ``pg_steps`` is deliberately far below the shipped default: the effect being exhibited is what
    a *short* first-order budget costs on an ill-conditioned instance, and at the shipped cap the
    projected gradient runs to its own stopping rule and the gap closes to a few hundredths of a
    percent. This measures the conditioning, not the library's behaviour.
    """
    x0 = jnp.array([1.0, 0.0])
    u_lo, u_hi = -5.0, 5.0
    known = DampedOscillator(omega=1.0, zeta=0.1)
    instances = (
        ("ill-conditioned, linear", 0.001, ZeroResidual(out_dim=2)),
        (
            "intermediate, learned residual",
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
    ratios = [
        comparison.projected_gradient_stationarity / comparison.lbfgs_stationarity
        for comparison in comparisons
    ]
    return NLPSolverCertificate(
        comparisons=tuple(comparisons),
        worst_relative_gap=max(gaps),
        best_relative_gap=min(gaps),
        worst_lbfgs_stationarity=max(c.lbfgs_stationarity for c in comparisons),
        least_stationarity_ratio=min(ratios),
        box_respected=box_ok,
        ok=(
            box_ok
            # Comparative, not absolute: the floor on a stationarity residual is set by the working
            # dtype, so "< 1e-3" would pass at float64 and fail at float32 for no reason but the
            # arithmetic. The claim is that one solver arrives and the other stops short.
            and min(ratios) > 10.0
            and min(gaps) > -1e-9  # and the quasi-Newton arm is never worse
            and max(gaps) > 0.05  # ill-conditioned: the first-order budget is not enough
            and min(gaps) < 0.005  # well-conditioned: it is, so the gap is about conditioning
        ),
    )
