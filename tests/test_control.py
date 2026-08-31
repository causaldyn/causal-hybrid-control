"""Optimal-control gate: projected gradient reduces cost and drives the state toward target."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc import (
    DampedOscillator,
    HybridDynamics,
    QuadraticCost,
    ZeroResidual,
    lbfgs_box_control,
    nlp_solver_certificate,
    projected_gradient_control,
    rollout,
)
from chc.adjoint import control_gradient_adjoint
from chc.cost import total_cost
from chc.residual import (
    ContractiveResidual,
    ControlAffineResidual,
    GraphResidual,
    KANResidual,
    LipschitzResidual,
    MLPResidual,
    PortHamiltonianResidual,
    SpectralResidual,
)

DT = 0.1


def test_projected_gradient_reduces_cost_and_reaches_target() -> None:
    dyn = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.01]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    x0 = jnp.array([1.0, 0.0])
    us0 = jnp.zeros((50, 1))

    us, history = projected_gradient_control(dyn, x0, us0, DT, cost, u_lo=-5.0, u_hi=5.0, steps=150)

    assert history[-1] < 0.7 * history[0]  # meaningful cost reduction
    assert bool((jnp.abs(us) <= 5.0 + 1e-6).all())  # respects box constraints
    xs = rollout(dyn, x0, us, DT)
    assert abs(float(xs[-1, 0])) < abs(float(x0[0]))  # ends closer to target position


def test_lbfgs_matches_the_projected_gradient_objective_and_beats_it() -> None:
    certificate = nlp_solver_certificate()
    assert certificate.ok
    assert certificate.box_respected
    # Comparative rather than absolute: an absolute residual threshold is a claim about the
    # working dtype, not about the solvers, and would fail under float32 alone.
    assert certificate.least_stationarity_ratio > 10.0
    # Both directions: the first-order budget is short where the problem is ill conditioned and
    # adequate where it is not, so the gap is a statement about conditioning, not about the plant.
    assert certificate.worst_relative_gap > 0.05
    assert certificate.best_relative_gap < 0.005


def test_lbfgs_box_control_preserves_the_caller_dtype() -> None:
    # SciPy is a float64 boundary; a float32 caller must not be silently promoted on the way back.
    dyn = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.01]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    us0 = jnp.zeros((20, 1), dtype=jnp.float32)
    us, history = lbfgs_box_control(dyn, jnp.array([1.0, 0.0]), us0, DT, cost, -5.0, 5.0)
    assert us.dtype == us0.dtype
    assert float(history[-1]) < float(history[0])
    assert bool((jnp.abs(us) <= 5.0 + 1e-6).all())


# --- the compiled descent must reproduce the loop it replaced, on every residual backend --------

_ULP_BUDGET = 500.0


def _naive_projected_gradient(
    dyn: object,
    x0: jnp.ndarray,
    us0: jnp.ndarray,
    dt: float,
    cost: QuadraticCost,
    u_lo: float,
    u_hi: float,
    steps: int,
    lr0: float = 0.2,
    tol: float = 1e-9,
) -> tuple[jnp.ndarray, list[float]]:
    """The plain Python recursion, kept as the oracle the compiled solver is checked against.

    It is deliberately *not* imported from ``chc``: an oracle that shares the implementation under
    test cannot detect the implementation changing.
    """
    us = jnp.clip(us0, u_lo, u_hi)
    current = total_cost(dyn, x0, us, dt, cost)
    history = [float(current)]
    for _ in range(steps):
        grad = control_gradient_adjoint(dyn, x0, us, dt, cost)
        lr, improved, candidate, candidate_cost = lr0, False, us, current
        for _ls in range(40):
            candidate = jnp.clip(us - lr * grad, u_lo, u_hi)
            candidate_cost = total_cost(dyn, x0, candidate, dt, cost)
            if candidate_cost < current - tol:
                improved = True
                break
            lr *= 0.5
        if not improved:
            break
        us, current = candidate, candidate_cost
        history.append(float(current))
    return us, history


def _ulp_gap(a: jnp.ndarray, b: jnp.ndarray) -> float:
    left, right = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    spacing = np.spacing(np.maximum(np.abs(left), np.abs(right)))
    return float(np.max(np.abs(left - right) / np.maximum(spacing, np.finfo(np.float64).tiny)))


def _backend(name: str, key: jnp.ndarray) -> tuple[object, int]:
    """``(residual, control_dim)`` for each backend; the state is 2-dimensional throughout."""
    if name == "zero":
        return ZeroResidual(out_dim=2), 1
    if name == "mlp":
        return MLPResidual(2, 1, 2, 16, 2, key=key), 1
    if name == "control_affine":
        k_drift, k_channel = jax.random.split(key)
        return (
            ControlAffineResidual(
                drift=0.05 * jax.random.normal(k_drift, (2, 3)),
                channel=0.05 * jax.random.normal(k_channel, (2, 1, 3)),
            ),
            1,
        )
    if name == "kan":
        return KANResidual(2, 1, 2, key=key), 1
    if name == "spectral":  # couples a periodic field to a co-located control, so control_dim = 2
        return SpectralResidual(2, 2, key=key), 2
    if name == "graph":
        return GraphResidual(jnp.array([[0.0, 1.0], [1.0, 0.0]]), 1, 1, key=key), 1
    if name == "port_hamiltonian":
        return PortHamiltonianResidual(2, 1, key=key), 1
    if name == "lipschitz":
        return LipschitzResidual(2, 1, 2, key=key), 1
    if name == "contractive":
        return ContractiveResidual(2, 1, key=key), 1
    raise AssertionError(name)


_BACKENDS = (
    "zero",
    "mlp",
    "control_affine",
    "kan",
    "spectral",
    "graph",
    "port_hamiltonian",
    "lipschitz",
    "contractive",
)


@pytest.mark.parametrize("name", _BACKENDS)
def test_compiled_descent_matches_the_python_recursion(name: str) -> None:
    residual, control_dim = _backend(name, jax.random.key(_BACKENDS.index(name)))
    dyn = HybridDynamics(known=DampedOscillator(omega=2.0, zeta=0.1), residual=residual)
    cost = QuadraticCost(
        Q=jnp.eye(2),
        R=0.05 * jnp.eye(control_dim),
        Qf=5.0 * jnp.eye(2),
        x_target=jnp.array([0.5, 0.0]),
    )
    x0, us0 = jnp.array([1.0, 0.0]), jnp.zeros((20, control_dim))

    us_ref, history_ref = _naive_projected_gradient(dyn, x0, us0, DT, cost, -2.0, 2.0, steps=60)
    us, history = projected_gradient_control(dyn, x0, us0, DT, cost, -2.0, 2.0, steps=60)

    assert len(history) == len(history_ref)  # the scan stops where the break would have
    assert _ulp_gap(us, us_ref) < _ULP_BUDGET
    assert _ulp_gap(history, jnp.asarray(history_ref)) < _ULP_BUDGET


def test_compiled_descent_stops_where_the_line_search_fails() -> None:
    # A heavy control penalty makes the descent converge in a handful of steps, so the history is
    # much shorter than the budget -- the case a fixed-length scan would get wrong if it padded.
    dyn = HybridDynamics(
        known=DampedOscillator(omega=2.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.eye(2), R=5.0 * jnp.eye(1), Qf=5.0 * jnp.eye(2), x_target=jnp.array([0.5, 0.0])
    )
    x0, us0 = jnp.array([1.0, 0.0]), jnp.zeros((20, 1))

    us_ref, history_ref = _naive_projected_gradient(dyn, x0, us0, DT, cost, -2.0, 2.0, steps=400)
    us, history = projected_gradient_control(dyn, x0, us0, DT, cost, -2.0, 2.0, steps=400)

    assert len(history_ref) < 400  # the oracle really does break early
    assert len(history) == len(history_ref)
    assert _ulp_gap(us, us_ref) < _ULP_BUDGET


def test_the_compiled_solver_amortises_across_calls() -> None:
    # The regression this guards: building the jitted loop inside the caller gives every call a
    # fresh compilation cache, so the second solve costs the same as the first.
    # The odd horizon and step count keep this instance off every other test's compilation key, so
    # the first solve here is genuinely cold rather than warmed by an earlier test in the module.
    dyn = HybridDynamics(
        known=DampedOscillator(omega=1.7, zeta=0.1),
        residual=MLPResidual(2, 1, 2, 16, 2, key=jax.random.key(11)),
    )
    cost = QuadraticCost(
        Q=jnp.eye(2), R=0.05 * jnp.eye(1), Qf=5.0 * jnp.eye(2), x_target=jnp.array([0.5, 0.0])
    )
    x0, us0 = jnp.array([1.0, 0.0]), jnp.zeros((23, 1))

    def solve_seconds() -> float:
        start = time.perf_counter()
        jax.block_until_ready(projected_gradient_control(dyn, x0, us0, DT, cost, -2.0, 2.0, 37))
        return time.perf_counter() - start

    cold = solve_seconds()
    warm = min(solve_seconds(), solve_seconds())
    assert warm < 0.5 * cold
