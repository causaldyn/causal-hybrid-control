"""Pessimism gate: support penalty keeps control near the data; greedy control extrapolates."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from chc import (
    DampedOscillator,
    HybridDynamics,
    QuadraticCost,
    ZeroResidual,
    projected_gradient_control,
    rollout,
)
from chc.cost import total_cost
from chc.support import SupportModel, pessimistic_control

DT = 0.1


def test_pessimism_keeps_control_in_support() -> None:
    k_x, k_u = jax.random.split(jax.random.key(0))
    xs_data = jax.random.normal(k_x, (2000, 2))  # x ~ N(0, I)
    us_data = 0.3 * jax.random.normal(k_u, (2000, 1))  # u ~ N(0, 0.3^2): narrow support
    support = SupportModel.fit(xs_data, us_data)

    model = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    # cheap control + a far target ⇒ the greedy optimum slams the actuator far beyond the u-support
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.001]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.array([-3.0, 0.0]),
    )
    x0 = jnp.zeros(2)
    us0 = jnp.zeros((20, 1))

    us_greedy, _ = projected_gradient_control(model, x0, us0, DT, cost, -5.0, 5.0, steps=200)
    us_pess, _ = pessimistic_control(
        model, x0, us0, DT, cost, support, lam_supp=20.0, u_lo=-5.0, u_hi=5.0, steps=200
    )

    greedy_max = float(jnp.max(jnp.abs(us_greedy)))
    pess_max = float(jnp.max(jnp.abs(us_pess)))
    assert greedy_max > 2.0  # greedy extrapolates far past the u-support (~0.3)
    assert pess_max < 0.6 * greedy_max  # pessimism shrinks control toward the data

    xs_greedy = rollout(model, x0, us_greedy, DT)
    xs_pess = rollout(model, x0, us_pess, DT)
    assert float(jnp.max(jnp.abs(xs_pess[:, 0]))) < float(jnp.max(jnp.abs(xs_greedy[:, 0])))


# --- the penalised descent must reproduce the loop it replaced ----------------------------------

_ULP_BUDGET = 500.0


def _naive_pessimistic(
    model: object,
    x0: jnp.ndarray,
    us0: jnp.ndarray,
    cost: QuadraticCost,
    support: SupportModel,
    lam_supp: float,
    u_lo: float,
    u_hi: float,
    steps: int,
    lr0: float = 0.2,
    tol: float = 1e-9,
) -> tuple[jnp.ndarray, list[float]]:
    """The plain Python recursion for the penalised objective, kept as the oracle."""

    def task(us: jnp.ndarray) -> jnp.ndarray:
        return total_cost(model, x0, us, DT, cost)

    def augmented(us: jnp.ndarray) -> jnp.ndarray:
        xs = rollout(model, x0, us, DT)
        return task(us) + lam_supp * support.penalty_trajectory(xs[:-1], us)

    grad_aug = jax.grad(augmented)
    us = jnp.clip(us0, u_lo, u_hi)
    current = augmented(us)
    history = [float(task(us))]
    for _ in range(steps):
        grad = grad_aug(us)
        lr, improved, candidate, candidate_cost = lr0, False, us, current
        for _ls in range(40):
            candidate = jnp.clip(us - lr * grad, u_lo, u_hi)
            candidate_cost = augmented(candidate)
            if candidate_cost < current - tol:
                improved = True
                break
            lr *= 0.5
        if not improved:
            break
        us, current = candidate, candidate_cost
        history.append(float(task(us)))
    return us, history


def _ulp_gap(a: jnp.ndarray, b: jnp.ndarray) -> float:
    left, right = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    spacing = np.spacing(np.maximum(np.abs(left), np.abs(right)))
    return float(np.max(np.abs(left - right) / np.maximum(spacing, np.finfo(np.float64).tiny)))


def test_compiled_pessimistic_descent_matches_the_python_recursion() -> None:
    k_x, k_u = jax.random.split(jax.random.key(4))
    support = SupportModel.fit(
        jax.random.normal(k_x, (500, 2)), 0.3 * jax.random.normal(k_u, (500, 1))
    )
    model = HybridDynamics(
        known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.001]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.array([-3.0, 0.0]),
    )
    x0, us0 = jnp.zeros(2), jnp.zeros((15, 1))

    us_ref, history_ref = _naive_pessimistic(
        model, x0, us0, cost, support, 20.0, -5.0, 5.0, steps=60
    )
    us, history = pessimistic_control(
        model, x0, us0, DT, cost, support, lam_supp=20.0, u_lo=-5.0, u_hi=5.0, steps=60
    )

    assert len(history) == len(history_ref)
    assert _ulp_gap(us, us_ref) < _ULP_BUDGET
    assert _ulp_gap(history, jnp.asarray(history_ref)) < _ULP_BUDGET


def test_the_compiled_pessimistic_solver_amortises_across_calls() -> None:
    # The regression this guards: the jitted augmented objective and its gradient used to be built
    # inside pessimistic_control, so every solve recompiled them. Odd sizes keep the compilation
    # key off every other test's.
    k_x, k_u = jax.random.split(jax.random.key(5))
    support = SupportModel.fit(
        jax.random.normal(k_x, (300, 2)), 0.3 * jax.random.normal(k_u, (300, 1))
    )
    model = HybridDynamics(
        known=DampedOscillator(omega=1.3, zeta=0.1), residual=ZeroResidual(out_dim=2)
    )
    cost = QuadraticCost(
        Q=jnp.eye(2), R=jnp.array([[0.01]]), Qf=jnp.eye(2), x_target=jnp.array([-1.0, 0.0])
    )
    x0, us0 = jnp.zeros(2), jnp.zeros((17, 1))

    def solve_seconds() -> float:
        start = time.perf_counter()
        jax.block_until_ready(
            pessimistic_control(
                model, x0, us0, DT, cost, support, lam_supp=3.0, u_lo=-5.0, u_hi=5.0, steps=41
            )
        )
        return time.perf_counter() - start

    cold = solve_seconds()
    warm = min(solve_seconds(), solve_seconds())
    assert warm < 0.5 * cold
