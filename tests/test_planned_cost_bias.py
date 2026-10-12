"""How far a plan's own cost reads from what it pays the plant, in expectation over the error of
the channel it was planned on: ``tr(W S) - tr(J_ee S) / 2 - J_e . E[e]``.

The scalar cases are Maxima's closed forms (``validation/planned_cost_bias.mac``). The multi-lever
case is a Monte Carlo of refits: one design, the noise drawn afresh, the channel re-estimated by
least squares and the plan re-solved on each estimate.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chc import CausalPlan, QuadraticCost, causal_plan
from chc.cost import total_cost
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.dynamics_id import fit_causal_residual
from chc.integrate import rk4_step
from chc.misspecification import planned_cost_bias
from chc.plan import PlannedCostBias, _perturbed_task_cost
from chc.residual import ControlAffineResidual

DT = 0.1
ONE = jnp.array([1.0])
ROOTS = ((5.0 - math.sqrt(17.0)) / 2.0, (5.0 + math.sqrt(17.0)) / 2.0)


def _rk4(a: float, b: float, dt: float) -> tuple[float, float]:
    """``x1 = phi x0 + gam u``: the RK4 step of ``xdot = a x + b u``."""
    h = a * dt
    return 1 + h + h**2 / 2 + h**3 / 6 + h**4 / 24, b * dt * (1 + h / 2 + h**2 / 6 + h**3 / 24)


def _at(plan: CausalPlan, actions: np.ndarray) -> CausalPlan:
    """The same problem, read at ``actions``: the closed form's optimum, not the solver's."""
    return replace(plan, actions=jnp.asarray(actions, dtype=plan.actions.dtype))


def _one_step(a: float, b: float, x0: float, target: float, r: float, qf: float) -> CausalPlan:
    """One step of ``xdot = a x + b u`` at its exact optimum, ``u* = -gam qf y / (r + gam^2 qf)``
    with ``y = phi x0 - target``."""
    phi, gam = _rk4(a, b, DT)
    cost = QuadraticCost(
        Q=jnp.eye(1), R=jnp.array([[r]]), Qf=jnp.array([[qf]]), x_target=jnp.array([target])
    )
    model = LinearDynamics(jnp.array([[a]]), jnp.array([[b]]))
    plan = causal_plan(model, jnp.array([x0]), cost, DT, 1, -1e3, 1e3, steps=1)
    optimum = -gam * qf * (phi * x0 - target) / (r + gam**2 * qf)
    return _at(plan, np.array([[optimum]]))


def _coefficient(a: float, b: float, x0: float, target: float, r: float, qf: float) -> float:
    """The bias per unit of ``S`` (STEP 3b): ``y^2 qf^2 (2 r^2 - 5 g^2 qf r + g^4 qf^2) / (2 (r +
    g^2 qf)^3)``, ``y = phi x0 - target``, ``g = gam``."""
    phi, g = _rk4(a, b, DT)
    y = phi * x0 - target
    return y**2 * qf**2 * (2 * r**2 - 5 * g**2 * qf * r + g**4 * qf**2) / (2 * (r + g**2 * qf) ** 3)


def test_one_step_reads_maxima_s_numbers() -> None:
    """STEP 5: ``a = -0.5, b = 1.2, x0 = 1``, ``r = 0.1, qf = 2``: the coefficient
    6.170016381437646, ``W`` 9.225094387724484 and ``J_e`` -2.610076308640553."""
    plan = _one_step(-0.5, 1.2, 1.0, 0.0, 0.1, 2.0)
    s, m = 0.003, 0.02
    found = plan.planned_cost_bias([[s]], error_mean=[[m]])
    assert isinstance(found, PlannedCostBias)
    assert found.optimism == pytest.approx(9.225094387724484 * s, rel=1e-10, abs=0.0)
    assert found.optimism + found.curvature == pytest.approx(
        6.170016381437646 * s, rel=1e-10, abs=0.0
    )
    assert found.first_order == pytest.approx(2.610076308640553 * m, rel=1e-10, abs=0.0)
    assert found.bias == found.optimism + found.curvature + found.first_order
    assert found.planned_cost == pytest.approx(plan.task_cost, rel=1e-6, abs=0.0)
    assert found.corrected_cost == found.planned_cost + found.bias
    assert found.direction == "optimistic"
    assert (found.free, found.unseen) == (1, None)
    assert found.residual < 1e-12
    unbiased = plan.planned_cost_bias([[s]])
    assert unbiased.first_order == 0.0
    assert unbiased.bias == pytest.approx(6.170016381437646 * s, rel=1e-10, abs=0.0)


@settings(max_examples=25, deadline=None)
@given(
    a=st.floats(-2.0, 1.0),
    b=st.floats(0.2, 3.0),
    x0=st.floats(-2.0, 2.0),
    target=st.floats(-2.0, 2.0),
    r=st.floats(0.01, 1.0),
    qf=st.floats(0.1, 5.0),
)
def test_a_one_step_bias_is_maxima_s_closed_form(
    a: float, b: float, x0: float, target: float, r: float, qf: float
) -> None:
    found = _one_step(a, b, x0, target, r, qf).planned_cost_bias(np.eye(1))
    expected = _coefficient(a, b, x0, target, r, qf)
    scale = found.optimism + abs(found.curvature)
    assert found.bias == pytest.approx(expected, rel=1e-8, abs=1e-12 * max(scale, 1e-300))


@pytest.mark.parametrize(
    ("ratio", "direction"),
    [
        (ROOTS[0] * 0.99, "optimistic"),
        (ROOTS[0] * 1.01, "pessimistic"),
        (ROOTS[1] * 0.99, "pessimistic"),
        (ROOTS[1] * 1.01, "optimistic"),
    ],
)
def test_the_sign_flips_at_both_roots(ratio: float, direction: str) -> None:
    """Between ``q b^2 / r = (5 -+ sqrt 17)/2`` the planned cost reads high, outside them low."""
    qf = 2.0
    gam = _rk4(-0.5, 1.2, DT)[1]
    r = gam**2 * qf / ratio
    found = _one_step(-0.5, 1.2, 1.0, 0.0, r, qf).planned_cost_bias([[1.0]])
    assert found.direction == direction
    assert found.optimism > 0.0 > found.curvature


@pytest.mark.parametrize("root", ROOTS)
def test_at_a_root_the_two_terms_cancel(root: float) -> None:
    qf = 2.0
    gam = _rk4(-0.5, 1.2, DT)[1]
    found = _one_step(-0.5, 1.2, 1.0, 0.0, gam**2 * qf / root, qf).planned_cost_bias([[1.0]])
    assert abs(found.bias) < 1e-9 * found.optimism


# ---- two states, two levers, three steps: a Monte Carlo of refits ----

A = np.array([[-0.3, 0.2], [0.1, -0.5]])
B = np.array([[1.0, 0.3], [-0.2, 0.8]])
COST = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.5])),
    R=jnp.diag(jnp.array([0.2, 0.1])),
    Qf=jnp.diag(jnp.array([2.0, 1.0])),
    x_target=jnp.array([1.5, -1.0]),
)
START = jnp.array([0.2, 0.4])
HORIZON = 3


def _cost(actions: jax.Array, channel: jax.Array) -> jax.Array:
    return total_cost(
        LinearDynamics(jnp.asarray(A), channel), START, actions.reshape(HORIZON, 2), DT, COST
    )


@jax.jit
def _optimum(channel: jax.Array) -> jax.Array:
    """The plan on ``channel``: the cost is quadratic in the actions, so one Newton step from 0."""
    still = jnp.zeros(HORIZON * 2)
    hessian = jax.hessian(_cost)(still, channel)
    return -jnp.linalg.solve(hessian, jax.grad(_cost)(still, channel))


def test_a_monte_carlo_of_refits_reads_the_bias() -> None:
    """One design of 60 dithered decisions; on each of 4000 draws of the noise the one-step
    channel is refitted by least squares, the plan re-solved on it, and the gap between what it
    pays the plant and what it reads is recorded. Each gap less its first-order term, ``-J_e . e``,
    whose mean is 0, averages to the bias the plan on the plant's channel predicts, within its
    Monte Carlo error, where the optimism alone, or half of it, does not."""
    channel = jnp.asarray(B)
    planned = causal_plan(
        LinearDynamics(jnp.asarray(A), channel), START, COST, DT, HORIZON, -1e2, 1e2
    )
    plan = _at(planned, np.asarray(_optimum(channel)).reshape(HORIZON, 2))
    # the one-step channel is M B: read M off the library's own step
    lift = np.asarray(
        jax.jacfwd(
            lambda c: jax.jacfwd(
                lambda u: rk4_step(LinearDynamics(jnp.asarray(A), c), 0.0, jnp.zeros(2), u, DT)
            )(jnp.zeros(2))
        )(jnp.eye(2))
    )
    step = lift.reshape(4, 4)  # vec of the one-step channel against vec of B, row-major
    gamma = (step @ B.ravel()).reshape(2, 2)
    rng = np.random.default_rng(12)
    design = rng.standard_normal((60, 2))
    noise = 0.05
    inverse = np.linalg.inv(design.T @ design)
    covariance = np.kron(np.eye(2), noise**2 * inverse)
    predicted = plan.planned_cost_bias(covariance)

    draws = 4000
    responses = design @ gamma.T + noise * rng.standard_normal((draws, 60, 2))
    fitted = np.linalg.solve(design.T @ design, design.T @ responses).transpose(0, 2, 1)
    errors = fitted - gamma[None]  # (draws, 2, 2): the estimate less the plant's one-step channel
    channels = np.linalg.solve(step, errors.reshape(draws, 4).T).T.reshape(draws, 2, 2) + B
    actions = jax.vmap(_optimum)(jnp.asarray(channels))
    truth = jax.vmap(_cost, in_axes=(0, None))(actions, channel)
    read = jax.vmap(_cost)(actions, jnp.asarray(channels))
    gaps = np.asarray(truth - read)

    problem = plan._problem
    assert problem is not None
    slope = np.asarray(
        jax.grad(lambda e: _perturbed_task_cost(problem, plan.actions, e.reshape(2, 2)))(
            jnp.zeros(4)
        )
    )
    second = gaps + errors.reshape(draws, 4) @ slope
    mean, error = float(second.mean()), float(second.std() / math.sqrt(draws))
    assert abs(mean - predicted.bias) < 4.0 * error
    assert predicted.bias > 10.0 * error
    for wrong in (predicted.optimism, predicted.optimism / 2.0):
        assert abs(mean - wrong) > 6.0 * error
    # the raw gaps carry the first-order term's spread, and agree only loosely
    assert abs(gaps.mean() - predicted.bias) < 4.0 * float(gaps.std() / math.sqrt(draws))


# ---- refusals ----


def test_it_refuses_a_plan_whose_active_set_moves_with_the_error() -> None:
    """A bound met exactly at the unconstrained optimum binds with a zero multiplier: an error
    one way frees the action, the other way pins it, and the bias is not one expansion."""
    free = _one_step(-0.5, 1.2, 1.0, 0.0, 0.1, 2.0)
    optimum = float(free.actions[0, 0])
    problem = free._problem
    assert problem is not None
    pinned = replace(free, _problem=dataclasses.replace(problem, u_lo=jnp.asarray(optimum)))
    assert pinned.decision_weight().weakly_active == 1
    with pytest.raises(ValueError, match="zero multiplier"):
        pinned.planned_cost_bias([[1.0]])


@pytest.mark.parametrize(
    ("covariance", "match"),
    [
        (None, "no covariance"),
        ([[1.0, 0.0]], "finite \\(1, 1\\)"),
        ([[math.nan]], "finite"),
        ([[-1.0]], "positive semidefinite"),
    ],
)
def test_it_refuses_a_covariance_it_cannot_read(covariance: object, match: str) -> None:
    plan = _one_step(-0.5, 1.2, 1.0, 0.0, 0.1, 2.0)
    with pytest.raises(ValueError, match=match):
        plan.planned_cost_bias(covariance)  # type: ignore[arg-type]


def test_it_refuses_a_covariance_that_is_not_symmetric_or_not_semidefinite() -> None:
    planned = causal_plan(
        LinearDynamics(jnp.asarray(A), jnp.asarray(B)), START, COST, DT, 1, -1e2, 1e2
    )
    plan = _at(planned, np.asarray(planned.actions))
    with pytest.raises(ValueError, match="symmetric"):
        plan.planned_cost_bias(np.eye(4) + np.triu(np.ones((4, 4)), 1) * 0.1)
    with pytest.raises(ValueError, match="positive semidefinite"):
        plan.planned_cost_bias(np.eye(4) - 2.0 * np.outer([1, 1, 0, 0], [1, 1, 0, 0]) / 2.0 * 1.5)
    with pytest.raises(ValueError, match="error_mean"):
        plan.planned_cost_bias(np.eye(4), error_mean=np.zeros(4))


# ---- the fit's own covariance ----

FIT_A = np.array([[-0.5, 0.2], [0.0, -0.3]])
FIT_B = np.array([0.8, -0.4])
FIT_COST = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.1])),
    R=jnp.array([[0.05]]),
    Qf=jnp.diag(jnp.array([1.0, 0.1])),
    x_target=jnp.array([1.0, 0.0]),
)


def _known(t: float | jax.Array, x: jax.Array, u: jax.Array) -> jax.Array:
    return jnp.zeros_like(x)


@pytest.fixture(scope="module")
def fitted():
    """A confounded log of a linear plant, its fit with each row's influence kept, and a plan on
    the fit's model at its exact optimum."""
    rng = np.random.default_rng(3)
    rows = 1500
    x = rng.normal(0.0, 1.0, (rows, 2))
    z = rng.normal(0.0, 1.0, rows)
    u = 0.9 * z - 0.3 * x[:, 0] + rng.normal(0.0, 0.5, rows)
    rate = x @ FIT_A.T + np.outer(u, FIT_B) + np.outer(z, [1.5, 0.0])
    data = {
        "x": jnp.asarray(x),
        "u": jnp.asarray(u[:, None]),
        "x_next": jnp.asarray(x + DT * rate + rng.normal(0.0, 0.05, (rows, 2))),
        "z": jnp.asarray(z[:, None]),
    }
    options = {"adjust_for": ("z",), "degree": 1, "channel_degree": 0, "seed": 0}
    fit = fit_causal_residual(_known, data, DT, influence=True, **options)
    bare = fit_causal_residual(_known, data, DT, **options)
    model = HybridDynamics(known=_known, residual=fit.residual)
    planned = causal_plan(model, jnp.zeros(2), FIT_COST, DT, 4, -1e2, 1e2)

    def cost(flat: jax.Array) -> jax.Array:
        return total_cost(model, jnp.zeros(2), flat.reshape(4, 1), DT, FIT_COST)

    still = jnp.zeros(4)
    optimum = -jnp.linalg.solve(jax.hessian(cost)(still), jax.grad(cost)(still))
    return fit, bare, _at(planned, np.asarray(optimum).reshape(4, 1))


def test_the_fit_s_channel_error_reads_as_the_one_step_channel_s(fitted) -> None:
    """With the drift's rows of the influence set to 0, the fit's covariance is the channel's
    alone, and on a field affine in the state the one-step channel is ``L`` times the field's:
    the bias read off the fit is the plan's own at ``L S L'``, to rounding."""
    fit, _, plan = fitted
    influence = np.asarray(fit.influence).copy()
    channel = fit.residual.channel.size
    influence[:, :, channel:] = 0.0
    alone = dataclasses.replace(fit, influence=jnp.asarray(influence))
    found = planned_cost_bias(plan, alone)
    terms = influence.reshape(-1, influence.shape[2])[:, :channel]
    covariance = terms.T @ terms

    def one_step(c: jax.Array) -> jax.Array:
        moved = HybridDynamics(
            known=_known,
            residual=ControlAffineResidual(
                drift=fit.residual.drift,
                channel=c.reshape(fit.residual.channel.shape),
                degree=1,
                channel_degree=0,
            ),
        )
        return jax.jacfwd(lambda u: rk4_step(moved, 0.0, jnp.zeros(2), u, DT))(jnp.zeros(1))

    lift = np.asarray(jax.jacfwd(one_step)(jnp.asarray(fit.residual.channel).ravel())).reshape(
        2, channel
    )
    own = plan.planned_cost_bias(lift @ covariance @ lift.T)
    assert found.optimism == pytest.approx(own.optimism, rel=1e-9, abs=0.0)
    assert found.curvature == pytest.approx(own.curvature, rel=1e-9, abs=0.0)
    assert found.planned_cost == pytest.approx(own.planned_cost, rel=1e-12, abs=0.0)
    assert (found.first_order, found.unseen) == (0.0, 0)
    # the drift's error is the plan's too
    whole = planned_cost_bias(plan, fit)
    assert whole.optimism > found.optimism
    assert whole.planned_cost == found.planned_cost


def test_it_refuses_a_fit_whose_covariance_is_missing(fitted) -> None:
    _, bare, plan = fitted
    with pytest.raises(ValueError, match="missing"):
        planned_cost_bias(plan, bare)


def test_it_refuses_a_plan_made_on_another_model(fitted) -> None:
    fit, _, _ = fitted
    other = causal_plan(
        LinearDynamics(jnp.asarray(FIT_A), jnp.asarray(FIT_B)[:, None]),
        jnp.zeros(2),
        FIT_COST,
        DT,
        4,
        -1e2,
        1e2,
    )
    with pytest.raises(ValueError, match="not made on the fit's model"):
        planned_cost_bias(other, fit)
