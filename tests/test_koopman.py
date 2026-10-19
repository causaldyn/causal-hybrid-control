"""Koopman gate: an EDMD lift makes the nonlinear system linear enough to predict and control."""

import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc import HybridDynamics
from chc.dynamics import DampedOscillator
from chc.integrate import rk4_step
from chc.regret import dlqr
from chc.research.koopman import KoopmanModel, _lift, koopman_controller, koopman_lqr_gain

DT = 0.05


class _Cubic(eqx.Module):
    beta: float

    def __call__(self, t: float, x: jax.Array, u: jax.Array) -> jax.Array:
        return jnp.array([0.0, -self.beta * x[0] ** 3])


def _plant() -> HybridDynamics:
    return HybridDynamics(known=DampedOscillator(omega=1.0, zeta=0.1), residual=_Cubic(beta=0.5))


def _transitions(n: int = 3000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_x, k_u = jax.random.split(jax.random.key(0))
    xs = jax.random.normal(k_x, (n, 2))
    us = 0.5 * jax.random.normal(k_u, (n, 1))
    x_next = jax.vmap(lambda x, u: rk4_step(_plant(), 0.0, x, u, DT))(xs, us)
    return np.asarray(xs), np.asarray(us), np.asarray(x_next)


def test_koopman_predicts_the_lifted_nonlinear_dynamics() -> None:
    xs, us, x_next = _transitions()
    model = KoopmanModel(degree=3).fit(xs, us, x_next)
    rmse = float(np.sqrt(np.mean((model.predict(xs, us) - x_next) ** 2)))
    assert rmse < 0.01  # the polynomial lift makes the cubic oscillator near-linear


@pytest.mark.parametrize("units", [2.0**-30, 1e-9, 1e6])
def test_the_lqr_gain_reads_the_same_in_any_units_of_the_action(units: float) -> None:
    """The model's B in units `s` of the action, and R in units `s^2`: the same gain in `s`.

    SciPy's DARE read B and R apart, so at a billionth of the action's units the gain was off by
    4e-4.
    """
    xs, us, x_next = _transitions()
    model = KoopmanModel(degree=3).fit(xs, us, x_next)
    q, r = np.diag([10.0, 1.0]), np.array([[0.1]])
    gain = koopman_lqr_gain(model, q, r)
    in_units = copy.copy(model)
    assert model._b is not None
    in_units._b = model._b / units
    read = koopman_lqr_gain(in_units, q, r / units**2) / units
    if units == 2.0**-30:
        np.testing.assert_array_equal(read, gain)
    else:
        assert np.max(np.abs(read - gain)) <= 1e-12 * np.max(np.abs(gain))


def test_koopman_lqr_regulates_the_true_system_to_target() -> None:
    xs, us, x_next = _transitions()
    model = KoopmanModel(degree=3).fit(xs, us, x_next)
    gain = koopman_lqr_gain(model, np.diag([10.0, 1.0]), np.array([[0.1]]))
    control = koopman_controller(model, gain, np.array([1.0, 0.0]))
    plant, x = _plant(), np.array([0.0, 0.0])
    for _ in range(80):
        u = np.clip(control(x), -10.0, 10.0)
        x = np.asarray(rk4_step(plant, 0.0, jnp.asarray(x), jnp.asarray(u), DT))
    assert (
        x[0] > 0.7
    )  # LQR on the Koopman matrices drives the true plant toward target position 1.0


UNITS = [1e-9, 1e-6, 1e-3, 1e3, 1e6]
Q, R, TARGET = np.diag([10.0, 1.0]), np.array([[0.1]]), np.array([1.0, 0.0])


def _assert_close(other: np.ndarray, one: np.ndarray, rel: float) -> None:
    """Agreement to ``rel`` of the reference's largest entry."""
    np.testing.assert_allclose(other, one, rtol=0.0, atol=rel * float(np.max(np.abs(one))))


def _actions(control, states: np.ndarray) -> np.ndarray:
    return np.stack([control(x) for x in states])


@pytest.mark.parametrize("units", UNITS)
def test_the_koopman_fit_reads_the_same_with_the_action_in_any_units(units: float) -> None:
    """The fit's ridge was a constant on the Gram of the raw ``[phi(x), u]``: with the action
    logged in millionths of its units, ``B`` read 0.00004 per original unit where it reads 0.0497,
    and a one-step prediction moved by 1.6% of the largest."""
    xs, us, x_next = _transitions()
    one = KoopmanModel(degree=3).fit(xs, us, x_next)
    other = KoopmanModel(degree=3).fit(xs, us * units, x_next)
    assert one._b is not None
    assert other._b is not None
    _assert_close(other._b * units, one._b, 1e-9)
    _assert_close(other.predict(xs[:50], us[:50] * units), one.predict(xs[:50], us[:50]), 1e-9)


@pytest.mark.parametrize("units", UNITS[1:])
def test_the_lqr_gain_and_its_controller_read_the_same_with_the_action_in_any_units(
    units: float,
) -> None:
    """With the action logged in millionths of its units, and its cost weighted to match, the gain
    was 0.99 off and the controller's action 0.998, because ``B`` was. The range stops at 1e-6:
    at 1e-9 the Riccati solve weighs the action by 1e17, and given ``B`` scaled exactly it is off
    by 4e-4 on its own, while the fit agrees to 1e-15."""
    xs, us, x_next = _transitions()
    one = KoopmanModel(degree=3).fit(xs, us, x_next)
    other = KoopmanModel(degree=3).fit(xs, us * units, x_next)
    gain_one, gain_other = koopman_lqr_gain(one, Q, R), koopman_lqr_gain(other, Q, R / units**2)
    _assert_close(gain_other / units, gain_one, 1e-6)
    _assert_close(
        _actions(koopman_controller(other, gain_other, TARGET), xs[:50]) / units,
        _actions(koopman_controller(one, gain_one, TARGET), xs[:50]),
        1e-6,
    )


# SciPy's matrix_balance casts its scaling factors to int with its permutation, and the Riccati
# solve's balancing reaches a factor past 2**63 at 1e-9: the cast warns, and nothing reads it.
@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
@pytest.mark.parametrize("units", UNITS)
def test_the_koopman_fit_and_its_controller_read_the_same_with_the_state_in_any_units(
    units: float,
) -> None:
    """With the state logged in millionths of its units the Gram's diagonal fell to 3e-9 and
    below, under a ridge of 1e-6: a one-step prediction was 0.98 off, and the controller's action,
    its cost on the state weighted to match, 1.0 off."""
    xs, us, x_next = _transitions()
    one = KoopmanModel(degree=3).fit(xs, us, x_next)
    other = KoopmanModel(degree=3).fit(xs * units, us, x_next * units)
    _assert_close(
        other.predict(xs[:50] * units, us[:50]) / units, one.predict(xs[:50], us[:50]), 1e-9
    )
    control_one = koopman_controller(one, koopman_lqr_gain(one, Q, R), TARGET)
    gain_other = koopman_lqr_gain(other, Q / units**2, R)
    control_other = koopman_controller(other, gain_other, TARGET * units)
    _assert_close(_actions(control_other, xs[:50] * units), _actions(control_one, xs[:50]), 1e-9)


def test_an_action_logged_at_zero_keeps_a_ridge_term_and_reads_no_channel() -> None:
    """A column of zeros has no mean square to scale the ridge by, and without a term of its own
    the Gram is singular there; with one, the unused action reads a channel of zero and the rest
    of the fit is as before."""
    xs, us, x_next = _transitions()
    one = KoopmanModel(degree=3).fit(xs, us, x_next)
    padded = KoopmanModel(degree=3).fit(xs, np.column_stack([us, np.zeros(len(us))]), x_next)
    assert one._a is not None
    assert one._b is not None
    assert padded._a is not None
    assert padded._b is not None
    assert np.all(padded._b[:, 1] == 0.0)
    _assert_close(padded._b[:, :1], one._b, 1e-12)
    _assert_close(padded._a, one._a, 1e-12)


# A bilinear plant x' = A x + b u + c x1 u (validation/koopman_bilinear_closure.mac (4)).
BIL_A = np.array([[0.9, 0.1], [-0.1, 0.8]])
BIL_B, BIL_C = np.array([0.0, 0.5]), np.array([0.2, 1.0])
BIL_RHO = np.sqrt([2.0 / 43.0, 10.0 / 19.0])  # the linear form's closure residuals, s = 1


def _bilinear_plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return x @ BIL_A.T + u * BIL_B + (x[:, :1] * u) * BIL_C


def _flipped(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``n / 4`` draws, each followed by its sign flips ``(-x, u)``, ``(x, -u)``, ``(-x, -u)``.

    Each fold of ``n / 5`` transitions, a multiple of four, holds whole quadruples, so every fit
    reads ``x1 u`` orthogonal to the linear columns, as in the .mac file's (1)."""
    rng = np.random.default_rng(seed)
    x, u = rng.standard_normal((n // 4, 2)), rng.standard_normal((n // 4, 1))
    signs = [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]
    xs = np.stack([sx * x for sx, _ in signs], axis=1).reshape(n, 2)
    us = np.stack([su * u for _, su in signs], axis=1).reshape(n, 1)
    return xs, us, _bilinear_plant(xs, us)


def _gaussian(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x, u = rng.standard_normal((n, 2)), rng.standard_normal((n, 1))
    return x, u, _bilinear_plant(x, u)


def test_the_default_form_fits_as_the_linear_lift_did_bit_for_bit() -> None:
    xs, us, x_next = _transitions()
    model = KoopmanModel(degree=3).fit(xs, us, x_next)
    phi = _lift(xs, 3)
    design = np.column_stack([phi, us])
    gram = design.T @ design + 1e-6 * np.diag(np.mean(design**2, axis=0))
    coef = np.linalg.solve(gram, design.T @ _lift(x_next, 3))
    np.testing.assert_array_equal(model._a, coef[: phi.shape[1]].T)
    np.testing.assert_array_equal(model._b, coef[phi.shape[1] :].T)
    assert model.form == "linear"
    assert model._h is None


def test_a_bilinear_lift_closes_on_a_bilinear_plant_where_a_linear_one_misses_x1_u() -> None:
    """Without a ridge the fits are exact: the bilinear form's error is round-off, and the linear
    form's is ``c x1 u`` at every transition, held out or not (the .mac file's (1) and (2))."""
    xs, us, x_next = _flipped(400)
    linear = KoopmanModel(degree=1, ridge=0.0).fit(xs, us, x_next)
    bilinear = KoopmanModel(degree=1, ridge=0.0, form="bilinear").fit(xs, us, x_next)
    scale = float(np.max(np.abs(x_next)))
    missed = (xs[:, :1] * us) * BIL_C
    np.testing.assert_allclose(
        linear.residuals(xs, us, x_next), missed, rtol=0.0, atol=1e-14 * scale
    )
    held_out = np.sqrt(np.sum(missed**2, axis=0) / np.sum(x_next**2, axis=0))
    assert linear.closure_residuals is not None
    np.testing.assert_allclose(linear.closure_residuals, held_out, rtol=1e-12, atol=0.0)
    assert np.max(np.abs(bilinear.residuals(xs, us, x_next))) <= 1e-14 * scale
    assert bilinear.closure_residuals is not None
    assert np.max(bilinear.closure_residuals) <= 1e-14
    fresh, fresh_u, fresh_next = _gaussian(50, seed=1)
    _assert_close(bilinear.predict(fresh, fresh_u), fresh_next, 1e-13)
    _assert_close(
        bilinear.rollout(fresh[0], fresh_u[:10])[-1], _plant_rollout(fresh[0], fresh_u[:10]), 1e-13
    )


def _plant_rollout(x0: np.ndarray, us: np.ndarray) -> np.ndarray:
    x = x0[None, :]
    for u in us:
        x = _bilinear_plant(x, u[None, :])
    return x[0]


def test_the_closure_residuals_match_their_closed_form_on_gaussian_transitions() -> None:
    """``rho_k = |c_k| / sqrt(|a_k|^2 + b_k^2 + c_k^2)`` (the .mac file's (3) and (4)); the
    bilinear form reads the ridge's shrinkage alone."""
    xs, us, x_next = _gaussian(20_000)
    linear = KoopmanModel(degree=1).fit(xs, us, x_next)
    bilinear = KoopmanModel(degree=1, form="bilinear").fit(xs, us, x_next)
    assert linear.closure_residuals is not None
    assert bilinear.closure_residuals is not None
    np.testing.assert_allclose(linear.closure_residuals, BIL_RHO, rtol=0.03, atol=0.0)
    assert np.max(bilinear.closure_residuals) <= 1e-5


def test_every_closure_residual_is_round_off_on_a_linear_plant() -> None:
    rng = np.random.default_rng(2)
    xs, us = rng.standard_normal((200, 2)), rng.standard_normal((200, 1))
    x_next = xs @ BIL_A.T + us * BIL_B
    for form in ("linear", "bilinear"):
        model = KoopmanModel(degree=1, ridge=0.0, form=form).fit(xs, us, x_next)
        assert model.closure_residuals is not None
        assert np.max(model.closure_residuals) <= 1e-14


def test_the_lqr_gain_refuses_a_lift_that_does_not_close() -> None:
    xs, us, x_next = _gaussian(400)
    linear = KoopmanModel(degree=1).fit(xs, us, x_next)
    q, r = np.eye(2), np.array([[1.0]])
    with pytest.raises(ValueError, match=r"feature 1's held-out closure residual is 0\.7"):
        koopman_lqr_gain(linear, q, r)
    assert linear.closure_residuals is not None
    worst = float(np.max(linear.closure_residuals))
    koopman_lqr_gain(linear, q, r, max_residual=worst)
    with pytest.raises(ValueError, match="does not close"):
        koopman_lqr_gain(linear, q, r, max_residual=np.nextafter(worst, 0.0))
    bilinear = KoopmanModel(degree=1, form="bilinear").fit(xs, us, x_next)
    koopman_lqr_gain(bilinear, q, r, x_target=np.zeros(2))


@pytest.mark.filterwarnings("error")
def test_the_lqr_gain_refuses_a_lift_whose_closure_was_not_read() -> None:
    """One transition leaves none to hold out: the closure is unread, and the gain refused."""
    xs, us, x_next = _gaussian(1)
    model = KoopmanModel(degree=1).fit(xs, us, x_next)
    assert model.closure_residuals is not None
    assert np.all(np.isnan(model.closure_residuals))
    with pytest.raises(ValueError, match="closure residual is nan"):
        koopman_lqr_gain(model, np.eye(2), np.array([[1.0]]), max_residual=np.inf)


def test_a_bilinear_gain_is_the_lqr_of_the_plant_linearised_at_the_target() -> None:
    """At ``x_t`` the plant's channel is ``b + c x1_t`` (the .mac file's (5))."""
    xs, us, x_next = _flipped(400)
    model = KoopmanModel(degree=1, ridge=0.0, form="bilinear").fit(xs, us, x_next)
    q, r, target = np.diag([10.0, 1.0]), np.array([[0.1]]), np.array([1.5, -0.5])
    channel = (BIL_B + BIL_C * target[0])[:, None]
    _assert_close(model.channel(target), channel, 1e-13)
    expected = dlqr(BIL_A, channel, q, r)[0]
    _assert_close(koopman_lqr_gain(model, q, r, x_target=target), expected, 1e-10)
    with pytest.raises(ValueError, match="pass x_target"):
        koopman_lqr_gain(model, q, r)


@pytest.mark.parametrize("form", ["linear", "bilinear"])
@pytest.mark.parametrize("units", UNITS)
def test_the_closure_residuals_read_the_same_in_any_units(form: str, units: float) -> None:
    xs, us, x_next = _gaussian(400)
    one = KoopmanModel(degree=2, form=form).fit(xs, us, x_next)
    state = KoopmanModel(degree=2, form=form).fit(xs * units, us, x_next * units)
    action = KoopmanModel(degree=2, form=form).fit(xs, us * units, x_next)
    assert one.closure_residuals is not None
    for other in (state, action):
        assert other.closure_residuals is not None
        np.testing.assert_allclose(
            other.closure_residuals, one.closure_residuals, rtol=0.0, atol=1e-7
        )
    _assert_close(action.predict(xs[:50], us[:50] * units), one.predict(xs[:50], us[:50]), 1e-9)


def test_an_unknown_form_is_refused() -> None:
    with pytest.raises(ValueError, match="form must be one of"):
        KoopmanModel(form="quadratic")


def test_the_closure_residuals_are_read_across_five_contiguous_blocks() -> None:
    xs, us, x_next = _transitions()
    model = KoopmanModel(degree=3).fit(xs, us, x_next)
    target = _lift(x_next, 3)
    error = np.empty_like(target)
    for block in np.array_split(np.arange(len(xs)), 5):
        train = np.setdiff1d(np.arange(len(xs)), block)
        fold = KoopmanModel(degree=3).fit(xs[train], us[train], x_next[train])
        error[block] = fold.residuals(xs[block], us[block], x_next[block])
    spread = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    expected = np.sqrt(np.sum(error**2, axis=0) / spread)
    assert model.closure_residuals is not None
    np.testing.assert_allclose(model.closure_residuals, expected, rtol=1e-12, atol=0.0)


def test_the_closure_residuals_are_held_out_of_the_fit() -> None:
    """Twenty transitions with no dynamics, ten coefficients per feature: the fit reads up to 0.87
    of each feature's root mean square in-sample, and more than all of it held out."""
    rng = np.random.default_rng(4)
    xs, us, x_next = (rng.standard_normal((20, d)) for d in (2, 1, 2))
    model = KoopmanModel(degree=3).fit(xs, us, x_next)
    in_sample = np.sqrt(
        np.sum(model.residuals(xs, us, x_next) ** 2, axis=0) / np.sum(_lift(x_next, 3) ** 2, axis=0)
    )
    assert model.closure_residuals is not None
    assert np.max(in_sample) < 1.0 < np.min(model.closure_residuals)


def test_a_bilinear_lift_closes_on_a_plant_with_two_actions() -> None:
    """``x' = A x + B u + u1 C1 x + u2 C2 x``: each action's term read apart, to round-off."""
    rng = np.random.default_rng(5)
    xs, us = rng.standard_normal((200, 2)), rng.standard_normal((200, 2))
    b = np.array([[0.5, 0.0], [0.1, -0.3]])
    c1, c2 = np.array([[0.0, 0.2], [0.4, 0.0]]), np.array([[-0.1, 0.0], [0.3, 0.6]])
    x_next = xs @ BIL_A.T + us @ b.T + us[:, :1] * (xs @ c1.T) + us[:, 1:] * (xs @ c2.T)
    model = KoopmanModel(degree=1, ridge=0.0, form="bilinear").fit(xs, us, x_next)
    assert model.closure_residuals is not None
    assert np.max(model.closure_residuals) <= 1e-14
    _assert_close(
        model.channel(np.array([1.0, -2.0])),
        b + np.column_stack([c1 @ [1.0, -2.0], c2 @ [1.0, -2.0]]),
        1e-13,
    )


def test_a_state_the_log_never_moves_reads_a_closure_residual_of_zero() -> None:
    xs, us, x_next = _gaussian(200)
    xs, x_next = np.column_stack([xs, np.zeros(200)]), np.column_stack([x_next, np.zeros(200)])
    model = KoopmanModel(degree=1).fit(xs, us, x_next)
    assert model.closure_residuals is not None
    assert model.closure_residuals[2] == 0.0
    koopman_lqr_gain(model, np.eye(3), np.array([[1.0]]), max_residual=0.8)


@pytest.mark.parametrize("level", [0.0, 273.15])
def test_a_lift_that_does_not_close_is_refused_at_any_level_of_the_state(level: float) -> None:
    """``x' = 2x - x^3 + u/10`` on ``[-1.5, 1.5]`` leaves a degree-one lift with no cube. Read over
    the feature's root mean square, the residual of the same plant logged 273.15 higher, as in
    kelvin, fell from 0.71 to 0.0024 and the gain was granted; read over its spread about its
    mean it is 0.85 there, and the gain refused."""
    rng = np.random.default_rng(0)
    xs, us = rng.uniform(-1.5, 1.5, (600, 1)), rng.uniform(-1.0, 1.0, (600, 1))
    x_next = xs + (xs - xs**3) + 0.1 * us
    model = KoopmanModel(degree=1).fit(xs + level, us, x_next + level)
    assert model.closure_residuals is not None
    assert float(model.closure_residuals[0]) > 0.6
    with pytest.raises(ValueError, match="does not close"):
        koopman_lqr_gain(model, np.eye(1), np.array([[0.1]]))


@pytest.mark.parametrize("level", [0.1, 273.15])
def test_a_state_the_log_holds_at_a_level_reads_a_closure_residual_near_zero(level: float) -> None:
    """A state held at one value has no spread about its mean, and its mean need not round to the
    value: read over that spread, its residual was the ridge's rounding over nothing, infinite or
    far above 1, and the gain refused. It is read over its root mean square."""
    xs, us, x_next = _gaussian(200)
    held = np.full(200, level)
    xs, x_next = np.column_stack([xs, held]), np.column_stack([x_next, held])
    model = KoopmanModel(degree=1).fit(xs, us, x_next)
    assert model.closure_residuals is not None
    assert model.closure_residuals[2] < 1e-6
    koopman_lqr_gain(model, np.eye(3), np.array([[1.0]]), max_residual=0.8)
