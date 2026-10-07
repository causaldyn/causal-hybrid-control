"""Koopman gate: an EDMD lift makes the nonlinear system linear enough to predict and control."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc import HybridDynamics
from chc.dynamics import DampedOscillator
from chc.integrate import rk4_step
from chc.koopman import KoopmanModel, koopman_controller, koopman_lqr_gain

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
