"""The channel's standard error on logs that break the fit's premises.

:func:`~chc.dynamics_id.fit_causal_residual` states a channel and its standard error, summed within
units. The contract read here: the 95 % interval ``channel +- 1.96 error`` covers the plant's
channel on 95 % of logs or more, less three Monte Carlo standard deviations
(``sqrt(0.95 * 0.05 / 40) = 0.034``), or, on each log it misses, the fit's own persistence check
(:func:`~chc.dynamics_id.persistence_check`) rejects its Markov premise at 5 %, which the
certificate reports as ``noise_persistence_p``, or, at ``nuisance_knots=None``, the fit warns that
a refit with spline nuisances moves its channel by more than three errors of the move.

The plant is ``x' = -0.5 x + 0.8 u + 1.5 z`` stepped by Euler at ``dt = 0.1``, 100 units over 20
periods, the lever set ``u = 0.9 z + 0.5 v`` from a recorded confounder ``z``; each test breaks one
premise.
"""

from __future__ import annotations

import functools
import math
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from chc.dynamics import LinearDynamics
from chc.dynamics_id import fit_causal_residual, persistence_check

pytestmark = pytest.mark.misspecification

CHANNEL, DT, LOGS = 0.8, 0.1, 40
FLOOR = 0.95 - 3.0 * math.sqrt(0.95 * 0.05 / LOGS)
KNOWN = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))


def _log(
    seed: int,
    *,
    noise_persistence: float = 0.0,
    lever_persistence: float = 0.0,
    heteroskedastic: bool = False,
    curved: bool = False,
    units: int = 100,
    periods: int = 20,
) -> tuple[dict[str, jnp.ndarray], np.ndarray, np.ndarray]:
    """One panel's transitions, with each one's unit and period. ``curved`` sets the drift to
    ``sin(2 x) - 0.5 x`` and the lever's rule to push back by ``1.5 tanh(2 x)``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=units)
    noise, part = rng.normal(size=units), rng.normal(size=units)
    columns: dict[str, list[np.ndarray]] = {"x": [], "u": [], "x_next": [], "z": []}
    for _ in range(periods):
        z = rng.normal(size=units)
        part = lever_persistence * part + math.sqrt(1.0 - lever_persistence**2) * rng.normal(
            size=units
        )
        u = 0.9 * z + 0.5 * part - (1.5 * np.tanh(2.0 * x) if curved else 0.0)
        noise = noise_persistence * noise + math.sqrt(1.0 - noise_persistence**2) * rng.normal(
            size=units
        )
        scale = 0.1 * (1.0 + np.abs(u)) if heteroskedastic else 0.1
        drift = np.sin(2.0 * x) - 0.5 * x if curved else -0.5 * x
        x_next = x + DT * (drift + CHANNEL * u + 1.5 * z) + scale * noise
        for name, value in (("x", x), ("u", u), ("x_next", x_next), ("z", z)):
            columns[name].append(value)
        x = x_next
    data = {name: jnp.asarray(np.concatenate(value))[:, None] for name, value in columns.items()}
    return data, np.tile(np.arange(units), periods), np.repeat(np.arange(periods), units)


@functools.cache
def _read(
    knots: str | None = "auto", **premises: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per log, whether the interval covers the channel, whether the persistence check rejects at
    5 %, and whether the fit warned that its nuisances miss the log's, which a fit with the
    polynomial alone checks. ``knots`` is the fit's ``nuisance_knots``, the default's."""
    covered, rejected, warned = [], [], []
    for seed in range(LOGS):
        data, units, periods = _log(seed, **premises)  # type: ignore[arg-type]
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            fit = fit_causal_residual(
                KNOWN,
                data,
                DT,
                adjust_for=("z",),
                channel_degree=0,
                clusters=units,
                nuisance_knots=knots,  # type: ignore[arg-type]
            )
        estimate = float(np.asarray(fit.residual.channel).ravel()[0])
        covered.append(abs(estimate - CHANNEL) <= 1.96 * fit.channel_error)
        rejected.append(persistence_check(fit, units, periods).p_value < 0.05)
        warned.append(
            any(r.category is UserWarning and "spline nuisances" in str(r.message) for r in records)
        )
    return np.array(covered), np.array(rejected), np.array(warned)


def _holds_or_says_so(covered: np.ndarray, rejected: np.ndarray, warned: np.ndarray) -> bool:
    return bool(np.mean(covered | rejected | warned) >= FLOOR)


def test_the_channel_error_covers_where_the_premises_hold() -> None:
    """The control case: 38 of 40 covered, the check rejects on none, and the spline refit moves
    the channel by 0.89 of its errors at most, so the fit warns on none."""
    covered, rejected, warned = _read()
    assert np.mean(covered) >= FLOOR
    assert np.mean(rejected) <= 0.05 + 3.0 * math.sqrt(0.05 * 0.95 / LOGS)
    assert np.mean(warned) <= 0.05 + 3.0 * math.sqrt(0.05 * 0.95 / LOGS)


def test_the_channel_error_covers_under_heteroskedastic_noise() -> None:
    """Noise whose sd grows with the lever, ``0.1 (1 + |u|)``: 38 of 40 covered."""
    covered, _, _ = _read(heteroskedastic=True)
    assert np.mean(covered) >= FLOOR


def test_noise_that_persists_within_units_is_covered_or_flagged() -> None:
    """AR(0.8) noise within units, the lever drawn afresh: the error, summed within units, covers
    on 37 of 40 (0.925); the check rejects on all 40."""
    assert _holds_or_says_so(*_read(noise_persistence=0.8))


def test_noise_and_a_lever_that_both_persist_are_flagged() -> None:
    """AR(0.8) noise and an AR(0.8) lever: the state is a common effect of the past lever and the
    past noise, and the channel reads 0.055 low, covered on 33 of 40 (0.825). The check rejects
    on all 40."""
    assert _holds_or_says_so(*_read(noise_persistence=0.8, lever_persistence=0.8))


def test_nuisances_outside_the_polynomial_are_covered_or_flagged() -> None:
    """Drift ``sin(2x) - 0.5x`` and the lever's rule ``-1.5 tanh(2x)``, outside the polynomial
    nuisances (degree 2, ``nuisance_knots=None``): the channel reads 0.42 low (8 errors) and is
    covered on none of 40, and the persistence check rejects on 26. A refit with spline nuisances
    moves it by 5.8 to 10.4 of its errors, and the fit's warning says so on all 40."""
    assert _holds_or_says_so(*_read(None, curved=True))


def test_the_default_spline_nuisances_cover_a_channel_outside_the_polynomial() -> None:
    """The same logs fitted at the default, ``nuisance_knots="auto"``: the bias is 0.0008, and the
    interval covers on 39 of 40."""
    covered, _, _ = _read(curved=True)
    assert np.mean(covered) >= FLOOR
