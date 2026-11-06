"""The channel's standard error on logs that break the fit's premises.

:func:`~chc.dynamics_id.fit_causal_residual` states a channel and its standard error, summed within
units. The contract read here: the 95 % interval ``channel +- 1.96 error`` covers the plant's
channel on 95 % of logs, or, on each log it misses, the fit's own persistence check
(:func:`~chc.dynamics_id.persistence_check`) rejects its Markov premise at 5 %, which the
certificate reports as ``noise_persistence_p``, or, at ``nuisance_knots=None``, the fit warns that
a refit with spline nuisances moves its channel by more than three errors of the move.

Each test reads 200 logs from its block of seeds (:mod:`_misspec_counts`) and counts the logs
covered, rejected and warned on apart. Where the premises hold, the count covered has the floor
179, below which a count of ``Binomial(200, 0.95)`` falls with probability at most
``alpha = 1/1000``; the sum over the logs of ``z^2``, ``z = (channel - 0.8) / error``, is at most
283.06, the ``1 - alpha / 10`` quantile of chi-square with 200 degrees, set past ``1 - alpha``
because ``z``'s tails are heavier than a normal's (kurtosis 3.4 to 3.6 where ``z^2`` averages 0.96
to 1.02), which widens the sum's spread; and the counts rejected and warned on have the ceiling 21
of ``Binomial(200, 0.05)``, so a check that flags every log fails. Where a premise fails, the logs
covered, rejected or warned on have the floor 179.

Against an error ``k`` times its own, a normal ``z`` is covered with probability
``p1 = erf(1.96 k / sqrt 2)``, 0.883 at ``k = 0.8``: the floor fails a block with probability 0.651
there and the sum with 0.826, and each fails with probability 0.8 below ``k = 0.778`` and
``k = 0.804``, its minimal detectable defect (``validation/misspec_power.mac``). Each test below
gives the counts of block 0, which CI reads, and the rates over the blocks measured.

The plant is ``x' = -0.5 x + 0.8 u + 1.5 z`` stepped by Euler at ``dt = 0.1``, 100 units over 20
periods, the lever set ``u = 0.9 z + 0.5 v`` from a recorded confounder ``z``; each test breaks one
premise.
"""

from __future__ import annotations

import functools
import math
import warnings
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest
from _misspec_counts import ALPHA, LOGS, at_least, at_most, seeds
from scipy.stats import chi2

from chc.dynamics import LinearDynamics
from chc.dynamics_id import fit_causal_residual, persistence_check

pytestmark = pytest.mark.misspecification

CHANNEL, DT = 0.8, 0.1
COVERED, FLAGGED = at_least("0.95"), at_most("0.05")
Z_SQUARED = float(chi2.ppf(1.0 - float(ALPHA) / 10.0, LOGS))
KNOWN = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))


class Counts(NamedTuple):
    covered: np.ndarray
    z: np.ndarray
    rejected: np.ndarray
    warned: np.ndarray

    def __repr__(self) -> str:
        return (
            f"of {self.z.size}: covered {self.covered.sum()}, rejected {self.rejected.sum()}, "
            f"warned {self.warned.sum()}, sum z^2 {np.sum(self.z**2):.1f}"
        )


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
def _read(knots: str | None = "auto", **premises: object) -> Counts:
    """Per log of the block, whether the interval covers the channel, its ``z``, whether the
    persistence check rejects at 5 %, and whether the fit warned that its nuisances miss the log's,
    which a fit with the polynomial alone checks. ``knots`` is the fit's ``nuisance_knots``, the
    default's."""
    covered, z, rejected, warned = [], [], [], []
    for seed in seeds(0):
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
        z.append((float(np.asarray(fit.residual.channel).ravel()[0]) - CHANNEL) / fit.channel_error)
        covered.append(abs(z[-1]) <= 1.96)
        rejected.append(persistence_check(fit, units, periods).p_value < 0.05)
        warned.append(
            any(r.category is UserWarning and "spline nuisances" in str(r.message) for r in records)
        )
    return Counts(np.array(covered), np.array(z), np.array(rejected), np.array(warned))


def _holds(counts: Counts) -> None:
    assert counts.covered.sum() >= COVERED, counts
    assert np.sum(counts.z**2) <= Z_SQUARED, counts
    assert counts.rejected.sum() <= FLAGGED, counts
    assert counts.warned.sum() <= FLAGGED, counts


def _holds_or_says_so(counts: Counts) -> None:
    assert np.sum(counts.covered | counts.rejected | counts.warned) >= COVERED, counts


def test_the_channel_error_covers_where_the_premises_hold() -> None:
    """The control case, ``p0 = 0.95``. Block 0: covered 189, sum of ``z^2`` 201.9, rejected 7,
    warned on none. Over blocks 0 to 4, covered 0.940 of 1000, where the floor fails a block with
    probability 0.005, and rejected 0.051. An error 0.8 times its own covers ``p1 = 0.888``: the
    floor fails a block with probability 0.57 there, and the sum, resampled from the 1000, 0.78."""
    _holds(_read())


def test_the_channel_error_covers_under_heteroskedastic_noise() -> None:
    """Noise whose sd grows with the lever, ``0.1 (1 + |u|)``, ``p0 = 0.95``. Block 0: covered
    190, sum of ``z^2`` 202.9, rejected 9, warned on none. Over blocks 0 to 4, covered 0.946 of
    1000. An error 0.8 times its own covers ``p1 = 0.890``: the floor fails a block with
    probability 0.53 there, and the sum 0.80."""
    _holds(_read(heteroskedastic=True))


def test_noise_that_persists_within_units_is_covered_or_flagged() -> None:
    """AR(0.8) noise within units, the lever drawn afresh: the error, summed within units, still
    covers, and the check rejects. Block 0: covered 192, rejected 200. Over blocks 0 to 4, covered
    0.951 of 1000, rejected every one."""
    _holds_or_says_so(_read(noise_persistence=0.8))


def test_noise_and_a_lever_that_both_persist_are_flagged() -> None:
    """AR(0.8) noise and an AR(0.8) lever: the state is a common effect of the past lever and the
    past noise, and the channel reads 0.050 low, 0.6 of its error. Block 0: covered 182, rejected
    200. Over blocks 0 to 4, covered 0.898 of 1000, rejected every one."""
    _holds_or_says_so(_read(noise_persistence=0.8, lever_persistence=0.8))


def test_nuisances_outside_the_polynomial_are_covered_or_flagged() -> None:
    """Drift ``sin(2x) - 0.5x`` and the lever's rule ``-1.5 tanh(2x)``, outside the polynomial
    nuisances (degree 2, ``nuisance_knots=None``): the channel reads 0.42 low, 8.3 errors. A refit
    with spline nuisances moves it by more than three errors of the move, and the fit warns. Block
    0: covered none, rejected 126, warned on 200. Over blocks 0 to 4, covered none of 1000,
    rejected 0.603, warned on every one."""
    _holds_or_says_so(_read(None, curved=True))


def test_the_default_spline_nuisances_cover_a_channel_outside_the_polynomial() -> None:
    """The same logs fitted at the default, ``nuisance_knots="auto"``, ``p0 = 0.95``: the channel
    reads 0.002 low. Block 0: covered 191, sum of ``z^2`` 194.4, rejected 8, warned on none. Over
    blocks 0 to 4, covered 0.950 of 1000. An error 0.8 times its own covers ``p1 = 0.892``: the
    floor fails a block with probability 0.50 there, and the sum 0.72."""
    _holds(_read(curved=True))
