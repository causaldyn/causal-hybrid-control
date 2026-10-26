"""The channel's region by a grid of priors (Simpson, Baumgartner, Koehler and Diehl, Theorem 2).

``channel_region(form="grid")`` reads the least over 65 fixed prior matrices of a radius that puts
the prior's term inside the root and pays for a prior near ``D'D`` by ``sqrt(1 + rho)``, where the
default reads Abbasi-Yadkori et al.'s against the fit's ridge (ADR 0138).
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.dynamics import LinearDynamics
from chc.dynamics_id import ChannelRegion, _grid_members, channel_region, fit_causal_residual

jax.config.update("jax_enable_x64", True)

_DT, _SIGMA = 0.2, 0.05
_TRUTH = np.array([[[1.0, 0.0]]])
_KNOWN = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))


def _closed_loop_log(seed: int, rows: int = 1500) -> dict[str, jnp.ndarray]:
    """One unit under ``u = -0.5 x + 0.8 z + own draw``, Euler steps of ``x' = -0.5 x + u + 0.8 z``
    with noise, ``z`` logged (``tests/test_tube_levels.py``'s log)."""
    rng = np.random.default_rng(seed)
    x = 0.0
    cols: dict[str, list[list[float]]] = {name: [] for name in ("x", "u", "x_next", "z")}
    for _ in range(rows):
        z = rng.normal()
        u = -0.5 * x + 0.8 * z + 0.5 * rng.normal()
        x_next = x + _DT * (-0.5 * x + u + 0.8 * z) + _SIGMA * rng.normal()
        for name, value in (("x", x), ("u", u), ("x_next", x_next), ("z", z)):
            cols[name].append([value])
        rng.normal()
        x = x_next
    return {name: jnp.asarray(values) for name, values in cols.items()}


def _fit(seed: int):
    return fit_causal_residual(
        _KNOWN, _closed_loop_log(seed), _DT, adjust_for=("z",), integrator="euler"
    )


def test_the_grid_s_members_are_their_closed_forms() -> None:
    """``rho_j`` and ``l_j`` read off the matrices themselves, at a prior and a design written out
    (``validation/channel_region_grid.mac``, (5))."""
    prior = np.diag([2.0, 0.5])
    design = np.array([[7.0, 2.0], [2.0, 3.0]])
    root = np.diag(1.0 / np.sqrt(np.diag(prior)))
    inflation, log_dets = _grid_members(np.linalg.eigvalsh(root @ design @ root))
    assert inflation.size == log_dets.size == 65
    assert inflation[0] == 0.0
    assert log_dets[0] == pytest.approx(
        np.linalg.slogdet(np.eye(2) + np.linalg.solve(prior, design))[1], rel=1e-13, abs=0.0
    )
    for j in (1, 3, 10):
        bar = 2.0**j * prior
        rho = max(np.linalg.eigvals(np.linalg.solve(prior + design, bar)).real)
        assert inflation[j] == pytest.approx(rho, rel=1e-12, abs=0.0)
        assert log_dets[j] == pytest.approx(
            np.linalg.slogdet(np.eye(2) + np.linalg.solve(prior + bar, design))[1],
            rel=1e-10,
            abs=0.0,
        )


def test_the_grid_s_radius_is_the_least_over_its_members() -> None:
    """Two states at ``delta = 0.05``: 2.525683 and 10.102732, where the ridge's form reads
    3.325873 and 13.299492 (``validation/channel_region_grid.mac``, (6))."""
    region = ChannelRegion(
        level=0.95,
        channel=np.zeros((2, 1, 2)),
        gram=np.eye(2),
        radius=np.zeros(2),
        noise=np.array([0.5, 2.0]),
        prior=np.array([1e-3, 0.0]),
        log_det_ratio=float(np.sum(np.log1p([4e8, 2.5e7]))),
        channel_degree=1,
        form="grid",
        spectrum=np.array([4e8, 2.5e7]),
    )
    np.testing.assert_allclose(region._radii(0.05), [2.52568319, 10.10273191], rtol=2e-9, atol=0)
    ridge = dataclasses.replace(region, form="ridge")
    np.testing.assert_allclose(ridge._radii(0.05), [3.32587295, 13.29949179], rtol=2e-9, atol=0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_grid_s_region_is_narrower_and_holds_the_channel(seed: int) -> None:
    """On closed-loop logs the grid's radius is about 0.68 of the ridge's (1.16 for 1.69, median
    of 100 logs), and both hold the channel."""
    fit = _fit(seed)
    ridge, grid = channel_region(fit, 0.95), channel_region(fit, 0.95, form="grid")
    assert ridge.form == "ridge"
    assert grid.form == "grid"
    assert grid.radius[0] < 0.75 * ridge.radius[0]
    assert ridge.contains(_TRUTH)
    assert grid.contains(_TRUTH)
    np.testing.assert_array_equal(grid.channel, ridge.channel)
    np.testing.assert_array_equal(grid.gram, ridge.gram)


def test_the_grid_s_region_holds_the_channel_on_closed_loop_logs() -> None:
    """20 logs at 0.9: every one held, as the ridge's form holds them."""
    held = [channel_region(_fit(seed), 0.9, form="grid").contains(_TRUTH) for seed in range(20)]
    assert all(held)


def test_the_grid_s_stability_level_inverts_its_radius() -> None:
    """The level at which the operator radius reaches ``rho`` is the level ``rho`` was read at."""
    fit = _fit(3)
    state = np.array([0.4])
    for level in (0.5, 0.9, 0.99):
        region = channel_region(fit, level, form="grid")
        rho = region.operator_radius(state)
        assert region.stability_level(rho, state) == pytest.approx(level, rel=1e-9, abs=0.0)
    region = channel_region(fit, 0.9, form="grid")
    assert region.stability_level(0.0, state) == 0.0
    assert region.stability_level(math.inf, state) == 1.0


def test_channel_region_refuses_a_form_it_does_not_know() -> None:
    with pytest.raises(ValueError, match="neither 'ridge' nor 'grid'"):
        channel_region(_fit(0), form="union")  # type: ignore[arg-type]
