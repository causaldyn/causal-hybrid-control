"""The switchback's design standard error, ``read_switchback(..., se="design")``: Liang and Recht's
(Biometrika 2025) moment estimator of ``tau_H`` on a lever drawn by a fair coin each period, and
its variance over the coins with the noise held fixed.

Its mean and variance are checked over every design of a short run, against their eq. (2.5)
(``validation/switchback_design_variance.mac`` derives both), and its interval by simulation on a
plant whose noise persists or trends, where the regression's interval does not hold its level.
"""

from __future__ import annotations

import itertools
import logging
import math

import numpy as np
import pytest

import chc.switchback as switchback
from chc.switchback import CHANNEL, STEADY_STATE, BlockDesign, Horizon, read_switchback

A, B = 0.8, 2.0
LEVEL = 0.95
BURN = 400


def _tau(h: int) -> float:
    return B * (1.0 - A**h) / (1.0 - A)


def _plant(u: np.ndarray, shocks: np.ndarray) -> np.ndarray:
    """``x_(t+1) = A x_t + B u_t + shocks_t`` from rest, over the last axis; one more period."""
    x = np.zeros((*u.shape[:-1], u.shape[-1] + 1))
    for t in range(u.shape[-1]):
        x[..., t + 1] = A * x[..., t] + B * u[..., t] + shocks[..., t]
    return x


def _persistent(rng, shape, rho: float) -> np.ndarray:
    """AR(``rho``) shocks with unit innovations, started ``BURN`` periods before."""
    innovations = rng.standard_normal((*shape[:-1], shape[-1] + BURN))
    shocks = np.zeros_like(innovations)
    for t in range(1, innovations.shape[-1]):
        shocks[..., t] = rho * shocks[..., t - 1] + innovations[..., t]
    return shocks[..., BURN:]


def _coins(rng, shape) -> np.ndarray:
    return (rng.random(shape) < 0.5).astype(np.float64)


def _moment_estimate(u: np.ndarray, y: np.ndarray, h: int) -> float:
    """The estimator, transcribed: ``sum_(s<H) 2/(T - s) sum_t (2 u_t - 1) (y_(t+1+s) - y_0)``."""
    periods = u.size
    z, after = 2.0 * u - 1.0, y[1:] - y[0]
    return sum(2.0 / (periods - s) * (z[: periods - s] * after[s:]).sum() for s in range(h))


def _toeplitz(v: np.ndarray, periods: int) -> np.ndarray:
    m = np.zeros((periods, periods))
    for i in range(periods):
        for j in range(max(0, i - v.size + 1), i + 1):
            m[i, j] = v[i - j]
    return m


def _dense_variance(q, g, g_long, h) -> float:
    """Liang and Recht's eq. (2.5) over full matrices, ``tr(M M)`` from ``g_long``."""
    periods = h.size
    tq = _toeplitz(q, periods)
    short, long = tq.T @ _toeplitz(g, periods), tq.T @ _toeplitz(g_long, periods)
    linear = tq.T @ h
    return float(
        (short**2).sum()
        + np.trace(long @ long)
        - 2.0 * (np.diag(short) ** 2).sum()
        + linear @ linear
    )


def _plug_in(u: np.ndarray, y: np.ndarray, h: int, lags: int) -> float:
    """The plug-in of eq. (2.5), transcribed over full matrices: ``g`` read over ``2 lags`` lags,
    the noise as ``2 y' - T_g z`` over the first ``lags``, and ``q_s = T / (T - s)``."""
    periods = u.size
    z, after = 2.0 * u - 1.0, y[1:] - y[0]
    g = np.array(
        [2.0 / (periods - s) * (z[: periods - s] * after[s:]).sum() for s in range(2 * lags)]
    )
    noise = 2.0 * after - _toeplitz(g[:lags], periods) @ z
    q = periods / (periods - np.arange(h, dtype=np.float64))
    return _dense_variance(q, g[:lags], g, noise) / periods**2


@pytest.mark.parametrize("h", [1, 2, 3])
def test_the_estimate_is_unbiased_and_its_variance_is_eq_2_5_over_every_design(monkeypatch, h):
    # the plug-in reads ceil(ln 10) = 3 lags here, so that a run of 10 can be read at all
    monkeypatch.setattr(switchback, "_DESIGN_LAGS", 1.0)
    rng = np.random.default_rng(1)
    periods = 10
    g = rng.normal(size=periods) * 0.7 ** np.arange(periods)
    # a trend and a level, fixed: the design holds them and redraws only the coins
    noise = np.concatenate([[4.0], 3.0 + np.cumsum(rng.normal(size=periods))])
    estimates = []
    for bits in itertools.product((0.0, 1.0), repeat=periods):
        u = np.array(bits)
        y = noise + np.concatenate([[0.0], np.convolve(u, g)[:periods]])
        estimate = _moment_estimate(u, y, h)
        if 0.0 < u.sum() < periods:
            reading = read_switchback(u, y, Horizon(h), "local_projection", se="design")
            assert reading.estimate == pytest.approx(estimate, rel=1e-12, abs=1e-12)
            # at H = 3 the plug-in's tr(M M) reads g beyond the 3 lags its other terms read
            plug_in = _plug_in(u, y, h, max(h, math.ceil(math.log(periods))))
            assert reading.se**2 == pytest.approx(plug_in, rel=1e-10, abs=0.0)
        estimates.append(estimate)
    estimates = np.array(estimates)
    assert estimates.mean() == pytest.approx(g[:h].sum(), rel=0.0, abs=1e-12)
    # T^2 var = |M|_F^2 + tr(M M) - 2 sum M_ii^2 + |T_q' h|^2, h = T_g 1 + 2 (e - y_0)
    q = periods / (periods - np.arange(h, dtype=np.float64))
    level = np.convolve(np.ones(periods), g)[:periods] + 2.0 * (noise[1:] - noise[0])
    exact = switchback._design_variance(q, g[None], g[None], level[None])[0] / periods**2
    assert estimates.var() == pytest.approx(exact, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(("h", "lags"), [(1, 3), (3, 4), (4, 9)])
def test_the_banded_variance_is_the_dense_one(h, lags):
    rng = np.random.default_rng(h)
    periods = 30
    q = periods / (periods - np.arange(h, dtype=np.float64))
    g_long = rng.normal(size=2 * lags)
    g = g_long[:lags]
    level = rng.normal(size=periods)
    banded = switchback._design_variance(q, g[None], g_long[None], level[None])[0]
    assert banded == pytest.approx(_dense_variance(q, g, g_long, level), rel=1e-12, abs=0.0)


def _coverage(u, y, h: int, se: str) -> tuple[float, np.ndarray, np.ndarray]:
    readings = [
        read_switchback(u[r], y[r], Horizon(h), "local_projection", se=se) for r in range(len(u))
    ]
    covered = np.mean([x.interval[0] <= _tau(h) <= x.interval[1] for x in readings])
    return (
        float(covered),
        np.array([x.estimate for x in readings]),
        np.array([x.se for x in readings]),
    )


def test_the_design_interval_holds_its_level_on_a_short_run_with_persistent_noise():
    # AR(0.8) shocks over 60 periods: the local projection's interval for tau_3 covered 0.903 of
    # the 995 runs it read; the design's covered 0.969, at 0.95 +- 0.028 four standard errors. It
    # pays in width: 23.5 against 6.2
    rng = np.random.default_rng(60)
    reps, periods, h = 1000, 60, 3
    u = _coins(rng, (reps, periods + BURN))
    y = _plant(u, _persistent(rng, u.shape, 0.8))
    u, y = u[:, BURN:], y[:, BURN:]
    covered, estimates, se = _coverage(u, y, h, "design")
    assert covered == pytest.approx(LEVEL, abs=4 * math.sqrt(LEVEL * (1 - LEVEL) / reps))
    assert estimates.mean() == pytest.approx(_tau(h), abs=4 * estimates.std() / math.sqrt(reps))
    assert se.mean() / estimates.std() == pytest.approx(1.0, abs=0.1)
    model = []
    for r in range(reps):
        try:
            reading = read_switchback(u[r], y[r], Horizon(h), "local_projection")
        except ValueError:
            continue
        model.append(reading.interval[0] <= _tau(h) <= reading.interval[1])
    assert np.mean(model) < 0.92


def test_the_design_interval_holds_its_level_given_the_noise():
    # one trending noise path, the coins redrawn: what the design's variance is over. The local
    # projection's interval for the channel covered 0.919 of 1000 runs, its estimate 0.42 of its
    # spread off the channel; the design's covered 0.960, at 0.95 +- 0.028
    rng = np.random.default_rng(7)
    reps, periods = 1000, 200
    shocks = np.broadcast_to(0.05 * np.arange(periods + BURN), (reps, periods + BURN))
    u = _coins(rng, (reps, periods + BURN))
    y = _plant(u, shocks)
    u, y = u[:, BURN:], y[:, BURN:]
    covered, estimates, _ = _coverage(u, y, 1, "design")
    assert covered == pytest.approx(LEVEL, abs=4 * math.sqrt(LEVEL * (1 - LEVEL) / reps))
    assert estimates.mean() == pytest.approx(B, abs=4 * estimates.std() / math.sqrt(reps))
    assert _coverage(u, y, 1, "model")[0] < 0.93


def _run(periods: int, seed: int = 9, share: float = 0.5, flip: float | None = None):
    rng = np.random.default_rng(seed)
    if flip is None:
        u = (rng.random(periods) < share).astype(np.float64)
    else:
        u = np.cumsum(rng.random(periods) < flip) % 2.0
    y = _plant(u, rng.standard_normal(periods))
    return u, y


def test_a_zones_level_does_not_move_the_design_reading():
    u, y = _run(300)
    alone = read_switchback(u, y, Horizon(2), "local_projection", se="design")
    pooled = read_switchback(
        np.vstack([u, u]), np.vstack([y, y + 1e3]), Horizon(2), "local_projection", se="design"
    )
    assert pooled.estimate == pytest.approx(alone.estimate, rel=1e-9, abs=0.0)
    assert pooled.se == pytest.approx(alone.se / math.sqrt(2.0), rel=1e-9, abs=0.0)
    assert pooled.zones == 2


def test_the_default_standard_error_is_the_models():
    u, y = _run(300)
    default = read_switchback(u, y, Horizon(2), "local_projection")
    assert read_switchback(u, y, Horizon(2), "local_projection", se="model") == default
    design = read_switchback(u, y, Horizon(2), "local_projection", se="design")
    assert design.estimate != default.estimate


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda u, y: read_switchback(u, y, CHANNEL, "local_projection", se="hac"), "'model'"),
        (
            lambda u, y: read_switchback(u, y, CHANNEL, "state_aware", se="design"),
            "not state_aware",
        ),
        (lambda u, y: read_switchback(u, y, CHANNEL, "iv", se="design"), "not iv"),
        (
            lambda u, y: read_switchback(
                u, y, Horizon(2), "block_dim", blocks=BlockDesign(2, 1), se="design"
            ),
            "not block_dim",
        ),
        (
            lambda u, y: read_switchback(u, y, STEADY_STATE, "local_projection", se="design"),
            "finite horizon",
        ),
        (
            lambda u, y: read_switchback(u[:12], y[:13], CHANNEL, "local_projection", se="design"),
            "more periods",
        ),
        (
            lambda u, y: read_switchback(
                u, 0.0 * y + 3.0, CHANNEL, "local_projection", se="design"
            ),
            "not positive",
        ),
    ],
)
def test_the_design_reading_refuses_what_it_cannot_read(call, match):
    u, y = _run(300)
    with pytest.raises(ValueError, match=match):
        call(u, y)


@pytest.mark.parametrize(
    ("lever", "match"),
    [({"flip": 0.1}, r"i\.i\.d\."), ({"share": 0.3}, "fair coin")],
)
def test_the_design_reading_needs_a_fair_coin_each_period(lever, match):
    u, y = _run(2000, **lever)
    with pytest.raises(ValueError, match=match):
        read_switchback(u, y, Horizon(3), "local_projection", se="design")


def test_a_design_reading_is_logged(caplog):
    u, y = _run(500)
    with caplog.at_level(logging.INFO, logger="chc.switchback"):
        reading = read_switchback(u, y, CHANNEL, "local_projection", se="design")
    (record,) = caplog.records
    assert record.chc_event == "switchback_reading"
    assert record.se_over == "design"
    assert record.se == reading.se
