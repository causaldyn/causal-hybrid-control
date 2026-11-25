"""The switchback's design standard error, ``read_switchback(..., se="design")``: Liang and Recht's
(Biometrika 2025) moment estimator of ``tau_H`` on a lever drawn by a fair coin each period, and
its variance over the coins with the noise held fixed.

Its mean and variance are checked over every design of a short run, against their eq. (2.5)
(``validation/switchback_design_variance.mac`` derives both), and its interval by simulation on a
plant whose noise persists or trends, where the regression's interval does not hold its level. The
moving baseline, ``baseline="moving"``, is the same estimator on the readings less the reading
``L`` periods back, and eq. (2.5) of the differenced response and noise
(``validation/switchback_moving_baseline.mac``): checked over every design, and its spread against
the derivation's on a persistent plant, a white-noise one and one whose noise alternates.
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


def _plant(u: np.ndarray, shocks: np.ndarray, a: float = A) -> np.ndarray:
    """``x_(t+1) = a x_t + B u_t + shocks_t`` from rest, over the last axis; one more period."""
    x = np.zeros((*u.shape[:-1], u.shape[-1] + 1))
    for t in range(u.shape[-1]):
        x[..., t + 1] = a * x[..., t] + B * u[..., t] + shocks[..., t]
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


def _after(y: np.ndarray, lag: int | None) -> np.ndarray:
    """The readings after the first less their baseline: the first reading, or with a ``lag`` the
    reading ``lag`` periods back once there is one, ``y_(t+1) - y_(max(t + 1 - lag, 0))``."""
    if lag is None:
        return y[1:] - y[0]
    return np.array([y[t + 1] - y[max(t + 1 - lag, 0)] for t in range(y.size - 1)])


def _differenced(v: np.ndarray, lag: int | None) -> np.ndarray:
    """``(1 - S^lag) v`` over the last axis, the moving baseline's filter; ``v`` without a lag."""
    if lag is None:
        return v
    out = v.copy()
    out[..., lag:] = v[..., lag:] - v[..., :-lag]
    return out


def _moment_estimate(u: np.ndarray, y: np.ndarray, h: int, lag: int | None = None) -> float:
    """The estimator, transcribed: ``sum_(s<H) 2/(T - s) sum_t (2 u_t - 1) (y_(t+1+s) - y_0)``,
    or less ``y_(t+1+s-L)`` in place of ``y_0`` with a ``lag``."""
    periods = u.size
    z, after = 2.0 * u - 1.0, _after(y, lag)
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


def _plug_in(u: np.ndarray, y: np.ndarray, h: int, lags: int, lag: int | None = None) -> float:
    """The plug-in of eq. (2.5), transcribed over full matrices: ``g`` read over ``2 lags`` lags,
    the noise as ``2 y' - T_g z`` over the first ``lags``, and ``q_s = T / (T - s)``."""
    periods = u.size
    z, after = 2.0 * u - 1.0, _after(y, lag)
    g = np.array(
        [2.0 / (periods - s) * (z[: periods - s] * after[s:]).sum() for s in range(2 * lags)]
    )
    noise = 2.0 * after - _toeplitz(g[:lags], periods) @ z
    q = periods / (periods - np.arange(h, dtype=np.float64))
    return _dense_variance(q, g[:lags], g, noise) / periods**2


@pytest.mark.parametrize(("h", "lag"), [(1, None), (2, None), (3, None), (1, 1), (1, 2), (2, 2)])
def test_the_estimate_is_unbiased_and_its_variance_is_eq_2_5_over_every_design(monkeypatch, h, lag):
    # the plug-in reads ceil(ln 10) = 3 lags here, so that a run of 10 can be read at all; under a
    # moving baseline it reads L more, and max(H, ceil(0.4 ln 10)) + L = H + L
    per_log = 1.0 if lag is None else 0.4
    monkeypatch.setattr(switchback, "_DESIGN_LAGS", per_log)
    rng = np.random.default_rng(1)
    periods = 10
    g = rng.normal(size=periods) * 0.7 ** np.arange(periods)
    # a trend and a level, fixed: the design holds them and redraws only the coins
    noise = np.concatenate([[4.0], 3.0 + np.cumsum(rng.normal(size=periods))])
    estimates = []
    for bits in itertools.product((0.0, 1.0), repeat=periods):
        u = np.array(bits)
        y = noise + np.concatenate([[0.0], np.convolve(u, g)[:periods]])
        estimate = _moment_estimate(u, y, h, lag)
        if 0.0 < u.sum() < periods:
            reading = read_switchback(
                u,
                y,
                Horizon(h),
                "local_projection",
                se="design",
                baseline="first" if lag is None else "moving",
                baseline_lag=lag,
            )
            assert reading.estimate == pytest.approx(estimate, rel=1e-12, abs=1e-12)
            # at H = 3 the plug-in's tr(M M) reads g beyond the 3 lags its other terms read
            lags = max(h, math.ceil(per_log * math.log(periods))) + (lag or 0)
            assert reading.se**2 == pytest.approx(_plug_in(u, y, h, lags, lag), rel=1e-10, abs=0.0)
        estimates.append(estimate)
    estimates = np.array(estimates)
    assert estimates.mean() == pytest.approx(g[:h].sum(), rel=0.0, abs=1e-12)
    # T^2 var = |M|_F^2 + tr(M M) - 2 sum M_ii^2 + |T_q' h|^2, h = T_g 1 + 2 (e - y_0); the moving
    # baseline's is the same of g - S^L g and (1 - S^L) h
    q = periods / (periods - np.arange(h, dtype=np.float64))
    level = np.convolve(np.ones(periods), g)[:periods] + 2.0 * (noise[1:] - noise[0])
    level, g = _differenced(level, lag), _differenced(g, lag)
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
    # AR(0.8) shocks over 60 periods: the design's interval for tau_3 covered 0.969, at 0.95 +-
    # 0.028 four standard errors, and pays in width: 23.5 against 6.2. The local projection's
    # covered 0.903 of the 995 runs it read, short through the run's length, not the noise's
    # persistence: with i.i.d. shocks it covered 0.880 over 60 periods
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


def _short_run_warnings(caplog, u, y, se: str = "model") -> list:
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="chc.switchback"):
        read_switchback(u, y, Horizon(3), "local_projection", se=se)
    return [r for r in caplog.records if getattr(r, "chc_event", None) == "switchback_short_run"]


def test_a_model_interval_on_a_short_run_warns_and_points_to_the_design(caplog):
    """The regression fits its intercept and ``y_t`` on the periods its sums read, which biases
    ``tau_H`` by ``-(H - 1) tau_H / (T - H + 1)`` to first order on one zone and reads its variance
    low: its 95% interval for ``tau_3`` covered 0.929 over 200 periods and 0.940 over 400 (200 000
    runs each), on i.i.d. noise."""
    u, y = _run(399)
    (record,) = _short_run_warnings(caplog, u, y)
    estimate = read_switchback(u, y, Horizon(3), "local_projection").estimate
    assert record.periods == 399
    assert record.bias_bound == pytest.approx(2 * abs(estimate) / 397, rel=1e-12, abs=0.0)
    assert "se='design'" in record.getMessage()
    assert not _short_run_warnings(caplog, u, y, se="design")
    u, y = _run(400)
    assert not _short_run_warnings(caplog, u, y)


def test_many_zones_warn_of_the_bias_a_long_run_still_carries(caplog):
    """The bias does not shrink with the zones, and the pooled standard error does: on 64 zones it
    weighs 0.42 standard errors over 600 periods, and the interval covered 0.926 over 300 periods
    and 0.944 over 600 (4000 runs each)."""
    rng = np.random.default_rng(64)
    u = _coins(rng, (64, 600 + BURN))
    y = _plant(u, rng.standard_normal(u.shape))
    u, y = u[:, BURN:], y[:, BURN:]
    (record,) = _short_run_warnings(caplog, u, y)
    assert record.periods == 600
    assert record.bias_in_se > 0.3
    assert not _short_run_warnings(caplog, u[:1], y[:1])


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


def _design_readings(u, y, h: int, tau: float, **baseline):
    """Each run's design reading of ``tau_H``: the estimates, the standard errors and the share of
    the intervals that cover ``tau``."""
    readings = [
        read_switchback(u[r], y[r], Horizon(h), "local_projection", se="design", **baseline)
        for r in range(len(u))
    ]
    return (
        np.array([x.estimate for x in readings]),
        np.array([x.se for x in readings]),
        float(np.mean([x.interval[0] <= tau <= x.interval[1] for x in readings])),
    )


def _spread(u, y, h: int, lag: int | None) -> float:
    """The variance over the runs that eq. (2.5) predicts on ``_plant``: the estimate's variance
    over the coins on each run's own noise path, averaged over the paths. Its mean is ``tau_H`` on
    every path, so that is its whole variance. ``lag`` reads it under the moving baseline."""
    reps, periods = u.shape
    g = B * A ** np.arange(periods)
    effect = np.stack([np.convolve(row, g)[:periods] for row in u])
    level = np.convolve(np.ones(periods), g)[:periods] + 2.0 * (y[:, 1:] - effect - y[:, :1])
    level, g = _differenced(level, lag), _differenced(g, lag)
    q = periods / (periods - np.arange(h, dtype=np.float64))
    g = np.broadcast_to(g, (reps, periods))
    return float(switchback._design_variance(q, g, g, level).mean()) / periods**2


def test_a_moving_baseline_narrows_the_design_reading_on_a_persistent_plant():
    """``x' = 0.8 x + 2 u + eps``, i.i.d. ``eps``: an outcome that persists. Read against the
    reading ``H`` periods back, ``tau_3``'s estimate over 60 periods spread 0.547 times as far as
    from the first reading, against 0.543 from eq. (2.5) of each one's own response and noise, and
    its interval covered 0.946. The plug-in reads ``L`` lags more than the first reading's, each
    with its error, and its standard error was 1.10 times the spread."""
    rng = np.random.default_rng(60)
    reps, periods, h = 1000, 60, 3
    u = _coins(rng, (reps, periods + BURN))
    y = _plant(u, rng.standard_normal(u.shape))
    u, y = u[:, BURN:], y[:, BURN:]
    first, _, _ = _design_readings(u, y, h, _tau(h))
    moving, se, covered = _design_readings(u, y, h, _tau(h), baseline="moving")
    assert covered == pytest.approx(LEVEL, abs=4 * math.sqrt(LEVEL * (1 - LEVEL) / reps))
    assert moving.mean() == pytest.approx(_tau(h), abs=4 * moving.std() / math.sqrt(reps))
    assert 1.0 <= se.mean() / moving.std() < 1.25
    for estimates, lag in ((first, None), (moving, h)):
        predicted = math.sqrt(_spread(u, y, h, lag))
        assert estimates.std() == pytest.approx(predicted, rel=4 / math.sqrt(2 * reps), abs=0.0)
    assert moving.std() / first.std() < 0.6


@pytest.mark.parametrize(
    ("rho", "first_sd", "moving_sd"), [(0.0, 0.4472136, 0.4472136), (-0.5, 0.4954110, 0.5773503)]
)
def test_a_moving_baseline_ties_on_white_noise_and_widens_on_a_noise_that_alternates(
    rho, first_sd, moving_sd
):
    """``x' = 2 u + eps``, ``eps`` AR(``rho``) of unit innovations, the channel over 60 periods
    at ``L = 1``. ``validation/switchback_moving_baseline.mac`` STEP 6 gives the spreads:
    ``T^2 var = T b^2 + 8 sum_j (gam_0 - gam_(j+1))`` from the first reading and
    ``T b^2 + 8 T (gam_0 - gam_1)`` from the moving baseline. On white noise they tie; where the
    noise alternates, ``gam_1 < 0``, the moving baseline spreads 1.165 times as far."""
    rng = np.random.default_rng(1)
    reps, periods = 1000, 60
    u = _coins(rng, (reps, periods + BURN))
    y = _plant(u, _persistent(rng, u.shape, rho), a=0.0)
    u, y = u[:, BURN:], y[:, BURN:]
    for sd, baseline in ((first_sd, "first"), (moving_sd, "moving")):
        estimates, _, _ = _design_readings(u, y, 1, B, baseline=baseline)
        assert estimates.mean() == pytest.approx(B, abs=4 * sd / math.sqrt(reps))
        assert estimates.std() == pytest.approx(sd, rel=4 / math.sqrt(2 * reps), abs=0.0)


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
    first = read_switchback(u, y, Horizon(2), "local_projection", se="design", baseline="first")
    assert first == design
    moving = read_switchback(u, y, Horizon(2), "local_projection", se="design", baseline="moving")
    assert moving.estimate != design.estimate


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
    ("options", "match"),
    [
        ({"baseline": "none"}, "'first' or 'moving'"),
        ({"se": "model", "baseline": "moving"}, "read by the design standard error"),
        ({"baseline_lag": 3}, "the moving baseline's"),
        ({"baseline": "moving", "baseline_lag": 2}, "at least the horizon's 3"),
        ({"baseline": "moving", "baseline_lag": 3.0}, "whole number"),
        ({"baseline": "moving", "baseline_lag": True}, "whole number"),
        # ceil(3 ln 300) = 18 lags and L more, twice over, must stay below the run's 300 periods
        ({"baseline": "moving", "baseline_lag": 132}, r"reads 300 lags .*\(264 for the moving"),
        ({"baseline": "moving", "baseline_lag": 300}, "more periods"),
    ],
)
def test_a_moving_baseline_refuses_a_lag_the_design_cannot_read(options, match):
    u, y = _run(300)
    with pytest.raises(ValueError, match=match):
        read_switchback(u, y, Horizon(3), "local_projection", **({"se": "design"} | options))


def test_a_moving_baseline_reads_up_to_the_longest_lag_its_run_holds():
    u, y = _run(300)
    longest = read_switchback(
        u, y, Horizon(3), "local_projection", se="design", baseline="moving", baseline_lag=131
    )
    assert longest.estimate == pytest.approx(_moment_estimate(u, y, 3, 131), rel=1e-12, abs=0.0)
    with pytest.raises(ValueError, match="finite horizon"):
        read_switchback(u, y, STEADY_STATE, "local_projection", se="design", baseline="moving")


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
    assert record.baseline == "first"
    assert record.se == reading.se
