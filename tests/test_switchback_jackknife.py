"""The half-panel jackknife of the switchback's plug-in and local projection: the estimator is
``2 tau_T - (tau_1 + tau_2) / 2`` over the run and its halves, it removes the readings' 1/T bias
(Dhaene and Jochmans, Review of Economic Studies 2015), and its standard error tracks its spread
on one zone and on several.

The plant is the working model ``x_{t+1} = a x_t + b u_t + eps_t``, fair coins, ``y = x``, started
in its stationary law."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from chc.switchback import (
    CHANNEL,
    STEADY_STATE,
    Horizon,
    _fit,
    _influence,
    _plant_fit,
    read_switchback,
)

Z = 1.959963984540054


def _plant(rng, zones: int, periods: int, a: float, b: float = 1.0):
    u = (rng.random((zones, periods)) < 0.5).astype(np.float64)
    x = np.empty((zones, periods + 1))
    x[:, 0] = b / 2.0 / (1.0 - a) + rng.standard_normal(zones) / math.sqrt(1.0 - a * a)
    eps = rng.standard_normal((zones, periods))
    for t in range(periods):
        x[:, t + 1] = a * x[:, t] + b * u[:, t] + eps[:, t]
    return u, x


def _tau(a: float, h: float, b: float = 1.0) -> float:
    return b / (1.0 - a) if math.isinf(h) else b * (1.0 - a**h) / (1.0 - a)


@pytest.mark.parametrize("periods", [200, 201])
@pytest.mark.parametrize(
    ("estimand", "analysis"),
    [(STEADY_STATE, "state_aware"), (Horizon(5), "state_aware"), (Horizon(3), "local_projection")],
)
def test_the_jackknife_is_twice_the_run_less_its_halves_each_read_on_its_own(
    periods, estimand, analysis
):
    u, y = _plant(np.random.default_rng(4), 3, periods, 0.8)
    half = (periods + 1) // 2
    full = read_switchback(u, y, estimand, analysis)
    first = read_switchback(u[:, :half], y[:, : half + 1], estimand, analysis)
    second = read_switchback(u[:, periods - half :], y[:, periods - half :], estimand, analysis)
    jackknife = read_switchback(u, y, estimand, analysis, jackknife=True)
    expected = 2.0 * full.estimate - 0.5 * (first.estimate + second.estimate)
    assert jackknife.estimate == pytest.approx(expected, rel=1e-12, abs=0.0)
    if math.isinf(estimand.periods):
        return
    assert jackknife.interval == pytest.approx(
        (jackknife.estimate - Z * jackknife.se, jackknife.estimate + Z * jackknife.se),
        rel=1e-12,
        abs=0.0,
    )


@pytest.mark.parametrize(("lags", "analysis"), [(0, "state_aware"), (2, "local_projection")])
def test_a_fits_influences_sum_to_its_covariance(lags, analysis):
    u, y = _plant(np.random.default_rng(5), 2, 300, 0.7)
    if analysis == "state_aware":
        target, regressors = y[:, 1:], np.stack([y[:, :-1], u])
    else:
        rows = u.shape[1] - lags
        target = sum(y[:, j : j + rows] for j in range(1, lags + 2))
        regressors = np.stack([u[:, :rows], y[:, :rows]])
    theta, cov = _fit(target, regressors, regressors, lags)
    found, influence = _influence(target, regressors, lags)
    total = np.einsum("izt,jzt->ij", influence, influence)
    for lag in range(1, lags + 1):
        cross = np.einsum("izt,jzt->ij", influence[..., lag:], influence[..., :-lag])
        total += cross + cross.T
    np.testing.assert_allclose(found, theta, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(total, cov, rtol=1e-10, atol=0.0)


def _study(estimand, analysis, a, periods, zones, reps, seed):
    rng = np.random.default_rng(seed)
    logging.getLogger("chc.switchback").setLevel(logging.ERROR)
    try:
        rows = []
        for _ in range(reps):
            u, y = _plant(rng, zones, periods, a)
            full = read_switchback(u, y, estimand, analysis)
            jackknife = read_switchback(u, y, estimand, analysis, jackknife=True)
            rows.append((full.estimate, jackknife.estimate, jackknife.se))
    finally:
        logging.getLogger("chc.switchback").setLevel(logging.NOTSET)
    return np.array(rows).T


@pytest.mark.parametrize("seed", [21, 22])
def test_the_jackknife_removes_the_plug_ins_bias_near_the_unit_root(seed):
    # the steady state at a = 0.9 over 400 periods: the plug-in reads about -0.33 of tau = 10 and
    # the jackknife about -0.02 (4000 runs), against a Monte Carlo error of about 0.06 here
    a, reps = 0.9, 1200
    full, jackknife, _ = _study(STEADY_STATE, "state_aware", a, 400, 1, reps, seed)
    _the_correction_lands_on_tau(full, jackknife, _tau(a, math.inf))


def _the_correction_lands_on_tau(full, jackknife, tau):
    reps = full.size
    correction = jackknife - full
    # the correction is real: its paired mean is many of its own standard errors from zero
    assert correction.mean() > 4.5 * correction.std(ddof=1) / math.sqrt(reps)
    assert abs(jackknife.mean() - tau) <= 4.5 * jackknife.std(ddof=1) / math.sqrt(reps)


@pytest.mark.parametrize("zones", [1, 8])
def test_the_jackknife_removes_the_local_projections_bias(zones):
    # tau_3 at a = 0.8 over 100 periods: -(H - 1) tau / (T - H + 1) = -0.050 to first order on one
    # zone (validation/local_projection_bias.mac); measured -0.060 and +0.005 over 4000 runs
    a, reps = 0.8, 1500
    full, jackknife, _ = _study(Horizon(3), "local_projection", a, 100, zones, reps, 30 + zones)
    _the_correction_lands_on_tau(full, jackknife, _tau(a, 3))


@pytest.mark.parametrize(
    ("estimand", "analysis", "a", "periods", "zones"),
    [
        (Horizon(3), "local_projection", 0.8, 100, 1),
        (Horizon(5), "state_aware", 0.9, 200, 1),
        (STEADY_STATE, "state_aware", 0.9, 200, 4),
    ],
)
def test_the_jackknifes_standard_error_is_its_spread(estimand, analysis, a, periods, zones):
    reps = 1200
    _, jackknife, se = _study(estimand, analysis, a, periods, zones, reps, 41)
    # the spread's own relative error over reps runs is about 1 / sqrt(2 (reps - 1)); the ratio
    # read over 2000-6000 runs was 0.96-1.05
    assert abs(se.mean() / jackknife.std(ddof=1) - 1.0) <= 0.06 + 4.5 / math.sqrt(2.0 * reps)


def test_the_jackknifes_interval_covers_the_local_projection():
    # covered 0.945 against the model interval's 0.912 over 4000 runs
    reps, a = 1500, 0.8
    _, jackknife, se = _study(Horizon(3), "local_projection", a, 100, 1, reps, 51)
    covered = float((np.abs(jackknife - _tau(a, 3)) <= Z * se).mean())
    assert abs(covered - 0.95) <= 4.5 * math.sqrt(0.95 * 0.05 / reps)


def test_the_steady_states_interval_is_fiellers_at_the_jackknifed_parameters():
    u, y = _plant(np.random.default_rng(12), 2, 301, 0.85)
    half = 151
    fits = [
        _plant_fit(u[:, start:stop], y[:, start : stop + 1], "state_aware")[0]
        for start, stop in ((0, 301), (0, half), (301 - half, 301))
    ]
    a, b = 2.0 * fits[0] - 0.5 * (fits[1] + fits[2])
    reading = read_switchback(u, y, STEADY_STATE, "state_aware", jackknife=True)
    low, high = reading.interval
    # Fieller's ends are where b - tau (1 - a) = 0 is rejected at the edge: there the ratio's
    # statistic equals z, with the jackknifed (a, b) at its centre
    assert low < b / (1.0 - a) < high
    assert low < reading.estimate < high


@pytest.mark.parametrize("seed", [61, 62])
def test_on_many_zones_the_jackknifed_steady_state_covers_and_the_reading_does_not(seed):
    # a = 0.9 over 200 periods on 16 zones, 1000 runs: Fieller's interval of the uncorrected
    # reading covered 0.821, since a_hat's O(1/T) bias does not shrink with zones, and the
    # jackknife's 0.934
    rng, reps, tau = np.random.default_rng(seed), 300, _tau(0.9, math.inf)
    logging.getLogger("chc.switchback").setLevel(logging.ERROR)
    try:
        hits = np.zeros((reps, 2), dtype=bool)
        for i in range(reps):
            u, y = _plant(rng, 16, 200, 0.9)
            for j, jackknife in enumerate((False, True)):
                low, high = read_switchback(
                    u, y, STEADY_STATE, "state_aware", jackknife=jackknife
                ).interval
                hits[i, j] = low <= tau <= high
    finally:
        logging.getLogger("chc.switchback").setLevel(logging.NOTSET)
    plain, jackknifed = hits.mean(axis=0)
    error = 4.5 * math.sqrt(0.95 * 0.05 / reps)
    assert jackknifed >= 0.934 - error
    assert plain <= 0.821 + 4.5 * math.sqrt(0.821 * 0.179 / reps)
    assert jackknifed > plain


def test_the_channel_at_an_iid_lever_moves_little():
    # no 1/T bias to remove there: the jackknife reads the run's estimate to its own noise
    u, y = _plant(np.random.default_rng(6), 1, 400, 0.8)
    full = read_switchback(u, y, CHANNEL, "state_aware")
    jackknife = read_switchback(u, y, CHANNEL, "state_aware", jackknife=True)
    assert abs(jackknife.estimate - full.estimate) <= 1.5 * full.se
    assert jackknife.se == pytest.approx(full.se, rel=0.1, abs=0.0)


def test_without_the_jackknife_the_reading_is_the_one_before():
    u, y = _plant(np.random.default_rng(7), 2, 300, 0.8)
    for estimand, analysis in ((STEADY_STATE, "state_aware"), (Horizon(3), "local_projection")):
        assert read_switchback(u, y, estimand, analysis, jackknife=False) == read_switchback(
            u, y, estimand, analysis
        )


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda u, y: read_switchback(u, y, CHANNEL, "iv", jackknife=True), "not iv"),
        (
            lambda u, y: read_switchback(
                u, y, Horizon(3), "local_projection", se="design", jackknife=True
            ),
            "se='design'",
        ),
        (
            lambda u, y: read_switchback(u[:, :6], y[:, :7], Horizon(3), "local_projection",
                                         jackknife=True),
            "fewer than 3 rows",
        ),
        (
            lambda u, y: read_switchback(
                np.where(np.arange(u.shape[1]) < 150, 0.0, u), y, CHANNEL, "state_aware",
                jackknife=True,
            ),
            "never switches in periods 0 to 149",
        ),
        (
            lambda u, y: read_switchback(u, y, STEADY_STATE, "local_projection", jackknife=True),
            "finite horizon",
        ),
    ],
)  # fmt: skip
def test_the_jackknife_refuses_what_it_cannot_read(call, match):
    u, y = _plant(np.random.default_rng(8), 1, 300, 0.8)
    with pytest.raises(ValueError, match=match):
        call(u, y)


def test_a_half_past_the_unit_root_has_no_steady_state():
    rng = np.random.default_rng(9)
    u = (rng.random((1, 200)) < 0.5).astype(np.float64)
    x = np.zeros((1, 201))
    for t in range(200):
        # explosive in the first half, stable and noisy in the second, which the run's fit reads
        if t < 100:
            x[0, t + 1] = 1.03 * x[0, t] + u[0, t] + 0.1 * rng.standard_normal()
        else:
            x[0, t + 1] = 0.5 * x[0, t] + u[0, t] + 20.0 * rng.standard_normal()
    assert read_switchback(u, x, STEADY_STATE, "state_aware").estimate > 0.0
    with pytest.raises(ValueError, match="over periods 0 to 99"):
        read_switchback(u, x, STEADY_STATE, "state_aware", jackknife=True)


def test_a_jackknife_reading_is_logged(caplog):
    u, y = _plant(np.random.default_rng(10), 1, 300, 0.8)
    with caplog.at_level(logging.INFO, logger="chc.switchback"):
        reading = read_switchback(u, y, Horizon(3), "local_projection", jackknife=True)
    (record,) = [r for r in caplog.records if r.chc_event == "switchback_reading"]
    assert record.estimate == reading.estimate
    assert record.jackknife is True
