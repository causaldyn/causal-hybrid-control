"""chc.independence: partial correlation removes a confounded path and stays calibrated on AR; the
generalised covariance measure holds its level on a logged panel whose units share noise."""

import numpy as np
import pytest

from chc.independence import gcm_test, partial_corr_test


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def test_partial_corr_test_removes_a_confounded_path() -> None:
    rng = np.random.default_rng(0)
    n = 500
    z = rng.standard_normal(n)
    x = z + 0.3 * rng.standard_normal(n)
    y = z + 0.3 * rng.standard_normal(n)  # x and y are linked ONLY through z
    _, p_marginal = partial_corr_test(x, y)  # marginally dependent (via z)
    _, p_conditional = partial_corr_test(x, y, z)  # independent given z
    assert float(p_marginal) < 0.01
    assert float(p_conditional) > 0.05


def test_partial_corr_test_is_calibrated_under_autocorrelation() -> None:
    phi, n, trials, alpha = 0.8, 250, 200, 0.05
    naive_rejections = mci_rejections = 0
    for trial in range(trials):
        rng = np.random.default_rng(1000 + trial)
        x = _ar1(rng, n, phi)
        y = _ar1(rng, n, phi)  # two INDEPENDENT AR(1) series: there is no true x-y link
        _, p_naive = partial_corr_test(x[1:], y[1:])
        conditioning = np.column_stack([x[:-1], y[:-1]])  # condition on the lagged parents (MCI)
        _, p_mci = partial_corr_test(x[1:], y[1:], conditioning)
        naive_rejections += float(p_naive) < alpha
        mci_rejections += float(p_mci) < alpha
    naive_rate, mci_rate = naive_rejections / trials, mci_rejections / trials
    assert naive_rate > 0.15  # the naive test over-rejects unrelated autocorrelated series
    assert mci_rate < 0.10  # conditioning on the lagged parents restores ~nominal calibration


def _logged_panel(
    rng: np.random.Generator, units: int, periods: int, *, reads: float = 0.0, shared: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(lever, columns, state, period)`` over ``units x periods`` rows: an autocorrelated state
    the lever moves, a season every unit sees, and a logger that reads the state, ``reads`` times
    the season, and noise of its own, a ``shared`` share of whose variance every unit draws at
    once. The columns are the season seen, and the state and the lever one period back."""
    burn = 20
    season = np.sin(2 * np.pi * np.arange(periods + burn) / 12) + 0.5 * rng.standard_normal(
        periods + burn
    )
    level = rng.standard_normal(units)
    state, lever = np.zeros(units), np.zeros(units)
    rows: list[tuple[np.ndarray, ...]] = []
    for t in range(periods + burn):
        before = state, lever
        state = 0.8 * state + 0.5 * lever + 0.3 * level + rng.standard_normal(units)
        seen = season[t] + rng.standard_normal(units)
        noise = np.sqrt(shared) * rng.standard_normal() + np.sqrt(1 - shared) * rng.standard_normal(
            units
        )
        lever = -0.6 * state + reads * seen + noise
        if t >= burn:
            rows.append((lever, np.column_stack([seen, *before]), state, np.full(units, t)))
    lever_rows, columns, states, periods_ = (
        np.concatenate(part) for part in zip(*rows, strict=True)
    )
    return lever_rows, columns, states, periods_


def test_gcm_test_holds_its_level_when_the_logger_shares_noise_across_units() -> None:
    """Grouped by period, the sign changes keep 5%; row by row, the shared noise rejects most."""
    by_period = by_row = 0
    trials = 200
    for trial in range(trials):
        rng = np.random.default_rng(trial)
        lever, columns, state, period = _logged_panel(rng, 40, 20, shared=0.5)
        by_period += (
            gcm_test(lever, columns, state, clusters=period, draws=499, seed=trial).p_value <= 0.05
        )
        by_row += gcm_test(lever, columns, state, draws=499, seed=trial).p_value <= 0.05
    assert 0.02 <= by_period / trials <= 0.09
    assert by_row / trials > 0.3


def test_gcm_test_catches_a_logger_that_also_reads_the_season() -> None:
    caught = 0
    for trial in range(40):
        lever, columns, state, period = _logged_panel(
            np.random.default_rng(trial), 40, 20, reads=0.1
        )
        test = gcm_test(lever, columns, state, clusters=period, draws=499, seed=trial)
        caught += test.p_value <= 0.05
        assert test.p_value >= 1 / 500  # the panel's own signs are one of the draws
    assert caught / 40 >= 0.8


def test_gcm_test_reads_partial_corr_tests_correlation_at_degree_one() -> None:
    rng = np.random.default_rng(3)
    z = rng.standard_normal((800, 2))
    x = z @ np.array([0.5, -0.3]) + rng.standard_normal(800)
    for link in (0.0, 0.2):
        y = z @ np.array([0.2, 0.4]) + link * x + rng.standard_normal(800)
        test = gcm_test(x, y, z, degree=1)
        rho, p_value = partial_corr_test(x, y, z)
        assert abs(float(test.partial_correlation[0, 0]) - rho) < 1e-12
        assert (test.p_value <= 0.05) == (p_value <= 0.05)


def test_gcm_test_leaves_out_a_column_its_conditioning_set_determines() -> None:
    """The last angle of a discretised pendulum is the current angle less dt times the velocity."""
    rng = np.random.default_rng(4)
    state = rng.standard_normal((500, 2))
    lever = -0.5 * state[:, 1] + rng.standard_normal(500)
    free = rng.standard_normal(500)
    determined = state[:, 0] - 0.05 * state[:, 1]
    both = gcm_test(lever, np.column_stack([free, determined]), state, degree=1)
    alone = gcm_test(lever, free, state, degree=1)
    assert np.isnan(both.partial_correlation[0, 1])
    assert np.isnan(both.detectable[0, 1])
    # one regression has two right-hand sides and the other one, and the BLAS kernel the CPU
    # dispatches to may round the two differently
    assert both.statistic == pytest.approx(alone.statistic, rel=1e-12)
    assert both.p_value == alone.p_value


@pytest.mark.parametrize("clusters", [1000, 10])
def test_gcm_test_detects_about_eight_in_ten_at_its_own_detectable_correlation(
    clusters: int,
) -> None:
    """Row by row, and over ten clusters, where the statistic's own normalisation asks for more
    dependence than reading it as a normal would."""
    rows = 1000
    labels = np.arange(rows) // (rows // clusters)
    nulls = (
        gcm_test(rng.standard_normal(rows), rng.standard_normal(rows), clusters=labels, draws=499)
        for rng in map(np.random.default_rng, range(200))
    )
    target = np.mean([null.detectable[0, 0] for null in nulls])
    caught = 0
    for trial in range(400):
        rng = np.random.default_rng(1000 + trial)
        x = rng.standard_normal(rows)
        y = target * x + np.sqrt(1 - target**2) * rng.standard_normal(rows)
        caught += gcm_test(x, y, clusters=labels, draws=499, seed=trial).p_value <= 0.05
    assert 0.7 <= caught / 400 <= 0.9


def test_gcm_test_reads_what_it_could_detect_off_the_clusters_spread_not_their_mean() -> None:
    """Rows correlated 0.3 in clusters of 50: each cluster's sum sits about 15 from zero, with a
    spread of about 7. Read about zero, that spread would more than double what the test calls
    detectable."""
    rng = np.random.default_rng(7)
    rows, rho = 2000, 0.3
    clusters = np.repeat(np.arange(rows // 50), 50)
    x, noise = rng.standard_normal(rows), rng.standard_normal(rows)
    null = gcm_test(x, noise, clusters=clusters).detectable[0, 0]
    found = gcm_test(x, rho * x + np.sqrt(1 - rho**2) * noise, clusters=clusters)
    assert found.p_value <= 0.05
    assert found.detectable[0, 0] < 1.5 * null


def test_gcm_test_refuses_fewer_rows_than_twice_its_regressions_terms() -> None:
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError, match="twice as many rows"):
        gcm_test(rng.standard_normal(25), rng.standard_normal(25), rng.standard_normal((25, 4)))
