"""The deployed loop's conformal radius, watched and adapted: :class:`chc.gate.MissRateAlarm` on
its closed forms and by simulation on an AR(1) plant, and :class:`chc.gate.OnlineRadius` against
every adversary on short sequences and a few on long ones.

The Beta law and the mixture's closed form are checked symbolically in
``validation/miss_rate_alarm.mac``; here they are checked against scipy's binomial law and
quadrature.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate, special, stats

from chc.gate import MissRateAlarm, OnlineRadius

# --- MissRateAlarm -------------------------------------------------------------------------------


def _calibration(n: int, seed: int = 0) -> np.ndarray:
    """``n + 1`` residuals of i.i.d. noise, so ``n`` innovations at persistence 0."""
    return np.random.default_rng(seed).standard_normal(n + 1)


def _mixture(misses: int, hits: int, bound: float) -> float:
    """The likelihood ratios of a miss rate ``q`` against ``bound``, mixed over ``q`` uniform on
    ``[bound, 1]``, by quadrature."""

    def ratio(q: float) -> float:
        return (q / bound) ** misses * ((1.0 - q) / (1.0 - bound)) ** hits

    return integrate.quad(ratio, bound, 1.0, epsabs=0.0, epsrel=1e-12, limit=200)[0] / (1.0 - bound)


def _fed(alarm: MissRateAlarm, misses: int, hits: int) -> MissRateAlarm:
    """``alarm`` after a seed residual, ``misses`` innovations outside its radius and ``hits``
    inside, at persistence 0."""
    big = 10.0 * alarm.radius
    alarm.update(np.r_[0.0, np.full(misses, big), np.zeros(hits)])
    return alarm


def test_the_radius_is_split_conformals_on_whitened_innovations() -> None:
    residual = _calibration(200, seed=1)
    alarm = MissRateAlarm(residual, persistence=0.5, alpha=0.2, delta=0.2)
    scores = np.sort(np.abs(residual[1:] - 0.5 * residual[:-1]))
    k = math.ceil(201 * 0.8)  # 161
    assert alarm.radius == scores[k - 1]
    # two states: the innovation's Euclidean norm
    pair = np.random.default_rng(2).standard_normal((101, 2))
    alarm = MissRateAlarm(pair, persistence=-0.3, alpha=0.1, delta=0.1)
    norms = np.sort(np.linalg.norm(pair[1:] + 0.3 * pair[:-1], axis=1))
    assert alarm.radius == norms[math.ceil(101 * 0.9) - 1]


@pytest.mark.parametrize(
    ("n", "alpha", "delta"), [(300, 0.1, 0.1), (19, 0.1, 0.2), (1000, 0.05, 0.05)]
)
def test_the_bound_is_exceeded_by_the_miss_rate_with_probability_half_delta(
    n: int, alpha: float, delta: float
) -> None:
    """The miss rate exceeds ``bound`` when at most ``n - k`` of the ``n`` scores lie above its
    ``1 - bound`` quantile: a binomial tail, read here through scipy's binomial law."""
    alarm = MissRateAlarm(_calibration(n), persistence=0.0, alpha=alpha, delta=delta)
    k = math.ceil((n + 1) * (1 - alpha))
    assert stats.binom.cdf(n - k, n, alarm.bound) == pytest.approx(delta / 2, rel=1e-9, abs=0.0)
    assert alarm.bound > (n + 1 - k) / (n + 1)  # above the mean, which is at most alpha


def test_the_bound_matches_maxima_at_the_monte_carlos_sizes() -> None:
    """``validation/miss_rate_alarm.mac`` STEP 4, in bfloat by bisection on the binomial tail."""
    alarm = MissRateAlarm(_calibration(300), persistence=0.0, alpha=0.1, delta=0.1)
    assert alarm.bound == pytest.approx(0.129460694055819, rel=1e-13, abs=0.0)


@pytest.mark.parametrize(("misses", "hits"), [(0, 0), (1, 0), (2, 0), (3, 17), (12, 40), (1, 60)])
def test_the_statistic_is_the_uniform_mixture_of_likelihood_ratios(misses: int, hits: int) -> None:
    alarm = _fed(
        MissRateAlarm(_calibration(300), persistence=0.0, alpha=0.1, delta=1e-12), misses, hits
    )
    assert (alarm.misses, alarm.steps) == (misses, misses + hits)
    expected = _mixture(misses, hits, alarm.bound)
    assert alarm.statistic == pytest.approx(expected, rel=1e-9, abs=0.0)


def test_the_statistic_is_a_martingale_at_the_bound_and_falls_below_it() -> None:
    """Over the binomial law of the misses after ``t`` steps, the statistic's mean is exactly 1 at
    a miss rate equal to ``bound`` (``validation/miss_rate_alarm.mac`` STEP 6) and below 1 under
    it: what Ville's inequality needs."""
    t = 9
    probe = MissRateAlarm(_calibration(300), persistence=0.0, alpha=0.1, delta=1e-12)
    bound = probe.bound
    values = [
        _fed(
            MissRateAlarm(_calibration(300), persistence=0.0, alpha=0.1, delta=1e-12), s, t - s
        ).statistic
        for s in range(t + 1)
    ]
    at_bound = sum(stats.binom.pmf(s, t, bound) * v for s, v in enumerate(values))
    below = sum(stats.binom.pmf(s, t, 0.6 * bound) * v for s, v in enumerate(values))
    assert at_bound == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert below < 0.9


def test_the_alarm_sounds_at_two_over_delta_and_starts_again() -> None:
    """At ``n = 200``, ``alpha = delta = 0.2`` the statistic after 1, 2 and 3 straight misses is
    2.62, 7.75 and 24.9: the alarm sounds at the third, past ``2 / delta = 10``, and not at the
    second, past ``1 / delta`` alone."""
    alarm = MissRateAlarm(_calibration(200), persistence=0.0, alpha=0.2, delta=0.2)
    big = 10.0 * alarm.radius
    assert alarm.statistic == 1.0
    assert not alarm.update([0.0])  # a seed only
    assert alarm.steps == 0
    assert not alarm.update([alarm.radius])  # on the radius: inside
    assert (alarm.misses, alarm.steps) == (0, 1)
    alarm = MissRateAlarm(_calibration(200), persistence=0.0, alpha=0.2, delta=0.2)
    assert not alarm.update([0.0, big, big])
    assert alarm.statistic == pytest.approx(7.75, abs=0.005)
    assert alarm.update([big, big, big])
    assert (alarm.misses, alarm.steps, alarm.statistic) == (0, 0, 1.0)  # the rest was dropped
    assert not alarm.update([big])  # the first residual after an alarm only whitens the next
    assert (alarm.misses, alarm.steps) == (0, 0)
    assert not alarm.update([big])
    assert (alarm.misses, alarm.steps) == (1, 1)


@settings(max_examples=60, deadline=None)
@given(rows=st.integers(0, 40), cut=st.floats(0.0, 1.0), seed=st.integers(0, 2**31 - 1))
def test_the_alarm_reads_the_same_whatever_the_batches(rows: int, cut: float, seed: int) -> None:
    """The whitening carries the last residual from one call to the next."""
    deployed = 1.5 * np.random.default_rng(seed).standard_normal((rows, 2))
    whole = MissRateAlarm(
        _calibration(99).reshape(-1, 1).repeat(2, 1), persistence=0.7, alpha=0.1, delta=1e-9
    )
    split = MissRateAlarm(
        _calibration(99).reshape(-1, 1).repeat(2, 1), persistence=0.7, alpha=0.1, delta=1e-9
    )
    whole.update(deployed)
    at = round(cut * rows)
    split.update(deployed[:at])
    split.update(deployed[at:])
    assert (split.misses, split.steps) == (whole.misses, whole.steps)


RHO, SIGMA = 0.8, 0.3


def _ar1(rng: np.random.Generator, paths: int, steps: int, scale: float = 1.0) -> np.ndarray:
    """Residuals of an exact model on a plant whose noise is AR(1) at ``RHO``: with the model
    exact, a residual is the noise, whatever the policy that closed the loop."""
    eta = np.empty((paths, steps))
    previous = rng.standard_normal(paths) * SIGMA / math.sqrt(1 - RHO**2)
    shocks = rng.standard_normal((steps, paths))
    for t in range(steps):
        previous = RHO * previous + scale * SIGMA * shocks[t]
        eta[:, t] = previous
    return eta


def _lag1(residual: np.ndarray) -> float:
    centred = residual - residual.mean()
    return float(centred[1:] @ centred[:-1] / (centred @ centred))


def _alarmed(calibration: np.ndarray, deployed: np.ndarray, persistence: float) -> bool:
    alarm = MissRateAlarm(calibration, persistence=persistence, alpha=0.1, delta=0.1)
    return any(alarm.update(deployed[t : t + 100]) for t in range(0, deployed.size, 100))


def test_on_a_correct_ar1_plant_whitened_innovations_alarm_at_most_delta_and_raw_ones_do_not(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """1000 paths, 300 calibration innovations, 1000 deployed steps, ``alpha = delta = 0.1``. With
    the noise's own persistence the alarm sounded on 0.016 of the paths, at most ``delta`` within
    four standard errors; with the lag-1 correlation fitted on the calibration log, on 0.020.
    Unwhitened, misses cluster, and it sounded on 0.209, twice ``delta``."""
    caplog.set_level("ERROR", logger="chc.gate")
    rng = np.random.default_rng(31)
    paths, delta = 1000, 0.1
    calibration, deployed = _ar1(rng, paths, 301), _ar1(rng, paths, 1000)
    whitened = np.mean([_alarmed(c, d, RHO) for c, d in zip(calibration, deployed, strict=True)])
    fitted = np.mean([_alarmed(c, d, _lag1(c)) for c, d in zip(calibration, deployed, strict=True)])
    raw = np.mean([_alarmed(c, d, 0.0) for c, d in zip(calibration, deployed, strict=True)])
    se = math.sqrt(delta * (1 - delta) / paths)
    assert whitened <= delta + 4 * se
    assert fitted <= delta + 4 * se
    assert raw > 1.5 * delta


def test_a_noise_that_grew_is_caught(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("ERROR", logger="chc.gate")
    rng = np.random.default_rng(32)
    calibration, deployed = _ar1(rng, 200, 301), _ar1(rng, 200, 1000, scale=1.5)
    caught = [_alarmed(c, d, RHO) for c, d in zip(calibration, deployed, strict=True)]
    assert np.mean(caught) >= 0.99


def test_an_alarm_logs_its_counts(caplog: pytest.LogCaptureFixture) -> None:
    alarm = MissRateAlarm(_calibration(200), persistence=0.0, alpha=0.2, delta=0.2)
    with caplog.at_level("WARNING", logger="chc.gate"):
        assert alarm.update(np.r_[0.0, np.full(5, 10.0 * alarm.radius)])
    (record,) = caplog.records
    assert record.chc_event == "miss_rate_alarm"  # type: ignore[attr-defined]
    assert (record.misses, record.steps) == (3, 3)  # type: ignore[attr-defined]
    assert "2 later residuals were dropped" in record.getMessage()


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"alpha": 0.0}, r"alpha must lie in \(0, 1\)"),
        ({"delta": 1.0}, r"delta must lie in \(0, 1\)"),
        ({"persistence": 1.0}, r"persistence must lie in \(-1, 1\)"),
        ({"persistence": math.nan}, r"persistence must lie in \(-1, 1\)"),
        ({"calibration": np.full(30, np.nan)}, "calibration is not finite"),
        ({"calibration": np.zeros((3, 3, 3))}, r"calibration must have shape \(steps,\)"),
        (
            {"calibration": np.zeros(9)},
            "8 calibration innovations cannot give a radius covering 0.9",
        ),
    ],
)
def test_an_alarm_outside_the_contract_is_refused(arguments: dict[str, object], match: str) -> None:
    given = {
        "calibration": np.zeros(10),
        "persistence": 0.0,
        "alpha": 0.1,
        "delta": 0.1,
    } | arguments
    with pytest.raises(ValueError, match=match):
        MissRateAlarm(**given)  # type: ignore[arg-type]


def test_residuals_outside_the_contract_are_refused() -> None:
    alarm = MissRateAlarm(np.zeros((10, 2)), persistence=0.0, alpha=0.1, delta=0.1)
    with pytest.raises(ValueError, match=r"residuals must have shape \(steps,\) or \(steps, 2\)"):
        alarm.update(np.zeros(3))
    with pytest.raises(ValueError, match="residuals is not finite"):
        alarm.update(np.full((2, 2), np.inf))
    assert not alarm.update(np.zeros((0, 2)))


# --- OnlineRadius --------------------------------------------------------------------------------


def test_one_horizon_is_the_published_update() -> None:
    """``q_(t+1) = q_t + eta_t (1{s_t > q_t} - alpha)``, ``eta_t = step t^(-decay)``: Angelopoulos,
    Barber and Bates's update (4)."""
    rng = np.random.default_rng(41)
    online = OnlineRadius([0.7], alpha=0.1, bound=3.0, step=0.4)
    q = 0.7
    for t, s in enumerate(rng.uniform(0.0, 3.0, 200), start=1):
        assert online.update([s])[0] == (s > q)
        q += 0.4 * t**-0.6 * (float(s > q) - 0.1)
        assert online.radius[0] == pytest.approx(q, rel=1e-13, abs=1e-15)
    assert online.scored[0] == 200


def test_a_prediction_is_scored_against_the_radius_it_was_issued_with() -> None:
    online = OnlineRadius([0.5, 0.5], alpha=0.5, bound=2.0, step=1.0, decay=0.0)
    assert online.update([0.6, np.nan]).tolist() == [True, False]
    assert online.radius.tolist() == [1.0, 0.5]
    # horizon 2 scores the prediction issued a period ago, at 0.5, not the current radius
    assert online.update([0.6, 0.6]).tolist() == [False, True]
    assert online.radius.tolist() == [0.5, 1.0]
    # the one issued at the second period, with horizon 2's radius still 0.5
    assert online.update([np.nan, 0.7]).tolist() == [False, True]
    assert online.radius.tolist() == [0.5, 1.5]
    assert online.scored.tolist() == [2, 2]
    assert online.miss_rate.tolist() == [0.5, 1.0]


def _deviations(online: OnlineRadius, choices: tuple[float, ...]) -> np.ndarray:
    """The largest ``|miss rate - alpha| / miss_bound`` of each horizon over a run of scores, the
    same choice for every horizon that may be scored."""
    horizons = online.radius.size
    worst = np.zeros(horizons)
    for period, choice in enumerate(choices, start=1):
        scores = np.where(np.arange(1, horizons + 1) <= period, choice, np.nan)
        online.update(scores)
        ratio = np.abs(online.miss_rate - online.alpha) / online.miss_bound
        worst = np.maximum(worst, np.nan_to_num(ratio))
    return worst


@pytest.mark.parametrize(("step", "decay"), [(1.0, 0.0), (0.3, 0.0), (1.0, 0.6), (0.3, 0.6)])
def test_every_adversary_keeps_each_horizon_within_its_bound(step: float, decay: float) -> None:
    """Whether a score misses depends only on whether it exceeds the radius it was issued with, so
    the scores 0 and ``bound`` reach every sequence of misses an adversary can force: all 2^12 of
    them, on three horizons at once. A delayed horizon needs its ``h``: at a constant step of 1,
    horizon 3 comes within 0.9 of ``(bound + 3 step) / (step n)``, past the
    ``(bound + step) / (step n)`` of a horizon scored at once."""
    worst = np.zeros(3)
    for choices in itertools.product((0.0, 1.0), repeat=12):
        online = OnlineRadius([0.0, 0.0, 0.0], alpha=0.1, bound=1.0, step=step, decay=decay)
        worst = np.maximum(worst, _deviations(online, choices))
    assert np.all(worst <= 1.0)
    if (step, decay) == (1.0, 0.0):
        assert worst[2] > (1.0 + step) / (1.0 + 3 * step)


@pytest.mark.parametrize("decay", [0.0, 0.6])
def test_long_adversarial_sequences_keep_each_horizon_within_its_bound(decay: float) -> None:
    rng = np.random.default_rng(42)
    period = np.arange(2000)
    sequences = {
        "uniform": rng.uniform(0.0, 2.0, period.size),
        "bound": np.full(period.size, 2.0),
        "switching": np.where((period // 50) % 2 == 1, 2.0, 0.0),
        "heavy": np.minimum(2.0, np.abs(rng.standard_cauchy(period.size))),
    }
    for scores in sequences.values():
        online = OnlineRadius([1.0] * 4, alpha=0.1, bound=2.0, step=0.5, decay=decay)
        assert np.all(_deviations(online, tuple(scores)) <= 1.0)


def test_a_radius_that_chases_its_own_scores_keeps_its_bound() -> None:
    """An adversary who scores each prediction just above or just below the radius it was issued
    with, whichever moves the miss rate further from ``alpha``."""
    online = OnlineRadius([0.5, 0.5], alpha=0.2, bound=1.0, step=0.5, decay=0.6)
    issued: list[np.ndarray] = []
    worst = np.zeros(2)
    for period in range(1, 1501):
        issued.append(online.radius.copy())
        scores = np.full(2, np.nan)
        for h in (1, 2):
            if period >= h:
                q = issued[period - h][h - 1]
                rate = online.miss_rate[h - 1]
                miss = math.isnan(rate) or rate >= online.alpha
                scores[h - 1] = min(1.0, q + 1e-9) if miss and q < 1.0 else max(0.0, min(1.0, q))
        online.update(scores)
        worst = np.maximum(worst, np.nan_to_num(np.abs(online.miss_rate - 0.2) / online.miss_bound))
    assert np.all(worst <= 1.0)


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"initial": []}, "one radius per horizon, got none"),
        ({"initial": [0.5, 2.5]}, r"initial radii must lie in \[0, bound = 2.0\]"),
        ({"initial": [np.nan]}, "initial is not finite"),
        ({"alpha": 1.0}, r"alpha must lie in \(0, 1\)"),
        ({"bound": 0.0}, "bound must be positive and finite"),
        ({"step": math.inf}, "step must be positive and finite"),
        ({"decay": 1.0}, r"decay must lie in \[0, 1\)"),
    ],
)
def test_an_online_radius_outside_the_contract_is_refused(
    arguments: dict[str, object], match: str
) -> None:
    given = {"initial": [0.5], "alpha": 0.1, "bound": 2.0, "step": 0.5} | arguments
    with pytest.raises(ValueError, match=match):
        OnlineRadius(**given)  # type: ignore[arg-type]


def test_scores_outside_the_contract_are_refused() -> None:
    online = OnlineRadius([0.5, 0.5], alpha=0.1, bound=2.0, step=0.5)
    with pytest.raises(ValueError, match=r"one entry per horizon, shape \(2,\)"):
        online.update([0.1])
    with pytest.raises(ValueError, match=r"scores must lie in \[0, bound = 2.0\]"):
        online.update([2.5, np.nan])
    with pytest.raises(ValueError, match="horizon 2 scored at period 1, before any prediction"):
        online.update([0.1, 0.1])
    assert online.scored.tolist() == [0, 0]  # a refused period closes nothing
    online.update([0.1, np.nan])
    online.update([0.1, 0.1])
    assert online.scored.tolist() == [2, 1]
    assert online.miss_bound[1] == pytest.approx((2.0 + 2 * 0.5) / 0.5, rel=1e-15, abs=0.0)
    assert math.isinf(OnlineRadius([0.5], alpha=0.1, bound=2.0, step=0.5).miss_bound[0])


def test_the_beta_law_of_the_miss_rate_holds_by_simulation() -> None:
    """Over calibrations, the miss rate of the radius exceeds ``bound`` on ``delta / 2`` of them."""
    rng = np.random.default_rng(43)
    n, alpha, delta = 99, 0.1, 0.2
    exceeded = 0
    draws = 4000
    for _ in range(draws):
        alarm = MissRateAlarm(rng.standard_normal(n + 1), persistence=0.0, alpha=alpha, delta=delta)
        miss_rate = 2.0 * special.ndtr(-alarm.radius)  # |N(0, 1)| above the radius
        exceeded += miss_rate > alarm.bound
    se = math.sqrt(delta / 2 * (1 - delta / 2) / draws)
    assert abs(exceeded / draws - delta / 2) <= 4 * se
