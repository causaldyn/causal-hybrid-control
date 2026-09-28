"""The switchback design: its closed forms against the lab's verified ones, its plans against the
lab's reference planner, and every planned standard error and power against simulation.

The plant is the working model, ``x_{t+1} = a x_t + b u_t + eps_t`` with ``y_t = x_t (+ eta_t)``.
The simulations here are the lab's, vectorised over replications and written independently of the
module: they read each plan's own design with the analysis it names.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from chc.switchback import (
    CHANNEL,
    STEADY_STATE,
    BlockDesign,
    Horizon,
    MarkovDesign,
    PersistencePrior,
    SwitchbackPlan,
    TargetMDE,
    _a1_markov,
    _dim_bias,
    _ds,
    _fieller,
    _information,
    _s,
    _v_local_projection,
    _v_state_aware,
    design_switchback,
    read_switchback,
)

A, SIGMA, B = 0.8, 1.0, 2.0
PRIOR = PersistencePrior(A, A, SIGMA, B)


def _tau(a: float, h: float) -> float:
    return B / (1.0 - a) if math.isinf(h) else B * (1.0 - a**h) / (1.0 - a)


# --- the lab's simulation, vectorised over replications -----------------------------------------


def _markov_signs(rng, reps: int, n: int, flip: float) -> np.ndarray:
    v = np.empty((reps, n))
    v[:, 0] = rng.choice([-1.0, 1.0], size=reps)
    flips = rng.random((reps, n)) < flip
    for t in range(1, n):
        v[:, t] = np.where(flips[:, t], -v[:, t - 1], v[:, t - 1])
    return v


def _block_signs(rng, reps: int, n: int, length: int) -> np.ndarray:
    return np.repeat(rng.choice([-1.0, 1.0], size=(reps, n // length + 1)), length, axis=1)[:, :n]


def _simulate(v, a: float, noise_sd: float, rng, burn: int) -> tuple[np.ndarray, np.ndarray]:
    """``(u, y)`` after ``burn`` periods; ``y[:, t + 1]`` is the first reading after ``u[:, t]``."""
    reps, n = v.shape
    u = (v + 1.0) / 2.0
    x = np.zeros((reps, n + 1))
    eps = SIGMA * rng.standard_normal((reps, n))
    for t in range(n):
        x[:, t + 1] = a * x[:, t] + B * u[:, t] + eps[:, t]
    y = x + noise_sd * rng.standard_normal(x.shape)
    return u[:, burn:], y[:, burn:]


def _centre(z: np.ndarray) -> np.ndarray:
    return z - z.mean(axis=1, keepdims=True)


def _two_regressors(y, x1, x2) -> tuple[np.ndarray, np.ndarray]:
    """OLS of ``y`` on ``(1, x1, x2)`` per row, or IV when ``x1`` is replaced by an instrument."""
    y, x1, x2 = _centre(y), _centre(x1), _centre(x2)
    s11, s12, s22 = (x1 * x1).sum(1), (x1 * x2).sum(1), (x2 * x2).sum(1)
    s1y, s2y = (x1 * y).sum(1), (x2 * y).sum(1)
    det = s11 * s22 - s12**2
    return (s22 * s1y - s12 * s2y) / det, (s11 * s2y - s12 * s1y) / det


def _plug_in(a_hat, b_hat, h: float) -> np.ndarray:
    return b_hat / (1.0 - a_hat) if math.isinf(h) else b_hat * (1.0 - a_hat**h) / (1.0 - a_hat)


def _state_aware(u, y, h: float) -> np.ndarray:
    return _plug_in(*_two_regressors(y[:, 1:], y[:, :-1], u), h)


def _iv(u, y, h: float) -> np.ndarray:
    y1, y0, now, lagged = y[:, 2:], y[:, 1:-1], u[:, 1:], u[:, :-1]
    y1, y0, now, lagged = _centre(y1), _centre(y0), _centre(now), _centre(lagged)
    zx11, zx12 = (lagged * y0).sum(1), (lagged * now).sum(1)
    zx21, zx22 = (now * y0).sum(1), (now * now).sum(1)
    zy1, zy2 = (lagged * y1).sum(1), (now * y1).sum(1)
    det = zx11 * zx22 - zx12 * zx21
    return _plug_in((zx22 * zy1 - zx12 * zy2) / det, (zx11 * zy2 - zx21 * zy1) / det, h)


def _local_projection(u, y, h: int) -> np.ndarray:
    n = u.shape[1]
    ahead = sum(y[:, j : j + n - h + 1] for j in range(1, h + 1))
    return _two_regressors(ahead, u[:, : n - h + 1], y[:, : n - h + 1])[0]


def _block_dim(u, y, design: BlockDesign) -> np.ndarray:
    keep = (np.arange(u.shape[1]) % design.length) >= design.washout
    on, off = (u == 1) & keep, (u == 0) & keep
    outcome = y[:, 1:]
    return (outcome * on).sum(1) / on.sum(1) - (outcome * off).sum(1) / off.sum(1)


def _read(plan: SwitchbackPlan, noise_sd: float, reps: int, seed: int) -> dict:
    """Each effect's reading on ``reps`` simulated runs of its plan, pooled over the zones."""
    rng = np.random.default_rng(seed)
    readings = {}
    for j, arm in enumerate(plan.arms):
        periods = round(arm.share * plan.periods)
        rows = reps * plan.zones
        design = arm.design
        if isinstance(design, MarkovDesign):
            burn = 400
            v = _markov_signs(rng, rows, periods + burn, design.flip)
        else:
            burn = design.length * math.ceil(400 / design.length)
            v = _block_signs(rng, rows, periods + burn, design.length)
        u, y = _simulate(v, A, noise_sd, rng, burn)
        for report in (r for r in plan.reports if r.arm == j):
            h = report.estimand.periods
            if report.analysis == "state_aware":
                x = _state_aware(u, y, h)
            elif report.analysis == "iv":
                x = _iv(u, y, h)
            elif report.analysis == "local_projection":
                x = _local_projection(u, y, int(h))
            else:
                assert isinstance(design, BlockDesign)
                x = _block_dim(u, y, design)
            readings[report.estimand] = x.reshape(reps, plan.zones).mean(axis=1)
    return readings


# --- closed forms -----------------------------------------------------------------------------


def test_the_variance_matches_the_labs_maxima_form():
    # the lab's form, transcribed from its Maxima derivation, against the module's, written as the
    # known-persistence variance plus the misalignment loss: two algebraically different forms
    rng = np.random.default_rng(1)
    for _ in range(200):
        a, q = rng.uniform(0.05, 0.97), rng.uniform(0.05, 5.0)
        h = [1.0, 2.0, 3.0, 7.0, math.inf][rng.integers(5)]
        a1 = _a1_markov(rng.uniform(-0.9, 0.99), a)
        s, ds = _s(a, h), _ds(a, h)
        mxx = (q * (1 + 2 * a * a1) + 1) / (1 - a**2)
        lab = (q * ds**2 - 2 * q * s * ds * a1 + s**2 * mxx) / (mxx - q * a1**2)
        assert _v_state_aware(a1, a, q, h) == pytest.approx(lab, rel=1e-10)
        assert _information(a1, a, q) == pytest.approx(mxx - q * a1**2, rel=1e-12)


@pytest.mark.parametrize("h", [1.0, 2.0, 5.0, math.inf])
def test_the_derivative_of_the_horizon_sum_is_its_derivative(h):
    for a in (0.1, 0.5, 0.8, 0.95):
        step = 1e-6
        numeric = (_s(a + step, h) - _s(a - step, h)) / (2 * step)
        assert _ds(a, h) == pytest.approx(numeric, rel=1e-7)


def test_the_block_difference_in_means_reads_the_mean_effect_over_the_positions_it_keeps():
    # position k of a block has held the lever for k + 1 periods, so it reads tau_{k+1}
    for a in (0.3, 0.8, 0.95):
        for length in (1, 2, 5, 12):
            for washout in range(length):
                kept = range(washout, length)
                mean_effect = np.mean([_s(a, k + 1) for k in kept])
                for h in (1.0, 3.0, math.inf):
                    assert _dim_bias(length, washout, a, h) == pytest.approx(
                        mean_effect - _s(a, h), abs=1e-12
                    )
        # DIM(H, H-1) keeps only the last period, which has held the lever for H
        for h in (1, 2, 5, 9):
            assert _dim_bias(h, h - 1, a, float(h)) == pytest.approx(0.0, abs=1e-12)


# --- the design rule ---------------------------------------------------------------------------


@pytest.mark.parametrize("a", [0.2, 0.5, 0.8])
def test_the_two_period_effect_flips_with_probability_a_over_1_plus_2a(a):
    plan = design_switchback((Horizon(2),), PersistencePrior(a, a, 1.0, 2.0), 100_000)
    (arm,) = plan.arms
    assert isinstance(arm.design, MarkovDesign)
    assert arm.design.flip == pytest.approx(a / (1 + 2 * a), rel=1e-6)
    assert plan.reports[0].loss == pytest.approx(0.0, abs=1e-9)


def test_the_aligned_design_does_not_depend_on_the_signal_to_noise_ratio():
    flips = set()
    for guess in (0.5, 2.0, 8.0):
        plan = design_switchback((Horizon(5),), PersistencePrior(A, A, 1.0, guess), 2000)
        design = plan.arms[0].design
        assert isinstance(design, MarkovDesign)
        flips.add(round(design.flip, 7))
    assert len(flips) == 1


@pytest.mark.parametrize("guess", [1.0, 2.0, 3.5])
def test_the_channel_and_the_steady_state_meet_at_the_chebyshev_midpoint(guess):
    # each loses q (A1 - A1*)^2 / D(A1) against its own optimum, A1* = 0 and 1 / (1 - a), so the
    # worst of the two is least at A1 = 1 / (2 (1 - a)); a long run keeps the switch floor slack
    q = guess**2 / 4
    plan = design_switchback(
        (CHANNEL, STEADY_STATE), PersistencePrior(A, A, 1.0, guess), 10**6, min_switches=1
    )
    design = plan.arms[0].design
    assert isinstance(design, MarkovDesign)
    assert design.flip == pytest.approx((1 - A) / (2 * (2 - A)), rel=1e-5)
    for report in plan.reports:
        assert report.loss == pytest.approx(q * (1 + A) / (4 * (1 - A) + q * (3 - A)), rel=1e-5)


def _arms(plan: SwitchbackPlan) -> list[tuple[float, object]]:
    return [(round(arm.share, 6), arm.design) for arm in plan.arms]


# the lab's reference planner at a = 0.8, sigma = 1, b = 2 and 2000 periods, each printed to four
# digits: (estimands, prior, options, arms, [(analysis, se, mde, bias, loss) per effect], zones)
LAB_PLANS = {
    "channel": (
        (CHANNEL,),
        PRIOR,
        {},
        [MarkovDesign(0.5)],
        [("state_aware", 0.0447, 0.1253, 0, 0)],
    ),
    "tau_5": (
        (Horizon(5),),
        PRIOR,
        {},
        [MarkovDesign(0.11884865406811923)],
        [("state_aware", 0.1503, 0.4212, 0, 0)],
    ),
    "steady state at the switch floor": (
        (STEADY_STATE,),
        PRIOR,
        {},
        [MarkovDesign(0.02)],
        [("state_aware", 0.2355, 0.6597, 0, 0)],
    ),
    "channel and steady state": (
        (CHANNEL, STEADY_STATE),
        PRIOR,
        {},
        [MarkovDesign(0.09341723161718518)],
        [("state_aware", 0.0551, 0.1543, 0, 0.516), ("state_aware", 0.2900, 0.8124, 0, 0.516)],
    ),
    "tau_5 over a in [0.7, 0.9]": (
        (Horizon(5),),
        PersistencePrior(0.7, 0.9, SIGMA, B),
        {},
        [MarkovDesign(0.11578500131974623)],
        [("state_aware", 0.1840, 0.5154, 0, 0.009)],
    ),
    "noisy channel and tau_5": (
        (CHANNEL, Horizon(5)),
        PersistencePrior(A, A, SIGMA, B, noise_ratio=1.0),
        {},
        [MarkovDesign(0.22715865216041542)],
        [("iv", 0.0867, 0.2429, 0, 0.479), ("iv", 0.2146, 0.6012, 0, 0.479)],
    ),
    "tau_5 without a model": (
        (Horizon(5),),
        PRIOR,
        {"trust_state_model": False},
        [BlockDesign(5, 4)],
        [("block_dim", 0.2034, 0.5698, 0, 0)],
    ),
    "channel and tau_5 without a model": (
        (CHANNEL, Horizon(5)),
        PRIOR,
        {"trust_state_model": False},
        [MarkovDesign(0.5), BlockDesign(5, 4)],
        [("state_aware", 0.0632, 0.1772, 0, 1.0), ("block_dim", 0.2876, 0.8059, 0, 1.0)],
    ),
    "steady state without a model, four zones": (
        (STEADY_STATE,),
        PRIOR,
        {"trust_state_model": False, "zones": 4},
        [BlockDesign(50, 16)],
        [("block_dim", 0.1264, 0.3873, -0.0331, 0)],
    ),
}


@pytest.mark.parametrize("case", LAB_PLANS)
def test_the_plans_reproduce_the_labs_reference_planner(case):
    estimands, prior, options, designs, readings = LAB_PLANS[case]
    plan = design_switchback(estimands, prior, 2000, **options)
    assert len(plan.arms) == len(designs)
    for arm, expected in zip(plan.arms, designs, strict=True):
        assert type(arm.design) is type(expected)
        if isinstance(expected, MarkovDesign):
            assert arm.design.flip == pytest.approx(expected.flip, rel=1e-7)
        else:
            assert arm.design == expected
    for report, (analysis, se, mde, bias, loss) in zip(plan.reports, readings, strict=True):
        assert report.analysis == analysis
        assert report.se == pytest.approx(se, abs=5e-5)
        assert report.mde == pytest.approx(mde, abs=5e-5)
        assert report.bias == pytest.approx(bias, abs=5e-5)
        assert report.loss == pytest.approx(loss, abs=5e-4)


def test_a_target_mde_takes_the_fewest_zones_that_reach_it():
    plan = design_switchback((Horizon(5),), PRIOR, 2000, TargetMDE(0.05))
    target = 0.05 * _tau(A, 5)
    assert plan.reports[0].mde <= target
    fewer = design_switchback((Horizon(5),), PRIOR, 2000, plan.zones - 1)
    assert fewer.reports[0].mde > target
    assert plan.zones == 2
    # the lab's: the steady state without a model at 5% of tau takes three zones
    steady = design_switchback(
        (STEADY_STATE,), PRIOR, 2000, TargetMDE(0.05), trust_state_model=False
    )
    assert steady.zones == 3
    assert steady.reports[0].mde == pytest.approx(0.4421, abs=5e-5)


def test_the_design_mode_warns_for_what_it_cannot_see(caplog):
    with caplog.at_level(logging.INFO, logger="chc.switchback"):
        steady = design_switchback((STEADY_STATE,), PRIOR, 2000)
    assert any("floor" in w for w in steady.warnings)
    assert any("first-order state" in w for w in steady.warnings)
    assert any("about 40 switches" in w for w in steady.warnings)
    assert [r.chc_event for r in caplog.records] == ["switchback_design"]

    noisy = design_switchback((Horizon(3),), PersistencePrior(A, A, SIGMA, B, 1.0), 2000)
    assert any("measured with noise" in w for w in noisy.warnings)
    assert noisy.reports[0].analysis == "iv"

    zoned = design_switchback((CHANNEL,), PRIOR, 2000, 3)
    assert any("several zones" in w for w in zoned.warnings)
    assert not design_switchback((CHANNEL,), PRIOR, 2000).warnings


def test_a_short_run_near_a_unit_root_is_flagged():
    # the verifier measured the finite-run variance at 1.07-1.11x the asymptotic one at a = 0.6
    # over 100 periods, and 1.02-1.04x at a = 0.95 over 2000
    def flagged(a: float, periods: int, zones: int = 1) -> bool:
        plan = design_switchback(
            (Horizon(3),), PersistencePrior(a, a, SIGMA, B), periods, zones, min_switches=20
        )
        return any("unit" in w or "persistence estimate" in w for w in plan.warnings)

    assert flagged(0.95, 200)
    assert flagged(0.6, 100)
    assert not flagged(0.8, 2000)
    # its O(1/T) bias does not shrink with more zones, so neither does the flag
    assert flagged(0.95, 200, zones=50)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: PersistencePrior(0.0, 0.5, 1.0, 1.0), "0 < a_lo"),
        (lambda: PersistencePrior(-0.3, 0.5, 1.0, 1.0), "0 < a_lo"),
        (lambda: PersistencePrior(0.5, 1.0, 1.0, 1.0), "0 < a_lo"),
        (lambda: PersistencePrior(0.6, 0.5, 1.0, 1.0), "0 < a_lo"),
        (lambda: PersistencePrior(0.5, 0.5, 0.0, 1.0), "shock_sd > 0"),
        (lambda: PersistencePrior(0.5, 0.5, 1.0, 0.0), "channel_guess != 0"),
        (lambda: PersistencePrior(0.5, 0.5, 1.0, 1.0, -0.1), "noise_ratio >= 0"),
        (lambda: PersistencePrior(0.5, math.nan, 1.0, 1.0), "finite"),
        (lambda: Horizon(0), "whole number"),
        (lambda: Horizon(2.5), "whole number"),
        (lambda: Horizon(-math.inf), "whole number"),
        (lambda: TargetMDE(0.0), "positive"),
        (lambda: TargetMDE(math.inf), "positive"),
        (lambda: design_switchback((), PRIOR, 2000), "distinct"),
        (lambda: design_switchback((CHANNEL, Horizon(1)), PRIOR, 2000), "distinct"),
        (lambda: design_switchback((CHANNEL,), PRIOR, 159), "at least 160"),
        (lambda: design_switchback((CHANNEL,), PRIOR, 2000, 0), "at least 1"),
        (lambda: design_switchback((CHANNEL,), PRIOR, 2000, alpha=1.0), r"\(0, 1\)"),
        (lambda: design_switchback((CHANNEL,), PRIOR, 2000, power=0.0), r"\(0, 1\)"),
        (
            lambda: design_switchback(
                (STEADY_STATE,), PRIOR, 2000, TargetMDE(0.001), trust_state_model=False
            ),
            "no number of zones",
        ),
    ],
)
def test_the_design_refuses_what_it_cannot_plan(make, match):
    with pytest.raises(ValueError, match=match):
        make()


# --- simulation --------------------------------------------------------------------------------

# (estimands, noise ratio, options): the lab's five checks, the local projection that the working
# model prefers to DIM(2, 1) at a = 0.8, with and without noise, and the channel alone under noise
SIMULATED = {
    "tau_5": ((Horizon(5),), 0.0, {}),
    "channel and steady state": ((CHANNEL, STEADY_STATE), 0.0, {}),
    "noisy channel and tau_5": ((CHANNEL, Horizon(5)), 1.0, {}),
    "channel and tau_5 without a model": (
        (CHANNEL, Horizon(5)),
        0.0,
        {"trust_state_model": False},
    ),
    "steady state without a model, four zones": (
        (STEADY_STATE,),
        0.0,
        {"trust_state_model": False, "zones": 4},
    ),
    "tau_2 by local projection": ((Horizon(2),), 0.0, {"trust_state_model": False}),
    "noisy tau_2 by local projection": ((Horizon(2),), 1.0, {"trust_state_model": False}),
    "noisy channel alone": ((CHANNEL,), 1.0, {}),
}


@pytest.mark.parametrize("case", SIMULATED)
def test_the_planned_standard_error_and_power_hold_in_simulation(case):
    estimands, kappa, options = SIMULATED[case]
    prior = PersistencePrior(A, A, SIGMA, B, noise_ratio=kappa)
    plan = design_switchback(estimands, prior, 2000, **options)
    reps = 2000
    readings = _read(
        plan, math.sqrt(kappa) * SIGMA, reps, seed=83_000 + list(SIMULATED).index(case)
    )
    for report in plan.reports:
        x = readings[report.estimand]
        tau = _tau(A, report.estimand.periods)
        sd = x.std(ddof=1)
        # the ratio's own sd is 1 / sqrt(2 reps) = 0.016; the lab measured 0.971 to 1.022
        assert sd / report.se == pytest.approx(1.0, abs=0.07), (report.estimand, sd / report.se)
        # a two-sided test at the MDE rejects with the planned power, 0.80 +- 0.009
        rejects = np.abs(x - (tau - report.mde)) / report.se > 1.959964
        assert rejects.mean() == pytest.approx(0.8, abs=0.05), (report.estimand, rejects.mean())
        # consistent readings are centred on tau; DIM carries its O(1/blocks) ratio bias, 0.25%
        # of tau_5 here, and the steady state the working-model bias the plan quotes
        slack = 4 * sd / math.sqrt(reps) + 0.005 * abs(tau)
        assert x.mean() - tau == pytest.approx(report.bias, abs=slack), report.estimand


def test_the_aligned_design_beats_an_iid_one_by_the_predicted_factor():
    # the gate: against a design that decorrelates the lever from the state (A1 = 0), the aligned
    # design removes the whole misalignment loss q S'^2 / D(0) from tau_5's variance
    plan = design_switchback((Horizon(5),), PRIOR, 2000)
    q = PRIOR.signal_to_noise
    predicted = _v_state_aware(0.0, A, q, 5.0) / _s(A, 5.0) ** 2
    assert predicted == pytest.approx(1.687, abs=1e-3)
    iid = design_switchback((CHANNEL,), PRIOR, 2000)
    reps = 4000
    aligned = _read(plan, 0.0, reps, seed=11)[Horizon(5)]
    rng = np.random.default_rng(12)
    u, y = _simulate(_markov_signs(rng, reps, 2400, 0.5), A, 0.0, rng, 400)
    assert iid.arms[0].design == MarkovDesign(0.5)
    ratio = _state_aware(u, y, 5.0).var() / aligned.var()
    # the ratio of two variances over 4000 runs each is known to about 3%
    assert ratio == pytest.approx(predicted, rel=0.1)


# --- reading the data ----------------------------------------------------------------------------


def _arm_runs(plan: SwitchbackPlan, arm: int, noise_sd: float, reps: int, rng):
    """``reps`` simulated runs of one arm, shaped ``(reps, zones, periods)`` and ``periods + 1``."""
    design = plan.arms[arm].design
    periods = round(plan.arms[arm].share * plan.periods)
    rows = reps * plan.zones
    if isinstance(design, MarkovDesign):
        burn = 400
        v = _markov_signs(rng, rows, periods + burn, design.flip)
    else:
        burn = design.length * math.ceil(400 / design.length)
        v = _block_signs(rng, rows, periods + burn, design.length)
    u, y = _simulate(v, A, noise_sd, rng, burn)
    return u.reshape(reps, plan.zones, -1), y.reshape(reps, plan.zones, -1)


@pytest.mark.parametrize("case", SIMULATED)
def test_the_data_standard_error_covers_at_the_nominal_rate(case):
    estimands, kappa, options = SIMULATED[case]
    prior = PersistencePrior(A, A, SIGMA, B, noise_ratio=kappa)
    plan = design_switchback(estimands, prior, 2000, **options)
    reps = 1000
    rng = np.random.default_rng(91_000 + list(SIMULATED).index(case))
    for j, arm in enumerate(plan.arms):
        u, y = _arm_runs(plan, j, math.sqrt(kappa) * SIGMA, reps, rng)
        blocks = arm.design if isinstance(arm.design, BlockDesign) else None
        for report in (r for r in plan.reports if r.arm == j):
            # the block difference reads the steady state with the bias the plan quotes
            target = _tau(A, report.estimand.periods) + report.bias
            readings = [
                read_switchback(u[r], y[r], report.estimand, report.analysis, blocks=blocks)
                for r in range(reps)
            ]
            estimates = np.array([x.estimate for x in readings])
            se = np.array([x.se for x in readings])
            covered = np.mean([x.interval[0] <= target <= x.interval[1] for x in readings])
            # 0.95 +- 0.007 over 1000 runs; measured 0.937 to 0.959
            assert covered == pytest.approx(0.95, abs=0.025), (report.estimand, covered)
            # the data's standard error is the spread it claims, to the ratio's own 0.022
            assert se.mean() / estimates.std() == pytest.approx(1.0, abs=0.07), report.estimand
            # and the spread is the plan's: 0.975 to 1.02, but 1.10 for the steady state's block
            # difference, whose 40 blocks of 50 a zone lose the first few to centring
            assert estimates.std() / report.se == pytest.approx(1.0, abs=0.13), report.estimand
            assert estimates.mean() == pytest.approx(target, abs=4 * estimates.std() / 30)


def test_the_local_projection_keeps_the_lags_a_noisy_state_creates():
    # a state read with noise leaves past levers in the error, so the lever's scores are correlated
    # across the H - 1 periods the sums overlap: at noise 4 and H = 5 dropping those lags puts the
    # standard error 7.8% low. Without noise they are uncorrelated, and the lags change nothing
    rng = np.random.default_rng(10)
    reps, periods, kappa = 4000, 2000, 4.0
    u, y = _simulate(_markov_signs(rng, reps, periods + 400, 0.5), A, 2.0, rng, 400)
    readings = [read_switchback(u[r], y[r], Horizon(5), "local_projection") for r in range(reps)]
    q = B**2 / (4 * SIGMA**2)
    planned = math.sqrt(_v_local_projection(A, q, kappa, 5) * 4 * SIGMA**2 / periods)
    se = np.mean([r.se for r in readings])
    assert se / planned == pytest.approx(1.0, abs=0.02)
    assert np.std([r.estimate for r in readings]) / planned == pytest.approx(1.0, abs=0.035)


def test_the_block_difference_is_unbiased_at_eight_blocks():
    # the verifier's case, a = 0.8 over eight one-period blocks: the realised means are 27% low,
    # since given how many periods were on, an on period's past held fewer on periods than an off
    # one's. Centred on the blocks before it, each block's coin is fair whatever the count
    rng = np.random.default_rng(5)
    reps = 20_000
    u, y = _simulate(_block_signs(rng, reps, 408, 1), A, 0.0, rng, 400)
    one = BlockDesign(1, 0)
    # with the first six coins alike, fewer than two blocks follow both settings
    alike = (u[:, :6] == u[:, :1]).all(axis=1)
    assert alike.mean() == pytest.approx(1 / 32, abs=0.005)
    with pytest.raises(ValueError, match="both settings"):
        read_switchback(u[alike][0], y[alike][0], CHANNEL, "block_dim", blocks=one)
    centred = np.array(
        [
            read_switchback(u[r], y[r], CHANNEL, "block_dim", blocks=one).estimate
            for r in np.flatnonzero(~alike)
        ]
    )
    assert centred.mean() == pytest.approx(B, abs=4 * centred.std() / math.sqrt(centred.size))
    both = (u.sum(axis=1) > 0) & (u.sum(axis=1) < 8)
    realised = _block_dim(u[both], y[both], one)
    assert realised.mean() / B - 1 == pytest.approx(-0.27, abs=0.02)


def test_near_a_unit_root_the_steady_state_interval_is_fiellers(caplog):
    # a = 0.95 over 200 periods: the delta method covers 0.877 and Fieller's 0.930; the rest is
    # a_hat's O(1/T) bias, which the reading warns of
    rng = np.random.default_rng(6)
    reps = 2000
    u, y = _simulate(_markov_signs(rng, reps, 600, 0.2), 0.95, 0.0, rng, 400)
    tau = B / 0.05
    fieller = delta = 0
    with caplog.at_level(logging.WARNING, logger="chc.switchback"):
        for r in range(reps):
            reading = read_switchback(u[r], y[r], STEADY_STATE, "state_aware")
            fieller += reading.interval[0] <= tau <= reading.interval[1]
            delta += abs(reading.estimate - tau) <= 1.959964 * reading.se
    assert fieller / reps == pytest.approx(0.930, abs=0.005)
    assert delta / reps == pytest.approx(0.877, abs=0.005)
    events = [r.chc_event for r in caplog.records]
    assert events.count("switchback_near_unit_root") == reps


def test_fiellers_interval_is_where_the_ratio_test_does_not_reject():
    z = 1.959964
    rng = np.random.default_rng(8)
    for _ in range(200):
        a, b = rng.uniform(0.3, 0.97), rng.uniform(0.5, 3.0)
        root = rng.normal(size=(2, 2)) * rng.uniform(0.001, 0.05)
        cov = root @ root.T + 1e-6 * np.eye(2)
        lo, hi = _fieller(a, b, cov, z)
        if (1 - a) ** 2 <= z * z * cov[0, 0]:
            assert (lo, hi) == (-math.inf, math.inf)
            continue
        for tau in (lo, hi):
            variance = cov[1, 1] + 2 * tau * cov[0, 1] + tau * tau * cov[0, 0]
            assert (b - tau * (1 - a)) ** 2 == pytest.approx(z * z * variance, rel=1e-8)
        assert lo < b / (1 - a) < hi


def test_the_readings_are_the_labs_estimators_on_one_zone():
    rng = np.random.default_rng(3)
    u, y = _simulate(_markov_signs(rng, 4, 1400, 0.5), A, 0.5, rng, 400)
    for r in range(4):
        one_u, one_y = u[r : r + 1], y[r : r + 1]
        for h in (1.0, 5.0, math.inf):
            for analysis, lab in (("state_aware", _state_aware), ("iv", _iv)):
                reading = read_switchback(u[r], y[r], Horizon(h), analysis)
                assert reading.estimate == pytest.approx(lab(one_u, one_y, h)[0], rel=1e-9)
        projection = read_switchback(u[r], y[r], Horizon(3), "local_projection")
        assert projection.estimate == pytest.approx(_local_projection(one_u, one_y, 3)[0], rel=1e-9)


def test_a_zones_level_does_not_move_the_reading():
    rng = np.random.default_rng(4)
    u, y = _simulate(_markov_signs(rng, 1, 1400, 0.3), A, 0.0, rng, 400)
    alone = read_switchback(u[0], y[0], Horizon(4), "state_aware")
    pooled = read_switchback(np.vstack([u, u]), np.vstack([y, y + 10.0]), Horizon(4), "state_aware")
    assert pooled.estimate == pytest.approx(alone.estimate, rel=1e-10)
    assert pooled.zones == 2
    assert pooled.se == pytest.approx(alone.se / math.sqrt(2), rel=0.01)


def _run(periods: int, flip: float = 0.5, a: float = A, seed: int = 9):
    rng = np.random.default_rng(seed)
    u, y = _simulate(_markov_signs(rng, 1, periods + 400, flip), a, 0.0, rng, 400)
    return u[0], y[0]


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda u, y: read_switchback(u, y[:-1], CHANNEL, "state_aware"), "one more period"),
        (lambda u, y: read_switchback(0.5 * u, y, CHANNEL, "state_aware"), "0 or 1"),
        (
            lambda u, y: read_switchback(
                u, np.where(np.arange(y.size) == 5, np.nan, y), CHANNEL, "iv"
            ),
            "finite",
        ),
        (lambda u, y: read_switchback(0 * u, y, CHANNEL, "state_aware"), "never switches"),
        (lambda u, y: read_switchback(u, y, CHANNEL, "state_aware", alpha=0.0), r"\(0, 1\)"),
        (
            lambda u, y: read_switchback(u, y, CHANNEL, "state_aware", blocks=BlockDesign(2, 1)),
            "blocks are read by block_dim",
        ),
        (lambda u, y: read_switchback(u, y, CHANNEL, "block_dim"), "blocks are read by block_dim"),
        (
            lambda u, y: read_switchback(u, y, Horizon(5), "block_dim", blocks=BlockDesign(5, 3)),
            "keeping the last",
        ),
        (
            lambda u, y: read_switchback(u, y, Horizon(2), "block_dim", blocks=BlockDesign(2, 1)),
            "changes inside a block",
        ),
        (
            lambda u, y: read_switchback(
                u[:10], y[:11], STEADY_STATE, "block_dim", blocks=BlockDesign(4, 1)
            ),
            "three whole blocks",
        ),
        (
            lambda u, y: read_switchback(u, y, STEADY_STATE, "local_projection"),
            "finite horizon",
        ),
    ],
)
def test_the_reading_refuses_what_it_cannot_read(call, match):
    u, y = _run(300)
    with pytest.raises(ValueError, match=match):
        call(u, y)


def test_a_local_projection_needs_an_iid_lever():
    u, y = _run(2000, flip=0.1)
    with pytest.raises(ValueError, match=r"i\.i\.d\."):
        read_switchback(u, y, Horizon(3), "local_projection")


def test_an_explosive_fit_has_no_steady_state():
    u, y = _run(300, a=1.02)
    with pytest.raises(ValueError, match="no steady state"):
        read_switchback(u, y, STEADY_STATE, "state_aware")


def test_a_reading_is_logged(caplog):
    u, y = _run(2000)
    with caplog.at_level(logging.INFO, logger="chc.switchback"):
        reading = read_switchback(u, y, CHANNEL, "state_aware")
    (record,) = caplog.records
    assert record.chc_event == "switchback_reading"
    assert record.estimate == reading.estimate
    assert reading.periods == 2000
    assert reading.zones == 1
