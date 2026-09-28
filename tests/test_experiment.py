"""The experiment design: its local model against the lab's verified numbers, its allocation against
closed forms, and its verdicts on the lab's five scenarios.

The market is the lab's: four zones on a ring, a lever pulling 15% of its effect from each
neighbour, ``coupling = I - P``. The scalar decision is the lab's tracking plant, ``q = 1``,
``r = 1/2``, ``e = 1``, whose knife edge sits at ``b = sqrt(r / q)``.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from scipy import integrate, optimize, stats

from chc.experiment import (
    ChannelPrior,
    ZoneDecision,
    ZoneExperiment,
    _local,
    _second_partial_moment,
    _sigma_weight,
    _water_filling,
    design_experiment,
)

K = 4
_RING = np.zeros((K, K))
for _k in range(K):
    _RING[_k, (_k + 1) % K] = _RING[_k, (_k - 1) % K] = 0.15
COUPLING = np.eye(K) - _RING
Q = np.diag([2.0, 1.0, 1.5, 0.7])
R = np.diag([0.4, 0.2, 0.6, 0.3])
B_TRUE = np.array([0.8, 1.5, 0.5, 1.1])
NOISE_SD = np.array([1.0, 2.0, 0.7, 1.5])
DEMO_EXPERIMENT = ZoneExperiment(
    unit_cost=np.array([1e-5, 0.5e-5, 3e-5, 1e-5]), noise_sd=NOISE_SD, probe=np.ones(K)
)


def _market(baseline, lo=0.0, hi=1.2) -> ZoneDecision:
    return ZoneDecision(
        coupling=COUPLING,
        baseline=np.asarray(baseline, dtype=float),
        target=np.zeros(K),
        state_weight=Q,
        action_weight=R,
        lo=np.full(K, lo),
        hi=np.full(K, hi),
    )


def _scalar(lo: float = -math.inf, hi: float = math.inf) -> ZoneDecision:
    return ZoneDecision(
        coupling=np.eye(1),
        baseline=np.array([-1.0]),
        target=np.zeros(1),
        state_weight=np.eye(1),
        action_weight=0.5 * np.eye(1),
        lo=np.array([lo]),
        hi=np.array([hi]),
    )


def test_the_solver_finds_the_box_optimum():
    rng = np.random.default_rng(0)
    for _ in range(40):
        a = rng.normal(size=(K, K))
        decision = ZoneDecision(
            coupling=np.eye(K) + 0.3 * rng.normal(size=(K, K)),
            baseline=rng.normal(size=K),
            target=rng.normal(size=K),
            state_weight=a @ a.T,
            action_weight=np.diag(rng.uniform(0.1, 1.0, K)),
            lo=-rng.uniform(0.0, 1.0, K),
            hi=rng.uniform(0.1, 1.0, K),
        )
        b = rng.uniform(-2.0, 2.0, K)
        u = decision.solve(b)
        reference = optimize.minimize(
            lambda v, b=b, decision=decision: decision.cost(v, b),
            np.zeros(K),
            method="L-BFGS-B",
            bounds=list(zip(decision.lo, decision.hi, strict=True)),
            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000},
        )
        assert decision.cost(u, b) <= reference.fun + 1e-12
        assert decision.regret(u, b) == pytest.approx(0.0, abs=1e-12)


def test_the_decision_weight_is_the_hessian_of_the_certainty_equivalent_regret():
    decision = _market([-1.0, -0.4, -1.3, 0.2], lo=-math.inf, hi=math.inf)
    weight = _local(decision, B_TRUE).weight
    lab = np.array(
        [
            [0.70061, -0.07414, 0.16649, 0.07374],
            [-0.07414, 0.03202, -0.05597, -0.00416],
            [0.16649, -0.05597, 0.51388, 0.02975],
            [0.07374, -0.00416, 0.02975, 0.02084],
        ]
    )
    np.testing.assert_allclose(weight, lab, atol=6e-6)

    def hessian(f, h=1e-4):
        out = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                ei, ej = h * np.eye(K)[i], h * np.eye(K)[j]
                out[i, j] = (
                    f(B_TRUE + ei + ej)
                    - f(B_TRUE + ei - ej)
                    - f(B_TRUE - ei + ej)
                    + f(B_TRUE - ei - ej)
                ) / (4 * h * h)
        return out

    action = decision.solve(B_TRUE)
    envelope = hessian(lambda b: decision.cost(action, b)) - hessian(
        lambda b: decision.cost(decision.solve(b), b)
    )
    regret = hessian(lambda b_hat: decision.regret(decision.solve(b_hat), B_TRUE))
    np.testing.assert_allclose(envelope, weight, atol=1e-6)
    np.testing.assert_allclose(regret, weight, atol=1e-6)


@pytest.mark.parametrize("z", [-1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
def test_the_second_partial_moment_is_its_integral(z):
    integral, _ = integrate.quad(
        lambda x: (x - z) ** 2 * stats.norm.pdf(x), z, math.inf, epsabs=0.0, epsrel=1e-13, limit=200
    )
    assert _second_partial_moment(z) == pytest.approx(integral, rel=1e-9, abs=1e-300)


@pytest.mark.parametrize("z", [3.0, 5.0, 10.0, 20.0])
def test_the_second_partial_moment_obeys_its_tail_bounds(z):
    phi = stats.norm.pdf(z)
    assert phi * (2 / z**3 - 12 / z**5) <= _second_partial_moment(z) < 2 * phi / z**3


def test_a_pinned_lever_costs_the_gaussian_tail_of_its_multiplier():
    # b_hat < 0 wants a negative action; the box [0, 10] pins it at 0, one prior sd from release
    decision, s = _scalar(0.0, 10.0), 0.1
    local = _local(decision, np.array([-0.1]))
    assert np.all(local.weight == 0.0)
    (pin,) = local.pins
    design = design_experiment(
        decision,
        ChannelPrior(np.array([-0.1]), np.array([[s * s]])),
        ZoneExperiment(unit_cost=np.ones(1), noise_sd=np.ones(1), probe=np.ones(1)),
        0.0,
        draws=200_000,
        seed=1,
    )
    (zone,) = design.pinned
    assert zone.z == pytest.approx(1.0)
    assert zone.regret == pytest.approx(
        (s * s) * _second_partial_moment(1.0) / (2 * pin.curvature), rel=1e-12
    )
    # the lab's quadrature of the true expected regret is 7.2807e-4; the tail law, with the
    # curvature at b_hat, reads 1.45% above it
    assert design.regret_now == pytest.approx(7.2807e-4, rel=0.03)
    assert zone.regret / 7.2807e-4 == pytest.approx(1.0145, abs=5e-4)


def test_the_knife_edge_keeps_a_weight_the_plug_in_loses():
    decision, s = _scalar(), 0.05
    edge = np.array([math.sqrt(0.5)])
    assert _local(decision, edge).weight[0, 0] == pytest.approx(0.0, abs=1e-12)
    prior = ChannelPrior(edge, np.array([[s * s]]))
    assert _sigma_weight(decision, prior)[0, 0] > 0.0
    design = design_experiment(
        decision,
        prior,
        ZoneExperiment(unit_cost=np.ones(1), noise_sd=np.ones(1), probe=np.ones(1)),
        0.0,
        draws=100_000,
        seed=2,
    )
    # fourth order at the edge, 3 e^2 q^3 s^4 / (16 r^2), up to a relative 10 q s^2 / r
    assert design.regret_now == pytest.approx(3 * s**4 / (16 * 0.25), rel=0.08)


def test_the_local_regret_is_the_monte_carlo_regret_on_the_market():
    decision = _market([-1.0, -0.4, -1.3, 0.2], lo=-math.inf, hi=math.inf)
    weight = _local(decision, B_TRUE).weight
    rng = np.random.default_rng(400)
    variance = NOISE_SD**2 / 400
    draws = B_TRUE + rng.standard_normal((8000, K)) * np.sqrt(variance)
    regret = np.array([decision.regret(decision.solve(b), B_TRUE) for b in draws])
    predicted = 0.5 * float(np.sum(np.diag(weight) * variance))
    # the lab: 0.9977 at 400 units a zone
    assert regret.mean() / predicted == pytest.approx(1.0, abs=0.04)


def test_decision_neyman_is_the_flat_prior_water_filling_and_beats_classical_neyman():
    decision = _market([-1.0, -0.4, -1.3, 0.2], lo=-math.inf, hi=math.inf)
    weight = _local(decision, B_TRUE).weight
    w = np.diag(weight)
    cost = np.array([1.0, 0.5, 3.0, 1.0])
    experiment = ZoneExperiment(unit_cost=cost, noise_sd=NOISE_SD, probe=np.ones(K))
    flat = ChannelPrior(B_TRUE, 1e12 * np.eye(K))
    units = _water_filling(weight, flat, experiment, 2000.0, uses=1e12)
    raw = NOISE_SD * np.sqrt(w / cost)
    np.testing.assert_allclose(units, 2000.0 * raw / (raw @ cost), rtol=1e-6)

    def leading(n):
        return 0.5 * float(np.sum(w * NOISE_SD**2 / n))

    optimum = 0.5 * float(np.sum(NOISE_SD * np.sqrt(w * cost))) ** 2 / 2000.0
    assert leading(units) == pytest.approx(optimum, rel=1e-6)
    assert optimum == pytest.approx(1.1835e-3, rel=1e-3)
    classical = NOISE_SD / np.sqrt(cost)
    assert leading(2000.0 * classical / (classical @ cost)) / optimum > 1.45


def test_water_filling_with_prior_precisions_is_the_bayes_optimum():
    decision = _market([-1.0, -0.4, -1.3, 0.2], lo=-math.inf, hi=math.inf)
    weight = _local(decision, B_TRUE).weight
    precision = np.array([400.0, 1.0, 5.0, 60.0])
    experiment = ZoneExperiment(
        unit_cost=np.array([1.0, 0.5, 3.0, 1.0]), noise_sd=NOISE_SD, probe=np.ones(K)
    )
    prior = ChannelPrior(B_TRUE, np.diag(1.0 / precision))
    units = _water_filling(weight, prior, experiment, 500.0, uses=1e12)
    objective = 0.5 * float(np.sum(np.diag(weight) / (precision + units / NOISE_SD**2)))
    np.testing.assert_allclose(units, [0.0, 225.73, 129.05, 0.0], atol=0.01)
    assert objective == pytest.approx(2.28568617e-3, rel=1e-7)


SCENARIOS = {
    "S1 tight posterior": (
        [-1.0, -0.4, -1.3, -0.6],
        [0.8, 1.5, 0.5, 1.1],
        [0.02, 0.02, 0.02, 0.02],
        1e-3,
        1.0,
        "deploy",
        (0, 0, 0, 0),
    ),
    "S2 wide on zones 0 and 2": (
        [-1.0, -0.4, -1.3, -0.6],
        [0.8, 1.5, 0.5, 1.1],
        [0.3, 0.02, 0.25, 0.02],
        1e-3,
        1.0,
        "experiment",
        (65, 0, 11, 0),
    ),
    "S3 unsigned zone, tiny budget": (
        [-1.0, -0.4, -1.3, -0.6],
        [0.8, 1.5, 0.5, 0.05],
        [0.05, 0.05, 0.05, 0.3],
        1e-6,
        1.0,
        "abstain",
        (0, 0, 0, 0),
    ),
    "S4 pinned zone far from activation": (
        [-1.0, 1.5, -1.3, -0.6],
        [0.8, 1.5, 0.5, 1.1],
        [0.1, 0.4, 0.1, 0.02],
        1e-1,
        30.0,
        "experiment",
        (1021, 0, 361, 0),
    ),
    "S5 pinned zone near activation": (
        [-1.0, -0.05, -1.3, -0.6],
        [0.8, 1.5, 0.5, 1.1],
        [0.1, 0.4, 0.1, 0.02],
        1e-1,
        30.0,
        "experiment",
        (924, 25, 332, 0),
    ),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_the_lab_scenarios_get_the_lab_verdicts(name):
    baseline, mean, sd, budget, uses, verdict, units = SCENARIOS[name]
    design = design_experiment(
        _market(baseline),
        ChannelPrior(np.array(mean), np.diag(np.array(sd) ** 2)),
        DEMO_EXPERIMENT,
        budget,
        uses=uses,
        draws=4000,
        seed=7,
    )
    assert design.verdict == verdict
    assert design.sample_sizes == units
    assert design.spend <= budget


def test_the_reported_error_is_the_half_width_of_the_value_over_seeds():
    # Five units: the regret after tracks the regret now draw by draw (correlation 0.93), so the
    # scatter of their difference is 2.8x smaller than of either. The allocation does not depend on
    # the seed, so only the Monte Carlo moves with it.
    baseline, mean, sd, *_ = SCENARIOS["S2 wide on zones 0 and 2"]
    designs = [
        design_experiment(
            _market(baseline),
            ChannelPrior(np.array(mean), np.diag(np.array(sd) ** 2)),
            DEMO_EXPERIMENT,
            5e-5,
            draws=200,
            seed=seed,
        )
        for seed in range(40)
    ]
    assert {d.sample_sizes for d in designs} == {(5, 0, 0, 0)}
    scatter = np.std([d.value for d in designs], ddof=1)
    # 0.88x measured here and 0.95x over 200 seeds; a spread over 40 seeds is itself ~11% noise
    assert scatter == pytest.approx(np.mean([d.error for d in designs]) / 1.96, rel=0.3)


def test_a_pinned_zone_earns_units_only_near_activation():
    far, near = (
        design_experiment(
            _market(baseline),
            ChannelPrior(B_TRUE, np.diag([0.1, 0.4, 0.1, 0.02]) ** 2),
            DEMO_EXPERIMENT,
            1e-1,
            uses=30.0,
            draws=1000,
        )
        for baseline in ([-1.0, 1.5, -1.3, -0.6], [-1.0, -0.05, -1.3, -0.6])
    )
    assert [(p.zone, round(p.z, 2)) for p in far.pinned] == [(1, 3.72)]
    assert [(p.zone, round(p.z, 2)) for p in near.pinned] == [(1, 0.74)]
    assert far.sample_sizes[1] == 0
    assert near.sample_sizes[1] > 0


def test_a_wide_prior_is_priced_by_monte_carlo_and_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="chc.experiment"):
        design = design_experiment(
            _market([-1.0, -0.4, -1.3, -0.6]),
            ChannelPrior(B_TRUE, np.diag([0.3, 0.02, 0.25, 0.02]) ** 2),
            DEMO_EXPERIMENT,
            1e-3,
            draws=20_000,
            seed=2,
        )
    assert design.model_gap > 0.25
    assert [getattr(r, "chc_event", None) for r in caplog.records] == ["design_gap"]
    # the local model puts the regret after at 1.46e-2; what is reported is the measurement
    assert design.regret_after > 1.9e-2


def test_a_channel_known_exactly_gets_no_units():
    design = design_experiment(
        _market([-1.0, -0.4, -1.3, -0.6]),
        ChannelPrior(B_TRUE, np.diag([0.0, 0.02, 0.25, 0.02]) ** 2),
        DEMO_EXPERIMENT,
        1e-3,
        draws=1000,
    )
    assert design.sample_sizes[0] == 0
    assert design.sample_sizes[2] > 0


def test_without_a_budget_nothing_is_run():
    design = design_experiment(
        _market([-1.0, -0.4, -1.3, -0.6]),
        ChannelPrior(B_TRUE, np.diag([0.3, 0.02, 0.25, 0.02]) ** 2),
        DEMO_EXPERIMENT,
        0.0,
        draws=1000,
    )
    assert design.sample_sizes == (0, 0, 0, 0)
    assert design.spend == 0.0
    assert design.regret_after == design.regret_now
    assert design.verdict == "deploy"


def test_a_deploy_verdict_meets_its_regret_bound_at_about_its_level():
    # worlds with a known truth: each draws an estimate around B_TRUE and designs from it
    decision = _market([-1.0, -0.4, -1.3, -0.6])
    sd = np.full(K, 0.05)
    rng = np.random.default_rng(11)
    met = []
    for replicate in range(60):
        estimate = B_TRUE + sd * rng.standard_normal(K)
        design = design_experiment(
            decision,
            ChannelPrior(estimate, np.diag(sd**2)),
            DEMO_EXPERIMENT,
            0.0,
            draws=400,
            seed=replicate,
        )
        assert design.verdict == "deploy"
        met.append(decision.regret(decision.solve(estimate), B_TRUE) <= design.regret_bound)
    # nominal 0.95; the 500-world campaign is scripts/bench_experiment.py
    assert np.mean(met) >= 0.85


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"lo": np.ones(K), "hi": np.ones(K)}, "lo >= hi"),
        ({"action_weight": np.zeros((K, K))}, "positive definite"),
        ({"state_weight": -np.eye(K)}, "positive semidefinite"),
        ({"baseline": np.zeros(K + 1)}, "shape"),
        ({"coupling": np.full((K, K), np.nan)}, "non-finite"),
    ],
)
def test_a_malformed_decision_is_refused(change, message):
    fields = {
        "coupling": COUPLING,
        "baseline": np.zeros(K),
        "target": np.zeros(K),
        "state_weight": Q,
        "action_weight": R,
        "lo": np.zeros(K),
        "hi": np.ones(K),
    }
    with pytest.raises(ValueError, match=message):
        ZoneDecision(**(fields | change))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"budget": -1.0}, "budget"),
        ({"uses": 0.0}, "uses"),
        ({"level": 1.0}, "level"),
        ({"draws": 99}, "draws"),
    ],
)
def test_a_malformed_request_is_refused(kwargs, message):
    arguments = {"budget": 1e-3} | kwargs
    budget = arguments.pop("budget")
    with pytest.raises(ValueError, match=message):
        design_experiment(
            _market([-1.0, -0.4, -1.3, -0.6]),
            ChannelPrior(B_TRUE, 0.01 * np.eye(K)),
            DEMO_EXPERIMENT,
            budget,
            **arguments,
        )


def test_an_experiment_needs_positive_costs_and_a_prior_its_size():
    with pytest.raises(ValueError, match="positive"):
        ZoneExperiment(unit_cost=np.zeros(K), noise_sd=NOISE_SD, probe=np.ones(K))
    with pytest.raises(ValueError, match="zones"):
        design_experiment(
            _market([-1.0, -0.4, -1.3, -0.6]),
            ChannelPrior(B_TRUE[:3], 0.01 * np.eye(3)),
            DEMO_EXPERIMENT,
            1e-3,
        )
