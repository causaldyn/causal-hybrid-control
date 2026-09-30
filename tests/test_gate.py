"""The deployment gate: its rules on deterministic batches, its error rates by simulation, and the
lab's eight-zone closed loop reproduced verdict for verdict.

The closed loop is the lab's synthetic market: the baseline acts ``N(0, 1)``, a zone's candidate
``N(m, s^2)``, and a decision's reward is Bernoulli with mean ``a_t + 0.6 g(u)``, where ``a_t`` is a
shock shared by every zone and ``g`` is a bump or a probit. Truths, chi-squares and total
variations are computed here in closed form or by quadrature, and the expected verdict counts were
produced by the lab's reference implementation on the same draws.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate, optimize, signal, stats

from chc.gate import (
    DecisionLog,
    DeploymentGate,
    DriftAlarm,
    GateConfig,
    GateMode,
    ZoneBatch,
    ZonePlan,
    channel_drift_evalues,
)

BATCH = 96
REWARD_SCALE = 0.6
LAB = GateConfig(
    delta=0.02,
    delta_harm=0.02,
    min_effect=0.02,
    horizon=150 * BATCH,
    alpha=0.10,
    alpha_harm=0.05,
    drift_arl=1e4,
    rho=0.5,
)


def _npdf(u: np.ndarray, m: float, s: float) -> np.ndarray:
    return np.exp(-((u - m) ** 2) / (2 * s * s)) / (math.sqrt(2 * math.pi) * s)


@dataclass(frozen=True)
class Truth:
    """A zone's candidate ``N(m, s^2)`` against the baseline ``N(0, 1)``, and its reward curve."""

    m: float
    s: float
    reward: str  # "bump": exp(-(u - 1)^2 / 2); "probit": Phi(u)

    def g(self, u: np.ndarray) -> np.ndarray:
        return np.exp(-((u - 1.0) ** 2) / 2) if self.reward == "bump" else stats.norm.cdf(u)

    @staticmethod
    def value(m: float, s: float, reward: str) -> float:
        v = 1 + s * s
        if reward == "bump":
            return math.exp(-((m - 1.0) ** 2) / (2 * v)) / math.sqrt(v)
        return float(stats.norm.cdf(m / math.sqrt(v)))

    @property
    def contrast(self) -> float:
        return REWARD_SCALE * (
            self.value(self.m, self.s, self.reward) - self.value(0.0, 1.0, self.reward)
        )

    @property
    def plan(self) -> ZonePlan:
        if self.s * self.s >= 2.0:
            chi2 = math.inf
        else:
            q = 1.0 / (self.s * self.s)
            chi2 = q / math.sqrt(2 * q - 1) * math.exp(self.m**2 / (2 - self.s**2)) - 1.0
        tv = (
            0.5
            * integrate.quad(
                lambda u: abs(_npdf(u, self.m, self.s) - _npdf(u, 0.0, 1.0)), -40.0, 40.0, limit=400
            )[0]
        )
        return ZonePlan(chi2, tv)

    def draw(
        self, rng: np.random.Generator, mode: GateMode, a: np.ndarray, rho: float
    ) -> ZoneBatch:
        """One batch logged under the policy ``mode`` asks for, with its logged propensities."""
        n = a.size
        if mode == "shadow":
            u = rng.normal(0.0, 1.0, n)
        elif mode == "deployed":
            u = rng.normal(self.m, self.s, n)
        else:
            pick = rng.random(n) < rho
            u = np.where(pick, rng.normal(self.m, self.s, n), rng.normal(0.0, 1.0, n))
        pn, pb = _npdf(u, self.m, self.s), _npdf(u, 0.0, 1.0)
        logged = {"shadow": pb, "deployed": pn, "experiment": (1 - rho) * pb + rho * pn}[mode]
        r = (rng.random(n) < a + REWARD_SCALE * self.g(u)).astype(float)
        return ZoneBatch(reward=r, candidate=pn, baseline=pb, logged=logged)


def _boundary(contrast: float, side: str) -> float:
    """The mean ``m`` of ``N(m, 1)`` whose bump contrast is exactly ``contrast``."""

    def gap(m: float) -> float:
        return Truth(m, 1.0, "bump").contrast - contrast

    return optimize.brentq(gap, -1.0, 1.0) if side == "low" else optimize.brentq(gap, 1.0, 3.0)


BOUNDARY = Truth(_boundary(LAB.delta, "low"), 1.0, "bump")  # contrast exactly delta, chi2 0.016
BOUNDARY_WIDE = Truth(_boundary(LAB.delta, "high"), 1.0, "bump")  # contrast delta, chi2 32.6
WIDE_NULL = Truth(1.0, 1.6, "bump")  # contrast -0.012, chi2 infinite
HARMFUL = Truth(-0.3, 1.0, "bump")  # contrast -0.052
BETTER = Truth(_boundary(LAB.delta + 0.03, "low"), 1.0, "bump")  # contrast 0.05
WIDE_BETTER = Truth(0.5, 1.6, "probit")  # contrast 0.063, chi2 infinite


def _closed_loop(truths: list[Truth], replications: int, seed: int) -> np.ndarray:
    """Verdicts, ``(replications, checks, zones)``, with each zone logged as its mode asks."""
    rng = np.random.default_rng(seed)
    names = [f"z{i}" for i in range(len(truths))]
    plans = {z: t.plan for z, t in zip(names, truths, strict=True)}
    checks = LAB.horizon // BATCH
    verdicts = np.empty((replications, checks, len(truths)), dtype=object)
    for rep in range(replications):
        gate = DeploymentGate(plans, LAB)
        state = np.zeros(1)
        for c in range(checks):
            eta, state = signal.lfilter(
                [1.0], [1.0, -0.995], 0.1 * rng.normal(size=BATCH), zi=state
            )
            a = 0.2 + 0.2 * np.tanh(eta)
            batch = {
                z: t.draw(rng, gate.mode(z), a, LAB.rho)
                for z, t in zip(names, truths, strict=True)
                if gate.mode(z) != "retired"
            }
            out = gate.update(batch)
            verdicts[rep, c] = [out[z] for z in names]
    return verdicts


def _paired_loop(
    truths: list[Truth], replications: int, seed: int, config: GateConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Verdicts as in :func:`_closed_loop`, and the decisions each zone logged under the mixture,
    ``(replications, zones)``. The shock and each zone draw from streams of their own, so two
    configs see the same draws in every zone whose modes agree."""
    names = [f"z{i}" for i in range(len(truths))]
    plans = {z: t.plan for z, t in zip(names, truths, strict=True)}
    checks = config.horizon // BATCH
    verdicts = np.empty((replications, checks, len(truths)), dtype=object)
    spent = np.zeros((replications, len(truths)), dtype=int)
    for rep in range(replications):
        shock = np.random.default_rng([seed, rep, 0])
        streams = [np.random.default_rng([seed, rep, 1 + i]) for i in range(len(truths))]
        gate = DeploymentGate(plans, config)
        state = np.zeros(1)
        for c in range(checks):
            eta, state = signal.lfilter(
                [1.0], [1.0, -0.995], 0.1 * shock.normal(size=BATCH), zi=state
            )
            a = 0.2 + 0.2 * np.tanh(eta)
            modes = [gate.mode(z) for z in names]
            spent[rep] += BATCH * (np.array(modes) == "experiment")
            batch = {
                z: t.draw(g, mode, a, config.rho)
                for z, t, g, mode in zip(names, truths, streams, modes, strict=True)
                if mode != "retired"
            }
            out = gate.update(batch)
            verdicts[rep, c] = [out[z] for z in names]
    return verdicts, spent


def _ones(n: int, *, candidate: float, baseline: float, logged: float, reward: float) -> ZoneBatch:
    one = np.ones(n)
    return ZoneBatch(
        reward=reward * one, candidate=candidate * one, baseline=baseline * one, logged=logged * one
    )


# Six decisions of a candidate that doubles the baseline's propensity on rewards of 1: one batch
# leaves a lone zone in shadow at alpha = 0.05, and two deploy it.
STRONG = _ones(6, candidate=2.0, baseline=1.0, logged=1.0, reward=1.0)
NEUTRAL = _ones(1, candidate=1.0, baseline=1.0, logged=1.0, reward=0.5)
LONE = GateConfig(delta=0.02, delta_harm=0.02, min_effect=0.02, horizon=10_000, drift_arl=100.0)


def test_the_lab_closed_loop_is_reproduced_verdict_for_verdict() -> None:
    """Twenty replications of each of the lab's two eight-zone worlds, 150 checks of 96
    decisions; every count below is the lab reference implementation's on the same draws."""
    null = _closed_loop(
        [BOUNDARY, BOUNDARY, BOUNDARY, BOUNDARY, BOUNDARY_WIDE, BOUNDARY_WIDE, WIDE_NULL, HARMFUL],
        20,
        9200,
    )
    assert not (null == "deploy").any()
    assert (null == "experiment").any(axis=1).sum(axis=0).tolist() == [0, 0, 0, 0, 20, 20, 20, 0]
    assert (null == "hold").any(axis=1).sum(axis=0).tolist() == [0, 0, 0, 0, 0, 0, 0, 20]

    mixed = _closed_loop(
        [BOUNDARY, BOUNDARY, BOUNDARY_WIDE, WIDE_NULL, BETTER, BETTER, BETTER, WIDE_BETTER],
        20,
        9201,
    )
    deployed = (mixed == "deploy").any(axis=1)
    first = np.where(deployed, (mixed == "deploy").argmax(axis=1), 0)
    assert deployed.sum(axis=0).tolist() == [0, 0, 0, 0, 20, 20, 20, 20]
    assert first.sum(axis=0).tolist() == [0, 0, 0, 0, 132, 144, 173, 350]
    assert (mixed == "experiment").any(axis=1).sum(axis=0).tolist() == [0, 0, 20, 20, 0, 0, 0, 20]
    assert not (mixed == "hold").any()
    assert not (mixed == "rollback").any()


def test_the_type_one_error_stays_below_alpha_when_read_after_every_decision() -> None:
    """A lone zone whose candidate is exactly ``delta`` better, read after each of 1000
    decisions, against a one-sided z-test of the same increments read as often: the gate deploys
    it on none of 100 paths, the z-test on 28."""
    config = GateConfig(delta=LAB.delta, delta_harm=0.02, min_effect=0.02, horizon=10**6)
    rng = np.random.default_rng(20260928)
    plan, paths, decisions = BOUNDARY.plan, 100, 1000
    deployed = z_deployed = 0
    for _ in range(paths):
        u = rng.normal(0.0, 1.0, decisions)
        reward = (rng.random(decisions) < 0.2 + REWARD_SCALE * BOUNDARY.g(u)).astype(float)
        candidate, baseline = _npdf(u, BOUNDARY.m, BOUNDARY.s), _npdf(u, 0.0, 1.0)
        gate = DeploymentGate({"z": plan}, config)
        for t in range(decisions):
            one = slice(t, t + 1)
            batch = ZoneBatch(reward[one], candidate[one], baseline[one], baseline[one])
            if gate.update({"z": batch})["z"] == "deploy":
                deployed += 1
                break
        x = (candidate / baseline - 1.0) * reward - config.delta
        n = np.arange(1, decisions + 1)
        mean = np.cumsum(x) / n
        sd = np.sqrt(np.maximum(np.cumsum(x * x) / n - mean**2, 1e-300))
        z_deployed += bool(np.any((mean / (sd / np.sqrt(n)))[29:] > 1.6448536269514722))
    assert deployed <= config.alpha * paths
    assert z_deployed > 4 * config.alpha * paths


def test_the_rules_apply_in_order() -> None:
    """DEPLOY on overwhelming evidence; a drift alarm rolls a deployed zone back and holds a
    shadow one; evidence against a deployed candidate rolls it back."""
    config = GateConfig(
        delta=0.02, delta_harm=0.02, min_effect=0.02, horizon=10_000, alpha=0.1, drift_arl=100.0
    )
    plans = {"a": ZonePlan(0.1, 0.1), "b": ZonePlan(0.1, 0.1), "c": ZonePlan(0.1, 0.1)}
    gate = DeploymentGate(plans, config)
    good = _ones(BATCH, candidate=2.0, baseline=1.0, logged=1.0, reward=1.0)
    calm = _ones(BATCH, candidate=1.0, baseline=1.0, logged=1.0, reward=0.5)
    assert gate.update({"a": good, "b": calm, "c": good}) == {
        "a": "deploy",
        "b": "shadow",
        "c": "deploy",
    }

    drifting = ZoneBatch(
        reward=calm.reward,
        candidate=calm.candidate,
        baseline=calm.baseline,
        logged=calm.logged,
        drift=np.full((BATCH, 2), 3.0),
    )
    worse = _ones(BATCH, candidate=1.0, baseline=2.0, logged=1.0, reward=1.0)
    verdicts = gate.update({"a": drifting, "b": drifting, "c": worse})
    assert verdicts == {"a": "rollback", "b": "hold", "c": "rollback"}
    assert (gate.mode("a"), gate.mode("b"), gate.mode("c")) == ("retired", "shadow", "retired")
    assert gate.update({"a": good, "b": calm}) == {"a": "hold", "b": "shadow", "c": "hold"}


def test_a_drift_alarm_discards_the_evidence_gathered_before_it() -> None:
    alarm = ZoneBatch(
        reward=NEUTRAL.reward,
        candidate=NEUTRAL.candidate,
        baseline=NEUTRAL.baseline,
        logged=NEUTRAL.logged,
        drift=np.full(1, 1e6),
    )
    steady = DeploymentGate({"z": ZonePlan(0.1, 0.1)}, LONE)
    assert [steady.update({"z": b})["z"] for b in (STRONG, NEUTRAL, STRONG)] == [
        "shadow",
        "shadow",
        "deploy",
    ]
    drifted = DeploymentGate({"z": ZonePlan(0.1, 0.1)}, LONE)
    assert [drifted.update({"z": b})["z"] for b in (STRONG, alarm, STRONG)] == [
        "shadow",
        "hold",
        "shadow",
    ]


def test_a_candidate_better_by_less_than_the_margin_is_neither_deployed_nor_held() -> None:
    """Per 100 decisions: three where only the candidate would have earned 1, three where only
    the baseline would have earned 0 with the candidate's action, 94 neutral. The candidate is
    0.03 better, short of ``delta = 0.05`` and not harmful; betting on ``r`` instead of ``1 - r``
    in the harm e-process would hold it."""
    config = GateConfig(delta=0.05, delta_harm=0.0, min_effect=0.02, horizon=10**6)
    reward = np.r_[np.ones(3), np.zeros(3), np.full(94, 0.5)]
    candidate = np.r_[np.full(3, 2.0), np.zeros(3), np.ones(94)]
    batch = ZoneBatch(
        reward=reward, candidate=candidate, baseline=np.ones(100), logged=np.ones(100)
    )
    gate = DeploymentGate({"z": ZonePlan(0.06, 0.03)}, config)
    assert {gate.update({"z": batch})["z"] for _ in range(100)} == {"shadow"}


def test_a_harmful_candidate_is_held_and_retired() -> None:
    harmful = _ones(BATCH, candidate=2.0, baseline=1.0, logged=1.0, reward=0.0)
    gate = DeploymentGate({"z": ZonePlan(0.1, 0.1)}, LONE)
    assert gate.update({"z": harmful}) == {"z": "hold"}
    assert gate.mode("z") == "retired"
    good = _ones(BATCH, candidate=2.0, baseline=1.0, logged=1.0, reward=1.0)
    assert gate.update({"z": good}) == {"z": "hold"}


def test_experiment_is_chosen_only_when_shadow_is_too_slow_and_the_mixture_faster() -> None:
    """At an exhausted horizon: a heavy-tailed candidate and a far one switch, a near one stays
    in shadow because the mixture would be slower still -- the rule on the budget alone sent
    99.8 % of the lab's slow nulls to EXPERIMENT."""
    plans = {
        "heavy": WIDE_NULL.plan,
        "far": BOUNDARY_WIDE.plan,
        "near": BOUNDARY.plan,
    }
    spent = DeploymentGate(
        plans, GateConfig(delta=0.02, delta_harm=0.02, min_effect=0.02, horizon=1)
    )
    assert spent.update({}) == {"heavy": "experiment", "far": "experiment", "near": "shadow"}
    ample = DeploymentGate(
        plans, GateConfig(delta=0.02, delta_harm=0.02, min_effect=0.02, horizon=10**9)
    )
    assert ample.update({}) == {"heavy": "experiment", "far": "shadow", "near": "shadow"}


FUTILE = dataclasses.replace(LAB, alpha_futility=0.1)


def test_the_futility_stop_ends_the_experiments_that_cannot_pay_and_nothing_else() -> None:
    """The lab's two worlds with and without the stop, on paired draws. The heavy-tailed null
    (contrast -0.012) leaves its experiment early, the boundary zones with chi2 32.6 (contrast
    exactly ``delta``) now and then, and every zone the stop does not reach reads the same verdicts
    in both runs: the better heavy-tailed zone is deployed on the same check."""
    for truths, seed, stopped, spent in (
        (
            [BOUNDARY] * 4 + [BOUNDARY_WIDE] * 2 + [WIDE_NULL, HARMFUL],
            9300,
            [4, 5, 6],
            ([0, 0, 0, 0, 286_080, 286_080, 286_080, 0], [0, 0, 0, 0, 243_840, 257_280, 35_232, 0]),
        ),
        (
            [BOUNDARY, BOUNDARY, BOUNDARY_WIDE, WIDE_NULL, BETTER, BETTER, BETTER, WIDE_BETTER],
            9301,
            [2, 3],
            ([0, 0, 286_080, 286_080, 0, 0, 0, 40_128], [0, 0, 274_272, 28_800, 0, 0, 0, 40_128]),
        ),
    ):
        without, spent_without = _paired_loop(truths, 20, seed, LAB)
        with_stop, spent_with = _paired_loop(truths, 20, seed, FUTILE)
        assert spent_without.sum(axis=0).tolist() == spent[0]
        assert spent_with.sum(axis=0).tolist() == spent[1]
        same = [bool((without[..., k] == with_stop[..., k]).all()) for k in range(len(truths))]
        assert [k for k, s in enumerate(same) if not s] == stopped
        assert not (with_stop[..., stopped] == "deploy").any()


def test_a_futility_stop_is_wrong_no_more_often_than_its_level_when_read_after_every_decision() -> (
    None
):
    """A lone zone whose candidate is exactly ``delta + min_effect`` better, in the experiment from
    the start, read after each of 2000 decisions: the stop ends it on 1 of 100 paths, where
    ``alpha_futility`` allows 10."""
    worth = FUTILE.delta + FUTILE.min_effect
    truth = Truth(_boundary(worth, "high"), 1.0, "bump")
    config = dataclasses.replace(FUTILE, horizon=1)
    rng = np.random.default_rng(20260930)
    stops = 0
    for _ in range(100):
        gate = DeploymentGate({"z": truth.plan}, config)
        assert gate.update({}) == {"z": "experiment"}
        batch = truth.draw(rng, "experiment", np.full(2000, 0.2), config.rho)
        for t in range(2000):
            one = slice(t, t + 1)
            verdict = gate.update(
                {
                    "z": ZoneBatch(
                        batch.reward[one],
                        batch.candidate[one],
                        batch.baseline[one],
                        batch.logged[one],
                    )
                }
            )["z"]
            if verdict != "experiment":
                stops += verdict == "shadow"
                break
    assert abs(truth.contrast - worth) < 1e-12
    assert stops == 1


@pytest.fixture
def futile() -> DeploymentGate:
    """A lone zone sent to the experiment by the rule on the horizon, whose candidate acts as the
    baseline does, so that every mode logs a propensity of 1. It cannot be ``delta + min_effect``
    better, and three batches in the experiment reject it."""
    config = GateConfig(
        delta=0.02,
        delta_harm=0.02,
        min_effect=0.02,
        horizon=1,
        drift_arl=100.0,
        alpha_futility=0.1,
    )
    gate = DeploymentGate({"z": WIDE_NULL.plan}, config)
    assert gate.update({}) == {"z": "experiment"}
    same = _ones(BATCH, candidate=1.0, baseline=1.0, logged=1.0, reward=0.5)
    verdicts = [gate.update({"z": same})["z"] for _ in range(3)]
    assert verdicts == ["experiment", "experiment", "shadow"]
    return gate


def test_a_futile_experiment_returns_to_shadow_and_stays_there(futile: DeploymentGate) -> None:
    """Back in shadow, the rule on the horizon would send the zone straight back. It stays, also
    once a batch of evidence for the candidate has pulled the futility e-value back under its
    threshold: a crossing rejects."""
    # Each of these decisions bets the futility e-value down, and no other e-value up.
    for_it = _ones(BATCH, candidate=0.0, baseline=1.0, logged=1.0, reward=0.0)
    assert [futile.update({"z": b})["z"] for b in (NEUTRAL, for_it, NEUTRAL)] == ["shadow"] * 3


def test_a_drift_alarm_lets_a_futile_zone_experiment_again(futile: DeploymentGate) -> None:
    """The alarm opens a new epoch for the futility evidence with the others, so a rejection made
    on the channel before it moved no longer stands."""
    alarm = dataclasses.replace(NEUTRAL, drift=np.full(1, 1e6))
    assert futile.update({"z": alarm}) == {"z": "hold"}
    assert futile.update({}) == {"z": "experiment"}


def test_a_candidate_better_by_delta_is_deployed_although_it_is_not_worth_min_effect(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Per 1000 decisions: 100 where only the candidate earns 1, 100 where it acts where the
    baseline earned 0 and does not, 800 neutral. The candidate is 0.1 better, above ``delta = 0``
    and below ``delta + min_effect = 0.2``. One batch rejects it for futility, as a gate that
    cannot select it logs, and selects it: DEPLOY, the claim ``V_new - V_base > delta``, comes
    first."""
    config = GateConfig(
        delta=0.0, delta_harm=0.0, min_effect=0.2, horizon=10**6, alpha_futility=0.1
    )
    reward = np.r_[np.ones(100), np.zeros(100), np.full(800, 0.5)]
    candidate = np.r_[np.full(100, 2.0), np.zeros(100), np.ones(800)]
    batch = ZoneBatch(
        reward=reward, candidate=candidate, baseline=np.ones(1000), logged=np.ones(1000)
    )
    unselectable = DeploymentGate(
        {"z": ZonePlan(0.2, 0.1)}, dataclasses.replace(config, alpha=1e-300)
    )
    with caplog.at_level(logging.INFO, logger="chc.gate"):
        assert unselectable.update({"z": batch}) == {"z": "shadow"}
    assert [getattr(r, "futile", None) for r in caplog.records if r.name == "chc.gate"] == [True]
    gate = DeploymentGate({"z": ZonePlan(0.2, 0.1)}, config)
    assert gate.update({"z": batch}) == {"z": "deploy"}


def test_an_unchanged_channel_alarms_no_more_often_than_drift_arl() -> None:
    """Likelihood-ratio e-values ``exp(theta z - theta^2/2)`` for two detectors on an unchanged
    channel, one decision per read; and the same detectors after a shift, which alarm fast."""
    rng = np.random.default_rng(7)
    theta = np.array([0.25, 0.5])
    reads, arl = 20_000, 50.0
    config = GateConfig(delta=0.02, delta_harm=0.02, min_effect=0.02, horizon=10**6, drift_arl=arl)

    def alarms(shift: float) -> int:
        gate = DeploymentGate({"z": ZonePlan(0.1, 0.1)}, config)
        z = rng.standard_normal((reads, 1)) + shift
        e = np.exp(theta * z - theta**2 / 2)
        count = 0
        for row in e:
            batch = ZoneBatch(
                reward=NEUTRAL.reward,
                candidate=NEUTRAL.candidate,
                baseline=NEUTRAL.baseline,
                logged=NEUTRAL.logged,
                drift=row[None, :],
            )
            count += gate.update({"z": batch})["z"] == "hold"
        return count

    assert 0 < alarms(0.0) <= reads / arl
    assert alarms(1.0) > 2 * reads / arl


@pytest.mark.parametrize(
    ("mode", "batch", "policy"),
    [
        ("shadow", _ones(4, candidate=2.0, baseline=1.0, logged=0.5, reward=1.0), "the baseline"),
        (
            "experiment",
            _ones(4, candidate=2.0, baseline=1.0, logged=2.0, reward=1.0),
            "the mixture 0.5 baseline",
        ),
        (
            "deployed",
            _ones(4, candidate=2.0, baseline=1.0, logged=1.0, reward=1.0),
            "the candidate",
        ),
    ],
)
def test_a_batch_logged_under_another_policy_is_refused(
    mode: GateMode, batch: ZoneBatch, policy: str
) -> None:
    """In the experiment, logging the propensity of the policy that drew the action rather than
    the mixture's is refused: it would leave the candidate's weight unbounded."""
    gate = DeploymentGate({"z": WIDE_NULL.plan}, LONE)
    if mode == "experiment":
        assert gate.update({}) == {"z": "experiment"}
    elif mode == "deployed":
        gate = DeploymentGate({"z": ZonePlan(0.1, 0.1)}, LONE)
        gate.update({"z": STRONG})
        assert gate.update({"z": STRONG}) == {"z": "deploy"}
    with pytest.raises(ValueError, match=f"must log under {policy}"):
        gate.update({"z": batch})


def test_a_refused_update_changes_no_zone() -> None:
    gate = DeploymentGate({"a": ZonePlan(0.1, 0.1), "b": ZonePlan(0.1, 0.1)}, LONE)
    wrong = _ones(1, candidate=1.0, baseline=1.0, logged=0.5, reward=0.5)
    with pytest.raises(ValueError, match="zone 'b'"):
        gate.update({"a": STRONG, "b": wrong})
    assert gate.update({"a": STRONG}) == {"a": "shadow", "b": "shadow"}


def test_a_refused_drift_batch_leaves_every_alarm_as_it_was() -> None:
    """Zone b's batch has the wrong number of detectors, and zone a's would have alarmed: the
    refusal comes first, so a's statistic is still 50 and the next e-value of 2 takes it to 102."""
    gate = DeploymentGate({"a": ZonePlan(0.1, 0.1), "b": ZonePlan(0.1, 0.1)}, LONE)

    def neutral(drift: list[list[float]]) -> ZoneBatch:
        return dataclasses.replace(NEUTRAL, drift=np.array(drift))

    gate.update({"a": neutral([[50.0]]), "b": neutral([[1.0, 1.0]])})
    with pytest.raises(ValueError, match="zone 'b': 1 drift detectors"):
        gate.update({"a": neutral([[10.0]]), "b": neutral([[1.0]])})
    assert gate.update({"a": neutral([[2.0]])})["a"] == "hold"


def test_inputs_outside_the_contract_are_refused() -> None:
    one = np.ones(3)
    with pytest.raises(ValueError, match="rewards must lie in"):
        ZoneBatch(reward=2 * one, candidate=one, baseline=one, logged=one)
    with pytest.raises(ValueError, match="entries for 3 rewards"):
        ZoneBatch(reward=one, candidate=np.ones(2), baseline=one, logged=one)
    with pytest.raises(ValueError, match="negative propensity"):
        ZoneBatch(reward=one, candidate=-one, baseline=one, logged=one)
    with pytest.raises(ValueError, match="logged propensity is 0"):
        ZoneBatch(reward=one, candidate=one, baseline=one, logged=np.zeros(3))
    with pytest.raises(ValueError, match="not finite"):
        ZoneBatch(reward=one, candidate=np.array([1.0, np.inf, 1.0]), baseline=one, logged=one)
    with pytest.raises(ValueError, match="drift must have shape"):
        ZoneBatch(reward=one, candidate=one, baseline=one, logged=one, drift=np.ones((2, 1)))
    with pytest.raises(ValueError, match="non-negative"):
        ZoneBatch(reward=one, candidate=one, baseline=one, logged=one, drift=-one)
    with pytest.raises(ValueError, match="saturated must hold 3 booleans"):
        DecisionLog(action=one, propensity=one, saturated=np.zeros(3))
    with pytest.raises(ValueError, match=r"action must have shape \(3,\)"):
        DecisionLog(action=np.ones(2), propensity=one, saturated=np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="chi2"):
        ZonePlan(-1.0, 0.1)
    with pytest.raises(ValueError, match="tv"):
        ZonePlan(0.1, 1.5)
    for bad in (
        {"delta": -0.1},
        {"alpha": 1.0},
        {"rho": 0.0},
        {"min_effect": 0.0},
        {"horizon": 0},
        {"drift_arl": 1.0},
    ):
        with pytest.raises(ValueError, match=next(iter(bad))):
            GateConfig(
                **({"delta": 0.02, "delta_harm": 0.02, "min_effect": 0.02, "horizon": 10} | bad)
            )
    with pytest.raises(ValueError, match="at least one zone"):
        DeploymentGate({}, LONE)
    gate = DeploymentGate({"z": ZonePlan(0.1, 0.1)}, LONE)
    with pytest.raises(KeyError, match="no plan for zones"):
        gate.update({"y": NEUTRAL})
    two = ZoneBatch(
        reward=np.full(2, 0.5), candidate=np.ones(2), baseline=np.ones(2), logged=np.ones(2)
    )
    gate.update({"z": ZoneBatch(two.reward, two.candidate, two.baseline, two.logged, np.ones(2))})
    with pytest.raises(ValueError, match="drift detectors"):
        gate.update(
            {"z": ZoneBatch(two.reward, two.candidate, two.baseline, two.logged, np.ones((2, 3)))}
        )


# ---- what a logged decision records (ADR 0015) ----


def _logged(size: int, seed: int, *, dither: bool = True) -> DecisionLog:
    """The baseline draws each action as its mean plus a standard normal dither, so the logged
    propensity is the dither's density."""
    rng = np.random.default_rng(seed)
    xi = rng.normal(size=size)
    return DecisionLog(
        action=0.2 * rng.normal(size=size) + xi,
        propensity=_npdf(xi, 0.0, 1.0),
        saturated=np.zeros(size, dtype=bool),
        dither=xi if dither else None,
    )


def test_a_batch_read_from_stored_records_is_the_batch_built_from_arrays() -> None:
    log = _logged(BATCH, 7)
    stored = json.loads(json.dumps(log.to_records()))
    assert {record["decision_log_version"] for record in stored} == {DecisionLog.VERSION}
    read = DecisionLog.from_records(stored)
    reward = np.random.default_rng(8).uniform(size=BATCH)
    candidate = _npdf(read.action, 0.3, 1.0)
    baseline = read.propensity
    from_log = ZoneBatch.from_log(read, reward=reward, candidate=candidate, baseline=baseline)
    from_arrays = ZoneBatch(reward, candidate, log.propensity, log.propensity)
    for name in ("reward", "candidate", "baseline", "logged"):
        np.testing.assert_array_equal(getattr(from_log, name), getattr(from_arrays, name))
    gates = [DeploymentGate({"z": ZonePlan(0.1, 0.1)}, LONE) for _ in range(2)]
    assert gates[0].update({"z": from_log}) == gates[1].update({"z": from_arrays})


@settings(max_examples=60, deadline=None)
@given(
    size=st.integers(0, 12),
    width=st.sampled_from([None, 1, 3]),
    dither=st.booleans(),
    seed=st.integers(0, 2**31 - 1),
)
def test_a_log_survives_its_stored_form_bit_for_bit(
    size: int, width: int | None, dither: bool, seed: int
) -> None:
    """Through JSON and back, over actions from 1e-300 to 1e300. An empty log cannot say whether it
    would have carried a dither, and reads back without one."""
    rng = np.random.default_rng(seed)
    shape = (size,) if width is None else (size, width)
    log = DecisionLog(
        action=rng.normal(size=shape) * 10.0 ** rng.integers(-300, 300, size=shape),
        propensity=np.exp(20.0 * rng.normal(size=size)),
        saturated=rng.uniform(size=size) < 0.3,
        dither=rng.normal(size=shape) if dither else None,
    )
    read = DecisionLog.from_records(json.loads(json.dumps(log.to_records())))
    np.testing.assert_array_equal(read.propensity, log.propensity)
    np.testing.assert_array_equal(read.saturated, log.saturated)
    np.testing.assert_array_equal(read.action.reshape(shape), log.action)
    if log.dither is None or size == 0:
        assert read.dither is None
    else:
        assert read.dither is not None
        np.testing.assert_array_equal(read.dither.reshape(shape), log.dither)


RECORD = {"decision_log_version": 1, "action": 0.4, "propensity": 0.35, "saturated": False}


def _without(key: str) -> dict[str, object]:
    return {k: v for k, v in RECORD.items() if k != key}


@pytest.mark.parametrize(
    ("records", "match"),
    [
        ([_without("decision_log_version")], "1 of 1 records have no 'decision_log_version'"),
        ([RECORD | {"decision_log_version": 2}], "decision_log_version 2;"),
        ([RECORD | {"decision_log_version": True}], "decision_log_version True;"),
        ([RECORD | {"decision_log_version": "1"}], "decision_log_version '1';"),
        ([RECORD, _without("propensity")], r"no 'propensity' \(the first is record 1\)"),
        ([_without("action")], "no 'action'"),
        ([_without("saturated")], "no 'saturated'"),
        ([RECORD | {"saturated": 0}], "true or false"),
        ([RECORD | {"dither": 0.1}, RECORD], "1 of 2 records carry a dither"),
        ([RECORD | {"propensity": 0.0}], "must be positive"),
        ([RECORD | {"dither": [0.1, 0.2]}], "dither has shape"),
    ],
)
def test_a_stored_record_is_refused_rather_than_filled_in(
    records: list[dict[str, object]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        DecisionLog.from_records(records)


def test_keys_the_log_does_not_define_are_left_to_the_caller() -> None:
    read = DecisionLog.from_records([RECORD | {"version": "policy-7", "reward": 0.9}])
    assert read.propensity.tolist() == [0.35]
    assert read.dither is None


def test_a_reader_of_the_draw_refuses_a_clipped_dither() -> None:
    log = _logged(5, 3)
    np.testing.assert_array_equal(log.dither_draws(), log.dither)
    clipped = dataclasses.replace(log, saturated=np.array([False, False, True, False, False]))
    clipped_draw = r"1 of 5 decisions were clipped \(the first is decision 2\)"
    with pytest.raises(ValueError, match=clipped_draw):
        clipped.dither_draws()
    with pytest.raises(ValueError, match="records no dither"):
        _logged(5, 3, dither=False).dither_draws()


# ---- the channel-drift monitor (ADR 0018) ----


def _dithered(
    action: np.ndarray, dither: np.ndarray, saturated: np.ndarray | None = None
) -> DecisionLog:
    size = action.shape[0]
    return DecisionLog(
        action=action,
        propensity=np.ones(size),
        saturated=np.zeros(size, dtype=bool) if saturated is None else saturated,
        dither=dither,
    )


def test_each_drift_evalue_averages_what_the_dither_identity_says() -> None:
    """Two states and two actions, each entry of the channel off the model's by its own amount,
    with its own radius, dither scale and residual scale, and a residual that also carries the other
    action's dither. Column by column, the mean is ``1 / |1 - theta k|``: at most 1 on each side an
    entry has not crossed, above 1 on the side it has."""
    rng = np.random.default_rng(20)
    size = 50_000
    sigma, scale = np.array([0.5, 2.0]), np.array([1.0, 0.25])
    radius = np.array([[0.1, 0.0], [0.05, 0.2]])
    moved = np.array([[0.05, -0.03], [0.1, 0.0]])  # the plant's channel less the model's
    dither = sigma * rng.standard_normal((size, 2))
    action = 0.5 * np.sin(np.arange(size)[:, None] / 7.0 + np.array([0.0, 1.0])) + dither
    residual = np.array([0.2, -0.05]) + action @ moved.T + 0.3 * scale * rng.normal(size=(size, 2))
    evalues = channel_drift_evalues(
        _dithered(action, dither), residual, dither_scale=sigma, radius=radius, residual_scale=scale
    ).reshape(size, 2, 2, 2, 8)
    side = np.array([1.0, -1.0])
    k = (moved[..., None] - side * radius[..., None]) * sigma[:, None] / scale[:, None, None]
    theta = side[:, None] * 2.0 ** -np.arange(8)
    expected = 1.0 / np.abs(1.0 - theta * k[..., None])
    z = (evalues.mean(axis=0) - expected) / (evalues.std(axis=0) / math.sqrt(size))
    assert np.abs(z).max() < 5.0
    crossed = expected > 1.0
    assert crossed.sum() == 16
    assert crossed[0, 1, 1].all()  # shrank past a radius of 0
    assert crossed[1, 0, 0].all()  # grew past its radius


def test_a_decision_reads_as_the_docstring_writes_it() -> None:
    """Two decisions, two states and one action, against the formula transcribed entry by entry:
    the residual moved to the edge by the action as applied, and the columns in the documented
    order."""
    action, drawn = np.array([0.7, -0.4]), np.array([0.2, -0.6])
    residual = np.array([[0.3, -1.1], [0.05, 0.4]])
    radius, sigma, scale = np.array([[0.1], [0.25]]), 0.5, np.array([0.8, 2.0])
    evalues = channel_drift_evalues(
        _dithered(action, drawn), residual, dither_scale=sigma, radius=radius, residual_scale=scale
    ).reshape(2, 2, 1, 2, 8)
    for t in range(2):
        for i in range(2):
            for s, side in enumerate((1.0, -1.0)):
                r = (residual[t, i] - side * radius[i, 0] * action[t]) / scale[i]
                for b in range(8):
                    theta = side * 2.0**-b
                    expected = math.exp(theta * r * drawn[t] / sigma - (theta * r) ** 2 / 2)
                    assert evalues[t, i, 0, s, b] == pytest.approx(expected, rel=1e-13)


def test_a_clipped_decision_is_read_with_its_draw() -> None:
    """A box clips ``u = p + sigma xi``, so the residual moves with what the clip left of the
    dither while the bet reads the draw. Without noise ``c`` is known decision by decision, and each
    column averages ``1 - (Phi(u2) - Phi(u1)) (1 - 1 / s)`` (``validation/dither_drift_evalue.mac``
    STEP 8), with the nominal action inside the box, on a bound and beyond one: exactly 1 on the
    radius's edge, at most 1 inside it and at least 1 past it. The flag is not read."""
    rng = np.random.default_rng(27)
    size, sigma, low, high = 70_000, 0.4, -0.6, 0.6
    nominal = np.resize([-0.9, -0.6, -0.3, 0.0, 0.45, 0.6, 1.0], size)
    drawn = sigma * rng.standard_normal(size)
    action = np.clip(nominal + drawn, low, high)
    scale, radius, offset = np.array([0.5, 1.0]), np.array([[0.1], [0.1]]), np.array([0.2, -0.3])
    moved = np.array([0.1, 0.3])  # state 0 on its radius's growth edge, state 1 past it
    residual = offset + action[:, None] * moved
    log = _dithered(action, drawn, action != nominal + drawn)
    common = {"dither_scale": sigma, "radius": radius, "residual_scale": scale}
    evalues = channel_drift_evalues(log, residual, **common)
    unflagged = dataclasses.replace(log, saturated=np.zeros(size, dtype=bool))
    np.testing.assert_array_equal(channel_drift_evalues(unflagged, residual, **common), evalues)
    side = np.array([1.0, -1.0])
    edge = moved[:, None] - side * radius  # (states, sides)
    theta = side[:, None] * 2.0 ** -np.arange(8)  # (sides, bets)
    c = (offset[:, None] + edge * nominal[:, None, None]) / scale[:, None]
    s = 1.0 - theta * (edge * sigma / scale[:, None])[..., None]
    below, above = (nominal - low) / sigma, (high - nominal) / sigma
    tc = theta * c[..., None]
    p1 = stats.norm.cdf(-s * below[:, None, None, None] - tc)
    p2 = stats.norm.cdf(s * above[:, None, None, None] - tc)
    expected = 1.0 - (p2 - p1) * (1.0 - 1.0 / s)  # (decisions, states, sides, bets)
    gap = evalues.reshape(size, 2, 2, 8) - expected
    z = gap.mean(axis=0) / (gap.std(axis=0) / math.sqrt(size))
    assert np.abs(z).max() < 5.0
    assert 0.3 < log.saturated.mean() < 0.7
    np.testing.assert_allclose(expected[:, 0, 0], 1.0, rtol=1e-12)  # the edge, whatever the clip
    assert (expected[:, :, 1] <= 1.0).all()
    assert (expected[:, 1, 0] >= 1.0).all()
    assert expected[:, 1, 0].mean() > 1.001


def _lab_plant(
    paths: int,
    channel: np.ndarray,
    dither: float,
    rng: np.random.Generator,
    box: float = math.inf,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The lab's plant ``x' = 0.8 x + 0.15 + b_t u + eps``, ``eps`` Laplace with standard
    deviation 0.5, under ``u = -0.5 x + dither xi`` clipped to ``[-box, box]``, and the residual
    against a model that gets the drift wrong, ``x' = 0.9 x + 0.05 + u``: per decision and path,
    the action as applied, the dither as drawn, the residual, and whether the box clipped."""
    x = rng.normal(0.0, 0.7, paths)
    action, drawn, residual = (np.empty((channel.size, paths)) for _ in range(3))
    saturated = np.empty((channel.size, paths), dtype=bool)
    for t, b in enumerate(channel):
        drawn[t] = dither * rng.standard_normal(paths)
        wanted = -0.5 * x + drawn[t]
        action[t] = np.clip(wanted, -box, box)
        saturated[t] = action[t] != wanted
        after = 0.8 * x + 0.15 + b * action[t] + rng.laplace(scale=0.5 / math.sqrt(2), size=paths)
        residual[t] = after - (0.9 * x + 0.05 + action[t])
        x = after
    return action, drawn, residual, saturated


def _alarms_by(
    plant: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ends: tuple[int, ...],
    *,
    dither: float,
    radius: float,
    arl: float,
) -> np.ndarray:
    """Per path, whether the alarm sounded before each end, the alarm fed the rows up to each end
    in one update."""
    action, drawn, residual, saturated = plant
    sounded = np.zeros((len(ends), action.shape[1]), dtype=bool)
    for p in range(action.shape[1]):
        evalues = channel_drift_evalues(
            _dithered(action[:, p], drawn[:, p], saturated[:, p]),
            residual[:, p],
            dither_scale=dither,
            radius=radius,
            residual_scale=0.5,
        )
        alarm, start = DriftAlarm(arl), 0
        for i, end in enumerate(ends):
            sounded[i:, p] |= alarm.update(evalues[start:end])
            start = end
    return sounded


def test_a_channel_inside_its_radius_alarms_by_h_no_more_often_than_h_over_arl() -> None:
    """``R_t - t`` is a supermartingale, so ``P(alarm by H) <= H / A``. The model's drift is wrong,
    the noise Laplace and the channel 0.3 from the model's; with a radius of 0.35 the bound holds,
    and with none the alarm sounds on almost every path."""
    rng = np.random.default_rng(21)
    horizon, arl = 300, 1000.0
    plant = _lab_plant(200, np.full(horizon, 1.3), 1.0, rng)
    covered = _alarms_by(plant, (horizon,), dither=1.0, radius=0.35, arl=arl)
    naive = _alarms_by(plant, (horizon,), dither=1.0, radius=0.0, arl=arl)
    assert covered.mean() <= horizon / arl
    assert naive.mean() > 3 * horizon / arl


def test_a_channel_that_moves_past_its_radius_is_caught_after_the_move() -> None:
    rng = np.random.default_rng(22)
    move, arl = 200, 1000.0
    channel = np.where(np.arange(move + 1000) < move, 1.07, 1.5)
    sounded = _alarms_by(
        _lab_plant(100, channel, 0.3, rng), (move, channel.size), dither=0.3, radius=0.1, arl=arl
    )
    assert sounded[0].mean() <= move / arl
    assert sounded[1].mean() >= 0.95


def test_a_boxed_plant_dithered_throughout_is_watched_as_an_unboxed_one() -> None:
    """The same move, with the lab's actions boxed to ``+-0.5`` and every decision dithered, so
    that a quarter of them clip: before the move the alarm sounds on no more than ``H / A`` of the
    paths, and after it on nearly every one."""
    rng = np.random.default_rng(28)
    move, arl = 200, 1000.0
    channel = np.where(np.arange(move + 1000) < move, 1.07, 1.5)
    plant = _lab_plant(100, channel, 0.3, rng, box=0.5)
    assert 0.1 < plant[3].mean() < 0.4
    sounded = _alarms_by(plant, (move, channel.size), dither=0.3, radius=0.1, arl=arl)
    assert sounded[0].mean() <= move / arl
    assert sounded[1].mean() >= 0.95


def test_the_alarm_is_shiryaev_roberts_averaged_over_detectors_and_starts_again() -> None:
    alarm = DriftAlarm(6.0)
    assert not alarm.update(np.ones((0, 5)))
    assert (alarm.detectors, alarm.statistic) == (None, 0.0)  # no decision fixes no count
    assert not alarm.update(np.array([[2.0, 0.5]]))
    assert (alarm.detectors, alarm.statistic) == (2, 1.25)
    assert not alarm.update(np.array([[3.0, 1.0]]))
    assert alarm.statistic == 5.25  # ((2 + 1) 3 + (0.5 + 1) 1) / 2
    assert alarm.update(np.array([[4.0, 4.0], [1e6, 1e6]]))
    assert (alarm.detectors, alarm.statistic) == (None, 0.0)  # the second row was dropped
    assert not alarm.update(np.ones((1, 3)))
    assert alarm.statistic == 1.0


@settings(max_examples=60, deadline=None)
@given(
    rows=st.integers(0, 30),
    cut=st.floats(0.0, 1.0),
    detectors=st.integers(1, 4),
    seed=st.integers(0, 2**31 - 1),
)
def test_the_alarm_reads_the_same_whatever_the_batches(
    rows: int, cut: float, detectors: int, seed: int
) -> None:
    evalues = np.random.default_rng(seed).uniform(0.0, 1.5, size=(rows, detectors))
    whole, split = DriftAlarm(1e12), DriftAlarm(1e12)
    whole.update(evalues)
    at = round(cut * rows)
    split.update(evalues[:at])
    split.update(evalues[at:])
    assert split.statistic == pytest.approx(whole.statistic, rel=1e-12)


def test_a_dither_that_is_not_the_stated_draw_is_refused() -> None:
    rng = np.random.default_rng(23)
    drawn = 0.3 * rng.standard_normal(40)
    log = _dithered(0.1 + drawn, drawn)
    residual = rng.normal(size=40)
    common = {"radius": 0.1, "residual_scale": 0.5}
    channel_drift_evalues(log, residual, dither_scale=0.3, **common)
    with pytest.raises(ValueError, match=r"mean square 1\d\.\d+ over 40 decisions"):
        channel_drift_evalues(log, residual, dither_scale=0.09, **common)
    with pytest.raises(ValueError, match="dither of action 0"):
        channel_drift_evalues(log, residual, dither_scale=3.0, **common)
    with pytest.raises(ValueError, match="records no dither"):
        channel_drift_evalues(
            dataclasses.replace(log, dither=None), residual, dither_scale=0.3, **common
        )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"residual": np.ones(39)}, r"residual must have shape \(40,\) or \(40, states\)"),
        ({"residual": np.full(40, np.nan)}, "residual is not finite"),
        ({"radius": -0.1}, "radius must be non-negative"),
        ({"radius": np.ones(2)}, r"radius must be a scalar or of shape \(1, 1\)"),
        ({"residual_scale": 0.0}, "residual_scale must be positive"),
        ({"dither_scale": -0.3}, "dither_scale must be positive"),
        ({"dither_scale": np.inf}, "dither_scale is not finite"),
    ],
)
def test_drift_evalues_outside_the_contract_are_refused(
    change: dict[str, object], match: str
) -> None:
    drawn = 0.3 * np.random.default_rng(24).standard_normal(40)
    arguments: dict[str, object] = {
        "residual": np.zeros(40),
        "dither_scale": 0.3,
        "radius": 0.1,
        "residual_scale": 0.5,
    } | change
    residual = arguments.pop("residual")
    with pytest.raises(ValueError, match=match):
        channel_drift_evalues(_dithered(drawn, drawn), residual, **arguments)  # type: ignore[arg-type]


def test_an_alarm_outside_the_contract_is_refused() -> None:
    with pytest.raises(ValueError, match="arl must exceed 1"):
        DriftAlarm(1.0)
    alarm = DriftAlarm(10.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        alarm.update(-np.ones(3))
    alarm.update(np.ones((2, 2)))
    with pytest.raises(ValueError, match="3 drift detectors, where the running statistic has 2"):
        alarm.update(np.ones((1, 3)))
    assert alarm.statistic == 2.0


def test_a_gate_fed_the_dither_evalues_holds_a_zone_whose_channel_moved() -> None:
    """A zone in shadow whose channel is 1.0 at first and 2.0 from the fourth batch: its evidence
    restarts, and it reads HOLD, once the dither shows the move. Before it, a false alarm has
    probability at most 288 / 10 000."""
    rng = np.random.default_rng(25)
    gate = DeploymentGate({"z": ZonePlan(0.1, 0.1)}, dataclasses.replace(LONE, drift_arl=1e4))
    x, verdicts = 0.0, []
    for batch in range(8):
        drawn = rng.standard_normal(BATCH)
        action = np.empty(BATCH)
        residual = np.empty(BATCH)
        for t in range(BATCH):
            action[t] = -0.5 * x + drawn[t]
            after = 0.5 * x + (1.0 if batch < 3 else 2.0) * action[t] + 0.5 * rng.standard_normal()
            residual[t] = after - (0.5 * x + action[t])
            x = after
        log = DecisionLog(action, _npdf(drawn, 0.0, 1.0), np.zeros(BATCH, dtype=bool), drawn)
        drift = channel_drift_evalues(
            log, residual, dither_scale=1.0, radius=0.1, residual_scale=0.5
        )
        verdicts.append(
            gate.update(
                {
                    "z": ZoneBatch.from_log(
                        log,
                        reward=np.full(BATCH, 0.5),
                        candidate=_npdf(drawn, 0.1, 1.0),
                        baseline=log.propensity,
                        drift=drift,
                    )
                }
            )["z"]
        )
    assert verdicts[:3] == ["shadow"] * 3
    assert "hold" in verdicts[3:]
