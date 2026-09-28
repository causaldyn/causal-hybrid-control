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
import math
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate, optimize, signal, stats

from chc.gate import DecisionLog, DeploymentGate, GateConfig, GateMode, ZoneBatch, ZonePlan

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
