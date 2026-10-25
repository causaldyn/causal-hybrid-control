"""The deployment gate in blocks, on plants with memory: per-decision increments certify the myopic
contrast there, and a block experiment on policies, read after a burn-in, certifies the sustained
one.

Two plants. The lab's: ``x' = 0.9 x + 0.1 u + 0.2 eps``, a reward that is Bernoulli with mean
``0.3 exp(-(u - 1)^2 / 2) + 0.7 exp(-x^2 / 2)``, the baseline acting ``N(0, 1)`` and the candidate
``N(0.4, 1)``: +0.029 better myopically, -0.0096 worse in the long run. And one with carryover of
order 1: the candidate acts 1 and the baseline 0, and the reward is Bernoulli with mean
``0.5 + gain u_t - cost u_(t-1)``, so the myopic contrast is ``gain`` and the sustained one
``gain - cost``. The carryover check and the centred increments read further plants: carryover
of order ``p`` on a reward with Gaussian noise, and blocks of one period whose read means are
drawn directly.

Every draw comes from a stream whose shape the gate's modes do not change, so two configs on one
seed see the same draws in every zone until their verdicts part.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numpy as np
import pytest

from chc.gate import BlockBatch, DeploymentGate, GateConfig, ZoneBatch, ZonePlan, _Level

# Read by no rule in blocks; the per-decision gate on the lab's plant reads it.
PLAN = ZonePlan(chi2=math.exp(0.4**2) - 1.0, tv=0.1585)

Period = Callable[[np.random.Generator, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _lab(rng: np.random.Generator, x: np.ndarray, candidate: np.ndarray) -> tuple:
    u = np.where(candidate, 0.4, 0.0) + rng.normal(size=x.shape)
    mean = 0.3 * np.exp(-((u - 1.0) ** 2) / 2.0) + 0.7 * np.exp(-(x**2) / 2.0)
    reward = (rng.random(x.shape) < mean).astype(float)
    return reward, 0.9 * x + 0.1 * u + 0.2 * rng.normal(size=x.shape)


def _order_one(gain: float, cost: float) -> Period:
    def period(rng: np.random.Generator, last: np.ndarray, candidate: np.ndarray) -> tuple:
        u = candidate.astype(float)
        return (rng.random(last.shape) < 0.5 + gain * u - cost * last).astype(float), u

    return period


AT_THE_MARGIN = _order_one(0.45, 0.45)  # myopic +0.45, sustained 0
BETTER = _order_one(0.4, 0.1)  # sustained +0.3


def _order(p: int, gain: float, cost: float) -> Period:
    """Carryover of order ``p``: the candidate acts 1 and the baseline 0, and the reward is
    ``clip(0.5 + gain u_t - cost u_(t-p) + 0.2 eta_t, 0, 1)``; the state is the last ``p``
    actions, oldest first."""

    def period(rng: np.random.Generator, past: np.ndarray, candidate: np.ndarray) -> tuple:
        u = candidate.astype(float)
        noise = 0.2 * rng.standard_normal(u.shape)
        reward = np.clip(0.5 + gain * u - cost * past[:, 0] + noise, 0.0, 1.0)
        return reward, np.concatenate([past[:, 1:], u[:, None]], axis=1)

    return period


def _slack(paths: int, level: float = 0.05) -> float:
    """A share of paths that ``level`` bounds, plus three binomial standard deviations."""
    return level + 3.0 * math.sqrt(level * (1.0 - level) / paths)


class _Warnings(logging.Handler):
    """The zones a gate's update warned of carryover in."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.zones: set[str] = set()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "chc_event", None) == "gate_carryover":
            self.zones.add(record.zone)  # type: ignore[attr-defined]


def _config(burn_in: int, block_length: int, **changes: float) -> GateConfig:
    settings = {"delta": 0.0, "delta_harm": 0.02, "min_effect": 0.02, "horizon": 10**6}
    return GateConfig(**(settings | changes), block_length=block_length, burn_in=burn_in)  # type: ignore[arg-type]


def _run(
    periods: list[Period],
    config: GateConfig,
    *,
    paths: int,
    blocks: int,
    per_read: int,
    seed: int,
    start: Callable[[np.random.Generator, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whether each path's gate ever deployed each zone, each zone's mode at the end, and whether
    the gate warned of carryover in it, all ``(paths, zones)``: one gate per path, every zone
    logged as its mode asks."""
    assert config.block_length is not None
    rng = np.random.default_rng(seed)
    names = [f"z{k}" for k in range(len(periods))]
    gates = [DeploymentGate(dict.fromkeys(names, PLAN), config) for _ in range(paths)]
    states = [np.zeros(paths) if start is None else start(rng, paths) for _ in periods]
    deployed = np.zeros((paths, len(periods)), dtype=bool)
    warned = np.zeros((paths, len(periods)), dtype=bool)
    handler = _Warnings()
    logger = logging.getLogger("chc.gate")
    logger.addHandler(handler)
    length = config.block_length
    for _ in range(blocks // per_read):
        batches: list[dict[str, BlockBatch]] = [{} for _ in range(paths)]
        for k, (name, period) in enumerate(zip(names, periods, strict=True)):
            modes = np.array([gate.mode(name) for gate in gates])[:, None]
            coin = rng.random((paths, per_read)) < config.rho
            experiment = modes == "experiment"
            ran = np.where(experiment, coin, modes == "deployed")
            logged = np.where(experiment, np.where(ran, config.rho, 1.0 - config.rho), 1.0)
            reward = np.empty((paths, per_read, length))
            for b in range(per_read):
                for t in range(length):
                    reward[:, b, t], states[k] = period(rng, states[k], ran[:, b])
            for i in range(paths):
                if modes[i, 0] != "retired":
                    batches[i][name] = BlockBatch(reward[i], ran[i], logged[i])
        for i, gate in enumerate(gates):
            handler.zones.clear()
            verdicts = gate.update(batches[i])
            deployed[i] |= [verdicts[name] == "deploy" for name in names]
            warned[i] |= [name in handler.zones for name in names]
    logger.removeHandler(handler)
    modes = np.array([[gate.mode(name) for name in names] for gate in gates])
    return deployed, modes, warned


def _per_decision_lab(paths: int, decisions: int, batch: int, seed: int) -> float:
    """The gate as it runs without blocks on the lab's plant: the share of paths it deploys on."""
    config = GateConfig(delta=0.0, delta_harm=0.02, min_effect=0.02, horizon=decisions)
    rng = np.random.default_rng(seed)
    deployed = 0
    for _ in range(paths):
        gate = DeploymentGate({"z": PLAN}, config)
        x = rng.normal(0.0, 0.51, 1)
        for _ in range(decisions // batch):
            mode = gate.mode("z")
            u, reward = np.empty(batch), np.empty(batch)
            for t in range(batch):
                pick = mode == "deployed" or (mode == "experiment" and rng.random() < config.rho)
                u[t] = (0.4 if pick else 0.0) + rng.normal()
                mean = 0.3 * math.exp(-((u[t] - 1.0) ** 2) / 2.0) + 0.7 * math.exp(-(x[0] ** 2) / 2)
                reward[t] = float(rng.random() < mean)
                x = 0.9 * x + 0.1 * u[t] + 0.2 * rng.normal(size=1)
            candidate = np.exp(-((u - 0.4) ** 2) / 2.0) / math.sqrt(2.0 * math.pi)
            baseline = np.exp(-(u**2) / 2.0) / math.sqrt(2.0 * math.pi)
            logged = {
                "shadow": baseline,
                "experiment": (1.0 - config.rho) * baseline + config.rho * candidate,
                "deployed": candidate,
            }[mode]
            if gate.update({"z": ZoneBatch(reward, candidate, baseline, logged)})["z"] == "deploy":
                deployed += 1
                break
    return deployed / paths


def test_in_blocks_the_lab_candidate_worse_in_the_long_run_is_not_deployed() -> None:
    """On the lab's plant the gate without blocks deploys the candidate on every path within 20 000
    decisions; in blocks of 50 periods, 40 of them burned, read every 5 blocks over the same 20 000
    periods, on at most ``alpha`` of 1000 paths."""
    assert _per_decision_lab(paths=40, decisions=20_000, batch=96, seed=20261012) == 1.0
    config = _config(burn_in=40, block_length=50)
    deployed, _, _ = _run(
        [_lab],
        config,
        paths=1000,
        blocks=400,
        per_read=5,
        seed=20261013,
        start=lambda rng, n: rng.normal(0.0, 0.51, n),
    )
    assert deployed.mean() <= config.alpha


def test_in_blocks_a_candidate_at_the_margin_under_carryover_is_deployed_at_most_alpha() -> None:
    """Carryover of order 1, a candidate +0.45 better myopically and exactly as good sustained, read
    after every block of 2 periods: with the first period burned, the gate deploys it on at most
    ``alpha`` of 300 paths, and the carryover check, whose null then holds, warns on at most its
    level of them; with none burned, the read is half myopic, and it deploys on most."""
    checked = _config(1, 2, check_burn_in=True)
    burned, _, warned = _run(
        [AT_THE_MARGIN], checked, paths=300, blocks=400, per_read=1, seed=20261014
    )
    assert burned.mean() <= 0.05
    assert warned.mean() <= _slack(300)
    unburned, _, _ = _run(
        [AT_THE_MARGIN], _config(0, 2), paths=300, blocks=400, per_read=1, seed=20261014
    )
    assert unburned.mean() >= 0.5


def test_in_blocks_e_bh_keeps_the_false_deploy_rate_across_zones() -> None:
    """Three zones at the margin under carryover and three better by 0.3 sustained, 200 paths of
    600 blocks read every 10, ``alpha = 0.1``: the false-discovery rate at the end is at most
    ``alpha`` and the better zones are deployed; with no burn-in, half of the deploys are false."""
    zones = [AT_THE_MARGIN] * 3 + [BETTER] * 3

    def rates(burn_in: int) -> tuple[float, float]:
        config = _config(burn_in, 2, alpha=0.1)
        deployed, _, _ = _run(zones, config, paths=200, blocks=600, per_read=10, seed=20261015)
        false = deployed[:, :3].sum(axis=1)
        fdp = false / np.maximum(deployed.sum(axis=1), 1)
        return float(fdp.mean()), float(deployed[:, 3:].mean())

    fdr, power = rates(1)
    assert fdr <= 0.1
    assert power >= 0.8
    fdr, _ = rates(0)
    assert fdr >= 0.4


def test_in_blocks_a_futility_stop_adds_no_wrong_deploy() -> None:
    """On the same draws, a zone the gate with a futility stop deploys is one the gate without it
    deploys too, and the stop ends most experiments on the zones at the margin."""
    zones = [AT_THE_MARGIN] * 3 + [BETTER]
    plain = _config(1, 2, alpha=0.1, min_effect=0.2)
    futile = _config(1, 2, alpha=0.1, min_effect=0.2, alpha_futility=0.1)
    runs = {"plain": plain, "futile": futile}
    deployed, modes = {}, {}
    for name, config in runs.items():
        deployed[name], modes[name], _ = _run(
            zones, config, paths=200, blocks=600, per_read=5, seed=20261016
        )
    assert not (deployed["futile"] & ~deployed["plain"]).any()
    assert deployed["futile"][:, :3].mean() <= 0.1
    assert deployed["futile"][:, 3].mean() >= 0.8
    assert (modes["futile"][:, :3] == "shadow").mean() >= 0.5
    assert not (modes["plain"] == "shadow").any()


def _blocks(
    ran: list[bool], reward: float = 0.5, *, rho: float = 0.5, length: int = 2
) -> BlockBatch:
    flags = np.array(ran)
    return BlockBatch(np.full((flags.size, length), reward), flags, np.where(flags, rho, 1.0 - rho))


def test_in_blocks_a_zone_starts_in_the_experiment_and_its_modes_log_blocks() -> None:
    """A zone in blocks starts in the experiment; deployed, it logs the candidate's blocks with
    probability 1, and in shadow, after a futility stop, the baseline's; a drift alarm sends a
    futile zone back to the experiment, whatever its plan says of shadow evidence."""
    config = _config(1, 2, alpha=0.1, min_effect=0.5, alpha_futility=0.1, drift_arl=5.0)
    slow = ZonePlan(chi2=0.0, tv=0.0)  # shadow is never too slow for this plan without blocks
    gate = DeploymentGate({"good": slow, "bad": slow}, config)
    assert gate.mode("good") == gate.mode("bad") == "experiment"
    win = BlockBatch(np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([True, False]), np.full(2, 0.5))
    feeds = {"good": win, "bad": _blocks([True, False])}
    for _ in range(40):
        live = {z: batch for z, batch in feeds.items() if gate.mode(z) == "experiment"}
        if not live:
            break
        gate.update(live)
    assert gate.mode("good") == "deployed"
    assert gate.mode("bad") == "shadow"
    with pytest.raises(ValueError, match="must log under the candidate"):
        gate.update({"good": _blocks([True, False])})
    with pytest.raises(ValueError, match="must log under the baseline"):
        gate.update({"bad": _blocks([True])})
    candidate_only = BlockBatch(np.ones((3, 2)), np.ones(3, dtype=bool), np.ones(3))
    baseline_only = BlockBatch(np.ones((3, 2)), np.zeros(3, dtype=bool), np.ones(3))
    assert gate.update({"good": candidate_only, "bad": baseline_only}) == {
        "good": "deploy",
        "bad": "shadow",
    }
    alarm = BlockBatch(np.ones((1, 2)), np.zeros(1, dtype=bool), np.ones(1), np.full((2, 1), 10.0))
    assert gate.update({"bad": alarm})["bad"] == "hold"
    assert gate.update({})["bad"] == "experiment"


def test_in_blocks_the_gate_logs_periods_as_its_decisions(caplog: pytest.LogCaptureFixture) -> None:
    """Four blocks of five periods a read: the futility stop is logged with the periods read."""
    gate = DeploymentGate({"z": PLAN}, _config(2, 5, alpha_futility=0.1, min_effect=0.2))
    batch = BlockBatch(np.ones((4, 5)), np.zeros(4, dtype=bool), np.full(4, 0.5))
    reads = 0
    with caplog.at_level(logging.INFO, logger="chc.gate"):
        while gate.mode("z") != "shadow" and reads < 30:
            gate.update({"z": batch})
            reads += 1
    events = [r for r in caplog.records if getattr(r, "chc_event", None) == "deployment_gate"]
    assert gate.mode("z") == "shadow"
    assert [event.decisions for event in events] == [20 * reads]  # type: ignore[attr-defined]


def test_without_blocks_the_gate_runs_as_before() -> None:
    config = GateConfig(delta=0.02, delta_harm=0.02, min_effect=0.02, horizon=100)
    assert (config.block_length, config.burn_in) == (None, 0)
    assert DeploymentGate({"z": PLAN}, config).mode("z") == "shadow"


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"burn_in": 1}, "without a block_length"),
        ({"block_length": 0}, "at least 1 period"),
        ({"block_length": 3, "burn_in": 3}, r"must lie in \[0, block_length\)"),
        ({"block_length": 3, "burn_in": -1}, r"must lie in \[0, block_length\)"),
        ({"check_burn_in": True}, "check_burn_in without a block_length"),
    ],
)
def test_a_config_with_blocks_outside_the_contract_is_refused(
    changes: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        GateConfig(delta=0.0, delta_harm=0.0, min_effect=0.1, horizon=10, **changes)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["check_burn_in", "centred"])
def test_a_switch_that_is_not_a_bool_is_refused(name: str) -> None:
    with pytest.raises(TypeError, match=f"{name} must be True or False, got 1"):
        _config(1, 2, **{name: 1})


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"reward": np.ones(2)}, r"shape \(blocks, periods\)"),
        ({"reward": np.ones((2, 0))}, r"shape \(blocks, periods\)"),
        ({"reward": np.full((2, 2), 1.5)}, r"must lie in \[0, 1\]"),
        ({"reward": np.full((2, 2), np.nan)}, "not finite"),
        ({"candidate": np.ones(2)}, "must hold 2 booleans"),
        ({"candidate": np.ones(3, dtype=bool)}, "must hold 2 booleans"),
        ({"logged": np.full(3, 0.5)}, "3 entries for 2 blocks"),
        ({"logged": np.array([0.5, 0.0])}, r"must lie in \(0, 1\]"),
        ({"logged": np.array([0.5, 1.5])}, r"must lie in \(0, 1\]"),
        ({"drift": np.ones((2, 1))}, r"shape \(4,\)"),
        ({"drift": -np.ones((4, 1))}, "non-negative"),
    ],
)
def test_blocks_outside_the_contract_are_refused(arguments: dict, match: str) -> None:
    valid = {"reward": np.ones((2, 2)), "candidate": np.array([True, False]), "logged": np.ones(2)}
    with pytest.raises(ValueError, match=match):
        BlockBatch(**(valid | arguments))


def test_a_gate_refuses_a_batch_of_the_other_kind_or_of_other_blocks() -> None:
    blocks = DeploymentGate({"z": PLAN}, _config(1, 2))
    plain = DeploymentGate({"z": PLAN}, GateConfig(0.0, 0.0, 0.1, 10))
    one = ZoneBatch(np.ones(1), np.ones(1), np.ones(1), np.ones(1))
    with pytest.raises(TypeError, match="reads a BlockBatch, got ZoneBatch"):
        blocks.update({"z": one})
    with pytest.raises(TypeError, match="reads a ZoneBatch, got BlockBatch"):
        plain.update({"z": _blocks([True])})
    with pytest.raises(ValueError, match="blocks of 3 periods, where the gate's are 2"):
        blocks.update({"z": _blocks([True], length=3)})
    with pytest.raises(ValueError, match="must log under the mixture"):
        blocks.update({"z": _blocks([True], rho=0.4)})


def test_in_blocks_carryover_past_the_burn_in_is_warned_of_and_moves_no_verdict(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Carryover of order 3 in blocks of 5 with 1 burned: two of a block's four read periods still
    see the block before, so its read mean falls by ``cost / 2 = 0.3`` when that block ran the
    candidate. With ``check_burn_in`` the check warns on every one of 100 paths, once, naming the
    burn-in to raise; by default it does not run, and every verdict is the same."""
    config = _config(1, 5, check_burn_in=True)
    order = _order(3, gain=0.4, cost=0.6)
    with caplog.at_level(logging.WARNING, logger="chc.gate"):
        deployed, modes, warned = _run(
            [order],
            config,
            paths=100,
            blocks=400,
            per_read=5,
            seed=20261025,
            start=lambda rng, n: np.zeros((n, 3)),
        )
    assert warned.all()
    records = [r for r in caplog.records if getattr(r, "chc_event", None) == "gate_carryover"]
    assert len(records) == 100
    for record in records:
        assert (record.lag, record.burn_in) == (1, 1)  # type: ignore[attr-defined]
        assert record.contrast < 0.0  # type: ignore[attr-defined]
        assert record.p_value <= 0.05  # type: ignore[attr-defined]
        assert "raise burn_in" in record.getMessage()
    again, again_modes, silent = _run(
        [order],
        _config(1, 5),
        paths=100,
        blocks=400,
        per_read=5,
        seed=20261025,
        start=lambda rng, n: np.zeros((n, 3)),
    )
    assert (again == deployed).all()
    assert (again_modes == modes).all()
    assert not silent.any()


def test_in_blocks_the_carryover_check_reads_the_experiment_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Deployed, a zone runs the candidate's blocks alone, whose policy is no draw: read means that
    climb from 0 to 1 over 400 blocks after the DEPLOY are not read as carryover."""
    gate = DeploymentGate({"z": PLAN}, _config(0, 1, check_burn_in=True))
    ran = np.arange(20) % 2 == 0
    win = BlockBatch(ran[:, None].astype(float), ran, np.full(20, 0.5))
    assert gate.update({"z": win}) == {"z": "deploy"}
    climb = BlockBatch(np.linspace(0.0, 1.0, 400)[:, None], np.ones(400, dtype=bool), np.ones(400))
    with caplog.at_level(logging.WARNING, logger="chc.gate"):
        assert gate.update({"z": climb}) == {"z": "deploy"}
    assert not [r for r in caplog.records if getattr(r, "chc_event", None) == "gate_carryover"]


def _one_period_blocks(
    config: GateConfig,
    *,
    mu: float,
    sd: float,
    effect: float,
    phi: float = 0.0,
    paths: int,
    blocks: int,
    seed: int,
    per_read: int = 10,
) -> np.ndarray:
    """The block after which each path's gate deployed, ``inf`` where it did not, in blocks of one
    period whose read means are ``clip(mu + effect u + eta, 0, 1)``: ``u`` the block's draw and
    ``eta`` an AR(``phi``) of standard deviation ``sd``, the same under both arms."""
    rng = np.random.default_rng(seed)
    first = np.full(paths, np.inf)
    for path in range(paths):
        ran = rng.random(blocks) < config.rho
        shocks = sd * rng.standard_normal(blocks)
        eta = shocks.copy()
        for k in range(1, blocks):
            eta[k] = phi * eta[k - 1] + math.sqrt(1.0 - phi**2) * shocks[k]
        reward = np.clip(mu + effect * ran + eta, 0.0, 1.0)[:, None]
        logged = np.where(ran, config.rho, 1.0 - config.rho)
        gate = DeploymentGate({"z": PLAN}, config)
        for start in range(0, blocks, per_read):
            cut = slice(start, start + per_read)
            if gate.update({"z": BlockBatch(reward[cut], ran[cut], logged[cut])})["z"] == "deploy":
                first[path] = start + per_read
                break
            if gate.mode("z") != "experiment":
                break
    return first


def _one_period(delta: float, *, centred: bool = True) -> GateConfig:
    return GateConfig(
        delta=delta,
        delta_harm=0.02,
        min_effect=0.02,
        horizon=10**6,
        block_length=1,
        centred=centred,
    )


NULLS = {
    "iid": {"mu": 0.5, "sd": 0.05, "effect": 0.0},
    "persistent": {"mu": 0.5, "sd": 0.05, "effect": 0.0, "phi": 0.9},
    "margin": {"mu": 0.5, "sd": 0.05, "effect": 0.02},
    "wide": {"mu": 0.5, "sd": 0.5, "effect": 0.0},
    "low": {"mu": 0.1, "sd": 0.05, "effect": 0.0},
}


@pytest.mark.parametrize("name", NULLS)
def test_centred_increments_deploy_a_candidate_no_better_than_delta_at_most_alpha(
    name: str,
) -> None:
    """Five nulls in blocks of one period, read every 10 blocks over 1000: no effect on read means
    of mean 0.5 and sd 0.05, independent or AR(0.9) across blocks, or of sd 0.5, or of mean 0.1;
    and an effect exactly at ``delta = 0.02``. Centred, the gate deploys on at most ``alpha`` of
    200 paths, within three binomial standard deviations: it deployed on 0, 0, 0, 2 and 0."""
    setting = NULLS[name]
    first = _one_period_blocks(
        _one_period(setting["effect"]), **setting, paths=200, blocks=1000, seed=20261027
    )
    assert np.isfinite(first).mean() <= _slack(200)


def _reads_the_draw(
    self: _Level, w_new: np.ndarray, w_base: np.ndarray, reward: np.ndarray, rho: float
) -> np.ndarray:
    """A level that reads the block it is for: the mean read of the arm the block did not draw."""
    levels = np.empty(reward.size)
    for k in range(reward.size):
        total, weight = self.sums
        means = np.divide(total, weight, out=np.full(2, 0.5), where=weight > 0.0)
        levels[k] = means[1] if w_new[k] > 0.0 else means[0]
        self.sums[0] += (w_new[k] * reward[k], w_base[k] * reward[k])
        self.sums[1] += (w_new[k], w_base[k])
    return levels


def test_a_centred_level_that_reads_the_block_s_own_draw_breaks_the_guarantee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At the margin, an effect of exactly ``delta = 0.02``, the gate's level deploys on at most
    ``alpha`` of 100 paths of 1000 blocks: on none. A level that reads the block's own draw, the
    mean of the arm the block did not run, makes the increment's mean ``2 Delta``, and on the same
    draws it deploys on most of them: on 94."""
    setting = {**NULLS["margin"], "paths": 100, "blocks": 1000, "seed": 20261028}
    config = _one_period(0.02)
    assert np.isfinite(_one_period_blocks(config, **setting)).mean() <= _slack(100)
    monkeypatch.setattr(_Level, "read", _reads_the_draw)
    assert np.isfinite(_one_period_blocks(config, **setting)).mean() >= 0.5


def test_centred_increments_deploy_a_better_candidate_where_plain_ones_do_not() -> None:
    """Read means ``0.5 + 0.03 u`` with sd 0.05, read every 10 blocks over 600: centred, the gate
    deploys on at least 0.95 of 50 paths; it did on all of them, after a median of 260 blocks,
    where the planning approximation in ``validation/gate_centred_increment.mac`` says 247. Plain,
    the increment's second moment is 98 times the centred one's, 1.07 against 0.0109, and the gate
    deploys on at most 0.1 of them."""
    setting = {"mu": 0.5, "sd": 0.05, "effect": 0.03, "paths": 50, "blocks": 600}
    centred = _one_period_blocks(_one_period(0.0), **setting, seed=20261029)
    assert np.isfinite(centred).mean() >= 0.95
    plain = _one_period_blocks(_one_period(0.0, centred=False), **setting, seed=20261029)
    assert np.isfinite(plain).mean() <= 0.1


def test_a_centred_level_reads_only_the_decisions_before_it() -> None:
    """Each level is read before its decision: changing a decision moves no level up to and
    including its own, and one batch reads what its decisions read one at a time. Before any
    decision the level is 1/2, while one arm is unseen it is the other's mean, and then it is
    ``(1 - rho) m_new + rho m_base``."""
    rng = np.random.default_rng(20261030)
    rho, n = 0.3, 40
    ran = rng.random(n) < rho
    ran[:3] = [False, False, True]  # the baseline's arm is seen first, the candidate's third
    w_new, w_base = ran / rho, ~ran / (1.0 - rho)
    reward = rng.random(n)
    levels = _Level().read(w_new, w_base, reward, rho)
    one_at_a_time = _Level()
    each = [
        one_at_a_time.read(w_new[k : k + 1], w_base[k : k + 1], reward[k : k + 1], rho)
        for k in range(n)
    ]
    np.testing.assert_array_equal(np.concatenate(each), levels)
    for k in range(n):
        moved = reward.copy()
        moved[k:] = rng.random(n - k)
        np.testing.assert_array_equal(
            _Level().read(w_new, w_base, moved, rho)[: k + 1], levels[: k + 1]
        )
    assert levels[0] == 0.5
    assert levels[1:3] == pytest.approx([reward[0], reward[:2].mean()], rel=1e-12, abs=0.0)
    for k in range(3, n):
        m_new, m_base = reward[:k][ran[:k]].mean(), reward[:k][~ran[:k]].mean()
        assert levels[k] == pytest.approx((1 - rho) * m_new + rho * m_base, rel=1e-12, abs=0.0)


def test_the_log_says_which_increments_produced_the_e_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each verdict's structured record carries ``centred``, whose increments its e-values are
    on: the same twenty blocks, the candidate's reading 1 and the baseline's 0, deploy either way,
    on other e-values."""
    ran = np.arange(20) % 2 == 0
    batch = BlockBatch(ran[:, None].astype(float), ran, np.full(20, 0.5))
    evidence = {}
    for centred in (False, True):
        gate = DeploymentGate({"z": PLAN}, _one_period(0.0, centred=centred))
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="chc.gate"):
            assert gate.update({"z": batch}) == {"z": "deploy"}
        (record,) = [
            r for r in caplog.records if getattr(r, "chc_event", None) == "deployment_gate"
        ]
        assert record.centred is centred  # type: ignore[attr-defined]
        evidence[centred] = record.log_evidence  # type: ignore[attr-defined]
    assert evidence[True] != evidence[False]
