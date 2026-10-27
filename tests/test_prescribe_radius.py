"""``prescribe``'s barrier reads the fit's own absolute radius on its channel by default.

The radius was a share of the fitted channel's size at the plan's start, so at every level it
stayed within the estimate's own size of the estimate, and ``gamma = 1`` was a radius of zero.
An Euler fit by adjustment gives an absolute one (:meth:`chc.plan.ChannelSensitivity.from_fit`),
read at each planned step's predicted state, and the held solve and the audit now read that one.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from chc import CausalGraph, Constraint, Lever, Panel, Target, prescribe
from chc.decision import DecisionError, Prescription
from chc.plan import ChannelSensitivity

DT = 0.1
EDGES = [
    ("demand", "incentive"),
    ("demand", "supply"),
    ("incentive", "supply"),
    ("incentive", "wait"),
    ("wait", "supply"),
]
REACH = 2.0  # the incentive's largest action


def _panel(seed: int = 0, units: int = 100, periods: int = 12) -> Panel:
    """Supply and wait under an incentive the log sets from demand, a confounder it records,
    stepped by Euler at ``DT``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.2, size=(units, 2))
    frames: dict[str, list[np.ndarray]] = {name: [] for name in ("supply", "wait", "incentive")}
    demand = []
    for _ in range(periods):
        z = rng.normal(size=units)
        u = 0.9 * z + 0.5 * rng.normal(size=units)
        frames["supply"].append(x[:, 0])
        frames["wait"].append(x[:, 1])
        frames["incentive"].append(u)
        demand.append(z)
        rate = np.stack(
            [-0.6 * x[:, 0] + 0.3 * x[:, 1] + 0.8 * u + 1.5 * z, 0.25 * x[:, 1] - 0.4 * u], 1
        )
        x = x + DT * rate + rng.normal(0.0, 0.01, size=x.shape)
    columns = {name: np.stack(values, 1).ravel() for name, values in frames.items()}
    columns["demand"] = np.stack(demand, 1).ravel()
    columns["unit"] = np.repeat(np.arange(units), periods)
    columns["time"] = np.tile(np.arange(periods), units)
    return Panel.from_frame(columns, unit="unit", time="time", seed=0)


def _ask(**options: object) -> Prescription:
    return prescribe(
        _panel(),
        levers=[Lever("incentive", lo=-REACH, hi=REACH, unit_cost=0.05)],
        target=Target("supply", value=1.0),
        adjustment=CausalGraph.from_edges(EDGES),
        horizon=15,
        dt=DT,
        **{  # type: ignore[arg-type]
            "integrator": "euler",
            "x0": np.zeros(2),
            "constraints": [Constraint("supply", hi=0.8), Constraint("wait", hi=0.5)],
            **options,
        },
    )


def _radius_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if getattr(record, "chc_event", None) == "radius"]


@pytest.mark.parametrize("held", [True, False])
def test_the_barrier_reads_the_fit_s_absolute_radius_at_each_step_by_default(
    held: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """On an Euler fit by adjustment the default audit reads the fit's own sensitivity model at
    each planned step's predicted state, held or not, as ``radius="absolute"`` does; ``gamma = 1``
    is then the channel's sampling error, a radius the plant's deviation is carried by, not the
    zero the share read there."""
    with caplog.at_level(logging.INFO, logger="chc.decision"):
        result = _ask(hold_constraints=held)
    assert not _radius_records(caplog)
    assert result.plan is not None
    safety = result.plan.safety if held else None
    certificate = result.certificate
    assert certificate.radius_model == "absolute"
    assert result.run is not None
    assert result.run.radius == "absolute"
    sensitivity = ChannelSensitivity.from_fit(result.model_fit)
    states = np.asarray(result.plan.trajectory, dtype=float)[:-1]
    # each margin's gradient has norm 1, so the audit's radius is the widest step's Delta(x_k)
    widest = max(sensitivity.radius(state, 1.0, u_max=[REACH]) for state in states)
    first = sensitivity.radius(states[0], 1.0, u_max=[REACH])
    assert widest > first > 0.0  # read at each step, not frozen at the start
    if safety is not None:
        assert safety.radius_model == "absolute"
        assert safety.radius == pytest.approx(widest, rel=1e-9, abs=0.0)
        assert float(np.max(np.asarray(safety.deviation))) > 0.0
        assert certificate.barrier_certified_steps == safety.certified_steps
    asked = _ask(hold_constraints=held, radius="absolute")
    assert (asked.certificate.radius_model, asked.certificate.barrier_certified_steps) == (
        "absolute",
        certificate.barrier_certified_steps,
    )
    assert asked.certificate.gamma_star == certificate.gamma_star
    relative = _ask(hold_constraints=held, radius="relative")
    assert relative.certificate.radius_model == "relative"
    assert relative.run is not None
    assert relative.run.radius == "relative"


def test_a_fit_with_no_absolute_radius_reads_the_relative_one_and_says_so_quietly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An ``rk4`` fit is a fixed point of its own step, whose error no bound on the levers' density
    limits: the default barrier reads the share of the fitted channel, and an ``INFO`` record, not
    a ``WARNING``, names the fit's refusal, so a default run prints nothing. Asked for, the
    relative radius is read with no record; with no constraint no radius is read at all."""
    with caplog.at_level(logging.INFO, logger="chc.decision"):
        result = _ask(integrator="rk4")
    assert result.certificate.radius_model == "relative"
    assert result.run is not None
    assert result.run.radius == "relative"
    (record,) = _radius_records(caplog)
    assert record.levelno == logging.INFO
    assert "rk4" in record.reason  # type: ignore[attr-defined]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="chc.decision"):
        asked = _ask(radius="relative")
        free = _ask(integrator="rk4", constraints=(), x0=np.zeros(1))
    assert asked.certificate.radius_model == "relative"
    assert not _radius_records(caplog)
    assert free.certificate.radius_model is None
    assert free.run is not None
    assert free.run.radius is None


def test_an_absolute_radius_asked_of_a_fit_that_gives_none_is_refused() -> None:
    """``radius="absolute"`` on an ``rk4`` fit names the fit's refusal and the two ways out, each
    once: up to 0.24.0 the advice to fit under Euler came twice."""
    with pytest.raises(DecisionError, match="rk4") as refused:
        _ask(integrator="rk4", radius="absolute")
    assert str(refused.value).count("integrator='euler'") == 1
    assert str(refused.value).count("radius='relative'") == 1


def test_every_finite_level_reads_the_absolute_radius() -> None:
    """The sensitivity model bounds the error at every finite level; up to 0.21 it bounded none
    from ``(3 + sqrt 5)/2`` on, and ``prescribe`` refused the absolute radius at such a level."""
    assert _ask(gamma=3.0).certificate.radius_model == "absolute"
    assert _ask(gamma=3.0, radius="relative").certificate.radius_model == "relative"
    assert _ask(gamma=3.0, integrator="rk4").certificate.radius_model == "relative"
    with pytest.raises(DecisionError, match="none of 'auto'"):
        _ask(radius="share")
