"""prescribe(dither=): the plan's channel against the one the logged draw instruments.

The plant is ``x' = 0.9 x + 0.5 u + c w + e``, its logger ``u = d w + eta``: ``w`` the column a
graph may leave out, ``eta`` the logged draw, all drawn afresh. The check adds a fit and a reading;
it moves nothing the plan was made from.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from chc import CausalGraph, Lever, Panel, Target, prescribe
from chc.decision import Constraint, DecisionError, DitherCheck, Driver

TRUE = CausalGraph.from_edges([("promo", "incentive"), ("promo", "x"), ("incentive", "x")])
EMPTY = CausalGraph.from_edges([("incentive", "x")])  # says nothing confounds the lever
LEVER = Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)


def _logs(c: float, d: float, seed: int, units: int = 100, periods: int = 6) -> Panel:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, units)
    names = ("unit", "time", "x", "incentive", "promo", "draw", "rule")
    rows: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for t in range(periods):
        w, eta, e = rng.standard_normal((3, units))
        u = d * w + eta
        for name, value in zip(
            names, (np.arange(units), np.full(units, t), x, u, w, eta, 0.5 * x), strict=True
        ):
            rows[name].append(value)
        x = 0.9 * x + 0.5 * u + c * w + e
    frame = {name: np.concatenate(value) for name, value in rows.items()}
    return Panel.from_frame(frame, unit="unit", time="time", seed=0)


def _prescribe(panel: Panel, graph, **kwargs):
    return prescribe(
        panel,
        levers=[LEVER],
        target=Target("x", value=1.0),
        adjustment=graph,
        horizon=5,
        dt=1.0,
        **kwargs,
    )


CONFOUNDED = _logs(c=0.8, d=1.0, seed=1003)


@pytest.fixture(scope="module")
def pair():
    return _prescribe(CONFOUNDED, EMPTY), _prescribe(CONFOUNDED, EMPTY, dither="draw")


def test_dither_moves_nothing_the_plan_reads(pair) -> None:
    plain, checked = pair
    assert plain.dither_check is None
    assert checked.dither_check is not None
    np.testing.assert_array_equal(
        np.asarray(plain.schedule.magnitudes), np.asarray(checked.schedule.magnitudes)
    )
    assert plain.to_json() == checked.to_json()
    for part in ("drift", "channel"):
        np.testing.assert_array_equal(
            np.asarray(getattr(plain.model_fit.residual, part)),
            np.asarray(getattr(checked.model_fit.residual, part)),
        )
    lines = [line for line in checked.report().splitlines() if "dither check" not in line]
    assert lines == plain.report().splitlines()


def test_a_planted_confounder_is_rejected(pair, caplog) -> None:
    check = pair[1].dither_check
    assert isinstance(check, DitherCheck)
    assert check.status == "tested"
    assert check.dither == ("draw",)
    assert check.contrast is not None
    assert check.contrast.p_value <= 0.05
    assert check.contrast.cost > 0.0
    assert check.share > 0.0
    assert check.relevance
    assert min(check.relevance) > 0.5
    assert "**the plan's channel is not the one the draw ['draw'] instruments**" in (
        pair[1].report()
    )
    with caplog.at_level(logging.WARNING, logger="chc"):
        _prescribe(CONFOUNDED, EMPTY, dither=["draw"])
    events = [r for r in caplog.records if getattr(r, "chc_event", None) == "dither"]
    assert len(events) == 1
    assert events[0].levelno == logging.WARNING
    assert events[0].status == "tested"


def test_the_adjusted_graph_is_tested_not_void() -> None:
    check = _prescribe(CONFOUNDED, TRUE, dither="draw").dither_check
    assert check.status == "tested"
    assert check.contrast is not None
    assert check.contrast.p_value > 0.05


def test_a_logger_of_the_state_and_its_draw_alone_is_void() -> None:
    # d = 0: the lever is the draw, so the plan's fit, on the states alone, is the draw's own
    result = _prescribe(_logs(c=0.5, d=0.0, seed=3), EMPTY, dither="draw")
    check = result.dither_check
    assert isinstance(check, DitherCheck)
    assert check.status == "void"
    assert check.contrast is None
    assert np.isfinite(check.share)
    assert "dither check: void, the logger read nothing but the state" in result.report()


@pytest.mark.parametrize(
    ("dither", "kwargs", "message"),
    [
        ([], {}, "names no column"),
        (["draw", "draw"], {}, "more than once"),
        ("incentive", {}, "as the draw and as a lever"),
        ("x", {}, "as the draw and as a lever"),
        ("draw", {"adjustment": ["draw"]}, "adjustment set"),
        ("draw", {"adjustment": []}, "asserted empty adjustment set"),
        (
            "draw",
            {"constraints": [Constraint("x", hi=3.0)], "hold_constraints": True},
            "hold_constraints",
        ),
        ("draw", {"drivers": [Driver("promo", forecast=np.zeros(6))]}, "with drivers"),
    ],
)
def test_dither_refusals(dither, kwargs, message) -> None:
    graph = kwargs.pop("adjustment", EMPTY)
    with pytest.raises(DecisionError, match=message):
        _prescribe(CONFOUNDED, graph, dither=dither, **kwargs)


def test_dither_names_a_column_the_panel_has() -> None:
    with pytest.raises(KeyError, match="nowhere"):
        _prescribe(CONFOUNDED, EMPTY, dither="nowhere")


def test_dither_refuses_a_lever_that_follows_its_logged_rule() -> None:
    graph = CausalGraph.from_edges([("incentive", "x"), ("rule", "x")])
    with pytest.raises(DecisionError, match="follow the rule of the state"):
        prescribe(
            CONFOUNDED,
            levers=[LEVER, Lever("rule", lo=-2.0, hi=2.0, unit_cost=0.05)],
            target=Target("x", value=1.0),
            adjustment=graph,
            horizon=5,
            dt=1.0,
            dither="draw",
        )
