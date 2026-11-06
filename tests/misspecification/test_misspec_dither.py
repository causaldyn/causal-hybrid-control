"""The dither check's size on a graph that is right and its power on a confounder the graph omits.

The plant is ``x' = a x + b u + c w + sg e``, its logger ``u = d w + s eta``, with ``w`` the column
the graph leaves out, ``eta`` the logged draw, both drawn afresh, at ``a = 0.9``, ``b = 0.5`` and
``d = s = sg = 1``, over 100 units and 5 transitions each. The graph says nothing confounds the
lever. At ``c = 0`` it is right, and the check rejects at a level ``a`` on at most ``a`` of the
logs. At ``c = 1/2`` it is wrong, and the closed form (``validation/dither_hausman.mac``) puts the
contrast's noncentrality at the fixed alternative,
``c^2 s^2 / (sg^2 (d^2 + s^2) + c^2 (d^2 + 4 s^2))`` a transition, at 500/13 = 38.46 over the 500.

The check reads the difference's variance from the two fits' influences, summed two ways and at
CR1's factor for the smaller dimension's count, here 5 periods, so it reads that variance high,
and its power sits below the one-degree power of that noncentrality, 1.000.

Each count is held to the bound a binomial count at the stated rate crosses with probability at
most ``alpha = 1/1000`` (:mod:`_misspec_counts`), on seeds read from its block.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from _misspec_counts import at_least, at_most, seeds

from chc import CausalGraph, Lever, Panel, Target, prescribe
from chc.decision import DitherCheck

pytestmark = pytest.mark.misspecification

A, B, D, S, SG = 0.9, 0.5, 1.0, 1.0, 1.0
UNITS, PERIODS, ALPHA = 100, 6, 0.05
GRAPH = CausalGraph.from_edges([("incentive", "x")])


def _logs(c: float, seed: int) -> Panel:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, UNITS)
    names = ("unit", "time", "x", "incentive", "draw")
    rows: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for t in range(PERIODS):
        w, eta, e = rng.standard_normal((3, UNITS))
        u = D * w + S * eta
        for name, value in zip(
            names, (np.arange(UNITS), np.full(UNITS, t), x, u, eta), strict=True
        ):
            rows[name].append(value)
        x = A * x + B * u + c * w + SG * e
    frame = {name: np.concatenate(value) for name, value in rows.items()}
    return Panel.from_frame(frame, unit="unit", time="time", seed=0)


def _p_values(c: float, seeds: range) -> np.ndarray:
    logging.getLogger("chc").setLevel(logging.ERROR)
    values = []
    for seed in seeds:
        check = prescribe(
            _logs(c, seed),
            levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
            target=Target("x", value=1.0),
            adjustment=GRAPH,
            horizon=5,
            dt=1.0,
            dither="draw",
        ).dither_check
        assert isinstance(check, DitherCheck)
        assert check.status == "tested"
        values.append(check.contrast.p_value)
    return np.array(values)


def _noncentrality(c: float) -> float:
    transitions = UNITS * (PERIODS - 1)
    return transitions * c**2 * S**2 / (SG**2 * (D**2 + S**2) + c**2 * (D**2 + 4.0 * S**2))


def test_size_on_a_graph_that_is_right() -> None:
    """200 logs, rejected at 0.05 on at most 21 and at 0.2 on at most 58, the ceilings of
    ``Binomial(200, 0.05)`` and ``Binomial(200, 0.2)`` at ``alpha = 1/1000``. Block 0: rejected 4
    and 30. Over 1000 logs, 0.023 and 0.157: the check reads the variance high, and rejects
    below its level. A p-value 0.33 times its own rejects ``p1`` = 0.110 and 0.553 of them, where
    the second ceiling fails a block with probability above 0.999999; it fails with probability
    0.8 down to a p-value 0.53 times its own, its minimal detectable defect. The second level is
    the test's power: at 0.05 alone a p-value a third of its own passes half the blocks."""
    p_values = _p_values(0.0, seeds(0))
    rejected = {level: int(np.sum(p_values <= level)) for level in (ALPHA, 0.2)}
    assert rejected[ALPHA] <= at_most("0.05"), rejected
    assert rejected[0.2] <= at_most("0.2"), rejected


def test_power_on_a_confounder_the_graph_omits() -> None:
    """20 logs, rejected at 0.05 on at least 10, the floor of ``Binomial(20, 0.8)`` at
    ``alpha = 1/1000``. At ``c = 1/2`` the check rejected 196 of 200 logs, 19 or 20
    in each 20, and all 20 on block 0."""
    assert _noncentrality(0.5) == pytest.approx(500.0 / 13.0, rel=1e-12, abs=0.0)
    rejected = int(np.sum(_p_values(0.5, seeds(1000, 20)) <= ALPHA))
    assert rejected >= at_least("0.8", 20), rejected
