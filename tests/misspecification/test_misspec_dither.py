"""The dither check's size on a graph that is right and its power on a confounder the graph omits.

The plant is ``x' = a x + b u + c w + sg e``, its logger ``u = d w + s eta``, with ``w`` the column
the graph leaves out, ``eta`` the logged draw, both drawn afresh, at ``a = 0.9``, ``b = 0.5`` and
``d = s = sg = 1``, over 100 units and 5 transitions each. The graph says nothing confounds the
lever. At ``c = 0`` it is right, and the check rejects at 5 % on at most ``alpha`` of the logs,
within three Monte Carlo standard deviations. At ``c = 1/2`` it is wrong, and the closed form
(``validation/dither_hausman.mac``) puts the contrast's noncentrality at the fixed alternative,
``c^2 s^2 / (sg^2 (d^2 + s^2) + c^2 (d^2 + 4 s^2))`` a transition, at 500/13 = 38.46 over the 500.

The check reads the difference's variance from the two fits' influences, summed two ways and at
CR1's factor for the smaller dimension's count, here 5 periods, so it reads that variance high,
and its power sits below the one-degree power of that noncentrality, 1.000. The contract read:
at least 0.8 of the logs rejected, within three Monte Carlo standard deviations.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

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
    paths = 40
    rejected = float(np.mean(_p_values(0.0, range(paths)) <= ALPHA))
    assert rejected <= ALPHA + 3.0 * math.sqrt(ALPHA * (1.0 - ALPHA) / paths)


def test_power_on_a_confounder_the_graph_omits() -> None:
    assert _noncentrality(0.5) == pytest.approx(500.0 / 13.0, rel=1e-12, abs=0.0)
    paths, floor = 20, 0.8
    rejected = float(np.mean(_p_values(0.5, range(1000, 1000 + paths)) <= ALPHA))
    assert rejected >= floor - 3.0 * math.sqrt(floor * (1.0 - floor) / paths)
