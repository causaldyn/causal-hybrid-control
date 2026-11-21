"""The dither check's size on a graph that is right and holds a covariate that persists.

The plant is ``x' = a x + b u + c w + sg e``, its logger ``u = d w + s eta``, ``eta`` the logged
draw, drawn afresh, and ``w`` AR(0.9) of unit variance, at ``a = 0.9``, ``b = 0.5``, ``c = 0.8`` and
``d = s = sg = 1``, over 100 units and 5 transitions each. The graph says ``w`` confounds the lever,
and it is right: the plan's fit holds ``w``, and its push in its drift. A covariate that persists
leans on the state, since its past pushed it, and a drift on the states alone reads that lean into
its slope, 0.154 of ``w``'s push here (``validation/dither_persistence.mac``). The instrumented fit
holds the adjustment set's push as the plan's fit does, so the contrast reads the two channels and
nothing more, and rejects at a level ``a`` on at most ``a`` of the logs.

The count is held to the bound a binomial count at the stated rate crosses with probability at
most ``alpha = 1/1000`` (:mod:`_misspec_counts`), on seeds read from its block.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _misspec_counts import at_most, seeds

from chc import CausalGraph, Lever, Panel, Target, prescribe
from chc.decision import DitherCheck

pytestmark = pytest.mark.misspecification

A, B, C, D, S, SG, RHO = 0.9, 0.5, 0.8, 1.0, 1.0, 1.0, 0.9
UNITS, PERIODS, ALPHA = 100, 6, 0.05
GRAPH = CausalGraph.from_edges([("w", "incentive"), ("w", "x"), ("incentive", "x")])


def _logs(seed: int) -> Panel:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, UNITS)
    w = rng.standard_normal(UNITS)
    names = ("unit", "time", "x", "incentive", "w", "draw")
    rows: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for t in range(PERIODS):
        fresh, eta, e = rng.standard_normal((3, UNITS))
        w = RHO * w + math.sqrt(1.0 - RHO**2) * fresh
        u = D * w + S * eta
        for name, value in zip(
            names, (np.arange(UNITS), np.full(UNITS, t), x, u, w, eta), strict=True
        ):
            rows[name].append(value)
        x = A * x + B * u + C * w + SG * e
    frame = {name: np.concatenate(value) for name, value in rows.items()}
    return Panel.from_frame(frame, unit="unit", time="time", seed=0)


def test_size_on_a_graph_that_holds_a_persistent_covariate() -> None:
    """200 logs, rejected at 0.05 on at most 21, the ceiling of ``Binomial(200, 0.05)`` at
    ``alpha = 1/1000``. Block 0: rejected 1, and 21 at 0.2; up to 1.3.1, whose instrumented drift
    held nothing, 57 and 132."""
    rejected = 0
    for seed in seeds(3000):
        check = prescribe(
            _logs(seed),
            levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
            target=Target("x", value=1.0),
            adjustment=GRAPH,
            horizon=5,
            dt=1.0,
            dither="draw",
        ).dither_check
        assert isinstance(check, DitherCheck)
        assert check.status == "tested"
        assert check.contrast is not None
        rejected += check.contrast.p_value <= ALPHA
    assert rejected <= at_most("0.05"), rejected
