"""Property-based (hypothesis) tests: one invariant per layer of the decision facade.

The example-based files pin *what the chain answers* on a fixed plant. These pin *what the answer
may never be*, on inputs nobody chose: a projection that moves a point already in the box, a
certificate that grows as the assumed confounding does, a schedule outside the lever band a client
signed off, a report that will not survive `json.dumps`, a fingerprint that depends on dictionary
order, a d-separation that depends on which end you ask from.

The expensive properties (anything that fits a model) run few examples with `deadline=None` on a
deliberately small panel -- hypothesis is here to vary the *shape* of the input, and a slow
generator would silence it in CI rather than sharpen it.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from chc.control import project_box
from chc.cost import QuadraticCost
from chc.decision import Constraint, Lever, Target, prescribe
from chc.dynamics import LinearDynamics
from chc.graph import CausalGraph
from chc.panel import Panel
from chc.plan import causal_plan, certify_safety

DT = 0.1


def finite(lo: float, hi: float) -> st.SearchStrategy[float]:
    """Bounded, non-NaN, non-inf floats."""
    return st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False)


# ---- projection: the primitive every clipped schedule rests on ----
#
# `deadline=None` on both: hypothesis times each example in wall clock, and every distinct list
# length is a distinct JAX shape, so the first example at a new size pays an XLA compile. That is
# 265 ms under a loaded machine against a 200 ms default -- a measurement of the compiler, not of
# the property. Varying the length is the point (scalar and per-lever bounds both broadcast), so
# the deadline goes rather than the variation, and `max_examples` caps how many shapes get built.


@given(
    us=st.lists(finite(-50.0, 50.0), min_size=1, max_size=40),
    lo=finite(-10.0, 0.0),
    width=finite(0.0, 20.0),
)
@settings(deadline=None, max_examples=40)
def test_project_box_is_idempotent_and_lands_in_the_box(
    us: list[float], lo: float, width: float
) -> None:
    once = project_box(jnp.asarray(us), lo, lo + width)
    assert bool(jnp.all(once >= lo - 1e-9))
    assert bool(jnp.all(once <= lo + width + 1e-9))
    assert bool(jnp.allclose(project_box(once, lo, lo + width), once))  # P o P == P


@given(us=st.lists(finite(-5.0, 5.0), min_size=1, max_size=20))
@settings(deadline=None, max_examples=40)
def test_project_box_fixes_points_already_inside(us: list[float]) -> None:
    xs = jnp.asarray(us)
    assert bool(jnp.array_equal(project_box(xs, -5.0, 5.0), xs))


# ---- §40: the safety certificate may only shrink as the assumed confounding grows ----


def _toy_plan():
    """One small plan, built once: hypothesis varies `gamma`, not the optimisation."""
    model = LinearDynamics(jnp.array([[-0.5]]), jnp.array([[1.0]]))
    return model, causal_plan(
        model,
        jnp.array([0.4]),
        QuadraticCost(Q=jnp.eye(1), R=0.05 * jnp.eye(1), Qf=jnp.eye(1), x_target=jnp.array([1.0])),
        DT,
        12,
        jnp.array([-1.0]),
        jnp.array([1.0]),
    )


MODEL, PLAN = _toy_plan()


@given(gamma=finite(1.0, 4.0), step=finite(0.05, 3.0))
@settings(deadline=None, max_examples=30)
def test_certified_prefix_is_monotone_in_gamma(gamma: float, step: float) -> None:
    def barrier(x):
        return 1.5 - x[0]

    weak = certify_safety(PLAN, MODEL, barrier, DT, gamma=gamma, u_max=1.0)
    weaker = certify_safety(PLAN, MODEL, barrier, DT, gamma=gamma + step, u_max=1.0)
    assert weaker.certified_steps <= weak.certified_steps  # more assumed confounding, never more
    assert not np.isnan(weak.gamma_star) or np.isnan(weaker.gamma_star)


# ---- the facade: whatever the box, the schedule is inside it, and the report is JSON ----


def _logs(n_units: int = 40, n_periods: int = 8, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    columns = ("unit", "time", "supply", "wait", "incentive", "demand")
    rows: dict[str, list[float]] = {name: [] for name in columns}
    for unit in range(n_units):
        supply, wait = rng.normal(0.0, 0.2), rng.normal(0.0, 0.2)
        for period in range(n_periods):
            demand = rng.normal(0.0, 1.0)
            incentive = 0.9 * demand + rng.normal(0.0, 0.5)
            rows["unit"].append(unit)
            rows["time"].append(period)
            rows["supply"].append(supply)
            rows["wait"].append(wait)
            rows["incentive"].append(incentive)
            rows["demand"].append(demand)
            supply_next = supply + DT * (
                -0.6 * supply + 0.3 * wait + 0.8 * incentive + 1.5 * demand
            )
            wait = wait + DT * (0.25 * wait - 0.4 * incentive) + rng.normal(0.0, 0.01)
            supply = supply_next + rng.normal(0.0, 0.01)
    return {name: np.asarray(values) for name, values in rows.items()}


PANEL = Panel.from_frame(_logs(), unit="unit", time="time", seed=0)
GRAPH = CausalGraph.from_edges(
    [
        ("demand", "incentive"),
        ("demand", "supply"),
        ("incentive", "supply"),
        ("incentive", "wait"),
        ("supply", "wait"),
    ]
)


def _prescribe(lo: float, hi: float, target: float):
    return prescribe(
        PANEL,
        levers=[Lever("incentive", lo=lo, hi=hi, unit_cost=0.05)],
        target=Target("supply", value=target),
        constraints=[Constraint("wait", hi=0.5)],
        adjustment=GRAPH,
        horizon=8,
        dt=DT,
    )


@given(lo=finite(-3.0, -0.1), width=finite(0.2, 6.0), target=finite(-2.0, 2.0))
@settings(deadline=None, max_examples=12, suppress_health_check=[HealthCheck.too_slow])
def test_schedule_never_leaves_the_lever_box(lo: float, width: float, target: float) -> None:
    """The band is a client commitment, so it is a hard constraint, not a penalty weight."""
    schedule = _prescribe(lo, lo + width, target).schedule
    magnitudes = np.asarray(schedule.magnitudes)
    assert magnitudes.shape[1] == 1
    assert np.all(magnitudes >= lo - 1e-6)
    assert np.all(magnitudes <= lo + width + 1e-6)


@given(lo=finite(-3.0, -0.1), width=finite(0.2, 6.0), target=finite(-2.0, 2.0))
@settings(deadline=None, max_examples=8, suppress_health_check=[HealthCheck.too_slow])
def test_to_json_round_trips_through_the_stdlib(lo: float, width: float, target: float) -> None:
    """`to_json` claims plain JSON-safe values; the only honest check is to serialise them."""
    payload = _prescribe(lo, lo + width, target).to_json()
    assert json.loads(json.dumps(payload)) == payload


# ---- provenance: a fingerprint of the data, not of how the dictionary was typed ----


@given(
    values=st.lists(finite(-10.0, 10.0), min_size=6, max_size=60),
    reverse=st.booleans(),
)
def test_fingerprint_ignores_column_order_and_notices_a_changed_value(
    values: list[float], reverse: bool
) -> None:
    n = len(values)
    frame = {
        "unit": np.zeros(n),
        "time": np.arange(float(n)),
        "y": np.asarray(values),
    }
    shuffled = dict(reversed(list(frame.items()))) if reverse else dict(frame)
    base = Panel.from_frame(frame, unit="unit", time="time")
    assert Panel.from_frame(shuffled, unit="unit", time="time").provenance.data_sha256 == (
        base.provenance.data_sha256
    )
    bumped = dict(frame, y=np.asarray(values) + 1.0)
    assert Panel.from_frame(bumped, unit="unit", time="time").provenance.data_sha256 != (
        base.provenance.data_sha256
    )


# ---- graphs: d-separation is a statement about a pair, not about an order ----


@st.composite
def dags(draw: st.DrawFn, max_nodes: int = 6) -> CausalGraph:
    """Random DAGs by construction: names in topological order, edges only ever forwards.

    ``from_edges`` takes no isolated nodes, so a draw that produced none is nudged to a single edge
    rather than rejected --- filtering here would cost hypothesis its shrinking on the same input.
    """
    n = draw(st.integers(min_value=2, max_value=max_nodes))
    names = [f"v{i}" for i in range(n)]
    edges = [(names[i], names[j]) for i in range(n) for j in range(i + 1, n) if draw(st.booleans())]
    return CausalGraph.from_edges(edges or [(names[0], names[1])])


def _two(*, draw: st.DataObject, graph: CausalGraph) -> tuple[str, str]:
    """Two distinct nodes.

    Drawn as a pair rather than filtered: ``pytest.skip`` inside a hypothesis test abandons the
    whole test at the first degenerate draw, which turns a 150-example property into a green skip.
    """
    names = sorted(graph.nodes)
    first = draw.draw(st.sampled_from(names))
    return first, draw.draw(st.sampled_from([name for name in names if name != first]))


@given(graph=dags(), data=st.data())
@settings(max_examples=150)
def test_d_separation_is_symmetric(graph: CausalGraph, data: st.DataObject) -> None:
    a, b = _two(draw=data, graph=graph)
    rest = [name for name in sorted(graph.nodes) if name not in (a, b)]
    given_ = frozenset(data.draw(st.sets(st.sampled_from(rest)))) if rest else frozenset()
    assert graph.d_separated(a, b, given_) == graph.d_separated(b, a, given_)


@given(graph=dags(), data=st.data())
@settings(max_examples=200)
def test_the_canonical_set_never_conditions_on_the_future(
    graph: CausalGraph, data: st.DataObject
) -> None:
    """No covariate may be the outcome or a descendant of the treatment.

    Stated on the *output* rather than by asking ``is_valid_adjustment_set``, deliberately. Both
    methods read the same ``forb``, so a consistency check between them is a tautology: dropping
    ``Y`` from ``forb`` -- the exact pre-release defect a brute-force cross-check found -- leaves
    such a check green. ``descendants`` is an independent BFS over ``children``, so this does not.
    """
    treatment, outcome = _two(draw=data, graph=graph)
    found = graph.adjustment_set(treatment=treatment, outcome=outcome)
    assert outcome not in found.covariates  # conditioning on the effect certifies nothing
    assert treatment not in found.covariates
    below = graph.descendants(treatment)
    assert not below.intersection(found.covariates)  # no post-treatment adjustment


@given(graph=dags(), data=st.data())
@settings(max_examples=150)
def test_the_canonical_set_passes_the_validity_check_it_is_offered_under(
    graph: CausalGraph, data: st.DataObject
) -> None:
    """The two public entry points must not drift apart: what `Adjust` returns, `is_valid` takes."""
    treatment, outcome = _two(draw=data, graph=graph)
    found = graph.adjustment_set(treatment=treatment, outcome=outcome)
    if found.status == "identified":
        assert graph.is_valid_adjustment_set(found.covariates, treatment=treatment, outcome=outcome)
