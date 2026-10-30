"""chc.discovery: each kept edge's p-value, and the graph a decision record carries it on."""

import hashlib
import json
import re
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import chc.discovery
from chc.decision import Lever, Prescription, Target, prescribe
from chc.discovery import LaggedGraph, _lagged_design, discover_lagged_parents
from chc.graph import CausalGraph
from chc.independence import partial_corr_test
from chc.panel import Panel
from chc.pathway import causal_pathway
from test_discovery import _simulate_var


def _digest(values: list) -> str:
    """Rounded to ten digits, so that a last bit a BLAS or a CPU moves does not move it."""
    text = repr([f"{v:.10g}" if isinstance(v, float) else v for v in values])
    return hashlib.sha256(text.encode()).hexdigest()


def _with_controls(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Three states and one randomised lever: ``u`` moves ``x0`` up and ``x2`` down a step later,
    and ``x0`` moves ``x1``."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, 1))
    x = np.zeros((n, 3))
    for t in range(1, n):
        x[t, 0] = 0.6 * x[t - 1, 0] + 0.8 * u[t - 1, 0] + 0.3 * rng.standard_normal()
        x[t, 1] = 0.5 * x[t - 1, 1] + 0.4 * x[t - 1, 0] + 0.3 * rng.standard_normal()
        x[t, 2] = 0.7 * x[t - 1, 2] - 0.5 * u[t - 1, 0] + 0.3 * rng.standard_normal()
    return x, u


def _graph_values(graph: LaggedGraph) -> list:
    control = None if graph.control_parents is None else np.asarray(graph.control_parents).tolist()
    return [np.asarray(graph.state_parents).tolist(), graph.max_lag, graph.edges(), control]


def test_the_discovered_graphs_and_a_pathway_are_what_they_were() -> None:
    """A VAR series and one with a lever, at ``alpha`` 0.01 and 0.05 and with ``max_parents``
    capped, and a pathway on the series with the lever: every parent, edge and impulse response,
    digested as the previous release produced them."""
    var = _simulate_var(600, seed=0)
    x, u = _with_controls(600, seed=1)
    values: list = []
    for alpha in (0.01, 0.05):
        values += _graph_values(discover_lagged_parents(var, max_lag=3, alpha=alpha))
        values += _graph_values(discover_lagged_parents(x, u, max_lag=2, alpha=alpha))
    values += _graph_values(discover_lagged_parents(var, max_lag=3, max_parents=2))
    values += _graph_values(discover_lagged_parents(x, u, max_lag=2, max_parents=1))
    assert _digest(values) == "888ed9ceccfe300fb374f435d23e4c52da4661b7c8881f2f31a035386e60d260"

    pathway = causal_pathway(x, 1, u, horizon=6, max_lag=2)
    read: list = [pathway.target, pathway.horizon]
    for edge in pathway.edges:
        read += [edge.source, edge.kind, edge.lag, edge.sign, edge.contribution, edge.peak_horizon]
        read += [edge.actionable, *map(float, edge.irf)]
    assert len(pathway.edges) == 3
    assert _digest(read) == "796047fe4499c4a2ea3be412085562a074626055ee76d9d937a11ad903b92965"


def _series(shape: str) -> tuple[np.ndarray, np.ndarray | None, int]:
    if shape == "var":
        return _simulate_var(600, seed=0), None, 3
    x, u = _with_controls(600, seed=1)
    return x, u, 2


@pytest.mark.parametrize("shape", ["var", "controls"])
def test_each_kept_p_value_is_its_test_given_the_other_kept_parents(
    shape: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The p-value at each edge is ``partial_corr_test`` of the parent's column with its target
    given the target's other kept parents, in the order they were selected, to the bit; nan where
    there is no edge. Several read below float32's least number, which a float32 array or one of
    jax without x64 reads as 0."""
    series, controls, max_lag = _series(shape)
    order: list[list[int]] = []  # each target's kept columns, as the selection returned them
    select = chc.discovery._select_parents

    def spy(*arguments: Any) -> dict[int, float]:
        kept = select(*arguments)
        order.append(list(kept))
        return kept

    monkeypatch.setattr(chc.discovery, "_select_parents", spy)
    graph = discover_lagged_parents(series, controls, max_lag=max_lag)
    design, tags = _lagged_design(series, controls, max_lag)
    held = {"state": graph.state_p_values, "control": graph.control_p_values}
    expected = {kind: np.full(np.shape(p), np.nan) for kind, p in held.items() if p is not None}
    for target_index, kept in enumerate(order):
        target = series[max_lag:, target_index]
        for c in kept:
            others = [s for s in kept if s != c]
            given = design[:, others] if others else None
            kind, source, lag = tags[c]
            expected[kind][target_index, source, lag - 1] = partial_corr_test(
                design[:, c], target, given
            )[1]
    assert len(order) == series.shape[1]
    for kind, p_values in expected.items():
        stored = held[kind]
        assert isinstance(stored, np.ndarray)
        assert stored.dtype == np.float64
        assert not stored.flags.writeable
        parents = graph.state_parents if kind == "state" else graph.control_parents
        assert np.array_equal(np.isnan(stored), ~np.asarray(parents))
        assert np.array_equal(stored, p_values, equal_nan=True)
    every = np.concatenate([p[~np.isnan(p)] for p in expected.values()])
    assert len(every) == len(graph.edges())
    assert np.any((every > 0.0) & (every < np.finfo(np.float32).smallest_subnormal))


def test_a_graph_built_by_hand_or_by_tigramite_holds_no_p_values() -> None:
    """``TigramiteDiscovery`` builds its graph from the parents alone, as a caller does."""
    state_parents = np.zeros((2, 2, 1), dtype=bool)
    state_parents[1, 0, 0] = True
    graph = LaggedGraph(jnp.asarray(state_parents), None, 1)
    assert graph.state_p_values is None
    assert graph.control_p_values is None
    with pytest.raises(ValueError, match="a discovered edge carries its test's p-value"):
        graph.causal_graph(["a", "b"])


def test_p_values_are_held_against_their_parents() -> None:
    parents = np.zeros((2, 2, 1), dtype=bool)
    parents[1, 0, 0] = True
    p_values = np.full((2, 2, 1), np.nan)
    p_values[1, 0, 0] = 0.003
    graph = LaggedGraph(jnp.asarray(parents), None, 1, p_values)
    p_values[1, 0, 0] = 0.5  # the graph holds a copy
    assert graph.state_p_values is not None
    assert graph.state_p_values[1, 0, 0] == 0.003
    with pytest.raises(ValueError, match="read-only"):
        graph.state_p_values[1, 0, 0] = 0.5

    p_values[1, 0, 0] = 0.003
    with pytest.raises(ValueError, match="shape"):
        LaggedGraph(jnp.asarray(parents), None, 1, p_values[:, :1])
    for wrong in ((0, 0, 0, 0.5), (1, 0, 0, np.nan), (1, 0, 0, 1.5)):
        bad = p_values.copy()
        bad[wrong[:3]] = wrong[3]
        with pytest.raises(ValueError, match=r"nan exactly where|outside"):
            LaggedGraph(jnp.asarray(parents), None, 1, bad)
    controls = jnp.zeros((2, 1, 1), dtype=bool)
    with pytest.raises(ValueError, match="control_p_values is given exactly"):
        LaggedGraph(jnp.asarray(parents), controls, 1, p_values)
    with pytest.raises(ValueError, match="control_p_values is given exactly"):
        LaggedGraph(jnp.asarray(parents), None, 1, p_values, np.full((2, 1, 1), np.nan))
    with pytest.raises(ValueError, match="without state_p_values"):
        LaggedGraph(jnp.asarray(parents), controls, 1, None, np.full((2, 1, 1), np.nan))


_NODE = re.compile(r"(.+)\[t(?:-([0-9]+))?\]")


def _slice(node: str) -> tuple[str, int]:
    found = _NODE.fullmatch(node)
    assert found is not None
    return found[1], int(found[2] or 0)


@pytest.mark.parametrize("shape", ["var", "controls"])
def test_the_causal_graph_is_the_unrolled_one_and_every_edge_holds_its_p_value(shape: str) -> None:
    """Every copy of an edge, at each slice it is unrolled to, is discovered with the edge's
    p-value; none is left asserted."""
    series, controls, max_lag = _series(shape)
    graph = discover_lagged_parents(series, controls, max_lag=max_lag)
    states = [f"x{i}" for i in range(series.shape[1])]
    levers = None if controls is None else ["u0"]
    unrolled = graph.causal_graph(states, levers)

    names = {"state": states, "control": levers or []}
    triples = [(names[kind][s], states[t], lag) for t, s, lag, kind in graph.edges()]
    expected = CausalGraph.unrolled(triples, lags=max_lag)
    assert unrolled == expected
    assert (unrolled.nodes, unrolled.lags) == (expected.nodes, max_lag)
    assert len(unrolled.provenance) == len(unrolled.edges)

    index = {name: ("state", i) for i, name in enumerate(states)}
    index |= {name: ("control", k) for k, name in enumerate(levers or [])}
    for parent, child in unrolled.edges:
        (source_name, at), (target_name, child_at) = _slice(parent), _slice(child)
        kind, source = index[source_name]
        held = graph.state_p_values if kind == "state" else graph.control_p_values
        assert held is not None
        p_value = held[index[target_name][1], source, at - child_at - 1]
        provenance = unrolled.edge_provenance(parent, child)
        assert (provenance.source, provenance.test) == ("discovered", "partial_corr_test")
        assert provenance.p_value == p_value
        assert type(provenance.p_value) is float


def test_the_causal_graph_refuses_names_that_do_not_fit() -> None:
    x, u = _with_controls(300, seed=1)
    graph = discover_lagged_parents(x, u, max_lag=1)
    with pytest.raises(ValueError, match="3 state components, and 2"):
        graph.causal_graph(["a", "b"], ["u"])
    with pytest.raises(ValueError, match="1 control components, and 0"):
        graph.causal_graph(["a", "b", "c"])
    with pytest.raises(ValueError, match=r"\['a'\] are given twice"):
        graph.causal_graph(["a", "a", "c"], ["u"])
    with pytest.raises(ValueError, match=r"\['c'\] are given twice"):
        graph.causal_graph(["a", "b", "c"], ["c"])
    with pytest.raises(ValueError, match="not one string"):
        graph.causal_graph("abc", ["u"])
    without = discover_lagged_parents(x, max_lag=1)
    with pytest.raises(ValueError, match="0 control components, and 1"):
        without.causal_graph(["a", "b", "c"], ["u"])
    assert without.causal_graph(["a", "b", "c"], []) == without.causal_graph(["a", "b", "c"])


def _one_unit(n: int = 300, dt: float = 0.5, seed: int = 7) -> Panel:
    """One unit's trajectory: a randomised lever ``u`` moves ``x`` a step later."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 1.0, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] + dt * (-0.5 * x[t - 1] + 0.8 * u[t - 1]) + rng.normal(0.0, 0.3)
    frame = {"unit": np.zeros(n), "time": np.arange(n, dtype=float), "x": x, "u": u}
    return Panel.from_frame(frame, unit="unit", time="time", seed=0)


def test_a_discovered_graph_goes_to_the_decision_record_and_replays() -> None:
    """Discovered on one unit's trajectory, the graph's two edges into ``x[t]`` reach
    ``inputs.adjustment`` discovered, each with its p-value bit for bit through JSON text, and the
    record replays."""
    panel = _one_unit()
    x, u = np.asarray(panel.columns["x"]), np.asarray(panel.columns["u"])
    graph = discover_lagged_parents(x[:, None], u[:, None], max_lag=1)
    assert graph.edges() == [(0, 0, 1, "state"), (0, 0, 1, "control")]
    assert graph.state_p_values is not None
    assert graph.control_p_values is not None
    made = prescribe(
        panel,
        levers=[Lever("u", lo=-1.0, hi=1.0, unit_cost=0.05)],
        target=Target("x", value=0.5),
        horizon=5,
        adjustment=graph.causal_graph(["x"], ["u"]),
        dt=0.5,
        tolerance=0.5,
        integrator="euler",
    )
    assert made.certificate.identification == "identified"
    record = json.loads(json.dumps(made.to_json(), allow_nan=False))
    written = record["inputs"]["adjustment"]
    assert written["lags"] == 1
    assert [
        (e["parent"], e["child"], e["source"], e["test"], e["p_value"]) for e in written["edges"]
    ] == [
        ("x[t-1]", "x[t]", "discovered", "partial_corr_test", float(graph.state_p_values[0, 0, 0])),
        (
            "u[t-1]",
            "x[t]",
            "discovered",
            "partial_corr_test",
            float(graph.control_p_values[0, 0, 0]),
        ),
    ]
    assert all(0.0 < e["p_value"] < 1e-45 for e in written["edges"])
    replayed = Prescription.from_json(record, panel)
    assert replayed.to_json() == record
