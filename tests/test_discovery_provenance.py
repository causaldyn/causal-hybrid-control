"""chc.discovery: each kept edge's p-value, and the graph a decision record carries it on."""

import hashlib

import numpy as np

from chc.discovery import LaggedGraph, discover_lagged_parents
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
