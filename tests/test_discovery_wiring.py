"""chc.discovery wiring: the discovered adjustment set de-biases the effect; node_adjacency sound.

Task C of plans/17. The headline is (ii): discovery *supplies the adjustment set* that turns a
sign-flipped confounded estimate back into the true effect. The residual wiring is covered at the
level of the ``node_adjacency`` helper feeding ``GraphResidual`` -- an end-to-end "beats a dense
MLP" claim only holds where discovery is reliable (sparse, well-excited, per-node-noise systems);
dense spatial coupling is a known hard case for linear discovery, left to the tigramite adapter.
"""

import jax
import jax.numpy as jnp
import numpy as np

from chc.causal import estimate_control_effect
from chc.discovery import LaggedGraph, discover_lagged_parents
from chc.residual import GraphResidual


def _confounded_trajectory(
    n: int, seed: int, a: float = 0.5, b_true: float = 1.0, c: float = 2.0, kappa: float = -1.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.7 * z[t - 1] + rng.standard_normal()  # an autocorrelated confounder
    u = kappa * z + 0.5 * rng.standard_normal(n)  # behaviour policy tied to the confounder
    x = np.zeros(n)
    noise = 0.1 * rng.standard_normal(n)
    for t in range(1, n):
        x[t] = a * x[t - 1] + b_true * u[t - 1] + c * z[t - 1] + noise[t]
    return x, u, z


def test_discovery_supplies_the_adjustment_set_that_debiases_the_effect() -> None:
    b_true = 1.0
    x, u, z = _confounded_trajectory(4000, seed=0, b_true=b_true)
    graph = discover_lagged_parents(np.column_stack([x, z]), np.column_stack([u]), max_lag=1)
    names = ["x", "z"]
    edges = graph.edges()
    parents = {names[s] for target, s, _lag, kind in edges if target == 0 and kind == "state"}
    adjust = tuple(p for p in sorted(parents) if p != "x")  # confounders = state parents minus x
    assert "z" in adjust  # discovery flags the confounder as something to adjust for

    columns = {"x": x[:-1], "u": u[:-1], "z": z[:-1], "x_next": x[1:]}
    data = {name: jnp.asarray(value) for name, value in columns.items()}
    biased = float(estimate_control_effect(data, adjust_for=()))
    adjusted = float(estimate_control_effect(data, adjust_for=adjust))
    assert abs(adjusted - b_true) < 0.1  # the discovered adjustment set recovers the true effect
    assert abs(biased - b_true) > 0.5  # the unadjusted estimate is badly confounded (it sign-flips)


def test_node_adjacency_collapses_component_blocks_to_nodes() -> None:
    state_parents = np.zeros((4, 4, 1), dtype=bool)  # 2 nodes of (pos, vel); components 2i, 2i+1
    state_parents[3, 0, 0] = True  # node1 velocity has node0 position as a lagged parent
    graph = LaggedGraph(jnp.asarray(state_parents), None, max_lag=1)
    node_adjacency = np.asarray(graph.node_adjacency(2))
    assert node_adjacency[1, 0] == 1.0  # node0 -> node1
    assert node_adjacency[0, 1] == 0.0  # but not the reverse


def test_graph_residual_consumes_a_discovered_adjacency() -> None:
    state_parents = np.zeros((4, 4, 1), dtype=bool)
    state_parents[[0, 1, 2, 3], [1, 0, 3, 2], 0] = True  # a 2-node coupling
    graph = LaggedGraph(jnp.asarray(state_parents), None, max_lag=1)
    adjacency = graph.node_adjacency(2)
    residual = GraphResidual(adjacency, node_dim=2, control_dim=1, key=jax.random.key(0))
    output = residual(0.0, jnp.zeros(4), jnp.zeros(1))
    assert output.shape == (4,)  # a discovered adjacency drops straight into the GNN residual
