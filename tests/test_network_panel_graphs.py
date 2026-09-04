"""The delayed-network panel off the cycle: any regular graph, and honest uncertainty."""

import jax
import numpy as np
import pytest

from chc.network_causal import (
    DelayedNetworkPanel,
    cycle_shells,
    estimate_network_effects,
    graph_shells,
    torus_adjacency,
)

CYCLE_6 = tuple(tuple(1 if (i - j) % 6 in (1, 5) else 0 for j in range(6)) for i in range(6))


def test_torus_adjacency_is_a_four_regular_simple_graph() -> None:
    adjacency = np.asarray(torus_adjacency(3, 4), dtype=float)
    assert adjacency.shape == (12, 12)
    assert np.array_equal(adjacency, adjacency.T)
    assert np.all(np.diag(adjacency) == 0)
    assert np.all(adjacency.sum(axis=1) == 4)


def test_torus_shells_partition_the_reachable_set() -> None:
    shells = graph_shells(np.asarray(torus_adjacency(4, 4), dtype=float), 2)
    for d in range(3):
        for e in range(3):
            if d != e:
                assert abs(float(np.trace(shells[d] @ shells[e]))) < 1e-12


def test_a_torus_is_vertex_transitive_but_not_circulant() -> None:
    """Which is why the design law has no closed form there and the exact route cannot run."""
    adjacency = np.asarray(torus_adjacency(3, 4), dtype=float)
    m = adjacency.shape[0]
    shifts = np.arange(m)
    rebuilt = adjacency[0][(shifts[:, None] - shifts[None, :]) % m]
    assert not np.allclose(adjacency, rebuilt)
    assert len(set(adjacency.sum(axis=1))) == 1  # regular, so every unit still looks alike


@pytest.mark.parametrize("side", [1, 2])
def test_a_degenerate_torus_is_refused(side: int) -> None:
    with pytest.raises(ValueError, match="both sides >= 3"):
        torus_adjacency(side, 5)


def test_an_explicit_cycle_reproduces_the_default_draw() -> None:
    key = jax.random.key(0)
    default = DelayedNetworkPanel(n_clusters=3, cluster_size=6, n_times=8).sample(key)
    explicit = DelayedNetworkPanel(n_clusters=3, cluster_size=6, n_times=8, graph=CYCLE_6).sample(
        key
    )
    for column in default:
        assert np.array_equal(np.asarray(default[column]), np.asarray(explicit[column]))


def test_the_torus_panel_widens_the_neighbours_column_and_recovers_the_effects() -> None:
    panel = DelayedNetworkPanel(
        n_clusters=20, cluster_size=12, n_times=16, graph=torus_adjacency(3, 4)
    )
    data = panel.sample(jax.random.key(0))
    assert np.asarray(data["neighbours"]).shape == (20 * 12 * 16, 4)
    out = estimate_network_effects(data, folds=2)
    assert out["direct"] == pytest.approx(panel.b_direct, abs=0.05)
    assert out["spillover"] == pytest.approx(panel.b_spillover, abs=0.05)


def test_an_irregular_graph_is_refused() -> None:
    path = tuple(tuple(1 if abs(i - j) == 1 else 0 for j in range(6)) for i in range(6))
    with pytest.raises(ValueError, match="must be regular"):
        DelayedNetworkPanel(n_clusters=2, cluster_size=6, n_times=6, graph=path).sample(
            jax.random.key(0)
        )


def test_a_graph_of_the_wrong_order_is_refused() -> None:
    with pytest.raises(ValueError, match="vertices for cluster_size"):
        DelayedNetworkPanel(n_clusters=2, cluster_size=8, n_times=6, graph=CYCLE_6).sample(
            jax.random.key(0)
        )


def test_clustering_inflates_the_spillover_se_and_leaves_the_direct_one_alone() -> None:
    """Which coefficient the cluster correction reaches is decided by which regressor is smooth.

    The sandwich meat's cross terms are ``E[x_i x_j eps_i eps_j]``. The panel's ``eta`` is i.i.d.
    across units, so the DIRECT regressor is white and its cross terms vanish however correlated
    the disturbance is; the EXPOSURE is a shell sum and is spatially smooth, so its cross terms
    survive. Measured over eight seeds: clustered / HC is ``0.93`` for direct against ``1.87`` for
    spillover -- reporting one i.i.d. SE for both would understate exactly the coefficient
    interference is about.
    """
    panel = DelayedNetworkPanel(n_clusters=16, cluster_size=8, n_times=16, disturbance_scale=2.0)
    data = panel.sample(jax.random.key(0))
    clustered = estimate_network_effects(data, folds=2)
    flat = estimate_network_effects({k: v for k, v in data.items() if k != "cid"}, folds=2)
    assert clustered["direct"] == pytest.approx(flat["direct"], abs=1e-9)
    assert clustered["spillover"] == pytest.approx(flat["spillover"], abs=1e-9)
    assert clustered["spillover_se"] > 1.2 * flat["spillover_se"]
    assert 0.5 < clustered["direct_se"] / flat["direct_se"] < 1.3


def test_the_intervals_cover_the_truth_across_seeds() -> None:
    panel = DelayedNetworkPanel(n_clusters=20, cluster_size=8, n_times=16, disturbance_scale=2.0)
    hits = {"direct": 0, "spillover": 0}
    seeds = 12
    for seed in range(seeds):
        out = estimate_network_effects(panel.sample(jax.random.key(seed)), folds=2)
        for name, truth in (("direct", panel.b_direct), ("spillover", panel.b_spillover)):
            hits[name] += abs(out[name] - truth) <= 1.96 * out[name + "_se"]
    assert hits["direct"] >= seeds - 3
    assert hits["spillover"] >= seeds - 3


def test_neighbour_exclusion_is_off_by_default_and_changes_the_fit_when_on() -> None:
    panel = DelayedNetworkPanel(n_clusters=20, cluster_size=12, n_times=16)
    data = panel.sample(jax.random.key(0))
    groups = np.asarray(data["unit"]) * 4 // 12
    plain = estimate_network_effects(data, folds=4, fold_groups=groups)
    excluded = estimate_network_effects(data, folds=4, fold_groups=groups, exclude_neighbours=True)
    assert plain["direct"] != excluded["direct"]
    assert estimate_network_effects(
        data, folds=4, fold_groups=groups, exclude_neighbours=False
    ) == (plain)


def test_neighbour_exclusion_refuses_a_graph_it_cannot_survive() -> None:
    """On a dense-enough graph a two-fold split's test neighbourhood covers the training fold."""
    panel = DelayedNetworkPanel(
        n_clusters=6, cluster_size=12, n_times=8, graph=torus_adjacency(3, 4)
    )
    data = panel.sample(jax.random.key(0))
    groups = np.asarray(data["unit"]) * 2 // 12
    with pytest.raises(ValueError, match="after neighbour exclusion"):
        estimate_network_effects(data, folds=2, fold_groups=groups, exclude_neighbours=True)


def test_the_designed_fold_is_expressible_as_fold_groups() -> None:
    from chc.regret import optimal_fold_partition

    m = 12
    design = optimal_fold_partition(cycle_shells(m, 2), (1.0, 0.7, 0.4), 0.6, lag=1)
    panel = DelayedNetworkPanel(n_clusters=6, cluster_size=m, n_times=8)
    data = panel.sample(jax.random.key(0))
    labels = design.fold[np.asarray(data["unit"])]
    out = estimate_network_effects(data, folds=2, fold_groups=labels)
    assert set(np.unique(labels).tolist()) == {0, 1}
    assert out["direct"] == pytest.approx(panel.b_direct, abs=0.1)
