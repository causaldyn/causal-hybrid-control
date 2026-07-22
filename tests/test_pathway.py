"""chc.pathway: recover a known lever -> mediator -> target chain and its structural laws."""

import numpy as np
import pytest

from chc.pathway import (
    CausalPathway,
    PathwayEdge,
    _ancestor_onsets,
    causal_pathway,
    causal_pathway_certificate,
)


def _lever_mediator_target(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """u -> m (+0.8) -> x (-0.9); an independent white-noise decoy has no path to x."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n)
    decoy = rng.standard_normal(n)
    m = np.zeros(n)
    x = np.zeros(n)
    noise = 0.1 * rng.standard_normal((n, 2))
    for t in range(1, n):
        m[t] = 0.5 * m[t - 1] + 0.8 * u[t - 1] + noise[t, 0]
        x[t] = 0.6 * x[t - 1] - 0.9 * m[t - 1] + noise[t, 1]
    return np.column_stack([x, m, decoy]), u.reshape(-1, 1)


def test_certificate_recovers_the_signed_chain_and_laws() -> None:
    cert = causal_pathway_certificate(seed=0)
    assert cert.ok  # signs, onset lag, weakest-link decoy, and the L2 geometric tail all hold
    assert cert.mediator_sign == -1  # m -> x is negative
    assert cert.control_sign == -1  # u -> m -> x is the +0.8 * -0.9 sign product
    assert cert.control_onset_lag == 2  # u reaches x through one mediator step
    assert cert.truncation_tail <= cert.geometric_bound  # L2: truncating loses geometrically little


def test_mediator_outranks_the_indirect_lever() -> None:
    series, controls = _lever_mediator_target(6000, seed=1)
    path = causal_pathway(series, target=0, controls=controls, horizon=8, max_lag=3)
    by_key = {(edge.kind, edge.source): edge for edge in path.edges}
    mediator, lever = by_key[("state", 1)], by_key[("control", 0)]
    assert mediator.contribution > lever.contribution  # the direct driver dominates the indirect
    assert mediator.sign == -1
    assert lever.sign == -1
    assert path.edges[0].source == 1  # m is ranked first
    assert path.edges[0].kind == "state"


def test_the_indirect_lever_is_the_only_actionable_edge() -> None:
    series, controls = _lever_mediator_target(6000, seed=2)
    path = causal_pathway(series, target=0, controls=controls, horizon=8, max_lag=3)
    actionable = path.actionable()
    assert [edge.kind for edge in actionable] == ["control"]  # only u is a lever
    assert actionable[0].source == 0
    assert actionable[0].actionable


def test_unlinked_decoy_is_negligible_in_the_pathway() -> None:
    series, controls = _lever_mediator_target(6000, seed=3)
    path = causal_pathway(series, target=0, controls=controls, horizon=8, max_lag=3)
    by_key = {(edge.kind, edge.source): edge for edge in path.edges}
    decoy = by_key.get(("state", 2))
    # weakest-link (L3): an unlinked variable either never enters or contributes negligibly
    assert decoy is None or decoy.contribution <= 0.1 * by_key[("state", 1)].contribution


def test_ancestor_onsets_take_the_shortest_lagged_walk() -> None:
    # u -> m (lag 1), m -> x (lag 1), x -> x (lag 1); x is target 0, m is 1, u is control 0.
    edges = [(0, 0, 1, "state"), (0, 1, 1, "state"), (1, 0, 1, "control"), (1, 1, 1, "state")]
    onsets = _ancestor_onsets(edges, target=0)
    assert onsets[("state", 1)] == 1  # the mediator reaches x in one step
    assert onsets[("control", 0)] == 2  # the lever reaches x through the mediator: 1 + 1
    assert onsets[("state", 0)] == 1  # the target's own autoregressive self-loop


def test_ancestor_walk_terminates_on_a_lag_cycle() -> None:
    # x <-> m mutual lagged coupling is a cycle; the strict-decrease guard must still terminate.
    edges = [(0, 1, 1, "state"), (1, 0, 1, "state")]
    onsets = _ancestor_onsets(edges, target=0)
    assert onsets[("state", 1)] == 1  # m -> x
    assert onsets[("state", 0)] == 2  # x -> m -> x, shortest return is two steps


def test_summary_lists_every_edge_ranked() -> None:
    series, controls = _lever_mediator_target(4000, seed=4)
    path = causal_pathway(series, target=0, controls=controls, horizon=6, max_lag=3)
    summary = path.summary()
    assert summary.startswith("causal pathway -> x0")
    assert summary.count("\n") == len(path.edges)  # header + one line per ranked edge


def test_returned_pathway_is_a_frozen_ranked_structure() -> None:
    series, controls = _lever_mediator_target(4000, seed=5)
    path = causal_pathway(series, target=0, controls=controls, horizon=6, max_lag=3)
    assert isinstance(path, CausalPathway)
    assert all(isinstance(edge, PathwayEdge) for edge in path.edges)
    contributions = [edge.contribution for edge in path.edges]
    assert contributions == sorted(contributions, reverse=True)  # strongest-first ordering


def test_series_must_be_two_dimensional() -> None:
    with pytest.raises(ValueError, match="must be"):
        causal_pathway(np.zeros(100), target=0)
