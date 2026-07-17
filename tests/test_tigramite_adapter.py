"""TigramiteDiscovery: a helpful error when tigramite is absent; a LaggedGraph when it is present.

tigramite is GPL-3.0 and never a chc dependency, so the happy path runs only where the user has
installed it (``pip install tigramite joblib``); otherwise the adapter must fail with a clear hint.
"""

import numpy as np
import pytest

from chc.discovery import LaggedGraph, TigramiteDiscovery

try:
    import tigramite  # noqa: F401

    _HAS_TIGRAMITE = True
except ImportError:
    _HAS_TIGRAMITE = False


@pytest.mark.skipif(_HAS_TIGRAMITE, reason="tigramite is installed; the absent-guard cannot fire")
def test_tigramite_discovery_points_to_the_install_when_absent() -> None:
    with pytest.raises(ImportError, match="tigramite"):
        TigramiteDiscovery().discover(np.zeros((20, 2)))


@pytest.mark.skipif(not _HAS_TIGRAMITE, reason="requires tigramite (bring-your-own-env)")
def test_tigramite_discovery_returns_a_lagged_graph() -> None:
    rng = np.random.default_rng(0)
    n, d = 600, 3
    x = np.zeros((n, d))
    for t in range(1, n):
        x[t, 0] = 0.6 * x[t - 1, 0] + rng.standard_normal()
        x[t, 1] = 0.5 * x[t - 1, 0] + rng.standard_normal()  # x0 -> x1
        x[t, 2] = 0.5 * x[t - 1, 2] + rng.standard_normal()
    graph = TigramiteDiscovery(pc_alpha=0.05).discover(x, max_lag=2)
    assert isinstance(graph, LaggedGraph)
    assert (1, 0, 1) in {(t, s, lag) for t, s, lag, _ in graph.edges()}  # recovers x0(t-1) -> x1(t)
