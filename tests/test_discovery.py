"""chc.discovery: MCI forward selection recovers true lagged parents; naive marginal over-links."""

import numpy as np
import pytest

import chc.discovery
from chc.discovery import discover_lagged_parents
from chc.independence import partial_corr_test

# True sparse VAR(2) over 4 variables, keyed (target, source, lag) -> coefficient.
_COEF = {
    (0, 0, 1): 0.5,
    (0, 1, 1): 0.3,
    (1, 1, 1): 0.5,
    (1, 2, 2): 0.4,
    (2, 2, 1): 0.6,
    (3, 3, 1): 0.5,
    (3, 0, 1): 0.35,
}
_TRUE_EDGES = set(_COEF)


def _simulate_var(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = 4
    x = np.zeros((n, d))
    for t in range(2, n):
        for j in range(d):
            x[t, j] = sum(c * x[t - lag, i] for (jj, i, lag), c in _COEF.items() if jj == j)
            x[t, j] += 0.3 * rng.standard_normal()
    return x


def _f1(found: set, truth: set) -> float:
    true_positive = len(found & truth)
    precision = true_positive / len(found) if found else 0.0
    recall = true_positive / len(truth) if truth else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def test_discovery_recovers_true_lagged_parents() -> None:
    x = _simulate_var(3000, seed=0)
    graph = discover_lagged_parents(x, max_lag=3, alpha=0.01)
    found = {(t, s, lag) for t, s, lag, _kind in graph.edges()}
    assert _f1(found, _TRUE_EDGES) >= 0.9


def test_discovery_beats_naive_marginal_screening() -> None:
    x = _simulate_var(3000, seed=0)
    n, d, max_lag = x.shape[0], x.shape[1], 3
    found = {(t, s, lag) for t, s, lag, _ in discover_lagged_parents(x, max_lag=max_lag).edges()}
    naive = set()  # mark an edge whenever the marginal (unconditioned) lagged correlation is sig
    for j in range(d):
        target = x[max_lag:n, j]
        for i in range(d):
            for lag in range(1, max_lag + 1):
                _, p = partial_corr_test(x[max_lag - lag : n - lag, i], target)
                if float(p) < 0.01:
                    naive.add((j, i, lag))
    assert _f1(found, _TRUE_EDGES) > _f1(naive, _TRUE_EDGES)


def test_a_nan_series_is_refused_rather_than_read_as_a_parent() -> None:
    """A nan reads a nan p-value, and the selection's minimum picked it when it came first: with one
    parent allowed, a nan in the first column made it a parent of every component."""
    series = np.random.default_rng(3).normal(size=(200, 2))
    series[50, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        discover_lagged_parents(series, max_parents=1)


def test_a_nan_p_value_is_not_a_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nan passed the significance test, and the selection's minimum picks whichever comes first:
    with every p-value nan, every candidate became a parent of every component."""
    monkeypatch.setattr(chc.discovery, "partial_corr_test", lambda x, y, z=None: (np.nan, np.nan))
    graph = discover_lagged_parents(np.random.default_rng(3).normal(size=(300, 2)), max_lag=2)
    assert not np.asarray(graph.state_parents).any()


@pytest.mark.parametrize("scale", [1e-8, 1e200])
def test_a_series_reads_the_same_parents_in_any_units(scale: float) -> None:
    """The screen's p-values read units: at 1e-8 every p-value of the series below read 1 or near
    it, and at 1e200 nan, so the edge from 0 to 1 was lost in both."""
    series = np.random.default_rng(3).normal(size=(300, 2))
    series[1:, 1] += 0.8 * series[:-1, 0]
    graph = discover_lagged_parents(scale * series, max_lag=2, max_parents=1)
    assert np.argwhere(np.asarray(graph.state_parents)).tolist() == [[1, 0, 0]]
