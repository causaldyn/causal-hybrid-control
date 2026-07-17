"""chc.discovery: MCI forward selection recovers true lagged parents; naive marginal over-links."""

import numpy as np

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
