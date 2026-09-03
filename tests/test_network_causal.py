"""Network causal gate: naive DML is blind to spillover; network DML recovers direct + spillover."""

import itertools

import jax
import numpy as np
import pytest
from numpy.typing import NDArray

from chc.estimators import DoubleML
from chc.network_causal import (
    ConfoundedNetworkSystem,
    DelayedNetworkPanel,
    _fold_chunks,
    cycle_shells,
    estimate_network_effects,
    estimate_network_effects_gnn,
    estimate_propagation,
    graph_shells,
    within_ar1,
)
from chc.regret import _fold_sandwich, delayed_network_certificate, fold_heuristic_certificate


def _data() -> dict[str, jax.Array]:
    return ConfoundedNetworkSystem().sample(jax.random.key(0))


def test_network_dml_recovers_direct_and_spillover() -> None:
    effects = estimate_network_effects(_data())
    assert abs(effects["direct"] - 1.0) < 0.1  # true direct = 1.0
    assert abs(effects["spillover"] - 0.6) < 0.1  # true spillover = 0.6


def test_naive_dml_is_blind_to_spillover() -> None:
    """The whole point: naive DML estimates a single effect and misses the interference channel."""
    data = _data()
    naive_direct = float(DoubleML().estimate(data, covariates=("x", "z")).effect)
    effects = estimate_network_effects(data)
    # naive's per-unit read (direct only) undershoots the true total effect direct + spillover:
    assert naive_direct < effects["direct"] + effects["spillover"] - 0.2


def test_gnn_nuisance_recovers_direct_and_spillover() -> None:
    """The learned message-passing nuisance recovers both effects from raw features + the graph."""
    data = ConfoundedNetworkSystem(n=1500).sample(jax.random.key(1))
    effects = estimate_network_effects_gnn(
        data, data["neighbours"], features=("x", "z"), hidden=12, steps=400, folds=2
    )
    assert abs(effects["direct"] - 1.0) < 0.2  # true direct = 1.0 (no hand-crafted features)
    assert abs(effects["spillover"] - 0.6) < 0.2  # true spillover = 0.6


def test_propagation_estimate_recovers_the_lag_the_persistence_and_the_truncation() -> None:
    """Result 51's ``delta`` and ``phi`` are fitted here, not assumed as model parameters.

    The panel's spillover reaches distance 2 (``gammas`` has three entries), so shell 3 carries no
    direct edge and its apparent peak must stop advancing -- which is what pins ``n_shells``.
    """
    m, clusters, times = 6, 80, 40
    for lag in (1, 2, 3):
        panel = DelayedNetworkPanel(
            lag=lag, n_clusters=clusters, cluster_size=m, n_times=times, phi=0.6
        )
        data = panel.sample(jax.random.key(7))
        shape = (clusters, m, times)
        u = np.asarray(data["u"]).reshape(shape)
        y = np.asarray(data["x_next"]).reshape(shape)
        got = estimate_propagation(u, y, cycle_shells(m, 3), horizon=3 * lag + 2, n_resamples=40)
        assert got.delay == lag  # an exact multiple of dt is identified exactly
        assert not got.censored
        assert got.lo <= lag <= got.hi
        assert got.n_shells == 2  # len(gammas) - 1, read off the flat tail
        assert abs(got.phi - 0.6) < 0.02  # the raw within estimator sits at 0.559 here


def test_within_ar1_removes_the_bias_the_within_transform_induces() -> None:
    """The correction is worth reporting only if the uncorrected number is visibly wrong."""
    m, clusters, times, phi = 6, 120, 40, 0.6
    data = DelayedNetworkPanel(n_clusters=clusters, cluster_size=m, n_times=times, phi=phi).sample(
        jax.random.key(5)
    )
    u = np.asarray(data["u"]).reshape(clusters, m, times)
    raw = within_ar1(u, corrections=0)
    assert abs(raw - (phi - (1.0 + phi) / (times - 1))) < 0.01  # Nickell, to the predicted size
    assert abs(within_ar1(u) - phi) < 0.01


def test_fold_groups_default_is_the_historical_path_and_groups_stay_intact() -> None:
    """The kwarg is opt-in: ``None`` must reproduce row permutation exactly, not approximately."""
    data = DelayedNetworkPanel(n_clusters=20, cluster_size=6, n_times=12).sample(jax.random.key(1))
    assert estimate_network_effects(data, folds=2, seed=0) == estimate_network_effects(
        data, folds=2, seed=0, fold_groups=None
    )
    chunks = _fold_chunks(int(data["unit"].shape[0]), 2, 0, data["unit"])
    labels = [set(np.asarray(data["unit"])[np.asarray(c)].tolist()) for c in chunks]
    assert not labels[0] & labels[1]  # no unit is split across folds
    assert labels[0] | labels[1] == set(range(6))


def test_the_delayed_network_law_is_an_estimator_variance_ratio() -> None:
    """Result 51's ``Psi`` predicts the VARIANCE OF AN ESTIMATOR, not only a property of a process.

    ``Psi = N^2 tr(A Sigma A)/(tr(A)^2 tr Sigma)`` was forced algebraically by
    ``validation/delayed_exposure_gate.mac`` STEP 6. For ``theta_hat = u'A eps / u'A u`` with an
    isotropic regressor independent of a disturbance carrying ``Sigma``, ``Var(theta_hat)``
    concentrates on ``tr(A Sigma A)/tr(A)^2``, so the ratio between two fold partitions is exactly
    the ratio of their ``Psi``. That concentration is what this measures; the certificate itself
    only checks ``Psi`` against the process.
    """
    m, times, folds, lag, phi = 6, 24, 2, 1, 0.6
    gammas = np.array([1.0, 0.7, 0.4])
    shells = cycle_shells(m, gammas.size - 1)
    gap = np.arange(times)[:, None] - np.arange(times)[None, :]
    sigma = sum(
        gammas[d] * gammas[e] * np.kron(shells[d] @ shells[e], phi ** np.abs(gap - lag * (d - e)))
        for d in range(gammas.size)
        for e in range(gammas.size)
    )
    values, vectors = np.linalg.eigh(sigma)
    assert values.min() > 0.0  # the construction is PSD, so the draw below is well posed
    root = vectors @ np.diag(np.sqrt(values)) @ vectors.T

    curve = delayed_network_certificate(
        cluster_size=m, n_times=times, lag=lag, phi_grid=(phi,), n_draws=2000
    )
    predicted = float(curve.psi_law_parity[0] / curve.psi_law_block[0])
    assert predicted < 0.8  # the two partitions must actually differ, or the test is vacuous

    rng, draws = np.random.default_rng(0), 25_000
    measured = []
    for fold in (np.arange(m) % folds, (np.arange(m) >= m // folds).astype(int)):
        a = np.kron(_fold_sandwich(m, folds, fold), np.eye(times))
        u = rng.standard_normal((draws, m * times)) @ a
        eps = rng.standard_normal((draws, m * times)) @ root
        theta = np.einsum("ni,ni->n", u, eps) / np.einsum("ni,ij,nj->n", u, np.eye(m * times), u)
        measured.append(theta.var(ddof=1))
    ratio = measured[0] / measured[1]
    assert abs(ratio - predicted) < 0.05  # ~5 Monte-Carlo standard errors at this draw count


def test_the_fold_design_crossover_is_a_threshold_on_phi_to_the_delay() -> None:
    """Neither fold partition wins outright: they swap at one point, and it moves with the delay.

    ``tr(Au)`` counts eigenvalue multiplicities and ``v0`` never sees the fold operator, so both
    normalisers are partition-free and equal ``Psi`` is a bare root of ``u_parity - u_block``. That
    difference carries no ``delta``, so the crossover is fixed in ``x = phi^delta`` -- which is
    exactly why reading ``phi`` off a panel is not enough to choose a fold design.
    ``proofs/delayed_network_exposure.v`` proves the root and its direction.
    """
    exact = (49.0 - np.sqrt(1101.0)) / 40.0  # Maxima: root of 32x^2 - (392/5)x + 26 on C_6
    curves = [
        delayed_network_certificate(
            cluster_size=6, n_times=24, lag=lag, phi_grid=(0.5,), n_draws=200
        )
        for lag in (1, 2, 3)
    ]
    for curve in curves:
        assert curve.trace_gap == 0.0  # the normaliser really is partition-free
        assert abs(curve.crossover - exact) < 1e-12
    # the phi threshold rises with the delay, so one persistence flips the recommendation
    thresholds = [c.crossover ** (1.0 / lag) for c, lag in zip(curves, (1, 2, 3), strict=True)]
    assert thresholds[0] < thresholds[1] < thresholds[2]
    assert abs(thresholds[0] - 0.3955) < 1e-3
    assert abs(thresholds[1] - 0.6289) < 1e-3

    dense = delayed_network_certificate(
        cluster_size=6, n_times=24, lag=1, phi_grid=(0.1, 0.9), n_draws=200
    )
    assert dense.psi_law_parity[0] > dense.psi_law_block[0]  # below x*: aligned folds win
    assert dense.psi_law_parity[1] < dense.psi_law_block[1]  # above x*: alternating folds win


def test_fold_choice_moves_realised_coverage_not_just_variance() -> None:
    """The point of Results 43 and 51 is INTERVALS, so measure intervals, not only second moments.

    The cross-fit hat ``M = I - F_opp`` leaves the estimator exactly unbiased -- the held-out
    prediction reproduces the design, so the moment stays orthogonal for ANY fold assignment. What
    the partition moves is the reported interval: a textbook standard error computed after row folds
    understates the realised spread in both arms, and by different amounts, so both under-cover and
    the misaligned one under-covers more. At ``phi = 0.6`` with ``delta = 1``,
    ``x = 0.6`` sits above
    the crossover ``0.395``, where the law says the ALTERNATING partition wins -- and it is the
    alternating arm that keeps the better coverage.
    """
    m, times, folds, lag, phi = 6, 24, 2, 1, 0.6
    clusters, draws, truth = 40, 3000, 1.0
    gammas = np.array([1.0, 0.7, 0.4])
    shells = cycle_shells(m, gammas.size - 1)
    gap = np.arange(times)[:, None] - np.arange(times)[None, :]
    sigma = sum(
        gammas[d] * gammas[e] * np.kron(shells[d] @ shells[e], phi ** np.abs(gap - lag * (d - e)))
        for d in range(gammas.size)
        for e in range(gammas.size)
    )
    root = np.linalg.cholesky(sigma + 1e-12 * np.eye(m * times))

    rng = np.random.default_rng(0)
    coverage, spread = {}, {}
    for name, fold in (("parity", np.arange(m) % folds), ("block", (np.arange(m) >= m // 2) * 1)):
        opposite = (fold[:, None] != fold[None, :]).astype(float)
        hat = np.kron(np.eye(m) - opposite / opposite.sum(1, keepdims=True), np.eye(times))
        # the cross-fit hat's Gram IS the projector the law is written in terms of
        assert np.allclose(hat.T @ hat, np.kron(_fold_sandwich(m, folds, fold), np.eye(times)))
        hits, estimates = 0, []
        for _ in range(draws):
            u = rng.standard_normal((clusters, m * times))
            eps = rng.standard_normal((clusters, m * times)) @ root.T
            ru = (u @ hat.T).reshape(-1)
            ry = ((truth * u + eps) @ hat.T).reshape(-1)
            theta = ru @ ry / (ru @ ru)
            resid = ry - theta * ru
            se = np.sqrt(resid @ resid / (resid.size - 1) / (ru @ ru))
            estimates.append(theta)
            hits += abs(theta - truth) <= 1.96 * se
        coverage[name] = hits / draws
        spread[name] = float(np.std(estimates, ddof=1))
        assert abs(float(np.mean(estimates)) - truth) < 0.005  # the damage is NOT a bias

    assert coverage["parity"] < 0.90  # both arms under-cover a nominal 95%
    assert coverage["block"] < 0.90
    assert coverage["parity"] - coverage["block"] > 0.02  # ~4 standard errors at this draw count

    curve = delayed_network_certificate(
        cluster_size=m, n_times=times, lag=lag, phi_grid=(phi,), n_draws=200
    )
    predicted = float(curve.psi_law_parity[0] / curve.psi_law_block[0])
    assert abs((spread["parity"] / spread["block"]) ** 2 - predicted) < 0.05


def _balanced_partitions(m: int) -> list[NDArray[np.int_]]:
    return [np.array(b) for b in itertools.product([0, 1], repeat=m) if sum(b) == m // 2]


def _path_shells(m: int) -> list[NDArray[np.float64]]:
    index = np.arange(m)
    return [np.eye(m), (np.abs(index[:, None] - index[None, :]) == 1) * 1.0]


def _crossover_inputs(
    adjacency: NDArray[np.float64], fold: NDArray[np.int_], gamma_1: float
) -> tuple[NDArray[np.float64], float, float]:
    """The two coefficients of the D = 1 polynomial, plus the two fold-overlap counts."""
    m = adjacency.shape[0]
    shells, gammas = [np.eye(m), adjacency], (1.0, gamma_1)
    sandwich = _fold_sandwich(m, 2, fold)
    coefficients = np.array(
        [
            sum(
                gammas[d] * gammas[e] * np.trace(sandwich @ shells[d] @ shells[e] @ sandwich)
                for d in range(2)
                for e in range(2)
                if abs(d - e) == lag
            )
            for lag in range(2)
        ]
    )
    same = (fold[:, None] == fold[None, :]).astype(float)
    return (
        coefficients,
        float((same * (adjacency @ adjacency)).sum()),
        float((same * adjacency).sum() / 2),
    )


def test_the_d1_crossover_is_the_spillover_decay_times_a_fold_overlap_ratio() -> None:
    """``x*(D=1) = -(g1/(4 g0)) * Delta(same-fold 2-walks) / Delta(same-fold edges)``.

    The ``d = 0`` block of ``u_0`` is ``tr(Au^2) = m - r^4 + (r^4-1)K``, which counts eigenvalue
    multiplicities and so is partition-free just as ``tr(Au)`` is; it cancels in the difference and
    takes ``r``, ``K`` and ``m`` with it. What is left is linear in the spillover decay ratio times
    a ratio of two integer counts -- derived in ``validation/delayed_network_exposure.mac`` STEP 11
    and proved in ``proofs/delayed_network_exposure.v``.
    """
    cycle = cycle_shells(6, 1)[1]
    for adjacency in (cycle, cycle_shells(8, 1)[1], _path_shells(6)[1]):
        for gamma_1 in (0.7, 1.4):
            m = adjacency.shape[0]
            rows = [_crossover_inputs(adjacency, f, gamma_1) for f in _balanced_partitions(m)]
            for (u_a, w_a, e_a), (u_b, w_b, e_b) in itertools.combinations(rows, 2):
                if abs(u_a[1] - u_b[1]) < 1e-9 or abs(e_a - e_b) < 1e-9:
                    continue
                got = -(u_a[0] - u_b[0]) / (u_a[1] - u_b[1])
                want = -(gamma_1 / 4.0) * (w_a - w_b) / (e_a - e_b)
                assert abs(got - want) < 1e-9


def test_the_crossover_is_graph_dependent_unlike_theta_star() -> None:
    """``theta*`` is graph-free; ``x*`` is not, because edges do not determine 2-walks.

    Two graphs can put their partitions at the same pair of same-fold EDGE fractions and still
    disagree on where the two designs swap, since the numerator counts same-fold 2-WALKS. This
    exhibits such a pair, so the natural generalisation of (d) is refuted by construction rather
    than left open.
    """

    def by_edge_fraction(adjacency: NDArray[np.float64]) -> dict[float, tuple]:
        table = {}
        for fold in _balanced_partitions(adjacency.shape[0]):
            same = (fold[:, None] == fold[None, :]).astype(float)
            share = float((same * adjacency).sum() / adjacency.sum())
            table.setdefault(round(share, 9), _crossover_inputs(adjacency, fold, 0.7))
        return table

    shapes = {
        "C10": by_edge_fraction(cycle_shells(10, 1)[1]),
        "P6": by_edge_fraction(_path_shells(6)[1]),
    }
    shared = set(shapes["C10"]) & set(shapes["P6"])
    clashes = []
    for left, right in itertools.combinations(sorted(shared), 2):
        values = []
        for table in shapes.values():
            (u_a, _, _), (u_b, _, _) = table[left], table[right]
            if abs(u_a[1] - u_b[1]) < 1e-9:
                break
            values.append(-(u_a[0] - u_b[0]) / (u_a[1] - u_b[1]))
        if len(values) == 2 and abs(values[0] - values[1]) > 1e-6:
            clashes.append((left, right, values))
    assert clashes, "no (theta_1, theta_2) pair separated the two graphs -- check the sweep"


def test_graph_shells_reproduces_the_cycle_and_partitions_any_graph() -> None:
    for m, dmax in ((6, 2), (8, 3), (12, 4)):
        adjacency = np.zeros((m, m))
        i = np.arange(m)
        adjacency[i, (i + 1) % m] = adjacency[(i + 1) % m, i] = 1.0
        general = graph_shells(adjacency, dmax)
        special = cycle_shells(m, dmax)
        assert all(np.array_equal(a, b) for a, b in zip(general, special, strict=True))

    # A path is the simplest non-vertex-transitive case: the ends see one neighbour, the interior
    # two, which is exactly the symmetry Result 52's closed form assumes and a path does not have.
    m = 6
    path = np.zeros((m, m))
    for i in range(m - 1):
        path[i, i + 1] = path[i + 1, i] = 1.0
    shells = graph_shells(path, 2)
    assert list(shells[1].sum(1)) == [1.0, 2.0, 2.0, 2.0, 2.0, 1.0]
    # The structural lemma Result 51 rests on: a vertex sits at exactly one distance from i.
    for d in range(3):
        for e in range(3):
            if d != e:
                assert abs(float(np.trace(shells[d] @ shells[e]))) < 1e-12

    # A vertex in another component belongs to no shell -- a truncated spillover model says
    # nothing about it, and inventing a distance would be worse than omitting it.
    split = np.zeros((4, 4))
    split[0, 1] = split[1, 0] = split[2, 3] = split[3, 2] = 1.0
    assert float(graph_shells(split, 2)[2].sum()) == 0.0

    for bad, message in (
        (np.zeros((2, 3)), "square"),
        (np.array([[0.0, 1.0], [0.0, 0.0]]), "symmetric"),
        (np.eye(2), "self-loops"),
    ):
        with pytest.raises(ValueError, match=message):
            graph_shells(bad, 1)


def test_the_fold_heuristic_finds_the_exact_optimum_off_the_cycle() -> None:
    # Result 52's design law is closed-form only on vertex-transitive graphs, and its honest-scope
    # note says the problem "degrades to combinatorial search" beyond them. It degrades in the
    # closed form, not in the answer: on nine topologies -- eight non-transitive, four random --
    # the spectral-plus-swap fallback returns exactly what enumeration does.
    certificate = fold_heuristic_certificate(m=12)
    assert certificate.ok
    assert len(certificate.names) == 9
    assert certificate.worst_ratio == pytest.approx(1.0, abs=1e-9)
    # The Ky Fan bound is a separate quantity and is loose even at the true optimum; conflating
    # "the design is optimal" with "the certificate is tight" is the easy mistake here.
    assert float(certificate.kyfan_gap.min()) > 0.0
