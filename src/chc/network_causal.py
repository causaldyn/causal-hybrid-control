"""Network / interference-aware causal inference: direct + spillover effects when SUTVA fails.

On a network a unit's outcome depends on its *neighbours'* treatments too (interference),
so an effect estimate that ignores the exposure is blind to the spillover. This adds a cross-fitted,
Neyman-orthogonal Double ML that estimates **both** the direct and the spillover effect, using
graph-aggregated (mean-neighbour) covariates as the nuisance features -- the lean, JAX-native
counterpart of GNN-nuisance network DML (cf. arXiv 2509.18484, 2211.07823); see ``plans/16``.
Identifiability caveat: like confounding, the spillover is only recovered under the stated exposure
model (mean neighbour treatment); real pilots need cluster/geo randomisation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from numpy.typing import NDArray

from chc.causal import _polynomial_features, _ridge_predict
from chc.irf import peak_lag


def cycle_shells(m: int, dmax: int) -> list[NDArray[np.float64]]:
    """0/1 distance-``d`` indicator matrices of the cycle ``C_m``, for ``d = 0..dmax``.

    ``S_d[i,j] = 1`` iff ``j`` is exactly ``d`` hops from ``i``. The shells partition the graph, so
    ``tr(S_d S_e) = 0`` for ``d != e`` -- a vertex cannot sit at two distinct distances from ``i``.
    That is the structural lemma Result 51 rests on.
    """
    i = np.arange(m)
    gap = np.abs(i[:, None] - i[None, :])
    dist = np.minimum(gap, m - gap)
    return [(dist == d).astype(float) for d in range(dmax + 1)]


def propagate_shells(
    innovations: NDArray[np.float64],
    shells: Sequence[NDArray[np.float64]],
    gammas: Sequence[float] | NDArray[np.float64],
    lag: int,
    n_times: int,
) -> NDArray[np.float64]:
    """Sum shell-``d`` neighbours' innovations arriving ``lag*d`` steps late.

    ``x_i(t) = sum_d gamma_d * sum_{j : dist(i,j) = d} innovations_j(t - lag*d)``, read off a
    burnt-in window so the result is stationary. ``innovations`` is ``(..., m, span)`` with
    ``span >= n_times + lag*(len(gammas)-1)``; the returned window is the LAST ``n_times`` columns
    reachable by every shell, so every lag indexes real history rather than the burn-in edge.

    Shared by ``DelayedNetworkPanel`` and ``chc.regret.delayed_network_certificate`` on purpose: the
    law and the design it is measured on must not drift apart.
    """
    dmax = len(gammas) - 1
    base = innovations.shape[-1] - n_times
    if base < lag * dmax:
        msg = (
            f"innovation span {innovations.shape[-1]} is too short: need at least "
            f"{n_times + lag * dmax} columns for {n_times} times at lag {lag} over {dmax} shells"
        )
        raise ValueError(msg)
    out = np.zeros((*innovations.shape[:-1], n_times))
    for d in range(dmax + 1):
        start = base - lag * d
        out += gammas[d] * np.einsum(
            "ij,...jt->...it", shells[d], innovations[..., start : start + n_times]
        )
    return out


def ar1_innovations(
    rng: np.random.Generator, shape: tuple[int, ...], span: int, phi: float
) -> NDArray[np.float64]:
    """Unit-variance AR(1) paths of length ``span``: ``z_t = phi z_{t-1} + sqrt(1-phi^2) e_t``.

    Drawn at the stationary variance from the first column, so no burn-in is needed for the MARGINAL
    scale; ``propagate_shells`` still needs slack for the lags.
    """
    z = np.empty((*shape, span))
    z[..., 0] = rng.standard_normal(shape)
    innov = np.sqrt(max(0.0, 1.0 - phi**2))
    for t in range(1, span):
        z[..., t] = phi * z[..., t - 1] + innov * rng.standard_normal(shape)
    return z


@dataclass(frozen=True)
class ConfoundedNetworkSystem:
    """Units on a graph; ``y = a*x + direct*u + spillover*e + c*z``, ``u`` confounded by ``z``.

    The confounder ``z`` is spatially smooth (neighbours share it), so the treatment ``u`` and the
    exposure ``e`` (mean neighbour treatment) correlate; an estimate that omits ``e`` misses it.
    """

    n: int = 4000
    degree: int = 6
    a: float = 0.5
    b_direct: float = 1.0
    b_spillover: float = 0.6
    c: float = 2.0
    kappa: float = -1.5
    noise_scale: float = 0.1

    def sample(self, key: Array) -> dict[str, Array]:
        """Draw ``n`` units as columns ``x, z, u, e, x_nb, z_nb, x_next`` (``e`` = exposure)."""
        k_x, k_z, k_eta, k_nb, k_noise = jax.random.split(key, 5)
        neighbours = jax.random.randint(k_nb, (self.n, self.degree), 0, self.n)
        x = jax.random.normal(k_x, (self.n,))
        raw_z = jax.random.normal(k_z, (self.n,))
        z = 0.5 * raw_z + 0.5 * jnp.mean(raw_z[neighbours], axis=1)  # spatially smooth confounder
        u = self.kappa * z + jax.random.normal(k_eta, (self.n,))  # confounded, spatially correlated
        e = jnp.mean(u[neighbours], axis=1)  # exposure = mean neighbour treatment
        x_nb = jnp.mean(x[neighbours], axis=1)
        z_nb = jnp.mean(z[neighbours], axis=1)
        noise = self.noise_scale * jax.random.normal(k_noise, (self.n,))
        y = self.a * x + self.b_direct * u + self.b_spillover * e + self.c * z + noise
        return {
            "x": x,
            "z": z,
            "u": u,
            "e": e,
            "x_nb": x_nb,
            "z_nb": z_nb,
            "x_next": y,
            "neighbours": neighbours,  # (n, degree) graph, for the GNN-nuisance estimator
        }


@dataclass(frozen=True)
class DelayedNetworkPanel:
    """Units on a graph observed over TIME, where a spillover takes ``lag`` steps per graph hop.

    ``ConfoundedNetworkSystem`` draws one cross-section, so a propagation delay cannot even be
    written down there, let alone estimated. This is its panel counterpart: ``n_clusters`` disjoint
    cycles of ``cluster_size`` units, each observed at ``n_times`` times, with

        e_i(t) = sum_d gamma_d * sum_{j : dist(i,j) = d} u_j(t - lag*d)

    so a shell-``d`` neighbour's treatment arrives ``lag*d`` steps late. That is exactly the
    generative model Result 51 is stated for, which is the point: it makes ``lag`` and the serial
    correlation ``phi`` ESTIMABLE from a log instead of assumed.

    ``disturbance_scale`` adds an INDEPENDENT draw of the same propagated process to the outcome,
    and it is what decides whether Result 51 says anything about an estimator rather than only
    about a process. The delayed-network covariance the law is a functional of is the one the
    SCORE's noise carries; with the default i.i.d. outcome noise the score is white, the fold
    operator cannot move the estimator's variance at all, and the two fold schemes measure equal
    variance whatever the law predicts. This is the delayed-network analogue of the cluster random
    effect that carries the ICC in :func:`chc.regret.cluster_fold_leakage_certificate`. It defaults
    to zero so the effect-recovery use of the panel is unchanged.

    Construction detail that decides whether the law applies. The exogenous part of the treatment,
    ``eta``, is i.i.d. ACROSS UNITS and AR(1) in time, while the confounder ``z`` carries the
    spatial smoothness. Result 51's covariance is the one the RESIDUALISED score has, and
    residualising on a nuisance that sees ``z`` leaves the ``eta``-driven part -- so putting the
    spatial correlation in
    ``z`` and not in ``eta`` is what makes the law's premise hold after cross-fitting rather than
    before it. A spatially smooth ``eta`` would add a second correlation layer and the polynomial
    would be an approximation; that variant is reachable by raising
    ``eta_smoothing``.

    ``sample`` takes a JAX key like its cross-sectional sibling and returns the same column-dict, so
    ``estimate_network_effects`` accepts a panel with no adapter. Internally the propagation runs
    through the NumPy routine ``chc.regret.delayed_network_certificate`` is verified against; the
    seed is derived from the key. One routine, one law -- a JAX re-implementation would be a second
    copy of the exact thing the certificate checks.
    """

    n_clusters: int = 40
    cluster_size: int = 6
    n_times: int = 24
    lag: int = 1
    phi: float = 0.6
    gammas: tuple[float, ...] = (1.0, 0.7, 0.4)
    b_direct: float = 1.0
    b_spillover: float = 0.6
    c: float = 2.0
    kappa: float = -1.5
    eta_smoothing: float = 0.0  # >0 puts spatial correlation into eta too; see the class docstring
    noise_scale: float = 0.1
    disturbance_scale: float = 0.0
    burn_in: int = 40

    def sample(self, key: Array) -> dict[str, Array]:
        """Draw the panel as flat columns, ordered cluster-major, then unit, then time."""
        m, p, g = self.cluster_size, self.n_times, self.n_clusters
        dmax = len(self.gammas) - 1
        shells = cycle_shells(m, dmax)
        ring = shells[1] / shells[1].sum(axis=1, keepdims=True)  # mean over cycle neighbours
        span = self.burn_in + p + self.lag * dmax
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))

        # z: spatially smooth AND serially correlated -- the confounder the nuisance must remove.
        # Built over the FULL span so the lagged exposure sees real confounded history, not zeros.
        z_full = ar1_innovations(rng, (g, m), span, self.phi)
        z_full = 0.5 * z_full + 0.5 * np.einsum("ij,gjt->git", ring, z_full)

        # eta: i.i.d. across units by default, so the RESIDUALISED score matches Result 51 exactly.
        eta_full = ar1_innovations(rng, (g, m), span, self.phi)
        if self.eta_smoothing > 0.0:
            eta_full = (1.0 - self.eta_smoothing) * eta_full + self.eta_smoothing * np.einsum(
                "ij,gjt->git", ring, eta_full
            )

        u_full = self.kappa * z_full + eta_full
        u, z = u_full[..., -p:], z_full[..., -p:]
        e = propagate_shells(u_full, shells, self.gammas, self.lag, p)

        x = rng.standard_normal((g, m, p))
        noise = self.noise_scale * rng.standard_normal((g, m, p))
        if self.disturbance_scale > 0.0:
            shock = ar1_innovations(rng, (g, m), span, self.phi)
            noise = noise + self.disturbance_scale * propagate_shells(
                shock, shells, self.gammas, self.lag, p
            )
        y = self.b_direct * u + self.b_spillover * e + self.c * z + 0.5 * x + noise

        # flat row index of (cluster, unit, time) is g*m*p + i*p + t, so a same-time neighbour of
        # unit i is at the same offset with i replaced -- the graph is per-cluster, never across.
        ring_idx = np.stack([(np.arange(m) - 1) % m, (np.arange(m) + 1) % m], axis=1)
        base = (np.arange(g) * m * p)[:, None, None, None]
        neigh = base + ring_idx[None, :, None, :] * p + np.arange(p)[None, None, :, None]

        flat = lambda a: jnp.asarray(a.reshape(-1))  # noqa: E731 -- one shape for every column
        return {
            "x": flat(x),
            "z": flat(z),
            "u": flat(u),
            "e": flat(e),
            "x_nb": flat(np.einsum("ij,gjt->git", ring, x)),
            "z_nb": flat(np.einsum("ij,gjt->git", ring, z)),
            "x_next": flat(y),
            "cid": jnp.asarray(np.repeat(np.arange(g), m * p)),
            "unit": jnp.asarray(np.tile(np.repeat(np.arange(m), p), g)),
            "time": jnp.asarray(np.tile(np.arange(p), g * m)),
            "neighbours": jnp.asarray(neigh.reshape(g * m * p, 2)),
        }


@dataclass(frozen=True)
class PropagationEstimate:
    """The two parameters Result 51's polynomial needs, read off a panel rather than assumed.

    ``delay`` is the per-shell propagation lag ``delta`` in steps: a shell-``d`` neighbour's
    treatment reaches the unit at lag ``delta * d``. ``phi`` is the treatment's own AR(1)
    coefficient in time. Together they fix ``x = phi ** delta``, the variable the delayed-network
    variance ratio is a polynomial in.

    ``shell_lags`` is the peak lag measured at each shell separately and ``n_shells`` is the
    spillover truncation those lags imply -- the polynomial's degree, not the graph's diameter.
    ``lo`` and ``hi`` bound ``delay`` by a cluster bootstrap. ``censored`` propagates
    :func:`chc.irf.peak_lag`'s flag: a peak on the end of the horizon bounds the delay from below
    and is not an estimate of it.
    """

    delay: float
    phi: float
    shell_lags: tuple[float, ...]
    n_shells: int
    lo: float
    hi: float
    censored: bool


def _shell_gram(
    exposure: NDArray[np.float64], outcomes: NDArray[np.float64], horizon: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-cluster normal equations of a panel local projection, one cluster per leading index.

    Rows are cut inside a single unit's trajectory, so none of them straddles the join between two
    units the way a projection over the flattened panel would. Returning ``(X'X, X'Y)`` per cluster
    rather than the rows themselves makes a bootstrap resample cost ``O(clusters)``: summing the
    selected clusters' Gram matrices is exactly the stacked least-squares problem.
    """
    n = outcomes.shape[-1] - horizon
    ones = np.ones_like(exposure[..., :n])
    rows = np.stack([exposure[..., :n], outcomes[..., :n], ones], axis=-1)
    resp = np.stack([outcomes[..., h : h + n] for h in range(horizon + 1)], axis=-1)
    rows = rows.reshape(rows.shape[0], -1, rows.shape[-1])
    resp = resp.reshape(resp.shape[0], -1, resp.shape[-1])
    return np.einsum("cnj,cnk->cjk", rows, rows), np.einsum("cnj,cnh->cjh", rows, resp)


def _peak_from_gram(gram: NDArray[np.float64], cross: NDArray[np.float64]) -> tuple[float, bool]:
    """Peak lag of the exposure coefficient across horizons, from summed normal equations."""
    return peak_lag(np.linalg.solve(gram.sum(0), cross.sum(0))[0], refine=False)


def _shell_slope(lags: Sequence[float]) -> tuple[float, int]:
    """Propagation lag per unit of distance, and the spillover truncation the lags imply.

    Fit through the origin: shell 0 is the unit itself, whose lag is zero by construction, so an
    intercept would estimate a quantity the model fixes. Only the longest strictly increasing
    prefix is used -- a shell past the spillover truncation carries no direct edge, so its apparent
    peak is inherited from the last shell that does and stops advancing. Including such a flat
    point drags the slope toward zero; on the shipped panel it halved it.
    """
    kept = 1
    while kept < len(lags) and lags[kept] > lags[kept - 1]:
        kept += 1
    d = np.arange(1, kept + 1, dtype=np.float64)
    return float(d @ np.asarray(lags[:kept]) / (d @ d)), kept


def within_ar1(series: NDArray[np.float64], corrections: int = 3) -> float:
    """AR(1) coefficient of a panel series, corrected for the bias the within transform induces.

    Sweeping out unit and time means to isolate the serial dependence also correlates the
    demeaned regressor with the demeaned innovation, biasing the lag-1 autocorrelation down by
    ``(1 + phi) / (n_times - 1)`` (Nickell 1981). The raw number is not reported: it is a
    prediction of a quantity smaller than the one Result 51 needs. Inverting that relation by
    fixed-point iteration recovered ``phi`` to within 0.007 for ``phi <= 0.6`` over
    ``n_times`` in 20-160, and to 0.022 in the worst corner measured (``phi = 0.85``,
    ``n_times = 20``), where the bias being removed is itself 0.097.
    """
    resid = (
        series
        - series.mean(-1, keepdims=True)
        - series.mean(-2, keepdims=True)
        + series.mean((-2, -1), keepdims=True)
    )
    lead, lag = resid[..., 1:].reshape(-1), resid[..., :-1].reshape(-1)
    raw = float(lag @ lead / (lag @ lag))
    estimate = raw
    for _ in range(corrections):
        estimate = raw + (1.0 + estimate) / (series.shape[-1] - 1)
    return estimate


def estimate_propagation(
    treatments: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    shells: Sequence[NDArray[np.float64]],
    horizon: int,
    n_resamples: int = 200,
    level: float = 0.95,
    seed: int = 0,
) -> PropagationEstimate:
    """Estimate the propagation lag and persistence of a delayed network exposure from a panel.

    ``treatments`` and ``outcomes`` are ``(clusters, units, times)``. ``shells[d]`` is the
    adjacency of the units exactly ``d`` apart, as :func:`cycle_shells` returns it; shell 0 is
    ignored, since a unit's own treatment carries no propagation lag.

    Each shell gets its own panel local projection of the outcome on that shell's mean treatment,
    controlling for the unit's own contemporaneous outcome, and the peak of the resulting response
    locates ``delta * d``. Regressing those peaks on ``d`` through the origin gives ``delta``.
    Doing it shell by shell rather than pooling is what makes the estimate falsifiable: if the
    peaks are not proportional to distance, the propagation model is wrong and the flat tail shows
    it, which is also how the spillover truncation is read off.

    :func:`chc.irf.delay_estimate` is the single-trajectory version of this and is the right tool
    for one series. It is not used here: its projection rows are cut by contiguous slicing, so on a
    flattened panel a fraction ``horizon / n_times`` of them span two different units. Measured on
    :class:`DelayedNetworkPanel` the flattened estimate happened to agree down to ``n_times = 8``,
    where three rows in four straddle a join -- the peak is a location statistic and the
    contaminated rows attenuate every horizon at once -- but agreement that depends on the
    contamination staying symmetric is not a guarantee, and cutting the rows inside a trajectory
    costs nothing.

    The interval resamples whole clusters with replacement. Clusters are the independent unit here,
    so this preserves both the serial dependence within a unit and the spatial dependence within a
    cluster exactly, which a moving block over time would break. It needs enough clusters to be
    meaningful: below roughly 20 the percentile interval is reporting the resampling grid.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1); got {level}")
    if len(shells) < 2:
        raise ValueError(f"need at least one non-zero shell; got {len(shells)}")
    if outcomes.shape[-1] - horizon < 2:
        raise ValueError(
            f"{outcomes.shape[-1]} times cannot carry a horizon of {horizon}: "
            f"a projection needs at least two rows per unit"
        )

    grams, crosses = [], []
    for shell in shells[1:]:
        weights = shell / np.maximum(shell.sum(axis=1, keepdims=True), 1.0)
        gram, cross = _shell_gram(np.einsum("ij,cjt->cit", weights, treatments), outcomes, horizon)
        grams.append(gram)
        crosses.append(cross)

    peaks, censored = zip(
        *(_peak_from_gram(g, c) for g, c in zip(grams, crosses, strict=True)), strict=True
    )
    delay, kept = _shell_slope(peaks)

    rng = np.random.default_rng(seed)
    n_clusters = outcomes.shape[0]
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        pick = rng.integers(0, n_clusters, n_clusters)
        resampled = [
            _peak_from_gram(g[pick], c[pick])[0] for g, c in zip(grams, crosses, strict=True)
        ]
        draws[i] = _shell_slope(resampled)[0]
    tail = 0.5 * (1.0 - level)
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])

    return PropagationEstimate(
        delay=delay,
        phi=within_ar1(treatments),
        shell_lags=tuple(float(p) for p in peaks),
        n_shells=kept,
        lo=float(lo),
        hi=float(hi),
        censored=any(censored[:kept]),
    )


def _fold_chunks(n: int, folds: int, seed: int, groups: Array | None) -> list[Array]:
    """Row indices per cross-fitting fold, optionally keeping labelled groups intact.

    ``groups is None`` is the historical path and stays byte-identical: permute rows, chunk them.
    Otherwise the permutation acts on the distinct labels, so every row carrying a label lands in
    the same fold as the rest of that label.
    """
    if groups is None:
        return jnp.array_split(jax.random.permutation(jax.random.key(seed), n), folds)
    labels = jnp.asarray(groups).reshape(-1)
    if labels.shape[0] != n:
        raise ValueError(f"fold_groups has {labels.shape[0]} entries for {n} rows")
    distinct = jnp.unique(labels)
    if distinct.shape[0] < folds:
        raise ValueError(f"{distinct.shape[0]} distinct groups cannot fill {folds} folds")
    order = jax.random.permutation(jax.random.key(seed), distinct.shape[0])
    assign = jnp.zeros(distinct.shape[0], dtype=jnp.int32)
    for k, part in enumerate(jnp.array_split(order, folds)):
        assign = assign.at[part].set(k)
    row_fold = assign[jnp.searchsorted(distinct, labels)]
    return [jnp.where(row_fold == k)[0] for k in range(folds)]


def estimate_network_effects(
    data: dict[str, Array],
    covariates: tuple[str, ...] = ("x", "z", "x_nb", "z_nb"),
    exposure: str = "e",
    degree: int = 2,
    folds: int = 5,
    ridge: float = 1.0,
    seed: int = 0,
    fold_groups: Array | None = None,
) -> dict[str, float]:
    """Cross-fitted DML for the direct and spillover effects (partials out graph-aware nuisances).

    Residualises the outcome, treatment ``u``, and exposure ``e`` on flexible predictions from the
    (own + mean-neighbour) covariates, then regresses the outcome residual on the two treatment
    residuals -- the coefficients are the direct and spillover effects.

    ``fold_groups`` labels each row with a unit that must not be split across folds; whole labels
    are permuted and chunked instead of rows. The default of ``None`` permutes rows, which is
    graph-blind: a unit's neighbour then usually lands in the training fold that predicts it, so
    the same-fold edge fraction sits near ``1 / folds``. Result 51 shows the delayed cross-fit
    variance ratio is affine in that fraction at spillover truncation ``D = 1``, with a root at
    ``theta* = K^3 / (4K^3 - 6K^2 + 4K - 1)``, which exceeds ``1 / K`` for every ``K`` -- so random
    rows always sit on the side where a propagation delay drives the ratio down. Passing the unit
    index recovers the graph structure the permutation destroys; passing a graph-aware grouping
    (contiguous blocks of a cycle, a partition of a real graph) moves ``theta`` toward the root.
    Folds become unbalanced when the labels are, which is the price of keeping them intact.
    """
    y, u, e = data["x_next"], data["u"], data[exposure]
    covs = jnp.stack([data[c] for c in covariates], axis=1)
    n = y.shape[0]
    chunks = _fold_chunks(n, folds, seed, fold_groups)
    y_res, u_res, e_res = (jnp.zeros(n), jnp.zeros(n), jnp.zeros(n))
    for k in range(folds):
        test = chunks[k]
        train = jnp.concatenate([chunks[j] for j in range(folds) if j != k])
        phi_tr = _polynomial_features(covs[train], degree)
        phi_te = _polynomial_features(covs[test], degree)
        y_res = y_res.at[test].set(y[test] - _ridge_predict(phi_tr, y[train], phi_te, ridge))
        u_res = u_res.at[test].set(u[test] - _ridge_predict(phi_tr, u[train], phi_te, ridge))
        e_res = e_res.at[test].set(e[test] - _ridge_predict(phi_tr, e[train], phi_te, ridge))
    coef, *_ = jnp.linalg.lstsq(jnp.stack([u_res, e_res], axis=1), y_res, rcond=None)
    return {"direct": float(coef[0]), "spillover": float(coef[1])}


class NeighbourMessagePassing(eqx.Module):
    """A mean-aggregation message-passing regressor that learns what to aggregate over neighbours.

    Each layer is ``h <- tanh(W_self h + W_neigh mean_{j in N(i)} h_j)``, so it can represent
    ``mean_j f(h_j)`` for a learned ``f`` -- richer than a fixed mean-neighbour covariate.
    """

    layers: tuple[tuple[eqx.nn.Linear, eqx.nn.Linear], ...]
    readout: eqx.nn.Linear

    def __init__(self, in_dim: int, hidden: int, *, key: Array):
        keys = jax.random.split(key, 5)
        self.layers = (
            (
                eqx.nn.Linear(in_dim, hidden, key=keys[0]),
                eqx.nn.Linear(in_dim, hidden, key=keys[1]),
            ),
            (
                eqx.nn.Linear(hidden, hidden, key=keys[2]),
                eqx.nn.Linear(hidden, hidden, key=keys[3]),
            ),
        )
        self.readout = eqx.nn.Linear(hidden, 1, key=keys[4])

    def __call__(self, feats: Array, neighbours: Array) -> Array:
        h = feats
        for self_lin, neigh_lin in self.layers:
            aggregate = jnp.mean(h[neighbours], axis=1)
            h = jax.nn.tanh(jax.vmap(self_lin)(h) + jax.vmap(neigh_lin)(aggregate))
        return jax.vmap(self.readout)(h)[:, 0]


def _fit_gnn_nuisance(
    feats: Array,
    neighbours: Array,
    target: Array,
    mask: Array,
    *,
    hidden: int,
    steps: int,
    lr: float,
    key: Array,
) -> Array:
    """Fit a message-passing regressor to ``target`` on the masked (train) nodes; predict all."""
    model = NeighbourMessagePassing(feats.shape[1], hidden, key=key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model: NeighbourMessagePassing, opt_state: optax.OptState):
        def loss(m: NeighbourMessagePassing) -> Array:
            pred = m(feats, neighbours)
            return jnp.sum(mask * (pred - target) ** 2) / jnp.sum(mask)  # message-pass full graph

        grads = eqx.filter_grad(loss)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state

    for _ in range(steps):
        model, opt_state = step(model, opt_state)
    return model(feats, neighbours)


def estimate_network_effects_gnn(
    data: dict[str, Array],
    neighbours: Array,
    *,
    features: tuple[str, ...] = ("x", "z"),
    treatment: str = "u",
    exposure: str = "e",
    folds: int = 2,
    hidden: int = 16,
    steps: int = 400,
    lr: float = 3e-3,
    seed: int = 0,
    fold_groups: Array | None = None,
) -> dict[str, float]:
    """Cross-fitted network DML with learned message-passing GNN nuisances, not fixed features.

    The GNN counterpart of :func:`estimate_network_effects`: it residualises the outcome, treatment,
    and exposure on a GNN of the raw node ``features`` plus the graph, so neighbour aggregation is
    learned, not hand-specified (no ``x_nb``/``z_nb`` columns needed). Same orthogonal score and
    estimand (direct + spillover). Accuracy matches the lean version on smooth confounding -- DML
    orthogonality makes the effect robust to nuisance form -- so it earns its cost when the graph
    representation is better learned than crafted, not as a free accuracy win.

    ``fold_groups`` behaves as in :func:`estimate_network_effects`, but its payoff here is
    CHANNEL-SPECIFIC and smaller than Result 51's scalar law predicts. Measured on
    :class:`DelayedNetworkPanel` at ``m=6, G=15, p=12, delta=1, phi=0.6``, 400 paired replications
    (both designs share each draw, so the ratio is bootstrapped over seeds; arm correlation 0.88 and
    0.97): parity-vs-block unit folds move the DIRECT effect's variance to ``0.887`` [0.803, 0.976]
    and the SPILLOVER effect's to ``0.999`` [0.950, 1.052]. The direct channel therefore gains a
    real ~11% variance reduction, the spillover channel is inert, and the law's ``0.715`` is outside
    BOTH intervals -- a scalar ``Psi`` cannot produce a channel-dependent answer at all, so the
    two-column sandwich is what governs here. An independent-arm standard error is not valid for
    this comparison; assuming one is what left an earlier 150-replication run unable to decide.
    """
    y, u, e = data["x_next"], data[treatment], data[exposure]
    feats = jnp.stack([data[f] for f in features], axis=1)
    n = int(y.shape[0])
    chunks = _fold_chunks(n, folds, seed, fold_groups)
    keys = jax.random.split(jax.random.key(seed + 1), 3 * folds)
    y_res, u_res, e_res = jnp.zeros(n), jnp.zeros(n), jnp.zeros(n)
    for k in range(folds):
        test = chunks[k]
        train = jnp.concatenate([chunks[j] for j in range(folds) if j != k])
        mask = jnp.zeros(n).at[train].set(1.0)
        cfg = {"hidden": hidden, "steps": steps, "lr": lr}
        pred_y = _fit_gnn_nuisance(feats, neighbours, y, mask, key=keys[3 * k], **cfg)
        pred_u = _fit_gnn_nuisance(feats, neighbours, u, mask, key=keys[3 * k + 1], **cfg)
        pred_e = _fit_gnn_nuisance(feats, neighbours, e, mask, key=keys[3 * k + 2], **cfg)
        y_res = y_res.at[test].set(y[test] - pred_y[test])
        u_res = u_res.at[test].set(u[test] - pred_u[test])
        e_res = e_res.at[test].set(e[test] - pred_e[test])
    coef, *_ = jnp.linalg.lstsq(jnp.stack([u_res, e_res], axis=1), y_res, rcond=None)
    return {"direct": float(coef[0]), "spillover": float(coef[1])}
