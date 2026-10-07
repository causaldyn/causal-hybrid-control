"""Confounded offline identification: why the control-effect residual needs the adjustment set.

A minimal linear demonstration of the CHC causal claim. The historical action
was chosen by a behaviour policy correlated with a covariate ``z`` that also drives the outcome.
Fitting the effect of ``u`` without adjusting for ``z`` is confounded (the estimate can flip sign);
conditioning the residual on the adjustment set recovers the true interventional effect.

Sequential-ignorability setting with history ``H = (x, z)``: with ``z`` in the adjustment set the
backdoor path ``u <- z -> x'`` is blocked and the effect is identified; omit ``z`` and it is not.
When ``z`` is *latent*, an instrument ``w`` identifies the effect via 2SLS; a Cinelli-Hazlett
robustness value bounds how much hidden confounding a control decision could tolerate. Double ML
recovers the effect under *nonlinear* confounding via cross-fitted residualisation.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations_with_replacement

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class ConfoundedLinearSystem:
    """One-step transition ``x' = a·x + b_true·u + c·z + noise`` logged under ``u = kappa·z + eta``.

    ``z`` is an observed confounder: it drives both the historical action (via ``kappa``) and the
    outcome (via ``c``). ``b_true`` is the causal effect we ultimately want for control.
    The defaults are tuned so the naive (unadjusted) estimate of ``b`` flips sign.
    """

    a: float = 0.5
    b_true: float = 1.0
    c: float = 2.0
    kappa: float = -1.5
    gamma: float = 0.0  # instrument strength; 0.0 = no instrument
    z_scale: float = 1.0
    eta_scale: float = 0.5
    noise_scale: float = 0.1

    def sample(self, n: int, key: Array) -> dict[str, Array]:
        """Draw ``n`` transitions as columns ``x, z, u, x_next, w`` (``w`` = instrument)."""
        k_x, k_z, k_eta, k_noise = jax.random.split(key, 4)
        x = jax.random.normal(k_x, (n,))
        z = self.z_scale * jax.random.normal(k_z, (n,))
        eta = self.eta_scale * jax.random.normal(k_eta, (n,))
        w = jax.random.normal(jax.random.fold_in(key, 7), (n,))  # instrument: drives u, not x_next
        u = self.kappa * z + self.gamma * w + eta  # behaviour policy tied to the confounder
        noise = self.noise_scale * jax.random.normal(k_noise, (n,))
        x_next = self.a * x + self.b_true * u + self.c * z + noise
        return {"x": x, "z": z, "u": u, "x_next": x_next, "w": w}


_STATE = {"x": "the state"}
_ACTION_AND_OUTCOME = {"u": "the action", "x_next": "the next state, the outcome"}
_TRANSITION = {**_STATE, **_ACTION_AND_OUTCOME}


def _refuse_reread(role: str, names: Iterable[str], taken: Mapping[str, str]) -> None:
    """Refuse a ``role`` column named for one the estimate reads in its own role already.

    A transition's state, action and next state are ``x``, ``u`` and ``x_next``, and every other
    column is read by the caller's name, all from one dict: a covariate named ``u`` adjusts the
    action for itself, which halves its coefficient in a regression and leaves the moment nothing
    to move in a partialling-out.
    """
    for name in names:
        if name in taken:
            raise ValueError(
                f"the {role} {name!r} is {taken[name]}, which the estimate already reads in that "
                "role; a column it stands for needs a name of its own"
            )


def _ols_with_intercept(features: Array, target: Array) -> Array:
    """Ordinary least squares with an intercept column appended; returns the coefficient vector."""
    design = jnp.concatenate([features, jnp.ones((features.shape[0], 1))], axis=1)
    coeffs, *_ = jnp.linalg.lstsq(design, target, rcond=None)
    return coeffs


def estimate_control_effect(data: dict[str, Array], adjust_for: tuple[str, ...] = ()) -> Array:
    """Estimate ``∂x_next/∂u`` from logged data, adjusting for the named covariates.

    Regresses ``x_next`` on ``[x, u, *adjust_for]``. With ``adjust_for=("z",)`` (the correct
    adjustment set) the ``u`` coefficient is causal; with ``adjust_for=()`` it stays confounded.

    Raises:
        ValueError: if ``adjust_for`` names ``u`` or ``x_next``, the action and the outcome.
    """
    _refuse_reread("covariate", adjust_for, _ACTION_AND_OUTCOME)
    columns = [data["x"], data["u"], *[data[name] for name in adjust_for]]
    features = jnp.stack(columns, axis=1)
    coeffs = _ols_with_intercept(features, data["x_next"])
    return coeffs[1]  # coefficient on u


def _ols_fit(features: Array, target: Array) -> tuple[Array, Array]:
    """OLS with intercept; returns (coefficients, fitted values)."""
    design = jnp.concatenate([features, jnp.ones((features.shape[0], 1))], axis=1)
    coeffs, *_ = jnp.linalg.lstsq(design, target, rcond=None)
    return coeffs, design @ coeffs


_ROUNDING = 64
"""A spread within this many eps of its column's root mean square is rounding, so no spread."""


def _instrument_relevance(state: Array, action: Array, instrument: Array) -> Array:
    """How far ``instrument`` moves ``action`` beyond what ``state`` and a constant explain: the
    canonical correlation between the first stage's push on the action and the action, each less
    its least-squares projection on a constant and the state. For one action and one instrument it
    is their partial correlation given the state, in absolute value: the root of the first stage's
    partial ``R^2``.

    It reads 0 where 2SLS's moment has no rank: where the product of the instrument and the action,
    each beyond the state, is no more than their rounding could make of it. Each column is known
    to ``_ROUNDING`` eps of its root mean square, the eps that of the coarsest precision the three
    columns come in, so the action's rounding moves the product by that much of the action's root
    mean square times the instrument's spread beyond the state, and the instrument's rounding by
    that much of its own times the action's spread. A column with no spread beyond the state, or
    none beyond its rounding, reads 0 so. Each column is centred before its projection, and the
    state scaled to unit spread, so the reading is the same in any units of the three and at any
    level of them.
    """
    rounding = _ROUNDING * max(
        float(jnp.finfo(jnp.result_type(column, float)).eps)
        for column in (state, action, instrument)
    )
    centred = state - jnp.mean(state)
    spread = jnp.sqrt(jnp.mean(centred**2))
    flat = spread <= rounding * jnp.sqrt(jnp.mean(state**2))
    scaled = jnp.where(flat, 0.0, centred / jnp.where(flat, 1.0, spread))
    basis = jnp.stack([jnp.ones_like(scaled), scaled], axis=1)

    def beyond_state(column: Array) -> Array:
        column = column - jnp.mean(column)
        return column - basis @ jnp.linalg.lstsq(basis, column)[0]

    shifted, moved = beyond_state(instrument), beyond_state(action)
    size, swing = jnp.linalg.norm(shifted), jnp.linalg.norm(moved)
    overlap = jnp.abs(shifted @ moved)
    if overlap <= rounding * (jnp.linalg.norm(action) * size + jnp.linalg.norm(instrument) * swing):
        return jnp.zeros((), overlap.dtype)
    return overlap / size / swing


def _two_stage(data: dict[str, Array], instrument: str) -> tuple[Array, Array]:
    """:func:`estimate_effect_iv`'s estimate and its instrument's relevance
    (:func:`_instrument_relevance`), refused where the relevance is 0."""
    _refuse_reread("instrument", (instrument,), _TRANSITION)
    x, u, w, y = data["x"], data["u"], data[instrument], data["x_next"]
    relevance = _instrument_relevance(x, u, w)
    if relevance == 0.0:
        raise ValueError(
            f"the instrument {instrument!r} does not move the action beyond what the state "
            "explains: 2SLS's moment has no rank, so the effect is not identified"
        )
    _, u_hat = _ols_fit(jnp.stack([x, w], axis=1), u)
    coeffs, _ = _ols_fit(jnp.stack([x, u_hat], axis=1), y)
    return coeffs[1], relevance  # coefficient on the fitted (exogenous) part of u


def estimate_effect_iv(data: dict[str, Array], instrument: str = "w") -> Array:
    """Two-stage least squares for ``∂x_next/∂u`` using an instrument for a *latent* confounder.

    Stage 1 regresses ``u`` on ``[x, instrument]``; stage 2 regresses ``x_next`` on ``[x, û]``. The
    instrument must drive ``u``, be independent of the confounder, and affect ``x_next`` only via
    ``u`` — then the effect is recovered even when the confounder ``z`` is unobserved.

    The log can check the first of the three, and the estimate does: the instrument must move
    ``u`` beyond what ``x`` and a constant explain. The product of the two beyond them must be more
    than 64 eps of each one's own size could make of it, read the same in any units and at any
    level. Short of that, 2SLS's moment has no rank, ``û`` is collinear with ``x`` and the
    constant, and least squares returned its minimum-norm coefficient: on 40,000 rows of
    :class:`ConfoundedLinearSystem` with ``gamma=1``, where the truth is 1.0, -0.0032 for an
    instrument of zeros and 0.63 for an affine copy of ``x`` where ``u`` moves with ``x``. A weak
    instrument keeps the rank and is estimated: a column of noise drawn apart from ``u`` read 0.263
    there, its partial correlation with ``u`` given ``x`` 0.0087 where the log's own instrument's
    is 0.53 (:class:`chc.estimators.IV2SLS` reports it). No inference here is robust to a weak
    instrument.

    Raises:
        ValueError: if ``instrument`` is ``x``, ``u`` or ``x_next``: the state reaches ``x_next``
            on its own path, and the action and the outcome are what it stands between; or if it
            does not move ``u`` beyond what ``x`` explains, which leaves the effect not identified.
    """
    return _two_stage(data, instrument)[0]


def _ols_with_se(features: Array, target: Array) -> tuple[Array, Array, int]:
    """OLS with intercept; returns (coefficients, standard errors, residual degrees of freedom)."""
    design = jnp.concatenate([features, jnp.ones((features.shape[0], 1))], axis=1)
    n, p = design.shape
    beta, *_ = jnp.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    dof = n - p
    sigma2 = (residual @ residual) / dof
    cov = sigma2 * jnp.linalg.inv(design.T @ design)
    return beta, jnp.sqrt(jnp.diag(cov)), dof


def sensitivity_analysis(
    data: dict[str, Array], adjust_for: tuple[str, ...] = (), q: float = 1.0
) -> dict[str, float]:
    """Effect estimate plus its Cinelli-Hazlett robustness value.

    The robustness value is the ``R^2`` an unobserved confounder would need with *both* ``u`` and
    ``x_next`` to reduce the estimated effect by ``q*100%`` (toward zero). High RV = robust;
    a controller can ship this bound on how much hidden confounding its decision could tolerate.

    Raises:
        ValueError: if ``adjust_for`` names ``x``, ``u`` or ``x_next``, or a column twice. The
            standard error inverts the design's Gram matrix, which a column read twice makes
            singular: the error, the robustness value and the interval's E-value came out nan,
            nan and 1.
    """
    _refuse_reread("covariate", adjust_for, _TRANSITION)
    twice = sorted({name for name in adjust_for if adjust_for.count(name) > 1})
    if twice:
        raise ValueError(f"covariates named more than once: {twice}")
    columns = [data["x"], data["u"], *[data[name] for name in adjust_for]]
    beta, se, dof = _ols_with_se(jnp.stack(columns, axis=1), data["x_next"])
    t_stat = jnp.abs(beta[1] / se[1])
    f = q * t_stat / jnp.sqrt(float(dof))
    rv = 0.5 * (jnp.sqrt(f**4 + 4.0 * f**2) - f**2)
    scale = float(jnp.std(data["u"]) / (jnp.std(data["x_next"]) + 1e-12))  # to standardised units
    report = {
        "effect": float(beta[1]),
        "std_error": float(se[1]),
        "robustness_value": float(rv),
    }
    report.update(e_value(float(beta[1]) * scale, float(se[1]) * scale))
    return report


_Z95 = 1.959964  # standard-normal 97.5th percentile: the 95% two-sided confidence multiplier


def e_value(standardized_effect: float, std_error: float | None = None) -> dict[str, float]:
    """VanderWeele-Ding E-value: the confounding strength needed to explain the effect away.

    The E-value is the minimum association (on the risk-ratio scale) an unmeasured confounder would
    need with *both* treatment and outcome, beyond the measured covariates, to reduce the estimate
    to the null. Larger = more robust. For a standardised (Cohen's d-scale) effect it maps
    ``d -> RR = exp(0.91 * |d|)`` (VanderWeele & Ding, 2017) then ``E = RR + sqrt(RR (RR - 1))``.
    ``std_error`` (same standardised scale) adds ``e_value_ci`` for the 95% confidence limit nearest
    the null -- ``1.0`` when the interval covers the null, i.e. no confounding need be invoked.
    """

    def _e(effect: float) -> float:
        rr = math.exp(0.91 * abs(effect))
        rr = rr if rr >= 1.0 else 1.0 / rr
        return rr + math.sqrt(rr * (rr - 1.0))

    report = {"e_value": _e(standardized_effect)}
    if std_error is not None:
        limit = abs(standardized_effect) - _Z95 * std_error
        report["e_value_ci"] = _e(limit) if limit > 0.0 else 1.0
    return report


def _stream_key(seed: int, stream: str) -> Array:
    """The key of the library's own random stream ``stream``, kept apart from a caller's keys.

    Under JAX's partitionable threefry, the default since jax 0.5.0, ``split(key(s), n)[i]`` is
    ``fold_in(key(s), i)`` for every ``n``. A stream drawn from ``key(seed)`` itself shares its
    children with data a caller drew from ``key(seed)``: cross-fitting folds drawn so put the rows
    in the order of the column drawn from the second child, in float32 up to 1625 rows, and the
    random common cause of :func:`refute_effect` was that column. The tag folded in is at least
    ``2**31``, past any child a caller can split off.
    """
    return jax.random.fold_in(jax.random.key(seed), zlib.crc32(stream.encode()) | 1 << 31)


def _polynomial_features(x: Array, degree: int) -> Array:
    """Monomials of ``x`` (n, d) up to total ``degree`` (with cross terms), plus a bias column."""
    n, d = x.shape
    features = [jnp.ones(n)]
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            term = jnp.ones(n)
            for idx in combo:
                term = term * x[:, idx]
            features.append(term)
    return jnp.stack(features, axis=1)


def _ridge_predict(x_train: Array, y_train: Array, x_test: Array, alpha: float) -> Array:
    p = x_train.shape[1]
    beta = jnp.linalg.solve(x_train.T @ x_train + alpha * jnp.eye(p), x_train.T @ y_train)
    return x_test @ beta


def _dml_residuals(
    data: dict[str, Array],
    covariates: tuple[str, ...],
    degree: int,
    folds: int,
    ridge: float,
    seed: int,
) -> tuple[Array, Array]:
    """Cross-fitted partialling-out residuals ``(y_res, u_res)`` -- the shared core of the DML point
    estimate and its influence-function SE. Nuisances are polynomial-ridge, fit out of fold.

    The state ``x`` is a covariate like any other here; ``u`` and ``x_next`` are refused, since
    a nuisance that reads the action or the outcome predicts it.
    """
    _refuse_reread("covariate", covariates, _ACTION_AND_OUTCOME)
    y, u = data["x_next"], data["u"]
    covs = jnp.stack([data[c] for c in covariates], axis=1)
    n = y.shape[0]
    chunks = jnp.array_split(jax.random.permutation(_stream_key(seed, "folds"), n), folds)

    y_res = jnp.zeros(n)
    u_res = jnp.zeros(n)
    for k in range(folds):
        test = chunks[k]
        train = jnp.concatenate([chunks[j] for j in range(folds) if j != k])
        phi_train = _polynomial_features(covs[train], degree)
        phi_test = _polynomial_features(covs[test], degree)
        y_res = y_res.at[test].set(y[test] - _ridge_predict(phi_train, y[train], phi_test, ridge))
        u_res = u_res.at[test].set(u[test] - _ridge_predict(phi_train, u[train], phi_test, ridge))
    return y_res, u_res


def estimate_effect_dml(
    data: dict[str, Array],
    covariates: tuple[str, ...] = ("x", "z"),
    degree: int = 3,
    folds: int = 2,
    ridge: float = 1e-2,
    seed: int = 0,
) -> Array:
    """Double / debiased ML estimate of ``∂x_next/∂u`` via cross-fitted residual-on-residual.

    Partials flexible (polynomial-ridge) predictions of ``x_next`` and ``u`` out of the covariates,
    then regresses the residuals. This is Neyman-orthogonal, so it recovers the effect even under
    *nonlinear* confounding, where the linear :func:`estimate_control_effect` adjustment is biased.

    Raises:
        ValueError: if ``covariates`` names ``u`` or ``x_next``, the action and the outcome.
    """
    y_res, u_res = _dml_residuals(data, covariates, degree, folds, ridge, seed)
    return jnp.sum(y_res * u_res) / jnp.sum(u_res * u_res)  # residual-on-residual through origin


def dml_point_and_se(
    data: dict[str, Array],
    covariates: tuple[str, ...] = ("x", "z"),
    degree: int = 3,
    folds: int = 2,
    ridge: float = 1e-2,
    seed: int = 0,
) -> tuple[float, float]:
    """DML point estimate ``theta`` and its heteroskedastic influence-function standard error.

    For the partially-linear score ``psi = u_res*(y_res - theta*u_res)`` the estimator is
    ``theta = <u_res, y_res>/<u_res, u_res>`` and the (robust, HC-style) SE is
    ``sqrt(sum u_res^2 * eps^2) / sum u_res^2`` with ``eps = y_res - theta*u_res`` -- the sandwich
    variance of the Neyman-orthogonal moment. Ships EconML-grade uncertainty with the cross-fit
    effect (a 95% CI is ``theta +/- 1.96*se``).

    Raises:
        ValueError: if ``covariates`` names ``u`` or ``x_next``, the action and the outcome.
    """
    y_res, u_res = _dml_residuals(data, covariates, degree, folds, ridge, seed)
    denom = jnp.sum(u_res * u_res)
    theta = jnp.sum(y_res * u_res) / denom
    eps = y_res - theta * u_res
    se = jnp.sqrt(jnp.sum(u_res**2 * eps**2)) / denom
    return float(theta), float(se)


def refute_effect(
    data: dict[str, Array],
    adjust_for: tuple[str, ...] = ("z",),
    subset_fraction: float = 0.5,
    seed: int = 0,
) -> dict[str, float | bool]:
    """DoWhy-style refutation tests for the adjusted effect estimate (a robustness gate).

    - **placebo**: permute the treatment — the effect must collapse toward 0 (else it is spurious);
    - **random common cause**: add an irrelevant covariate — the effect must stay stable;
    - **subset**: re-estimate on a random subsample — the effect must stay stable.

    Returns the estimates and ``passes`` (placebo near 0, the others near the original).
    """
    k_perm, k_rcc, k_sub = jax.random.split(_stream_key(seed, "refute_effect"), 3)
    n = data["x"].shape[0]
    original = float(estimate_control_effect(data, adjust_for))

    placebo_data = {**data, "u": data["u"][jax.random.permutation(k_perm, n)]}
    placebo = float(estimate_control_effect(placebo_data, adjust_for))

    common = "_rcc"
    while common in data:  # a name of the caller's would be replaced by the random cause
        common = f"_{common}"
    rcc_data = {**data, common: jax.random.normal(k_rcc, (n,))}
    rcc = float(estimate_control_effect(rcc_data, (*adjust_for, common)))

    idx = jax.random.permutation(k_sub, n)[: int(subset_fraction * n)]
    subset = float(
        estimate_control_effect({key: val[idx] for key, val in data.items()}, adjust_for)
    )

    scale = abs(original) + 1e-9
    passes = (
        abs(placebo) < 0.1 * scale
        and abs(rcc - original) < 0.1 * scale
        and abs(subset - original) < 0.15 * scale
    )
    return {
        "original": original,
        "placebo": placebo,
        "random_common_cause": rcc,
        "subset": subset,
        "passes": passes,
    }
