"""Confounded offline identification: why the control-effect residual needs the adjustment set.

A minimal linear demonstration of the CHC causal claim (see ``plans/02``). The historical action
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


def _ols_with_intercept(features: Array, target: Array) -> Array:
    """Ordinary least squares with an intercept column appended; returns the coefficient vector."""
    design = jnp.concatenate([features, jnp.ones((features.shape[0], 1))], axis=1)
    coeffs, *_ = jnp.linalg.lstsq(design, target, rcond=None)
    return coeffs


def estimate_control_effect(data: dict[str, Array], adjust_for: tuple[str, ...] = ()) -> Array:
    """Estimate ``∂x_next/∂u`` from logged data, adjusting for the named covariates.

    Regresses ``x_next`` on ``[x, u, *adjust_for]``. With ``adjust_for=("z",)`` (the correct
    adjustment set) the ``u`` coefficient is causal; with ``adjust_for=()`` it stays confounded.
    """
    columns = [data["x"], data["u"], *[data[name] for name in adjust_for]]
    features = jnp.stack(columns, axis=1)
    coeffs = _ols_with_intercept(features, data["x_next"])
    return coeffs[1]  # coefficient on u


def _ols_fit(features: Array, target: Array) -> tuple[Array, Array]:
    """OLS with intercept; returns (coefficients, fitted values)."""
    design = jnp.concatenate([features, jnp.ones((features.shape[0], 1))], axis=1)
    coeffs, *_ = jnp.linalg.lstsq(design, target, rcond=None)
    return coeffs, design @ coeffs


def estimate_effect_iv(data: dict[str, Array], instrument: str = "w") -> Array:
    """Two-stage least squares for ``∂x_next/∂u`` using an instrument for a *latent* confounder.

    Stage 1 regresses ``u`` on ``[x, instrument]``; stage 2 regresses ``x_next`` on ``[x, û]``. The
    instrument must drive ``u``, be independent of the confounder, and affect ``x_next`` only via
    ``u`` — then the effect is recovered even when the confounder ``z`` is unobserved.
    """
    x, u, w, y = data["x"], data["u"], data[instrument], data["x_next"]
    _, u_hat = _ols_fit(jnp.stack([x, w], axis=1), u)
    coeffs, _ = _ols_fit(jnp.stack([x, u_hat], axis=1), y)
    return coeffs[1]  # coefficient on the fitted (exogenous) part of u


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
    """
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
    """
    y, u = data["x_next"], data["u"]
    covs = jnp.stack([data[c] for c in covariates], axis=1)
    n = y.shape[0]
    chunks = jnp.array_split(jax.random.permutation(jax.random.key(seed), n), folds)

    y_res = jnp.zeros(n)
    u_res = jnp.zeros(n)
    for k in range(folds):
        test = chunks[k]
        train = jnp.concatenate([chunks[j] for j in range(folds) if j != k])
        phi_train = _polynomial_features(covs[train], degree)
        phi_test = _polynomial_features(covs[test], degree)
        y_res = y_res.at[test].set(y[test] - _ridge_predict(phi_train, y[train], phi_test, ridge))
        u_res = u_res.at[test].set(u[test] - _ridge_predict(phi_train, u[train], phi_test, ridge))

    return jnp.sum(y_res * u_res) / jnp.sum(
        u_res * u_res
    )  # residual-on-residual through the origin


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
    k_perm, k_rcc, k_sub = jax.random.split(jax.random.key(seed), 3)
    n = data["x"].shape[0]
    original = float(estimate_control_effect(data, adjust_for))

    placebo_data = {**data, "u": data["u"][jax.random.permutation(k_perm, n)]}
    placebo = float(estimate_control_effect(placebo_data, adjust_for))

    rcc_data = {**data, "_rcc": jax.random.normal(k_rcc, (n,))}
    rcc = float(estimate_control_effect(rcc_data, (*adjust_for, "_rcc")))

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
