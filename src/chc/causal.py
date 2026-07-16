"""Confounded offline identification: why the control-effect residual needs the adjustment set.

A minimal linear demonstration of the CHC causal claim (see ``plans/02``). The historical action
was chosen by a behaviour policy correlated with a covariate ``z`` that also drives the outcome.
Fitting the effect of ``u`` without adjusting for ``z`` is confounded (the estimate can flip sign);
conditioning the residual on the adjustment set recovers the true interventional effect.

Sequential-ignorability setting with history ``H = (x, z)``: with ``z`` in the adjustment set the
backdoor path ``u <- z -> x'`` is blocked and the effect is identified; omit ``z`` and it is not.
When ``z`` is *latent*, an instrument ``w`` identifies the effect via 2SLS; a Cinelli-Hazlett
robustness value bounds how much hidden confounding a control decision could tolerate.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    return {
        "effect": float(beta[1]),
        "std_error": float(se[1]),
        "robustness_value": float(rv),
    }
