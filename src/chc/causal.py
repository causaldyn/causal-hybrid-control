"""Confounded offline identification: why the control-effect residual needs the adjustment set.

A minimal linear demonstration of the CHC causal claim (see ``plans/02``). The historical action
was chosen by a behaviour policy correlated with a covariate ``z`` that also drives the outcome.
Fitting the effect of ``u`` without adjusting for ``z`` is confounded (the estimate can flip sign);
conditioning the residual on the adjustment set recovers the true interventional effect.

Sequential-ignorability setting with history ``H = (x, z)``: with ``z`` in the adjustment set the
backdoor path ``u <- z -> x'`` is blocked and the effect is identified; omit ``z`` and it is not.
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
    z_scale: float = 1.0
    eta_scale: float = 0.5
    noise_scale: float = 0.1

    def sample(self, n: int, key: Array) -> dict[str, Array]:
        """Draw ``n`` logged transitions as a dict of columns ``x, z, u, x_next``."""
        k_x, k_z, k_eta, k_noise = jax.random.split(key, 4)
        x = jax.random.normal(k_x, (n,))
        z = self.z_scale * jax.random.normal(k_z, (n,))
        eta = self.eta_scale * jax.random.normal(k_eta, (n,))
        u = self.kappa * z + eta  # behaviour policy — action correlated with the confounder
        noise = self.noise_scale * jax.random.normal(k_noise, (n,))
        x_next = self.a * x + self.b_true * u + self.c * z + noise
        return {"x": x, "z": z, "u": u, "x_next": x_next}


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
