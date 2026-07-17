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

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from chc.causal import _polynomial_features, _ridge_predict


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
        return {"x": x, "z": z, "u": u, "e": e, "x_nb": x_nb, "z_nb": z_nb, "x_next": y}


def estimate_network_effects(
    data: dict[str, Array],
    covariates: tuple[str, ...] = ("x", "z", "x_nb", "z_nb"),
    exposure: str = "e",
    degree: int = 2,
    folds: int = 5,
    ridge: float = 1.0,
    seed: int = 0,
) -> dict[str, float]:
    """Cross-fitted DML for the direct and spillover effects (partials out graph-aware nuisances).

    Residualises the outcome, treatment ``u``, and exposure ``e`` on flexible predictions from the
    (own + mean-neighbour) covariates, then regresses the outcome residual on the two treatment
    residuals -- the coefficients are the direct and spillover effects.
    """
    y, u, e = data["x_next"], data["u"], data[exposure]
    covs = jnp.stack([data[c] for c in covariates], axis=1)
    n = y.shape[0]
    chunks = jnp.array_split(jax.random.permutation(jax.random.key(seed), n), folds)
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
