"""Off-policy evaluation: estimate a policy's value from logged data before deploying it.

The pre-deployment safety gate (``plans/02`` §6): given logs ``(x, u, r)`` collected under a
behaviour policy, estimate the value of a candidate target policy by inverse-propensity weighting,
and refuse deployment when the target's actions leave the logged support (no overlap => no
evidence). Overlap is summarised by the effective sample size; a low ESS fraction is untrustworthy.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class GaussianPolicy(eqx.Module):
    """Diagonal-Gaussian policy ``u ~ N(W x + b, diag(exp(log_std))^2)``."""

    weight: Array  # (m, n)
    bias: Array  # (m,)
    log_std: Array  # (m,)

    def mean(self, x: Array) -> Array:
        return self.weight @ x + self.bias

    def log_prob(self, x: Array, u: Array) -> Array:
        std = jnp.exp(self.log_std)
        z = (u - self.mean(x)) / std
        return jnp.sum(-0.5 * z**2 - self.log_std - 0.5 * jnp.log(2 * jnp.pi))


def fit_behavior_policy(xs: Array, us: Array) -> GaussianPolicy:
    """Least-squares Gaussian fit of ``u ~ N(W x + b, sigma^2)`` from logged ``(x, u)``."""
    n = xs.shape[1]
    design = jnp.concatenate([xs, jnp.ones((xs.shape[0], 1))], axis=1)
    coef, *_ = jnp.linalg.lstsq(design, us, rcond=None)  # (n+1, m)
    residual = us - design @ coef
    std = jnp.std(residual, axis=0)
    return GaussianPolicy(weight=coef[:n].T, bias=coef[n], log_std=jnp.log(std + 1e-8))


def off_policy_value(
    data: dict[str, Array],
    target: GaussianPolicy,
    behavior: GaussianPolicy,
    ess_fraction_threshold: float = 0.1,
) -> dict[str, float | bool]:
    """IPS / self-normalised value estimate plus overlap diagnostics.

    ``data`` has keys ``x`` (N, n), ``u`` (N, m), ``r`` (N,). Returns IPS and self-normalised
    (SNIPS) value estimates, the effective sample size and its fraction, the max weight, and
    ``overlap_ok`` (whether the ESS fraction clears the threshold — the deployment gate).
    """
    xs, us, rs = data["x"], data["u"], data["r"]
    log_w = jax.vmap(target.log_prob)(xs, us) - jax.vmap(behavior.log_prob)(xs, us)
    raw_w = jnp.exp(log_w)
    # ESS and SNIPS are scale-invariant; subtract max(log_w) so weights never underflow to 0/0 (NaN)
    # under total non-overlap — that degenerate case then reads as ESS -> 1 (one effective sample).
    stable_w = jnp.exp(log_w - jnp.max(log_w))
    ess = (jnp.sum(stable_w) ** 2) / jnp.sum(stable_w**2)
    ess_fraction = ess / xs.shape[0]
    return {
        "ips_value": float(jnp.mean(raw_w * rs)),
        "snips_value": float(jnp.sum(stable_w * rs) / jnp.sum(stable_w)),
        "ess": float(ess),
        "ess_fraction": float(ess_fraction),
        "max_weight": float(jnp.max(stable_w) / jnp.sum(stable_w)),
        "overlap_ok": bool(ess_fraction >= ess_fraction_threshold),
    }
