"""Off-policy evaluation of a one-step policy value from logged data.

Given logs ``(x, u, r)`` collected under a behaviour policy, :func:`off_policy_value` estimates by
inverse-propensity weighting the value of a candidate policy **on the logger's own states**:
``E_{x ~ d_b} E_{u ~ pi(.|x)} [r]``, the contextual-bandit value. For a policy that moves the
state -- a feedback plan on a plant with memory -- that is not the value of deploying it. On a loop
whose candidate spreads the state past twice the logger's variance, SNIPS converges to -2.92
against a deployed value of -5.76, with the overlap flag set
(``test_the_one_step_estimand_is_not_the_deployed_value``).

Overlap is summarised by the effective sample size of the one-step weights, and ``overlap_ok``
reports whether its fraction clears a threshold. Nothing here refuses: the flag is the caller's to
read, and it sees one-step overlap only -- the stationary state-action ratio a deployed plan needs
can have infinite variance while it reads True. :func:`chc.evaluation.evaluate_plan` estimates the
deployed value, and refuses by name when the logs cannot give it.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc import _units


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
    """Least-squares Gaussian fit of ``u ~ N(W x + b, sigma^2)`` from logged ``(x, u)``.

    One ``sigma`` per action for every state. On logs whose noise varies with the state the fitted
    density is not the logger's, and weights built on it are biased: a candidate identical to the
    logger has mean weight ``1 + chi^2(pi_b || pi_hat)``, which is ``r^2 / sqrt(2 r^2 - 1)`` for a
    standard-deviation misfit ``r`` and infinite at ``r^2 <= 1/2``. In a sequential deployment gate
    simulated on heteroscedastic logs, that bias took the type-I error from 0.013 to 1.000. Where
    the propensity was logged at decision time, use it instead.
    """
    n = xs.shape[1]
    # lstsq cuts singular values below a share of the largest, so in float32 a state logged in
    # units far from the others', or far from zero, fell under the cut and read a weight of exactly
    # 0. Each state is solved for centred, the intercept carrying its level, in its own units
    # (chc._units): scaled by the power of two nearest its spread, exact, so a column whose spread
    # is near 1 keeps every bit. A state the log moved by rounding alone is zeroed: weight 0.
    states = _units.centred(xs)
    design = jnp.concatenate([states.deviations, jnp.ones((xs.shape[0], 1))], axis=1)
    coef = _units.least_squares(design, us)  # (n+1, m)
    std = jnp.std(us - design @ coef, axis=0)
    # a logger that set its action from the state alone leaves no spread to read; the floor is a
    # share of the actions' own spread, or of their root mean square where that spread is rounding,
    # since an absolute one outweighed actions logged in small units, and one of their size grew
    # with their distance from zero
    spread, rounding = _units.spread_and_rounding(us)
    size = jnp.where(rounding, _units.root_mean_square(us), spread)
    floor = 1e-8 * jnp.where(size > 0.0, size, 1.0)
    bias = coef[n] - states.centre @ coef[:n] - states.shift @ coef[:n]
    return GaussianPolicy(weight=coef[:n].T, bias=bias, log_std=jnp.log(jnp.maximum(std, floor)))


def off_policy_value(
    data: dict[str, Array],
    target: GaussianPolicy,
    behavior: GaussianPolicy,
    ess_fraction_threshold: float = 0.1,
) -> dict[str, float | bool]:
    """IPS / self-normalised estimates of the one-step value, plus overlap diagnostics.

    ``data`` has keys ``x`` (N, n), ``u`` (N, m), ``r`` (N,). Returns IPS and self-normalised
    (SNIPS) estimates of ``E_{x ~ d_b} E_{u ~ pi}[r]`` (the module docstring says why that is not a
    deployed plan's value), the effective sample size and its fraction, ``max_weight`` -- the
    largest normalised weight, ``max w / sum w``, a share of the total rather than a weight -- and
    ``overlap_ok``, whether the ESS fraction clears ``ess_fraction_threshold``.
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
