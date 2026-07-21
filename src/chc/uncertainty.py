"""Calibrated pessimism: a deep-ensemble residual whose disagreement, conformalised for coverage,
scores predictive uncertainty for the offline-safety penalty (plans/19 A; the ``U`` term, plans/02).

``chc.support`` scores *density distance* ``D((x,u),D)``; this module supplies the complementary
*calibrated predictive uncertainty* ``U(x,u)``. Fit K residuals as a deep ensemble; their member
disagreement is the epistemic uncertainty of the learned dynamics (large where the members, trained
on the same offline data, extrapolate apart), and split conformal turns it into interval widths with
a finite-sample coverage guarantee. Both plug into the same penalty channel of
:func:`chc.support.pessimistic_control` via ``penalty_trajectory(xs, us) -> scalar``, so model
exploitation is bounded, not merely discouraged.

A third scorer, :class:`WassersteinPenalty`, targets the *deployment-shift* failure mode rather than
in-distribution epistemic spread: a Wasserstein-1 distributionally-robust margin
(``radius * Sigma ||d r / d x||``) that keeps control where a small shift of the state distribution
cannot move the learned dynamics much (plans/20 §B).
"""

from __future__ import annotations

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc.dynamics import Dynamics, HybridDynamics
from chc.integrate import rk4_step
from chc.residual import MLPResidual
from chc.train import fit_residual


class EnsembleResidual(eqx.Module):
    """Deep ensemble of residuals: K members whose mean is the field, whose spread is uncertainty.

    Duck-types the ``Dynamics`` protocol -- ``__call__`` returns the member mean, so it drops into
    ``HybridDynamics.residual``. ``disagreement`` is the total variance across members (epistemic
    uncertainty of the learned part), ~0 where members agree and large where they extrapolate apart.
    """

    members: tuple[Dynamics, ...]

    def member_outputs(self, t: float | Array, x: Array, u: Array) -> Array:
        """Stack the K member vector fields at ``(t, x, u)`` -> shape ``(K, out_dim)``."""
        return jnp.stack([member(t, x, u) for member in self.members])

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return jnp.mean(self.member_outputs(t, x, u), axis=0)

    def disagreement(self, t: float | Array, x: Array, u: Array) -> Array:
        """Total variance across members at ``(t, x, u)`` -- the epistemic uncertainty scalar."""
        return jnp.sum(jnp.var(self.member_outputs(t, x, u), axis=0))


def fit_ensemble(
    model: HybridDynamics,
    data: dict[str, Array],
    dt: float,
    n_members: int = 5,
    width: int = 16,
    depth: int = 2,
    bootstrap: bool = True,
    seed: int = 0,
    steps: int = 2000,
    lr: float = 1e-2,
) -> tuple[HybridDynamics, list[Array]]:
    """Fit K residuals independently into an ``EnsembleResidual`` (split keys + optional bootstrap).

    Each member is a fresh ``MLPResidual`` (dims inferred from ``data = {x, u, x_next}``) trained by
    :func:`chc.train.fit_residual`. With ``bootstrap`` each member sees a row-resample of the data,
    so disagreement reflects both init randomness and data variation. Returns
    ``HybridDynamics(model.known, EnsembleResidual)`` and the per-member loss histories.
    """
    known = model.known
    xs, us, x_next = data["x"], data["u"], data["x_next"]
    n = xs.shape[0]
    state_dim, control_dim, out_dim = xs.shape[1], us.shape[1], x_next.shape[1]
    members: list[Dynamics] = []
    histories: list[Array] = []
    for member_key in jax.random.split(jax.random.key(seed), n_members):
        k_init, k_boot = jax.random.split(member_key)
        if bootstrap:
            idx = jax.random.randint(k_boot, (n,), 0, n)
            member_data = {"x": xs[idx], "u": us[idx], "x_next": x_next[idx]}
        else:
            member_data = data
        residual = MLPResidual(state_dim, control_dim, out_dim, width, depth, key=k_init)
        fitted, history = fit_residual(
            HybridDynamics(known=known, residual=residual), member_data, dt, steps=steps, lr=lr
        )
        members.append(fitted.residual)
        histories.append(history)
    return HybridDynamics(known=known, residual=EnsembleResidual(members=tuple(members))), histories


class EnsembleUncertainty(eqx.Module):
    """Predictive-uncertainty penalty ``U`` from an ensemble residual -- its member disagreement.

    Mirrors ``SupportModel.penalty_trajectory`` so it plugs into the same ``lam_unc`` channel of
    :func:`chc.support.pessimistic_control`. Penalises visiting states/actions where the learned
    dynamics are uncertain (members disagree) -- exactly where model exploitation happens.
    """

    ensemble: EnsembleResidual

    def penalty_trajectory(self, xs: Array, us: Array) -> Array:
        """Total member disagreement over the visited ``(x, u)`` pairs (xs: (H,n), us: (H,m))."""
        return jnp.sum(jax.vmap(lambda x, u: self.ensemble.disagreement(0.0, x, u))(xs, us))


def _member_next_states(
    known: Dynamics, ensemble: EnsembleResidual, x: Array, u: Array, dt: float
) -> Array:
    """Per-member one-step next-state predictions -> shape ``(K, n)``."""

    def step(member: Dynamics) -> Array:
        return rk4_step(HybridDynamics(known=known, residual=member), 0.0, x, u, dt)

    return jnp.stack([step(member) for member in ensemble.members])


def _predictive_std(
    known: Dynamics, ensemble: EnsembleResidual, x: Array, u: Array, dt: float
) -> Array:
    preds = _member_next_states(known, ensemble, x, u, dt)
    return jnp.sqrt(jnp.sum(jnp.var(preds, axis=0)))


class SplitConformal(eqx.Module):
    """Split-conformal calibration of the ensemble's next-state uncertainty (coverage guarantee).

    Calibrated on a held-out split: the normalised nonconformity score is
    ``s = ||x_next - mean|| / (sigma + eps)`` with ``sigma`` the ensemble predictive std; ``q_hat``
    is the conformal ``(1 - alpha)`` quantile of ``s``. ``interval_width = q_hat*(sigma + eps)`` has
    marginal coverage ``>= 1 - alpha`` on exchangeable data (measured by ``coverage``).
    """

    model: HybridDynamics  # known + EnsembleResidual
    q_hat: Array
    dt: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    @classmethod
    def calibrate(
        cls,
        model: HybridDynamics,
        data: dict[str, Array],
        dt: float,
        alpha: float = 0.1,
        eps: float = 1e-6,
    ) -> SplitConformal:
        known = model.known
        ensemble = cast(EnsembleResidual, model.residual)  # calibrate is only called on ensembles

        def score(x: Array, u: Array, x_next: Array) -> Array:
            mean = jnp.mean(_member_next_states(known, ensemble, x, u, dt), axis=0)
            sigma = _predictive_std(known, ensemble, x, u, dt) + eps
            return jnp.linalg.norm(x_next - mean) / sigma

        scores = jax.vmap(score)(data["x"], data["u"], data["x_next"])
        n = scores.shape[0]
        level = jnp.minimum(jnp.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
        return cls(model=model, q_hat=jnp.quantile(scores, level), dt=dt, alpha=alpha, eps=eps)

    def interval_width(self, x: Array, u: Array) -> Array:
        """Calibrated prediction-interval half-width ``q_hat * (sigma(x,u) + eps)`` (scalar)."""
        ensemble = cast(EnsembleResidual, self.model.residual)
        sigma = _predictive_std(self.model.known, ensemble, x, u, self.dt)
        return self.q_hat * (sigma + self.eps)

    def coverage(self, data: dict[str, Array]) -> float:
        """Empirical coverage on ``data``: fraction of true next-states within the interval."""
        known = self.model.known
        ensemble = cast(EnsembleResidual, self.model.residual)

        def covered(x: Array, u: Array, x_next: Array) -> Array:
            mean = jnp.mean(_member_next_states(known, ensemble, x, u, self.dt), axis=0)
            return jnp.linalg.norm(x_next - mean) <= self.interval_width(x, u)

        return float(jnp.mean(jax.vmap(covered)(data["x"], data["u"], data["x_next"])))

    def penalty_trajectory(self, xs: Array, us: Array) -> Array:
        """Total calibrated interval width over the visited ``(x, u)`` pairs (a penalty)."""
        return jnp.sum(jax.vmap(self.interval_width)(xs, us))


class WassersteinPenalty(eqx.Module):
    """Wasserstein-1 distributionally-robust penalty on the learned residual (plans/20 §B).

    Where ``SupportModel`` scores *in-distribution* density distance ``D`` and the ensemble scores
    epistemic *spread* ``U``, this scores robustness to a *distribution shift* of the states.
    By Kantorovich-Rubinstein W1 duality the worst-case cost over a Wasserstein-1 ball of radius
    ``radius`` equals the empirical cost plus ``radius`` times the cost's Lipschitz constant
    (Gao-Kleywegt; Blanchet-Kang-Murthy; Shafieezadeh-Abadeh-Esfahani-Kuhn). The fragile quantity is
    the *learned* residual ``r`` (the known physics ``f_known`` is trusted, hence excluded), so the
    penalty is ``radius * Sigma_t ||d r / d x||`` -- a gradient-norm margin that steers control away
    from states where a small deployment shift moves the learned dynamics a lot. Satisfies the
    ``PenaltyModel`` protocol, so it drops into the ``lam_unc`` channel of
    :func:`chc.support.pessimistic_control` unchanged; ``radius`` is the distribution-shift budget.
    """

    residual: Dynamics
    radius: float = eqx.field(static=True)

    @classmethod
    def from_model(cls, model: HybridDynamics, radius: float) -> WassersteinPenalty:
        """Penalise the learned residual of a hybrid model (``f_known`` is trusted, so excluded)."""
        return cls(residual=model.residual, radius=radius)

    def local_lipschitz(self, x: Array, u: Array) -> Array:
        """Local Lipschitz constant of the residual in ``x``: Frobenius norm of ``d r / d x``."""
        jac = jax.jacobian(lambda z: self.residual(0.0, z, u))(x)
        return jnp.linalg.norm(jac)  # ||dr/dx||_F; = |grad r| for a scalar residual

    def penalty_trajectory(self, xs: Array, us: Array) -> Array:
        """W1-DRO margin ``radius * Sigma_t ||d r/d x||`` over the visited ``(x, u)`` pairs."""
        return self.radius * jnp.sum(jax.vmap(self.local_lipschitz)(xs, us))
