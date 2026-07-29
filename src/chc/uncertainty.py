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

from dataclasses import dataclass
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from chc.dynamics import Dynamics, HybridDynamics, LinearDynamics
from chc.integrate import rk4_step
from chc.residual import ContractiveResidual, LipschitzResidual, MLPResidual
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


def cvar_upper(values: Array, alpha: float) -> Array:
    """Upper CVaR (superquantile) of equally-weighted ``values`` at level ``alpha``.

    Mean of the worst ``alpha`` fraction, with the exact fractional-tail weight rather than a
    rounded count, so it agrees with :func:`_top_tail_mean`'s convention and is continuous in
    ``alpha``. Differentiable: ``jnp.sort`` is a permutation, so this is a fixed linear functional
    of the order statistics almost everywhere. ``alpha = 1`` gives the mean, ``alpha -> 0`` the max.
    """
    n = values.shape[0]
    ordered = jnp.sort(values)[::-1]
    mass = alpha * n
    weights = jnp.clip(mass - jnp.arange(n), 0.0, 1.0)
    return jnp.sum(weights * ordered) / mass


class NestedCVaRPenalty(eqx.Module):
    """Time-consistent (nested) CVaR of the ensemble's disagreement, as a ``PenaltyModel``.

    :class:`EnsembleUncertainty` sums the member *variance* along the trajectory, which is a risk
    *neutral* aggregation: one very bad step averages away against many quiet ones. This replaces
    the aggregation with a nested conditional risk measure, ``rho_t = c_t + CVaR_alpha[rho_{t+1}]``
    (Ruszczynski 2010; Shapiro-Dentcheva-Ruszczynski 2021), evaluated on the per-member deviations
    ``||r_k - mean_k r_k||``. Same ``penalty_trajectory(xs, us) -> scalar`` shape, so it drops into
    the ``lam_unc`` channel of :func:`chc.support.pessimistic_control` unchanged.

    The distinction that matters is **which adversary the number prices**, and both are available:

    * ``penalty_trajectory`` (nested) -- the adversary re-picks the worst ``alpha`` fraction of
      members *at every step*, so a model may be wrong here and a different one wrong there.
    * ``static_penalty_trajectory`` -- the adversary must commit to one member for the whole
      horizon, then the tail is taken once over those horizon-summed scenarios.

    Nested is therefore never smaller (subadditivity of CVaR,
    ``Sum_t CVaR[c_t] >= CVaR[Sum_t c_t]`` -- a standard coherence property, not a result of this
    repo), and the gap is the price of time consistency. Time consistency is the reason to pay it:
    a plan optimal for the *static* measure at ``t = 0`` need not be the plan you would re-choose at
    ``t = 1`` after seeing the first step, so a receding-horizon controller that re-solves -- which
    is what :func:`chc.mpc.mpc_control` does -- can chase its own tail. The nested measure has no
    such gap by construction.

    HONEST SCOPE: with the members re-evaluated independently at each visited pair, the recursion
    collapses to ``Sum_t CVaR_alpha[c_t]``; this is the *aggregation* rule made risk-averse, not a
    dynamic-programming solve of a nested-risk MDP, and it does not certify a bound on the cost.
    """

    ensemble: EnsembleResidual
    alpha: float = eqx.field(static=True, default=0.2)

    def member_costs(self, xs: Array, us: Array) -> Array:
        """``(H, K)`` per-member deviation from the ensemble mean field at each visited pair."""

        def step(x: Array, u: Array) -> Array:
            outputs = self.ensemble.member_outputs(0.0, x, u)
            return jnp.linalg.norm(outputs - jnp.mean(outputs, axis=0), axis=-1)

        return jax.vmap(step)(xs, us)

    def penalty_trajectory(self, xs: Array, us: Array) -> Array:
        """Nested: the tail is taken per step, so the worst members may differ across steps."""
        per_step = self.member_costs(xs, us)
        return jnp.sum(jax.vmap(lambda c: cvar_upper(c, self.alpha))(per_step))

    def static_penalty_trajectory(self, xs: Array, us: Array) -> Array:
        """Static: one tail over horizon-summed per-member scenarios (time-INCONSISTENT)."""
        return cvar_upper(jnp.sum(self.member_costs(xs, us), axis=0), self.alpha)


@dataclass(frozen=True)
class NestedRiskCertificate:
    """Numeric evidence that the nested measure dominates the static one, and by how much."""

    nested: float
    static: float
    risk_neutral: float  # the same aggregation at alpha = 1, where every arm must coincide
    gap: float  # nested - static >= 0: the price of time consistency
    ok: bool


def nested_risk_certificate(
    xs: Array, us: Array, ensemble: EnsembleResidual, alpha: float = 0.2
) -> NestedRiskCertificate:
    """Check ``nested >= static >= risk-neutral``, and that ``alpha = 1`` collapses the three."""
    penalty = NestedCVaRPenalty(ensemble=ensemble, alpha=alpha)
    nested = float(penalty.penalty_trajectory(xs, us))
    static = float(penalty.static_penalty_trajectory(xs, us))
    neutral = NestedCVaRPenalty(ensemble=ensemble, alpha=1.0)
    mean_nested = float(neutral.penalty_trajectory(xs, us))
    mean_static = float(neutral.static_penalty_trajectory(xs, us))
    return NestedRiskCertificate(
        nested=nested,
        static=static,
        risk_neutral=mean_nested,
        gap=nested - static,
        ok=(
            nested >= static - 1e-6
            and static >= mean_nested - 1e-6
            and abs(mean_nested - mean_static) <= 1e-4 * max(1.0, abs(mean_nested))
        ),
    )


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


class ConfoundingRobustPenalty(eqx.Module):
    """Pessimism against a HIDDEN-CONFOUNDING effect error, for the general OC stack (§34/§38).

    The §35 controller robustifies a scalar tracker; this lifts the same sensitivity radius into the
    multivariate hybrid solver. Under hidden confounding the control-effect matrix ``B`` is only
    partially identified: the offline estimate ``B_hat`` sits within a sensitivity half-width
    ``radius`` (the §32 ``Delta(Gamma)``) of the truth. By the §34 dimensional insight a per-step
    control ``u_t`` then carries a per-step **transition** error
    ``||Delta_B @ u_t|| <= radius * ||u_t||``, and ``radius * Sigma_t ||u_t||`` accumulates that
    over a trajectory. Penalising it steers control away from *exploiting* a partially-identified
    effect with large actions -- the confounding analogue of :class:`WassersteinPenalty`'s
    deployment-shift margin. Satisfies the ``PenaltyModel`` protocol
    (``penalty_trajectory(xs, us) -> scalar``), so it drops into the ``lam_unc`` channel of
    :func:`chc.support.pessimistic_control` unchanged.

    HONEST SCOPE: this is an **identification-radius regulariser**, not a certified cost bound. The
    §34 inequality bounds the *state-transition* error; converting it into a bound on the objective
    needs a sensitivity multiplier -- ``Delta J <= Sigma_t L_{V,t+1} * radius * ||u_t||`` with
    ``L_{V,t+1}`` the Lipschitz constant of the cost-to-go (locally, the adjoint norm
    ``||lambda_{t+1}||``) -- which is *not* supplied here, so ``lam_unc`` absorbs it as an
    unidentified scale rather than deriving it. The COEFFICIENT is nonetheless derived from the §32
    sensitivity rather than being an arbitrary actuation budget; ``Gamma`` and the CVaR-gap
    calibration remain the analyst's inputs. It does NOT test for confounding.
    """

    radius: float = eqx.field(static=True)

    @classmethod
    def from_sensitivity(cls, cvar_gap: float, gamma: float) -> ConfoundingRobustPenalty:
        """Radius = the §32 bounded-density-ratio inflation ``(Gamma-1)/(Gamma+1) * cvar_gap``."""
        return cls(radius=confounding_robust_inflation(cvar_gap, 0.0, gamma))

    def penalty_trajectory(self, xs: Array, us: Array) -> Array:
        """Confounding pessimism ``radius * Sigma_t ||u_t||`` over the controls (``xs`` unused)."""
        del xs  # the confounded effect error scales with the ACTION magnitude (§34), not the state
        # smoothed L2 norm sqrt(||u||^2 + eps^2): ||u|| is non-differentiable at u=0 (NaN grad) and
        # the solver starts from us0=0 exactly on that singularity, so the floor is squared -- it
        # lives in ||u||^2 units, smoothing over a length scale eps=1e-6. Stays ABOVE ||u||, which
        # is what the §34 upper bound needs; the price is a constant eps per step at u=0, which
        # shifts the reported objective by lam_unc*radius*T*eps without moving the optimiser.
        per_step = jnp.sqrt(jnp.sum(us**2, axis=-1) + 1e-6**2)
        return self.radius * jnp.sum(per_step)


def lipschitz_rollout_bound(lipschitz: float, model_error: float, dt: float, horizon: int) -> float:
    """Certified H-step trajectory-error bound from a Lipschitz field via discrete Gronwall.

    Two Euler rollouts of an ``L``-Lipschitz field, one perturbed per step by ``<= model_error``,
    deviate by ``e_k`` obeying ``e_{k+1} <= (1 + L*dt)*e_k + dt*model_error``, ``e_0 = 0``, so

        ``e_H <= model_error * ((1 + L*dt)^H - 1) / L``   (``L>0``; the ``L->0`` limit is linear,
        ``model_error * dt * H``, no blow-up).

    Turns the CERTIFIED constant ``L`` of :class:`~chc.residual.LipschitzResidual` (plus the trusted
    known-field norm) into a *certified* pessimism radius that propagates a per-step error BUDGET
    through time. This COMPLEMENTS (does not replace) :class:`chc.support.SupportModel`: the support
    density-distance scores WHERE the one-step error ``model_error`` is large (off-support), and
    the bound propagates it (see :func:`support_calibrated_error`). ``model_error`` should be a
    one-step bound (a conformal upper quantile or a bounded-disturbance envelope), not a validation
    average. Machine-checked in ``proofs/lipschitz_rollout.v`` (``rollout_error_bound``); derived in
    ``validation/lipschitz_rollout.mac``. HONEST SCOPE: the bound is ``exp(L*T)`` (``T = H*dt``) --
    tight for small ``L*T`` (bounded-gain residual, short horizon, safety-critical), loose else; a
    contraction metric (:class:`~chc.residual.ContractiveResidual`, log-norm ``mu<0``) removes the
    exponential; the open loop assumes a FIXED action sequence -- for re-planning use
    :func:`closed_loop_rollout_bound`; for per-step ``L_k`` use :func:`time_varying_rollout_bound`.
    """
    growth = 1.0 + lipschitz * dt
    if lipschitz <= 0.0:  # L -> 0: the Gronwall closed form degrades to the linear envelope
        return model_error * dt * horizon
    return model_error * (growth**horizon - 1.0) / lipschitz


class _PerturbedField(eqx.Module):
    """A field with a fixed additive per-step error -- a model-error stand-in for the cert."""

    field: Dynamics
    perturb: Array

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.field(t, x, u) + self.perturb


@dataclass(frozen=True)
class LipschitzRolloutCertificate:
    """Numeric evidence: the measured rollout deviation stays under the certified Gronwall bound."""

    lipschitz: float  # L = ||A||_2 (trusted known field) + certified residual constant
    model_error: float  # per-step field error eps injected into the learned part
    certified_bound: float  # Gronwall e_H <= eps*((1+L*dt)^H - 1)/L
    measured_deviation: float  # actual ||x_H - x~_H|| under Euler rollout from a shared start
    ok: bool  # measured <= certified bound (the guarantee holds)


def _euler_rollout(dyn: Dynamics, x0: Array, us: Array, dt: float) -> Array:
    """Explicit-Euler rollout ``x_{k+1}=x_k+dt*f`` -- matches the proven Gronwall recurrence."""
    x = x0
    states = [x0]
    for k in range(us.shape[0]):
        x = x + dt * dyn(0.0, x, us[k])
        states.append(x)
    return jnp.stack(states)


def lipschitz_rollout_certificate(
    seed: int = 0, state_dim: int = 2, horizon: int = 8, dt: float = 0.05, model_error: float = 0.1
) -> LipschitzRolloutCertificate:
    """Roll two fields differing by a bounded per-step error; confirm the deviation obeys the bound.

    The field is ``f_known + r`` with ``f_known`` a linear map of known spectral norm and ``r`` a
    :class:`~chc.residual.LipschitzResidual` with certified constant ``L_r``; ``L = ||A||_2 + L_r``
    upper-bounds its ``x``-Lipschitz constant. A fixed error of norm ``model_error`` is added each
    step (the learned part is the fragile one); the Euler deviation from a shared start stays under
    ``lipschitz_rollout_bound(L, model_error, dt, horizon)``.
    """
    k_a, k_res, k_x, k_p = jax.random.split(jax.random.PRNGKey(seed), 4)
    a_raw = jax.random.normal(k_a, (state_dim, state_dim))
    a_matrix = 0.5 * a_raw / jnp.linalg.norm(a_raw, 2)  # a known field with ||A||_2 = 0.5
    known = LinearDynamics(a_matrix, jnp.zeros((state_dim, 1)))
    residual = LipschitzResidual(state_dim, 1, state_dim, key=k_res)
    lipschitz = float(jnp.linalg.norm(a_matrix, 2)) + float(residual.lipschitz_constant())
    field = HybridDynamics(known, residual)

    direction = jax.random.normal(k_p, (state_dim,))
    perturb = (
        model_error * direction / jnp.linalg.norm(direction)
    )  # per-step error, norm = model_error
    perturbed = _PerturbedField(field, perturb)

    x0 = jax.random.normal(k_x, (state_dim,))
    us = jnp.zeros((horizon, 1))
    deviation = jnp.linalg.norm(
        _euler_rollout(field, x0, us, dt) - _euler_rollout(perturbed, x0, us, dt), axis=1
    )
    measured = float(jnp.max(deviation))  # e is monotone, so the max is e_H
    bound = lipschitz_rollout_bound(lipschitz, model_error, dt, horizon)
    return LipschitzRolloutCertificate(
        lipschitz=lipschitz,
        model_error=model_error,
        certified_bound=bound,
        measured_deviation=measured,
        ok=measured <= bound + 1e-9,
    )


def contractive_rollout_bound(
    contraction_rate: float, lipschitz: float, model_error: float, dt: float, horizon: int
) -> float:
    """UNIFORMLY BOUNDED explicit-Euler rollout radius for a CONTRACTING field (log-norm mu=-c<0).

    The discrete Euler contraction factor is ``q = sqrt(1+2*mu*dt+L^2*dt^2)`` (NOT ``1+mu*dt``;
    ``L`` = the full Lipschitz constant), since two Euler steps deviate by
    ``|d+dt*(f(x)-f(y))|^2 <= (1+2*mu*dt+L^2*dt^2)|d|^2`` (Maxima ``contractive_euler.mac``; Rocq
    ``contractive_euler.v``). Contraction (``q<1``) needs the step ``dt<2c/L^2`` (sufficient, from
    the worst-case ``(mu,L)`` bounds); under it the discrete Gronwall bound
    ``e_H <= model_error*dt*(1 - q^H)/(1 - q)`` is capped at
    ``model_error*dt/(1-q)`` for ALL horizons -- no ``e^{L*T}`` blow-up -- tends to the continuous
    radius ``model_error/c`` as ``dt -> 0``. The payoff of a certified negative log-norm
    (:class:`~chc.residual.ContractiveResidual`) over the non-negative ``||.||``-Lipschitz of
    :class:`~chc.residual.LipschitzResidual`; the ``q < 1`` cap uses ``gronwall_bounded``. Returns
    ``+inf`` if not contracting or the step is too large (``dt >= 2c/L^2``), where Euler overshoots.
    """
    c = contraction_rate
    q_squared = 1.0 - 2.0 * c * dt + lipschitz**2 * dt**2  # = 1 + 2*mu*dt + L^2*dt^2, mu = -c
    if c <= 0.0 or q_squared >= 1.0 or q_squared < 0.0:  # not contracting / step too large
        return float("inf")
    q = q_squared**0.5
    return model_error * dt * (1.0 - q**horizon) / (1.0 - q)


@dataclass(frozen=True)
class ContractiveRolloutCertificate:
    """Evidence a contractive residual has log-norm mu<0 and a flat (bounded) rollout radius."""

    contraction_rate: float  # certified |mu| = min softplus(eta) > 0
    empirical_one_sided: (
        float  # max_t <dr, dx>/||dx||^2 -- must be <= -contraction_rate (contracting)
    )
    measured_deviation: float  # ||x_H - x~_H|| under the contractive Euler rollout
    bounded_radius: (
        float  # eps*dt/(1-q) -- horizon-independent ceiling (-> eps/c as dt->0, vs e^{L*T})
    )
    lipschitz_blowup: (
        float  # the norm-Lipschitz radius at the same horizon (what contraction avoids)
    )
    ok: bool


def contractive_rollout_certificate(
    seed: int = 0, state_dim: int = 3, horizon: int = 60, dt: float = 0.05, model_error: float = 0.1
) -> ContractiveRolloutCertificate:
    """Confirm a :class:`~chc.residual.ContractiveResidual` contracts and its radius stays flat.

    Checks (i) the empirical one-sided Lipschitz ``<r(x)-r(y),x-y>/||x-y||^2 <= -rate`` (certified
    contraction), and (ii) the perturbed-rollout deviation stays under the BOUNDED explicit-Euler
    radius ``eps*dt/(1-q)`` (step ``dt<2c/L^2``) even as the horizon grows -- contrasted with
    the ``e^{L*T}`` that the same field's norm-Lipschitz ``L`` would incur (``lipschitz_blowup``).
    """
    k_model, k_x, k_p, k_a, k_b = jax.random.split(jax.random.PRNGKey(seed), 5)
    model = ContractiveResidual(state_dim, 1, key=k_model)
    rate = float(model.contraction_rate())

    def residual(x: Array) -> Array:
        return model(0.0, x, jnp.zeros(1))

    a = jax.random.normal(k_a, (200, state_dim))
    b = jax.random.normal(k_b, (200, state_dim))
    delta = a - b
    inner = jax.vmap(lambda p, q, d: (residual(p) - residual(q)) @ d)(a, b, delta)
    one_sided = float(jnp.max(inner / (jnp.sum(delta**2, axis=1) + 1e-12)))

    field = HybridDynamics(
        LinearDynamics(jnp.zeros((state_dim, state_dim)), jnp.zeros((state_dim, 1))), model
    )
    direction = jax.random.normal(k_p, (state_dim,))
    perturb = model_error * direction / jnp.linalg.norm(direction)
    perturbed = _PerturbedField(field, perturb)
    x0 = jax.random.normal(k_x, (state_dim,))
    us = jnp.zeros((horizon, 1))
    deviation = jnp.linalg.norm(
        _euler_rollout(field, x0, us, dt) - _euler_rollout(perturbed, x0, us, dt), axis=1
    )
    measured = float(jnp.max(deviation))
    lipschitz = float(model.lipschitz_constant())
    ceiling = contractive_rollout_bound(rate, lipschitz, model_error, dt, horizon)
    blowup = lipschitz_rollout_bound(  # the same field's norm-Lipschitz e^{L*T} envelope
        lipschitz, model_error, dt, horizon
    )
    return ContractiveRolloutCertificate(
        contraction_rate=rate,
        empirical_one_sided=one_sided,
        measured_deviation=measured,
        bounded_radius=ceiling,
        lipschitz_blowup=blowup,
        ok=one_sided <= -rate + 1e-6 and measured <= ceiling + 1e-6,
    )


# ---- Result 28 UPGRADES (review): time-varying budget, safety tightening, closed loop. ----


def time_varying_rollout_bound(
    lipschitz: list[float], model_error: list[float], dt: float
) -> Array:
    """Per-step certified error tube ``e_0..e_H`` for time-varying ``L_k`` and BUDGET ``eps_k``.

    ``e_{k+1} = (1+L_k*dt) e_k + dt*eps_k``, ``e_0 = 0`` (Rocq ``gronwall_var_comparison``). Unlike
    the constant-``L`` :func:`lipschitz_rollout_bound`, this exposes WHICH step / channel drives the
    growth -- feed :func:`certified_horizon` for the honest ``certified_until_step``. ``eps_k`` is a
    *budget*: it should be a CERTIFIED per-step bound (a conformal quantile, a bounded-disturbance
    envelope, or :func:`support_calibrated_error` combining a base error with
    :class:`chc.support.SupportModel`), not a validation-set average.
    """
    e = 0.0
    tube = [0.0]
    for lk, ek in zip(lipschitz, model_error, strict=True):
        e = (1.0 + lk * dt) * e + dt * ek
        tube.append(e)
    return jnp.asarray(tube)


def certified_horizon(
    lipschitz: list[float], model_error: list[float], dt: float, tolerance: float
) -> int:
    """The largest step ``H`` whose certified error ``e_H`` stays within ``tolerance``.

    Past it the plan is flagged uncertain; ``e`` is monotone, so this is the first crossing.
    """
    e = 0.0
    for h in range(len(lipschitz)):
        e = (1.0 + lipschitz[h] * dt) * e + dt * model_error[h]
        if e > tolerance:
            return h  # e_h <= tolerance but e_{h+1} > tolerance: certified through h steps
    return len(lipschitz)


def closed_loop_rollout_bound(
    state_lipschitz: float,
    control_lipschitz: float,
    policy_lipschitz: float,
    model_error: float,
    dt: float,
    horizon: int,
) -> float:
    """Closed-loop rollout radius when the plan is RE-PLANNED each step by an ``L_pi``-Lipschitz pi.

    A state error perturbs the action, which perturbs the next state, so the growth rate is
    ``L_x + L_u*L_pi`` (not ``L_x`` alone): ``e_{k+1} <= (1 + dt(L_x + L_u*L_pi)) e_k + dt*eps``.
    Reduces to :func:`lipschitz_rollout_bound` with the combined constant. CAVEAT: for a clipping /
    active-set / threshold MPC ``L_pi`` may be unbounded (not globally Lipschitz); use this only
    where the controller is Lipschitz (a fixed feedback gain, or a smooth relaxation).
    """
    return lipschitz_rollout_bound(
        state_lipschitz + control_lipschitz * policy_lipschitz, model_error, dt, horizon
    )


def support_calibrated_error(base_error: float, scale: float, support_distances: Array) -> Array:
    """Per-step error BUDGET ``eps_k = base_error + scale * D(x_k, u_k)`` from a support distance.

    This COMBINES (does not replace) :class:`chc.support.SupportModel`: the certified rollout bound
    propagates a per-step budget through time, and the SupportModel says WHERE that budget is large
    (off-support, where the one-step model error grows). SupportModel scores the local risk, the
    Gronwall tube propagates it.
    """
    return base_error + scale * jnp.asarray(support_distances)


def constraint_margin(nominal_g: Array, lipschitz_g: float, error_radii: Array) -> Array:
    """Tightened constraint margin ``g(x_hat_k) + L_g * e_k`` for an ``L_g``-Lipschitz constraint.

    By Rocq ``constraint_tightening``, if this margin is ``<= 0`` at step ``k`` then the TRUE
    trajectory (within the certified radius ``e_k`` of the nominal ``x_hat_k``) satisfies
    ``g(x_k) <= 0`` -- the tube becomes a *robust feasibility* check. Admissible while all ``<= 0``.
    """
    return jnp.asarray(nominal_g) + lipschitz_g * jnp.asarray(error_radii)


@dataclass(frozen=True)
class TimeVaryingRolloutCertificate:
    """Evidence the time-varying tube is a valid, tighter-than-constant envelope with a cutoff."""

    certified_until_step: int  # largest k with e_k <= tolerance (honest planning horizon)
    varying_final: float  # e_H under the per-step budget
    constant_final: float  # e_H under the constant max-L bound (>= varying_final)
    safe_until_step: int  # largest k with the tightened constraint g(x_hat)+L_g*e_k <= 0
    ok: bool


def time_varying_rollout_certificate(
    seed: int = 0, horizon: int = 20, dt: float = 0.05, tolerance: float = 0.3
) -> TimeVaryingRolloutCertificate:
    """Rising per-step ``L_k`` + a bumpy budget: the tube stays under the constant-max-L bound.

    Identifies ``certified_until_step`` and the safety margin gives a ``safe_until_step`` cutoff.
    """
    key = jax.random.PRNGKey(seed)
    ramp = jnp.linspace(0.5, 3.0, horizon)  # L_k rising over the horizon
    lipschitz = [float(v) for v in ramp]
    budget = [float(0.05 + 0.05 * jnp.abs(jnp.sin(3.0 * v))) for v in ramp]  # eps_k, a bumpy budget
    varying = time_varying_rollout_bound(lipschitz, budget, dt)
    constant = time_varying_rollout_bound([max(lipschitz)] * horizon, budget, dt)
    until = certified_horizon(lipschitz, budget, dt, tolerance)
    nominal_g = -0.5 * jnp.ones(horizon + 1)  # a nominal margin of 0.5 to the constraint g <= 0
    margins = constraint_margin(nominal_g, 1.0, varying)  # L_g = 1
    safe = int(jnp.argmax(margins > 0.0)) if bool(jnp.any(margins > 0.0)) else horizon + 1
    _ = key
    return TimeVaryingRolloutCertificate(
        certified_until_step=until,
        varying_final=float(varying[-1]),
        constant_final=float(constant[-1]),
        safe_until_step=safe,
        ok=float(varying[-1]) <= float(constant[-1]) + 1e-9 and until <= horizon,
    )


def _top_tail_mean(outcomes: NDArray[np.float64], tau: float) -> float:
    """Mean of the top ``tau``-fraction (by mass) of ``outcomes`` -- the upper CVaR / superquantile.

    Exact fractional-tail correction: with ``m = tau * n`` real, average the largest ``floor(m)``
    points at full weight plus the next point at the fractional remainder, normalised by ``m``. This
    is the sharp tail mean the MSM worst-case puts weight ``Gamma`` on. NumPy float64 (like
    :mod:`chc.independence` / :mod:`chc.did`): a sharp quantile bound must not depend on the JAX
    ``jax_enable_x64`` flag.
    """
    y = np.sort(np.asarray(outcomes, dtype=np.float64))[::-1]  # descending
    n = int(y.shape[0])
    m = tau * n
    full = int(m)  # floor for m >= 0
    frac = m - full
    head = float(np.sum(y[:full]))
    if frac > 0.0 and full < n:
        head += frac * float(y[full])
    return head / m


def confounding_robust_inflation(cvar_upper: float, cvar_lower: float, gamma: float) -> float:
    """MSM confounding inflation ``(Gamma-1)/(Gamma+1) * (CVaR_up - CVaR_lo)`` over a point effect.

    Under a BOUNDED DENSITY-RATIO sensitivity model (the weight lies in ``[1/Gamma, Gamma]`` with
    mean 1 -- a marginal, covariate-unconditional special case of Tan's MSM, whose full form
    constrains the treatment-assignment odds with propensity/covariate-dependent bounds) the sharp
    worst-case ``E[wY]`` puts ``w=Gamma`` on the top-``tau`` tail (``tau=1/(Gamma+1)``,
    mean-preserving) and ``w=1/Gamma`` elsewhere. The gap over the nominal mean is this closed form
    (Maxima ``confounding_robust_cvar.mac``; Rocq ``msm_inflation_*``): ``0`` at ``Gamma=1`` (point
    ID), nonnegative, monotone in ``Gamma``. Dorn-Guo (2023); Oprescu et al., B-Learner (2023); Tan
    (2006).
    """
    if gamma < 1.0:
        raise ValueError(f"MSM sensitivity Gamma must be >= 1, got {gamma}")
    return (gamma - 1.0) / (gamma + 1.0) * (cvar_upper - cvar_lower)


def msm_worst_case_mean(outcomes: NDArray[np.float64], gamma: float) -> float:
    """Sharp MSM worst-case (upper) mean of ``outcomes`` over the ``[1/Gamma, Gamma]`` weight box.

    The confounding-robust *pessimistic* value a controller should plan against when the treated
    outcome sample may be confounded up to sensitivity ``Gamma``. Equals ``mean + inflation`` with
    the CVaR tails of :func:`confounding_robust_inflation`; reduces to the sample mean at
    ``Gamma=1``.
    """
    y = np.asarray(outcomes, dtype=np.float64)
    mu = float(np.mean(y))
    if gamma <= 1.0:
        return mu
    tau = 1.0 / (gamma + 1.0)
    cvar_upper = _top_tail_mean(y, tau)  # mean of the worst (largest) tau-tail
    cvar_lower = (mu - tau * cvar_upper) / (1.0 - tau)  # complementary bottom (1-tau) mean
    return mu + confounding_robust_inflation(cvar_upper, cvar_lower, gamma)


def confounding_robust_radius(
    nominal_radius: float, outcomes: NDArray[np.float64], gamma: float
) -> float:
    """Inflate a pessimism radius by the MSM confounding gap: ``rho0 + (worst_case - mean)``.

    Rocq ``robust_radius_ge_nominal`` / ``robust_radius_monotone``: the returned radius is never
    below ``nominal_radius`` (pessimism only grows under assumed confounding) and is monotone in
    ``Gamma``. Feeds :func:`chc.support.pessimistic_control` as a widened uncertainty budget.
    """
    mu = float(np.mean(np.asarray(outcomes, dtype=np.float64)))
    return nominal_radius + (msm_worst_case_mean(outcomes, gamma) - mu)


@dataclass(frozen=True)
class ConfoundingRobustCertificate:
    """Evidence the closed-form MSM worst-case equals the brute-force sharp bound and behaves."""

    closed_form: float  # mean + (Gamma-1)/(Gamma+1)*(CVaR_up - CVaR_lo)
    brute_force: float  # max over integer weight assignments in the [1/Gamma, Gamma] box
    at_gamma_one: float  # worst-case at Gamma=1 (must equal the sample mean)
    sample_mean: float
    monotone: bool  # worst-case nondecreasing over a Gamma grid
    ok: bool


def confounding_robust_certificate(
    seed: int = 0, n: int = 8, gamma: float = 3.0
) -> ConfoundingRobustCertificate:
    """Confirm the CVaR closed form matches the sharp LP worst-case on a mean-preserving grid.

    Chooses ``(n, Gamma)`` so ``tau*n = n/(Gamma+1)`` is an integer, making the sharp box-LP
    optimum a pure top-``tau*n`` assignment (weight ``Gamma`` on the largest points, ``1/Gamma`` on
    the rest) that a brute-force sort computes exactly -- a real check, no LP solver needed.
    """
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n).astype(np.float64)
    mu = float(np.mean(y))
    closed = msm_worst_case_mean(y, gamma)

    tau = 1.0 / (gamma + 1.0)
    k = round(tau * n)  # integer by construction
    ys = np.sort(y)[::-1]  # descending
    brute = float((gamma * np.sum(ys[:k]) + (1.0 / gamma) * np.sum(ys[k:])) / n)

    grid = [1.0, 1.5, 2.0, 3.0, 5.0]
    vals = [msm_worst_case_mean(y, g) for g in grid]
    monotone = all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))

    at_one = msm_worst_case_mean(y, 1.0)
    ok = (
        abs(closed - brute) < 1e-12
        and abs(at_one - mu) < 1e-12
        and monotone
        and closed >= mu - 1e-12  # never optimistic
    )
    return ConfoundingRobustCertificate(
        closed_form=closed,
        brute_force=brute,
        at_gamma_one=at_one,
        sample_mean=mu,
        monotone=monotone,
        ok=ok,
    )


def confounding_robust_closed_loop_bound(
    state_lip: float,
    control_lip: float,
    policy_lip: float,
    base_error: float,
    control_magnitude: float,
    cvar_gap: float,
    gamma: float,
    dt: float,
    horizon: int,
) -> float:
    """Closed-loop rollout radius (Result 31) with the per-step budget inflated by MSM confounding.

    Composes the confounding radius (§32) with the closed-loop replan tube (§31): the growth rate is
    ``L_x + L_u*L_pi`` (re-planning feeds state error through an ``L_pi``-Lipschitz policy). The §32
    half-width ``Delta_B = confounding_robust_inflation(cvar_gap, 0, gamma)`` is an *effect* (``B``)
    error; the per-step *state-transition* error it induces is ``||Delta_B * u_k|| <=
    control_magnitude*Delta_B``, so the budget is ``eps_k = base_error + control_magnitude*Delta_B``
    (the action multiplier makes the units match -- effect error times control is a state error).
    ``gamma=1`` recovers the plain closed-loop tube. Rocq ``closed_loop_confounding_monotone``: the
    tube is monotone in the confounding at every horizon.
    """
    inflated = base_error + control_magnitude * confounding_robust_inflation(cvar_gap, 0.0, gamma)
    return closed_loop_rollout_bound(state_lip, control_lip, policy_lip, inflated, dt, horizon)


@dataclass(frozen=True)
class ConfoundingRobustClosedLoopCertificate:
    """Evidence confounding widens the closed-loop tube and shortens the certified-safe horizon."""

    nominal_radius: float  # Gamma=1: the plain §31 closed-loop tube (no confounding)
    robust_radius: float  # Gamma>1: a wider tube
    nominal_safe_step: int  # certified-safe horizon with no confounding
    robust_safe_step: int  # <= nominal: confounding erodes the safe horizon
    radius_monotone: bool  # tube nondecreasing over a Gamma grid
    horizon_monotone: bool  # safe horizon nonincreasing over a Gamma grid
    ok: bool


def confounding_robust_closed_loop_certificate(
    gamma: float = 3.0, horizon: int = 25, dt: float = 0.05
) -> ConfoundingRobustClosedLoopCertificate:
    """Show the §32-inflated §31 closed-loop tube widens and its safe horizon shrinks with Gamma.

    A re-planning controller (``L_pi``-Lipschitz policy) on an ``L_x``/``L_u`` plant, a nominal
    margin to a constraint ``g <= 0``: as the assumed MSM sensitivity ``Gamma`` grows the
    confounding-inflated tube eats the margin sooner, so the safe planning horizon contracts.
    """
    state_lip, control_lip, policy_lip = 1.0, 0.5, 1.5  # L_cl = 1.0 + 0.5*1.5 = 1.75
    base_error, cvar_gap, control_magnitude = 0.05, 0.4, 0.8  # ||u_k||: effect error -> state error
    nominal_g, lipschitz_g = -0.6, 1.0  # margin 0.6 to the constraint g <= 0
    l_cl = state_lip + control_lip * policy_lip

    def tube_and_safe(g: float) -> tuple[float, int]:
        inflated = base_error + control_magnitude * confounding_robust_inflation(cvar_gap, 0.0, g)
        tube = time_varying_rollout_bound([l_cl] * horizon, [inflated] * horizon, dt)
        margins = constraint_margin(nominal_g * jnp.ones(horizon + 1), lipschitz_g, tube)
        safe = int(jnp.argmax(margins > 0.0)) if bool(jnp.any(margins > 0.0)) else horizon + 1
        return float(tube[-1]), safe

    nominal_radius, nominal_safe = tube_and_safe(1.0)
    robust_radius, robust_safe = tube_and_safe(gamma)
    grid = [1.0, 1.5, 2.0, 3.0, 5.0]
    radii, safes = zip(*(tube_and_safe(g) for g in grid), strict=True)
    radius_monotone = all(radii[i] <= radii[i + 1] + 1e-9 for i in range(len(radii) - 1))
    horizon_monotone = all(safes[i] >= safes[i + 1] for i in range(len(safes) - 1))
    return ConfoundingRobustClosedLoopCertificate(
        nominal_radius=nominal_radius,
        robust_radius=robust_radius,
        nominal_safe_step=nominal_safe,
        robust_safe_step=robust_safe,
        radius_monotone=radius_monotone,
        horizon_monotone=horizon_monotone,
        ok=(
            robust_radius >= nominal_radius - 1e-9
            and robust_safe <= nominal_safe
            and radius_monotone
            and horizon_monotone
        ),
    )
