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

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from numpy.typing import NDArray

from chc.cost import QuadraticCost
from chc.dynamics import Dynamics, HybridDynamics, LinearDynamics
from chc.integrate import rk4_step
from chc.residual import ContractiveResidual, LipschitzResidual, MLPResidual
from chc.train import fit_residual, one_step_mse


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


def _member_mesh(n_members: int) -> Mesh:
    """Largest device mesh whose size divides ``n_members``, so every device holds whole members.

    A mesh that does not divide the member count makes JAX pad the leading axis, which trains
    phantom members to fill the shards; taking the largest divisor instead degrades to a
    single-device mesh (the CPU default) with no branch at the call site.
    """
    devices = jax.devices()
    size = max(d for d in range(1, len(devices) + 1) if n_members % d == 0)
    return Mesh(np.asarray(devices[:size]), ("member",))


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
    """Fit K residuals into an ``EnsembleResidual`` (split keys + optional bootstrap), in parallel.

    Each member is a fresh ``MLPResidual`` (dims inferred from ``data = {x, u, x_next}``) trained by
    the same Adam recursion :func:`chc.train.fit_residual` runs. With ``bootstrap`` each member sees
    a row-resample of the data, so disagreement reflects both init randomness and data variation.
    Returns ``HybridDynamics(model.known, EnsembleResidual)`` and the per-member loss histories.

    The members are trained *together*: one stacked parameter pytree, ``jax.vmap`` over the member
    axis, ``jax.lax.scan`` over the Adam steps, and the stack sharded over the member axis of
    :func:`_member_mesh`. This is one XLA program rather than K x ``steps`` dispatches, and it is
    the same recursion -- :func:`sharded_ensemble_certificate` pins the agreement in ULP.
    """
    known = model.known
    xs, us, x_next = data["x"], data["u"], data["x_next"]
    n = xs.shape[0]
    state_dim, control_dim, out_dim = xs.shape[1], us.shape[1], x_next.shape[1]

    split = jax.vmap(jax.random.split)(jax.random.split(jax.random.key(seed), n_members))
    k_init, k_boot = split[:, 0], split[:, 1]
    if bootstrap:
        rows = jax.vmap(lambda key: jax.random.randint(key, (n,), 0, n))(k_boot)
    else:
        rows = jnp.broadcast_to(jnp.arange(n), (n_members, n))
    stacked = eqx.filter_vmap(
        lambda key: MLPResidual(state_dim, control_dim, out_dim, width, depth, key=key)
    )(k_init)
    params, static = eqx.partition(stacked, eqx.is_inexact_array)
    optimizer = optax.adam(lr)

    def member_loss(member: Any, x: Array, u: Array, y: Array) -> Array:
        residual = eqx.combine(member, static)
        return one_step_mse(HybridDynamics(known=known, residual=residual), x, u, y, dt)

    @jax.jit
    def train(params: Any, x: Array, u: Array, y: Array) -> tuple[Any, Array]:
        def adam_step(carry: Any, _: None) -> tuple[Any, Array]:
            params, opt_state = carry
            loss, grads = jax.vmap(jax.value_and_grad(member_loss))(params, x, u, y)
            updates, opt_state = jax.vmap(optimizer.update)(grads, opt_state, params)
            return (jax.vmap(optax.apply_updates)(params, updates), opt_state), loss

        init = (params, jax.vmap(optimizer.init)(params))
        (params, _), history = jax.lax.scan(adam_step, init, None, length=steps)
        return params, history

    shard = NamedSharding(_member_mesh(n_members), PartitionSpec("member"))
    batch = jax.device_put((params, xs[rows], us[rows], x_next[rows]), shard)
    trained, history = train(*batch)

    members = tuple(
        cast(Dynamics, eqx.combine(jax.tree.map(operator.itemgetter(i), trained), static))
        for i in range(n_members)
    )
    return (
        HybridDynamics(known=known, residual=EnsembleResidual(members=members)),
        [history[:, i] for i in range(n_members)],
    )


_PARITY_ULP_BUDGET = 2000.0
"""Agreement budget in ULP of the working dtype: tight enough that a changed recursion, a
mis-derived key or a dropped bootstrap blows past it by orders of magnitude, loose enough to
absorb the reduction-order difference between K small programs and one batched one."""


@dataclass(frozen=True)
class ShardedEnsembleCertificate:
    """Evidence that the stacked, sharded ensemble fit *is* the serial per-member recursion."""

    n_members: int
    n_devices: int
    mesh_size: int
    dtype: str
    parity_steps: int
    parity_ulp: float
    loss_ulp: float
    shard_devices: int
    disagreement_ulp: float
    ok: bool


def sharded_ensemble_certificate(
    seed: int = 0,
    n_members: int = 8,
    n_samples: int = 96,
    parity_steps: int = 60,
    dt: float = 0.05,
) -> ShardedEnsembleCertificate:
    """Check :func:`fit_ensemble` against a serial oracle built from the untouched public trainer.

    Two claims, each pinned by the measurement that can falsify it. *Numerics*: every member of the
    stacked fit matches the member :func:`chc.train.fit_residual` produces from the same key and the
    same bootstrap rows, reported in ULP of the working dtype rather than as an absolute -- the
    threshold is a precision claim, so it is stated in the unit precision actually has. *Layout*:
    the member axis is genuinely distributed, so a stacked array committed to :func:`_member_mesh`
    and consumed by a jitted ``vmap`` comes back spanning the whole mesh; on a one-device host that
    is trivially 1, and the subprocess test forces eight.

    ``parity_steps`` is deliberately short. Adam on this loss is chaotic: past a few hundred
    steps a one-ULP difference in reduction order is amplified without bound (measured 6.6 ULP
    at 200 float32 steps, 7557 at 400), so a long-horizon parity threshold would be either
    vacuous or flaky. The equivalence certified is of the *recursion*, which a short horizon
    tests exactly.
    """
    known = LinearDynamics(jnp.array([[0.0, 1.0], [-1.0, -0.1]]), jnp.zeros((2, 1)))
    k_x, k_u = jax.random.split(jax.random.PRNGKey(seed))
    xs = jax.random.normal(k_x, (n_samples, 2))
    us = jax.random.normal(k_u, (n_samples, 1))
    x_next = jax.vmap(lambda x, u: x + dt * known(0.0, x, u))(xs, us)
    data = {"x": xs, "u": us, "x_next": x_next}

    seed_model = HybridDynamics(known=known, residual=MLPResidual(2, 1, 2, key=k_x))
    fitted, histories = fit_ensemble(
        seed_model, data, dt, n_members=n_members, seed=seed, steps=parity_steps
    )
    stacked = cast(EnsembleResidual, fitted.residual)

    oracle: list[Dynamics] = []
    oracle_losses: list[float] = []
    for member_key in jax.random.split(jax.random.key(seed), n_members):
        k_init, k_boot = jax.random.split(member_key)
        rows = jax.random.randint(k_boot, (n_samples,), 0, n_samples)
        member = MLPResidual(2, 1, 2, key=k_init)
        trained, history = fit_residual(
            HybridDynamics(known=known, residual=member),
            {"x": xs[rows], "u": us[rows], "x_next": x_next[rows]},
            dt,
            steps=parity_steps,
        )
        oracle.append(trained.residual)
        oracle_losses.append(float(history[-1]))

    unit = float(np.finfo(np.asarray(xs).dtype).eps)
    worst = 0.0
    for reference, candidate in zip(oracle, stacked.members, strict=True):
        pairs = zip(
            jax.tree.leaves(eqx.filter(reference, eqx.is_inexact_array)),
            jax.tree.leaves(eqx.filter(candidate, eqx.is_inexact_array)),
            strict=True,
        )
        for a, b in pairs:
            scale = max(float(jnp.max(jnp.abs(a))), unit)
            worst = max(worst, float(jnp.max(jnp.abs(a - b))) / scale)
    loss_gap = max(
        abs(float(h[-1]) - ref) / max(abs(ref), unit)
        for h, ref in zip(histories, oracle_losses, strict=True)
    )

    mesh = _member_mesh(n_members)
    probe = jax.device_put(jnp.zeros((n_members, 4)), NamedSharding(mesh, PartitionSpec("member")))
    spread = jax.jit(jax.vmap(lambda row: row + 1.0))(probe)
    shard_devices = len(spread.sharding.device_set)

    reference_spread = float(
        EnsembleResidual(members=tuple(oracle)).disagreement(0.0, xs[0], us[0])
    )
    stacked_spread = float(stacked.disagreement(0.0, xs[0], us[0]))
    spread_gap = abs(stacked_spread - reference_spread) / max(reference_spread, unit)

    mesh_size = int(mesh.devices.size)
    return ShardedEnsembleCertificate(
        n_members=n_members,
        n_devices=len(jax.devices()),
        mesh_size=mesh_size,
        dtype=str(jnp.zeros(()).dtype),
        parity_steps=parity_steps,
        parity_ulp=worst / unit,
        loss_ulp=loss_gap / unit,
        shard_devices=shard_devices,
        disagreement_ulp=spread_gap / unit,
        ok=(
            worst < _PARITY_ULP_BUDGET * unit
            and loss_gap < _PARITY_ULP_BUDGET * unit
            and spread_gap < _PARITY_ULP_BUDGET * unit
            and shard_devices == mesh_size
            and n_members % mesh_size == 0
        ),
    )


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

    HONEST SCOPE, and it now depends on which constructor was used. Plain
    ``ConfoundingRobustPenalty(radius=...)`` is an **identification-radius regulariser**: the §34
    inequality bounds the *state-transition* error, and converting that into a bound on the
    objective needs the sensitivity multiplier ``Delta J <= Sigma_t L_{V,t+1} * radius * ||u_t||``,
    which this path does not supply -- so ``lam_unc`` absorbs it as an unidentified scale.
    :meth:`certified` supplies it -- the adjoint norm ``||lambda_{t+1}||``, the RK4 injection gain
    and a second-order deviation tube -- and on that path ``lam_unc = 1`` *is* the bound rather than
    a knob. :func:`confounding_cost_bound_certificate` is the gate that can fail it.

    Either way the COEFFICIENT is derived from the §32 sensitivity rather than being an arbitrary
    actuation budget; ``Gamma`` and the CVaR-gap calibration remain the analyst's inputs, and it
    does NOT test for confounding.
    """

    radius: float = eqx.field(static=True)
    cost_to_go: Array | None = None  # (H,) ||lambda_{t+1}||; None keeps the unweighted regulariser

    @classmethod
    def from_sensitivity(cls, cvar_gap: float, gamma: float) -> ConfoundingRobustPenalty:
        """Radius = the §32 bounded-density-ratio inflation ``(Gamma-1)/(Gamma+1) * cvar_gap``."""
        return cls(radius=confounding_robust_inflation(cvar_gap, 0.0, gamma))

    @classmethod
    def certified(
        cls,
        radius: float,
        dyn: Dynamics,
        x0: Array,
        us_reference: Array,
        dt: float,
        cost: QuadraticCost,
    ) -> ConfoundingRobustPenalty:
        """Supply the missing multiplier, so ``lam_unc = 1`` *is* the bound rather than a free knob.

        The §34 inequality bounds the per-step transition error; turning it into a bound on the
        objective needs ``Delta J <= Sigma_t L_{V,t+1} * radius * ||u_t||``, and ``L_{V,t+1}`` is
        locally the adjoint norm ``||lambda_{t+1}||``. That is a quantity the planner already
        computes, so leaving it to be absorbed by ``lam_unc`` was throwing away information, not
        avoiding an assumption. :func:`chc.adjoint.costate_norms` returns it along ``us_reference``.

        The first-order term alone is *not* an upper bound, and measuring rather than assuming is
        what showed it: at the optimum the Cauchy-Schwarz step is nearly tight, so the dropped
        curvature term -- positive, O(radius^2) -- pushes the realised worst case above it at every
        radius: an adversary reaches 1.014 of it at radius 0.005 and 1.69 at 0.2 on a two-lever
        oscillator, while the weights below hold at 0.995 and 0.953.
        :func:`chc.adjoint.perturbation_cost_weights` therefore carries the deviation tube and the
        cost curvature as well, which is why ``radius`` is an argument here: a second-order term
        cannot be folded into a radius-free weight.

        Three things this is and is not:

        * **Second order, exact for LQ.** For a linear plant with a quadratic cost the objective is
          exactly quadratic along a perturbation direction, so the expansion closes and the only
          slack left is Cauchy-Schwarz. Nonlinear plants keep an ``O(radius^3)`` remainder;
          :func:`confounding_cost_bound_certificate` measures whether it matters.
        * **At a reference.** The weights are frozen at ``us_reference``, which breaks the
          circularity of weights that depend on the plan that depends on the weights. Iterating
          (re-solve, re-weight) is available to the caller and is not done here, because a fixed
          point is a different claim from a bound at a named trajectory.
        * **Not a confounding test.** ``radius`` still comes from an assumed ``Gamma``.
        """
        from chc.adjoint import perturbation_cost_weights

        return cls(
            radius=radius,
            cost_to_go=perturbation_cost_weights(dyn, x0, us_reference, dt, cost, radius),
        )

    def penalty_trajectory(self, xs: Array, us: Array) -> Array:
        """Confounding pessimism ``radius * Sigma_t L_t ||u_t||`` (``xs`` unused).

        ``L_t`` is 1 when no weights were supplied -- the identification-radius regulariser -- and
        :func:`chc.adjoint.perturbation_cost_weights` when they were, on which path the sum is a
        certified upper bound on the cost gap rather than a scale-free direction.
        """
        del xs  # the confounded effect error scales with the ACTION magnitude (§34), not the state
        # smoothed L2 norm sqrt(||u||^2 + eps^2): ||u|| is non-differentiable at u=0 (NaN grad) and
        # the solver starts from us0=0 exactly on that singularity, so the floor is squared -- it
        # lives in ||u||^2 units, smoothing over a length scale eps=1e-6. Stays ABOVE ||u||, which
        # is what the §34 upper bound needs; the price is a constant eps per step at u=0, which
        # shifts the reported objective by lam_unc*radius*T*eps without moving the optimiser.
        per_step = jnp.sqrt(jnp.sum(us**2, axis=-1) + 1e-6**2)
        if self.cost_to_go is not None:
            per_step = per_step * self.cost_to_go
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


class _ControlChannelOffset(eqx.Module):
    """``f(t, x, u) + M u`` -- the support model's admissible field error, made concrete.

    ``||M u|| <= ||M||_2 ||u||``, so a spectral-norm ball of radius ``r`` around zero is exactly the
    set the §34 inequality allows. Distinct from :class:`_PerturbedField`, whose offset is constant
    in ``u`` and therefore models a *drift*, not a mis-identified effect.
    """

    field: Dynamics
    offset: Array  # (n, m)

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.field(t, x, u) + self.offset @ u


def _project_spectral(matrix: Array, radius: float) -> Array:
    """Nearest point of ``{M : ||M||_2 <= radius}`` -- singular values clipped, vectors kept."""
    u, s, vt = jnp.linalg.svd(matrix, full_matrices=False)
    return (u * jnp.clip(s, 0.0, radius)) @ vt


@dataclass(frozen=True)
class ConfoundingBoundCurve:
    """Predicted vs adversarially realised cost gap, radius by radius. ``ok`` iff none exceeded."""

    radii: tuple[float, ...]
    predicted: tuple[float, ...]
    realised: tuple[float, ...]
    ratio: tuple[float, ...]  # realised / predicted; > 1 is the bound failing
    first_order_ratio: tuple[float, ...]  # the same against the first-order term alone
    worst_ratio: float
    ok: bool


def confounding_cost_bound_certificate(
    radii: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
    horizon: int = 20,
    dt: float = 0.05,
    restarts: int = 6,
    ascent_steps: int = 250,
    seed: int = 0,
) -> ConfoundingBoundCurve:
    """Try to break :meth:`ConfoundingRobustPenalty.certified`, and report where it bends.

    A two-lever damped oscillator, planned to optimality, then *attacked*: the adversary picks the
    control-channel error ``M`` with ``||M||_2 <= radius`` that moves the objective most, by
    projected gradient ascent from several starts on both signs of ``Delta J``. Random sampling was
    the first version and it is not good enough -- in four parameters it underestimates the worst
    case by enough to hide a violated bound.

    Two ratios are reported because they answer different questions. ``ratio`` is the certificate:
    realised over the full second-order weight, and ``ok`` demands every entry stay at or below one.
    ``first_order_ratio`` is the diagnostic that motivated the second-order term: it exceeds one at
    every radius, which is why ``||lambda||`` alone was a calibrated *estimate*, not a bound.
    """
    system = LinearDynamics(a_matrix=jnp.array([[0.0, 1.0], [-1.0, -0.2]]), b_matrix=jnp.eye(2))
    cost = QuadraticCost(
        Q=jnp.eye(2), R=0.05 * jnp.eye(2), Qf=5.0 * jnp.eye(2), x_target=jnp.zeros(2)
    )
    x0 = jnp.array([1.0, -0.5])

    from chc.adjoint import perturbation_cost_weights
    from chc.control import projected_gradient_solve
    from chc.cost import total_cost

    us = projected_gradient_solve(
        system, x0, jnp.zeros((horizon, 2)), dt, cost, -3.0, 3.0, steps=30_000
    ).actions
    base = total_cost(system, x0, us, dt, cost)
    # radius 0 zeroes the tube, so the weights collapse to exactly the first-order term.
    first_order = perturbation_cost_weights(system, x0, us, dt, cost, 0.0)

    def gap(offset: Array) -> Array:
        return (
            total_cost(_ControlChannelOffset(field=system, offset=offset), x0, us, dt, cost) - base
        )

    @eqx.filter_jit
    def attack(start: Array, radius: float, sign: float) -> Array:
        step = radius / 5.0

        def ascend(offset: Array, _: None) -> tuple[Array, None]:
            grad = jax.grad(lambda m: sign * gap(m))(offset)
            scale = jnp.maximum(jnp.linalg.norm(grad), 1e-12)
            return _project_spectral(offset + step * grad / scale, radius), None

        final, _ = jax.lax.scan(ascend, _project_spectral(start, radius), None, length=ascent_steps)
        return jnp.abs(gap(final))

    key = jax.random.key(seed)
    starts = jax.random.normal(key, (restarts, 2, 2))
    predicted, realised, first_ratio = [], [], []
    for radius in radii:
        weights = perturbation_cost_weights(system, x0, us, dt, cost, radius)
        per_step = radius * jnp.linalg.norm(us, axis=1)
        predicted.append(float(jnp.sum(weights * per_step)))
        worst = max(float(attack(start, radius, sign)) for start in starts for sign in (1.0, -1.0))
        realised.append(worst)
        first_ratio.append(worst / float(jnp.sum(first_order * per_step)))
    ratio = tuple(r / p for r, p in zip(realised, predicted, strict=True))
    return ConfoundingBoundCurve(
        radii=tuple(radii),
        predicted=tuple(predicted),
        realised=tuple(realised),
        ratio=ratio,
        first_order_ratio=tuple(first_ratio),
        worst_ratio=max(ratio),
        ok=max(ratio) <= 1.0,
    )


# --- Result 32 (A19): Gamma is unfalsifiable only if nobody benchmarks it ---


def _logistic_fit(
    design: NDArray[np.float64],
    treated: NDArray[np.float64],
    ridge: float = 1e-6,
    steps: int = 60,
    tol: float = 1e-10,
) -> NDArray[np.float64]:
    """Ridge-penalised logistic regression by IRLS -- the propensity model benchmarking needs.

    Newton on the penalised log-likelihood; the ridge is on the slopes only, so an intercept-only
    model is unpenalised and the fit stays invariant to shifting the outcome's base rate. Converges
    in a handful of steps on the well-separated designs a benchmark sweeps, and the iteration is
    stopped on the coefficient step rather than the likelihood, which is what a caller comparing two
    *nested* fits needs: the difference of two half-converged logits is not an odds ratio.
    """
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(steps):
        probability = 1.0 / (1.0 + np.exp(-design @ coefficients))
        weights = np.clip(probability * (1.0 - probability), 1e-12, None)
        gradient = design.T @ (treated - probability) - penalty @ coefficients
        hessian = design.T @ (design * weights[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        coefficients = coefficients + step
        if float(np.max(np.abs(step))) < tol:
            break
    return coefficients


@dataclass(frozen=True)
class GammaBenchmark:
    """``Gamma`` expressed in units of the confounding the OBSERVED covariates actually carry."""

    names: tuple[str, ...]
    implied_gamma: tuple[float, ...]  # the odds-ratio range each covariate alone induces
    quantile: float  # of the per-unit logit gap; 1.0 is the MSM's own sup and is n-dependent
    strongest: str
    strongest_gamma: float
    assumed_gamma: float
    multiples_of_strongest: float  # log(assumed)/log(strongest): odds ratios compose, so log scale


def benchmark_gamma(
    treated: NDArray[np.float64],
    covariates: NDArray[np.float64],
    assumed_gamma: float,
    names: Sequence[str] | None = None,
    quantile: float = 0.95,
    ridge: float = 1e-6,
) -> GammaBenchmark:
    """Price an assumed MSM ``Gamma`` against the observed covariates, in MSM units.

    ``Gamma`` is the analyst's unfalsifiable input, and Result 32 says so every time it is used. It
    is unfalsifiable; it is not *uncalibrated*. The MSM bounds the assignment odds ratio between the
    true propensity and the modelled one uniformly over units, and dropping an observed covariate
    produces exactly such a pair -- so

        ``Gamma_j = exp( quantile_i | logit e(x_i) - logit e_{-j}(x_i) | )``

    is the sensitivity a confounder as strong as covariate ``j`` would generate, measured rather
    than assumed.

    **``quantile = 1.0`` is the MSM's own quantity and it is not the default, because it grows
    with the sample.** The model bounds the odds ratio *uniformly* over units, so the sup is the
    faithful statistic -- but under an unbounded covariate the sup is an extreme order statistic.
    Measured on a standard-normal design, the sup runs ``309 -> 382 -> 397 -> 734`` as ``n`` goes
    ``500 -> 4000 -> 32000 -> 128000``, while the 95th percentile sits at ``22, 24, 19, 19`` and
    the median at ``2.7, 3.0, 2.7, 2.8``. A benchmark that quadruples because more data arrived is
    not a benchmark, so the default reports a quantile and says which one; pass ``quantile=1.0``
    when the uniform bound is the point and the covariates are bounded.

    ``multiples_of_strongest`` then reads ``Gamma`` as "an unobserved confounder ``k`` times as
    strong as the strongest thing we did observe", on the log scale odds ratios compose in. A
    ``k`` far below 1 is an assumption nobody should be impressed by; a ``k`` far above 1 is one
    the analyst has to defend.

    SCOPE. Per-covariate and one-at-a-time: this is the *marginal* strength of each covariate, not a
    policy-level aggregation over a set of them, and the two differ whenever covariates are
    correlated. Dropping a covariate from the propensity is a benchmark, not a claim that the
    unobserved confounder resembles it. The propensity is logistic-linear in the columns as passed;
    a benchmark is only as good as that model, which is why the covariates should already carry the
    basis expansion the analyst believes.
    """
    if assumed_gamma < 1.0:
        raise ValueError(f"MSM sensitivity Gamma must be >= 1, got {assumed_gamma}")
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile must lie in (0, 1]; got {quantile}")
    x = np.asarray(covariates, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"covariates must be 2-D (n, p); got shape {x.shape}")
    if x.shape[1] == 0:
        raise ValueError("benchmarking needs at least one observed covariate")
    t = np.asarray(treated, dtype=np.float64).ravel()
    labels = tuple(names) if names is not None else tuple(f"x{j}" for j in range(x.shape[1]))
    if len(labels) != x.shape[1]:
        raise ValueError(f"got {len(labels)} names for {x.shape[1]} covariates")

    design = np.column_stack([np.ones(x.shape[0]), x])
    full_logit = design @ _logistic_fit(design, t, ridge)
    # The gap compares two INDEPENDENT fits, so dropping a column that carries no signal leaves a
    # residue at rounding scale rather than exactly zero. That residue must be snapped away here
    # and not at the division below: exp() turns 1e-16 into 1 + 1e-16, log() turns it back, and the
    # ratio then reports a confounding strength of 1e16 with a straight face. The floor is the
    # backward-stability scale of the matvec that produced the logits, sqrt(n) * eps * |logit|.
    noise = (
        float(np.sqrt(design.shape[0]))
        * float(np.finfo(np.float64).eps)
        * max(1.0, float(np.max(np.abs(full_logit))))
    )
    implied = []
    for j in range(x.shape[1]):
        reduced = np.delete(design, j + 1, axis=1)
        gap = np.abs(full_logit - reduced @ _logistic_fit(reduced, t, ridge))
        spread = float(np.quantile(gap, quantile))
        implied.append(float(np.exp(spread)) if spread > noise else 1.0)

    best = int(np.argmax(implied))
    strongest = implied[best]
    return GammaBenchmark(
        names=labels,
        implied_gamma=tuple(implied),
        quantile=quantile,
        strongest=labels[best],
        strongest_gamma=strongest,
        assumed_gamma=assumed_gamma,
        # log(1) = 0 would divide: a covariate that moves the odds not at all sets no scale, and
        # saying so beats reporting a finite multiple of nothing.
        multiples_of_strongest=(
            float("inf") if strongest <= 1.0 else float(np.log(assumed_gamma) / np.log(strongest))
        ),
    )


def negative_control_gamma(
    outcomes: NDArray[np.float64], tol: float = 1e-9, gamma_max: float = 1e6
) -> float:
    """Smallest ``Gamma`` whose MSM interval covers zero on a KNOWN-NULL outcome.

    The other half of calibration, and the half that can refute. On an outcome whose true effect is
    zero, any nonzero estimate is confounding, so the smallest ``Gamma`` that reconciles the two is
    a **lower bound on the confounding actually present**: assuming less than it is refuted by the
    data rather than merely unappealing. Monotonicity of the MSM inflation in ``Gamma`` makes the
    search a bisection, exact to ``tol``.

    Returns ``inf`` when no ``Gamma`` reconciles the null -- the endpoint saturates at the extreme
    order statistic as ``Gamma -> inf``, so a sample whose values all share the mean's sign cannot
    be pulled across zero by a bounded density ratio at all, and the negative control has refuted
    the model class instead of calibrating it.

    The interval is NOT symmetric about the mean, and which endpoint has to travel depends on the
    sign of the estimate: a positive estimate is reconciled by the *lower* endpoint reaching zero, a
    negative one by the upper. Reflecting the sample by ``-sign(mean)`` maps both onto the single
    upper-endpoint routine :func:`msm_worst_case_mean` computes, which is why the sharp CVaR tails
    are used in the direction that actually binds rather than the convenient one.
    """
    y = np.asarray(outcomes, dtype=np.float64).ravel()
    mu = float(np.mean(y))
    if mu == 0.0:
        return 1.0
    reflected = -np.sign(mu) * y  # mean is now -|mu| < 0; the binding endpoint is the upper one
    if float(np.max(reflected)) < 0.0:
        return float("inf")
    low, high = 1.0, gamma_max
    if msm_worst_case_mean(reflected, gamma_max) < 0.0:
        return float("inf")
    while high - low > tol * max(1.0, low):
        mid = 0.5 * (low + high)
        if msm_worst_case_mean(reflected, mid) >= 0.0:
            high = mid
        else:
            low = mid
    return high


@dataclass(frozen=True)
class GammaBenchmarkCertificate:
    """Falsifiable gates for :func:`benchmark_gamma` and :func:`negative_control_gamma`."""

    strengths: tuple[float, ...]
    implied_by_strength: tuple[float, ...]
    monotone_in_strength: bool
    null_covariate_gamma: float
    ranks_with_truth: bool
    sizes: tuple[int, ...]
    sup_by_size: tuple[float, ...]
    quantile_by_size: tuple[float, ...]
    null_by_size: tuple[float, ...]
    null_floor_scaled: float  # median of log(null_gamma)*sqrt(n): bounded iff the floor is root-n
    sup_growth: float  # max/min of the sup over the sample sizes swept
    quantile_growth: float  # the same ratio for the reported quantile
    quantile_is_stabler: bool
    biases: tuple[float, ...]
    calibrated_gamma: tuple[float, ...]
    endpoint_residual: float  # |binding endpoint| at the calibrated Gamma, worst over the sweep
    calibration_monotone: bool
    unreconcilable_is_infinite: bool
    ok: bool


def gamma_benchmark_certificate(
    strengths: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
    sizes: Sequence[int] = (500, 4000, 32000),
    biases: Sequence[float] = (0.1, 0.2, 0.4),
    quantile: float = 0.95,
    samples: int = 4000,
    seed: int = 0,
) -> GammaBenchmarkCertificate:
    """Check that the benchmark measures confounding strength and that the calibration binds.

    Five gates, each able to fail:

    1. **Monotone in strength.** A covariate whose logit coefficient grows induces a larger implied
       ``Gamma``. A benchmark that does not order confounders by strength orders nothing.
    2. **The null covariate's score is a root-``n`` noise floor.** Dropping a coefficient that is
       truly zero must not manufacture sensitivity beyond sampling noise -- and "beyond sampling
       noise" is a rate, not a constant, so the gate is that ``log(Gamma_null) * sqrt(n)`` stays
       bounded as ``n`` grows rather than that a single draw sits below a fixed number. Measured on
       six seeds per size, that product runs ``2.1, 4.0, 5.0, 4.4`` at ``n = 1000 .. 64000`` while
       ``Gamma_null`` itself falls ``1.069 -> 1.017``; a single draw of ``1.27`` at ``n = 4000`` is
       inside that distribution, which is exactly why one draw cannot be the gate.
    3. **Ranking.** On one design carrying strong / weak / null covariates, the implied ``Gamma``
       ranks them in that order.
    4. **The reported quantile is stabler in ``n`` than the sup.** This is the measurement the
       default rests on, so it is run rather than remembered: the sup is an extreme order statistic
       under an unbounded covariate and grows with the sample; the quantile should not.
    5. **The negative control lands on the endpoint, and grows with the planted bias.** At the
       returned ``Gamma`` the binding interval endpoint sits at zero to solver tolerance, and a
       larger planted bias needs a larger ``Gamma`` -- plus ``inf`` when the sample cannot be
       reconciled at all.
    """
    rng = np.random.default_rng(seed)

    def draw(n: int, beta: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = rng.standard_normal((n, beta.shape[0]))
        p = 1.0 / (1.0 + np.exp(-(x @ beta)))
        return (rng.uniform(size=n) < p).astype(np.float64), x

    implied_by_strength = []
    for strength in strengths:
        treated, x = draw(samples, np.array([float(strength), 0.0]))
        implied_by_strength.append(
            benchmark_gamma(treated, x, 2.0, quantile=quantile).implied_gamma[0]
        )
    monotone = bool(np.all(np.diff(implied_by_strength) > 0.0))

    treated, x = draw(samples, np.array([1.5, 0.4, 0.0]))
    ranked = benchmark_gamma(treated, x, 2.0, names=("strong", "weak", "null"), quantile=quantile)
    null_gamma = ranked.implied_gamma[2]
    ranks = bool(
        ranked.implied_gamma[0] > ranked.implied_gamma[1] > ranked.implied_gamma[2]
        and ranked.strongest == "strong"
    )

    sup_by_size, quantile_by_size, null_by_size = [], [], []
    for n in sizes:
        treated, x = draw(int(n), np.array([1.5, 0.4, 0.0]))
        sup_by_size.append(benchmark_gamma(treated, x, 2.0, quantile=1.0).strongest_gamma)
        at_quantile = benchmark_gamma(treated, x, 2.0, quantile=quantile)
        quantile_by_size.append(at_quantile.strongest_gamma)
        null_by_size.append(at_quantile.implied_gamma[2])
    null_floor_scaled = float(
        np.median([np.log(g) * np.sqrt(n) for g, n in zip(null_by_size, sizes, strict=True)])
    )
    sup_growth = float(max(sup_by_size) / min(sup_by_size))
    quantile_growth = float(max(quantile_by_size) / min(quantile_by_size))

    calibrated, residuals = [], []
    for bias in biases:
        null_outcome = rng.standard_normal(samples) + float(bias)
        gamma = negative_control_gamma(null_outcome)
        calibrated.append(gamma)
        # positive mean: the LOWER endpoint is the one that has to reach zero
        residuals.append(abs(-msm_worst_case_mean(-null_outcome, gamma)))
    calibration_monotone = bool(np.all(np.diff(calibrated) > 0.0))
    unreconcilable = negative_control_gamma(np.abs(rng.standard_normal(samples)) + 1.0)

    ok = bool(
        monotone
        and ranks
        and null_floor_scaled < 12.0
        and quantile_growth < sup_growth
        and calibration_monotone
        and max(residuals) < 1e-6
        and np.isinf(unreconcilable)
    )
    return GammaBenchmarkCertificate(
        strengths=tuple(float(s) for s in strengths),
        implied_by_strength=tuple(implied_by_strength),
        monotone_in_strength=monotone,
        null_covariate_gamma=float(null_gamma),
        ranks_with_truth=ranks,
        sizes=tuple(int(n) for n in sizes),
        sup_by_size=tuple(sup_by_size),
        quantile_by_size=tuple(quantile_by_size),
        null_by_size=tuple(null_by_size),
        null_floor_scaled=null_floor_scaled,
        sup_growth=sup_growth,
        quantile_growth=quantile_growth,
        quantile_is_stabler=bool(quantile_growth < sup_growth),
        biases=tuple(float(b) for b in biases),
        calibrated_gamma=tuple(calibrated),
        endpoint_residual=float(max(residuals)),
        calibration_monotone=calibration_monotone,
        unreconcilable_is_infinite=bool(np.isinf(unreconcilable)),
        ok=ok,
    )
