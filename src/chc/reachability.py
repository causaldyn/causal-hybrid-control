"""Hamilton-Jacobi backward reachable tube under a partially identified control effect.

``chc.barrier`` answers a *pointwise* question -- at this state, does some admissible action keep
``d/dt h >= -alpha*h`` once the effect matrix is only set-identified? That is sufficient for forward
invariance but not necessary: a function that fails the barrier condition may still admit a
controller that never leaves ``{h >= 0}``, because a barrier certificate demands the safe set be
invariant *through h itself* rather than merely be avoidable. This module computes the question a
barrier only approximates,

    V(x, T) = max_u min_{Delta_B} min_{s in [0, T]} h(xi(s; x, u, Delta_B)),

so ``{V(., T) >= 0}`` is the set from which safety is guaranteed for ``T`` more seconds against
*every* effect matrix in the identified ball. ``V <= h`` always, and the gap is exactly the price of
a finite horizon; the gap against the barrier's own certified set is the price of using ``h`` as its
own certificate.

The two are not independent. For a control-affine plant with a Euclidean actuation ball and an
operator-norm identification radius ``Delta`` (the same assumptions :mod:`chc.barrier` needs for its
closed form), the upper Hamiltonian is

    H(x, p) = max_{||u|| <= U} min_{||Delta_B|| <= Delta} p . (f + (B + Delta_B) u)
            = p . f + U * (||B^T p|| - Delta*||p||)_+,

which is :func:`chc.barrier.robust_barrier_margin` with ``p`` in place of ``grad h``. The
reachability solver and the barrier certificate therefore run on the *same* robust-margin algebra;
what differs is that ``p = grad V`` is solved for rather than assumed. That also means the §40
zero-action rule reappears here as a property of the Hamiltonian: once ``Delta*||p|| >= ||B^T p||``
the control term vanishes and the tube collapses at the drift's rate.

Scope: a uniform 2-D grid with the standard Lax-Friedrichs scheme (Osher-Fedkiw; Mitchell's
level-set toolbox). Two states because that is where a grid method is honest -- the cost is
exponential in the dimension, and this exists to *calibrate* the barrier certificate on small
problems, not to replace it on large ones.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class ReachableTube:
    """``V(., T_k)`` on a grid: the guaranteed-safe value after ``T_k`` more seconds."""

    axes: tuple[Array, Array]  # the two coordinate vectors the grid is a product of
    times: Array  # (K+1,) backward horizons, times[0] = 0 where V == h
    values: Array  # (K+1, n1, n2)

    @property
    def initial(self) -> Array:
        """``V(., 0) = h`` -- the safe set with no horizon demanded of it."""
        return self.values[0]

    @property
    def final(self) -> Array:
        """``V(., T)`` after the full horizon; ``>= 0`` is guaranteed-safe for all of it."""
        return self.values[-1]

    def safe_fraction(self, index: int = -1) -> float:
        """Fraction of grid points still guaranteed safe at ``times[index]``."""
        return float(jnp.mean(self.values[index] >= 0.0))

    def interpolate(self, point: Array, index: int = -1) -> Array:
        """Bilinear value at an arbitrary ``point``, so a plan's states can be scored off-grid."""
        coords = [
            jnp.clip((point[i] - axis[0]) / (axis[1] - axis[0]), 0.0, float(axis.shape[0] - 1))
            for i, axis in enumerate(self.axes)
        ]
        return jax.scipy.ndimage.map_coordinates(
            self.values[index], coords, order=1, mode="nearest"
        )


def robust_hamiltonian(
    p: Array, drift: Array, b_matrix: Array, u_max: float, radius: float
) -> Array:
    """``p.f + U*(||B^T p|| - Delta*||p||)_+`` -- the §40 margin with a solved-for gradient.

    The inner ``min`` over ``||Delta_B||_op <= radius`` is attained by aligning ``Delta_B u`` with
    ``-p``, which costs ``radius*||p||*||u||``; the outer ``max`` over ``||u|| <= u_max`` then
    spends the whole budget along ``B^T p``, or nothing once the radius swallows the channel.
    """
    channel = jnp.linalg.norm(b_matrix.T @ p)
    return jnp.dot(p, drift) + u_max * jnp.maximum(channel - radius * jnp.linalg.norm(p), 0.0)


def _gradients(values: Array, spacing: tuple[float, float]) -> tuple[Array, Array]:
    """Backward and forward one-sided differences per axis, extending the *gradient* at the edge.

    Repeating the edge **value** instead would make the boundary cell absorbing: its one-sided
    difference is then zero by construction, the Lax-Friedrichs dissipation balances the Hamiltonian
    exactly, and a level set flowing out of the box freezes at the edge and diffuses that error back
    inwards. Repeating the edge *slope* (Mitchell's ``addGhostExtrapolate``) lets it leave.
    """
    backward, forward = [], []
    for axis, step in enumerate(spacing):
        inner = jnp.diff(values, axis=axis) / step  # (n-1,) along axis
        edges = [
            jnp.take(inner, jnp.array([0]), axis),
            inner,
            jnp.take(inner, jnp.array([-1]), axis),
        ]
        diffs = jnp.concatenate(edges, axis=axis)  # (n+1,) along axis
        n = values.shape[axis]
        backward.append(jnp.take(diffs, jnp.arange(n), axis))
        forward.append(jnp.take(diffs, jnp.arange(1, n + 1), axis))
    return jnp.stack(backward), jnp.stack(forward)


def _wave_speeds(drifts: Array, b_matrix: Array, u_max: float, radius: float) -> Array:
    """Per-axis bound on ``|dH/dp|`` -- how much Lax-Friedrichs dissipation the scheme must add.

    ``dH/dp = f + U*(B B^T p/||B^T p|| - radius*p/||p||)`` wherever the control term is active. The
    textbook bound ``|f_i| + U*(||B_i.|| + radius)`` is valid but **discontinuous in the radius**:
    past §40's zero-action threshold the control term vanishes identically, so the bound should drop
    to ``|f_i|`` and instead keeps growing. A scheme that dissipates for authority it does not have
    smears a tube it never had to smear. The derivative is 1-homogeneous in ``p``, so maximising it
    over the unit circle is continuous in the radius and far tighter.

    The margin covers the angular discretisation, which is only ``O(dtheta)`` accurate at the kink
    where the control term switches off. Over-estimating a wave speed costs accuracy;
    under-estimating it costs stability, so the error is deliberately one-sided.
    """
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 4096, endpoint=False)
    p = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    channel = jnp.linalg.norm(p @ b_matrix, axis=-1)  # ||B^T p||, and ||p|| == 1
    direction = (p @ b_matrix) / jnp.maximum(channel, 1e-12)[:, None]
    active = (channel > radius)[:, None]
    speeds = jnp.abs(u_max * (direction @ b_matrix.T - radius * p)) * active
    return jnp.max(jnp.abs(drifts), axis=(0, 1)) + 1.05 * jnp.max(speeds, axis=0)


def backward_reachable_tube(
    barrier: Callable[[Array], Array],
    drift: Callable[[Array], Array],
    b_matrix: Array,
    *,
    lower: tuple[float, float],
    upper: tuple[float, float],
    resolution: tuple[int, int] = (81, 81),
    horizon: float = 1.0,
    steps: int = 200,
    u_max: float = 1.0,
    radius: float = 0.0,
) -> ReachableTube:
    """Solve ``dV/dT = min(0, H(x, grad V))`` from ``V(., 0) = h`` by Lax-Friedrichs.

    Args:
        barrier: ``h(x)``, safe where ``h >= 0``; the same function :func:`chc.plan.certify_safety`
            takes, so the two answers are about the same set.
        drift: ``f(x)``, the uncontrolled field. Control-affine is assumed, the control entering
            through the constant ``b_matrix`` -- the assumption :mod:`chc.barrier`'s closed form
            needs anyway, and the reason this module does not accept a general ``Dynamics``.
        radius: the §32 operator-norm identification radius on ``B``. ``0`` recovers the ordinary
            (exactly identified) reachable tube.

    Raises:
        ValueError: if the step violates the CFL condition, which would let the scheme report a
            *larger* safe set than the truth -- the one failure mode a safety tool must not have.
    """
    axes = tuple(
        jnp.linspace(lo, hi, n) for lo, hi, n in zip(lower, upper, resolution, strict=True)
    )
    spacing = (float(axes[0][1] - axes[0][0]), float(axes[1][1] - axes[1][0]))
    mesh = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)

    h_values = jax.vmap(jax.vmap(lambda x: jnp.squeeze(barrier(x))))(mesh)
    drifts = jax.vmap(jax.vmap(drift))(mesh)  # (n1, n2, 2)

    alpha = _wave_speeds(drifts, b_matrix, u_max, radius)
    dt = horizon / steps
    cfl = float(dt * jnp.sum(alpha / jnp.asarray(spacing)))
    if cfl > 1.0:
        raise ValueError(
            f"CFL number {cfl:.3f} > 1: the scheme would be unstable and could report an "
            f"optimistic safe set. Raise `steps` above {int(steps * cfl) + 1} or coarsen the grid."
        )

    def hamiltonian(p: Array, f: Array) -> Array:
        return robust_hamiltonian(p, f, b_matrix, u_max, radius)

    def step(values: Array, _: None) -> tuple[Array, Array]:
        backward, forward = _gradients(values, spacing)
        central = 0.5 * (backward + forward)
        raw = jax.vmap(jax.vmap(hamiltonian))(jnp.moveaxis(central, 0, -1), drifts)
        # The dissipation ADDS to the numerical Hamiltonian for dV/dT = H: writing the PDE as
        # V_T + (-H) = 0, Lax-Friedrichs subtracts sum_i alpha_i (p+ - p-)/2 from (-H), which lands
        # as a PLUS here. (p+ - p-) ~ dx * V_xx, so this is diffusion; getting the sign wrong makes
        # it ANTI-diffusion and the solve blows up rather than merely losing accuracy.
        dissipation = 0.5 * jnp.sum(alpha[:, None, None] * (forward - backward), axis=0)
        updated = values + dt * jnp.minimum(0.0, raw + dissipation)
        return updated, updated

    _, trajectory = jax.lax.scan(step, h_values, None, length=steps)
    return ReachableTube(
        axes=(axes[0], axes[1]),
        times=jnp.arange(steps + 1) * dt,
        values=jnp.concatenate([h_values[None], trajectory]),
    )


@dataclass(frozen=True)
class BarrierReachabilityGap:
    """What a *pointwise* §40 check does and does not tell you, measured against the true tube.

    The CBF theorem is a statement about a **set**: if the robust barrier condition holds at every
    point of ``{h >= 0}``, that set is forward invariant. Checking it at one state -- which is what
    :func:`chc.plan.certify_safety` does per step, and what any online safety filter does -- is a
    strictly weaker thing, and these fields separate the two.
    """

    safe_fraction: float  # grid fraction with h >= 0: the set the barrier is about
    reachable_fraction: float  # grid fraction with V(., T) >= 0: safe for T more seconds, truly
    barrier_fraction: float  # grid fraction where h >= 0 AND the pointwise condition holds
    valid_cbf: bool  # the condition holds at EVERY point of {h >= 0}, so the theorem applies
    certified_but_unreachable: float  # pointwise-certified yet outside the tube -- the trap
    radius: float
    ok: bool  # valid_cbf implies the tube must be all of {h >= 0}


def barrier_reachability_gap(
    barrier: Callable[[Array], Array],
    drift: Callable[[Array], Array],
    b_matrix: Array,
    *,
    lower: tuple[float, float],
    upper: tuple[float, float],
    resolution: tuple[int, int] = (81, 81),
    horizon: float = 1.0,
    steps: int = 200,
    u_max: float = 1.0,
    radius: float = 0.0,
    alpha: float = 1.0,
    tolerance: float = 0.02,
) -> BarrierReachabilityGap:
    """Run the §40 condition and the horizon-``T`` tube on one grid and report where they part.

    The barrier condition asks ``max_u [grad h . f + <B^T grad h, u> - radius*||grad h||*||u||]
    >= -alpha*h``; the tube asks whether *any* controller holds ``h >= 0`` for ``horizon`` seconds
    against every effect matrix in the identified ball. The implication runs one way and only with a
    quantifier: if the condition holds at **every** point of ``{h >= 0}``, that set is invariant and
    the tube must be all of it -- which is what ``ok`` checks. Pointwise the implication is simply
    false, and ``certified_but_unreachable`` is how much of the grid it is false on: states where
    the §40 margin is satisfied right now, yet no controller survives the horizon. Relative degree
    is the usual reason -- when ``B^T grad h == 0`` the condition degenerates into a statement about
    the drift and stops seeing the actuator at all.

    Args:
        alpha: the class-K gain of the barrier condition, matching :func:`chc.plan.certify_safety`.
        tolerance: grid-fraction slack when comparing the tube to ``{h >= 0}``, absorbing the
            Lax-Friedrichs dissipation that erodes the zero level set by ``O(dx)``.
    """
    tube = backward_reachable_tube(
        barrier,
        drift,
        b_matrix,
        lower=lower,
        upper=upper,
        resolution=resolution,
        horizon=horizon,
        steps=steps,
        u_max=u_max,
        radius=radius,
    )
    mesh = jnp.stack(jnp.meshgrid(*tube.axes, indexing="ij"), axis=-1)

    def condition(x: Array) -> Array:
        scalar = lambda z: jnp.squeeze(barrier(z))  # noqa: E731
        h, grad_h = scalar(x), jax.grad(scalar)(x)
        margin = robust_hamiltonian(grad_h, drift(x), b_matrix, u_max, radius)
        return margin >= -alpha * h

    safe_mask = tube.initial >= 0.0
    holds = jax.vmap(jax.vmap(condition))(mesh)
    barrier_mask = safe_mask & holds
    reachable_mask = tube.final >= 0.0

    safe = float(jnp.mean(safe_mask))
    reachable = tube.safe_fraction()
    valid_cbf = bool(jnp.all(~safe_mask | holds))
    return BarrierReachabilityGap(
        safe_fraction=safe,
        reachable_fraction=reachable,
        barrier_fraction=float(jnp.mean(barrier_mask)),
        valid_cbf=valid_cbf,
        certified_but_unreachable=float(jnp.mean(barrier_mask & ~reachable_mask)),
        radius=radius,
        ok=(not valid_cbf) or reachable >= safe - tolerance,
    )
