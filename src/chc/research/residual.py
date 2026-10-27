"""The research half of :mod:`chc.residual`: the evidence for its circulant backbone.

:func:`spectral_residual_certificate` fits :class:`~chc.residual.SpectralResidual` and an MLP to
the periodic advection-diffusion plant of :mod:`chc.research.transport`, and reports what the
circulant structure buys against the MLP: the symbol's error, held-out and rollout error, and
equivariance. It sits outside the surface 0.20.0 freezes (:mod:`chc.research`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from chc.dynamics import Dynamics
from chc.residual import (
    MLPResidual,
    SpectralResidual,
    fit_spectral_residual,
    lipschitz_certificate,
)
from chc.toeplitz import (
    circulant_matvec,
    circulant_operator_norm,
)

__all__ = [
    "SpectralResidualCurve",
    "spectral_residual_certificate",
]


@dataclass(frozen=True)
class SpectralResidualCurve:
    """Evidence that a circulant backbone earns its place on a translation-invariant plant."""

    symbol_error: float  # max |lambda_hat - lambda_true| / max |lambda_true|: RELATIVE, because
    # the symbol's own scale is nu*n^2/L^2 and an absolute threshold would be a precision claim
    spectral_test_mse: float  # held-out one-step error, closed-form fit
    mlp_test_mse: float  # ... for an MLP with 130x more parameters, fit by Adam
    mse_ratio: float  # mlp / spectral -- THE KILL-CRITERION: this must exceed 1
    spectral_adam_test_mse: float  # the same circulant fit by Adam on the MLP's budget: it LOSES
    kernel_scale: float  # max |truth kernel| = O(nu n^2 / L^2), the distance Adam has to travel
    spectral_rollout_error: float  # after a 40-step Euler rollout, same integrator for both
    mlp_rollout_error: float
    rollout_ratio: float  # mlp / spectral: wider than the one-step gap, because bias compounds
    spectral_equivariance: float  # ||r(roll x) - roll r(x)||: machine zero, by construction
    mlp_equivariance: float  # ... and O(1) for the MLP -- STRUCTURAL, not a training deficit
    spectral_params: int
    mlp_params: int
    norm_attained_ratio: float  # ||C v*|| / (||v*|| ||C||) on the top mode: exactly 1
    norm_random_ratio: float  # the same on random inputs: < 1, so the witness is what matters
    schur_slack: float  # LipschitzResidual's measured ratio / its certified constant: << 1
    tube_conservatism: float  # product-of-norms / exact norm for a two-circulant composition: > 1
    ok: bool


def _band_limited_fields(key: Array, count: int, n: int, decay: float) -> Array:
    """Random periodic fields with a decaying spectrum -- smooth, and exciting every mode."""
    bins = n // 2 + 1
    k_re, k_im = jax.random.split(key)
    envelope = jnp.exp(-decay * jnp.arange(bins))
    spectrum = envelope * (
        jax.random.normal(k_re, (count, bins)) + 1j * jax.random.normal(k_im, (count, bins))
    )
    return jax.vmap(lambda s: jnp.fft.irfft(s, n=n))(spectrum)


def _fit_by_adam(
    model: Dynamics, xs: Array, us: Array, ys: Array, steps: int, lr: float
) -> Dynamics:
    """Adam on the vector-field mean squared error. Same loop and budget for every backbone.

    Not :func:`chc.train.fit_residual`, which fits through an RK4 step from ``(x, u, x_next)``
    transitions: here the target IS the vector field, so both backbones see the same quantity with
    no integrator error between them and the comparison is of operators alone.
    """

    @eqx.filter_grad
    def gradient(m: Dynamics) -> Array:
        pred = jax.vmap(lambda x, u: m(0.0, x, u))(xs, us)
        return jnp.mean((pred - ys) ** 2)

    optimizer = optax.adam(lr)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    step = eqx.filter_jit(gradient)
    for _ in range(steps):
        updates, state = optimizer.update(step(model), state)
        model = eqx.apply_updates(model, updates)
    return model


def _param_count(model: Dynamics) -> int:
    return int(sum(p.size for p in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))))


def spectral_residual_certificate(
    *,
    seed: int = 0,
    n: int = 64,
    length: float = 1.0,
    speed: float = 0.8,
    viscosity: float = 0.01,
    n_train: int = 256,
    n_test: int = 128,
    steps: int = 400,
    lr: float = 0.02,
    rollout: int = 40,
    dt: float = 2e-3,
) -> SpectralResidualCurve:
    """Does a circulant backbone beat an MLP on the plant that justifies it -- and how, exactly.

    This is the backbone's kill-criterion: if the MLP matches it on its own home ground, the
    backbone is deleted and the finding recorded.

    The plant is ``chc.research.transport``'s periodic advection-diffusion field plus a
    Gaussian-smoothed control channel -- translation-invariant by construction, so the truth lies
    IN the circulant hypothesis class. That makes the fit-quality arms a test of inductive bias
    rather than of approximation power (the MLP carries ~130x more parameters and is not
    handicapped), and it makes the EQUIVARIANCE arm the decisive one: that gap cannot be closed by
    more training.

    Each backbone is fit the way it should be: the circulant by the closed-form per-mode least
    squares of :func:`fit_spectral_residual`, the MLP by Adam. ``spectral_adam_test_mse`` records
    what happens when the circulant is instead fit by Adam on the MLP's own budget -- it loses, and
    the reason is arithmetic rather than structural (see that function's docstring).
    """
    from chc.research.transport import (
        advection_diffusion_field,
        advection_diffusion_kernel,
        advection_diffusion_symbol,
        periodic_smoothing_kernel,
    )

    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    control_kernel = periodic_smoothing_kernel(n, length, 0.04)

    def truth(x: Array, u: Array) -> Array:
        field = advection_diffusion_field(x, length, speed=speed, viscosity=viscosity)
        return field + circulant_matvec(control_kernel, u)

    xs = _band_limited_fields(keys[0], n_train, n, 0.25)
    us = _band_limited_fields(keys[1], n_train, n, 0.4)
    ys = jax.vmap(truth)(xs, us)
    xt = _band_limited_fields(keys[2], n_test, n, 0.25)
    ut = _band_limited_fields(keys[3], n_test, n, 0.4)
    yt = jax.vmap(truth)(xt, ut)

    blank = SpectralResidual(n, n, key=keys[4])
    spectral = fit_spectral_residual(blank, xs, us, ys)
    spectral_adam = _fit_by_adam(blank, xs, us, ys, steps=steps, lr=lr)
    mlp = _fit_by_adam(
        MLPResidual(n, n, n, width=n, depth=2, key=keys[5]), xs, us, ys, steps=steps, lr=lr
    )

    def test_mse(m: Dynamics) -> float:
        pred = jax.vmap(lambda x, u: m(0.0, x, u))(xt, ut)
        return float(jnp.mean((pred - yt) ** 2))

    truth_symbol = advection_diffusion_symbol(n, length, speed=speed, viscosity=viscosity)
    truth_kernel = advection_diffusion_kernel(n, length, speed=speed, viscosity=viscosity)

    # Rollout with the SAME explicit-Euler integrator for both, so the comparison is of operators.
    def roll_out(m: Dynamics) -> float:
        def advance(model) -> Array:
            state = xt[0]
            for _ in range(rollout):
                state = state + dt * model(0.0, state, ut[0])
            return state

        reference = advance(lambda _t, x, u: truth(x, u))
        return float(jnp.linalg.norm(advance(m) - reference))

    shift = 7
    rolled = jax.vmap(lambda x, u: (jnp.roll(x, shift), jnp.roll(u, shift)))(xt[:16], ut[:16])

    def equivariance(m: Dynamics) -> float:
        direct = jax.vmap(lambda x, u: jnp.roll(m(0.0, x, u), shift))(xt[:16], ut[:16])
        shifted = jax.vmap(lambda x, u: m(0.0, x, u))(*rolled)
        return float(jnp.max(jnp.abs(shifted - direct)))

    # The norm is ATTAINED: feed the maximising Fourier mode and the ratio is exactly 1.
    top = int(jnp.argmax(jnp.abs(spectral.symbol())))
    witness = jnp.cos(2.0 * jnp.pi * top * jnp.arange(n) / n)
    op_norm = float(spectral.operator_norm())
    attained = float(
        jnp.linalg.norm(spectral(0.0, witness, jnp.zeros(n))) / jnp.linalg.norm(witness) / op_norm
    )
    probes = _band_limited_fields(keys[0], 64, n, 0.0)
    random_ratio = float(
        jnp.max(
            jax.vmap(lambda v: jnp.linalg.norm(spectral(0.0, v, jnp.zeros(n))))(probes)
            / jnp.linalg.norm(probes, axis=1)
        )
        / op_norm
    )

    schur = lipschitz_certificate(seed=seed, state_dim=n, control_dim=n, out_dim=n)

    # Two circulants whose maximising modes differ -- the advection-diffusion field peaks at the
    # highest resolved wavenumber, the smoothing actuator at k = 0.
    composed = float(circulant_operator_norm(circulant_matvec(truth_kernel, control_kernel)))
    separate = float(
        circulant_operator_norm(truth_kernel) * circulant_operator_norm(control_kernel)
    )

    spectral_mse, mlp_mse = test_mse(spectral), test_mse(mlp)
    spectral_roll, mlp_roll = roll_out(spectral), roll_out(mlp)
    curve = SpectralResidualCurve(
        symbol_error=float(
            jnp.max(jnp.abs(spectral.symbol() - truth_symbol)) / jnp.max(jnp.abs(truth_symbol))
        ),
        spectral_test_mse=spectral_mse,
        mlp_test_mse=mlp_mse,
        mse_ratio=mlp_mse / spectral_mse,
        spectral_adam_test_mse=test_mse(spectral_adam),
        kernel_scale=float(jnp.max(jnp.abs(truth_kernel))),
        spectral_rollout_error=spectral_roll,
        mlp_rollout_error=mlp_roll,
        rollout_ratio=mlp_roll / spectral_roll,
        spectral_equivariance=equivariance(spectral),
        mlp_equivariance=equivariance(mlp),
        spectral_params=_param_count(spectral),
        mlp_params=_param_count(mlp),
        norm_attained_ratio=attained,
        norm_random_ratio=random_ratio,
        schur_slack=schur.max_empirical_ratio / schur.constant,
        tube_conservatism=separate / composed,
        ok=False,
    )
    ok = (
        curve.symbol_error < 1e-5  # the operator is RECOVERED -- a few ULPs of float32
        and curve.mse_ratio > 100.0  # THE KILL-CRITERION: the backbone must actually win
        and curve.rollout_ratio > 10.0
        and curve.spectral_params < curve.mlp_params / 50  # and win with far fewer parameters
        and curve.spectral_equivariance < 1e-4 * curve.mlp_equivariance  # structural, not learned
        and abs(curve.norm_attained_ratio - 1.0) < 1e-3  # the norm is realised by a named input
        and curve.norm_random_ratio < 0.99  # ... and not by a generic one, so the witness matters
        and curve.schur_slack < 0.2  # the Schur bound has no witness and is measurably slack
        and curve.tube_conservatism > 1.0  # composing bounds loses what composing symbols does not
        and curve.spectral_adam_test_mse > curve.mlp_test_mse  # the honest negative arm, recorded
    )
    return replace(curve, ok=ok)
