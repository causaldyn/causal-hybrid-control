"""chc.residual structured backbones: port-Hamiltonian passivity + certified-Lipschitz gain."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from chc.dynamics import DampedOscillator, HybridDynamics
from chc.integrate import rollout
from chc.residual import (
    LipschitzResidual,
    PortHamiltonianResidual,
    SpectralResidual,
    damping_injection_certificate,
    fit_spectral_residual,
    lipschitz_certificate,
    port_hamiltonian_certificate,
    spectral_residual_certificate,
)
from chc.transport import advection_diffusion_field, advection_diffusion_symbol


def test_port_hamiltonian_certificate_confirms_passivity() -> None:
    cert = port_hamiltonian_certificate(seed=0)
    assert cert.ok
    assert cert.skew_residual < 1e-5  # dH^T J dH = 0 for skew J
    assert cert.min_dissipation_eig >= -1e-9  # R = L L^T is PSD
    assert cert.max_energy_rate <= 1e-5  # autonomous energy is non-increasing


def test_port_hamiltonian_structure_matrices_are_skew_and_psd() -> None:
    model = PortHamiltonianResidual(4, 1, key=jax.random.PRNGKey(1))
    skew, dissipation = model.structure_matrices()
    assert jnp.max(jnp.abs(skew + skew.T)) < 1e-6  # J + J^T = 0 (skew-symmetric)
    assert jnp.min(jnp.linalg.eigvalsh(dissipation)) >= -1e-9  # R is positive-semidefinite


def test_port_hamiltonian_output_has_state_shape_and_composes() -> None:
    model = PortHamiltonianResidual(2, 1, key=jax.random.PRNGKey(2))
    x, u = jnp.array([0.4, -0.2]), jnp.array([0.1])
    assert model(0.0, x, u).shape == (2,)  # a state-dimension vector field
    hybrid = HybridDynamics(DampedOscillator(1.0, 0.1), model)
    assert hybrid(0.0, x, u).shape == (2,)  # slots into the additive hybrid unchanged


def test_damping_injection_certificate_dissipates_closed_loop_energy() -> None:
    cert = damping_injection_certificate(seed=0)
    assert cert.ok
    assert cert.max_energy_rate <= 1e-5  # closed-loop Hdot = -dH^T R dH - kappa*y^2 <= 0
    assert cert.damping_dissipation >= -1e-9  # kappa*y^2 >= 0 (control injects dissipation)
    assert cert.energy_dissipated >= -1e-6  # H strictly decays along the closed loop


def test_damping_injection_holds_across_seeds() -> None:
    for seed in range(5):
        cert = damping_injection_certificate(seed=seed)
        assert cert.max_energy_rate <= 1e-5  # the closed-loop decay is not seed-luck


def test_lipschitz_certificate_respects_the_constant() -> None:
    cert = lipschitz_certificate(seed=0)
    assert cert.ok
    assert cert.max_empirical_ratio <= cert.constant + 1e-6  # never exceeds the certified L


def test_lipschitz_constant_bounds_the_true_ratio_after_scaling() -> None:
    model = LipschitzResidual(3, 1, 3, key=jax.random.PRNGKey(3))
    scaled = eqx.tree_at(lambda m: m.log_scale, model, jnp.asarray(2.0))  # push L up to softplus(2)
    key = jax.random.PRNGKey(9)
    a = jax.random.normal(key, (300, 4))
    b = jax.random.normal(jax.random.PRNGKey(10), (300, 4))

    def gain(m: LipschitzResidual) -> float:
        ev = jax.vmap(lambda z: m(0.0, z[:3], z[3:]))
        num = jnp.linalg.norm(ev(a) - ev(b), axis=1)
        den = jnp.linalg.norm(a - b, axis=1) + 1e-12
        return float(jnp.max(num / den))

    assert gain(scaled) <= float(scaled.lipschitz_constant()) + 1e-6  # bound holds at the new L
    assert float(scaled.lipschitz_constant()) > float(model.lipschitz_constant())  # L grew


def test_lipschitz_output_has_out_shape() -> None:
    model = LipschitzResidual(3, 2, 5, key=jax.random.PRNGKey(4))
    out = model(0.0, jnp.ones(3), jnp.ones(2))
    assert out.shape == (5,)


def test_structured_backbones_are_differentiable_through_a_rollout() -> None:
    # a short rollout gradient is the adjoint-compatibility smoke test both backbones must pass
    x0, u_seq, dt = jnp.array([0.3, -0.1]), jnp.zeros((5, 1)), 0.1
    for residual in (
        PortHamiltonianResidual(2, 1, key=jax.random.PRNGKey(5)),
        LipschitzResidual(2, 1, 2, key=jax.random.PRNGKey(6)),
    ):
        hybrid = HybridDynamics(DampedOscillator(1.0, 0.2), residual)

        def loss(model: HybridDynamics) -> Array:
            return jnp.sum(rollout(model, x0, u_seq, dt) ** 2)

        grads = eqx.filter_grad(loss)(hybrid)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)  # finite gradients


def test_spectral_residual_is_exactly_translation_equivariant() -> None:
    # The structural property, and the one an MLP cannot reach by training: a circulant commutes
    # with the cyclic shift by construction, so this is machine zero rather than small.
    model = SpectralResidual(32, 32, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (32,))
    u = jax.random.normal(jax.random.PRNGKey(2), (32,))
    for shift in (1, 7, 31):
        direct = jnp.roll(model(0.0, x, u), shift)
        shifted = model(0.0, jnp.roll(x, shift), jnp.roll(u, shift))
        assert float(jnp.max(jnp.abs(shifted - direct))) < 1e-5


def test_spectral_residual_operator_norm_is_attained_on_its_top_mode() -> None:
    # proofs/spectral_circulant.v: two_mode_bounded gives <=, two_mode_norm_attained gives = on the
    # maximising mode. The gap between the two lines is the whole difference from a Schur bound.
    n = 32
    model = SpectralResidual(n, 0, key=jax.random.PRNGKey(3))
    norm = float(model.operator_norm())
    top = int(jnp.argmax(jnp.abs(model.symbol())))
    witness = jnp.cos(2.0 * jnp.pi * top * jnp.arange(n) / n)
    ratio = float(jnp.linalg.norm(model(0.0, witness, jnp.zeros(0))) / jnp.linalg.norm(witness))
    assert abs(ratio / norm - 1.0) < 1e-4
    generic = jax.random.normal(jax.random.PRNGKey(4), (16, n))
    achieved = jax.vmap(lambda v: jnp.linalg.norm(model(0.0, v, jnp.zeros(0))))(generic)
    assert float(jnp.max(achieved / jnp.linalg.norm(generic, axis=1))) < norm  # not generic


def test_spectral_residual_rejects_a_control_it_cannot_couple() -> None:
    # A circulant is square and co-located; a 3-dim control over a 32-cell field is not a thing it
    # can represent. Fail at construction rather than silently ignoring the channel.
    with pytest.raises(ValueError, match="control_dim must be 0 or state_dim"):
        SpectralResidual(32, 3, key=jax.random.PRNGKey(5))


def test_closed_form_fit_recovers_the_advection_diffusion_operator() -> None:
    # The truth lies in the hypothesis class, and the fit is a linear solve, so recovery is exact up
    # to arithmetic -- no learning rate, no step count, no seed dependence.
    n, length, speed, viscosity = 64, 1.0, 0.8, 0.01
    keys = jax.random.split(jax.random.PRNGKey(6), 3)
    spectra = jax.random.normal(keys[0], (192, n // 2 + 1)) + 1j * jax.random.normal(
        keys[2], (192, n // 2 + 1)
    )
    xs = jax.vmap(lambda c: jnp.fft.irfft(c * jnp.exp(-0.25 * jnp.arange(n // 2 + 1)), n=n))(
        spectra
    )
    us = jnp.zeros_like(xs)
    ys = jax.vmap(lambda x: advection_diffusion_field(x, length, speed=speed, viscosity=viscosity))(
        xs
    )
    fitted = fit_spectral_residual(SpectralResidual(n, 0, key=keys[1]), xs, us, ys)
    truth = advection_diffusion_symbol(n, length, speed=speed, viscosity=viscosity)
    relative = float(jnp.max(jnp.abs(fitted.symbol() - truth)) / jnp.max(jnp.abs(truth)))
    assert relative < 1e-5


def test_spectral_certificate_beats_the_mlp_and_records_where_it_does_not() -> None:
    # Result 48. plans/18 E was skipped under a kill-criterion whose reopening condition was tying a
    # learned spectral operator into chc.transport; both halves exist now, so the criterion is live
    # and every block below can fail on its own.
    curve = spectral_residual_certificate()
    # (1) THE OPERATOR IS RECOVERED. Relative, because the symbol's own scale is nu*n^2/L^2.
    assert curve.symbol_error < 1e-5
    # (2) THE KILL-CRITERION. It must actually win, and with far fewer parameters.
    assert curve.mse_ratio > 100.0
    assert curve.rollout_ratio > 10.0
    assert curve.spectral_params * 50 < curve.mlp_params
    # (3) THE STRUCTURAL ARM, which no amount of MLP training closes.
    assert curve.spectral_equivariance < 1e-4 * curve.mlp_equivariance
    # (4) THE NORM IS ATTAINED, with a named witness -- and not by a generic input, which is why
    # the witness is the point. Contrast the Schur bound, measured slack by two orders here.
    assert abs(curve.norm_attained_ratio - 1.0) < 1e-3
    assert curve.norm_random_ratio < 0.99
    assert curve.schur_slack < 0.2
    # (5) Composing norms loses what composing symbols does not -- the tube's conservatism per step.
    assert curve.tube_conservatism > 1.0
    # (6) THE HONEST NEGATIVE ARM. Fit by Adam on the MLP's own budget the same circulant LOSES,
    # because its kernel entries are O(nu*n^2/L^2) away from a small initialisation. The structure
    # buys a linear solve; it does not buy faster gradient descent.
    assert curve.spectral_adam_test_mse > curve.mlp_test_mse
    assert curve.kernel_scale > 100.0
    assert curve.ok
