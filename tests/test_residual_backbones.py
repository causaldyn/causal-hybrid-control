"""chc.residual structured backbones: port-Hamiltonian passivity + certified-Lipschitz gain."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from chc.dynamics import DampedOscillator, HybridDynamics
from chc.integrate import rollout
from chc.residual import (
    LipschitzResidual,
    PortHamiltonianResidual,
    damping_injection_certificate,
    lipschitz_certificate,
    port_hamiltonian_certificate,
)


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
