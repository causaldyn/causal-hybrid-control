"""Learned residual backends (Strategy): MLP, RBF Kolmogorov-Arnold, graph, and two structured
backbones that carry a guarantee -- port-Hamiltonian (passive, Lyapunov-stable) and Lipschitz
(certified bounded gain). GP is future."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class ZeroResidual(eqx.Module):
    """A residual that contributes nothing — recovers the pure known dynamics."""

    out_dim: int = eqx.field(static=True)

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return jnp.zeros((self.out_dim,))


class MLPResidual(eqx.Module):
    """Learned residual ``r_θ(x, u)`` backed by an MLP over the concatenated ``[x, u]``.

    One interchangeable backend; the point of the abstraction is that KAN / RBF / GP slot in here
    without touching dynamics or control.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        out_dim: int,
        width: int = 16,
        depth: int = 2,
        *,
        key: Array,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=state_dim + control_dim,
            out_size=out_dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.mlp(jnp.concatenate([x, u]))


class RBFKANLayer(eqx.Module):
    """One Kolmogorov-Arnold layer with RBF edge functions (FastKAN-style).

    Each input-output edge is a learnable 1D map ``phi(x) = sum_g c_g rbf_g(x) + w silu(x)`` over a
    fixed radial-basis grid; the output is ``bias_j + sum_i phi_{ji}(z_i)``. Each edge is an
    extractable 1D curve (interpretable) and cheap to evaluate.
    """

    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    num_grid: int = eqx.field(static=True)
    grid_range: float = eqx.field(static=True)
    coeff: Array
    base_weight: Array
    bias: Array

    def __init__(
        self, in_dim: int, out_dim: int, num_grid: int, grid_range: float, *, key: Array
    ) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_grid = num_grid
        self.grid_range = grid_range
        k_coeff, k_base = jax.random.split(key)
        self.coeff = (in_dim * num_grid) ** -0.5 * jax.random.normal(
            k_coeff, (out_dim, in_dim, num_grid)
        )
        self.base_weight = in_dim**-0.5 * jax.random.normal(k_base, (out_dim, in_dim))
        self.bias = jnp.zeros(out_dim)

    def __call__(self, z: Array) -> Array:
        grid = jnp.linspace(-self.grid_range, self.grid_range, self.num_grid)
        inv_h = (self.num_grid - 1) / (2.0 * self.grid_range)
        phi = jnp.exp(-(((z[:, None] - grid[None, :]) * inv_h) ** 2))  # (in_dim, num_grid)
        spline = jnp.einsum("oig,ig->o", self.coeff, phi)
        return self.bias + spline + self.base_weight @ jax.nn.silu(z)


class KANResidual(eqx.Module):
    """Learned residual ``r_θ(x, u)`` backed by a Kolmogorov-Arnold layer (RBF edges).

    A drop-in :class:`ResidualModel` alternative to :class:`MLPResidual`; interpretable, and the
    same training / adjoint machinery applies unchanged.
    """

    layer: RBFKANLayer

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        out_dim: int,
        num_grid: int = 8,
        grid_range: float = 3.0,
        *,
        key: Array,
    ) -> None:
        self.layer = RBFKANLayer(state_dim + control_dim, out_dim, num_grid, grid_range, key=key)

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.layer(jnp.concatenate([x, u]))


class GraphResidual(eqx.Module):
    """Message-passing residual over a fixed graph -- a GNN backend for networked/spatial dynamics.

    The state is ``n_nodes`` blocks of ``node_dim``. Each node's update is an MLP of its features,
    the mean of its neighbours' encoded features (one message-passing round), and the control. It is
    permutation-equivariant and parameter-shared, learning a coupling a pointwise MLP re-learns per
    node. The adjacency is frozen (``stop_gradient``); see ``plans/16``.
    """

    adjacency: Array
    n_nodes: int = eqx.field(static=True)
    node_dim: int = eqx.field(static=True)
    encoder: eqx.nn.MLP
    message: eqx.nn.MLP

    def __init__(
        self, adjacency: Array, node_dim: int, control_dim: int, hidden: int = 16, *, key: Array
    ) -> None:
        k_enc, k_msg = jax.random.split(key)
        degree = jnp.sum(adjacency, axis=1, keepdims=True)
        self.adjacency = adjacency / jnp.maximum(degree, 1.0)  # row-normalised (mean aggregation)
        self.n_nodes = adjacency.shape[0]
        self.node_dim = node_dim
        self.encoder = eqx.nn.MLP(node_dim, hidden, hidden, 1, activation=jax.nn.tanh, key=k_enc)
        self.message = eqx.nn.MLP(
            node_dim + hidden + control_dim, node_dim, hidden, 1, activation=jax.nn.tanh, key=k_msg
        )

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        adjacency = jax.lax.stop_gradient(self.adjacency)  # graph structure is fixed, not trained
        nodes = x.reshape(self.n_nodes, self.node_dim)
        messages = adjacency @ jax.vmap(self.encoder)(nodes)
        control = jnp.broadcast_to(u, (self.n_nodes, u.shape[0]))
        update = jax.vmap(self.message)(jnp.concatenate([nodes, messages, control], axis=1))
        return update.reshape(-1)


class PortHamiltonianResidual(eqx.Module):
    """Port-Hamiltonian residual ``x' = (J - R) grad H(x) + g(x) u`` -- passive, Lyapunov-stable.

    Encodes energy + dissipation + a control port: ``J = A - A^T`` skew (lossless interconnection),
    ``R = L L^T >= 0`` dissipation, ``H_theta`` a scalar energy MLP, ``g(x)`` the input matrix. With
    no input the energy obeys ``H' = dH . (J - R) dH = -dH . R dH <= 0`` (the skew term
    ``dH . J dH = 0``, ``dH := grad H``), so ``H`` is a Lyapunov function -- the residual can't blow
    up off-support like a black box can, exactly the offline-control failure mode. Cleanest
    when the known part is itself port-Hamiltonian (as :class:`~chc.dynamics.DampedOscillator` is),
    so ``known + residual`` stays port-Hamiltonian; as a pure additive correction the bound is on
    the residual's own contribution. Pure autograd (one scalar-MLP gradient), no inner solve, so it
    composes with the discrete/diffrax adjoints. Prefer over plain HNN/LNN (no port or dissipation).
    """

    energy: eqx.nn.MLP  # H_theta : R^state -> scalar
    input_map: eqx.nn.MLP  # g_theta : R^state -> R^(state*control), reshaped to the input matrix
    a_raw: Array  # J = a_raw - a_raw.T (skew)
    l_raw: Array  # R = l_raw @ l_raw.T (positive-semidefinite dissipation)
    state_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)

    def __init__(
        self, state_dim: int, control_dim: int, width: int = 16, depth: int = 2, *, key: Array
    ) -> None:
        k_h, k_g, k_a, k_l = jax.random.split(key, 4)
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.energy = eqx.nn.MLP(state_dim, "scalar", width, depth, activation=jax.nn.tanh, key=k_h)
        self.input_map = eqx.nn.MLP(
            state_dim, state_dim * control_dim, width, depth, activation=jax.nn.tanh, key=k_g
        )
        self.a_raw = 0.1 * jax.random.normal(k_a, (state_dim, state_dim))
        self.l_raw = 0.1 * jax.random.normal(k_l, (state_dim, state_dim))

    def energy_gradient(self, x: Array) -> Array:
        """``grad_x H(x)`` -- the port-Hamiltonian co-energy vector."""
        return jax.grad(lambda state: self.energy(state))(x)

    def structure_matrices(self) -> tuple[Array, Array]:
        """The interconnection ``J`` (skew) and dissipation ``R`` (PSD) matrices."""
        return self.a_raw - self.a_raw.T, self.l_raw @ self.l_raw.T

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        skew, dissipation = self.structure_matrices()
        input_matrix = self.input_map(x).reshape(self.state_dim, self.control_dim)
        return (skew - dissipation) @ self.energy_gradient(x) + input_matrix @ u


class LipschitzResidual(eqx.Module):
    """Residual with a CERTIFIED Lipschitz constant ``L`` in ``[x, u]`` -- bounded off-support gain.

    Each linear layer's weight is divided by a rigorous spectral-norm upper bound
    ``sigma_max(W) <= sqrt(||W||_1 ||W||_inf)`` (Schur), so every layer is <= 1-Lipschitz; with the
    1-Lipschitz ``tanh`` the composition is <= 1-Lipschitz, and one learnable output scale sets the
    overall constant ``L = softplus(log_scale)``. This turns the *soft* Lipschitz penalty in
    :mod:`chc.uncertainty` into a by-construction invariant (invariants over guards): non-explosive
    rollouts and an honest ``L`` for the pessimism / regret bounds. The bound is on the residual's
    own contribution; total-system contraction still needs the known Jacobian. Pure matmuls,
    adjoint-friendly.
    """

    weights: list[Array]
    biases: list[Array]
    log_scale: Array  # L = softplus(log_scale), the certified Lipschitz constant in [x, u]
    out_dim: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        out_dim: int,
        width: int = 16,
        depth: int = 2,
        *,
        key: Array,
    ) -> None:
        sizes = [state_dim + control_dim] + [width] * depth + [out_dim]
        keys = jax.random.split(key, len(sizes) - 1)
        self.weights = [
            s_in**-0.5 * jax.random.normal(k, (s_out, s_in))
            for k, s_in, s_out in zip(keys, sizes[:-1], sizes[1:], strict=True)
        ]
        self.biases = [jnp.zeros(s_out) for s_out in sizes[1:]]
        self.log_scale = jnp.asarray(0.0)
        self.out_dim = out_dim

    @staticmethod
    def _spectral_normalized(weight: Array) -> Array:
        # sigma_max(W) <= sqrt(max abs col-sum * max abs row-sum) (Schur): a certified upper bound.
        col = jnp.max(jnp.sum(jnp.abs(weight), axis=0))
        row = jnp.max(jnp.sum(jnp.abs(weight), axis=1))
        return weight / (jnp.sqrt(col * row) + 1e-12)

    def lipschitz_constant(self) -> Array:
        """The certified Lipschitz constant ``L`` of ``r_theta`` with respect to ``[x, u]``."""
        return jax.nn.softplus(self.log_scale)

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        z = jnp.concatenate([x, u])
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            z = jnp.tanh(self._spectral_normalized(weight) @ z + bias)
        z = self._spectral_normalized(self.weights[-1]) @ z + self.biases[-1]
        return self.lipschitz_constant() * z


class ContractiveResidual(eqx.Module):
    """Residual with a CERTIFIED negative one-sided Lipschitz (log-norm) mu < 0 -- contracting.

    ``r(x,u) = -(rate + softplus(eta)) * x + rate * g(x,u)`` with ``g`` a Schur-normalized tanh-MLP
    (``||dg/dx||_2 <= 1``, as in :class:`LipschitzResidual`). By subadditivity of the log-norm and
    ``mu_2(-D) = -min_i D_ii``, the state-Jacobian obeys ``mu_2(J_r) <= -(rate + min softplus(eta))
    + rate = -min softplus(eta) < 0`` -- a certified contraction rate, not a penalty.
    Unlike the non-negative ``||.||``-Lipschitz ``L`` of :class:`LipschitzResidual` (rollout bound
    ``e^{L*T}``), a negative log-norm ``mu = -c`` gives a UNIFORMLY BOUNDED CONTINUOUS-time radius
    ``eps/c`` (no ``e^{L*T}``). For EXPLICIT EULER the discrete contraction factor is
    ``q = sqrt(1 + 2*mu*dt + L^2*dt^2)`` (NOT ``1+mu*dt``; ``L=lipschitz_constant()`` the full
    Lipschitz constant), so contraction needs the step ``dt < 2c/L^2`` (sufficient); under it the
    rollout radius stays bounded and tends to ``eps/c`` as ``dt -> 0`` (see
    ``chc.uncertainty.contractive_rollout_bound``; Rocq ``contractive_euler.v``).
    ``mu`` bounds only the residual's contribution; total-system contraction needs
    ``mu_2(J_known)+margin < 0`` (a gate to check offline). Pure matmuls; the L2/eigen bound is
    exact, general-n verification would use the L-inf (Gershgorin) variant.
    """

    weights: list[Array]  # Schur-normalized -> each layer is 1-Lipschitz; g is 1-Lipschitz
    biases: list[Array]
    log_drift: Array  # eta: the contraction margin per state coord is softplus(eta) > 0
    rate: float = eqx.field(static=True)  # the bounded-coupling scale (the 1-Lipschitz cap on g)
    state_dim: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        width: int = 16,
        depth: int = 2,
        rate: float = 1.0,
        *,
        key: Array,
    ) -> None:
        sizes = [state_dim + control_dim] + [width] * depth + [state_dim]
        keys = jax.random.split(key, len(sizes) - 1)
        self.weights = [
            s_in**-0.5 * jax.random.normal(k, (s_out, s_in))
            for k, s_in, s_out in zip(keys, sizes[:-1], sizes[1:], strict=True)
        ]
        self.biases = [jnp.zeros(s_out) for s_out in sizes[1:]]
        self.log_drift = jnp.zeros(state_dim)
        self.rate = rate
        self.state_dim = state_dim

    def contraction_rate(self) -> Array:
        """The certified contraction rate ``c = |mu| = min_i softplus(eta_i) > 0``."""
        return jnp.min(jax.nn.softplus(self.log_drift))

    def lipschitz_constant(self) -> Array:
        """Full Lipschitz constant ``L`` of ``r`` in ``x`` (needed for the explicit-Euler CFL).

        ``||J_r|| <= ||-(rate + softplus(eta)) I|| + rate*||dg/dx|| <= (rate + max softplus(eta)) +
        rate`` since ``g`` is 1-Lipschitz -- so ``L = 2*rate + max softplus(eta)``. Sets the Euler
        contraction step ``dt < 2c/L^2`` in :func:`chc.uncertainty.contractive_rollout_bound`.
        """
        return 2.0 * self.rate + jnp.max(jax.nn.softplus(self.log_drift))

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        z = jnp.concatenate([x, u])
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            z = jnp.tanh(LipschitzResidual._spectral_normalized(weight) @ z + bias)
        bounded = self.rate * (
            LipschitzResidual._spectral_normalized(self.weights[-1]) @ z + self.biases[-1]
        )
        drift = -(self.rate + jax.nn.softplus(self.log_drift)) * x  # -D x, D = diag(rate+softplus)
        return drift + bounded


@dataclass(frozen=True)
class PortHamiltonianCertificate:
    """Numeric evidence that a :class:`PortHamiltonianResidual` is passive (Lyapunov-stable)."""

    skew_residual: float  # max |dH . J dH| over sampled states -- must be ~0 (skew form vanishes)
    min_dissipation_eig: float  # smallest eigenvalue of R -- must be >= 0 (PSD dissipation)
    max_energy_rate: float  # max autonomous H' = dH . (J-R) dH -- must be <= 0 (non-increasing)
    ok: bool


def port_hamiltonian_certificate(
    seed: int = 0, state_dim: int = 3, control_dim: int = 1, n: int = 64
) -> PortHamiltonianCertificate:
    """Sample states and check passivity: skew ``J`` adds no energy, ``R >= 0``, and ``H' <= 0``."""
    k_model, k_x = jax.random.split(jax.random.PRNGKey(seed))
    model = PortHamiltonianResidual(state_dim, control_dim, key=k_model)
    skew, dissipation = model.structure_matrices()
    grads = jax.vmap(model.energy_gradient)(jax.random.normal(k_x, (n, state_dim)))
    skew_form = jax.vmap(lambda g: g @ skew @ g)(grads)  # exactly 0 in exact arithmetic
    energy_rate = jax.vmap(lambda g: g @ (skew - dissipation) @ g)(grads)  # = -g.R.g <= 0
    skew_residual = float(jnp.max(jnp.abs(skew_form)))
    min_eig = float(jnp.min(jnp.linalg.eigvalsh(dissipation)))
    max_rate = float(jnp.max(energy_rate))
    ok = skew_residual < 1e-5 and min_eig >= -1e-9 and max_rate <= 1e-5
    return PortHamiltonianCertificate(skew_residual, min_eig, max_rate, ok)


@dataclass(frozen=True)
class LipschitzCertificate:
    """Numeric evidence that a :class:`LipschitzResidual` respects its certified constant ``L``."""

    constant: float  # the certified L = softplus(log_scale)
    max_empirical_ratio: float  # max ||dr|| / ||d[x,u]|| over sampled pairs -- must be <= L
    ok: bool


def lipschitz_certificate(
    seed: int = 0, state_dim: int = 3, control_dim: int = 1, out_dim: int = 3, n: int = 200
) -> LipschitzCertificate:
    """Sample pairs and confirm ``||r(a) - r(b)|| <= L * ||a - b||`` for the certified ``L``."""
    k_model, k_a, k_b = jax.random.split(jax.random.PRNGKey(seed), 3)
    model = LipschitzResidual(state_dim, control_dim, out_dim, key=k_model)
    a = jax.random.normal(k_a, (n, state_dim + control_dim))
    b = jax.random.normal(k_b, (n, state_dim + control_dim))
    evaluate = jax.vmap(lambda z: model(0.0, z[:state_dim], z[state_dim:]))
    numerator = jnp.linalg.norm(evaluate(a) - evaluate(b), axis=1)
    denominator = jnp.linalg.norm(a - b, axis=1) + 1e-12
    ratio = float(jnp.max(numerator / denominator))
    constant = float(model.lipschitz_constant())
    return LipschitzCertificate(constant, ratio, ratio <= constant + 1e-6)


@dataclass(frozen=True)
class DampingInjectionCertificate:
    """Evidence that damping injection makes the port-Hamiltonian residual dissipate energy."""

    max_energy_rate: float  # max_t closed-loop Hdot = -dH^T R dH - kappa*y^2 -- must be <= 0
    damping_dissipation: (
        float  # min_t kappa*y_t^2 -- the extra dissipation the control injects (>= 0)
    )
    energy_dissipated: float  # H(x_0) - H(x_final) over the closed-loop Euler rollout (>= 0)
    ok: bool


def damping_injection_certificate(
    seed: int = 0, state_dim: int = 3, kappa: float = 1.0, horizon: int = 40, dt: float = 0.02
) -> DampingInjectionCertificate:
    """Apply u = -kappa*g^T dH to a :class:`PortHamiltonianResidual` and confirm the energy decays.

    The shipped ``port_hamiltonian_certificate`` checks only AUTONOMOUS (u=0) passivity. Here the
    control adds dissipation: the closed-loop rate ``Hdot = dH^T (J-R) dH - kappa*(g^T dH)^2 =
    -dH^T R dH - kappa*(g^T dH)^2 <= 0`` (skew term exactly 0), so ``H`` strictly decays along the
    closed loop. Machine-checked in ``proofs/port_hamiltonian_lyapunov.v`` (scalar/2x2 case).
    HONEST SCOPE: this is the RESIDUAL field's closed loop, and ``H`` (an MLP) is not enforced
    positive-definite, so it decays to ``argmin H``, not necessarily 0 -- convergence to the origin
    needs a PD energy; for the full hybrid ``f_known + r`` the whole-system claim needs ``f_known``
    matching or energy-orthogonality (see the proof header).
    """
    k_model, k_x = jax.random.split(jax.random.PRNGKey(seed))
    model = PortHamiltonianResidual(state_dim, control_dim=1, key=k_model)

    def damping_control(x: Array) -> Array:
        grad_h = model.energy_gradient(x)
        input_matrix = model.input_map(x).reshape(state_dim, 1)
        return -kappa * (input_matrix.T @ grad_h)  # u = -kappa * g^T dH, shape (1,)

    def closed_field(x: Array) -> Array:
        return model(0.0, x, damping_control(x))

    def energy_rate(x: Array) -> Array:
        return model.energy_gradient(x) @ closed_field(x)  # closed-loop Hdot

    def output_power(x: Array) -> Array:
        input_matrix = model.input_map(x).reshape(state_dim, 1)
        y = input_matrix.T @ model.energy_gradient(x)
        return kappa * (y @ y)  # kappa * y^2, the control's injected dissipation

    x = jax.random.normal(k_x, (state_dim,))
    states = [x]
    for _ in range(horizon):
        x = x + dt * closed_field(x)  # explicit-Euler closed-loop rollout
        states.append(x)
    trajectory = jnp.stack(states)
    rates = jax.vmap(energy_rate)(trajectory)
    energies = jax.vmap(model.energy)(trajectory)
    max_rate = float(jnp.max(rates))
    damping = float(jnp.min(jax.vmap(output_power)(trajectory)))
    dissipated = float(energies[0] - energies[-1])
    return DampingInjectionCertificate(
        max_energy_rate=max_rate,
        damping_dissipation=damping,
        energy_dissipated=dissipated,
        ok=max_rate <= 1e-5 and damping >= -1e-9 and dissipated >= -1e-6,
    )
