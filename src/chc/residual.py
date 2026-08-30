"""Learned residual backends (Strategy): MLP, RBF Kolmogorov-Arnold, graph, control-affine, and two
structured backbones that carry a guarantee -- port-Hamiltonian (passive, Lyapunov-stable) and
Lipschitz (certified bounded gain). GP is future."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations_with_replacement

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from chc.dynamics import Dynamics
from chc.toeplitz import (
    circulant_matvec,
    circulant_operator_norm,
    circulant_symbol,
)


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


def control_affine_features(x: Array, degree: int) -> Array:
    """Monomials of a *single* state up to total ``degree``, bias first.

    Owned here rather than shared with :mod:`chc.causal`'s batch version because it indexes the
    parameters of :class:`ControlAffineResidual`: the estimator that fits those parameters
    (:func:`chc.dynamics_id.fit_causal_residual`) has to use this exact basis, and a second
    implementation is a silent way for the two to disagree about what a coefficient means.
    """
    terms = [jnp.ones(())]
    for deg in range(1, degree + 1):
        terms.extend(
            jnp.prod(jnp.stack([x[i] for i in combo]))
            for combo in combinations_with_replacement(range(x.shape[0]), deg)
        )
    return jnp.stack(terms)


class ControlAffineResidual(eqx.Module):
    """``r_θ(x, u) = a_θ(x) + B_θ(x) u``, both coefficients linear in ``control_affine_features``.

    This is the plant class the rest of the library already speaks about:
    :func:`chc.plan.certify_safety` reads the control channel off the Jacobian at ``u = 0`` and
    :mod:`chc.spine` requires a control-affine plant, so identifying a residual *in this class* puts
    the identification layer and the safety layer on the same object rather than on two different
    linearisations of it.

    It is also the class in which the control channel is a **parameter** rather than a derivative of
    a black box, which is what makes it estimable by a partialling-out moment: see
    :func:`chc.dynamics_id.fit_causal_residual`. A general ``r_θ(x, u)`` fitted by prediction error
    has no such guarantee -- under a confounded logging policy it learns the observational response.

    ``degree = 1`` gives a constant channel (the case the orthogonality results §18/§19 cover);
    higher degrees make the channel state-dependent at the cost of leaving that scope.
    """

    drift: Array  # (out_dim, n_features)
    channel: Array  # (out_dim, control_dim, n_features)
    degree: int = eqx.field(static=True, default=1)

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        phi = control_affine_features(x, self.degree)
        return self.drift @ phi + (self.channel @ phi) @ u

    def control_channel(self, x: Array) -> Array:
        """``B_θ(x)`` -- the ``(out_dim, control_dim)`` response of the state rate to the action."""
        return self.channel @ control_affine_features(x, self.degree)

    def drift_jacobian(self, x: Array) -> Array:
        """``∂a_θ/∂x`` at ``x`` -- the ``(out_dim, n)`` local linearisation of the *drift*.

        The companion to :meth:`control_channel`, and it exists because an MPC plans on both halves
        of ``a_θ(x) + B_θ(x) u`` while only the second is identified causally by
        :func:`chc.dynamics_id.fit_causal_residual`.

        This is the drift at ``u = 0`` and **nothing else**: at ``degree >= 1`` the channel depends
        on the state, so the vector field the MPC integrates has Jacobian
        ``∂a_θ/∂x + ∂(B_θ(x) u)/∂x`` and this method returns only the first term. Reading stability
        off it is then a statement about where the actuator's coordinates put their zero rather than
        about the plant: under ``u = alpha v + beta`` the fitted class is closed and the drift
        absorbs ``beta`` times the state-dependent part of the channel, so a plant that decays
        everywhere its actuator can reach can still show a positive drift eigenvalue. On BOPTEST, a
        setpoint-actuated zone reporting its action in ``[15, 25] °C`` came back at ``+6.42`` here
        while :meth:`closed_loop_jacobian` at the setpoint it actually held read ``-1.40``.

        For a stability check use :meth:`closed_loop_jacobian` over the admissible action set.
        """
        return jax.jacobian(lambda z: self.drift @ control_affine_features(z, self.degree))(x)

    def closed_loop_jacobian(self, x: Array, u: Array) -> Array:
        """``∂(a_θ + B_θ u)/∂x`` at ``(x, u)`` -- the linearisation the horizon actually follows.

        Equals :meth:`drift_jacobian` exactly at ``degree = 0``, where the channel is constant and
        the two questions coincide. Everywhere else the difference is the whole of the coordinate
        dependence described above, and it is the difference that decides whether a plan is being
        integrated against a decaying model or an extrapolation.
        """
        return jax.jacobian(lambda z: self(0.0, z, u))(x)


class RBFKANLayer(eqx.Module):
    """One Kolmogorov-Arnold layer with RBF edge functions (FastKAN-style).

    Each input-output edge is a learnable 1D map ``phi(x) = sum_g c_g rbf_g(x) + w silu(x)`` over a
    fixed radial-basis grid; the output is ``bias_j + sum_i phi_{ji}(z_i)``. Each edge is an
    extractable 1D curve (interpretable) and cheap to evaluate.

    ``chc.symbolic`` performs that extraction, and carries the two caveats it needs: an edge's
    intercept is a gauge (a constant moves freely between an edge and the bias, so only the total is
    identified), and this layer represents only additively separable functions -- for an interaction
    it silently returns the best additive fit, whose error is bounded BELOW by ``r^2`` on
    ``[-r, r]^2``.
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


class SpectralResidual(eqx.Module):
    """Residual that IS a circulant operator on a periodic grid, parameterised by its kernel.

    The justifying plant is :func:`chc.transport.advection_diffusion_field` -- a translation-
    invariant vector field on a periodic domain. Every such linear operator is circulant, every
    circulant is diagonalised exactly by the DFT, and its eigenvalues are the DFT of its first
    column (``validation/spectral_circulant.mac`` STEP 1). So this backbone's hypothesis class is
    *exactly* the class of translation-invariant linear maps -- ``n`` parameters, not ``n^2``.

    Two properties no MLP backbone has, both structural rather than matters of fit quality:

    - **Translation equivariance to machine precision.** ``r(roll(x, s)) == roll(r(x), s)`` for
      every shift, by construction. An MLP can be trained towards it and never attains it.
    - **An operator norm that is ATTAINED.** :meth:`operator_norm` returns ``max_k |lambda_k|``,
      the exact spectral norm, with the maximising Fourier mode as an explicit witness. Compare
      :class:`LipschitzResidual`, whose constant comes from the Schur bound
      ``sigma_max(W) <= sqrt(||W||_1 ||W||_inf)`` -- valid, but with no witness and, measured,
      slack by more than an order of magnitude. Because gains multiply exactly under composition
      (Brahmagupta-Fibonacci, ``proofs/spectral_circulant.v`` ``gain_multiplies``), the Result 28/30
      rollout tube built on this constant is tight rather than merely valid.

    The parameterisation is the kernel, not the symbol, deliberately: the map kernel -> circulant is
    a bijection, whereas a free half-spectrum carries two unidentified imaginary parts (at DC and,
    on an even grid, at Nyquist) that ``irfft`` silently discards -- the same gauge problem
    ``chc.symbolic`` prices for KAN edges.

    A circulant is square, so ``out_dim`` is not a free parameter here; and the control channel is
    a circulant too, which is what lets it represent a NONLOCAL actuator. ``control_dim`` must
    therefore be ``0`` or ``state_dim``.
    """

    state_kernel: Array
    control_kernel: Array
    grid: int = eqx.field(static=True)
    has_control: bool = eqx.field(static=True)

    def __init__(
        self, state_dim: int, control_dim: int, *, key: Array, scale: float = 1e-2
    ) -> None:
        if control_dim not in (0, state_dim):
            msg = (
                f"SpectralResidual couples a periodic state field to a co-located control field, "
                f"so control_dim must be 0 or state_dim={state_dim}; got {control_dim}"
            )
            raise ValueError(msg)
        k_state, k_control = jax.random.split(key)
        self.state_kernel = scale * jax.random.normal(k_state, (state_dim,))
        self.control_kernel = scale * jax.random.normal(k_control, (state_dim,))
        self.grid = state_dim
        self.has_control = control_dim == state_dim

    def symbol(self) -> Array:
        """The state operator's eigenvalues as a half spectrum -- complex, length ``n//2 + 1``."""
        return circulant_symbol(self.state_kernel)

    def operator_norm(self) -> Array:
        """Exact spectral norm of the state operator: ``max_k |lambda_k|``, attained not bounded."""
        return circulant_operator_norm(self.state_kernel)

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        out = circulant_matvec(self.state_kernel, x)
        if self.has_control:
            out = out + circulant_matvec(self.control_kernel, u)
        return out


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


def fit_spectral_residual(
    model: SpectralResidual, xs: Array, us: Array, ys: Array
) -> SpectralResidual:
    """Fit a :class:`SpectralResidual` by CLOSED-FORM least squares, per Fourier mode.

    A circulant is linear in its kernel, so gradient descent is the wrong tool: in the Fourier basis
    the design decouples completely and each bin ``k`` is its own two-parameter complex regression
    ``y_k = lambda_k x_k + mu_k u_k``. One 2x2 normal system per bin, solved exactly -- no learning
    rate, no step count, no seed. Same argument as the ordinary least squares behind
    :func:`chc.symbolic.extract_symbolic`, and it stops being available the moment a nonlinearity
    is stacked on top.

    That this matters is a MEASUREMENT, not a preference. The circulant kernel of a spectral
    diffusion operator has entries of order ``nu*n^2/L^2`` -- 134.8 at the certificate's settings --
    and a first-order optimiser starting near zero needs ``O(nu*n^2/lr)`` steps to travel that far.
    Fit by Adam on the same budget as the MLP, this backbone loses; fit properly, it wins by four
    orders of magnitude. The optimiser, not the hypothesis class, decided the naive comparison.

    ``irfft`` projects away any imaginary residue at the DC and Nyquist bins. That is the correct
    projection rather than a loss: for real data those bins of ``x``, ``u`` and ``y`` are all real,
    so the exact solution there is real too, and what is discarded is float noise.
    """
    spec_x = jnp.fft.rfft(xs, axis=1)
    spec_y = jnp.fft.rfft(ys, axis=1)
    n = model.grid
    if not model.has_control:
        gain = jnp.sum(jnp.conj(spec_x) * spec_y, axis=0) / jnp.sum(jnp.abs(spec_x) ** 2, axis=0)
        return eqx.tree_at(lambda m: m.state_kernel, model, jnp.fft.irfft(gain, n=n))

    spec_u = jnp.fft.rfft(us, axis=1)

    def solve_bin(x_k: Array, u_k: Array, y_k: Array) -> Array:
        design = jnp.stack([x_k, u_k], axis=1)
        gram = jnp.conj(design).T @ design
        return jnp.linalg.solve(gram, jnp.conj(design).T @ y_k)

    coefficients = jax.vmap(solve_bin, in_axes=(1, 1, 1))(spec_x, spec_u, spec_y)
    return eqx.tree_at(
        lambda m: (m.state_kernel, m.control_kernel),
        model,
        (
            jnp.fft.irfft(coefficients[:, 0], n=n),
            jnp.fft.irfft(coefficients[:, 1], n=n),
        ),
    )


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

    ``plans/18`` E was skipped under its own kill-criterion, whose sole reopening condition was
    tying a learned spectral operator into ``chc.transport``. Both halves are here, so the
    criterion is live: if the MLP matches this backbone on its own home ground, the backbone is
    deleted and the finding recorded.

    The plant is ``chc.transport``'s periodic advection-diffusion field plus a Gaussian-smoothed
    control channel -- translation-invariant by construction, so the truth lies IN the circulant
    hypothesis class. That makes the fit-quality arms a test of inductive bias rather than of
    approximation power (the MLP carries ~130x more parameters and is not handicapped), and it
    makes the EQUIVARIANCE arm the decisive one: that gap cannot be closed by more training.

    Each backbone is fit the way it should be: the circulant by the closed-form per-mode least
    squares of :func:`fit_spectral_residual`, the MLP by Adam. ``spectral_adam_test_mse`` records
    what happens when the circulant is instead fit by Adam on the MLP's own budget -- it loses, and
    the reason is arithmetic rather than structural (see that function's docstring).
    """
    from chc.transport import (
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
