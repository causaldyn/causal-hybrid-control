"""Deep Galerkin Method -- a neural PDE solver, meeting classical Galerkin/Marchuk FEM on one PDE,
and coupled forward-backward into a mean-field game.

The Deep Galerkin Method (Sirignano-Spiliopoulos) trains a network to satisfy a PDE by minimising
its residual at random points -- a mesh-free Galerkin scheme. ``solve_poisson_dgm`` applies it to
the same 1-D Poisson BVP ``-V''(x) = f(x)``, ``V(0)=V(1)=0`` that ``chc.galerkin`` solves with a
variational-difference FEM (progonka), so the *neural* Galerkin can be checked against the analytic
and the *classical* one. The bridge from ``plans/01`` (Marchuk/Galerkin) to learning-based PDE
solvers.

The second half is the mean-field game that Poisson solve was only a stepping stone to. A backward
HJB for the value ``V`` and a forward Fokker-Planck for the density ``rho``, joined by the optimal
feedback ``alpha* = -(b/r) V_x`` and by the population mean ``m(t) = int x rho(x,t) dx``, are solved
as one coupled system by DGM (cf. arXiv 2405.13346) -- ``solve_mfg_dgm``. Both boundary conditions
are hard constraints rather than penalties: ``rho(0,.)`` is the prescribed Gaussian by construction
and ``V(T,.)`` is the prescribed terminal cost evaluated at the network's *own* terminal mean.

The falsifiable gate is that the linear-quadratic case has a CLOSED FORM: ``LQMeanFieldGame.solve``
returns it exactly, from a stationary Riccati root plus a 2x2 trace-free two-point boundary value
problem. That reduction also exposes an obstruction -- the fixed point degenerates where one
denominator vanishes, always on the anti-monotone branch ``c > 1 + r a^2/(q b^2)`` -- which is what
``lq_mean_field_certificate`` measures the DGM against. Derived in ``validation/lq_mean_field.mac``,
proved in ``proofs/lq_mean_field.v``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array


class ScalarMLP(eqx.Module):
    """A small tanh MLP ``x -> V(x)`` (scalar in, scalar out) for a 1-D field."""

    layers: list

    def __init__(self, width: int, key: Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(1, width, key=k1),
            eqx.nn.Linear(width, width, key=k2),
            eqx.nn.Linear(width, 1, key=k3),
        ]

    def __call__(self, x: Array) -> Array:
        h = jnp.atleast_1d(x)
        for lin in self.layers[:-1]:
            h = jax.nn.tanh(lin(h))
        return self.layers[-1](h)[0]


def solve_poisson_dgm(
    source: Callable[[Array], Array],
    width: int = 32,
    steps: int = 4000,
    n_collocation: int = 128,
    seed: int = 0,
) -> ScalarMLP:
    """Deep Galerkin solve of ``-V''(x) = source(x)`` on ``[0,1]``, ``V(0)=V(1)=0``.

    Minimises the mean-squared PDE residual at random collocation points plus the boundary term.
    """
    key = jax.random.key(seed)
    model = ScalarMLP(width, key)

    def second_derivative(m: ScalarMLP, x: Array) -> Array:
        return jax.grad(jax.grad(m.__call__))(x)

    def loss(m: ScalarMLP, xs: Array) -> Array:
        residual = jax.vmap(lambda x: second_derivative(m, x) + source(x))(xs)  # -V'' = f
        boundary = m(jnp.array(0.0)) ** 2 + m(jnp.array(1.0)) ** 2
        return jnp.mean(residual**2) + boundary

    optimizer = optax.adam(2e-3)
    state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(
        m: ScalarMLP, opt_state: optax.OptState, xs: Array
    ) -> tuple[ScalarMLP, optax.OptState]:
        grads = eqx.filter_grad(loss)(m, xs)
        updates, opt_state = optimizer.update(grads, opt_state)
        return eqx.apply_updates(m, updates), opt_state

    for _ in range(steps):
        key, sample_key = jax.random.split(key)
        xs = jax.random.uniform(sample_key, (n_collocation,))
        model, state = step(model, state, xs)
    return model


# --------------------------------------------------------------------------------------------
# The linear-quadratic mean-field game: the closed form that gates the neural solve.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LQMeanFieldGame:
    """A scalar linear-quadratic mean-field game with a closed-form solution.

    ``dX = (a X + b alpha) dt + sigma dW`` under
    ``E[int (q/2)(X - c m)^2 + (r/2) alpha^2 dt + (q_T/2)(X_T - c_T m_T)^2]``, with ``m(t)`` the
    population mean. The terminal weight ``q_T`` is pinned to the stationary Riccati root so the
    Riccati coefficient is constant and the remaining two-point boundary value problem has an
    exact matrix-exponential solution -- that is what makes this a *gate* rather than itself a
    numerical solve. The general case has the same structure with a time-varying transition.
    """

    a: float
    b: float
    q: float
    r: float
    coupling: float
    terminal_coupling: float
    sigma: float
    horizon: float
    mean_initial: float
    variance_initial: float

    @property
    def riccati_root(self) -> float:
        """The stationary root of ``(b^2/r) P^2 - 2 a P - q = 0``; also the terminal weight."""
        return (self.a + math.sqrt(self.a**2 + self.q * self.b**2 / self.r)) * self.r / self.b**2

    @property
    def closed_loop_rate(self) -> float:
        """``A = a - b^2 P/r = -sqrt(a^2 + q b^2/r)`` -- negative whatever the sign of ``a``."""
        return -math.sqrt(self.a**2 + self.q * self.b**2 / self.r)

    @property
    def branch_threshold(self) -> float:
        """The coupling ``c`` above which the transition matrix turns oscillatory."""
        return 1.0 + self.r * self.a**2 / (self.q * self.b**2)

    @property
    def lambda_squared(self) -> float:
        """``lam^2 = A^2 - q c b^2/r = a^2 + (q b^2/r)(1-c)``; negative on the oscillatory arm."""
        return self.a**2 + (self.q * self.b**2 / self.r) * (1.0 - self.coupling)

    @property
    def obstruction_gain(self) -> float:
        """``k = A + q_T c_T b^2/r``. The fixed point survives every horizon when ``k < lam``."""
        terminal = self.terminal_coupling * self.riccati_root * self.b**2 / self.r
        return self.closed_loop_rate + terminal

    def obstruction_horizon(self) -> float:
        """First horizon at which the mean-field fixed point degenerates, or ``inf`` if none.

        On the oscillatory branch this is finite for *every* terminal weight; on the real branch
        it is finite exactly when ``k > lam``. Proved in ``proofs/lq_mean_field.v``.
        """
        k = self.obstruction_gain
        lam_sq = self.lambda_squared
        if lam_sq < 0.0:
            w = math.sqrt(-lam_sq)
            return (math.pi / 2.0 - math.atan(k / w)) / w
        lam = math.sqrt(lam_sq)
        if lam_sq == 0.0:
            return 1.0 / k if k > 0.0 else math.inf
        if k <= lam:
            return math.inf
        return math.log((k + lam) / (k - lam)) / (2.0 * lam)

    def _transition_scalars(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(cosh(lam t), sinh(lam t)/lam)``, one path across both branches via a complex lam."""
        lam = np.sqrt(complex(self.lambda_squared))
        if abs(lam) < 1e-12:
            return np.ones_like(t), t
        ch = np.cosh(lam * t)
        sh = np.sinh(lam * t) / lam
        return np.real(ch), np.real(sh)

    def _transition(self, t: np.ndarray) -> np.ndarray:
        """``exp(M t)`` for trace-free ``M = [[A, -b^2/r], [qc, -A]]``; shape ``(len(t), 2, 2)``."""
        ch, sh = self._transition_scalars(t)
        big_a = self.closed_loop_rate
        phi = np.zeros((t.shape[0], 2, 2))
        phi[:, 0, 0] = ch + sh * big_a
        phi[:, 0, 1] = -sh * self.b**2 / self.r
        phi[:, 1, 0] = sh * self.q * self.coupling
        phi[:, 1, 1] = ch - sh * big_a
        return phi

    def fixed_point_denominator(self, horizon: float | None = None) -> float:
        """``cosh(lam T) - k sinh(lam T)/lam`` -- zero exactly at the obstruction horizon."""
        t = np.array([self.horizon if horizon is None else horizon])
        ch, sh = self._transition_scalars(t)
        return float(ch[0] - self.obstruction_gain * sh[0])

    def solve(self, n_time: int = 401) -> MeanFieldSolution:
        """The exact solution on a uniform time grid.

        ``Z`` is the one component without a closed form (it is a quadrature of ``S^2`` and
        ``m^2``); it is integrated by the trapezoid rule on the same grid and affects only the
        level of ``V``, never the control.
        """
        times = np.linspace(0.0, self.horizon, n_time)
        phi = self._transition(times)
        p_bar = self.riccati_root
        e_t = phi[-1]
        denominator = e_t[1, 1] + p_bar * self.terminal_coupling * e_t[0, 1]
        numerator = e_t[1, 0] + p_bar * self.terminal_coupling * e_t[0, 0]
        s_initial = -self.mean_initial * numerator / denominator
        y0 = np.array([self.mean_initial, s_initial])
        traj = phi @ y0
        mean, value_s = traj[:, 0], traj[:, 1]

        big_a = self.closed_loop_rate
        decay = np.exp(2.0 * big_a * times)
        variance = decay * self.variance_initial + self.sigma**2 * (decay - 1.0) / (2.0 * big_a)

        dz = (
            self.b**2 / (2.0 * self.r) * value_s**2
            - self.sigma**2 / 2.0 * p_bar
            - self.q / 2.0 * self.coupling**2 * mean**2
        )
        z_terminal = 0.5 * p_bar * self.terminal_coupling**2 * mean[-1] ** 2
        tail = np.concatenate([[0.0], np.cumsum(np.diff(times) * 0.5 * (dz[:-1] + dz[1:]))])
        value_z = z_terminal - (tail[-1] - tail)
        return MeanFieldSolution(
            game=self,
            times=times,
            mean=mean,
            variance=variance,
            value_s=value_s,
            value_z=value_z,
            denominator=float(denominator),
        )


@dataclass(frozen=True)
class MeanFieldSolution:
    """The exact LQ mean-field equilibrium: ``V = P x^2/2 + S x + Z`` against a Gaussian ``rho``."""

    game: LQMeanFieldGame
    times: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    value_s: np.ndarray
    value_z: np.ndarray
    denominator: float

    def _interp(self, series: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.interp(t, self.times, series)

    def value(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        p_bar = self.game.riccati_root
        linear = self._interp(self.value_s, t) * x
        return 0.5 * p_bar * x**2 + linear + self._interp(self.value_z, t)

    def value_gradient(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """``V_x = P x + S`` -- exact, no quadrature, and the only part the control depends on."""
        return self.game.riccati_root * x + self._interp(self.value_s, t)

    def control(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        return -(self.game.b / self.game.r) * self.value_gradient(t, x)

    def density(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        m, v = self._interp(self.mean, t), self._interp(self.variance, t)
        return np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2.0 * np.pi * v)

    def hjb_residual(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Residual of the backward HJB at the closed form. Zero up to the ``Z`` quadrature."""
        g, p_bar = self.game, self.game.riccati_root
        big_a = g.closed_loop_rate
        s = self._interp(self.value_s, t)
        m = self._interp(self.mean, t)
        ds = -big_a * s + g.q * g.coupling * m
        dz = (
            g.b**2 / (2.0 * g.r) * s**2
            - g.sigma**2 / 2.0 * p_bar
            - g.q / 2.0 * g.coupling**2 * m**2
        )
        v_x = p_bar * x + s
        return (
            ds * x
            + dz
            + g.a * x * v_x
            - g.b**2 / (2.0 * g.r) * v_x**2
            + g.sigma**2 / 2.0 * p_bar
            + g.q / 2.0 * (x - g.coupling * m) ** 2
        )

    def fokker_planck_residual(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Residual of the forward Fokker-Planck at the Gaussian, in log form (divided by rho)."""
        g, p_bar = self.game, self.game.riccati_root
        big_a = g.closed_loop_rate
        m, v = self._interp(self.mean, t), self._interp(self.variance, t)
        s = self._interp(self.value_s, t)
        dm = big_a * m - g.b**2 / g.r * s
        dv = 2.0 * big_a * v + g.sigma**2
        z = x - m
        # d/dt log rho for a Gaussian, then the transport and diffusion terms.
        log_rho_t = -0.5 * dv / v + z * dm / v + 0.5 * z**2 * dv / v**2
        log_rho_x = -z / v
        drift = g.a * x - g.b**2 / g.r * (p_bar * x + s)
        drift_x = g.a - g.b**2 / g.r * p_bar
        return (
            log_rho_t + drift_x + drift * log_rho_x - g.sigma**2 / 2.0 * (-1.0 / v + log_rho_x**2)
        )


# --------------------------------------------------------------------------------------------
# The coupled Deep Galerkin solve.
# --------------------------------------------------------------------------------------------


class FieldMLP(eqx.Module):
    """A small tanh MLP ``(t, x) -> R`` for a space-time field."""

    layers: list

    def __init__(self, width: int, key: Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(2, width, key=k1),
            eqx.nn.Linear(width, width, key=k2),
            eqx.nn.Linear(width, 1, key=k3),
        ]

    def __call__(self, t: Array, x: Array) -> Array:
        h = jnp.stack([t, x])
        for lin in self.layers[:-1]:
            h = jax.nn.tanh(lin(h))
        return self.layers[-1](h)[0]


class MeanFieldDGM(eqx.Module):
    """A Deep Galerkin solution of the coupled HJB / Fokker-Planck system.

    Both boundary conditions are structural rather than penalised: the density carries the
    prescribed initial Gaussian as an exact factor, and the value function carries the prescribed
    terminal cost evaluated at the network's *own* terminal mean -- which is where the mean-field
    coupling enters the value side. What is left to minimise is only the two interior residuals.
    """

    value_core: FieldMLP
    log_density_core: FieldMLP
    game: LQMeanFieldGame = eqx.field(static=True)
    half_width: float = eqx.field(static=True)
    n_quadrature: int = eqx.field(static=True)

    @property
    def quadrature(self) -> Array:
        """The integration nodes, derived rather than stored.

        An array field of an ``eqx.Module`` is an inexact-array leaf, so
        ``eqx.filter(model, eqx.is_inexact_array)`` hands it to the optimiser along with the
        weights -- and an optimiser that moves the quadrature grid corrupts every mean and mass
        computed on it, silently and without touching the density.
        """
        return jnp.linspace(-self.half_width, self.half_width, self.n_quadrature)

    def log_density(self, t: Array, x: Array) -> Array:
        g = self.game
        initial = -0.5 * (x - g.mean_initial) ** 2 / g.variance_initial - 0.5 * jnp.log(
            2.0 * jnp.pi * g.variance_initial
        )
        scaled_t = t / g.horizon
        return initial + scaled_t * self.log_density_core(scaled_t, x / self.half_width)

    def density(self, t: Array, x: Array) -> Array:
        return jnp.exp(self.log_density(t, x))

    def mean(self, t: Array) -> Array:
        """``int x rho / int rho`` on the quadrature grid -- normalised, so mass drift cannot
        contaminate the coupling."""
        weights = jax.vmap(lambda node: self.density(t, node))(self.quadrature)
        mass = jnp.trapezoid(weights, self.quadrature)
        return jnp.trapezoid(weights * self.quadrature, self.quadrature) / mass

    def mass(self, t: Array) -> Array:
        weights = jax.vmap(lambda node: self.density(t, node))(self.quadrature)
        return jnp.trapezoid(weights, self.quadrature)

    def value_at(self, t: Array, x: Array, mean_terminal: Array) -> Array:
        g = self.game
        terminal = 0.5 * g.riccati_root * (x - g.terminal_coupling * mean_terminal) ** 2
        scaled_t = t / g.horizon
        return terminal + (1.0 - scaled_t) * self.value_core(scaled_t, x / self.half_width)

    def value(self, t: Array, x: Array) -> Array:
        return self.value_at(t, x, self.mean(jnp.asarray(self.game.horizon)))

    def control(self, t: Array, x: Array) -> Array:
        gradient = jax.grad(lambda xx: self.value(t, xx))(x)
        return -(self.game.b / self.game.r) * gradient


def _mfg_residuals(
    model: MeanFieldDGM, t: Array, x: Array, mean_t: Array, mean_terminal: Array
) -> tuple[Array, Array]:
    """The HJB and Fokker-Planck residuals at one space-time point.

    The Fokker-Planck half is evaluated in log form and then multiplied back by the density: the
    log form keeps the derivatives well scaled where the density underflows, and the factor puts
    the weight where the mass actually is instead of spreading it over empty tails.
    """
    g = model.game

    def value(tt: Array, xx: Array) -> Array:
        return model.value_at(tt, xx, mean_terminal)

    v_t = jax.grad(value, 0)(t, x)
    v_x = jax.grad(value, 1)(t, x)
    v_xx = jax.grad(jax.grad(value, 1), 1)(t, x)
    hjb = (
        v_t
        + g.a * x * v_x
        - g.b**2 / (2.0 * g.r) * v_x**2
        + g.sigma**2 / 2.0 * v_xx
        + g.q / 2.0 * (x - g.coupling * mean_t) ** 2
    )

    log_rho = model.log_density
    g_t = jax.grad(log_rho, 0)(t, x)
    g_x = jax.grad(log_rho, 1)(t, x)
    g_xx = jax.grad(jax.grad(log_rho, 1), 1)(t, x)
    drift = g.a * x - g.b**2 / g.r * v_x
    drift_x = g.a - g.b**2 / g.r * v_xx
    log_residual = g_t + drift_x + drift * g_x - g.sigma**2 / 2.0 * (g_xx + g_x**2)
    reference = 0.5 * jnp.log(2.0 * jnp.pi * g.variance_initial)
    return hjb, jnp.exp(model.log_density(t, x) + reference) * log_residual


def solve_mfg_dgm(
    game: LQMeanFieldGame,
    width: int = 32,
    steps: int = 3000,
    n_time: int = 16,
    n_space: int = 16,
    n_quadrature: int = 96,
    learning_rate: float = 2e-3,
    fp_weight: float = 1.0,
    mass_weight: float = 1.0,
    half_width: float | None = None,
    seed: int = 0,
) -> MeanFieldDGM:
    """Solve the coupled forward-backward mean-field system by Deep Galerkin.

    By default the spatial box comes from problem data alone -- the initial mean, the initial
    variance and the stationary variance ``sigma^2/(2|A|)`` -- never from the closed-form
    solution, so the comparison in ``lq_mean_field_certificate`` stays honest. Pass ``half_width``
    to widen it when the population is known to travel further than that; the certificate uses it
    near the obstruction horizon deliberately, to hand the solver a domain it could not have
    guessed and show that this still does not save it.
    """
    if half_width is None:
        stationary_variance = game.sigma**2 / (2.0 * abs(game.closed_loop_rate))
        variance_ceiling = max(game.variance_initial, stationary_variance)
        half_width = abs(game.mean_initial) + 5.0 * math.sqrt(variance_ceiling) + 1.0
    key = jax.random.key(seed)
    value_key, density_key = jax.random.split(key)
    model = MeanFieldDGM(
        value_core=FieldMLP(width, value_key),
        log_density_core=FieldMLP(width, density_key),
        game=game,
        half_width=half_width,
        n_quadrature=n_quadrature,
    )

    def loss(m: MeanFieldDGM, ts: Array, xs: Array) -> Array:
        mean_t = jax.vmap(m.mean)(ts)
        mean_terminal = m.mean(jnp.asarray(game.horizon))
        rows = jax.vmap(
            lambda t, row, mt: jax.vmap(lambda x: _mfg_residuals(m, t, x, mt, mean_terminal))(row)
        )
        hjb, fokker_planck = rows(ts, xs, mean_t)
        mass_error = jax.vmap(m.mass)(ts) - 1.0
        return (
            jnp.mean(hjb**2)
            + fp_weight * jnp.mean(fokker_planck**2)
            + mass_weight * jnp.mean(mass_error**2)
        )

    optimizer = optax.adam(learning_rate)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(
        m: MeanFieldDGM, opt_state: optax.OptState, ts: Array, xs: Array
    ) -> tuple[MeanFieldDGM, optax.OptState]:
        grads = eqx.filter_grad(loss)(m, ts, xs)
        updates, opt_state = optimizer.update(grads, opt_state)
        return eqx.apply_updates(m, updates), opt_state

    for _ in range(steps):
        key, time_key, space_key = jax.random.split(key, 3)
        ts = jax.random.uniform(time_key, (n_time,), maxval=game.horizon)
        xs = jax.random.uniform(space_key, (n_time, n_space), minval=-half_width, maxval=half_width)
        model, state = step(model, state, ts, xs)
    return model


# --------------------------------------------------------------------------------------------
# The certificate: the closed form is the gate, and the obstruction is what it catches.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MeanFieldCurve:
    """Evidence for Result 49 -- the coupled solve, and what its own residual fails to see."""

    closed_form_hjb_residual: float
    closed_form_fp_residual: float
    bisected_obstruction_horizon: float
    predicted_obstruction_horizon: float
    horizon_relative_error: float
    pole_exponent: float
    branch_threshold: float
    safe_lambda_squared: float
    oscillatory_lambda_squared: float
    safe_obstruction_gain: float
    safe_minimum_denominator: float
    dgm_control_error: float
    dgm_mean_error: float
    dgm_density_error: float
    dgm_initial_mean_error: float
    far_control_error: float
    near_control_error: float
    far_residual: float
    near_residual: float
    residual_blindness: float
    ok: bool


def _mass_band(solution: MeanFieldSolution, n_time: int, n_space: int) -> tuple[np.ndarray, ...]:
    """A space-time grid covering +-3 standard deviations around the true mean.

    Errors are read where the population actually is: a mean-field control that is wrong five
    standard deviations out is wrong about nobody.
    """
    times = np.linspace(0.0, solution.game.horizon, n_time)
    spread = np.sqrt(np.interp(times, solution.times, solution.variance))
    centre = np.interp(times, solution.times, solution.mean)
    offsets = np.linspace(-3.0, 3.0, n_space)
    grid_x = centre[:, None] + spread[:, None] * offsets[None, :]
    return times, np.repeat(times[:, None], n_space, axis=1), grid_x


def _dgm_errors(
    game: LQMeanFieldGame, model: MeanFieldDGM, n_time: int = 21, n_space: int = 15
) -> tuple[float, float, float, float]:
    """Relative control, mean, density and initial-mean errors of a trained model."""
    reference = game.solve()
    times, grid_t, grid_x = _mass_band(reference, n_time, n_space)
    mean_terminal = model.mean(jnp.asarray(game.horizon))
    gradient = np.asarray(
        jax.vmap(
            jax.vmap(lambda t, x: jax.grad(lambda xx: model.value_at(t, xx, mean_terminal))(x))
        )(jnp.asarray(grid_t), jnp.asarray(grid_x))
    )
    exact_gradient = reference.value_gradient(grid_t.ravel(), grid_x.ravel()).reshape(grid_t.shape)
    density = np.asarray(
        jax.vmap(jax.vmap(model.density))(jnp.asarray(grid_t), jnp.asarray(grid_x))
    )
    exact_density = reference.density(grid_t.ravel(), grid_x.ravel()).reshape(grid_t.shape)
    fitted_mean = np.asarray(jax.vmap(model.mean)(jnp.asarray(times)))
    exact_mean = np.interp(times, reference.times, reference.mean)
    return (
        float(np.abs(gradient - exact_gradient).max() / np.abs(exact_gradient).max()),
        float(np.abs(fitted_mean - exact_mean).max() / np.abs(exact_mean).max()),
        float(np.abs(density - exact_density).max() / np.abs(exact_density).max()),
        float(abs(fitted_mean[0] - game.mean_initial)),
    )


def _dgm_residual(game: LQMeanFieldGame, model: MeanFieldDGM, n_time: int = 21) -> float:
    """RMS interior residual, normalised by the running-cost scale so horizons are comparable."""
    times, grid_t, grid_x = _mass_band(game.solve(), n_time, 15)
    mean_terminal = model.mean(jnp.asarray(game.horizon))
    mean_t = jax.vmap(model.mean)(jnp.asarray(times))
    hjb, fokker_planck = jax.vmap(
        jax.vmap(
            lambda t, x, m: _mfg_residuals(model, t, x, m, mean_terminal), in_axes=(0, 0, None)
        )
    )(jnp.asarray(grid_t), jnp.asarray(grid_x), mean_t)
    deviation = jnp.asarray(grid_x) - game.coupling * mean_t[:, None]
    scale = float(jnp.max(game.q / 2.0 * deviation**2))
    return float(jnp.sqrt(jnp.mean(hjb**2) + jnp.mean(fokker_planck**2))) / scale


def lq_mean_field_certificate(steps: int = 2500, seed: int = 0) -> MeanFieldCurve:
    """Measure the coupled DGM against the exact LQ mean-field equilibrium.

    Three arms. The first checks that the gate itself is exact -- the closed form annihilates both
    PDE residuals, and the obstruction horizon bisected from the denominator matches the formula.
    The second checks that the neural solve reproduces it on a monotone instance. The third is the
    one that can fail loudly: on the anti-monotone branch, as the horizon approaches the
    obstruction, the DGM's error grows while its own residual *shrinks*, so a residual-based
    stopping rule reports success exactly where the answer is worst.
    """
    safe = LQMeanFieldGame(
        a=-0.5,
        b=1.0,
        q=1.0,
        r=1.0,
        coupling=0.5,
        terminal_coupling=0.5,
        sigma=0.7,
        horizon=1.0,
        mean_initial=1.0,
        variance_initial=0.25,
    )
    oscillatory = replace(safe, coupling=3.0, terminal_coupling=3.0)

    # --- arm 1: the gate is exact ---------------------------------------------------------
    reference = safe.solve()
    rng = np.random.default_rng(seed)
    probe_t = rng.uniform(0.0, safe.horizon, 3000)
    probe_x = rng.uniform(-5.0, 5.0, 3000)
    hjb_residual = float(np.abs(reference.hjb_residual(probe_t, probe_x)).max())
    fp_residual = float(np.abs(reference.fokker_planck_residual(probe_t, probe_x)).max())

    predicted = oscillatory.obstruction_horizon()
    low, high = 0.1, 1.2
    for _ in range(90):
        middle = 0.5 * (low + high)
        if (
            oscillatory.fixed_point_denominator(low) * oscillatory.fixed_point_denominator(middle)
            <= 0.0
        ):
            high = middle
        else:
            low = middle
    bisected = 0.5 * (low + high)

    gaps = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
    linear_coefficients = np.array(
        [
            abs(replace(oscillatory, horizon=predicted - gap).solve(n_time=201).value_s[0])
            for gap in gaps
        ]
    )
    pole_exponent = float(np.polyfit(np.log(gaps), np.log(linear_coefficients), 1)[0])

    long_horizons = np.linspace(0.1, 40.0, 400)
    safe_minimum = min(safe.fixed_point_denominator(t) for t in long_horizons)

    # --- arm 2: the neural solve reproduces the equilibrium ------------------------------
    trained = solve_mfg_dgm(safe, steps=steps, seed=seed)
    control_error, mean_error, density_error, initial_mean_error = _dgm_errors(safe, trained)

    # --- arm 3: the residual stops tracking the error near the obstruction ----------------
    # One box for both horizons, wide enough for the near one, so the only difference is T.
    box = 10.0
    far, near = replace(oscillatory, horizon=0.35), replace(oscillatory, horizon=0.76)
    far_model = solve_mfg_dgm(far, steps=steps, half_width=box, seed=seed)
    near_model = solve_mfg_dgm(near, steps=steps, half_width=box, seed=seed)
    far_error = _dgm_errors(far, far_model)[0]
    near_error = _dgm_errors(near, near_model)[0]
    far_residual = _dgm_residual(far, far_model)
    near_residual = _dgm_residual(near, near_model)
    blindness = (near_error / far_error) / (near_residual / far_residual)

    ok = bool(
        hjb_residual < 1e-10
        and fp_residual < 1e-10
        and abs(bisected - predicted) / predicted < 1e-9
        and -1.05 < pole_exponent < -0.90
        and safe.lambda_squared > 0.0
        and oscillatory.lambda_squared < 0.0
        and safe.coupling < safe.branch_threshold < oscillatory.coupling
        and safe.obstruction_gain <= 0.0
        and math.isinf(safe.obstruction_horizon())
        and safe_minimum >= 1.0
        and control_error < 0.02
        and mean_error < 0.03
        and density_error < 0.05
        and initial_mean_error < 1e-10
        and near_error > 5.0 * far_error
        and near_residual < far_residual
        and blindness > 5.0
    )
    return MeanFieldCurve(
        closed_form_hjb_residual=hjb_residual,
        closed_form_fp_residual=fp_residual,
        bisected_obstruction_horizon=bisected,
        predicted_obstruction_horizon=predicted,
        horizon_relative_error=abs(bisected - predicted) / predicted,
        pole_exponent=pole_exponent,
        branch_threshold=safe.branch_threshold,
        safe_lambda_squared=safe.lambda_squared,
        oscillatory_lambda_squared=oscillatory.lambda_squared,
        safe_obstruction_gain=safe.obstruction_gain,
        safe_minimum_denominator=safe_minimum,
        dgm_control_error=control_error,
        dgm_mean_error=mean_error,
        dgm_density_error=density_error,
        dgm_initial_mean_error=initial_mean_error,
        far_control_error=far_error,
        near_control_error=near_error,
        far_residual=far_residual,
        near_residual=near_residual,
        residual_blindness=blindness,
        ok=ok,
    )
