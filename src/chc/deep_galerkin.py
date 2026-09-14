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
from functools import partial
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

PopulationNoise = Literal["independent", "common"]
"""Whether each agent gets its own Brownian path or the whole population shares one."""

_SHOOTING_STEPS = 40
_SHOOTING_TOLERANCE = 1e-6  # terminal-row miss above which the congested solve is a failure


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

    def finite_population_rms(self, n_agents: int) -> np.ndarray:
        """``sqrt(v(t)/N)`` on :attr:`times`: what ``N`` agents cost against the density.

        Exact at every ``t`` and every ``N``, not asymptotic. A mean-field feedback is a function
        of an agent's *own* state and of deterministic coefficients, so the closed-loop agents are
        independent Ornstein-Uhlenbeck processes and ``m_N`` is a sample mean of ``N`` i.i.d.
        draws; the ``1/sqrt(N)`` is then an identity and the constant is the closed-loop variance.
        Derived in ``validation/finite_population_gap.mac``, measured by
        :func:`finite_population_gap_certificate`.
        """
        if n_agents < 1:
            raise ValueError(f"a population needs at least one agent; got {n_agents}")
        return np.sqrt(self.variance / n_agents)

    def simulate_population(
        self,
        n_agents: int,
        *,
        replicates: int = 1,
        seed: int = 0,
        n_step: int = 400,
        mean_feedback: float = 0.0,
        noise: PopulationNoise = "independent",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run ``n_agents`` under this feedback; return ``(times, empirical mean paths)``.

        The plant stepped here is the *primitive* one, ``dX = (a X + b alpha) dt + sigma dW`` by
        Euler-Maruyama. Only the control is taken from the solution, so the paths are an
        independent check on :meth:`finite_population_rms` rather than a restatement of it; the
        Euler bias on the variance is ``|A| dt`` relative, two orders below the Monte-Carlo error
        at the default step.

        ``mean_feedback`` adds ``(kappa/b) m_N`` to every agent's control. That is the difference
        between an N-player equilibrium and a mean-field control applied to N players: it couples
        the agents, moves the closed-loop rate to ``A + kappa`` and biases the mean away from
        ``m``. ``noise="common"`` drives every agent with one Brownian path instead of its own,
        which is the independence assumption the identity rests on -- it is exposed so it can be
        violated on purpose.
        """
        if n_agents < 1:
            raise ValueError(f"a population needs at least one agent; got {n_agents}")
        if replicates < 1:
            raise ValueError(f"a Monte-Carlo sweep needs at least one replicate; got {replicates}")
        game = self.game
        dt = game.horizon / n_step
        times = np.linspace(0.0, game.horizon, n_step + 1)
        drive = self._interp(self.value_s, times)
        rng = np.random.default_rng(seed)
        x = game.mean_initial + math.sqrt(game.variance_initial) * rng.standard_normal(
            (replicates, n_agents)
        )
        # One shared column under a common shock, one column per agent otherwise.
        shock = (replicates, 1) if noise == "common" else (replicates, n_agents)
        paths = np.empty((replicates, n_step + 1))
        paths[:, 0] = x.mean(axis=1)
        root_dt = math.sqrt(dt)
        for step in range(n_step):
            control = -(game.b / game.r) * (game.riccati_root * x + drive[step])
            if mean_feedback:
                control = control + (mean_feedback / game.b) * x.mean(axis=1, keepdims=True)
            x = (
                x
                + (game.a * x + game.b * control) * dt
                + game.sigma * root_dt * rng.standard_normal(shock)
            )
            paths[:, step + 1] = x.mean(axis=1)
        return times, paths

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
    far_value_error: float  # |S_hat(0) - S(0)|, the quantity the dual-weighted estimator targets
    near_value_error: float
    far_dual_weighted: float  # the estimator, computed WITHOUT the closed form
    near_dual_weighted: float
    dual_weighted_accuracy: float  # worst relative discrepancy against the true error
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


def _model_traces(
    game: LQMeanFieldGame, model: MeanFieldDGM, n_time: int
) -> tuple[np.ndarray, ...]:
    """``(t, S, m, S', m', A)`` read off the trained model along the axis ``x = 0``.

    ``V = P x^2/2 + S x + Z`` makes ``S(t) = V_x(t, 0)`` and ``P(t) = V_xx(t, 0)``, so the whole
    reduced state and its own closed-loop rate ``A = a - b^2 P/r`` come from the network by
    differentiation -- no reference solution anywhere.
    """
    mean_terminal = model.mean(jnp.asarray(game.horizon))

    def value_slope(t: Array) -> Array:
        return jax.grad(lambda x: model.value_at(t, x, mean_terminal))(jnp.asarray(0.0))

    def value_curvature(t: Array) -> Array:
        return jax.grad(jax.grad(lambda x: model.value_at(t, x, mean_terminal)))(jnp.asarray(0.0))

    times = jnp.linspace(0.0, game.horizon, n_time)
    curvature = np.asarray(jax.vmap(value_curvature)(times))
    return (
        np.asarray(times),
        np.asarray(jax.vmap(value_slope)(times)),
        np.asarray(jax.vmap(model.mean)(times)),
        np.asarray(jax.vmap(jax.grad(value_slope))(times)),
        np.asarray(jax.vmap(jax.grad(model.mean))(times)),
        game.a - game.b**2 * curvature / game.r,
    )


def dual_weighted_error_estimate(
    game: LQMeanFieldGame, model: MeanFieldDGM, n_time: int = 401
) -> float:
    """An a-posteriori estimate of ``|S_hat(0) - S(0)|`` from the model alone -- Result 55.

    Result 49 measured a residual that FALLS as the answer degrades near the mean-field
    obstruction, so a residual-based stopping rule reports its cleanest convergence exactly where
    the solution is worst. The remedy is not a bigger residual budget but the right functional.
    Because the reduced fixed point is AFFINE, the error is an exact quotient
    (``validation/mean_field_dwr.mac`` STEP 2, ``proofs/mean_field_dwr.v``)::

        S_hat(0) - S(0) = (eps - int_0^T z(s).g(s) ds) / den(T),   z(s) = Phi(T-s)^T v / den(T)

    with ``g = y' - M y`` the reduced defect of the model's own ``y = (m, S)``, ``eps = v.y(T)``
    the terminal-consistency defect (structurally zero here), and ``z`` the exact ADJOINT
    solution. Everything is read from the trained network and the cost parameters: the transition
    matrix is integrated from the model's OWN closed-loop rate ``A = a - b^2 V_xx/r``, never from
    the reference solution, so this is an estimator and not an oracle comparison.

    Two facts make it work where the raw residual does not. The dual weight carries ``1/den(T)``
    and nothing else that can blow up, so it diverges exactly at the obstruction; and the reduced
    residual is homogeneous of degree 1 in ``(m, S)``, so a bounded approximator facing a
    diverging solution keeps a small residual by construction. Conditioning the residual by
    ``1/|den|`` alone is NOT enough -- measured rank correlation with the error ``0.41`` against
    ``-0.67`` for the raw residual and ``1.00`` for this estimator; what matters is the projection
    onto the adjoint mode, not the scalar rescaling.

    Scope: the estimate is exact for this affine family up to quadrature and autodiff error
    (measured within ``6%`` at a horizon where the error is ``72`` and the raw residual is at its
    smallest). For a non-quadratic problem the same construction is the standard dual-weighted
    residual and is first-order, not exact.
    """
    times, slope, mean, slope_rate, mean_rate, rate = _model_traces(game, model, n_time)
    b2r = game.b**2 / game.r
    coupling = game.q * game.coupling
    terminal_weight = game.riccati_root * game.terminal_coupling

    def reduced(index: int) -> np.ndarray:
        return np.array([[rate[index], -b2r], [coupling, -rate[index]]])

    transition = np.empty((times.size, 2, 2))
    transition[0] = np.eye(2)
    for i in range(times.size - 1):
        step = times[i + 1] - times[i]
        k1 = reduced(i) @ transition[i]
        k2 = reduced(i) @ (transition[i] + 0.5 * step * k1)
        k3 = reduced(i) @ (transition[i] + 0.5 * step * k2)
        k4 = reduced(i + 1) @ (transition[i] + step * k3)
        transition[i + 1] = transition[i] + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    functional = np.array([terminal_weight, 1.0])
    denominator = float(functional @ transition[-1] @ np.array([0.0, 1.0]))
    if denominator == 0.0:
        return math.inf  # the obstruction itself: no fixed point, hence no finite error
    weights = np.einsum("i,ij,sjk->sk", functional, transition[-1], np.linalg.inv(transition))
    defect_mean = mean_rate - (rate * mean - b2r * slope)
    defect_slope = slope_rate - (coupling * mean - rate * slope)
    interior = float(
        np.trapezoid(weights[:, 0] * defect_mean + weights[:, 1] * defect_slope, times)
    )
    terminal = float(terminal_weight * mean[-1] + slope[-1])
    return abs(terminal - interior) / abs(denominator)


def _value_slope_error(game: LQMeanFieldGame, model: MeanFieldDGM) -> float:
    """``|S_hat(0) - S(0)|`` against the closed form -- the truth the estimator is scored on."""
    mean_terminal = model.mean(jnp.asarray(game.horizon))
    fitted = float(
        jax.grad(lambda x: model.value_at(jnp.asarray(0.0), x, mean_terminal))(jnp.asarray(0.0))
    )
    return abs(fitted - float(game.solve().value_s[0]))


_MONOTONE_GAME = LQMeanFieldGame(
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
"""The instance both certificates measure, so the obstruction and the gap describe one system."""


def _ou_variance(game: LQMeanFieldGame, rate: float, t: float) -> float:
    """``v0 e^{2 rate t} + sigma^2 (e^{2 rate t} - 1)/(2 rate)``, the closed-loop variance."""
    decay = math.exp(2.0 * rate * t)
    return game.variance_initial * decay + game.sigma**2 * (decay - 1.0) / (2.0 * rate)


def lq_mean_field_certificate(steps: int = 2500, seed: int = 0) -> MeanFieldCurve:
    """Measure the coupled DGM against the exact LQ mean-field equilibrium.

    Three arms. The first checks that the gate itself is exact -- the closed form annihilates both
    PDE residuals, and the obstruction horizon bisected from the denominator matches the formula.
    The second checks that the neural solve reproduces it on a monotone instance. The third is the
    one that can fail loudly: on the anti-monotone branch, as the horizon approaches the
    obstruction, the DGM's error grows while its own residual *shrinks*, so a residual-based
    stopping rule reports success exactly where the answer is worst.
    """
    safe = _MONOTONE_GAME
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

    # --- arm 4: the dual-weighted estimator sees what the raw residual cannot -------------
    far_value_error = _value_slope_error(far, far_model)
    near_value_error = _value_slope_error(near, near_model)
    far_dual = dual_weighted_error_estimate(far, far_model)
    near_dual = dual_weighted_error_estimate(near, near_model)
    dual_accuracy = max(
        abs(far_dual / far_value_error - 1.0), abs(near_dual / near_value_error - 1.0)
    )

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
        and near_value_error > 5.0 * far_value_error
        and near_dual > far_dual  # the estimator ORDERS the two horizons the residual inverts
        and dual_accuracy < 0.1
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
        far_value_error=far_value_error,
        near_value_error=near_value_error,
        far_dual_weighted=far_dual,
        near_dual_weighted=near_dual,
        dual_weighted_accuracy=dual_accuracy,
        ok=ok,
    )


# --------------------------------------------------------------------------------------------
# The finite-population gap: what N agents cost against the Fokker-Planck density.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FinitePopulationGap:
    """Evidence that the mean-field density is *priced* at finite ``N``, not merely approached."""

    sizes: tuple[int, ...]
    measured_rms: tuple[float, ...]
    closed_form_rms: tuple[float, ...]
    worst_relative_error: float
    measured_exponent: float
    exponent_standard_error: float
    common_shock_exponent: float  # the independence assumption, violated on purpose
    coupled_variance_ratio: float  # measured / the (A + kappa) closed form -- should be 1
    coupled_wrong_rate_ratio: float  # measured / the (A) closed form -- must NOT be 1
    predicted_bias_constant: float  # c * int_0^T exp(A(T-s)) m(s) ds
    measured_bias_constant: float  # N * (coupled mean - mean-field mean), common random numbers
    bias_to_fluctuation: float  # at the largest N: the O(1/N) bias over the O(1/sqrt N) gap
    ok: bool


def finite_population_gap_certificate(
    replicates: int = 400, seed: int = 0, n_step: int = 400
) -> FinitePopulationGap:
    """Price Result 49's modelling gap: the Fokker-Planck density is the ``N -> infinity`` limit.

    Four arms, in increasing order of what they can tell you.

    The first measures the empirical-mean gap of a simulated population against the closed form
    ``sqrt(v(t)/N)``, and fits the exponent. The second is the falsification: drive every agent
    with one shared Brownian path and the exponent must collapse to zero, because independence --
    not large ``N`` -- is what makes the identity exact.

    The third and fourth ask what an *N-player* equilibrium would change. A mean-feedback gain
    ``kappa`` moves the closed-loop rate to ``A + kappa``: arm three checks the variance against
    the moved constant and against the unmoved one, so the second number is there to be wrong.
    Arm four takes ``kappa = c/N``, which is the scaling a real Nash equilibrium has, and measures
    the resulting bias under common random numbers -- it is ``O(1/N)``, so it never catches up
    with the ``O(1/sqrt N)`` fluctuation, which is why the exponent is the part that transfers.

    Derived in ``validation/finite_population_gap.mac``; the exponent's Monte-Carlo standard error
    is derived there too, and reported here rather than assumed.
    """
    game = _MONOTONE_GAME
    solution = game.solve()
    # Both grids end at the horizon, so the last column of a simulation and the last entry of the
    # closed form are the same instant and no interpolation is needed to compare them.
    end = -1
    terminal_mean = float(solution.mean[end])
    sizes = (8, 16, 32, 64, 128, 256, 512)
    log_sizes = np.log(np.asarray(sizes, dtype=float))

    # --- arm 1: the gap against the closed form, and the exponent --------------------------
    measured, predicted = [], []
    for n_agents in sizes:
        _, paths = solution.simulate_population(
            n_agents, replicates=replicates, seed=seed + n_agents, n_step=n_step
        )
        measured.append(float(np.sqrt(((paths[:, end] - terminal_mean) ** 2).mean())))
        predicted.append(float(solution.finite_population_rms(n_agents)[end]))
    exponent = float(np.polyfit(log_sizes, np.log(measured), 1)[0])
    worst = max(abs(m / p - 1.0) for m, p in zip(measured, predicted, strict=True))
    # Two-point slope error at this R and this lever arm; the 7-point fit is at least this good.
    slope_se = float(
        math.sqrt(2.0 / (replicates - 1)) / 2.0 * math.sqrt(2.0) / math.log(sizes[-1] / sizes[0])
    )

    # --- arm 2: remove the independence and the identity goes with it ----------------------
    shared = []
    for n_agents in sizes:
        _, paths = solution.simulate_population(
            n_agents, replicates=replicates, seed=seed + n_agents, n_step=n_step, noise="common"
        )
        shared.append(float(np.sqrt(((paths[:, end] - terminal_mean) ** 2).mean())))
    common_exponent = float(np.polyfit(log_sizes, np.log(shared), 1)[0])

    # --- arm 3: a fixed coupling moves the constant, not the exponent ----------------------
    kappa, coupled_size = 0.6, 512
    rate = game.closed_loop_rate + kappa
    _, coupled_paths = solution.simulate_population(
        coupled_size,
        replicates=replicates,
        seed=seed + coupled_size,
        n_step=n_step,
        mean_feedback=kappa,
    )
    coupled_variance = float(coupled_paths[:, end].var(ddof=1))
    moved = _ou_variance(game, rate, game.horizon)
    unmoved = _ou_variance(game, game.closed_loop_rate, game.horizon)

    # --- arm 4: at kappa = c/N the bias is O(1/N) and loses to the fluctuation --------------
    gain, bias_size = 6.0, 512
    weight = np.exp(game.closed_loop_rate * (game.horizon - solution.times)) * solution.mean
    predicted_bias = gain * float(np.trapezoid(weight, solution.times))
    # Common random numbers: the same seed and the same draw sequence in both runs, so the
    # O(1/sqrt N) fluctuation cancels in the difference and only the O(1/N) drift survives.
    _, free = solution.simulate_population(
        bias_size, replicates=replicates, seed=seed + bias_size, n_step=n_step
    )
    _, tied = solution.simulate_population(
        bias_size,
        replicates=replicates,
        seed=seed + bias_size,
        n_step=n_step,
        mean_feedback=gain / bias_size,
    )
    measured_bias = float(bias_size * (tied[:, end] - free[:, end]).mean())
    fluctuation = float(solution.finite_population_rms(bias_size)[end])

    ok = bool(
        worst < 0.08
        and abs(exponent + 0.5) < 0.05
        and slope_se < 0.02
        and common_exponent > -0.15  # the falsification arm: NOT -1/2
        and abs(coupled_variance * coupled_size / moved - 1.0) < 0.08
        and coupled_variance * coupled_size / unmoved > 1.5  # the unmoved constant is wrong
        and abs(measured_bias / predicted_bias - 1.0) < 0.05
        and abs(measured_bias / bias_size) < 0.3 * fluctuation
    )
    return FinitePopulationGap(
        sizes=sizes,
        measured_rms=tuple(measured),
        closed_form_rms=tuple(predicted),
        worst_relative_error=worst,
        measured_exponent=exponent,
        exponent_standard_error=slope_se,
        common_shock_exponent=common_exponent,
        coupled_variance_ratio=coupled_variance * coupled_size / moved,
        coupled_wrong_rate_ratio=coupled_variance * coupled_size / unmoved,
        predicted_bias_constant=predicted_bias,
        measured_bias_constant=measured_bias,
        bias_to_fluctuation=abs(measured_bias / bias_size) / fluctuation,
        ok=ok,
    )


# --------------------------------------------------------------------------------------------
# A generic dual-weighted estimator: the adjoint as a linearisation, not as a known matrix.
# --------------------------------------------------------------------------------------------


@partial(jax.jit, static_argnums=0)
def _congested_field(game: CongestedMeanFieldGame, state: Array) -> Array:
    """:meth:`CongestedMeanFieldGame.reduced_field`, jitted at module level.

    The game is a frozen dataclass of scalars, hence hashable, hence a legal static argument;
    binding it inside the method would give every call a fresh cache.
    """
    base = game.base
    mean, slope = state[0], state[1]
    response = base.coupling * mean + game.congestion * mean**3
    return jnp.stack(
        [
            base.closed_loop_rate * mean - base.b**2 / base.r * slope,
            base.q * response - base.closed_loop_rate * slope,
        ]
    )


@dataclass(frozen=True)
class CongestedMeanFieldGame:
    """:class:`LQMeanFieldGame` with a cubic congestion coupling: the reduction survives, the
    linearity does not.

    The running cost pays for distance from a *congestion-shifted* target,
    ``(q/2)(x - c m - gamma m^3)^2``, so the value function stays quadratic in ``x`` -- the Riccati
    coefficient never sees ``gamma`` -- while the reduced two-point boundary value problem for
    ``(m, S)`` becomes nonlinear:

        m' = A m - (b^2/r) S,   S' = q (c m + gamma m^3) - A S,   S(T) + P c_T m(T) = 0

    That is the smallest change that breaks :meth:`LQMeanFieldGame.solve`'s closed form without
    breaking the reduction it rests on, which is what makes it a usable gate for a *generic*
    dual-weighted estimator. At ``congestion = 0`` it is the LQ game exactly, and
    :func:`adjoint_weighted_error` is then exact rather than first-order -- Result 55, recovered as
    a special case. Derived in ``validation/nonlinear_dwr.mac``.
    """

    base: LQMeanFieldGame
    congestion: float

    @property
    def terminal_row(self) -> np.ndarray:
        """``B_T = (P c_T, 1)``, the row the terminal condition sends to zero. ``gamma``-free."""
        return np.array([self.base.riccati_root * self.base.terminal_coupling, 1.0])

    def reduced_field(self, state: Array) -> Array:
        """``F(m, S)``, in jax so the adjoint is :func:`jax.vjp` of this and of nothing else."""
        return _congested_field(self, state)

    def trajectory(self, slope_initial: float, times: np.ndarray) -> np.ndarray:
        """RK4 the reduced system from ``(m_0, slope_initial)``; shape ``(len(times), 2)``."""
        path = np.empty((times.size, 2))
        path[0] = (self.base.mean_initial, slope_initial)
        for i in range(times.size - 1):
            step = times[i + 1] - times[i]
            k1 = np.asarray(self.reduced_field(path[i]))
            k2 = np.asarray(self.reduced_field(path[i] + 0.5 * step * k1))
            k3 = np.asarray(self.reduced_field(path[i] + 0.5 * step * k2))
            k4 = np.asarray(self.reduced_field(path[i] + step * k3))
            path[i + 1] = path[i] + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return path

    def solve(self, n_time: int = 401) -> tuple[np.ndarray, np.ndarray]:
        """Shoot on ``S(0)`` until the terminal row vanishes; return ``(times, trajectory)``.

        The LQ closed form supplies the first iterate, so the secant converges in a handful of
        steps even at a congestion that moves the answer by tens of percent. There is no closed
        form to fall back on here -- that is the point of the class, and it is why this returns
        the two arrays rather than a :class:`MeanFieldSolution`: three of that type's five fields
        do not exist without one.
        """
        times = np.linspace(0.0, self.base.horizon, n_time)
        row = self.terminal_row

        def miss(slope_initial: float) -> float:
            return float(row @ self.trajectory(slope_initial, times)[-1])

        guess = float(self.base.solve(n_time=n_time).value_s[0])
        previous, current = guess, guess * 1.01 + 1e-3
        miss_previous, miss_current = miss(previous), miss(current)
        for _ in range(_SHOOTING_STEPS):
            # The secant plateaus above the target once RK4 noise in `miss` swamps the step, so
            # the loop aims low and the check below decides what actually counts as a failure.
            if abs(miss_current) < 1e-13 or miss_current == miss_previous:
                break
            step = miss_current * (current - previous) / (miss_current - miss_previous)
            previous, miss_previous = current, miss_current
            current = current - step
            miss_current = miss(current)
        if abs(miss_current) > _SHOOTING_TOLERANCE:
            raise ValueError(
                f"the congested two-point problem did not close: terminal row {miss_current:.3e} "
                f"at S(0) = {current:.6f}; the fixed point may have degenerated"
            )
        return times, self.trajectory(current, times)


def adjoint_weighted_error(
    field: Callable[[Array], Array],
    times: np.ndarray,
    trajectory: np.ndarray,
    rate: np.ndarray,
    terminal_row: np.ndarray,
    *,
    free_component: int = 1,
) -> float:
    """Estimate ``J(yhat) - J(y)`` for ``J(y) = y(0)[free_component]``, from ``yhat`` alone.

    The generic form of Result 55. Pairing the linearised primal with the transposed adjoint gives
    ``d/dt (z . delta) = z . g`` for *any* field (``validation/nonlinear_dwr.mac`` STEP 2, residual
    0), and integrating it across the interval leaves

        J(yhat) - J(y) = (eps - int_0^T z . g dt) / z(0)[free_component]

    with ``g = yhat' - F(yhat)`` the defect the caller supplies as ``rate``, ``eps = B_T yhat(T)``
    the terminal defect, and ``z`` the solution of ``-z' = F'(yhat)^T z`` from ``z(T) = B_T``. The
    only thing needed of ``F`` is a vector-Jacobian product, which is :func:`jax.vjp` -- so the
    hand-written affine transition matrix of :func:`dual_weighted_error_estimate` is replaced by a
    linearisation of whatever field is passed in.

    ``free_component`` names the component of ``y(0)`` the boundary value problem does *not* pin;
    the other one is an initial condition and cannot be in error. The returned value is SIGNED,
    because for a nonlinear field the sign of the remainder is part of the measurement.

    Exactness is a property of the field, not of this function. For an affine ``F`` the second
    variation vanishes and the formula has no remainder at any perturbation size; otherwise the
    remainder is ``O(eta^2)`` in the size of the error, so the estimate is first-order.
    :func:`nonlinear_dwr_certificate` measures both exponents rather than asserting them.
    """
    if trajectory.shape != rate.shape or trajectory.shape[0] != times.size:
        raise ValueError(
            f"trajectory {trajectory.shape} and rate {rate.shape} must agree and have "
            f"{times.size} rows, one per time"
        )

    # jax.jacrev IS reverse-mode -- vjp against each basis vector -- so the Jacobian here comes
    # from autodiff of the field and never from a hand-written matrix. Materialising it is the
    # right trade at this size: the system is 2x2, and one vmapped call replaces ~3200 sequential
    # vjps through the RK4 stages. A large system would keep the loop and call jax.vjp per stage.
    midpoints = 0.5 * (times[:-1] + times[1:])
    columns = trajectory.T
    middle_states = np.stack([np.interp(midpoints, times, column) for column in columns], axis=1)
    node_jacobians = np.asarray(jax.vmap(jax.jacrev(field))(jnp.asarray(trajectory)))
    middle_jacobians = np.asarray(jax.vmap(jax.jacrev(field))(jnp.asarray(middle_states)))

    # Backwards in t is forwards in tau = T - t, so the adjoint is an ordinary forward RK4 on the
    # reversed grid, with -F'(yhat)^T contributing the transpose at each stage.
    weights = np.empty_like(trajectory)
    weights[-1] = terminal_row
    for i in range(times.size - 1, 0, -1):
        step = times[i] - times[i - 1]
        upper, lower = node_jacobians[i].T, node_jacobians[i - 1].T
        middle = middle_jacobians[i - 1].T
        k1 = upper @ weights[i]
        k2 = middle @ (weights[i] + 0.5 * step * k1)
        k3 = middle @ (weights[i] + 0.5 * step * k2)
        k4 = lower @ (weights[i] + step * k3)
        weights[i - 1] = weights[i] + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    defect = rate - np.asarray(jax.vmap(field)(jnp.asarray(trajectory)))
    interior = float(np.trapezoid(np.einsum("ij,ij->i", weights, defect), times))
    terminal = float(terminal_row @ trajectory[-1])
    denominator = float(weights[0][free_component])
    if denominator == 0.0:
        return math.inf  # the obstruction: no fixed point, hence no finite error to report
    return (terminal - interior) / denominator


@dataclass(frozen=True)
class NonlinearDwrCurve:
    """Evidence for A16: the adjoint built by autodiff, and what the linearisation costs."""

    perturbations: tuple[float, ...]
    affine_relative_errors: tuple[float, ...]  # flat: exact up to quadrature, at every size
    affine_quadrature_ratio: float  # halving the step divides the floor by ~4
    denominator_against_result_55: float  # |z(0)[1] - den(T)| at congestion 0
    congested_errors: tuple[float, ...]
    congested_exponent: float  # -> 2: the remainder is the second variation
    congested_relative_error: float  # at the largest perturbation, so it is not trivially small
    affine_adjoint_errors: tuple[float, ...]  # same defect, wrong linearisation
    affine_adjoint_exponent: float  # -> 1: the right adjoint is worth exactly one order
    affine_model_error: float  # affine field for BOTH defect and adjoint: stops tracking
    affine_model_exponent: float  # -> 0
    ok: bool


def _perturbation(times: np.ndarray, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    """A smooth ``(p, p')`` with ``p_m(0) = 0``, so the initial condition survives the shift.

    Both components are non-zero at ``T`` on purpose: a perturbation that vanished there would
    leave the terminal defect at zero and exercise only half the estimator.
    """
    s = times / horizon
    shape = np.stack([0.5 * s + np.sin(np.pi * s), 1.0 + 0.4 * np.cos(np.pi * s)], axis=1)
    rate = np.stack(
        [(0.5 + np.pi * np.cos(np.pi * s)) / horizon, -0.4 * np.pi / horizon * np.sin(np.pi * s)],
        axis=1,
    )
    return shape, rate


def _manufactured(
    game: CongestedMeanFieldGame, n_time: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(times, solution, field-at-solution, p, p')`` for the manufactured-error sweep."""
    times, solution = game.solve(n_time=n_time)
    rate = np.asarray(jax.vmap(game.reduced_field)(jnp.asarray(solution)))
    shape, shape_rate = _perturbation(times, game.base.horizon)
    return times, solution, rate, shape, shape_rate


def nonlinear_dwr_certificate(
    congestion: float = 0.6, n_time: int = 801, seed: int = 0
) -> NonlinearDwrCurve:
    """Measure the generic dual-weighted estimator, and price the linearisation it rests on.

    Result 55 shipped an *exact* error quotient for the LQ mean-field game and recorded that the
    exactness belonged to the affine reduced problem. This replaces the hand-written affine adjoint
    with :func:`jax.vjp` of whatever field is handed over, and measures the four statements that
    makes, on a manufactured error ``yhat = y + eta p`` whose true value is known exactly.

    The perturbation is manufactured rather than taken from a trained network deliberately: a
    network's error is not known, so it could not separate "the estimate is first-order" from "the
    network is bad". Here the true error is ``eta p_S(0)`` by construction and the only thing being
    measured is the estimator.

    ``seed`` is accepted for signature parity with the other certificates in this module and is
    unused: nothing here is random.
    """
    del seed
    base_game = _MONOTONE_GAME
    affine = CongestedMeanFieldGame(base=base_game, congestion=0.0)
    congested = CongestedMeanFieldGame(base=base_game, congestion=congestion)
    sizes = tuple(0.05 * 0.5**k for k in range(5))
    logs = np.log(np.asarray(sizes))

    # --- arm 1: an affine field makes the formula EXACT, at every perturbation size ----------
    def affine_relative(grid: int) -> list[float]:
        times, solution, rate, shape, shape_rate = _manufactured(affine, grid)
        out = []
        for size in sizes:
            got = adjoint_weighted_error(
                affine.reduced_field,
                times,
                solution + size * shape,
                rate + size * shape_rate,
                affine.terminal_row,
            )
            out.append(abs(got / (size * shape[0, 1]) - 1.0))
        return out

    affine_errors = affine_relative(n_time)
    refined = affine_relative(2 * n_time - 1)
    quadrature_ratio = float(np.mean(affine_errors) / np.mean(refined))

    # STEP 3e: at congestion 0 the adjoint's free component at t = 0 IS Result 55's den(T). Shoot
    # from S(0) + eta instead of perturbing the trajectory: the result still solves the ODE, so the
    # interior defect drops out and the estimate is exactly eps/z(0)[1]. On an affine field eps is
    # den(T) * eta exactly, so the estimate over eta measures den(T)/z(0)[1] and nothing else --
    # which is a claim about the adjoint, not a restatement of the formula that produced it.
    times, solution = affine.solve(n_time=n_time)
    shot = affine.trajectory(float(solution[0, 1]) + sizes[0], times)
    denominator_gap = abs(
        adjoint_weighted_error(
            affine.reduced_field,
            times,
            shot,
            np.asarray(jax.vmap(affine.reduced_field)(jnp.asarray(shot))),
            affine.terminal_row,
        )
        / sizes[0]
        - 1.0
    )

    # --- arms 2-4: the congested field, with the right adjoint, the wrong one, and neither ----
    times, solution, rate, shape, shape_rate = _manufactured(congested, n_time)
    right, wrong_adjoint, wrong_model = [], [], []
    for size in sizes:
        trajectory, trajectory_rate = solution + size * shape, rate + size * shape_rate
        truth = size * shape[0, 1]
        right.append(
            abs(
                adjoint_weighted_error(
                    congested.reduced_field,
                    times,
                    trajectory,
                    trajectory_rate,
                    congested.terminal_row,
                )
                - truth
            )
        )
        # Same defect, affine linearisation: shifting the rate keeps `rate - F` unchanged, so the
        # only thing that differs from the arm above is which Jacobian the adjoint was built from.
        shift = np.asarray(jax.vmap(affine.reduced_field)(jnp.asarray(trajectory))) - np.asarray(
            jax.vmap(congested.reduced_field)(jnp.asarray(trajectory))
        )
        wrong_adjoint.append(
            abs(
                adjoint_weighted_error(
                    affine.reduced_field,
                    times,
                    trajectory,
                    trajectory_rate + shift,
                    congested.terminal_row,
                )
                - truth
            )
        )
        # And the affine field for both, which is what the Result 55 estimator does if pointed at
        # a congested game: its transition matrix is the LQ one and cannot be told otherwise.
        wrong_model.append(
            abs(
                adjoint_weighted_error(
                    affine.reduced_field,
                    times,
                    trajectory,
                    trajectory_rate,
                    congested.terminal_row,
                )
                - truth
            )
        )

    congested_exponent = float(np.polyfit(logs, np.log(right), 1)[0])
    adjoint_exponent = float(np.polyfit(logs, np.log(wrong_adjoint), 1)[0])
    model_exponent = float(np.polyfit(logs, np.log(wrong_model), 1)[0])
    congested_relative = float(right[0] / abs(sizes[0] * shape[0, 1]))

    ok = bool(
        max(affine_errors) < 1e-5
        and max(affine_errors) / min(affine_errors) < 1.1  # flat in eta: exact, not asymptotic
        and quadrature_ratio > 3.0  # and the floor is quadrature, falling like the step squared
        and denominator_gap < 1e-6
        and abs(congested_exponent - 2.0) < 0.15
        and congested_relative > 0.01  # the arm is not measuring an error too small to matter
        and abs(adjoint_exponent - 1.0) < 0.15
        and abs(model_exponent) < 0.15
        and wrong_adjoint[-1] > 100.0 * right[-1]
    )
    return NonlinearDwrCurve(
        perturbations=sizes,
        affine_relative_errors=tuple(affine_errors),
        affine_quadrature_ratio=quadrature_ratio,
        denominator_against_result_55=denominator_gap,
        congested_errors=tuple(right),
        congested_exponent=congested_exponent,
        congested_relative_error=congested_relative,
        affine_adjoint_errors=tuple(wrong_adjoint),
        affine_adjoint_exponent=adjoint_exponent,
        affine_model_error=float(wrong_model[-1]),
        affine_model_exponent=model_exponent,
        ok=ok,
    )
