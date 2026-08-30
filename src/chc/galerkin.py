"""1D Galerkin finite elements with a tridiagonal (Thomas / progonka) solve.

Weak form of ``-u'' = f`` on ``[0,1]`` with ``u(0)=u(1)=0`` and a piecewise-linear hat basis: the 1D
analogue of the user's 2D bilinear coursework (``plans/11`` §5). Hat stiffness assembles to the
tridiagonal stencil ``(1/h)[-1, 2, -1]``, solved by the Thomas sweep (the "progonka" kernel of
Marchuk-Agoshkov projection-grid methods). Seeds the weak-form / Galerkin track (``plans/01`` §3.3).

That operator is symmetric positive definite, where testing with the trial space is optimal by
Cea's lemma. The second half of the module is the case where that fails: adding advection,
``-eps u'' + s u' = 0``, makes the stencil antisymmetric and the matrix non-SPD, and above the cell
Peclet number ``Pe = s*h/(2*eps) = 1`` the same scheme oscillates. ``convection_diffusion_1d``
carries the Petrov-Galerkin test space that fixes it; the threshold, the oscillation criterion and
the optimal parameter are derived in ``validation/convection_diffusion.mac`` and proved in
``proofs/convection_diffusion.v``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np
from jax import Array, lax


def thomas_solve(sub: Array, diag: Array, sup: Array, rhs: Array) -> Array:
    """Solve a tridiagonal system (Thomas / progonka). ``sub[0]`` and ``sup[-1]`` are unused."""

    def forward(
        carry: tuple[Array, Array], inp: tuple[Array, Array, Array, Array]
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        c_prev, d_prev = carry
        sub_i, diag_i, sup_i, rhs_i = inp
        m = diag_i - sub_i * c_prev
        c_i = sup_i / m
        d_i = (rhs_i - sub_i * d_prev) / m
        return (c_i, d_i), (c_i, d_i)

    _, (cs, ds) = lax.scan(forward, (jnp.zeros(()), jnp.zeros(())), (sub, diag, sup, rhs))

    def backward(x_next: Array, inp: tuple[Array, Array]) -> tuple[Array, Array]:
        c_i, d_i = inp
        x_i = d_i - c_i * x_next
        return x_i, x_i

    _, x_head = lax.scan(backward, ds[-1], (cs[:-1], ds[:-1]), reverse=True)
    return jnp.concatenate([x_head, ds[-1:]])


def poisson_1d(f: Callable[[Array], Array], n: int) -> tuple[Array, Array]:
    """FEM solution of ``-u'' = f`` on ``[0,1]``, ``u(0)=u(1)=0``, on ``n`` uniform elements.

    Returns the node coordinates (length ``n+1``) and the nodal solution (with the zero boundaries).
    """
    h = 1.0 / n
    nodes = jnp.linspace(0.0, 1.0, n + 1)
    interior = nodes[1:-1]  # n-1 unknowns
    diag = jnp.full(n - 1, 2.0 / h)
    off = jnp.full(n - 1, -1.0 / h)
    sub = off.at[0].set(0.0)
    sup = off.at[-1].set(0.0)
    rhs = f(interior) * h  # lumped load, O(h^2) like the FEM itself
    u_interior = thomas_solve(sub, diag, sup, rhs)
    return nodes, jnp.concatenate([jnp.zeros(1), u_interior, jnp.zeros(1)])


# Q1 bilinear element stiffness for the 2D Laplacian on a square (scale-invariant in 2D).
_Q1_STIFFNESS = (1.0 / 6.0) * np.array(
    [[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]], dtype=float
)


def poisson_2d(f: Callable[[Array, Array], Array], n: int) -> tuple[Array, Array]:
    """FEM solution of ``-Δu = f`` on ``[0,1]^2``, ``u=0`` on the boundary, bilinear (Q1) elements.

    The 2D analogue of the user's coursework (``plans/11`` §5): a bilinear tensor-product basis
    assembled element-by-element on an ``n x n`` grid; ``f`` is vectorised over the grids.
    Returns the 1D node coordinates and the ``(n+1, n+1)`` nodal solution.
    """
    h = 1.0 / n
    m = n - 1  # interior nodes per axis
    coords = np.linspace(0.0, 1.0, n + 1)

    def gidx(i: int, j: int) -> int:
        return (i - 1) * m + (j - 1)

    stiffness = np.zeros((m * m, m * m))
    for ex in range(n):
        for ey in range(n):
            corners = [(ex, ey), (ex + 1, ey), (ex + 1, ey + 1), (ex, ey + 1)]
            for a, (ia, ja) in enumerate(corners):
                if 1 <= ia <= n - 1 and 1 <= ja <= n - 1:
                    for b, (ib, jb) in enumerate(corners):
                        if 1 <= ib <= n - 1 and 1 <= jb <= n - 1:
                            stiffness[gidx(ia, ja), gidx(ib, jb)] += _Q1_STIFFNESS[a, b]

    xs, ys = np.meshgrid(coords[1:-1], coords[1:-1], indexing="ij")
    load = (np.asarray(f(jnp.asarray(xs), jnp.asarray(ys))) * h**2).reshape(m * m)
    u_interior = np.linalg.solve(stiffness, load).reshape(m, m)

    u = np.zeros((n + 1, n + 1))
    u[1:-1, 1:-1] = u_interior
    return jnp.asarray(coords), jnp.asarray(u)


def optimal_upwind(peclet: float) -> float:
    """The SUPG parameter ``coth(Pe) - 1/Pe`` that makes 1-D convection-diffusion nodally exact.

    Derived in ``validation/convection_diffusion.mac`` STEP 4: demanding that the stabilised
    amplification factor equal the exact ``exp(2*Pe)`` forces the effective cell Peclet to be
    ``tanh(Pe)``, and solving ``Pe/(1 + alpha*Pe) = tanh(Pe)`` gives this. Christie-Griffiths-
    Mitchell-Zienkiewicz (1976); Brooks-Hughes (1982).

    ``coth(Pe) - 1/Pe`` is a difference of two quantities that both blow up as ``Pe -> 0``, so the
    naive form loses every significant digit there. The series ``Pe/3 - Pe^3/45`` takes over below
    the crossover, where it is already accurate to well past double precision.
    """
    if abs(peclet) < 1e-3:
        return peclet / 3.0 - peclet**3 / 45.0
    # At the other end, coth(Pe) - 1 = 2/(exp(2*Pe) - 1) drops below double precision around
    # Pe = 19, so this saturates at exactly 1 - 1/Pe: the boundary of the stability range, where
    # the super-diagonal vanishes and the scheme degenerates to the exact upwind limit.
    return 1.0 / np.tanh(peclet) - 1.0 / peclet


def convection_diffusion_exact(x: Array, eps: float, s: float) -> Array:
    """Exact solution of ``-eps u'' + s u' = 0`` on ``[0,1]`` with ``u(0)=0``, ``u(1)=1``.

    Written about the layer at ``x = 1`` (``validation/convection_diffusion.mac`` STEP 1d) rather
    than as ``(exp(s x/eps) - 1)/(exp(s/eps) - 1)``: the two are equal, but the latter overflows
    once ``s/eps`` passes ~700 while this form stays bounded, and the convection-dominated regime is
    the whole point of the problem.
    """
    ratio = s / eps
    return (jnp.exp(ratio * (x - 1.0)) - jnp.exp(-ratio)) / (1.0 - jnp.exp(-ratio))


def convection_diffusion_1d(
    eps: float, s: float, n: int, *, alpha: float = 0.0
) -> tuple[Array, Array]:
    """Petrov-Galerkin FEM for ``-eps u'' + s u' = 0`` on ``[0,1]``, ``u(0)=0``, ``u(1)=1``.

    Unlike :func:`poisson_1d`, the operator here is **not symmetric**: the advection term
    ``s integral(u' phi_i)`` assembles antisymmetrically to ``+-s/2``, so the stiffness matrix is
    not SPD and testing with the trial space is no longer the natural choice.

    ``alpha`` selects the test space ``w_i = phi_i + alpha*(h/2)*phi_i'``. For piecewise-linear
    elements the added piece contributes exactly an artificial diffusion ``alpha*s*h/2`` (the
    ``phi_i''`` term vanishes inside every element), so the assembled scheme is the central one with
    ``eps -> eps*(1 + alpha*Pe)``. That is the Petrov-Galerkin matrix, not a shortcut around it.

    - ``alpha = 0`` -- Bubnov-Galerkin (test space = trial space), central differencing. Oscillates
      once the cell Peclet ``Pe = s*h/(2*eps)`` exceeds 1.
    - ``alpha = 1`` -- full upwind. Never oscillates, but its amplification factor is exactly
      ``1 + 2*Pe``, the first two terms of ``exp(2*Pe)``, so it is only first-order accurate.
    - ``alpha = optimal_upwind(Pe)`` -- nodally exact.

    Returns the node coordinates (length ``n+1``) and the nodal solution including both boundaries.
    """
    h = 1.0 / n
    peclet = s * h / (2.0 * eps)
    eps_eff = eps * (1.0 + alpha * peclet)
    nodes = jnp.linspace(0.0, 1.0, n + 1)
    diag = jnp.full(n - 1, 2.0 * eps_eff / h)
    sub_val = -eps_eff / h - s / 2.0
    sup_val = -eps_eff / h + s / 2.0
    sub = jnp.full(n - 1, sub_val).at[0].set(0.0)
    sup = jnp.full(n - 1, sup_val).at[-1].set(0.0)
    # u(0) = 0 contributes nothing; u(1) = 1 moves the last super-diagonal entry to the load.
    rhs = jnp.zeros(n - 1).at[-1].set(-sup_val)
    u_interior = thomas_solve(sub, diag, sup, rhs)
    return nodes, jnp.concatenate([jnp.zeros(1), u_interior, jnp.ones(1)])


@dataclass(frozen=True)
class ConvectionDiffusionCurve:
    """Evidence for the cell-Peclet threshold and for what a Petrov-Galerkin test space buys."""

    peclet: float  # s*h/(2*eps) on the coarse mesh -- above 1 by construction
    alpha_optimal: float  # coth(Pe) - 1/Pe, the SUPG parameter
    bubnov_undershoot: (
        float  # min nodal value; the exact solution lives in [0,1], so < 0 is spurious
    )
    bubnov_sign_changes: int  # alternations in the nodal differences -- the oscillation itself
    upwind_undershoot: float  # full upwind is monotone: must be >= 0
    bubnov_error: float  # max nodal error of the Bubnov-Galerkin scheme
    upwind_error: float  # ... of full upwind: stable, but over-diffusive
    supg_error: float  # ... of the optimal Petrov-Galerkin scheme: nodally exact
    supg_error_ulps: float  # supg_error / machine epsilon -- the precision-INDEPENDENT statement
    fine_peclet: float  # the same problem resolved so that Pe < 1
    fine_bubnov_undershoot: float  # there Bubnov-Galerkin is already monotone: the other side
    measured_threshold: float  # Pe at which the undershoot appears, found by bisection: must be ~1
    upwind_order: float  # measured convergence order of full upwind: ~1, not 2
    ok: bool


def convection_diffusion_certificate(
    *, eps: float = 0.01, s: float = 1.0, n_coarse: int = 20, n_fine: int = 200
) -> ConvectionDiffusionCurve:
    """Exhibit the cell-Peclet instability, its threshold, and the Petrov-Galerkin cure.

    Derived in ``validation/convection_diffusion.mac``, machine-checked in
    ``proofs/convection_diffusion.v``. Three schemes on the same mesh, one of which must fail:

    - Bubnov-Galerkin above ``Pe = 1`` oscillates, because its amplification factor
      ``(1+Pe)/(1-Pe)`` turns negative there while the exact ``exp(2*Pe)`` never does;
    - full upwind never oscillates but its amplification is exactly ``1 + 2*Pe``, the first two
      terms of ``exp(2*Pe)``, so it is first-order;
    - the optimal ``alpha`` matches the exact factor and is nodally exact.

    ``supg_error`` is reported both absolutely and **in units of the working machine epsilon**. The
    absolute number is meaningless without the dtype -- it is 6e-9 under float32 and 1e-16 under
    float64 -- and a threshold tuned to one of those would silently pass on the other.
    """
    h = 1.0 / n_coarse
    peclet = s * h / (2.0 * eps)
    alpha_opt = optimal_upwind(peclet)

    def solve(alpha: float, n: int, epsilon: float) -> tuple[Array, Array, Array]:
        nodes, u = convection_diffusion_1d(epsilon, s, n, alpha=alpha)
        return nodes, u, convection_diffusion_exact(nodes, epsilon, s)

    _, u_bub, exact = solve(0.0, n_coarse, eps)
    _, u_up, _ = solve(1.0, n_coarse, eps)
    _, u_supg, _ = solve(alpha_opt, n_coarse, eps)
    diffs = np.diff(np.asarray(u_bub))
    sign_changes = int(np.sum(np.sign(diffs[:-1]) * np.sign(diffs[1:]) < 0))

    # Below the threshold the same scheme is fine -- the failure is the Peclet number, not the mesh.
    _, u_fine, _ = solve(0.0, n_fine, eps)
    fine_peclet = s / (2.0 * eps * n_fine)

    # Bisect on Pe (through eps) for the value at which the undershoot first appears.
    def undershoot_at(pe: float) -> float:
        _, u = convection_diffusion_1d(s * h / (2.0 * pe), s, n_coarse, alpha=0.0)
        return float(jnp.min(u))

    lo, hi = 0.5, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if undershoot_at(mid) < -1e-12:
            hi = mid
        else:
            lo = mid

    # Convergence order of full upwind, measured where the layer is resolved so that the rate is
    # the scheme's own and not the unresolved-layer O(1) plateau.
    errs = []
    for n in (40, 80, 160, 320):
        _, u_n, ex_n = solve(1.0, n, 0.1)
        errs.append(float(jnp.max(jnp.abs(u_n - ex_n))))
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]

    machine_eps = float(np.finfo(np.asarray(u_supg).dtype).eps)
    supg_error = float(jnp.max(jnp.abs(u_supg - exact)))
    curve = ConvectionDiffusionCurve(
        peclet=peclet,
        alpha_optimal=alpha_opt,
        bubnov_undershoot=float(jnp.min(u_bub)),
        bubnov_sign_changes=sign_changes,
        upwind_undershoot=float(jnp.min(u_up)),
        bubnov_error=float(jnp.max(jnp.abs(u_bub - exact))),
        upwind_error=float(jnp.max(jnp.abs(u_up - exact))),
        supg_error=supg_error,
        supg_error_ulps=supg_error / machine_eps,
        fine_peclet=fine_peclet,
        fine_bubnov_undershoot=float(jnp.min(u_fine)),
        measured_threshold=0.5 * (lo + hi),
        upwind_order=float(np.mean(orders)),
        ok=False,
    )
    ok = (
        curve.peclet > 1.0
        and curve.bubnov_undershoot < -0.01  # it really does oscillate
        and curve.bubnov_sign_changes > 0
        and curve.upwind_undershoot >= -1e-12  # and full upwind really does not
        and curve.supg_error_ulps < 10.0  # nodally exact, at whatever precision is in force
        and curve.fine_bubnov_undershoot >= -1e-12  # below the threshold Bubnov is fine
        and abs(curve.measured_threshold - 1.0) < 0.01  # the derived threshold, measured
        and abs(curve.upwind_order - 1.0) < 0.15  # first order, not second
    )
    return replace(curve, ok=ok)
