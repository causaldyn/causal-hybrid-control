"""Offline optimal control over a box: projected gradient, and a bound-constrained quasi-Newton.

The projected gradient also takes linear constraints on the whole action sequence
(:class:`LinearConstraint` -- a budget, a rate limit), projected onto together with the box so that
every iterate stays feasible; see ``docs/adr/0001-linear-action-constraints.md``.

Both minimise the same Bolza objective from the same discrete adjoint
(:func:`chc.adjoint.control_gradient_adjoint`); they differ only in how they use it.
:func:`projected_gradient_control` takes backtracked steepest-descent steps, which is monotone and
dependency-free but first-order, so an ill-conditioned instance needs *many* of them -- thousands,
not hundreds. Its ``steps`` is therefore a cap rather than a bill: the descent runs inside one
compiled program that stops the moment the line search fails, so an unused step costs nothing and
the default is loose enough for the stopping rule to decide. :func:`lbfgs_box_control` hands the
same gradient to SciPy's L-BFGS-B, which curves the step with a limited-memory secant
approximation and reaches stationarity in tens of iterations rather than thousands -- but crosses
the Python boundary on each one, so it cannot be compiled and is the reference rather than the
workhorse. :func:`nlp_solver_certificate` measures the gap rather than asserting it in prose.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray
from scipy.optimize import linprog, minimize

from chc.adjoint import control_gradient_adjoint
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import DampedOscillator, Dynamics, HybridDynamics
from chc.residual import MLPResidual, ZeroResidual

_MAX_BACKTRACK = 40
_MAX_SWEEPS = 4_096  # Dykstra sweeps per projection, a multiple of the polishing period
_POLISH_EVERY = 32  # sweeps between attempts to finish a projection by the active-set polish
_ACTIVE_SET_STEPS = 16  # dual active-set steps per polishing attempt

Bound = float | Array
"""One side of the action box: a scalar shared by every lever, or a per-lever array.

A real actuator set is rarely a cube -- a budget and a discount move in different units and over
different ranges -- so a single scalar pair forces the caller to widen every lever to the loosest
one, which is a larger feasible set than the plant has.
"""


def broadcast_box(bound: Bound, shape: tuple[int, ...], name: str, dtype: Any) -> Array:
    """Expand a scalar or per-lever bound to the full ``(horizon, m)`` action shape.

    Accepts a scalar, a ``(m,)`` per-lever vector, or an already-full ``(horizon, m)`` schedule.
    A 1-D array is read as **per-lever**, never as per-step: for ``m == 1`` a ``(horizon,)`` array
    would broadcast along the lever axis and silently constrain the wrong axis, so time-varying
    bounds have to be spelled out in two dimensions.

    ``dtype`` is the *actions'* dtype, and it is required rather than defaulted: the box is
    validated in float64 but returned in the caller's precision, because a float64 box clipped
    against float32 actions promotes the answer and silently changes what the caller asked for.

    Raises:
        ValueError: on a shape that is neither, or on an inverted box. ``jnp.clip`` with
            ``lo > hi`` returns ``hi`` everywhere without complaint, which is a wrong answer
            rather than a failure, so the ordering is checked here where it can still be reported.
    """
    values = np.asarray(bound, dtype=np.float64)
    if values.ndim == 1 and values.shape != shape[1:]:
        raise ValueError(
            f"{name} has shape {values.shape}; a 1-D bound is per-lever and must be "
            f"{shape[1:]}. For a bound that varies over time pass the full {shape}."
        )
    if values.ndim > 2 or (values.ndim == 2 and values.shape != shape):
        raise ValueError(
            f"{name} has shape {values.shape}; expected a scalar, {shape[1:]} or {shape}"
        )
    return jnp.asarray(np.broadcast_to(values, shape), dtype=dtype)


def check_box(lo: Array, hi: Array) -> None:
    """Reject an inverted or empty box before ``jnp.clip`` turns it into a silent answer."""
    bad = np.asarray(lo) > np.asarray(hi)
    if bool(bad.any()):
        first = tuple(int(i) for i in np.argwhere(bad)[0])
        raise ValueError(
            f"empty action box at index {first}: u_lo {float(np.asarray(lo)[first])} > "
            f"u_hi {float(np.asarray(hi)[first])}"
        )


def project_box(us: Array, lo: Bound, hi: Bound) -> Array:
    """Euclidean projection onto the box ``[lo, hi]`` (elementwise clip).

    ``lo`` and ``hi`` may be scalars or per-lever arrays; both broadcast against ``us``.
    """
    return jnp.clip(us, lo, hi)


@dataclass(frozen=True)
class LinearConstraint:
    """``lower <= matrix @ us.ravel() <= upper``: linear inequalities over the action sequence.

    The actions are flattened row-major, so column ``k * m + j`` of ``matrix`` is lever ``j`` at
    step ``k``. An absent side is ``-inf`` or ``inf``, which lets one type state a half-space, a
    band or an equality: a total budget is a single row of ones with ``upper`` the budget, and a
    limit on how fast a lever may move is a band on consecutive differences (:meth:`rate_limit`).
    ``lower`` and ``upper`` may be scalars shared by every row.

    Every iterate of the solve satisfies the constraints, not only the answer: the step is projected
    onto ``box ∩ constraints`` by Dykstra's algorithm, so a solve stopped by its step budget still
    hands back an executable plan. Why that, and not an augmented Lagrangian, is measured in
    ``docs/adr/0001-linear-action-constraints.md``.

    Raises:
        ValueError: on a row of zeros (it constrains nothing, or nothing satisfies it, and either
            way it is a mistake), a non-finite coefficient, a NaN bound, or ``lower > upper``.
    """

    matrix: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]

    def __post_init__(self) -> None:
        matrix = np.atleast_2d(np.asarray(self.matrix, dtype=np.float64))
        if matrix.ndim != 2:
            raise ValueError(f"constraint matrix must be 2-D, got shape {matrix.shape}")
        rows = matrix.shape[0]
        try:
            lower = np.broadcast_to(np.asarray(self.lower, dtype=np.float64), (rows,)).copy()
            upper = np.broadcast_to(np.asarray(self.upper, dtype=np.float64), (rows,)).copy()
        except ValueError as error:
            raise ValueError(
                f"constraint bounds must be scalars or length {rows}, one per row of the matrix"
            ) from error
        if not np.isfinite(matrix).all():
            raise ValueError("constraint matrix has a non-finite coefficient")
        if rows and not np.any(matrix, axis=1).all():
            zero = int(np.flatnonzero(~np.any(matrix, axis=1))[0])
            raise ValueError(f"row {zero} of the constraint matrix is zero")
        if np.isnan(lower).any() or np.isnan(upper).any():
            raise ValueError("constraint bounds contain NaN; an absent side is -inf or inf")
        if (lower > upper).any():
            bad = int(np.flatnonzero(lower > upper)[0])
            raise ValueError(f"row {bad} has lower {lower[bad]} above upper {upper[bad]}")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def rate_limit(cls, horizon: int, caps: Sequence[float]) -> LinearConstraint:
        """``|u[k+1, j] - u[k, j]| <= caps[j]`` between consecutive planned steps, per lever.

        ``caps`` holds one entry per lever and ``inf`` leaves that lever free; only finite caps
        produce rows, so an all-``inf`` call returns a constraint with none. The band is between
        planned steps only: the move from the action already applied to the first planned one is
        not constrained, because the solver is not told what that action was.
        """
        limits = np.asarray(caps, dtype=np.float64)
        if limits.ndim != 1 or np.isnan(limits).any() or (limits < 0.0).any():
            raise ValueError(f"caps must be one non-negative number per lever, got {caps!r}")
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        levers = limits.shape[0]
        capped = np.flatnonzero(np.isfinite(limits))
        matrix = np.zeros((capped.size * (horizon - 1), horizon * levers))
        for block, lever in enumerate(capped):
            for step in range(horizon - 1):
                row = block * (horizon - 1) + step
                matrix[row, (step + 1) * levers + lever] = 1.0
                matrix[row, step * levers + lever] = -1.0
        bound = np.repeat(limits[capped], horizon - 1)
        return cls(matrix, -bound, bound)


Blocks = tuple[tuple[Array, Array, Array, Array], ...]
"""Constraint rows coloured into mutually orthogonal classes: ``(rows, lower, upper, |row|^2)``."""

Duals = tuple[Array, tuple[Array, ...]]
"""Dykstra's increments: one full vector for the box, one multiplier per constraint row."""


def _constraint_blocks(
    constraints: Sequence[LinearConstraint], lo: Array, hi: Array, dtype: Any
) -> Blocks:
    """Stack, check and colour the rows; ``()`` means the box alone and the old code path.

    Rows in one class are mutually orthogonal, so Dykstra can project onto all of them at once --
    exactly, since orthogonal normals do not interact -- and a sweep costs one matrix-vector product
    per class rather than one sequential update per row. A rate band on a chain needs two classes
    (even and odd steps); a budget row of ones is orthogonal to every difference row and joins one.

    Raises:
        ValueError: if a constraint's width is not the flattened action size, or if no action
            sequence satisfies the box and the constraints together. Dykstra does not detect an
            empty intersection -- it would run every projection to its sweep cap and return a point
            that satisfies nothing -- so emptiness is settled once here, by a linear program.
    """
    size = int(np.prod(lo.shape))
    present = [constraint for constraint in constraints if constraint.matrix.shape[0]]
    for constraint in present:
        if constraint.matrix.shape[1] != size:
            raise ValueError(
                f"a constraint has {constraint.matrix.shape[1]} columns, but the actions "
                f"{tuple(lo.shape)} flatten to {size}"
            )
    if not present:
        return ()
    matrix = np.vstack([constraint.matrix for constraint in present])
    lower = np.concatenate([constraint.lower for constraint in present])
    upper = np.concatenate([constraint.upper for constraint in present])

    has_upper, has_lower = np.isfinite(upper), np.isfinite(lower)
    feasibility = linprog(
        np.zeros(size),
        A_ub=np.vstack([matrix[has_upper], -matrix[has_lower]]),
        b_ub=np.concatenate([upper[has_upper], -lower[has_lower]]),
        bounds=list(
            zip(
                np.asarray(lo, dtype=np.float64).ravel(),
                np.asarray(hi, dtype=np.float64).ravel(),
                strict=True,
            )
        ),
        method="highs",
    )
    if feasibility.status == 2:
        raise ValueError(
            f"no action sequence satisfies both the box and the {matrix.shape[0]} constraint rows"
        )

    overlaps = np.abs(matrix @ matrix.T) > 0.0
    classes: list[list[int]] = []
    blocked: list[NDArray[np.bool_]] = []
    for row in range(matrix.shape[0]):
        for members, mask in zip(classes, blocked, strict=True):
            if not mask[row]:
                members.append(row)
                mask |= overlaps[row]
                break
        else:
            classes.append([row])
            blocked.append(overlaps[row].copy())
    return tuple(
        (
            jnp.asarray(matrix[members], dtype=dtype),
            jnp.asarray(lower[members], dtype=dtype),
            jnp.asarray(upper[members], dtype=dtype),
            jnp.asarray(np.sum(matrix[members] ** 2, axis=1), dtype=dtype),
        )
        for members in classes
    )


def _no_duals(size: int, blocks: Blocks, dtype: Any) -> Duals:
    return jnp.zeros(size, dtype), tuple(jnp.zeros(rows.shape[0], dtype) for rows, *_ in blocks)


def _side(value: Array, low: Array, high: Array) -> Array:
    """``+1`` above ``high``, ``-1`` below ``low``, ``0`` between: the bound a value presses on."""
    return jnp.where(value > high, 1, jnp.where(value < low, -1, 0)).astype(jnp.int8)


def _polish(
    y: Array,
    lo: Array,
    hi: Array,
    blocks: Blocks,
    duals: Duals,
    tolerance: Array,
    steps: int = _ACTIVE_SET_STEPS,
) -> tuple[Array, Array, Duals]:
    """Finish the projection with a dual active-set method started from Dykstra's increments.

    The increments name the active set long before they settle: alternating projections converge
    linearly at a rate set by the angles between the sets, which on an unlucky polytope is thousands
    of sweeps. With the active set fixed the projection is an equality-constrained least-squares
    problem, one linear solve (as in OSQP's solution polishing, Stellato et al. 2020). The guess
    can be wrong in one way that matters: at a vertex, a constraint Dykstra has not yet released
    makes the set over-determined, so the solve is singular, and swapping sides by the signs of
    its solution (a primal-dual active-set update) has nothing to swap on.

    So each step moves the multipliers towards the solution on the current set and stops at the
    first multiplier that would change sign, which then leaves the set -- the ratio test of the dual
    method of Goldfarb & Idnani (1983). Started from Dykstra's increments the multipliers already
    have the right signs, and the ratio test keeps them so. The solve carries a ridge of
    ``256 eps |row|^2``: below rounding on a well-posed set, while on an over-determined one it
    turns the step into the dual ascent direction along the dependency, so the ratio test drops the
    constraint that direction releases first. A full step with every row and coordinate in bounds
    ends the method; one that breaks bounds adds the broken ones, with a zero multiplier.

    The result is kept only if it passes the projection's KKT conditions -- primal feasibility and
    the right sign on every multiplier -- which for this strictly convex problem make it the
    projection; otherwise Dykstra carries on. Each step is one ``O(p^3)`` solve in the number of
    constraint rows, and there are at most ``steps``.
    """
    _, multipliers = duals  # the box's sides are read off the pull the row multipliers leave
    rows = jnp.concatenate([block[0] for block in blocks])
    lower = jnp.concatenate([block[1] for block in blocks])
    upper = jnp.concatenate([block[2] for block in blocks])
    norms = jnp.concatenate([block[3] for block in blocks])
    root = jnp.sqrt(norms)
    two_sided = lower < upper
    ridge = 256 * jnp.finfo(y.dtype).eps * norms

    def point(box_side: Array, nu: Array) -> tuple[Array, Array, Array]:
        pull = y - rows.T @ nu
        pinned = jnp.where(box_side > 0, hi, lo)
        return jnp.where(box_side == 0, pull, pinned), pull, pinned

    State = tuple[Array, Array, Array, Array, Array]

    def unsettled(state: State) -> Array:
        *_, count, settled = state
        return ~settled & (count < steps)

    def step(state: State) -> State:
        row_side, box_side, nu, count, _ = state
        active, free = row_side != 0, box_side == 0
        x, pull, pinned = point(box_side, nu)
        free_rows = jnp.where(free, rows, 0.0)
        system = jnp.where(active[:, None] & active[None, :], free_rows @ free_rows.T, 0.0)
        system = system + jnp.diag(jnp.where(active, ridge, 1.0))
        target = jnp.where(row_side > 0, upper, lower)
        residual = jnp.where(active, rows @ x - target, 0.0)
        move = jnp.linalg.solve(system, residual)
        shift = -(rows.T @ move)
        # How far each multiplier can travel before its sign turns; equality rows take either.
        row_room = jnp.where(
            active & two_sided & (row_side * move < 0), row_side * nu / -(row_side * move), jnp.inf
        )
        box_room = jnp.where(
            ~free & (box_side * shift < 0),
            box_side * (pull - pinned) / -(box_side * shift),
            jnp.inf,
        )
        length = jnp.clip(jnp.minimum(jnp.min(row_room), jnp.min(box_room)), 0.0, 1.0)
        blocked = length < 1
        leaving_rows = blocked & (row_room <= length)
        nu = jnp.where(leaving_rows, 0.0, nu + length * move)
        row_side = jnp.where(leaving_rows, 0, row_side).astype(jnp.int8)
        box_side = jnp.where(blocked & (box_room <= length), 0, box_side).astype(jnp.int8)
        x, _, _ = point(box_side, nu)
        level = rows @ x
        slack = tolerance * root
        broken_rows = (
            ~blocked & (row_side == 0) & ((level > upper + slack) | (level < lower - slack))
        )
        broken_box = ~blocked & (box_side == 0) & ((x > hi + tolerance) | (x < lo - tolerance))
        row_side = jnp.where(broken_rows, _side(level, lower, upper), row_side)
        box_side = jnp.where(broken_box, _side(x, lo, hi), box_side)
        on_target = jnp.abs(jnp.where(row_side != 0, level - target, 0.0)) <= slack
        settled = ~blocked & ~jnp.any(broken_rows) & ~jnp.any(broken_box) & jnp.all(on_target)
        return row_side, box_side, nu, count + 1, settled

    nu = jnp.concatenate(multipliers)
    start = (
        jnp.sign(nu).astype(jnp.int8),
        _side(y - rows.T @ nu, lo, hi),
        nu,
        jnp.asarray(0),
        jnp.asarray(False),
    )
    _, box_side, nu, _, _ = jax.lax.while_loop(unsettled, step, start)
    x, pull, _ = point(box_side, nu)
    level = rows @ x
    slack = tolerance * root
    # Every multiplier in the units of x; ``x = y - rows' nu - push`` holds by construction, so
    # what remains of the KKT conditions is checked on the pair itself, not on the method's sides.
    force, push = nu * root, pull - x
    kkt = (
        jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(nu))
        & jnp.all((x >= lo - tolerance) & (x <= hi + tolerance))
        & jnp.all((level >= lower - slack) & (level <= upper + slack))
        # Complementary slackness with the sign built in: a multiplier that pushes down holds its
        # constraint at the upper bound, one that pushes up at the lower. A step the ratio test cut
        # short leaves rows off their bounds with multipliers still on them, and fails here.
        & jnp.all(jnp.where(force > tolerance, level >= upper - slack, True))
        & jnp.all(jnp.where(force < -tolerance, level <= lower + slack, True))
        & jnp.all(jnp.where(push > tolerance, x >= hi - tolerance, True))
        & jnp.all(jnp.where(push < -tolerance, x <= lo + tolerance, True))
    )
    sizes = np.cumsum([block[0].shape[0] for block in blocks])[:-1]
    return kkt, x, (pull - x, tuple(jnp.split(nu, sizes)))


def _dykstra(y: Array, lo: Array, hi: Array, blocks: Blocks, duals: Duals) -> tuple[Array, Duals]:
    """Euclidean projection of flat ``y`` onto ``box ∩ blocks``, warm-started from ``duals``.

    Dykstra's algorithm is block-coordinate ascent on the dual of the projection (Gaffke & Mathar
    1989; Tibshirani 2017), so the increments left by the previous projection are a valid starting
    dual for the next one, and ``x = y - box_increment - sum_c rows_c' nu_c`` recovers its primal.
    Warm-starting roughly halves the sweeps a solve needs. The box goes LAST in every sweep, so the
    returned point satisfies the box exactly and the rows to the sweep tolerance.

    It stops when the INCREMENTS stop moving, not when ``x`` does (Birgin & Raydan 2005). A sweep
    can hand ``x`` back exactly where it started while the increments are still travelling -- a
    random three-row instance did so after 31 sweeps at a point that broke a row by ``7.8e-2`` --
    so an unchanged ``x`` proves nothing. Unchanged increments do: every set then fixes ``x``, each
    increment lies in its set's normal cone there, and ``y - x`` is their sum, which is exactly the
    optimality condition of the projection.

    Every ``_POLISH_EVERY`` sweeps, and once before the first with the warm-start duals, a dual
    active-set method started from the increments finishes the projection (:func:`_polish`), and its
    answer is kept if it passes the KKT check. Warm-started inside a descent the active set rarely
    changes between steps, so most projections end at that first attempt without a sweep; on the
    random polytopes where plain Dykstra ran out its cap, the polish is what makes the answer exact
    rather than merely reported.
    """
    box_increment, multipliers = duals
    x = y - box_increment
    for (rows, *_), nu in zip(blocks, multipliers, strict=True):
        x = x - rows.T @ nu
    tolerance = 256 * jnp.finfo(y.dtype).eps * (1 + jnp.max(jnp.abs(y)))

    def unsettled(state: tuple[Array, Array, tuple[Array, ...], Array, Array]) -> Array:
        _, _, _, count, change = state
        return jnp.logical_and(count < _MAX_SWEEPS, change > tolerance)

    def polished(
        state: tuple[Array, Array, tuple[Array, ...], Array, Array],
    ) -> tuple[Array, Array, tuple[Array, ...], Array, Array]:
        _, box_increment, multipliers, count, change = state
        kkt, x, (box_increment, multipliers) = _polish(
            y, lo, hi, blocks, (box_increment, multipliers), tolerance
        )
        exact = (x, box_increment, multipliers, count, jnp.zeros_like(change))
        return jax.tree.map(lambda new, old: jnp.where(kkt, new, old), exact, state)

    def sweep(
        state: tuple[Array, Array, tuple[Array, ...], Array, Array],
    ) -> tuple[Array, Array, tuple[Array, ...], Array, Array]:
        x, box_increment, multipliers, count, _ = state
        change = jnp.zeros((), dtype=x.dtype)
        updated = []
        for (rows, lower, upper, norms), nu in zip(blocks, multipliers, strict=True):
            level = rows @ x + nu * norms
            renewed = (level - jnp.clip(level, lower, upper)) / norms
            x = x + rows.T @ (nu - renewed)
            # |Δnu_i| * |row_i| is how far row i's increment vector moved.
            change = jnp.maximum(change, jnp.max(jnp.abs(renewed - nu) * jnp.sqrt(norms)))
            updated.append(renewed)
        shifted = x + box_increment
        x = jnp.clip(shifted, lo, hi)
        change = jnp.maximum(change, jnp.max(jnp.abs(shifted - x - box_increment)))
        state = (x, shifted - x, tuple(updated), count + 1, change)
        return jax.lax.cond((count + 1) % _POLISH_EVERY == 0, polished, lambda kept: kept, state)

    start = (x, box_increment, multipliers, jnp.asarray(0), jnp.asarray(jnp.inf, dtype=y.dtype))
    x, box_increment, multipliers, _, _ = jax.lax.while_loop(unsettled, sweep, polished(start))
    return x, (box_increment, multipliers)


# Compiled once per problem shape. Called eagerly, the ``while_loop`` inside would be re-traced on
# every solve, since its body closures are rebuilt per call.
_project_once = eqx.filter_jit(_dykstra)


def _violation(us: Array, constraints: Sequence[LinearConstraint]) -> float:
    """Worst amount by which ``us`` breaks a constraint row, in that row's own units."""
    flat = np.asarray(us, dtype=np.float64).ravel()
    worst = 0.0
    for constraint in constraints:
        if constraint.matrix.shape[0]:
            level = constraint.matrix @ flat
            excess = np.maximum(constraint.lower - level, level - constraint.upper)
            worst = max(worst, float(np.max(excess)))
    return worst


def _polytope_stationarity(
    us: Array, gradient: Array, lo: Array, hi: Array, blocks: Blocks
) -> float:
    """``||u - P(u - grad J)||``, ``P`` onto ``box ∩ constraints``: zero exactly at a KKT point."""
    stepped = (us - gradient).astype(us.dtype).ravel()
    projected, _ = _project_once(
        stepped, lo.ravel(), hi.ravel(), blocks, _no_duals(stepped.size, blocks, us.dtype)
    )
    return float(jnp.linalg.norm(us.ravel() - projected))


def _backtrack(
    us: Array,
    current: Array,
    grad: Array,
    u_lo: Array,
    u_hi: Array,
    lr0: float,
    tol: float,
    value_of: Callable[[Array], Array],
    blocks: Blocks,
    duals: Duals,
) -> tuple[Array, Array, Array, Duals]:
    """Halve the step until it strictly decreases ``value_of``; report whether one ever did.

    With constraint ``blocks`` every trial is projected by Dykstra, starting from the ``duals`` of
    the current iterate, and the accepted trial's duals are handed back to warm-start the next step.
    """

    def searching(state: tuple[Array, Array, Array, Array, Array, Duals]) -> Array:
        trial, _, _, _, accepted, _ = state
        return jnp.logical_and(trial < _MAX_BACKTRACK, jnp.logical_not(accepted))

    def halve(
        state: tuple[Array, Array, Array, Array, Array, Duals],
    ) -> tuple[Array, Array, Array, Array, Array, Duals]:
        trial, lr, _, _, _, _ = state
        if blocks:
            flat, trial_duals = _dykstra(
                (us - lr * grad).astype(us.dtype).ravel(),
                u_lo.ravel(),
                u_hi.ravel(),
                blocks,
                duals,
            )
            candidate = flat.reshape(us.shape)
        else:
            # `.astype(us.dtype)` is load-bearing, not defensive. The actions carry the working
            # precision, but the gradient does not: `control_gradient_adjoint` differentiates a
            # cost whose Q/R/Qf and x_target are whatever `jnp.array` produced, which under
            # `jax_enable_x64` is float64 even when the actions are float32. The subtraction then
            # promotes, the candidate re-enters the carry one dtype wider than it left, and
            # `lax.while_loop` rejects the body outright. Casting here -- at the point that decides
            # what the carry holds -- keeps the whole descent in the caller's precision.
            candidate = jnp.clip(us - lr * grad, u_lo, u_hi).astype(us.dtype)
            trial_duals = duals
        value = value_of(candidate)
        return trial + 1, lr * 0.5, candidate, value, value < current - tol, trial_duals

    _, _, candidate, value, accepted, trial_duals = jax.lax.while_loop(
        searching,
        halve,
        (jnp.asarray(0), jnp.asarray(lr0, dtype=us.dtype), us, current, jnp.asarray(False), duals),
    )
    kept = jax.tree.map(lambda new, old: jnp.where(accepted, new, old), trial_duals, duals)
    return jnp.where(accepted, candidate, us), jnp.where(accepted, value, current), accepted, kept


@eqx.filter_jit
def _projected_gradient_loop(
    dyn: Dynamics,
    x0: Array,
    us0: Array,
    dt: float,
    cost: QuadraticCost,
    u_lo: Array,
    u_hi: Array,
    steps: int,
    lr0: float,
    tol: float,
    blocks: Blocks = (),
) -> tuple[Array, Array, Array]:
    """The whole descent as one XLA program, outer ``while_loop`` and inner ``while_loop``.

    Module level rather than a closure inside the caller, because ``filter_jit`` caches on the
    wrapped function object: a wrapper rebuilt per call is recompiled per call and never amortises.

    The outer loop is a ``while_loop``, not a ``scan``, so ``steps`` is a *cap* rather than a bill:
    the descent stops the moment the line search fails, exactly where the Python ``break`` did, and
    an unused step costs nothing at all. That is what lets the default budget be loose enough for
    the stopping rule -- not the caller's guess -- to decide when the solve is finished. The cost
    history is written into a preallocated buffer, so a real early exit keeps the ``1 + accepted
    steps`` return shape.

    With constraint ``blocks`` the initial guess and every trial step are projected onto
    ``box ∩ constraints`` instead of clipped, and Dykstra's duals ride in the carry so each
    projection starts from the last accepted one's; without them the program is the box-only one.
    """
    duals = _no_duals(us0.size, blocks, us0.dtype)
    if blocks:
        flat, duals = _dykstra(us0.ravel(), u_lo.ravel(), u_hi.ravel(), blocks, duals)
        us = flat.reshape(us0.shape)
    else:
        us = project_box(us0, u_lo, u_hi)
    initial = total_cost(dyn, x0, us, dt, cost)
    values = jnp.zeros((steps + 1,), dtype=initial.dtype).at[0].set(initial)

    def descending(carry: tuple[Array, Array, Array, Array, Array, Duals]) -> Array:
        taken, _, _, _, alive, _ = carry
        return jnp.logical_and(taken < steps, alive)

    def descend(
        carry: tuple[Array, Array, Array, Array, Array, Duals],
    ) -> tuple[Array, Array, Array, Array, Array, Duals]:
        taken, us, current, values, _, duals = carry
        grad = control_gradient_adjoint(dyn, x0, us, dt, cost)
        us, current, accepted, duals = _backtrack(
            us,
            current,
            grad,
            u_lo,
            u_hi,
            lr0,
            tol,
            lambda candidate: total_cost(dyn, x0, candidate, dt, cost),
            blocks,
            duals,
        )
        # On rejection ``taken`` does not advance and the write lands back on its own slot, so the
        # buffer holds exactly the accepted prefix whichever way the step went.
        taken = jnp.where(accepted, taken + 1, taken)
        return taken, us, current, values.at[taken].set(current), accepted, duals

    taken, optimised, _, values, _, _ = jax.lax.while_loop(
        descending, descend, (jnp.asarray(0), us, initial, values, jnp.asarray(True), duals)
    )
    return optimised, values, taken


SolverStatus = Literal["converged", "max_iterations", "no_progress"]
"""Why the descent stopped -- reported, because "it stopped" and "it arrived" are different claims.

``converged`` is the method's own stopping rule: the backtracking line search could not find a step
that lowered the cost by more than ``tol``. ``max_iterations`` means the budget ran out first, so
the answer is wherever the descent happened to be. ``no_progress`` means not one step was accepted,
so the result *is* the caller's initial guess -- which is either an already-optimal guess or a badly
scaled problem, and :attr:`SolverResult.stationarity` is what tells the two apart.

The status deliberately does not claim optimality. A stalled line search is a statement about steps,
not about gradients, and on an ill-scaled instance the two come apart; the stationarity residual is
returned beside it so the caller judges rather than trusts a label.
"""


@dataclass(frozen=True)
class SolverResult:
    """A finished box-constrained solve, with the evidence for how far to trust it."""

    actions: Array
    cost_history: Array  # task cost per accepted step, length 1 + iterations
    status: SolverStatus
    iterations: int  # accepted steps, not gradient evaluations
    stationarity: float  # ||u - P(u - grad J)||, P onto the feasible set; zero exactly at KKT
    # Worst excess of a linear-constraint row, in its own units; 0.0 without constraints. The box is
    # exact by construction, the rows only to the projection's tolerance -- this says how close.
    constraint_violation: float = 0.0


def _status(iterations: int, steps: int) -> SolverStatus:
    if iterations == 0:
        return "no_progress"
    return "max_iterations" if iterations >= steps else "converged"


def _history(values: Array, taken: int) -> Array:
    """The accepted prefix of the cost buffer, as the default float dtype, converted on the host.

    ``jnp.asarray`` of a Python list compiles a conversion for every new length, and the length is
    the step count, so a replanning loop compiled one per replan.
    """
    prefix = np.asarray(values)[: taken + 1]
    return jax.device_put(prefix.astype(jax.dtypes.canonicalize_dtype(np.float64)))


def projected_gradient_solve(
    dyn: Dynamics,
    x0: Array,
    us0: Array,
    dt: float,
    cost: QuadraticCost,
    u_lo: Bound,
    u_hi: Bound,
    steps: int = 10_000,
    lr0: float = 0.2,
    tol: float = 1e-9,
    *,
    constraints: Sequence[LinearConstraint] = (),
) -> SolverResult:
    """:func:`projected_gradient_control`, returning why it stopped as well as where.

    Same solve, same numbers; the tuple-returning function below is a thin wrapper on this one.
    It exists because the compact form cannot distinguish a descent that reached its stopping rule
    from one that ran out of budget, and a planner that acts on the second without knowing is
    acting on an unfinished solve. Costs one extra gradient evaluation, for the stationarity
    residual -- negligible against the thousands the descent already spent.

    ``constraints`` add linear rows over the whole sequence (:class:`LinearConstraint`); the
    stationarity residual then projects onto the box *and* the rows, and
    :attr:`SolverResult.constraint_violation` reports how closely the rows hold.
    """
    lo = broadcast_box(u_lo, us0.shape, "u_lo", us0.dtype)
    hi = broadcast_box(u_hi, us0.shape, "u_hi", us0.dtype)
    check_box(lo, hi)
    blocks = _constraint_blocks(constraints, lo, hi, us0.dtype)
    optimised, values, taken = _projected_gradient_loop(
        dyn, x0, us0, dt, cost, lo, hi, steps, lr0, tol, blocks
    )
    iterations = int(taken)
    return SolverResult(
        actions=optimised,
        cost_history=_history(values, iterations),
        status=_status(iterations, steps),
        iterations=iterations,
        stationarity=(
            _polytope_stationarity(
                optimised, control_gradient_adjoint(dyn, x0, optimised, dt, cost), lo, hi, blocks
            )
            if blocks
            else box_stationarity(dyn, x0, optimised, dt, cost, lo, hi)
        ),
        constraint_violation=_violation(optimised, constraints),
    )


def projected_gradient_control(
    dyn: Dynamics,
    x0: Array,
    us0: Array,
    dt: float,
    cost: QuadraticCost,
    u_lo: Bound,
    u_hi: Bound,
    steps: int = 10_000,
    lr0: float = 0.2,
    tol: float = 1e-9,
    *,
    constraints: Sequence[LinearConstraint] = (),
) -> tuple[Array, Array]:
    """Minimise ``J`` over the control sequence subject to box constraints.

    Uses the discrete adjoint (:func:`chc.adjoint.control_gradient_adjoint`) for the gradient and
    backtracking line search, so every accepted step strictly decreases the cost. Returns the
    optimised controls and the cost history (length ``1 + accepted steps``).

    The descent runs inside a single compiled program (:func:`_projected_gradient_scan`); only the
    trim to the accepted prefix happens on the host, so the return shape stays what a Python loop
    would have produced.

    ``u_lo`` and ``u_hi`` are a scalar shared by every lever, a per-lever ``(m,)`` array, or a full
    ``(horizon, m)`` schedule. They are normalised to the action shape here and enter the compiled
    program as *arrays*, so a caller sweeping boxes compiles once rather than once per box value.
    ``dt``, ``steps`` and the line-search scalars stay static to the compilation, the same
    convention :func:`chc.cost.total_cost` already uses for ``dt``.

    ``constraints`` intersect the box with linear rows over the whole sequence
    (:class:`LinearConstraint`), and every iterate stays inside both.

    Returns only where the descent landed. :func:`projected_gradient_solve` returns *why it
    stopped* as well, which is what a caller needs before acting on the answer.
    """
    lo = broadcast_box(u_lo, us0.shape, "u_lo", us0.dtype)
    hi = broadcast_box(u_hi, us0.shape, "u_hi", us0.dtype)
    check_box(lo, hi)
    blocks = _constraint_blocks(constraints, lo, hi, us0.dtype)
    optimised, values, taken = _projected_gradient_loop(
        dyn, x0, us0, dt, cost, lo, hi, steps, lr0, tol, blocks
    )
    return optimised, _history(values, int(taken))


def lbfgs_box_control(
    dyn: Dynamics,
    x0: Array,
    us0: Array,
    dt: float,
    cost: QuadraticCost,
    u_lo: Bound,
    u_hi: Bound,
    steps: int = 300,
) -> tuple[Array, Array]:
    """Minimise ``J`` over the control sequence subject to box constraints, by L-BFGS-B.

    Same objective, same discrete-adjoint gradient and the same ``(controls, cost history)`` return
    as :func:`projected_gradient_control` -- a drop-in whose step is curved by a limited-memory
    secant approximation instead of scaled by a backtracked constant. ``steps`` caps L-BFGS-B
    iterations, not gradient evaluations, and it converges well inside the default on the instances
    :func:`nlp_solver_certificate` sweeps.

    SciPy is the trust boundary: the objective and gradient cross it as float64 NumPy and the answer
    is cast back to ``us0``'s dtype, so a float32 caller is not silently promoted.
    """
    shape = us0.shape
    lo = broadcast_box(u_lo, shape, "u_lo", us0.dtype)
    hi = broadcast_box(u_hi, shape, "u_hi", us0.dtype)
    check_box(lo, hi)
    history = [float(total_cost(dyn, x0, project_box(us0, lo, hi), dt, cost))]
    seen: dict[bytes, float] = {}

    def objective(flat: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        us = jnp.asarray(flat, dtype=us0.dtype).reshape(shape)
        value = float(total_cost(dyn, x0, us, dt, cost))
        seen[flat.tobytes()] = value
        gradient = control_gradient_adjoint(dyn, x0, us, dt, cost)
        return value, np.asarray(gradient, dtype=np.float64).ravel()

    def record(xk: NDArray[np.float64]) -> None:
        history.append(seen.get(xk.tobytes(), history[-1]))

    result = minimize(
        objective,
        np.asarray(project_box(us0, lo, hi), dtype=np.float64).ravel(),
        jac=True,
        method="L-BFGS-B",
        # Per element, not per problem: L-BFGS-B keeps one bound pair per coordinate, which is
        # what makes a per-lever box expressible here at all.
        bounds=list(
            zip(
                np.asarray(lo, dtype=np.float64).ravel(),
                np.asarray(hi, dtype=np.float64).ravel(),
                strict=True,
            )
        ),
        callback=record,
        options={"maxiter": steps},
    )
    optimised = jnp.asarray(result.x, dtype=us0.dtype).reshape(shape)
    return project_box(optimised, lo, hi), jnp.asarray(history)


def box_stationarity(
    dyn: Dynamics, x0: Array, us: Array, dt: float, cost: QuadraticCost, u_lo: Bound, u_hi: Bound
) -> float:
    """First-order optimality residual ``||u - P_box(u - grad J)||`` -- zero exactly at a KKT point.

    The convergence measure that does not depend on knowing the optimum: it is the natural
    stationarity map of a box-constrained problem, so it separates "the solver stopped" from "the
    solver arrived" without a reference solution and without a wall clock.
    """
    gradient = control_gradient_adjoint(dyn, x0, us, dt, cost)
    return float(jnp.linalg.norm(us - project_box(us - gradient, u_lo, u_hi)))


@dataclass(frozen=True)
class SolverComparison:
    """One instance of the box-constrained OC problem solved both ways."""

    label: str
    control_weight: float
    projected_gradient_cost: float
    lbfgs_cost: float
    projected_gradient_stationarity: float
    lbfgs_stationarity: float
    relative_gap: float


@dataclass(frozen=True)
class NLPSolverCertificate:
    """Evidence for where the first-order solver suffices and where it stops short."""

    comparisons: tuple[SolverComparison, ...]
    worst_relative_gap: float
    best_relative_gap: float
    worst_lbfgs_stationarity: float
    least_stationarity_ratio: float  # min over instances of PG stationarity / L-BFGS-B's
    box_respected: bool
    ok: bool


def nlp_solver_certificate(
    horizon: int = 40, dt: float = 0.1, pg_steps: int = 150, seed: int = 3
) -> NLPSolverCertificate:
    """Sweep the control weight and confirm both halves of the claim about the first-order solver.

    The instances differ only in ``R``, which sets the conditioning of the reduced Hessian: a large
    control weight makes the objective well conditioned and steepest descent adequate, a small one
    makes it ill conditioned and a short step budget nowhere near enough. Asserting *both* is the
    point -- the certificate fails if the well-conditioned instance develops a gap (the comparison
    would be measuring something other than conditioning) and equally if the ill-conditioned one
    stops showing one (the reframing would no longer be justified).

    ``pg_steps`` is deliberately far below the shipped default: the effect being exhibited is what
    a *short* first-order budget costs on an ill-conditioned instance, and at the shipped cap the
    projected gradient runs to its own stopping rule and the gap closes to a few hundredths of a
    percent. This measures the conditioning, not the library's behaviour.
    """
    x0 = jnp.array([1.0, 0.0])
    u_lo, u_hi = -5.0, 5.0
    known = DampedOscillator(omega=1.0, zeta=0.1)
    instances = (
        ("ill-conditioned, linear", 0.001, ZeroResidual(out_dim=2)),
        (
            "intermediate, learned residual",
            0.01,
            MLPResidual(2, 1, 2, 16, 2, key=jax.random.key(seed)),
        ),
        (
            "well-conditioned, learned residual",
            0.1,
            MLPResidual(2, 1, 2, 16, 2, key=jax.random.key(seed + 8)),
        ),
    )

    comparisons: list[SolverComparison] = []
    box_ok = True
    for label, control_weight, residual in instances:
        cost = QuadraticCost(
            Q=jnp.diag(jnp.array([1.0, 0.0])),
            R=jnp.array([[control_weight]]),
            Qf=jnp.diag(jnp.array([10.0, 1.0])),
            x_target=jnp.zeros(2),
        )
        dyn = HybridDynamics(known=known, residual=residual)
        us0 = jnp.zeros((horizon, 1))
        pg_us, pg_history = projected_gradient_control(
            dyn, x0, us0, dt, cost, u_lo, u_hi, steps=pg_steps
        )
        qn_us, qn_history = lbfgs_box_control(dyn, x0, us0, dt, cost, u_lo, u_hi)
        box_ok = box_ok and bool((jnp.abs(qn_us) <= u_hi + 1e-9).all())
        pg_cost, qn_cost = float(pg_history[-1]), float(qn_history[-1])
        comparisons.append(
            SolverComparison(
                label=label,
                control_weight=control_weight,
                projected_gradient_cost=pg_cost,
                lbfgs_cost=qn_cost,
                projected_gradient_stationarity=box_stationarity(
                    dyn, x0, pg_us, dt, cost, u_lo, u_hi
                ),
                lbfgs_stationarity=box_stationarity(dyn, x0, qn_us, dt, cost, u_lo, u_hi),
                relative_gap=(pg_cost - qn_cost) / abs(qn_cost),
            )
        )

    gaps = [comparison.relative_gap for comparison in comparisons]
    ratios = [
        comparison.projected_gradient_stationarity / comparison.lbfgs_stationarity
        for comparison in comparisons
    ]
    return NLPSolverCertificate(
        comparisons=tuple(comparisons),
        worst_relative_gap=max(gaps),
        best_relative_gap=min(gaps),
        worst_lbfgs_stationarity=max(c.lbfgs_stationarity for c in comparisons),
        least_stationarity_ratio=min(ratios),
        box_respected=box_ok,
        ok=(
            box_ok
            # Comparative, not absolute: the floor on a stationarity residual is set by the working
            # dtype, so "< 1e-3" would pass at float64 and fail at float32 for no reason but the
            # arithmetic. The claim is that one solver arrives and the other stops short.
            and min(ratios) > 10.0
            and min(gaps) > -1e-9  # and the quasi-Newton arm is never worse
            and max(gaps) > 0.05  # ill-conditioned: the first-order budget is not enough
            and min(gaps) < 0.005  # well-conditioned: it is, so the gap is about conditioning
        ),
    )
