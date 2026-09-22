"""The numbers behind docs/adr/0001-linear-action-constraints.md. Counts only, no wall time.

Gradient and cost evaluations, Dykstra sweeps and active-set solves are deterministic, so they are
exempt from the state of the machine that runs them; a timing would not be, and none is taken.

    instances   the shipped projected gradient (D) against a PHR augmented Lagrangian with the box
                kept in its inner projected-gradient solve (A), on three plans where a rate limit
                and a budget bind, measured against an exact or best-of-two reference optimum.
    polytopes   the shipped projection against the exact least-distance one, on 3000 random
                polytopes with an interior.

Arm D's counts come from a copy of the projection and the descent with counters threaded through;
the copy's actions are asserted equal to the shipped solver's, so a copy that drifts from the
library stops the run instead of reporting on an algorithm that no longer ships. The polish is
counted from its own source with only its step count exposed.

Run: uv run python scripts/bench_linear_constraints.py {instances,polytopes} > out.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import LinearConstraint as ScipyLinearConstraint
from scipy.optimize import minimize, nnls

jax.config.update("jax_enable_x64", True)

import chc.control as control  # noqa: E402
from chc import (  # noqa: E402
    DampedOscillator,
    LinearConstraint,
    LinearDynamics,
    MarketingMixSystem,
    QuadraticCost,
    projected_gradient_solve,
)
from chc.adjoint import control_gradient_adjoint  # noqa: E402
from chc.cost import total_cost  # noqa: E402

STEPS, LR0, TOL = 10_000, 0.2, 1e-9  # projected_gradient_solve's defaults, used by both arms


# ── exact oracles (host numpy; share nothing with the solver) ─────────────────────────────────


def least_distance(e: np.ndarray, f: np.ndarray) -> np.ndarray:
    """``argmin |w|`` subject to ``e @ w >= f`` (Lawson & Hanson 1974, ch. 23), by one NNLS."""
    n = e.shape[1]
    stacked = np.vstack([e.T, f[None, :]])
    target = np.zeros(n + 1)
    target[-1] = 1.0
    weights, _ = nnls(stacked, target, maxiter=100 * max(e.shape))
    residual = stacked @ weights - target
    if abs(residual[-1]) < 1e-14:
        raise RuntimeError("the least-distance program is infeasible")
    return -residual[:n] / residual[-1]


def rows_of(lo, hi, a, lower, upper) -> tuple[np.ndarray, np.ndarray]:
    """``box ∩ {lower <= a u <= upper}`` as ``G u <= h``, keeping only the sides present."""
    n = a.shape[1]
    has_upper, has_lower = np.isfinite(upper), np.isfinite(lower)
    g = np.vstack([a[has_upper], -a[has_lower], np.eye(n), -np.eye(n)])
    return g, np.concatenate([upper[has_upper], -lower[has_lower], hi, -lo])


def project_exact(y, lo, hi, a, lower, upper) -> np.ndarray:
    g, h = rows_of(lo, hi, a, lower, upper)
    return y + least_distance(-g, g @ y - h)


def violation(u, lo, hi, a, lower, upper) -> float:
    level = a @ u
    excess = [lower - level, level - upper, lo - u, u - hi]
    return float(max(np.max(e, initial=0.0) for e in excess))


# ── instances ─────────────────────────────────────────────────────────────────────────────────


def oscillator():
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.0])),
        R=jnp.array([[0.01]]),
        Qf=jnp.diag(jnp.array([10.0, 1.0])),
        x_target=jnp.zeros(2),
    )
    dyn = DampedOscillator(omega=1.0, zeta=0.1)
    return "oscillator (H=40, m=1)", dyn, jnp.array([1.0, 0.0]), 40, 1, 0.1, cost, -5.0, 5.0, True


def two_lever():
    dyn = LinearDynamics(
        jnp.array([[0.0, 1.0, 0.0], [-1.0, -0.2, 0.5], [0.0, 0.0, -0.3]]),
        jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    )
    cost = QuadraticCost(
        Q=jnp.diag(jnp.array([1.0, 0.1, 0.1])),
        R=0.02 * jnp.eye(2),
        Qf=jnp.diag(jnp.array([10.0, 1.0, 1.0])),
        x_target=jnp.zeros(3),
    )
    return "linear (H=30, m=2)", dyn, jnp.array([1.5, 0.0, 1.0]), 30, 2, 0.1, cost, -3.0, 3.0, True


def marketing_mix():
    system = MarketingMixSystem()
    c = len(system.channels)
    q = jnp.diag(jnp.array([1.0] + [0.0] * c))
    cost = QuadraticCost(Q=q, R=0.08 * jnp.eye(c), Qf=q, x_target=jnp.array([8.0] + [0.0] * c))
    x0 = jnp.array([system.base] + [0.0] * c)
    lo, hi = system.spend_floor, system.spend_ceiling
    return (
        "marketing mix (H=12, m=3)",
        system.plant(season=0.0),
        x0,
        12,
        c,
        1.0,
        cost,
        lo,
        hi,
        False,
    )


def reference(dyn, x0, horizon, m, dt, cost, lo, hi, a, lower, upper, linear) -> np.ndarray:
    """The optimum: exact for a linear plant (a QP), else the better of SLSQP and trust-constr."""
    n = horizon * m

    def value(flat):
        return total_cost(dyn, x0, flat.reshape(horizon, m), dt, cost)

    if linear:  # RK4 on a linear plant is linear in the actions, so J is exactly quadratic
        hessian = np.asarray(jax.hessian(value)(jnp.zeros(n)))
        hessian = 0.5 * (hessian + hessian.T)
        free = -np.linalg.solve(hessian, np.asarray(jax.grad(value)(jnp.zeros(n))))
        back = np.linalg.inv(np.linalg.cholesky(hessian)).T  # u = free + back @ w, |w|^2 = 2J + c
        g, h = rows_of(lo, hi, a, lower, upper)
        return free + back @ least_distance(-g @ back, g @ free - h)
    fun, grad = jax.jit(value), jax.jit(jax.grad(value))
    any_bound = bool(np.isfinite(np.concatenate([lower, upper])).any())
    best = None
    for method, options in (
        ("SLSQP", {"ftol": 1e-15, "maxiter": 5000}),
        ("trust-constr", {"gtol": 1e-12, "xtol": 1e-14, "maxiter": 20000}),
    ):
        result = minimize(
            lambda u: float(fun(jnp.asarray(u))),
            np.clip(np.zeros(n), lo, hi),
            jac=lambda u: np.asarray(grad(jnp.asarray(u))),
            method=method,
            bounds=list(zip(lo, hi, strict=True)),
            constraints=[ScipyLinearConstraint(a, lower, upper)] if any_bound else [],
            options=options,
        )
        u = project_exact(result.x, lo, hi, a, lower, upper)
        if best is None or float(fun(jnp.asarray(u))) < float(fun(jnp.asarray(best))):
            best = u
    assert best is not None
    return best


# ── arm A: PHR augmented Lagrangian, box kept in the inner projected gradient ───────────────


@eqx.filter_jit
def _alm_inner(dyn, x0, us0, dt, cost, lo, hi, g, h, lam, rho):
    shape = us0.shape

    def value(flat):
        shifted = jnp.maximum(0.0, lam + rho * (g @ flat - h))
        penalty = (jnp.sum(shifted**2) - jnp.sum(lam**2)) / (2.0 * rho)
        return total_cost(dyn, x0, flat.reshape(shape), dt, cost) + penalty

    def gradient(flat):
        shifted = jnp.maximum(0.0, lam + rho * (g @ flat - h))
        plant = control_gradient_adjoint(dyn, x0, flat.reshape(shape), dt, cost).ravel()
        return plant + g.T @ shifted

    u = jnp.clip(us0.ravel(), lo, hi)

    def backtrack(u, current, direction):
        def searching(s):
            return jnp.logical_and(s[0] < control._MAX_BACKTRACK, jnp.logical_not(s[4]))

        def halve(s):
            trial, lr, _, _, _ = s
            candidate = jnp.clip(u - lr * direction, lo, hi)
            v = value(candidate)
            return trial + 1, lr * 0.5, candidate, v, v < current - TOL

        start = (jnp.asarray(0), jnp.asarray(LR0), u, current, jnp.asarray(False))
        trial, _, candidate, v, ok = jax.lax.while_loop(searching, halve, start)
        return jnp.where(ok, candidate, u), jnp.where(ok, v, current), ok, trial

    def descend(c):
        taken, u, current, _, grads, evals = c
        u, current, ok, trials = backtrack(u, current, gradient(u))
        return jnp.where(ok, taken + 1, taken), u, current, ok, grads + 1, evals + trials

    zero = jnp.asarray(0)
    start = (zero, u, value(u), jnp.asarray(True), zero, jnp.asarray(1))
    _, u, _, _, grads, evals = jax.lax.while_loop(
        lambda c: jnp.logical_and(c[0] < STEPS, c[3]), descend, start
    )
    return u.reshape(shape), grads, evals


def augmented_lagrangian(dyn, x0, us0, dt, cost, lo, hi, g, h, rho=10.0, outers=40, feas=1e-10):
    """Powell-Hestenes-Rockafellar: multiplier update, rho x10 when violation falls by < 4x."""
    lam, u, grads, evals, previous = jnp.zeros(g.shape[0]), us0, 0, 0, np.inf
    for outer in range(1, outers + 1):
        u, used_g, used_e = _alm_inner(dyn, x0, u, dt, cost, lo, hi, g, h, lam, jnp.asarray(rho))
        grads, evals = grads + int(used_g), evals + int(used_e)
        gap = g @ u.ravel() - h
        worst = float(jnp.max(jnp.abs(jnp.maximum(gap, -lam / rho))))
        lam = jnp.maximum(0.0, lam + rho * gap)
        if worst <= feas:
            return u, grads, evals, outer
        if worst > 0.25 * previous:
            rho *= 10.0
        previous = worst
    return u, grads, evals, outers


# ── arm D: the shipped projection, counted ────────────────────────────────────────────────────


def _counted_polish_from_source() -> Callable:
    source = inspect.getsource(control._polish)
    for old, new in (
        ("def _polish(", "def counted_polish("),
        (
            "_, box_side, nu, _, _ = jax.lax.while_loop(",
            "_, box_side, nu, used, _ = jax.lax.while_loop(",
        ),
        (
            "return kkt, x, (pull - x, tuple(jnp.split(nu, sizes)))",
            "return kkt, x, (pull - x, tuple(jnp.split(nu, sizes))), used",
        ),
    ):
        assert source.count(old) == 1, f"control._polish changed shape near {old!r}"
        source = source.replace(old, new)
    namespace = dict(vars(control))
    exec(compile(source, "counted_polish", "exec"), namespace)
    return namespace["counted_polish"]


counted_polish = _counted_polish_from_source()


def counted_projection(y, lo, hi, blocks, duals):
    """``control._dykstra`` with three counters: sweeps, polish attempts, active-set solves."""
    box_increment, multipliers = duals
    x = y - box_increment
    for (rows, *_), nu in zip(blocks, multipliers, strict=True):
        x = x - rows.T @ nu
    tolerance = 256 * jnp.finfo(y.dtype).eps * (1 + jnp.max(jnp.abs(y)))

    def polished(s):
        _, box_increment, multipliers, count, change, attempts, solves = s
        kkt, x, new_duals, used = counted_polish(
            y, lo, hi, blocks, (box_increment, multipliers), tolerance
        )
        exact = (x, *new_duals, count, jnp.zeros_like(change))
        kept = jax.tree.map(lambda new, old: jnp.where(kkt, new, old), exact, s[:5])
        return (*kept, attempts + 1, solves + used)

    def sweep(s):
        x, box_increment, multipliers, count, _, attempts, solves = s
        change = jnp.zeros((), dtype=x.dtype)
        updated = []
        for (rows, lower, upper, norms), nu in zip(blocks, multipliers, strict=True):
            level = rows @ x + nu * norms
            renewed = (level - jnp.clip(level, lower, upper)) / norms
            x = x + rows.T @ (nu - renewed)
            change = jnp.maximum(change, jnp.max(jnp.abs(renewed - nu) * jnp.sqrt(norms)))
            updated.append(renewed)
        shifted = x + box_increment
        x = jnp.clip(shifted, lo, hi)
        change = jnp.maximum(change, jnp.max(jnp.abs(shifted - x - box_increment)))
        state = (x, shifted - x, tuple(updated), count + 1, change, attempts, solves)
        return jax.lax.cond(
            (count + 1) % control._POLISH_EVERY == 0, polished, lambda kept: kept, state
        )

    def unsettled(s):
        return jnp.logical_and(s[3] < control._MAX_SWEEPS, s[4] > tolerance)

    zero = jnp.asarray(0)
    start = (x, box_increment, multipliers, zero, jnp.asarray(jnp.inf, y.dtype), zero, zero)
    x, box_increment, multipliers, count, _, attempts, solves = jax.lax.while_loop(
        unsettled, sweep, polished(start)
    )
    return x, (box_increment, multipliers), jnp.stack([count, attempts, solves])


@eqx.filter_jit
def counted_descent(dyn, x0, us0, dt, cost, lo, hi, blocks):
    """The constrained branch of ``control._projected_gradient_loop``, counting its work."""
    shape = us0.shape
    flat, duals, work = counted_projection(
        us0.ravel(), lo.ravel(), hi.ravel(), blocks, control._no_duals(us0.size, blocks, us0.dtype)
    )
    us = flat.reshape(shape)

    def backtrack(us, current, direction, duals):
        def searching(s):
            return jnp.logical_and(s[0] < control._MAX_BACKTRACK, jnp.logical_not(s[4]))

        def halve(s):
            trial, lr, _, _, _, _, work = s
            stepped = (us - lr * direction).astype(us.dtype).ravel()
            flat, trial_duals, used = counted_projection(
                stepped, lo.ravel(), hi.ravel(), blocks, duals
            )
            candidate = flat.reshape(shape)
            v = total_cost(dyn, x0, candidate, dt, cost)
            return trial + 1, lr * 0.5, candidate, v, v < current - TOL, trial_duals, work + used

        start = (
            jnp.asarray(0),
            jnp.asarray(LR0, us.dtype),
            us,
            current,
            jnp.asarray(False),
            duals,
            jnp.zeros(3, jnp.int32),
        )
        trial, _, candidate, v, ok, trial_duals, work = jax.lax.while_loop(searching, halve, start)
        kept = jax.tree.map(lambda new, old: jnp.where(ok, new, old), trial_duals, duals)
        return jnp.where(ok, candidate, us), jnp.where(ok, v, current), ok, kept, trial, work

    def descend(c):
        taken, us, current, _, duals, grads, evals, work = c
        direction = control_gradient_adjoint(dyn, x0, us, dt, cost)
        us, current, ok, duals, trials, used = backtrack(us, current, direction, duals)
        return (
            jnp.where(ok, taken + 1, taken),
            us,
            current,
            ok,
            duals,
            grads + 1,
            evals + trials,
            work + used,
        )

    start = (
        jnp.asarray(0),
        us,
        total_cost(dyn, x0, us, dt, cost),
        jnp.asarray(True),
        duals,
        jnp.asarray(0),
        jnp.asarray(1),
        work.astype(jnp.int32),
    )
    _, us, _, _, _, grads, evals, work = jax.lax.while_loop(
        lambda c: jnp.logical_and(c[0] < STEPS, c[3]), descend, start
    )
    return us, grads, evals, work


# ── the two parts ─────────────────────────────────────────────────────────────────────────────


def instances() -> list[dict]:
    out = []
    for build, budget_share in ((oscillator, 0.6), (two_lever, 0.6), (marketing_mix, 0.7)):
        label, dyn, x0, horizon, m, dt, cost, lo_s, hi_s, linear = build()
        n = horizon * m
        lo, hi = np.full(n, lo_s), np.full(n, hi_s)
        # Rate cap and budget sized off the box-only optimum, so that both bind.
        nothing = (np.zeros((1, n)), np.array([-np.inf]), np.array([np.inf]))
        box = reference(dyn, x0, horizon, m, dt, cost, lo, hi, *nothing, linear).reshape(horizon, m)
        rate = LinearConstraint.rate_limit(
            horizon, list(0.35 * np.max(np.abs(np.diff(box, axis=0)), 0))
        )
        # A signed budget on the total action, in the direction the box-only optimum spends.
        sign = float(np.sign(box.sum()) or 1.0) if linear else 1.0
        budget = LinearConstraint(sign * np.ones((1, n)), -np.inf, budget_share * abs(box.sum()))
        a = np.vstack([rate.matrix, budget.matrix])
        lower = np.concatenate([rate.lower, budget.lower])
        upper = np.concatenate([rate.upper, budget.upper])
        u_ref = reference(dyn, x0, horizon, m, dt, cost, lo, hi, a, lower, upper, linear)
        j_ref = float(total_cost(dyn, x0, jnp.asarray(u_ref).reshape(horizon, m), dt, cost))
        level = a @ u_ref
        binding = int(np.sum((level > upper - 1e-8) | (level < lower + 1e-8)))

        us0 = jnp.zeros((horizon, m))
        shipped = projected_gradient_solve(
            dyn, x0, us0, dt, cost, lo_s, hi_s, constraints=(rate, budget)
        )
        box_lo = control.broadcast_box(lo_s, us0.shape, "u_lo", us0.dtype)
        box_hi = control.broadcast_box(hi_s, us0.shape, "u_hi", us0.dtype)
        blocks = control._constraint_blocks((rate, budget), box_lo, box_hi, us0.dtype)
        us, grads, evals, work = counted_descent(dyn, x0, us0, dt, cost, box_lo, box_hi, blocks)
        assert np.array_equal(np.asarray(us), np.asarray(shipped.actions)), (
            "the counted copy drifted"
        )
        sweeps, attempts, solves = (int(w) for w in work)
        u_d = np.asarray(shipped.actions).ravel()
        j_d = float(total_cost(dyn, x0, shipped.actions, dt, cost))

        has_upper, has_lower = np.isfinite(upper), np.isfinite(lower)
        g = np.vstack([a[has_upper], -a[has_lower]])
        h = np.concatenate([upper[has_upper], -lower[has_lower]])
        u_a, grads_a, evals_a, outers = augmented_lagrangian(
            dyn, x0, us0, dt, cost, jnp.asarray(lo), jnp.asarray(hi), jnp.asarray(g), jnp.asarray(h)
        )
        u_a = np.asarray(u_a).ravel()
        u_a_feasible = project_exact(u_a, lo, hi, a, lower, upper)
        j_a = float(total_cost(dyn, x0, jnp.asarray(u_a_feasible).reshape(horizon, m), dt, cost))

        work_d = int(grads) + int(evals) + sweeps  # a sweep priced as one evaluation, as registered
        work_a = grads_a + evals_a
        out.append(
            {
                "instance": label,
                "constraint_rows": int(a.shape[0]),
                "binding_rows": binding,
                "J_reference": j_ref,
                "D": {
                    "relative_gap": (j_d - j_ref) / abs(j_ref),
                    "violation": violation(u_d, lo, hi, a, lower, upper),
                    "status": shipped.status,
                    "stationarity": shipped.stationarity,
                    "gradients": int(grads),
                    "evaluations": int(evals),
                    "sweeps": sweeps,
                    "polish_attempts": attempts,
                    "active_set_solves": solves,
                },
                "A": {
                    "relative_gap": (j_a - j_ref) / abs(j_ref),
                    "violation_as_returned": violation(u_a, lo, hi, a, lower, upper),
                    "outer_iterations": outers,
                    "gradients": grads_a,
                    "evaluations": evals_a,
                },
                "work_D_over_A_solve_as_one_evaluation": (work_d + solves) / work_a,
                "solve_price_at_which_A_breaks_even": (work_a - work_d) / max(solves, 1),
            }
        )
        print(json.dumps(out[-1]), file=sys.stderr, flush=True)
    return out


def polytopes(seeds: int = 750) -> dict:
    # Both compiled: an eager run of the same algorithm can round differently from a compiled one.
    project, counted = eqx.filter_jit(control._dykstra), eqx.filter_jit(counted_projection)
    worst, capped, sweeps_needed = 0.0, 0, []
    for rows, size in ((3, 4), (6, 5), (9, 6), (12, 8)):
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            a = rng.normal(size=(rows, size))
            level = a @ rng.uniform(-0.5, 0.5, size)
            lower = np.where(rng.random(rows) < 0.3, -np.inf, level - rng.uniform(0.05, 1.0, rows))
            upper = np.where(rng.random(rows) < 0.3, np.inf, level + rng.uniform(0.05, 1.0, rows))
            lo, hi, y = -np.ones(size), np.ones(size), rng.normal(0.0, 3.0, size)
            blocks = control._constraint_blocks(
                [LinearConstraint(a, lower, upper)], jnp.asarray(lo), jnp.asarray(hi), jnp.float64
            )
            zero = control._no_duals(size, blocks, jnp.float64)
            args = (jnp.asarray(y), jnp.asarray(lo), jnp.asarray(hi), blocks, zero)
            x, _ = project(*args)
            x_counted, _, work = counted(*args)
            assert np.array_equal(np.asarray(x), np.asarray(x_counted)), "the counted copy drifted"
            worst = max(
                worst,
                float(np.max(np.abs(np.asarray(x) - project_exact(y, lo, hi, a, lower, upper)))),
            )
            sweeps_needed.append(int(work[0]))
            capped += int(work[0]) >= control._MAX_SWEEPS
    counts = np.array(sweeps_needed)
    return {
        "instances": int(counts.size),
        "worst_distance_to_exact_projection": worst,
        "ran_out_of_sweeps": capped,
        "sweeps": {q: float(np.percentile(counts, p)) for q, p in (("median", 50), ("p99", 99))}
        | {"max": int(counts.max())},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=("instances", "polytopes"))
    part = parser.parse_args().part
    json.dump(instances() if part == "instances" else polytopes(), sys.stdout, indent=1)
    print()
