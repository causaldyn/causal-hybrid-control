"""The Python side of the Milestone-J measurement: the same solve, timed the same two ways.

``golden`` prints the trajectory ``parity_check.py`` compares against the Rust binary; ``steady``
reports per-solve time after a warm-up solve, which is what a control loop pays; bare invocation
does one solve and exits, which is what ``hyperfine`` times as the deployable cold path.

Two Python arms, and what separates them has changed. When this harness was written
``chc.control.projected_gradient_control`` was a *Python* loop paying one dispatch per gradient and
one per backtracking trial, so ``steady`` measured Python orchestration rather than a runtime and
``steady-jit`` existed to supply the honest comparator. The library now compiles that recursion
itself, so both arms are single XLA programs and the remaining gap is the shipped path's host-side
work: it syncs the accepted-step count and copies the cost history back so the return shape stays
what the Python loop produced. ``steady-jit`` is the same solve without that contract.
"""

from __future__ import annotations

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from chc import QuadraticCost, projected_gradient_control  # noqa: E402
from chc.adjoint import control_gradient_adjoint  # noqa: E402
from chc.control import project_box  # noqa: E402
from chc.cost import total_cost  # noqa: E402
from chc.dynamics import LinearDynamics  # noqa: E402

A = jnp.array([[0.0, 1.0], [-1.0, -0.2]])
B = jnp.array([[0.0], [1.0]])
COST = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.1])),
    R=jnp.array([[0.05]]),
    Qf=jnp.diag(jnp.array([5.0, 1.0])),
    x_target=jnp.zeros(2),
)
X0 = jnp.array([1.0, 0.0])
US0 = jnp.zeros((30, 1))


def solve() -> tuple[jnp.ndarray, jnp.ndarray]:
    return projected_gradient_control(
        LinearDynamics(A, B), X0, US0, 0.1, COST, -5.0, 5.0, steps=400, lr0=0.2, tol=1e-9
    )


DT, LO, HI, TOL, LR0, OUTER, BACKTRACK = 0.1, -5.0, 5.0, 1e-9, 0.2, 400, 40


@jax.jit
def solve_compiled() -> tuple[jnp.ndarray, jnp.ndarray]:
    """The same recursion as one XLA program, returning the full history rather than its prefix.

    The Python original breaks out of the outer loop when no backtracked step improves. A fixed
    ``OUTER``-length scan is equivalent, not an approximation: failure is deterministic in ``us``,
    so once a step fails every later step from the same iterate fails identically. On this instance
    every step is accepted, so the two arms run the same 400 gradients and differ only in the trim.
    """
    dyn = LinearDynamics(A, B)

    def outer(carry: tuple[jnp.ndarray, jnp.ndarray], _: None) -> tuple[tuple, jnp.ndarray]:
        us, current = carry
        grad = control_gradient_adjoint(dyn, X0, us, DT, COST)

        def cond(state: tuple) -> jnp.ndarray:
            trial, _, _, accepted = state
            return jnp.logical_and(trial < BACKTRACK, jnp.logical_not(accepted))

        def body(state: tuple) -> tuple:
            trial, lr, _, _ = state
            candidate = project_box(us - lr * grad, LO, HI)
            value = total_cost(dyn, X0, candidate, DT, COST)
            return (trial + 1, lr * 0.5, candidate, value < current - TOL)

        seed = project_box(us - LR0 * grad, LO, HI)
        _, _, candidate, accepted = jax.lax.while_loop(cond, body, (0, LR0, seed, jnp.bool_(False)))
        value = total_cost(dyn, X0, candidate, DT, COST)
        us = jnp.where(accepted, candidate, us)
        current = jnp.where(accepted, value, current)
        return (us, current), current

    initial = total_cost(dyn, X0, US0, DT, COST)
    (us, _), history = jax.lax.scan(outer, (US0, initial), None, length=OUTER)
    return us, history


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "golden":
        us, history = solve()
        print(f"cost {float(history[-1]):.9f}")
        print(f"u0 {float(us[0, 0]):.9f}")
        print(f"accepted {len(history) - 1}")
    elif mode == "golden-jit":
        us, history = solve_compiled()
        print(f"cost {float(history[-1]):.9f}")
        print(f"u0 {float(us[0, 0]):.9f}")
    elif mode == "steady-jit":
        repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        jax.block_until_ready(solve_compiled())  # warm up: pay tracing and compilation once
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            jax.block_until_ready(solve_compiled())
            times.append(time.perf_counter() - start)
        times.sort()
        print(f"solves {repeats}")
        print(f"best_ms {times[0] * 1e3:.4f}")
        print(f"median_ms {times[repeats // 2] * 1e3:.4f}")
    elif mode == "steady":
        repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        solve()  # warm up: the first solve pays tracing and compilation
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            us, _ = solve()
            us.block_until_ready()
            times.append(time.perf_counter() - start)
        times.sort()
        print(f"solves {repeats}")
        print(f"best_ms {times[0] * 1e3:.4f}")
        print(f"median_ms {times[repeats // 2] * 1e3:.4f}")
    else:
        solve()


if __name__ == "__main__":
    main()
