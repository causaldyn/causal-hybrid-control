"""Model predictive control: receding-horizon optimal control with warm starts.

At each step MPC solves a finite-horizon OC problem from the current measured state (against the
*model*), applies only the first control, advances the *plant*, and re-solves. Planning and
reality are separate objects (``model`` vs ``plant``), so the offline/confounded setting — plan
with the learned hybrid model, act on the true system — needs no change to this loop.

:func:`mpc_control` runs that loop against a simulated plant. :class:`RecedingHorizon` is its
online form -- the caller owns the plant and the clock -- and each of its steps is a whole
:func:`chc.plan.causal_plan`, certificate included.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from chc.control import Bound, LinearConstraint, projected_gradient_control
from chc.cost import QuadraticCost
from chc.dynamics import Dynamics
from chc.integrate import rk4_step
from chc.plan import BarrierConstraint, CausalPlan, _plan
from chc.support import PenaltyModel, SupportModel


def mpc_control(
    model: Dynamics,
    x0: Array,
    cost: QuadraticCost,
    dt: float,
    horizon: int,
    u_lo: Bound,
    u_hi: Bound,
    n_steps: int,
    *,
    plant: Dynamics | None = None,
    inner_steps: int = 40,
    warm_start: bool = True,
) -> tuple[Array, Array]:
    """Run closed-loop MPC for ``n_steps``; return the realised trajectory and applied controls.

    Unlike the one-shot solvers, ``inner_steps`` here is a **per-decision latency budget** and is
    deliberately far below their default: a warm start hands each replan a nearly-optimal iterate,
    so the truncation buys latency rather than hiding an unconverged answer. Priced rather than
    assumed -- against the same loop run to convergence over 25 replans, 40 steps costs 0.3% of
    closed-loop cost on a known plant and 0.4% with an MLP residual, for 1.5x and 2.4x less time.
    Raise it if the loop is offline; the solve stops on its own once the line search fails.

    Args:
        model: dynamics used for planning (the controller's belief).
        plant: true dynamics the control is applied to (defaults to ``model``).

    Returns:
        ``(xs, us)`` with ``xs`` of shape ``(n_steps + 1, n)`` and ``us`` of shape ``(n_steps, m)``.
    """
    plant = model if plant is None else plant
    control_dim = cost.R.shape[0]
    guess = jnp.zeros((horizon, control_dim))
    x = x0
    states = [x0]
    applied: list[Array] = []

    for _ in range(n_steps):
        us_opt, _ = projected_gradient_control(
            model, x, guess, dt, cost, u_lo, u_hi, steps=inner_steps
        )
        u0 = us_opt[0]
        applied.append(u0)
        x = rk4_step(plant, 0.0, x, u0, dt)
        states.append(x)
        guess = (
            jnp.concatenate([us_opt[1:], us_opt[-1:]], axis=0)
            if warm_start
            else jnp.zeros((horizon, control_dim))
        )

    return jnp.stack(states), jnp.stack(applied)


def _shift(sequence: NDArray[np.float64]) -> NDArray[np.float64]:
    """One step on: drop the first entry, repeat the last."""
    return np.concatenate([sequence[1:], sequence[-1:]])


@dataclass(eq=False)
class RecedingHorizon:
    """:func:`chc.plan.causal_plan` from each measured state, warm-started from the last plan.

    ::

        controller = RecedingHorizon(model, cost, dt=0.1, horizon=20, u_lo=-5.0, u_hi=5.0,
                                     barrier=BarrierConstraint(h, alpha=2.0))
        while running:
            plan = controller.step(measure())
            apply(plan.actions[0])

    :meth:`step` returns the whole :class:`~chc.plan.CausalPlan`, so the certificate reaches the
    caller with the action it covers; read ``plan.safety`` and ``plan.solver_status`` before
    applying ``plan.actions[0]``. The fields are :func:`~chc.plan.causal_plan`'s arguments, with
    the same meaning and defaults.

    Each step starts the descent from the last plan shifted one step, its final action repeated,
    and a held barrier's rounds from the last multipliers shifted the same way, under a penalty
    chosen afresh. That moves where the solves start, not what they minimise.

    **Compiled once.** Each program compiles the first time a step needs it -- the barrier's rounds
    the first time the barrier binds -- and is reused after, so steps stop compiling. What compiles
    again is a new shape, dtype or Python function: build the barrier's function once, outside the
    loop, as this object does by holding it; a fresh ``lambda`` per step compiles the descent per
    step. Across processes, set ``jax_compilation_cache_dir`` and lower
    ``jax_persistent_cache_min_compile_time_secs`` to 0: at its default of one second JAX wrote
    none of a first step's programs on the tests' oscillator, and at 0 a second process loaded
    every one of them from disk.

    Each step is one :func:`~chc.plan.causal_plan`, so nothing outside the horizon carries over:
    ``constraints`` bind each horizon afresh -- a budget row caps every window, not the run's
    total, and a rate limit does not reach back to the action already applied -- and ``steps``
    caps each descent, of which a held barrier runs up to ``1 + _BARRIER_ROUNDS`` a step.

    One controller per control loop: :meth:`step` updates the warm start in place, unguarded. A
    new controller over the same model, cost and barrier is a cold start that reuses the compiled
    programs.
    """

    model: Dynamics
    cost: QuadraticCost
    dt: float
    horizon: int
    u_lo: Bound
    u_hi: Bound
    support: SupportModel | None = None
    lam_supp: float = 0.0
    uncertainty: PenaltyModel | None = None
    lam_unc: float = 0.0
    lipschitz: float = 0.0
    model_error: float = 0.0
    tolerance: float = float("inf")
    steps: int = 10_000
    constraints: Sequence[LinearConstraint] = ()
    barrier: BarrierConstraint | None = None
    _warm: tuple[NDArray[np.float64], NDArray[np.float64] | None] | None = field(
        default=None, init=False, repr=False
    )

    def step(self, x: Array) -> CausalPlan:
        """Plan from the measured state ``x``, and keep the plan as the next step's start."""
        actions, multipliers = (None, None) if self._warm is None else self._warm
        plan, multipliers = _plan(
            self.model,
            x,
            self.cost,
            self.dt,
            self.horizon,
            self.u_lo,
            self.u_hi,
            support=self.support,
            lam_supp=self.lam_supp,
            uncertainty=self.uncertainty,
            lam_unc=self.lam_unc,
            lipschitz=self.lipschitz,
            model_error=self.model_error,
            tolerance=self.tolerance,
            steps=self.steps,
            constraints=self.constraints,
            barrier=self.barrier,
            warm_start=actions,
            multipliers=multipliers,
        )
        self._warm = (
            _shift(np.asarray(plan.actions, dtype=np.float64)),
            None if multipliers is None else _shift(multipliers),
        )
        return plan
