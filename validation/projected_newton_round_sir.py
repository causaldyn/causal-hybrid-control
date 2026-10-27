"""Export the SIR round's Gauss-Newton matrix for projected_newton_round.mac, part (5).

The held plan of ADR 0144's epidemic (beta 0.6, gamma 0.1, 60 steps of 1, actions in [0, 0.9],
at most a tenth infected, alpha 0.5), solved by the diagonal rounds. At the plan they end on, the
round's Hessian (chc.plan._round_terms) over the actions off the box, in the variables the
diagonal rounds step in, scaled by the root of the Gauss-Newton diagonal (chc.plan._round_metric),
at the rounds' last penalty and at a hundredth and a tenth of it, with numpy's condition number
of each. It writes projected_newton_round_sir.dat to the working directory; run Maxima on
projected_newton_round.mac from there.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from chc import BarrierConstraint, QuadraticCost, causal_plan  # noqa: E402
from chc.plan import _round_terms  # noqa: E402
from chc.research.epidemic import SIRDynamics  # noqa: E402

model = SIRDynamics(beta=0.6, gamma=0.1)
x0 = jnp.array([0.99, 0.01])
cost = QuadraticCost(
    Q=jnp.zeros((2, 2)), R=jnp.array([[1.0]]), Qf=jnp.zeros((2, 2)), x_target=jnp.zeros(2)
)
plan = causal_plan(
    model, x0, cost, 1.0, 60, 0.0, 0.9, barrier=BarrierConstraint(lambda x: 0.1 - x[1], alpha=0.5)
)
assert plan._problem is not None
assert plan._problem.held is not None
held = plan._problem.held
actions = np.asarray(plan.actions, np.float64).ravel()
free = (actions > 0.0) & (actions < 0.9)
lines, names = [], []
for share, name in ((1e-2, "sir_rho_hundredth"), (1e-1, "sir_rho_tenth"), (1.0, "sir_rho_last")):
    scaled = eqx.tree_at(lambda term: term.weight, held, held.weight * share)
    _, _, curved, gauss_newton = _round_terms(model, x0, plan.actions, 1.0, cost, scaled)
    # the round's Hessian in the diagonal descent's variables: over the Gauss-Newton diagonal
    root = np.sqrt(np.diag(np.asarray(gauss_newton, np.float64)))
    jacobi = (np.asarray(curved, np.float64) / np.outer(root, root))[np.ix_(free, free)]
    jacobi = 0.5 * (jacobi + jacobi.T)
    values = np.linalg.eigvalsh(jacobi)
    rows = ",\n  ".join("[" + ", ".join(repr(float(v)) for v in row) + "]" for row in jacobi)
    lines.append(f"{name} : matrix(\n  {rows})$")
    lines.append(f"{name}_numpy : {float(values[-1] / values[0])!r}$")
    names.append(name)
    print(
        name,
        "penalty",
        float(held.weight) * share,
        "free",
        int(free.sum()),
        "condition",
        float(values[-1] / values[0]),
    )
lines.append(f"sir_matrices : [{', '.join(names)}]$")
lines.append(f"sir_numpy : [{', '.join(n + '_numpy' for n in names)}]$")
labels = ", ".join(f'"{name}"' for name in names)
lines.append(f"sir_labels : [{labels}]$")
Path("projected_newton_round_sir.dat").write_text("\n".join(lines) + "\n")
print("status", plan.solver_status, "stationarity", plan.stationarity)
