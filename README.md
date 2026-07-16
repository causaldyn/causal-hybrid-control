# causal-hybrid-control

Physics-structured dynamics with a **learned causal residual**, controlled by **constrained optimal
control / MPC**, and made safe on offline, confounded data by an explicit **pessimism / support** layer.

```
ẋ = f_known(x, u, t; p) + r_θ(x, u, t)                         # known mechanism + learned residual
u* = argmin_u  J_task(u) + λ_unc·U(x,u) + λ_supp·D((x,u), 𝒟)   # pessimistic constrained control
```

Most data science stops at prediction. The value is in *decisions* — and a decision changes the future,
so it must be evaluated as an **intervention**, not a correlation, and chosen by **optimal control**, not
by argmax over a predictive score. `chc` is a small JAX library for that step.

## The one result

On a confounded offline log, fitting the effect of the action *without* adjusting for the confounder
flips its sign (true `+1.0` → naive `-0.2`). Control the true system with each estimate:

```
controller          cost      regret    viol     ood
oracle              4.59        0.00    0.00    0.00
causal-CHC          4.59       -0.00    0.00    0.00
predictive      13740.08    13735.49    0.97    1.00
```

The **causal** controller matches the oracle; the **predictive** one is catastrophic on every metric —
it drives the state the wrong way (`x → -20` for target `+2`), violates constraints 97% of the time, and
acts entirely out of the logged support. Reproduce: `uv run python scripts/run_benchmark.py`, or
`uv run --group viz python scripts/flagship_demo.py` for the figure.

## Install

```bash
uv sync            # JAX + Diffrax + Equinox + Optax + SciPy
uv run pytest      # 25 tests
```

## Quickstart

```python
import jax, jax.numpy as jnp
from chc import DampedOscillator, HybridDynamics, KANResidual, QuadraticCost, mpc_control

# hybrid dynamics: known oscillator + a learnable (KAN) residual, swappable for MLP/linear
model = HybridDynamics(
    known=DampedOscillator(omega=1.0, zeta=0.1),
    residual=KANResidual(state_dim=2, control_dim=1, out_dim=2, key=jax.random.key(0)),
)
cost = QuadraticCost(Q=jnp.diag(jnp.array([1.0, 0.1])), R=jnp.array([[0.05]]),
                     Qf=jnp.diag(jnp.array([5.0, 1.0])), x_target=jnp.zeros(2))

xs, us = mpc_control(model, jnp.array([1.0, 0.0]), cost, dt=0.1,
                     horizon=20, u_lo=-5.0, u_hi=5.0, n_steps=40)   # closed-loop MPC
```

## Example notebooks

Worked, executed notebooks (figures + tables) under [`notebooks/`](notebooks/) — open in JupyterLab
(`uv sync --group notebooks && uv run --group notebooks jupyter lab`) or read on GitHub:

| notebook | what it shows |
|---|---|
| [`01_causal_vs_predictive_control`](notebooks/01_causal_vs_predictive_control.ipynb) | the headline: predictive control diverges under confounding, causal control matches the oracle |
| [`02_learn_hidden_physics`](notebooks/02_learn_hidden_physics.ipynb) | hybrid dynamics + system ID: recover an omitted cubic term; multi-step training cuts drift |
| [`03_causal_inference_toolkit`](notebooks/03_causal_inference_toolkit.ipynb) | adjustment · IV/2SLS · Double ML · sensitivity · refutation, side by side |
| [`04_epidemic_and_pessimism`](notebooks/04_epidemic_and_pessimism.ipynb) | flatten an epidemic curve under a capacity cap; pessimism vs a greedy controller |

Sources are paired `.py` (jupytext) next to each `.ipynb`.

## What's inside

| area | module | what it does |
|---|---|---|
| dynamics | `dynamics`, `residual`, `integrate` | hybrid `f_known + r_θ`; MLP / **RBF-KAN** / linear residuals; RK4 |
| sensitivity | `adjoint` | discrete adjoint (verified == autodiff == finite differences) |
| classical OC | `lqr` | LQR / AKOR (Riccati) — the `r_θ→0` limit and correctness baseline |
| identification | `train`, `causal` | system ID (one/multi-step); effect — adjustment, **IV/2SLS**, **DML**, sensitivity, refutation |
| control | `control`, `mpc`, `splitting` | projected-gradient OC; receding-horizon MPC; **Strang–Marchuk** splitting |
| offline safety | `support`, `offpolicy` | pessimism penalty; IPS/SNIPS off-policy value + overlap gate |
| evaluation | `benchmark`, `flagship` | pricing / inventory / **support-shift** oracle-regret tasks + leaderboard |
| scientific / PDE | `epidemic`, `galerkin` | SIR epidemic control (flatten the curve); 1D/2D Galerkin FEM (progonka) |

## Validation

Correctness is cross-checked in independent tools, symbolic first (`validation/`): the ARE / matrix
exponential are verified **Maxima**-authoritative (exact + high-precision `bfloat`) against **PARI/GP**
(50-digit) and **Octave**, with SciPy used only as the fast float64 numeric. Control invariants
(box-projection bounds + idempotence) are **formally proved in Rocq** (`proofs/box_projection.v`).

## Honest positioning

`chc` composes ideas that exist — hybrid dynamics (SciML UDE), pessimistic offline control
(MOPO/MOReL/Delphic), sequential causal identification (g-methods / dynamic treatment regimes),
differentiable control (Neuromancer). The contribution is the *integration behind one API* plus a
benchmark with ground-truth interventional effects. KAN is **one interpretable residual backend**, not
the identity of the framework. See `plans/` for the full analysis and roadmap.

## Status

Early (`v0.0.1`), single-author, research code (34 Python tests + a Rust runtime). Working: hybrid
dynamics + adjoint, LQR, system ID (one-/multi-step), causal identification (adjustment / IV / sensitivity),
pessimism, MPC, Strang–Marchuk splitting, off-policy gate, KAN backend, three benchmark tasks (pricing,
inventory, support-shift), two flagships (pricing, epidemic), 1D/2D Galerkin FEM, and a golden-parity Rust
runtime. All demonstrations are on synthetic/semi-synthetic problems — real-data validation is the next
credibility step. Roadmap: real-data case study, more tasks, and the Medium/paper writeups.

## License

MIT
