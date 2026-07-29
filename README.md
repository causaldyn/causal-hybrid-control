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
uv sync            # JAX + Diffrax + Equinox + Optax + NumPy + SciPy (Python 3.12–3.14)
uv run pytest      # 353 passed, 2 skipped (tigramite, lightgbm: bring-your-own-env)
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
| [`05_benchmark_scoreboard`](notebooks/05_benchmark_scoreboard.ipynb) | the scoreboard: regret vs oracle across every task — CHC lands next to the oracle, the baseline blows up |
| [`05_confounding_robust_control`](notebooks/05_confounding_robust_control.ipynb) | when **no adjustment set exists**: a sensitivity level `Γ` → identification radius → minimax action. Worst-case cost 1.23 → 0.35; 96% cheaper at realistic confounding, and the price is a 26%-of-the-CE-downside premium when there is none |
| [`06_cruise_control_confounded`](notebooks/06_cruise_control_confounded.ipynb) | relatable end-to-end: adaptive cruise control from confounded fleet logs (Simpson's paradox → IV → control) |
| [`07_real_data_lalonde`](notebooks/07_real_data_lalonde.ipynb) | **real data, experimental ground truth**: on LaLonde NSW the naive estimate flips sign (−$8.5k), Double ML recovers the randomised truth (+$1.8k, within $234) |

Sources are paired `.py` (jupytext) next to each `.ipynb`.

## What's inside

| area | module | what it does |
|---|---|---|
| dynamics | `dynamics`, `residual`, `integrate` | hybrid `f_known + r_θ`; MLP / **RBF-KAN** / graph / **port-Hamiltonian** (passive, Lyapunov-stable) / **Lipschitz-certified** residuals; RK4 |
| sensitivity | `adjoint` | discrete adjoint (verified == autodiff == finite differences) |
| classical OC | `lqr` | LQR / AKOR (Riccati) — the `r_θ→0` limit and correctness baseline |
| identification | `train`, `causal`, `estimators` | system ID (one/multi-step); pluggable effect backend — adjustment, **IV/2SLS**, **DML**, sensitivity, refutation, + optional **EconML/DoWhy** adapters |
| control | `control`, `mpc`, `splitting`, `plan` | projected-gradient OC; receding-horizon MPC; **Strang–Marchuk** splitting; `causal_plan` — the one-call spine returning a plan *with* its uncertainty tube and certified horizon attached |
| offline safety | `support`, `offpolicy`, `uncertainty` | pessimism penalty; IPS/SNIPS off-policy value + overlap gate; **calibrated** deep-ensemble + split-conformal uncertainty; **Wasserstein-1 DRO** distribution-shift margin; **certified rollout tubes** (Lipschitz / contractive-log-norm Grönwall bounds → time-varying uncertainty tube, safety-tightening, certified-safe horizon), **Rocq-proved** |
| guarantee | `regret` | LQ certainty-equivalence bound — quadratic in model error (Dean–Mania–Tu–Recht–Matni); **interference-aware regret certificate** (extra exposure-map-error term), **machine-checked in Rocq** |
| sensitivity-aware control | `sensitivity` (facade over `regret`, `uncertainty`, `barrier`) | **control under HIDDEN CONFOUNDING**: bounded-density-ratio (MSM) CVaR worst-case → pessimism-radius inflation; the confounding-regret floor is *second-order* in the effect bias; a **minimax controller** that shifts the gain under asymmetric (over/under-shoot) loss and beats certainty-equivalence — now a **closed-loop** controller on a confounded dynamic plant (bounds the worst-case downside, 82% cheaper over 30 steps), plus a `ConfoundingRobustPenalty` that carries the sensitivity radius into the general pessimistic-control stack — all **Rocq-certified**. `chc.sensitivity` is the one-import surface (estimate→radius→control) |
| safety under partial ID | `barrier`, `plan` | the same sensitivity radius spent on a **constraint**: robust control-barrier margin, a least-restrictive safety filter (closed-form certified action interval, no QP), and `Gamma*` — **the largest sensitivity-model level under which the barrier stays certified** (a model parameter, not measured confounding). Safety degrades at *first* order in the effect bias (until the radius swallows the channel and the loss saturates) where performance regret degrades at second (the envelope theorem protects objectives, not binding constraints), **Rocq-certified**; in closed loop a regret-sized budget violates the limit on 93% of steps where the constraint-sized one never does. `certify_safety` audits a finished plan against all of it — the certified prefix next to the plan's `Gamma*` (the weakest step's, exactly) |
| what the certificate is worth | `reachability` | the **Hamilton–Jacobi** answer the barrier only approximates: `V(x,T) = max_u min_{ΔB} min_s h(ξ(s))` on a Lax–Friedrichs grid, with the §32 identification radius as the adversary. Same robust-margin algebra as `barrier`, but `p = ∇V` is *solved for* rather than assumed. Turns the CBF theorem into an executable check (condition on all of `{h ≥ 0}` ⟹ the tube **is** `{h ≥ 0}`) and prices what pointwise certification misses — on a relative-degree-2 barrier the §40 verdict is identical at every radius while the true tube shrinks, so `certify_safety`'s per-step prefix is a filter, not a proof |
| end to end | `spine` | all four layers on **one** decision — confounded logs → causal gain → constrained plan → `Gamma*` certificate → the same plan run on the *true* plant. Two zones of a mobile driver pool, one incentive lever whose `[+b, -b]` column is driver conservation, a supply floor in the zone it drains. The confounded arm plans 13.6 and pays 38.5; `Gamma*` tells the two arms apart (7.46 vs 1.18) **before either acts**, without ground truth. `uv run python scripts/spine_demo.py` |
| causal frontier | `did`, `scm`, `estimators`, `causal` | Callaway–Sant'Anna staggered **DiD**; **augmented synthetic control**; **R-learner** CATE; **E-values** beside Cinelli–Hazlett; **influence-function CIs** on cross-fit DML |
| dynamic effects | `irf`, `toeplitz` | impulse-response / local-projection dynamic effects; Toeplitz / Levinson–Durbin / Gohberg–Semencul operators |
| structure discovery | `discovery`, `independence`, `network_causal`, `pathway` | lagged-parent discovery; MCI partial-correlation test; network/spillover orthogonal DML; **ranked temporal causal pathway** — which lagged variables & multi-step chains drive a target, signed + actionable (Rocq-certified walk-sum / geometric-truncation / weakest-link laws) |
| advanced control | `koopman`, `meanfield`, `transport`, `matching`, `games`, `mintime` | Koopman-LQR; mean-field control; continuum + discrete **Kantorovich OT** (driver↔rider matching → **dual surge prices**); differentiable Stackelberg games over a **certified** congestion equilibrium (implicit-function gradients, contraction certificate, optimal damping — the solver reports its residual instead of silently returning a non-equilibrium); PMP time-optimal bang-bang |
| marketplace moat | `marketplace` | **offline causal control under equilibrium interference**: learn incentives from confounded switchback logs where SUTVA fails — de-confounded + equilibrium-aware + W-DRO-pessimistic control recovers the oracle where MOPO / naive-causal go *negative* |
| evaluation | `benchmark`, `flagship`, `lalonde`, `metrics` | pricing / inventory / support-shift / **model-uncertainty** / **confounding-robust** oracle-regret tasks + leaderboard with multi-seed bootstrap CIs; real-data **LaLonde** validation; step-response quality metrics |
| scientific / PDE | `epidemic`, `galerkin`, `deep_galerkin` | SIR epidemic control (flatten the curve); 1D/2D Galerkin FEM (progonka); mesh-free **Deep Galerkin** neural Poisson solver |

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

Early (`v0.1.0`), single-author, research code (354 collected tests; Python 3.12–3.14, astral `ruff` + `ty`).
Working: hybrid dynamics + adjoint (discrete and adaptive `diffrax`), LQR, system ID (one-/multi-step),
causal identification (adjustment / IV / DML / sensitivity / refutation) plus the modern frontier —
Callaway–Sant'Anna staggered DiD, augmented synthetic control, R-learner CATE, E-values; **calibrated**
pessimism (deep ensemble + split conformal) and an LQ certainty-equivalence regret guarantee; MPC,
Strang–Marchuk splitting, off-policy gate, KAN/MLP/Graph residual backends; advanced control backends
(Koopman-LQR, mean-field, optimal transport, differentiable Stackelberg games, PMP time-optimal
bang-bang); dynamic-effect IRFs + structured Toeplitz/Levinson/Gohberg–Semencul operators; lagged
structure discovery; five benchmark tasks (pricing, inventory, support-shift, model-uncertainty,
confounding-robust) with multi-seed bootstrap CIs; two flagships (pricing, epidemic); 1D/2D Galerkin FEM + a mesh-free Deep
Galerkin neural Poisson solver; and step-response quality metrics. Both halves are now validated on
**real** targets, not just synthetic ones: the **causal identification core** on real data with an
experimental ground truth (notebook 07, LaLonde NSW: the naive estimate flips sign, Double ML recovers
the randomised benchmark), and the **control loop** on a real building emulator — the identification +
forecast-MPC of this library, run via `causaldyn-bench` against a live **BOPTEST**
`bestest_hydronic_heat_pump`, beats the tuned built-in baseline on *every* KPI at once (thermal
discomfort 8.01→7.32, energy 0.393→0.354, cost 0.100→0.090, emissions 0.066→0.059 — a clean Pareto
win). Roadmap: more real tasks, the Medium/paper writeups, and — only if a real-time/edge deployment
target appears — a compiled runtime.

## License

MIT
