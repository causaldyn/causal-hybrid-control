# causal-hybrid-control

[![ci](https://github.com/causaldyn/causal-hybrid-control/actions/workflows/ci.yml/badge.svg)](https://github.com/causaldyn/causal-hybrid-control/actions/workflows/ci.yml)
[![pypi](https://img.shields.io/pypi/v/causal-hybrid-control)](https://pypi.org/project/causal-hybrid-control/)
[![python](https://img.shields.io/pypi/pyversions/causal-hybrid-control)](https://pypi.org/project/causal-hybrid-control/)
[![license](https://img.shields.io/pypi/l/causal-hybrid-control)](LICENSE)
[![doi](https://zenodo.org/badge/DOI/10.5281/zenodo.21737789.svg)](https://doi.org/10.5281/zenodo.21737789)
[![docs](https://img.shields.io/website?url=https%3A%2F%2Fcausaldyn.github.io%2Fcausal-hybrid-control%2F&label=docs)](https://causaldyn.github.io/causal-hybrid-control/)
[![scorecard](https://api.scorecard.dev/projects/github.com/causaldyn/causal-hybrid-control/badge)](https://scorecard.dev/viewer/?uri=github.com/causaldyn/causal-hybrid-control)

Physics-structured dynamics with a **learned causal residual**, controlled by **constrained optimal
control / MPC**, and made safe on offline, confounded data by an explicit **pessimism / support** layer.

```
ẋ = f_known(x, u, t; p) + r_θ(x, u, t)                         # known mechanism + learned residual
u* = argmin_u  J_task(u) + λ_unc·U(x,u) + λ_supp·D((x,u), 𝒟)   # pessimistic constrained control
```

Most data science stops at prediction. The value is in *decisions* — and a decision changes the future,
so it must be evaluated as an **intervention**, not a correlation, and chosen by **optimal control**, not
by argmax over a predictive score. `chc` is a small JAX library for that step.

## Why the usual stacks do not close this

Each of the three obvious approaches solves a different problem and leaves the same gap:

- **Forecast + argmax.** Optimises a correlation. Under a confounded logging policy the fitted action
  response *is* the observational one — measured on the synthetic plant below, a control channel of
  `0.02` where the truth is `1.0`, so the planner barely acts and never notices.
- **Causal effect estimation alone** (DML, staggered DiD, synthetic control, causal forests). Returns
  an effect, not a decision. The effect can be dynamic — EconML's `DynamicDML` estimates each
  period's within the logged horizon — but nothing chooses the actions: no actuation limits, no plan
  past the logged horizon, no notion of a state the action must not reach. `chc` treats these as
  *backends* (`chc.estimators`) rather than rivals.
- **Offline RL / MPC on a learned model.** Fits the dynamics by residual MSE, which is not
  identification, and calibrates its pessimism to sampling noise rather than to unmeasured
  confounding — so the uncertainty penalty shrinks with `N` while the bias does not.

What is here is the seam: identify the *interventional* control channel from confounded logs
(`chc.dynamics_id`), plan against it under constraints, and price what unmeasured confounding can do
to the plan's performance (`chc.sensitivity`) and to its safety (`chc.barrier`).

## The one result

On a confounded offline log, fitting the effect of the action *without* adjusting for the confounder
flips its sign (true `+1.0` → naive `-0.2`). Control the true system with each estimate:

```
controller            cost      regret    viol     ood
causal-CHC            4.59       -0.00    0.00    0.00
oracle                4.59        0.00    0.00    0.00
predictive        13740.08    13735.49    0.97    1.00
```

The **causal** controller matches the oracle; the **predictive** one is catastrophic on every metric —
it drives the state the wrong way (`x → -20` for target `+2`), violates constraints 97% of the time, and
acts entirely out of the logged support. That is one seed, with the constraint and support columns;
`uv run python scripts/run_benchmark.py` runs this task and four others over 12 seeds with bootstrap
CIs (predictive regret `13734.15 [13732.55, 13735.31]`), and
`uv run --group viz python scripts/flagship_demo.py` draws the figure.

## Install

```bash
uv sync            # JAX + Diffrax + Equinox + Optax + NumPy + SciPy (Python 3.11–3.15)
uv run pytest      # the tigramite and lightgbm tests skip: bring-your-own-env
```

**GPU, TPU and other hardware.** chc has no device-specific code: install JAX's build for your
hardware through the extra of the same name -- `pip install "causal-hybrid-control[cuda13]"` on an
NVIDIA GPU whose driver is 580 or newer -- and JAX places the arrays there. The
[installation page](docs/installation.md) lists every extra, CUDA 12 and 13, TPU, ROCm and oneAPI,
JAX's conda and nightly builds, and what changes for chc on an accelerator. In a development
checkout, `just test` installs the build this machine's GPU calls for and runs the suite on it.

## Quickstart

From a panel of logs to a certified schedule, in one call. The causal assumption is a **required**
argument, because the default would be "adjust for nothing", which is a claim rather than its
absence — and when the graph says the effect is not identified there is no schedule at all.

```python
from chc import CausalGraph, Constraint, Lever, Panel, Target, prescribe

panel = Panel.from_frame(logs, unit="region", time="week")  # pandas, polars or a dict
graph = CausalGraph.from_edges(
    [("demand", "incentive"), ("demand", "supply"), ("incentive", "supply"), ("incentive", "wait")]
)

out = prescribe(
    panel,
    adjustment=graph,
    horizon=15,
    dt=0.1,
    tolerance=0.5,
    levers=[Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
    target=Target("supply", value=1.0),
    constraints=[Constraint("wait", hi=0.5)],
)

out.certificate.trustworthy_steps  # 15 -- the prefix that survives BOTH identification and safety
out.schedule.magnitudes  # (15, 1); raises instead, if the effect is not identified
print(out.report())  # Markdown: decision, both certificate axes, provenance
```

`prescribe` composes what is below it and adds no new estimator, solver or guarantee: the
adjustment set comes from the graph, the control channel from cross-fit Robinson DML
(`fit_causal_residual`), the plan from `causal_plan`, the safety price from `certify_safety`,
and how far the plan can be from the best one its own box allows from `plan_regret_bound` --
`0.0` at a plan every lever bound pins, where the textbook `|grad J|^2/(2 mu)` reports `55.83`.

### The expert path

```python
import jax, jax.numpy as jnp
from chc import HybridDynamics, QuadraticCost
from chc.dynamics import DampedOscillator
from chc.mpc import mpc_control
from chc.residual import KANResidual

# hybrid dynamics: known oscillator + a learnable (KAN) residual, swappable for MLP/linear
model = HybridDynamics(
    known=DampedOscillator(omega=1.0, zeta=0.1),
    residual=KANResidual(state_dim=2, control_dim=1, out_dim=2, key=jax.random.key(0)),
)
cost = QuadraticCost(
    Q=jnp.diag(jnp.array([1.0, 0.1])),
    R=jnp.array([[0.05]]),
    Qf=jnp.diag(jnp.array([5.0, 1.0])),
    x_target=jnp.zeros(2),
)

xs, us = mpc_control(
    model, jnp.array([1.0, 0.0]), cost, dt=0.1, horizon=20, u_lo=-5.0, u_hi=5.0, n_steps=40
)  # closed-loop MPC; u_lo/u_hi also take a per-lever array when the actuators differ
```

Online, where the plant is the world rather than a simulation, `RecedingHorizon` plans from each
measured state and returns the whole `causal_plan` -- actions, tube and audit together:

```python
from chc import BarrierConstraint, RecedingHorizon

floor = BarrierConstraint(lambda x: x[1] + 0.8, alpha=2.0)  # built once, not per step
controller = RecedingHorizon(model, cost, dt=0.1, horizon=20, u_lo=-5.0, u_hi=5.0, barrier=floor)
plan = controller.step(x)  # every step: read plan.safety, then apply plan.actions[0]
```

Each step starts from the last plan and the barrier's multipliers, shifted one step on -- 35 % fewer
descent steps than cold solves on an oscillator's velocity floor, for the same closed-loop cost to
`2e-6` (`docs/adr/0003-receding-horizon-warm-starts.md`). A program compiles the first time a step
needs it and is reused after, so steps stop compiling as long as the model, the cost and the
barrier's function stay the same objects: a new `lambda` per step compiles the descent per step.
For a process that restarts, set `jax_compilation_cache_dir` and lower
`jax_persistent_cache_min_compile_time_secs` to 0 -- at its default of one second JAX wrote none of
a first step's programs to disk.

## Example notebooks

Worked, executed notebooks (figures + tables) under [`notebooks/`](notebooks/) — open in JupyterLab
(`uv sync --group notebooks && uv run --group notebooks jupyter lab`) or read on GitHub:

| notebook | what it shows |
|---|---|
| [`01_causal_vs_predictive_control`](notebooks/01_causal_vs_predictive_control.ipynb) | the headline: predictive control diverges under confounding, causal control matches the oracle |
| [`02_learn_hidden_physics`](notebooks/02_learn_hidden_physics.ipynb) | hybrid dynamics + system ID: recover an omitted cubic term; multi-step training cuts drift |
| [`03_causal_inference_toolkit`](notebooks/03_causal_inference_toolkit.ipynb) | adjustment · IV/2SLS · Double ML · sensitivity · refutation, side by side |
| [`04_epidemic_and_pessimism`](notebooks/04_epidemic_and_pessimism.ipynb) | flatten an epidemic curve under a capacity cap; pessimism vs a greedy controller |
| [`05_benchmark_scoreboard`](notebooks/05_benchmark_scoreboard.ipynb) | the scoreboard: regret vs oracle on three tasks (pricing, inventory, support shifts) — CHC lands next to the oracle, the baseline blows up |
| [`05_confounding_robust_control`](notebooks/05_confounding_robust_control.ipynb) | when **no adjustment set exists**: a sensitivity level `Γ` → identification radius → minimax action. Worst-case cost 1.23 → 0.35; 96% cheaper at realistic confounding, and the price is a 26%-of-the-CE-downside premium when there is none |
| [`06_cruise_control_confounded`](notebooks/06_cruise_control_confounded.ipynb) | relatable end-to-end: adaptive cruise control from confounded fleet logs (Simpson's paradox → IV → control) |
| [`07_real_data_lalonde`](notebooks/07_real_data_lalonde.ipynb) | **real data, experimental ground truth**: on LaLonde NSW the naive estimate flips sign (−$8.5k), Double ML recovers the randomised truth — +$1.6k against the experiment's +$1.8k, within $234 |

Sources are paired `.py` (jupytext) next to each `.ipynb`.

## What's inside

| area | module | what it does |
|---|---|---|
| dynamics | `dynamics`, `residual`, `integrate`, `symbolic` | hybrid `f_known + r_θ`; MLP / **RBF-KAN** / graph / **control-affine** (`a_θ(x) + B_θ(x)u`, the class the identification and safety layers share) / **port-Hamiltonian** (passive; `energy="icnn"` for a *coercive* one) / **Lipschitz-certified** / **spectral** residuals; RK4. `H' <= 0` is an identity of `(J - R) grad H` in which `H` never appears, so it holds for any energy network and separates none of them (`2.7e-15` for both arms); what it confines the state to is `{ H <= H(x0) }`, and a `tanh` energy read out linearly is bounded in `x`, so that set is the whole space and `invariant_radius` is honestly `inf`. The input-convex energy with a quadratic floor makes it a ball: realised excursion `6.00` inside a predicted `25.81`, and one critical point instead of three. The spectral one IS a circulant on a periodic grid, so its operator norm `max_k |lambda_k|` is *attained* rather than bounded (the Lipschitz backbone's Schur bound measures 113x slack), it is translation-equivariant to machine precision, and being linear in its kernel it is fitted by a closed-form per-mode least squares rather than by Adam. `symbolic` turns a fitted RBF-KAN edge back into a closed form and states what that is worth: the intercept is a gauge (only the total is identified), the layer is additive so an interaction has a *proved* error floor `r²` on `[−r,r]²`, and the extracted formula extrapolates where the layer (RBF support gone) does not — 4.22e-4 against 34.65 |
| sensitivity | `adjoint` | discrete adjoint (verified == autodiff == finite differences) |
| classical OC | `lqr` | LQR / AKOR (Riccati) — the `r_θ→0` limit and correctness baseline |
| identification | `train`, `dynamics_id`, `causal`, `estimators`, `gmethods`, `frames` | system ID (one/multi-step); pluggable effect backend — adjustment, **IV/2SLS**, **DML**, sensitivity, refutation, + optional **EconML/DoWhy** adapters; Robins' **g-formula** (cross-fitted) for a treatment *sequence* under time-varying confounding. `dynamics_id` is the one that makes the *plant* causal: prediction-error fitting learns the **observational** control channel, so under a confounded logging policy the planner inherits the bias (measured: channel `0.02` where the truth is `1.0`). `fit_causal_residual` estimates it by Robinson partialling-out lifted to a state-dependent matrix — channel error `0.002`, control regret `0.014` against the biased fit's `6.41` — or by 2SLS when the confounder is never logged, at a real variance premium (`0.10` error, regret `0.13`, because the shifter explains only 18% of the action). Reports `identified=False` instead of a confident wrong answer when nothing in the log can pin it down. Data goes in as a mapping of arrays, a **pandas** frame or a **polars** frame — `frames.as_columns` recognises a frame structurally and normalises once at the boundary, so neither library is a dependency of the wheel. `uv run python scripts/dynamics_id_demo.py` |
| control | `cost`, `control`, `mpc`, `splitting`, `plan` | Bolza objective; projected-gradient OC and a bound-constrained quasi-Newton (`lbfgs_box_control`) sharing the same discrete-adjoint gradient, with `box_stationarity` as the reference-free convergence measure; receding-horizon MPC (`mpc_control` against a simulated plant, `RecedingHorizon.step(x)` online: one `causal_plan` per measured state, warm-started from the last plan and its barrier multipliers); **Strang–Marchuk** splitting; `causal_plan` — the one-call spine returning a plan *with* its uncertainty tube and certified horizon attached. Three modes are named apart on purpose: **plan** (`causal_plan`, the box and linear constraints — budgets, rate limits — in the solve, and with `barrier=` the audit's own condition held by augmented-Lagrangian rounds), **audit** (`certify_safety`, read-only on a finished plan), **filter** (`robust_safety_filter`, one action at a time, online). The tube never enters the *objective* and a barrier does only when given one; even then the verdict is the audit's, which the plan carries as `plan.safety`, because a solve stopped by its budget can come back short of the condition; with no error model supplied the certificate reports `not_evaluated` rather than a vacuous full-horizon pass |
| offline safety | `support`, `evaluation`, `offpolicy`, `uncertainty` | pessimism penalty; a plan's **deployed** value from logs (per-decision IS, marginalised IS, doubly robust, fitted Q), **certified before any cost is read** and refused by name when a weight has infinite variance; one-step IPS/SNIPS off-policy value + overlap flag; **calibrated** deep-ensemble + split-conformal uncertainty; a **time-consistent nested-CVaR** aggregation of that disagreement (the risk-neutral sum averages one very bad step away); **Wasserstein-1 DRO** distribution-shift margin; **certified rollout tubes** (Lipschitz / contractive-log-norm Grönwall bounds → time-varying uncertainty tube, safety-tightening, certified-safe horizon), **Rocq-proved**. The ensemble trains as one sharded program (`vmap` over a member axis, `lax.scan` over the Adam steps, `NamedSharding` over the device mesh) rather than K sequential fits — 3.6× on one device, 10.3× over an 8-device mesh at K=8, agreeing with the serial recursion to 232 ULP |
| experiment design | `experiment` | which experiment, if any, to run before a one-shot decision over zones: units per zone from the **decision weight** (the Hessian of the certainty-equivalent regret, averaged over the prior's sigma points), a Gaussian-tail price for a lever pinned at its bound, and reverse water-filling at the price of information; then **Monte Carlo re-solves the decision on every draw**, and every number reported is that measurement. The verdict is deploy, experiment or abstain, with the reason. At one budget on the lab's four-zone market, the decision-weighted allocation leaves `1.18e-3` of regret where classical Neyman leaves `1.84e-3` |
| switchback | `switchback` | which switchback to run on a plant with memory, and how to read it, for a **named** effect: the channel, the effect of holding the lever on `H` periods, or the steady state. The design is a Markov chain **aligned to the effect**, so the persistence estimate is orthogonal to it (at two periods it flips with probability `a/(1+2a)`); IV under measurement noise; block or local-projection readings when the state model is not trusted; standard error, MDE and loss at the least favourable persistence. `read_switchback` reads the effect off the data: robust or HAC standard errors, Fieller's interval for the steady state, and a block difference centred on earlier blocks, unbiased at any number of blocks where the realised-means form is 27% low at eight. Planned standard errors were 0.99–1.04 of the simulated spread, and the data's intervals covered 0.937–0.959 of 0.95. On the market of `zones`, whose state is not first order, the plug-in read 3–14% off and its first-order test warned on 99.9–100% of runs; the model-free readings covered 0.944–0.953, but the plan's variances are the working model's, and power held within three points of nominal only for the channel and, at the edge, `tau_2` |
| deployment | `gate` | a candidate policy in shadow of the baseline, per zone, until an **anytime-valid** gate says deploy, hold, experiment or roll back: at every read, including one chosen by looking at the data, the expected share of wrong deploys across zones stays at most `alpha` (e-processes over the logged propensities, e-BH over e-values frozen at selection). It takes the three propensities rather than weights, and refuses a batch whose logged ones are not those of the policy it asked to log. Its conditions — logged propensities, no spillover, no carryover — are in the docstring |
| guarantee | `regret` | LQ certainty-equivalence bound — quadratic in model error (Dean–Mania–Tu–Recht–Matni); **interference-aware regret certificate** (extra exposure-map-error term), **machine-checked in Rocq**; the **van Trees floor on control regret** — `multivariate_action_floor` for a matrix effect, where the bound is a trace and confounding is priced by *alignment* with `du*/dθ` rather than by a ratio; and `capped_exploration_policy`, which takes a per-round cap **schedule** and a spending **budget** and stops on a delivered exploration *mass* rather than a round count |
| sensitivity-aware control | `sensitivity` (facade over `regret`, `uncertainty`, `barrier`) | **control under HIDDEN CONFOUNDING**: bounded-density-ratio (MSM) CVaR worst-case → pessimism-radius inflation; the confounding-regret floor is *second-order* in the effect bias; a **minimax controller** that shifts the gain under asymmetric (over/under-shoot) loss and beats certainty-equivalence — now a **closed-loop** controller on a confounded dynamic plant (bounds the worst-case downside, 82% cheaper over 30 steps), plus a `ConfoundingRobustPenalty` that carries the sensitivity radius into the general pessimistic-control stack — all **Rocq-certified**. `Γ` itself is **calibrated before it is spent**: `benchmark_gamma` prices it in units of the confounding the observed covariates carry (an exponent, `log Γ / log Γ_strongest`, because odds ratios compose) and `negative_control_gamma` inverts a known-null outcome for the smallest `Γ` that reconciles it — a *lower bound* on the confounding present, or `inf` when the model class is refuted instead. `chc.sensitivity` is the one-import surface (calibrate→radius→control) |
| safety under partial ID | `barrier`, `plan` | the same sensitivity radius spent on a **constraint**: robust control-barrier margin, a least-restrictive safety filter (closed-form certified action interval, no QP), and `Gamma*` — **the largest level of the marginal sensitivity model under which the barrier stays certified** (a model parameter, not measured confounding; Rosenbaum's `Γ` is another model's, and `docs/concepts/gamma.md` gives the bracket between them). Safety degrades at *first* order in the effect bias (until the radius swallows the channel and the loss saturates) where performance regret degrades at second (the envelope theorem protects objectives, not binding constraints), **Rocq-certified**; in closed loop a regret-sized budget violates the limit on 93% of steps where the constraint-sized one never does. `certify_safety` audits a finished plan against all of it — the certified prefix next to the plan's `Gamma*` (the weakest step's, exactly) — and `causal_plan(barrier=BarrierConstraint(...))` holds the same worst-case condition inside the solve instead of pricing it after |
| what the certificate is worth | `reachability` | the **Hamilton–Jacobi** answer the barrier only approximates: `V(x,T) = max_u min_{ΔB} min_s h(ξ(s))` on a Lax–Friedrichs grid, with the §32 identification radius as the adversary. Same robust-margin algebra as `barrier`, but `p = ∇V` is *solved for* rather than assumed. Turns the CBF theorem into an executable check (condition on all of `{h ≥ 0}` ⟹ the tube **is** `{h ≥ 0}`) and prices what pointwise certification misses — on a relative-degree-2 barrier the §40 verdict is identical at every radius while the true tube shrinks (6.4% of the grid certified-and-unreachable), so `certify_safety`'s per-step prefix is a filter, not a proof. `uv run python scripts/reachability_demo.py` |
| end to end | `spine` | all four layers on **one** decision — confounded logs → causal gain → constrained plan → `Gamma*` certificate → the same plan run on the *true* plant. Two zones of a mobile driver pool, one incentive lever whose `[+b, -b]` column is driver conservation, a supply floor in the zone it drains. The confounded arm plans 13.59 and pays 38.97; `Gamma*` tells the two arms apart (7.46 vs 1.17) **before either acts**, without ground truth. `uv run python scripts/spine_demo.py` |
| decisions from a panel | `decision`, `panel`, `graph` | the façade: `prescribe(panel, adjustment=graph, ...)` runs adjustment set → causal channel → constrained plan → safety certificate as one call, and adds no new estimator, solver or guarantee. `graph.CausalGraph` derives the adjustment set (the canonical Perković–Textor–Kalisch–Maathuis set, valid **iff any observed set is**) instead of asking a caller to type `covariates=("x", "z")` by hand, so a collider is never adjusted for and a mediator never removed — the two failures that raise nothing and change the number. `panel.Panel` checks a long panel's `(unit, time)` index once and does the wide pivot DiD/SCM need, naming the column *and* the entity when it refuses. `DecisionCertificate` keeps identification and certification apart, and an unidentified effect yields **no schedule at all** rather than one that looks like every other schedule. Measured on one synthetic plant read three ways: adjusted channel `0.84` against a true `0.80`, empty adjustment asserted `2.09`, confounder latent → no schedule |
| case study: media budgets | `mmm` | **marketing-mix budget scheduling** on a saturating carryover plant, where the confounding is not hypothetical: spend is planned *against demand*, so a model fitted on the log credits the channel with the season. Each channel acts twice — an immediate incremental return (the **control channel**, and what gets confounded) and a carried-over one through a saturating adstock (the **drift**); the mechanical adstock rows go in as `known=`. Three readings of one log, every arm audited on the *true* plant: at matched budget the prescribed schedule buys **+4.5% more lift** (cumulative sales over the do-nothing arm) than an equal split by front-loading carryover then tapering, while the confounded arm inflates the channels `2.3–8.1x`, under-invests by 15% and buys 87% of the lift. It also shows why return-per-unit-spend is the wrong scoreboard: under diminishing returns the confounded arm scores *higher* on it while earning less. `uv run python scripts/mmm_demo.py` |
| case study: a pendulum | script, over `decision` | **Pendulum-v1 raised 0.30 rad from hanging and held there**, planned by `prescribe` from a log in which an operator cancelled half the wind torque it measured — so the logged torque moves with a disturbance that enters `omega'` exactly where the torque does, and the confounding sits on the control channel itself. Gravity goes in as `known=`, the actuator is what the log has to identify, and every schedule is run on the *true* pendulum. Adjusted for the wind (the graph derives `{wind}`), the torque channel is `+3.002` against a true `3.000` and the schedule ends `0.003` rad from the target, with the speed held within `1.0` rad/s in the solve as the barrier; asserting an empty adjustment set returns `-1.064` (the white-wind omitted-variable formula predicts `-1.052`), so the plan pushes the wrong way and ends `0.610` rad off, `2.0x` as far as never acting; declared latent, the wind leaves no schedule. The certificate trusts the adjusted schedule for 4 steps of 40 and the asserted one for none. Printed in float32, the default, and the script says which of its numbers float64 moves. `uv run python scripts/pendulum_demo.py` |
| causal frontier | `did`, `scm`, `estimators`, `causal` | Callaway–Sant'Anna staggered **DiD**; **augmented synthetic control**; **R-learner** CATE; **E-values** beside Cinelli–Hazlett; **influence-function CIs** on cross-fit DML |
| dynamic effects | `irf`, `toeplitz` | impulse-response / local-projection dynamic effects; Toeplitz / Levinson–Durbin / Gohberg–Semencul operators |
| delay | `delay`, `irf` | a discrete delay as a **plain `Dynamics`** — the m-stage linear chain, so `rollout`, the adjoint, both solvers, `causal_plan`, `certify_safety` and `mpc_control` run on a delayed plant unchanged (augment, don't write a DDE solver); the exact **delay margin** `arccos(a/K)/sqrt(K²−a²)`; a causal **delay estimate with a moving-block bootstrap interval**; and the stabilising ball in *delay* space, which is a **half-line with a relative radius** whose performance loss is a **square root** one side and linear the other — the decay-optimal design sits at a defective root, so neither a symmetric ball nor a single regret constant exists. `robust_delay_design` turns an interval into the minimax design, which is *below* its centre. **Rocq-proved** |
| structure discovery | `discovery`, `independence`, `network_causal`, `pathway` | lagged-parent discovery; MCI partial-correlation test; network/spillover orthogonal DML; **ranked temporal causal pathway** — which lagged variables & multi-step chains drive a target, signed + actionable (Rocq-certified walk-sum / geometric-truncation / weakest-link laws) |
| advanced control | `koopman`, `meanfield`, `transport`, `matching`, `games`, `mintime` | Koopman-LQR; mean-field control; a periodic **advection-diffusion** field with an exact spectral propagator (the translation-invariant plant that justifies the spectral residual); continuum + discrete **Kantorovich OT** (driver↔rider matching → **dual surge prices**); differentiable Stackelberg games over a **certified** congestion equilibrium (implicit-function gradients, contraction certificate, optimal damping — the solver reports its residual instead of silently returning a non-equilibrium); PMP time-optimal bang-bang |
| media mix: coefficients that move | `dlm` | West and Harrison's **discount dynamic linear model**: a level or trend, Fourier seasonality over a period that need not be an integer, and random-walk coefficients on media, each block with its own discount; the observational variance learned in closed form; one-step Student `t` forecasts and their log-likelihood; a smoother whose covariances are in the units of the variance the whole series supports and whose lag-one covariances give a period's total an interval; backward sampling; `k`-step forecasts under a stated evolution policy; e-values of the one-step errors for `DriftAlarm`. The filter matches PyBATS to `2.7e-16`. The one-step forecasts cover 0.90 of 0.90; a channel's 13-week contribution, with the discounts picked by that likelihood, covers 0.78, and 0.88–0.90 with the coefficients' discount fixed at 0.85–0.9: a period's return is not reported from the likelihood's pick alone. **Over geos** (`forward_filter_geos`): national blocks every geo reads and regional blocks each geo has its own copy of, a geo's coefficient the national one plus its deviation, filtered as one state with one variance scale; one geo is the filter's own to `1e-12`, and with nothing discounted many geos are the conjugate regression on every observation to `1e-10`; `smooth` and `backward_sample` run back over the stacked state, the smoother the joint Gaussian posterior of every state to `1e-8`; `fit_geo_spread` chooses the geos' spread by that likelihood and draws it from its posterior, so that an effect's interval carries the spread's uncertainty. On `chc.geo_world`'s worlds pooling puts each geo's error 9–11 % below its own alone at a noise of 30 and 33 % at 100, and the intervals mixed over the spread's draws cover 0.88–0.89 of 0.90. `uv run python scripts/bench_dlm.py`, `scripts/bench_geo_dlm.py` |
| media mix: response and carryover | `response` | **seventeen saturation families** under one contract — zero at zero spend, increasing, ceiling 1, one scale `K` in currency so a change of currency moves `K` alone: Michaelis–Menten, exponential, tanh, arctan, algebraic, half-normal; Hill, Weibull, logistic, Gompertz, Richards, Chapman–Richards, gamma, log-normal, Burr XII, beta and Kumaraswamy CDFs; plus `log(1 + x/K)`, constant elasticity and Ricker's inverted U. Each S-shaped family states its **inflection** and the **tangency** of its concave envelope, in closed form where one exists (Maxima) and as a root otherwise; given no warm start, `causal_plan` plans on each such curve's concave envelope first (`relax`), since on an S-curve `h'(0) = 0` and a descent from zero spend stops there, and reports the relaxed problem's cost, which bounds the plan's distance from the best. **Carryover kernels** (geometric, delayed peak, Weibull), each with a length of its own, so logging another period moves no earlier adstock; a `Channel` reads spend through a kernel and a curve; and **one definition of return**, carryover included: `contribution`, and per currency unit `roi`, `marginal_roi`, `steady_state_marginal_roi`. Mapped onto PyMC-Marketing's transforms and run beside them (`scripts/pymc_marketing_reference.py`) |
| media mix: lift tests | `lift` | **geo tests read as structure**: a channel's carryover, curve and size fitted to the gap between a test's groups period by period (Heusch's differencing equation), by least squares over every test with one noise scale, each parameter with a profile-likelihood interval that says when the tests leave it open; the coefficient projected out and the rest in logs, so the fit follows the ridges a line and a step make; and the adstock the readouts covered, the only range over which the curve is identified. On Heusch's endogenous-spend generator, written clean-room in causaldyn-bench, his four go-dark tests at his noise: the carryover's 95 % interval covered 0.954 of 500 histories; at three times the noise it still covers, and 15 % of those intervals close | `just track-m2-lift` in causaldyn-bench |
| media mix: the budget's split | `allocation` | **a quarter's budget across channels**: one spend a period for each `chc.response` channel within its box, for the most return, with the history's carryover running into the plan and the plan's running on after it; exact where every curve is concave, by bisection on the budget's price, a linear channel's jump spent to the budget exactly, and the budget's shadow price reported; an S-curve planned on its concave envelope, with the gap that bounds the plan's shortfall. The box is required: it is where the channels were seen. **Every geo's channels under one budget** (`allocate_geos`): caps and floors on what each geo and each channel spends, a price for each, certified by cutting planes and made exact by Newton's method on the totals that bind; a return, a marginal return or a return on spend over the grid in place of the budget (`budget_for_geos`), in a few plans | `tests/test_allocation.py`, `tests/test_geo_allocation.py`, `tests/test_geo_goal_seek.py` |
| media mix: a geo world | `geo_world` | **geos to score a geo model and a geo plan on**: each geo's KPI made by its own media, each geo's effect on a channel drawn log-normal about the channel's national median with a stated spread, populations and bases per head likewise; a lift per head, the curve per head of the spend per head through a normalised carryover, so a geo twice the size spending twice as much lifts twice as much; national media allotted by population; the log's spend raised with the season by a policy, the confounding; spillover into a geo's two neighbours on a ring, and a national drift in each channel's lift, as options. Every variate is drawn whatever the parameters, so two mixes compare paired. Each geo's channels come back as `chc.response` cells that lift exactly what the world's media lift, for `allocate_geos` to plan on | `tests/test_geo_world.py` |
| marketplace moat | `marketplace` | **offline causal control under equilibrium interference**: learn incentives from confounded switchback logs where SUTVA fails — de-confounded + equilibrium-aware + W-DRO-pessimistic control recovers the oracle where MOPO / naive-causal go *negative* |
| marketplace plant | `zones` | the **zones × time plant the marketplace loop runs on**, control-affine by construction: idle supply, open requests and an incentive stock per zone; an incentive and a price per zone as levers; recruits drawn from neighbouring zones and supply that answers late. Its logged operator raised both levers with each zone's demand shock, so a fit that ignores the shock reads a price rise as all but free, in most zones as raising demand, and an incentive at 18–39% of what it recruits; adjusted, it reads both within 22% of one period's response, zone by zone, over six logs. Bridges to `ZoneDecision` and `LinearGaussianPlant`, exact under linear matching |
| evaluation | `benchmark`, `causal_bench`, `flagship`, `lalonde`, `metrics`, `surrogate` | pricing / inventory / support-shift / **model-uncertainty** / **confounding-robust** / **causal-dynamics** (the confounding is in the plant's own channel; the failure is invisible to the constraint and support columns) oracle-regret tasks + leaderboard with multi-seed bootstrap CIs; a causal-methods table scoring every frontier estimator against the naive baseline it is meant to beat; real-data **LaLonde** validation; step-response quality metrics; a gradient-boosted tree surrogate as the tabular prediction competitor (optional `trees` extra) |
| scientific / PDE | `epidemic`, `galerkin`, `deep_galerkin` | SIR epidemic control (flatten the curve); 1D/2D Galerkin FEM (progonka) plus the non-symmetric **convection-diffusion** case with the cell-Peclet threshold and the optimal SUPG parameter; mesh-free **Deep Galerkin** — a neural Poisson solver, and the coupled **mean-field game** (backward HJB + forward Fokker-Planck joined by `alpha* = -(b/r)V_x` and the population mean), with both boundary conditions structural rather than penalised. Gated on an exact LQ closed form, which also prices the failure: past the anti-monotone threshold `c = 1 + ra²/(qb²)` the equilibrium degenerates at a horizon in closed form, and there the solver's own residual *falls* while its error rises. The density it returns is the `N -> infinity` limit, so the **finite-population gap** is priced exactly rather than fitted: `E[(m_N - m)^2] = v(t)/N` at every `t` and every `N`, because a mean-field feedback leaves the closed-loop agents independent. The a-posteriori estimator generalises off the closed form too: `adjoint_weighted_error` linearises whatever field it is given, so the same construction runs on a **congestion-shifted** game with no closed form — exact on the affine problem, second-order otherwise, and one order better than reusing the affine adjoint |

## Validation

Correctness is cross-checked in independent tools, symbolic first (`validation/`): the ARE / matrix
exponential are verified **Maxima**-authoritative (exact + high-precision `bfloat`) against **PARI/GP**
(50-digit) and **Octave**, with SciPy used only as the fast float64 numeric. The control and guarantee
invariants are **formally proved in Rocq** — the files under `proofs/`, from the box-projection bounds
and idempotence (`box_projection.v`) to the interference-aware regret certificate — and where Stdlib
could state only a scalar shadow, `proofs/mathcomp/` proves the matrix statement with MathComp.

## Honest positioning

`chc` composes ideas that exist — hybrid dynamics (SciML UDE), pessimistic offline control
(MOPO/MOReL/Delphic), sequential causal identification (g-methods / dynamic treatment regimes),
differentiable control (Neuromancer). The contribution is the *integration behind one API* plus a
benchmark with ground-truth interventional effects. KAN is **one interpretable residual backend**, not
the identity of the framework — and `chc.symbolic` makes "interpretable" checkable rather than
asserted, including the two places where the interpretation stops being valid.

Worth knowing before you rely on a fitted model: **a low residual MSE is not causal identification.**
Fitting `r_θ` by prediction error recovers the *observational* control response, which is the wrong
one whenever the logged action was chosen from something that also moved the state — on the synthetic
plant the trained channel is `0.02` where the truth is `1.0`, and no amount of extra training fixes
it because it is not a fitting problem. `chc.dynamics_id` is the identified route, and it is
restricted to control-affine residuals; outside that class this library offers a sensitivity radius
(`chc.sensitivity`), not an unbiased estimate.

### Where it sits

Read by the two questions this library refuses to merge: does the tool **identify** the effect of an
action from data a policy generated, and does it **certify** the plan it hands you. Most tools answer
one; the ones that answer both are papers, not packages. Versions and licences read 2026-09-28.

| | what it is for | identifies an interventional effect | produces a schedule | ships a certificate | licence |
|---|---|---|---|---|---|
| **`chc`** | decisions from a confounded log, over a plant | yes, the **control channel** of a control-affine residual (cross-fit Robinson DML), and it says `not_identified` rather than guessing | yes, projected gradient over a box and linear constraints (budgets, rate limits), with a confounding-robust barrier held in the solve | yes — identification status, trajectory tube, barrier `Γ*` | MIT |
| **DoWhy / DoWhy-GCM** 0.14 | identify and refute an effect on a DAG | yes — back-door, front-door, IV, and the Rotnitzky–Smucler **efficient** backdoor set, which minimises asymptotic variance among backdoor sets. CHC's `CausalGraph` answers the other question, Perković et al.'s canonical set, which is valid **iff any observed set is** | no | refutation tests, not a control guarantee | MIT |
| **EconML** 0.17.0 | heterogeneous treatment effects, DML/DR/orthogonal forests | yes, static **and** sequential: its DML estimators return a matrix θ(X), and `DynamicDML` estimates each period's effect of an adaptively assigned treatment on the last period's outcome (sequential ignorability, a linear-Markov state, a balanced panel). CHC applies the same orthogonal moment one step at a time to a control-affine plant | no — it chooses no sequence | confidence intervals | MIT |
| **DCBO**, last commit 2023-04 | sequential interventions in a time-varying SCM | yes, by GP emulation over an SCM | yes, a sequence of interventions | regret empirics, no feasibility guarantee | **ambiguous**: MIT in `LICENSE`, GPL-3.0-or-later in the README and `setup.py`; research code, not on PyPI |
| **Google Meridian** 2.1.0 | Bayesian marketing-mix modelling | partially — priors and geo experiments calibrate it; the estimand is the media response | no — it splits a budget across channels and holds each channel's flighting over geos and time fixed; a flighting can be supplied, not optimised | posterior intervals | Apache-2.0 |
| **do-mpc** 5.1.2 | robust and economic nonlinear MPC | **no** — the model is yours and assumed correct | yes, and more general constraints than CHC's: nonlinear path constraints on the state itself, where CHC holds linear rows on the actions and a barrier's decay condition | robust multi-stage MPC guarantees, under a correct model | **LGPL-3.0** |
| **d3rlpy** 2.8.1 | offline deep RL from logged trajectories | no — conservatism bounds value error, not confounding | yes, a policy | pessimistic value bounds | MIT |
| **causaLens `decisionOS`** | enterprise causal decision platform | yes, per its own account | yes | not publicly auditable | commercial, closed |

Two rows that are **not** here, and the reason is the same. The 2024–25 literature on causal
Bayesian optimisation under safety constraints, and on causal optimal control ("COAST"-style), has
no shipped, installable implementation this could be run against. That is the gap CHC is aimed at,
and stating it as an absence is more honest than a row of dashes against a paper.

Where a row says *no* it is not a criticism: do-mpc solves control problems CHC cannot state, and
EconML answers effect questions CHC does not ask. The claim is narrower — that **going from a
confounded log to a certified schedule in one place** is what nothing above does end to end.

What that claim concedes:

- **Within the logged horizon, `DynamicDML` is the better tool** for a dynamic effect. CHC's
  addition is a plant that carries the answer past that horizon, under constraints, with a
  certificate. [`scripts/econml_reference.py`](scripts/econml_reference.py) runs EconML 0.17.0 on
  both questions: LinearDML returns a 2×2 `B(x)` to 0.068, and DynamicDML a linear plant's impulse
  response to 0.013.
- **Planning a marketing-mix schedule through the adstock state is Nerlove and Arrow (1962).** The
  shipped tooling is new; the idea is not.
- **Off-policy evaluation is weaker than SCOPE-RL's.** CHC's weights one step on the logger's states
  (IPS/SNIPS with an overlap flag); SCOPE-RL has the sequential estimators, doubly robust and DICE.
- **CasADi, acados and do-mpc are far stronger solvers**, and CHC has nothing for embedded MPC. Its
  claim is where the plant comes from, not how it is solved.

The release-by-release record, scope corrections included, is in [`CHANGELOG.md`](CHANGELOG.md).

## Status

Early (`v0.9.0`), single-author, research code (`just counts` prints the test and proof counts;
Python 3.11–3.15 with the free-threaded 3.14t and 3.15t, astral `ruff` + `ty`).
Working: hybrid dynamics + adjoint (discrete and adaptive `diffrax`), LQR, system ID (one-/multi-step),
causal identification (adjustment / IV / DML / sensitivity / refutation) plus the modern frontier —
Callaway–Sant'Anna staggered DiD, augmented synthetic control, R-learner CATE, E-values; **calibrated**
pessimism (deep ensemble + split conformal) and an LQ certainty-equivalence regret guarantee; MPC,
Strang–Marchuk splitting, off-policy gate, KAN/MLP/Graph residual backends; advanced control backends
(Koopman-LQR, mean-field, optimal transport, differentiable Stackelberg games, PMP time-optimal
bang-bang); dynamic-effect IRFs + structured Toeplitz/Levinson/Gohberg–Semencul operators; lagged
structure discovery; seven benchmark tasks (pricing, inventory, support-shift, model-uncertainty,
confounding-robust, causal-dynamics, delay-oscillation) with multi-seed bootstrap CIs; two flagships (pricing, epidemic); 1D/2D Galerkin FEM + a mesh-free Deep
Galerkin neural Poisson solver; and step-response quality metrics. Both halves are now validated on
**real** targets, not just synthetic ones: the **causal identification core** on real data with an
experimental ground truth (notebook 07, LaLonde NSW: the naive estimate flips sign, Double ML recovers
the randomised benchmark), and the **control loop** on a real building emulator — the identification +
forecast-MPC of this library, run via `causaldyn-bench` against a live **BOPTEST**
`bestest_hydronic_heat_pump`, beats the tuned built-in baseline on *every* KPI at once (thermal
discomfort 8.01→7.32, energy 0.393→0.354, cost 0.100→0.090, emissions 0.066→0.059 — a clean Pareto
win). Planned through `prescribe` instead, from a weather-compensated log, the same plant gave a
pre-registered comparison of the adjusted call against the naive one whose verdict is void, and a
naive loop that ran away where its fitted channel changed sign
([case study](docs/case-studies/boptest.md)). Roadmap: more real tasks, the Medium/paper writeups,
and — only if a real-time/edge deployment target appears — a compiled runtime.

### What "0.x" promises

Pre-1.0, so SemVer's major-version protection does not apply yet. What *does* apply, and what you can
plan against:

- **A minor release may change behaviour, and the changelog says which.** `CHANGELOG.md` is written
  to be read before upgrading — it carries scope corrections and retractions alongside the additions,
  because a number that quietly changed meaning is worse than one that broke loudly. 0.5.0 moved the
  marketing-mix headline figures by fixing the integrator the fit used; that is the kind of thing it
  records.
- **A patch release changes no signature and no number.** 0.5.1 fixed a wrong `__version__` and
  nothing else.
- **Renames get one minor of alias.** Removals get one minor of `DeprecationWarning` first, naming the
  replacement in the message.

Three tiers, by what a break costs you:

| tier | modules | promise |
|---|---|---|
| **stable** | `dynamics` `integrate` `cost` `control` `plan` `barrier` `residual` `lqr` `mpc` `train` `adjoint` `decision` `panel` `graph` `dynamics_id` | the plant/control spine and the façade over it. Breaking changes wait for 1.0 and get a deprecation cycle |
| **evolving** | the estimator, certificate and domain layers — `causal` `sensitivity` `uncertainty` `regret` `spine` `irf` `did` `scm` `matching` `marketplace` `mmm` and their neighbours | may gain keyword arguments in a minor; defaults may change with a changelog entry arguing why |
| **experimental** | modules that exist to carry one research result — `deep_galerkin` `galerkin` `transport` `meanfield` `games` `epidemic` `discovery` `symbolic` `koopman` `surrogate` `flagship` `benchmark` `causal_bench` `lalonde` `mintime` — and modules built ahead of the release that verifies them — `experiment` `switchback` on a plant CHC did not write, `zones` in the pre-registered study it is the plant of, `dlm`, `lift` and `allocation` on an external media-mix generator, `misspecification` on the zone plant, `response` in the plans it warm-starts, `geo_world` in the geo models scored on it | may change or be withdrawn in any release. Pin an exact version if you depend on one |

Twenty-four entries in modules of other tiers are experimental on the same terms, for the same
reason:
the `weights` argument of `fit_causal_residual` and `solve_channel_moment`, its `influence` argument
with the fields it keeps, the `unmoved` field of `CausalDynamicsFit`, and `omitted_confounder_bound`
and `OmittedConfounderBound` in `dynamics_id`,
`callaway_santanna_inference` and `EventStudyInference` in `did`, `synthetic_control_inference` and
`SyntheticControlInference` in `scm`, `gcm_test` and `GcmTest` in `independence`, `LoggerCheck` in
`evaluation`, the `logger_check` field of `Prescription` and `PlanEvaluation`, `shadow_price_effect`
and `shadow_price_interval` in `matching`, `channel_drift_evalues`, `DriftAlarm`, `channel_move`,
`ChannelMove`, `MovePrice` and the `alpha_futility` field of `GateConfig` in `gate`, and
`CausalPlan.decision_weight`, `DecisionWeight` and `CausalPlan.relaxed_cost` in `plan`.

Roadmap: **0.10.0** media mix — the discount dynamic linear model verified on an external
generator, response curves with one definition of ROI, a budget in `prescribe`, and lift tests as
identification → **0.11.0** the marketplace study, and the adaptation modules leave experimental →
**1.0.0**, which is when the stable tier stops moving and the top level holds only the lifecycle's
names: the 421 names that warn since 0.8.0 leave it then.

Supply chain: every artifact carries a PEP 740 attestation and a SLSA build provenance; see
[`SECURITY.md`](SECURITY.md) for how to verify one and what is in scope for a report.

## Contributing and citation

[`CONTRIBUTING.md`](CONTRIBUTING.md) lists the gates a change has to pass — ruff, `ty`, the pytest
suite, and `rocq compile` over `proofs/*.v`. Machine-readable citation metadata is in
[`CITATION.cff`](CITATION.cff); every release is archived on Zenodo.

<!-- --8<-- [start:bibtex] -->
```bibtex
@software{gradina_causal_hybrid_control,
  author  = {Gradina, Ilia},
  title   = {causal-hybrid-control: physics-structured dynamics with a learned causal residual},
  year    = {2026},
  version = {0.9.0},
  doi     = {10.5281/zenodo.21737789},
  license = {MIT},
  url     = {https://github.com/causaldyn/causal-hybrid-control}
}
```

The `doi` is the *concept* DOI: it resolves to the newest release rather than freezing at the
`version` above, so a reader following the citation lands on current code.

<!-- --8<-- [end:bibtex] -->

## License

MIT © Ilia Gradina
