# What's inside

Every public module, grouped by what it is for. Each section names its modules, says what they do,
and quotes what was measured. Each module's API page states its stability tier; the
[API reference](api/index.md) says what each tier promises.

## Hybrid dynamics

[`dynamics`](api/dynamics.md), [`residual`](api/residual.md), [`integrate`](api/integrate.md),
[`symbolic`](api/symbolic.md)

The plant is `ẋ = f_known + r_θ`: a known mechanism plus a learned residual, rolled out with RK4.
The residual is one of:

- an MLP, an **RBF-KAN** or a graph network;
- **control-affine**, `a_θ(x) + B_θ(x)u`, the class the identification and safety layers share;
- **port-Hamiltonian**, passive by construction, with `energy="icnn"` for a *coercive* one;
- **Lipschitz-certified**;
- **spectral**.

What passivity is worth: `H' <= 0` is an identity of `(J - R) grad H` in which `H` never appears, so
it holds for any energy network and separates none of them (`2.7e-15` for both arms). What it
confines the state to is `{ H <= H(x0) }`, and a `tanh` energy read out linearly is bounded in `x`,
so that set is the whole space and `invariant_radius` is honestly `inf`. The input-convex energy
with a quadratic floor makes it a ball: realised excursion `6.00` inside a predicted `25.81`, and
one critical point instead of three.

The spectral residual *is* a circulant on a periodic grid, so its operator norm `max_k |lambda_k|`
is *attained* rather than bounded (the Lipschitz backbone's Schur bound measures 113x slack). It is
translation-equivariant to machine precision, and, being linear in its kernel, it is fitted by a
closed-form per-mode least squares rather than by Adam.

`symbolic` turns a fitted RBF-KAN edge back into a closed form and states what that is worth. The
intercept is a gauge: only the total is identified. The layer is additive, so an interaction has a
*proved* error floor `r²` on `[−r,r]²`. And the extracted formula extrapolates where the layer,
its RBF support gone, does not: 4.22e-4 against 34.65.

## Gradients and the classical baseline

[`adjoint`](api/adjoint.md), [`lqr`](api/lqr.md)

The discrete adjoint, verified against autodiff and finite differences. LQR and AKOR by the Riccati
equation: the `r_θ → 0` limit, and the correctness baseline for everything above it.

## Identification

[`train`](api/train.md), [`dynamics_id`](api/dynamics_id.md), [`causal`](api/causal.md),
[`estimators`](api/estimators.md), [`gmethods`](api/gmethods.md), [`frames`](api/frames.md)

System identification, one-step and multi-step. A pluggable effect backend: adjustment, **IV/2SLS**,
**DML**, sensitivity and refutation, with optional **EconML** and **DoWhy** adapters. Robins'
**g-formula**, cross-fitted, for a treatment *sequence* under time-varying confounding.

`dynamics_id` is the module that makes the *plant* causal. Prediction-error fitting learns the
**observational** control channel, so under a confounded logging policy the planner inherits the
bias: measured, a channel of `0.02` where the truth is `1.0`. `fit_causal_residual` estimates it by
Robinson partialling-out lifted to a state-dependent matrix: channel error `0.003`, control regret
`0.014` against the biased fit's `6.20`. When the confounder is never logged it uses 2SLS, at a real
variance premium (`0.10` error, regret `0.13`), because the shifter explains only 18 % of the
action. It reports `identified=False` instead of a confident wrong answer when nothing in the log
can pin the channel down.

Data goes in as a mapping of arrays, a **pandas** frame or a **polars** frame. `frames.as_columns`
recognises a frame structurally and normalises it once at the boundary, so neither library is a
dependency of the wheel. `uv run python scripts/dynamics_id_demo.py`

## Planning and control

[`cost`](api/cost.md), [`control`](api/control.md), [`mpc`](api/mpc.md),
[`splitting`](api/splitting.md), [`plan`](api/plan.md)

A Bolza objective. Projected-gradient optimal control and a bound-constrained quasi-Newton
(`lbfgs_box_control`), sharing the same discrete-adjoint gradient, with `box_stationarity` as the
reference-free convergence measure. Receding-horizon MPC: `mpc_control` against a simulated plant,
and `RecedingHorizon.step(x)` online, one `causal_plan` per measured state, warm-started from the
last plan and its barrier multipliers. **Strang–Marchuk** splitting.

`causal_plan` is the one-call spine. It returns a plan *with* its error tube (RK4, at a rate or a
state matrix you supply) and the horizon the tube trusts. Three modes are named apart on purpose:

- **plan**: `causal_plan`, with the box and linear constraints (budgets, rate limits) in the solve,
  and with `barrier=` the audit's own condition held by augmented-Lagrangian rounds;
- **audit**: `certify_safety`, read-only on a finished plan;
- **filter**: `robust_safety_filter`, one action at a time, online.

The tube never enters the *objective*, and a barrier does only when given one. Even then the
verdict is the audit's, which the plan carries as `plan.safety`, because a solve stopped by its
budget can come back short of the condition. With no error model supplied, the certificate reports
`not_evaluated` rather than a vacuous full-horizon pass.

## Decisions from a panel

[`decision`](api/decision.md), [`panel`](api/panel.md), [`graph`](api/graph.md)

The façade: `prescribe(panel, adjustment=graph, ...)` runs adjustment set → causal channel →
constrained plan → safety certificate as one call, and adds no new estimator, solver or guarantee.

`graph.CausalGraph` derives the adjustment set, the canonical Perković–Textor–Kalisch–Maathuis
set, which is valid **iff any observed set is**, instead of asking a caller to type
`covariates=("x", "z")` by hand. So a collider is never adjusted for and a mediator never removed:
the two failures that raise nothing and change the number. `panel.Panel` checks a long panel's
`(unit, time)` index once and does the wide pivot that DiD and synthetic control need, naming the
column *and* the entity when it refuses.

`DecisionCertificate` keeps identification and certification apart, and an unidentified effect
yields **no schedule at all** rather than one that looks like every other schedule. Measured on one
synthetic plant read three ways: the adjusted channel is `0.84` against a true `0.80`; an empty
adjustment set asserted gives `2.15`; with the confounder latent there is no schedule.

## Offline safety

[`support`](api/support.md), [`evaluation`](api/evaluation.md), [`offpolicy`](api/offpolicy.md),
[`uncertainty`](api/uncertainty.md)

- A pessimism penalty.
- A plan's **deployed** value from logs: per-decision importance sampling, marginalised importance
  sampling, doubly robust and fitted Q. It is **certified before any cost is read** on a
  linear-Gaussian plant with affine policies, and refused by name when a weight has infinite
  variance.
- One-step IPS and SNIPS off-policy values, with an overlap flag.
- **Calibrated** uncertainty: a deep ensemble with split-conformal calibration, which covers
  marginally on exchangeable data, from `(1 - alpha)/alpha` calibration scores. A
  **time-consistent nested-CVaR** aggregates that disagreement, where the risk-neutral sum averages
  one very bad step away.
- A **Wasserstein-1 DRO** margin for distribution shift.
- **Rollout error tubes**: RK4 recursions at a Lipschitz rate or a state matrix, or Euler's on
  request, giving a time-varying uncertainty tube, safety tightening, and the horizon the tube
  trusts. The tube is a bound where the per-step error is one, and a scale where it is a standard
  error. Rocq proves that the recursions bound the error under those premises.

The ensemble trains as one sharded program (`vmap` over a member axis, `lax.scan` over the Adam
steps, `NamedSharding` over the device mesh) rather than K sequential fits: 3.6× on one device and
10.3× over an 8-device mesh at K=8, agreeing with the serial recursion to 232 ULP.

## Hidden confounding

[`sensitivity`](api/sensitivity.md), a façade over `regret`, `uncertainty` and `barrier`

**Control under hidden confounding.** A bounded-density-ratio (MSM) CVaR worst case becomes an
inflated pessimism radius. The confounding-regret floor is *second-order* in the effect bias. A
**minimax controller** shifts the gain under asymmetric (over- or under-shoot) loss and beats
certainty equivalence once the confounding is real, at a premium where there is none. It is also a
**closed-loop** controller on a confounded dynamic plant: over 30 steps it cuts the worst case
across the confounding sweep from 18.84 to 4.84, and costs 82 % less than certainty equivalence at
a confounding of 0.8. A `ConfoundingRobustPenalty` carries the sensitivity radius into the general
pessimistic-control stack. Their algebraic sign facts are **Rocq-checked**, with the loss weights
and the radius taken as inputs.

`Γ` itself is **calibrated before it is spent**. `benchmark_gamma` prices it in units of the
confounding the observed covariates carry, as an exponent, `log Γ / log Γ_strongest`, because odds
ratios compose. `negative_control_gamma` inverts a known-null outcome for the smallest `Γ` that
reconciles it: a *lower bound* on the confounding present, or `inf` when the model class is refuted
instead. `chc.sensitivity` is the one-import surface: calibrate, then radius, then control.

## Safety under partial identification

[`barrier`](api/barrier.md), [`plan`](api/plan.md)

The same sensitivity radius, spent on a **constraint**: a robust control-barrier margin, a
least-restrictive safety filter (a closed-form certified action interval, no QP), and `Gamma*`,
**the largest level of the marginal sensitivity model under which the barrier stays certified**.
`Gamma*` is a model parameter, not measured confounding; Rosenbaum's `Γ` is another model's, and
[the sensitivity level Γ](concepts/gamma.md) gives the bracket between them.

Safety degrades at *first* order in the effect bias, until the radius swallows the channel and the
loss saturates, where performance regret degrades at second: the envelope theorem protects
objectives, not binding constraints. Its algebraic core is **Rocq-checked**. In closed loop a
regret-sized budget violates the limit on 93 % of steps, where the constraint-sized one never does.

`certify_safety` audits a finished plan against all of it, pointwise along the model's predicted
path: a filter rather than a proof of invariance. It reports the certified prefix next to the
plan's `Gamma*`, exactly the weakest step's. `causal_plan(barrier=BarrierConstraint(...))` holds
the same worst-case condition inside the solve instead of pricing it after.

## What the certificate is worth

[`reachability`](api/reachability.md)

The **Hamilton–Jacobi** answer the barrier only approximates: `V(x,T) = max_u min_{ΔB} min_s
h(ξ(s))` on a Lax–Friedrichs grid, with the identification radius as the adversary. It is the same
robust-margin algebra as `barrier`, but `p = ∇V` is *solved for* rather than assumed. It turns the
control-barrier theorem into an executable check (the condition on all of `{h ≥ 0}` implies the tube
**is** `{h ≥ 0}`) and prices what pointwise certification misses. On a relative-degree-2 barrier the
pointwise verdict is identical at every radius while the true tube shrinks (6.4 % of the grid
certified and unreachable), so `certify_safety`'s per-step prefix is a filter, not a proof.
`uv run python scripts/reachability_demo.py`

## Regret guarantees

[`regret`](api/regret.md)

- The LQ certainty-equivalence bound, quadratic in the model error (Dean, Mania, Tu, Recht and
  Matni).
- An **interference-aware regret certificate**, with an extra exposure-map-error term; its algebra
  is **machine-checked in Rocq** from an assumed bound on the planned policy's error.
- The **van Trees floor on control regret**: `multivariate_action_floor` for a matrix effect, where
  the bound is a trace and confounding is priced by its *alignment* with `du*/dθ` rather than by a
  ratio.
- `capped_exploration_policy`, which takes a per-round cap **schedule** and a spending **budget**,
  and stops on a delivered exploration *mass* rather than a round count.

## Experiments, switchbacks and deployment

[`experiment`](api/experiment.md), [`switchback`](api/switchback.md), [`gate`](api/gate.md),
[`misspecification`](api/misspecification.md)

**Which experiment, if any, to run before a one-shot decision over zones** (`experiment`). Units per
zone come from the **decision weight**: the Hessian of the certainty-equivalent regret, averaged
over the prior's sigma points, with a Gaussian-tail price for a lever pinned at its bound and
reverse water-filling at the price of information. Then **Monte Carlo re-solves the decision on
every draw**, and every number reported is that measurement. The verdict is deploy, experiment or
abstain, with the reason. At one budget on the lab's four-zone market, the decision-weighted
allocation leaves `1.18e-3` of regret where classical Neyman allocation leaves `1.84e-3`.

**Which switchback to run on a plant with memory, and how to read it** (`switchback`), for a
**named** effect: the channel, the effect of holding the lever on `H` periods, or the steady state.
The design is a Markov chain **aligned to the effect**, so the persistence estimate is orthogonal to
it (at two periods it flips with probability `a/(1+2a)`). It handles IV under measurement noise,
block or local-projection readings when the state model is not trusted, and gives the standard
error, the MDE and the loss at the least favourable persistence. `read_switchback` reads the effect
off the data: robust or HAC standard errors, Fieller's interval for the steady state, and a block
difference centred on earlier blocks, unbiased at any number of blocks where the realised-means form
is 27 % low at eight. Planned standard errors were 0.99–1.04 of the simulated spread, and the data's
intervals covered 0.937–0.959 of 0.95. On the market of `zones`, whose state is not first order, the
plug-in read 3–14 % off and its first-order test warned on 99.9–100 % of runs. The model-free
readings covered 0.944–0.953, but the plan's variances are the working model's, and power held
within three points of nominal only for the channel and, at the edge, `tau_2`.

**A candidate policy in shadow of the baseline** (`gate`), per zone, until an **anytime-valid** gate
says deploy, hold, experiment or roll back. At every read, including one chosen by looking at the
data, the expected share of wrong deploys across zones stays at most `alpha` (e-processes over the
logged propensities, e-BH over e-values frozen at selection). It takes the three propensities rather
than weights, and refuses a batch whose logged ones are not those of the policy it asked to log. Its
conditions — logged propensities, no spillover, no carryover — are in the docstring.

**Whether a plan pays for its model class being wrong** (`misspecification`): two fits of one class
on one log, their difference priced in the plan's own regret, and tested against the law it has
when the class holds — a Hausman test in the plan's metric.

## End to end

[`spine`](api/spine.md)

All four layers on **one** decision: confounded logs → causal gain → constrained plan → `Gamma*`
certificate → the same plan run on the *true* plant. Two zones of a mobile driver pool, one
incentive lever whose `[+b, -b]` column is driver conservation, and a supply floor in the zone it
drains. The confounded arm plans 13.59 and pays 38.96. `Gamma*` tells the two arms apart (7.46
against 1.17) **before either acts**, without ground truth. It is a ceiling for the problem along
each plan's path, not for the plan: the adjusted plan's own actions certify 6 of 25 steps, and it
crosses the floor at step 16. `uv run python scripts/spine_demo.py`

## Case studies in code

[`mmm`](api/mmm.md), and a script over [`decision`](api/decision.md)

**Media budgets** (`mmm`): marketing-mix budget scheduling on a saturating carryover plant, where
the confounding is not hypothetical: spend is planned *against demand*, so a model fitted on the
log credits the channel with the season. Each channel acts twice: an immediate incremental return
(the **control channel**, and what gets confounded) and a carried-over one through a saturating
adstock (the **drift**); the mechanical adstock rows go in as `known=`. Two readings of one log and
three fixed rules, every arm audited on the *true* plant. At matched budget the prescribed schedule
buys **+4.2 % more lift** (cumulative sales over the do-nothing arm) than an equal split, and the
myopic rule on the same fit, which only splits each week's budget across the channels differently,
buys +7.3 %: on this plant the gain is the identified allocation, and the whole-horizon plan buys
2.9 % less than the myopic rule. The confounded arm inflates the channels `2.5–15.4x`, under-invests
by 14 % and buys 87 % of the lift. It also shows why return per unit of spend is the wrong
scoreboard: under diminishing returns the confounded arm scores *higher* on it while earning less.
`uv run python scripts/mmm_demo.py`

**A pendulum**: Pendulum-v1 raised 0.30 rad from hanging and held there, planned by `prescribe`
from a log in which an operator cancelled half the wind torque it measured. So the logged torque
moves with a disturbance that enters `omega'` exactly where the torque does, and the confounding
sits on the control channel itself. Gravity goes in as `known=`, the actuator is what the log has
to identify, and every schedule is run on the *true* pendulum. Adjusted for the wind (the graph
derives `{wind}`), the torque channel is `+3.001` against a true `3.000`, and the schedule ends
`0.004` rad from the target, with the speed held within `1.0` rad/s in the solve as the barrier.
Asserting an empty adjustment set returns `-1.090` (the white-wind omitted-variable formula
predicts `-1.052`), so the plan pushes the wrong way and ends `0.608` rad off, `2.0x` as far as
never acting; declared latent, the wind leaves no schedule. The certificate trusts the adjusted
schedule for 9 steps of 40 and the asserted one for none. Printed in float32, the default, and the
script says which of its numbers float64 moves. `uv run python scripts/pendulum_demo.py`

## The causal frontier

[`did`](api/did.md), [`scm`](api/scm.md), [`estimators`](api/estimators.md),
[`causal`](api/causal.md)

Callaway–Sant'Anna staggered **DiD**; **augmented synthetic control**; the **R-learner** for
conditional effects; **E-values** beside Cinelli–Hazlett; **influence-function confidence
intervals** on cross-fit DML.

## Dynamic effects and delays

[`irf`](api/irf.md), [`toeplitz`](api/toeplitz.md), [`delay`](api/delay.md)

Impulse-response and local-projection dynamic effects; Toeplitz, Levinson–Durbin and
Gohberg–Semencul operators.

A discrete delay as a **plain `Dynamics`**: the m-stage linear chain, so `rollout`, the adjoint,
both solvers, `causal_plan`, `certify_safety` and `mpc_control` run on a delayed plant unchanged
(augment, don't write a DDE solver). The exact **delay margin** `arccos(a/K)/sqrt(K²−a²)`. A causal
**delay estimate with a moving-block bootstrap interval**. And the stabilising ball in *delay*
space, which is a **half-line with a relative radius** whose performance loss is a **square root**
on one side and linear on the other: the decay-optimal design sits at a defective root, so neither a
symmetric ball nor a single regret constant exists. `robust_delay_design` turns an interval into
the minimax design, which is *below* its centre. The margin's algebraic core is **Rocq-proved**,
and its smallest crossing derived in Maxima.

## Structure discovery

[`discovery`](api/discovery.md), [`independence`](api/independence.md),
[`network_causal`](api/network_causal.md), [`pathway`](api/pathway.md)

Lagged-parent discovery; the MCI partial-correlation test; orthogonal DML on a network with
spillover; and a **ranked temporal causal pathway**: which lagged variables and multi-step chains
drive a target, signed and actionable, with Rocq-certified walk-sum, geometric-truncation and
weakest-link laws.

## Advanced control

[`koopman`](api/koopman.md), [`meanfield`](api/meanfield.md), [`transport`](api/transport.md),
[`matching`](api/matching.md), [`games`](api/games.md), [`mintime`](api/mintime.md)

- Koopman-LQR.
- Mean-field control.
- A periodic **advection-diffusion** field with an exact spectral propagator: the
  translation-invariant plant that justifies the spectral residual.
- Continuum and discrete **Kantorovich optimal transport**: driver-to-rider matching, with **dual
  surge prices**.
- Differentiable Stackelberg games over a **certified** congestion equilibrium: implicit-function
  gradients, a contraction certificate and optimal damping. The solver reports its residual instead
  of silently returning a non-equilibrium.
- Time-optimal bang-bang control by Pontryagin's maximum principle.

## Media mix

[`dlm`](api/dlm.md), [`response`](api/response.md), [`lift`](api/lift.md),
[`allocation`](api/allocation.md), [`geo_world`](api/geo_world.md)

**Coefficients that move** (`dlm`). West and Harrison's **discount dynamic linear model**: a level
or a trend, Fourier seasonality over a period that need not be an integer, and random-walk
coefficients on media, each block with its own discount. The observational variance is learned in
closed form. It gives one-step Student `t` forecasts and their log-likelihood; a smoother whose
covariances are in the units of the variance the whole series supports, and whose lag-one
covariances give a period's total an interval; backward sampling; `k`-step forecasts under a stated
evolution policy; and e-values of the one-step errors for `DriftAlarm`. The filter matches PyBATS to
`2.7e-16`. The one-step forecasts cover 0.90 of 0.90. A channel's 13-week contribution, with the
discounts picked by that likelihood, covers 0.78, and 0.88–0.90 with the coefficients' discount
fixed at 0.85–0.9: a period's return is not reported from the likelihood's pick alone.

**Over geos** (`forward_filter_geos`): national blocks every geo reads, and regional blocks each geo
has its own copy of, so that a geo's coefficient is the national one plus its deviation, filtered as
one state with one variance scale. One geo is the filter's own to `1e-12`, and with nothing
discounted many geos are the conjugate regression on every observation to `1e-10`. `smooth` and
`backward_sample` run back over the stacked state, the smoother the joint Gaussian posterior of
every state to `1e-8`. `fit_geo_spread` chooses the geos' spread by that likelihood and draws it
from its posterior, so that an effect's interval carries the spread's uncertainty. On
`chc.geo_world`'s worlds pooling puts each geo's error 9–11 % below its own alone at a noise of 30,
and 33 % at 100, and the intervals mixed over the spread's draws cover 0.88–0.89 of 0.90.
`uv run python scripts/bench_dlm.py`, `scripts/bench_geo_dlm.py`

**Response and carryover** (`response`). **Seventeen saturation families** under one contract: zero
at zero spend, increasing, a ceiling of 1, and one scale `K` in currency, so a change of currency
moves `K` alone. They are Michaelis–Menten, exponential, tanh, arctan, algebraic and half-normal;
Hill, Weibull, logistic, Gompertz, Richards, Chapman–Richards, gamma, log-normal, Burr XII, beta and
Kumaraswamy CDFs; plus `log(1 + x/K)`, constant elasticity and Ricker's inverted U. Each S-shaped
family states its **inflection** and the **tangency** of its concave envelope, in closed form where
one exists (Maxima) and as a root otherwise. Given no warm start, `causal_plan` plans on each such
curve's concave envelope first (`relax`), since on an S-curve `h'(0) = 0` and a descent from zero
spend stops there, and reports the relaxed problem's cost, which bounds the plan's distance from the
best. **Carryover kernels** (geometric, delayed peak, Weibull) each have a length of their own, so
logging another period moves no earlier adstock. A `Channel` reads spend through a kernel and a
curve, and there is **one definition of return**, carryover included: `contribution`, and per
currency unit `roi`, `marginal_roi` and `steady_state_marginal_roi`. Mapped onto PyMC-Marketing's
transforms and run beside them (`scripts/pymc_marketing_reference.py`).

**Lift tests** (`lift`): **geo tests read as structure**. A channel's carryover, curve and size are
fitted to the gap between a test's groups period by period (Heusch's differencing equation), by
least squares over every test with one noise scale, each parameter with a profile-likelihood
interval that says when the tests leave it open. The coefficient is projected out and the rest fitted
in logs, so the fit follows the ridges a line and a step make; the adstock the readouts covered is
the only range over which the curve is identified. On Heusch's endogenous-spend generator, written
clean-room in `causaldyn-bench`, with his four go-dark tests at his noise, the carryover's 95 %
interval covered 0.954 of 500 histories; at three times the noise it still covers, and 15 % of those
intervals close. `just track-m2-lift` in `causaldyn-bench`

**The budget's split** (`allocation`): a quarter's budget across channels, one spend a period for
each `chc.response` channel within its box, for the most return, with the history's carryover
running into the plan and the plan's running on after it. Exact where every curve is concave, by
bisection on the budget's price, a linear channel's jump spent to the budget exactly, and the
budget's shadow price reported. S-curves are searched in boxes to the best split by `allocate` and
`cvar_allocate`, and planned on their concave envelopes by the others, each such plan's gap logged.
The box is required: it is where the channels were seen. **Every geo's channels under one budget**
(`allocate_geos`): caps and floors on what each geo and each channel spends, with a price for each,
certified by cutting planes and made exact by Newton's method on the totals that bind. A return, a
marginal return or a return on spend can stand over the grid in place of the budget
(`budget_for_geos`), met in a few plans. [Tutorial 8](tutorials/08_splitting_a_budget.md) splits a
budget step by step.

**A geo world** (`geo_world`): geos to score a geo model and a geo plan on. Each geo's KPI is made
by its own media, each geo's effect on a channel drawn log-normal about the channel's national
median with a stated spread, and populations and bases per head likewise. The lift is per head: the
curve per head of the spend per head through a normalised carryover, so a geo twice the size
spending twice as much lifts twice as much. National media are allotted by population; the log's
spend rises with the season by a policy, the confounding; and spillover into a geo's two neighbours
on a ring, and a national drift in each channel's lift, are options. Every variate is drawn whatever
the parameters, so two mixes compare paired. Each geo's channels come back as `chc.response` cells
that lift exactly what the world's media lift, for `allocate_geos` to plan on.

## Marketplaces

[`marketplace`](api/marketplace.md), [`zones`](api/zones.md)

**Offline causal control under equilibrium interference** (`marketplace`): incentives learned from
confounded switchback logs where SUTVA fails. De-confounded, equilibrium-aware and W-DRO-pessimistic
control recovers the oracle where MOPO and naive causal control go *negative*.

**The zones × time plant the marketplace loop runs on** (`zones`), control-affine by construction:
idle supply, open requests and an incentive stock per zone; an incentive and a price per zone as
levers; recruits drawn from neighbouring zones, and supply that answers late. Its logged operator
raised both levers with each zone's demand shock, so a fit that ignores the shock reads a price rise
as all but free, in most zones as raising demand, and an incentive at 20–36 % of what it recruits.
Adjusted, it reads both within 23 % of one period's response, zone by zone, over six logs. It
bridges to `ZoneDecision` and `LinearGaussianPlant`, exact under linear matching.

## Evaluation

[`benchmark`](api/benchmark.md), [`causal_bench`](api/causal_bench.md),
[`flagship`](api/flagship.md), [`lalonde`](api/lalonde.md), [`metrics`](api/metrics.md),
[`surrogate`](api/surrogate.md)

Oracle-regret tasks with a leaderboard and multi-seed bootstrap confidence intervals: pricing,
inventory, support shift, **model uncertainty**, **confounding-robust**, and **causal dynamics**,
where the confounding is in the plant's own channel and the failure is invisible to the constraint
and support columns. A causal-methods table scores every frontier estimator against the naive
baseline it is meant to beat. Real-data **LaLonde** validation; step-response quality metrics; and a
gradient-boosted tree surrogate as the tabular prediction competitor (the optional `trees` extra).
The tables are on the [benchmarks](benchmarks.md) page.

## Scientific computing

[`epidemic`](api/epidemic.md), [`galerkin`](api/galerkin.md),
[`deep_galerkin`](api/deep_galerkin.md)

SIR epidemic control (flatten the curve). 1D and 2D Galerkin FEM (progonka), plus the non-symmetric
**convection-diffusion** case with the cell-Péclet threshold and the optimal SUPG parameter.

Mesh-free **Deep Galerkin**: a neural Poisson solver, and the coupled **mean-field game** (backward
HJB and forward Fokker–Planck joined by `alpha* = -(b/r)V_x` and the population mean), with both
boundary conditions structural rather than penalised. It is gated on an exact LQ closed form, which
also prices the failure: past the anti-monotone threshold `c = 1 + ra²/(qb²)` the equilibrium
degenerates at a horizon in closed form, and there the solver's own residual *falls* while its
error rises. The density it returns is the `N → ∞` limit, so the **finite-population gap** is
priced exactly rather than fitted: `E[(m_N - m)^2] = v(t)/N` at every `t` and every `N`, because a
mean-field feedback leaves the closed-loop agents independent. The a-posteriori estimator
generalises off the closed form too: `adjoint_weighted_error` linearises whatever field it is given,
so the same construction runs on a **congestion-shifted** game with no closed form, exact on the
affine problem, second-order otherwise, and one order better than reusing the affine adjoint.
