# Why, and when not to use it

## The gap

Each of the three obvious approaches solves a different problem and leaves the same gap:

- **Forecast + argmax.** Optimises a correlation. Under a confounded logging policy the fitted
  action response *is* the observational one, so the planner acts on the wrong effect and never
  notices. [Tutorial 1](tutorials/01_causal_vs_predictive_control.md) shows the sign flip and what
  it does to control.
- **Causal effect estimation alone** (DML, staggered DiD, synthetic control, causal forests).
  Returns an effect, not a decision. The effect can be dynamic — EconML's `DynamicDML` estimates
  each period's within the logged horizon — but the estimator chooses no actions: no actuation
  limits, no plan past the logged horizon, no notion of a state the action must not reach. `chc`
  treats these as *backends* (`chc.estimators`) rather than rivals.
- **Offline RL / MPC on a learned model.** Fits the dynamics by residual MSE, which is not
  identification, and calibrates its pessimism to sampling noise rather than to unmeasured
  confounding — so the uncertainty penalty shrinks with `N` while the bias does not. Delphic offline
  RL (Pace et al., 2024) is the exception: it penalises the uncertainty that hidden confounding
  leaves.

What is here is the seam: identify the *interventional* control channel from confounded logs
([`chc.dynamics_id`](api/dynamics_id.md)), plan against it under constraints, and price what
unmeasured confounding can do to the plan's performance ([`chc.sensitivity`](api/sensitivity.md))
and to its safety ([`chc.barrier`](api/barrier.md)).

## Where it sits

Read by the two questions this library refuses to merge: does the tool **identify** the effect of
an action from data a policy generated, and does it **certify** the plan it hands you. Most tools
answer one. Of the packages here only the closed `decisionOS` claims both; elsewhere both are
answered in papers. Versions and licences read 2026-10-06; what each of CHC's certificates bounds is
in [the scopes](concepts/certificates.md#scopes).

| | what it is for | identifies an interventional effect | produces a schedule | ships a certificate | licence |
|---|---|---|---|---|---|
| **`chc`** | decisions from a confounded log, over a plant | yes, the **control channel** of a control-affine residual (cross-fit Robinson DML), and it says `not_identified` rather than guessing | yes, projected gradient over a box and linear constraints (budgets, rate limits), with a confounding-robust barrier held in the solve | yes — identification status, trajectory tube, barrier `Γ*`, each with [its scope](concepts/certificates.md#scopes) | Apache-2.0 |
| **DoWhy / DoWhy-GCM** 0.14 | identify and refute an effect on a DAG | yes — back-door, front-door, IV, and the Rotnitzky–Smucler **efficient** backdoor set, which minimises asymptotic variance among backdoor sets. CHC's `CausalGraph` answers the other question, Perković et al.'s canonical set, which is valid **iff any observed set is** | no | refutation tests, not a control guarantee | MIT |
| **EconML** 0.17.0 | heterogeneous treatment effects, DML/DR/orthogonal forests | yes, static **and** sequential: its DML estimators return a matrix θ(X), and `DynamicDML` estimates each period's effect of an adaptively assigned treatment on the last period's outcome (sequential ignorability, a linear-Markov state, a balanced panel). CHC applies the same orthogonal moment one step at a time to a control-affine plant | no — it chooses no sequence | confidence intervals | MIT |
| **DCBO**, last commit 2023-04 | sequential interventions in a time-varying SCM | yes, by GP emulation over an SCM | yes, a sequence of interventions | regret empirics, no feasibility guarantee | **ambiguous**: MIT in `LICENSE`, GPL-3.0-or-later in the README and `setup.py`; research code, not on PyPI |
| **Google Meridian** 2.1.0 | Bayesian marketing-mix modelling | partially — priors and geo experiments calibrate it; the estimand is the media response | no — it splits a budget across channels and holds each channel's flighting over geos and time fixed; a flighting can be supplied, not optimised | posterior intervals, and a model review of five checks | Apache-2.0 |
| **PyMC-Marketing** 1.2.0 | Bayesian marketing-mix modelling, with time-varying media and intercept | partially — lift tests enter the likelihood; the estimand is the media response | a budget split across channels, and across the dims of its multidimensional model, with value-at-risk, CVaR and Sharpe utilities; a flighting is supplied, not optimised | posterior intervals | Apache-2.0 |
| **Robyn** 3.12.1, R | marketing-mix modelling by ridge regression under an evolutionary hyperparameter search | partially — lift tests calibrate the search; the estimand is the media response | a budget split across channels (`robyn_allocator`), not a schedule over time | no | MIT |
| **Orbit** 1.1.5.1 | Bayesian time-series forecasting, with time-varying regression (KTR) | no — a regression on the log | no | posterior intervals | Apache-2.0 |
| **do-mpc** 5.1.2 | robust and economic nonlinear MPC | **no** — the model is yours and assumed correct | yes, and more general constraints than CHC's: nonlinear path constraints on the state itself, where CHC holds linear rows on the actions and a barrier's decay condition | robust multi-stage MPC guarantees, under a correct model | **LGPL-3.0** |
| **d3rlpy** 2.8.1 | offline deep RL from logged trajectories | no — conservatism bounds value error, not confounding | yes, a policy | pessimistic value bounds | MIT |
| **causaLens `decisionOS`** | enterprise causal decision platform | yes, per its own account | yes | not publicly auditable | commercial, closed |

Two rows that are **not** here, and the reason is the same. The 2024–25 literature on causal
Bayesian optimisation under safety constraints, and on causal optimal control ("COAST"-style), has no
shipped, installable implementation this could be run against. That is the gap CHC is aimed at,
and stating it as an absence is more honest than a row of dashes against a paper.

Where a row says *no* it is not a criticism: do-mpc solves control problems CHC cannot state, and
EconML answers effect questions CHC does not ask. The EconML row's *yes* was run, not read:
[`scripts/econml_reference.py`](https://github.com/causaldyn/causal-hybrid-control/blob/main/scripts/econml_reference.py)
gives `LinearDML` a 2×2 channel `B(x)` behind a confounded action, which it recovers to 0.068 at
worst, and `DynamicDML` a linear plant logged over three periods, whose impulse response it
recovers to 0.013, every period's 95 % interval covering the truth (EconML 0.17.0, 2026-10-06).
The claim is narrower — that **going from a confounded log to a schedule and its certificate in
one place** is what nothing above does end to end.

## When not to use it

The same table, read the other way.

- **Your model is known and trusted, and the constraints are nonlinear in the state.** That is
  do-mpc's problem, and it states control problems CHC cannot: CHC holds linear rows on the actions
  and a barrier's decay condition, not general path constraints on the state. CasADi, acados and
  do-mpc are far stronger solvers, and CHC has nothing for embedded MPC.
- **You need an effect, not a decision.** Heterogeneous effects are EconML's question, and within
  the logged horizon so are dynamic ones: `DynamicDML` estimates each period's effect with
  intervals, and CHC adds nothing there. Identifying and refuting an effect on a DAG is DoWhy's.
  CHC calls such estimators through `chc.estimators` rather than competing with them.
- **You need sequential off-policy evaluation beyond a linear-Gaussian plant.** `chc.evaluation`
  has per-decision and marginalised importance sampling, doubly robust and fitted Q, and certifies
  a log in advance only on a linear-Gaussian plant with affine policies; SCOPE-RL 0.2.1 has more
  sequential estimators, the DICE family among them.
- **You want a Bayesian marketing-mix model** calibrated by priors and geo experiments: that is
  Meridian. The [media-budgets case study](case-studies/media-budgets.md) is about the other half —
  scheduling spend when the log was planned against demand. Planning that schedule through the
  adstock state is Nerlove and Arrow (1962); the shipped tooling is new, the idea is not.
- **You can run experiments on the system.** Causal Bayesian optimisation intervenes and learns
  from each intervention; `prescribe` reads a log and never experiments. On DCBO's own three dynamic
  SCMs ([Track L](https://github.com/causaldyn/causaldyn-bench/blob/main/results/track_l.md) of
  `causaldyn-bench`), `prescribe` from the log alone is level with DCBO on the stationary one and
  behind every method that experiments, plain BO included, on the other two. The track names what
  it lacks there: a *minimise* objective, a choice of which variables to intervene on at each step
  (`max_levers` chooses one set for the whole horizon), experiments, and a model that changes in
  time. The converse holds too: DCBO reads the true SEM for every explorative evaluation, so it
  cannot run from a log alone.
- **Your residual is not control-affine.** `chc.dynamics_id` is the identified route, and it is
  restricted to control-affine residuals; outside that class this library offers a sensitivity
  radius (`chc.sensitivity`), not an unbiased estimate.
- **Nothing in your log identifies the effect** — no adjustment set and no instrument. Then there is
  no schedule to be had: `prescribe` returns none, on purpose, and what remains is to price the
  radius ([the sensitivity level Γ](concepts/gamma.md)).
- **You need a frozen API today.** This is early, single-author research code before 1.0. The
  [stability tiers](api/index.md) say what each module promises; an experimental module may change
  or be withdrawn in any release.

## Honest positioning

`chc` composes ideas that exist — hybrid dynamics (SciML UDE), pessimistic offline control
(MOPO/MOReL/Delphic), sequential causal identification (g-methods / dynamic treatment regimes),
differentiable control (Neuromancer). The contribution is the *integration behind one API* plus a
benchmark with ground-truth interventional effects. KAN is **one interpretable residual backend**,
not the identity of the framework — and `chc.symbolic` makes "interpretable" checkable rather than
asserted, including the two places where the interpretation stops being valid.

Worth knowing before you rely on a fitted model: **a low residual MSE is not causal
identification.** Fitting `r_θ` by prediction error recovers the *observational* control response,
which is the wrong one whenever the logged action was chosen from something that also moved the
state, and no amount of extra training fixes it because it is not a fitting problem. See
[identification](concepts/identification.md).
