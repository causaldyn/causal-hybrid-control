# causal-hybrid-control

[![ci](https://github.com/causaldyn/causal-hybrid-control/actions/workflows/ci.yml/badge.svg)](https://github.com/causaldyn/causal-hybrid-control/actions/workflows/ci.yml)
[![pypi](https://img.shields.io/pypi/v/causal-hybrid-control)](https://pypi.org/project/causal-hybrid-control/)
[![python](https://img.shields.io/pypi/pyversions/causal-hybrid-control)](https://pypi.org/project/causal-hybrid-control/)
[![license](https://img.shields.io/pypi/l/causal-hybrid-control)](https://github.com/causaldyn/causal-hybrid-control/blob/main/LICENSE)
[![doi](https://zenodo.org/badge/DOI/10.5281/zenodo.21737789.svg)](https://doi.org/10.5281/zenodo.21737789)
[![docs](https://img.shields.io/website?url=https%3A%2F%2Fcausaldyn.github.io%2Fcausal-hybrid-control%2F&label=docs)](https://causaldyn.github.io/causal-hybrid-control/)
[![scorecard](https://api.scorecard.dev/projects/github.com/causaldyn/causal-hybrid-control/badge)](https://scorecard.dev/viewer/?uri=github.com/causaldyn/causal-hybrid-control)

**Decisions from logged data, when the data is confounded.** `chc` learns what an action does from
logs that a policy wrote, plans actions under limits, and states how far each plan can be trusted.

Most logs were written by someone acting on what they saw: an operations team that paid drivers
more when demand spiked, a thermostat that heated harder on cold nights, a retention team that gave
discounts to the customers most likely to leave. In such data the action moves together with
whatever else drives the outcome. A model fitted to predict the outcome then learns the wrong
effect, sometimes with the wrong sign, and a planner that trusts it acts on it. `chc` is a small
JAX library for the step from those logs to a decision:

1. **Identify** the effect of each action from the logs, given the causal graph you state, or say
   that the logs cannot identify it.
2. **Plan** a schedule of actions under limits (bounds, budgets, a safety constraint), on a model
   that keeps the physics you know and learns the rest: `ẋ = f_known(x, u) + r_θ(x, u)`.
3. **Certify** the plan: whether the effect was identified, for how many steps the model's error
   stays inside your tolerance, and how much hidden confounding the safety constraint survives.
   Each certificate states what it bounds, on what, and under which premises:
   [the scopes](https://causaldyn.github.io/causal-hybrid-control/concepts/certificates/#scopes).

When nothing in the logs identifies the effect, `chc` returns no plan rather than a confident wrong
one.

## The one result

On a confounded log, fitting the action's effect without adjusting for the confounder flips its
sign (true `+1.0`, naive `-0.2`). Control the true system with each estimate:

```text
controller            cost      regret    viol     ood
causal-CHC            4.59       -0.00    0.00    0.00
oracle                4.59        0.00    0.00    0.00
predictive        13740.08    13735.49    0.97    1.00
```

The causal controller matches the oracle. The predictive one drives the state the wrong way
(`x → -20` for a target of `+2`), violates the constraint on 97 % of steps (`viol`) and acts
entirely outside the logged actions (`ood`). That is one seed of the pricing task in notebook 5;
over 12 seeds the predictive controller's regret is `13734.15 [13732.55, 13735.31]`
([benchmarks](https://causaldyn.github.io/causal-hybrid-control/benchmarks/)).

## Install

```bash
pip install causal-hybrid-control    # or: uv add causal-hybrid-control
```

Python 3.11–3.15. JAX runs it on the CPU. For a GPU or a TPU, install JAX's build through the extra
of the same name, for example `pip install "causal-hybrid-control[cuda13]"`; the
[installation page](https://causaldyn.github.io/causal-hybrid-control/installation/) lists them
all. To work on the library itself, see
[CONTRIBUTING.md](https://github.com/causaldyn/causal-hybrid-control/blob/main/CONTRIBUTING.md).

## Quickstart

From a panel of logs to a schedule and its certificate, in one call. The causal graph is a required
argument: leaving it out would mean "adjust for nothing", which is itself a causal claim.

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

out.certificate.trustworthy_steps  # 15 -- identified, inside the tube's tolerance, barrier met
out.schedule.magnitudes  # (15, 1); raises instead, if the effect is not identified
print(out.report())  # Markdown: decision, both certificate axes, provenance
```

`prescribe` adds no estimator of its own. The graph gives the adjustment set, cross-fitted double
machine learning gives the action's effect, and the planner and the certificates do the rest. The
[quickstart](https://causaldyn.github.io/causal-hybrid-control/quickstart/) runs this end to end,
from simulated logs, and reads the report line by line;
[the expert path](https://causaldyn.github.io/causal-hybrid-control/quickstart/#the-expert-path)
builds the model, the cost and the controller by hand.

## Examples

Executed notebooks under
[`notebooks/`](https://github.com/causaldyn/causal-hybrid-control/tree/main/notebooks), also on the
[documentation site](https://causaldyn.github.io/causal-hybrid-control/tutorials/). New here?
Start with notebook 0.

| | notebook | what it shows |
|---|---|---|
| 0 | [Start here: a decision from logs](https://github.com/causaldyn/causal-hybrid-control/blob/main/notebooks/00_start_here.ipynb) | the whole loop on one everyday problem: the trap in the logs, the effect, the plan, and how far to trust it |
| 1 | [Causal vs predictive control](https://github.com/causaldyn/causal-hybrid-control/blob/main/notebooks/01_causal_vs_predictive_control.ipynb) | under confounding, predictive control drives the state the wrong way (to −20, target +2); causal control matches the oracle |
| 2 | [Learning what physics misses](https://github.com/causaldyn/causal-hybrid-control/blob/main/notebooks/02_learn_hidden_physics.ipynb) | a hybrid model recovers an omitted cubic term; multi-step training cuts drift from noisy measurements |
| 3 | [The causal inference toolkit](https://github.com/causaldyn/causal-hybrid-control/blob/main/notebooks/03_causal_inference_toolkit.ipynb) | adjustment, IV/2SLS, double ML, sensitivity and refutation, side by side |
| 4 | [An epidemic under a capacity cap](https://github.com/causaldyn/causal-hybrid-control/blob/main/notebooks/04_epidemic_and_pessimism.ipynb) | flatten the curve down to hospital capacity; pessimism against a greedy controller |
| 5 | [The benchmark scoreboard](https://github.com/causaldyn/causal-hybrid-control/blob/main/notebooks/05_benchmark_scoreboard.ipynb) | regret against an oracle on pricing, inventory and support-shift tasks |
| 5 | [Robust control under hidden confounding](https://github.com/causaldyn/causal-hybrid-control/blob/main/notebooks/05_confounding_robust_control.ipynb) | when no adjustment set exists: a sensitivity level `Γ`, a pessimism radius, a minimax action, and when it pays |
| 6 | [Cruise control from fleet logs](https://github.com/causaldyn/causal-hybrid-control/blob/main/notebooks/06_cruise_control_confounded.ipynb) | Simpson's paradox in fleet logs; the effect by adjustment or IV; control with the adjusted effect |
| 7 | [Real data: LaLonde](https://github.com/causaldyn/causal-hybrid-control/blob/main/notebooks/07_real_data_lalonde.ipynb) | the naive estimate flips sign; adjustment and double ML land inside the randomised experiment's 95% interval |
| 8 | [Splitting an advertising budget](https://github.com/causaldyn/causal-hybrid-control/blob/main/notebooks/08_splitting_a_budget.ipynb) | three channels, one budget: why the best split equalises the next euro's return, and when TV switches on |
| 9 | [Heating a room on a winter night](https://github.com/causaldyn/causal-hybrid-control/blob/main/notebooks/09_heating_a_room.ipynb) | a thermostat's logs, the weather that confounds them, and a night plan that keeps the room warm |

**Case studies**, with what each printed, are on the
[documentation site](https://causaldyn.github.io/causal-hybrid-control/case-studies/): pricing under
confounding, an epidemic, a pendulum, media budgets, marketplace dispatch, driver supply end to end
and confounded incentives in closed loop, one script each here, and a heat pump on a live building
emulator, run in `causaldyn-bench`.

## What's inside

| area | modules | in one line |
|---|---|---|
| hybrid dynamics | `dynamics` `residual` `integrate` `symbolic` | known physics plus a learned residual (MLP, RBF-KAN, graph, control-affine, port-Hamiltonian, Lipschitz-certified, spectral), rolled out with RK4 |
| identification | `dynamics_id` `causal` `estimators` `gmethods` `train` `frames` | the action's causal effect from confounded logs: adjustment, IV/2SLS, double ML, the g-formula, sensitivity, refutation |
| planning and control | `plan` `control` `mpc` `cost` `lqr` `splitting` `adjoint` | constrained optimal control and MPC on the fitted plant, with an error tube and a safety audit |
| decisions from a panel | `decision` `panel` `graph` | `prescribe`: adjustment set, effect, plan and certificate in one call |
| offline safety | `support` `uncertainty` `evaluation` `offpolicy` | pessimism, calibrated uncertainty, error tubes, a plan's value from logs |
| hidden confounding | `sensitivity` `barrier` `reachability` `regret` | what unmeasured confounding can do to a plan's cost and its safety, and control that hedges it |
| experiments and deployment | `experiment` `switchback` `gate` `misspecification` | which experiment to run first, how to read a switchback, an anytime-valid deployment gate |
| causal frontier | `did` `scm` `irf` `toeplitz` `delay` `discovery` `independence` `network_causal` `pathway` | staggered DiD, synthetic control, dynamic effects, delays, lagged structure discovery |
| media mix | `response` `allocation` `dlm` `lift` `geo_world` `mmm` | response curves, a budget's split, a dynamic linear model, lift tests, geo worlds |
| markets and games | `marketplace` `zones` `matching` `transport` `meanfield` `games` `koopman` `mintime` | control under interference, optimal transport, mean-field control, Stackelberg games |
| end to end and evaluation | `spine` `benchmark` `causal_bench` `flagship` `lalonde` `metrics` `surrogate` | one decision through every layer, oracle-regret leaderboards, real-data LaLonde |
| scientific computing | `epidemic` `galerkin` `deep_galerkin` | epidemic control, Galerkin FEM, neural PDE solvers |

Every module, with what it does and what was measured:
[what's inside](https://causaldyn.github.io/causal-hybrid-control/modules/).

## How it is checked

Symbolic results are derived in Maxima and cross-checked in PARI/GP and Octave
([`validation/`](https://github.com/causaldyn/causal-hybrid-control/tree/main/validation)). The
algebraic cores of the control and guarantee invariants are proved in Rocq
([`proofs/`](https://github.com/causaldyn/causal-hybrid-control/tree/main/proofs)), with MathComp
for the matrix statements. A published statistical result a proof relies on enters it as a named,
cited premise, and each file's header says what it proves and what it assumes
([theory](https://causaldyn.github.io/causal-hybrid-control/theory/)).

## Status

Early, single-author research code, before 1.0. Both halves have been checked on real targets: the
identification on the LaLonde data against its randomised experiment (notebook 7), and the control
loop on a live BOPTEST building emulator, where it beat the tuned baseline controller on four KPIs
of five and drew a 2.0 % higher peak demand
([case study](https://causaldyn.github.io/causal-hybrid-control/case-studies/boptest/)).

What each module promises before 1.0 (stable, evolving or experimental) and the roadmap are in the
[API reference](https://causaldyn.github.io/causal-hybrid-control/api/). The
[changelog](https://github.com/causaldyn/causal-hybrid-control/blob/main/CHANGELOG.md) is written to
be read before upgrading, scope corrections and retractions included. Every release artifact carries
a PEP 740 attestation and SLSA build provenance;
[SECURITY.md](https://github.com/causaldyn/causal-hybrid-control/blob/main/SECURITY.md) says how to
verify one.

## How it compares

DoWhy and EconML estimate effects; Meridian, PyMC-Marketing and Robyn model media; do-mpc and d3rlpy
plan or learn a policy on a model you trust. `chc` works at the seam between them: from a confounded
log to a schedule and its certificate, in one place. It composes ideas that exist — hybrid
dynamics, pessimistic offline control, sequential causal identification, differentiable control —
and its contribution is the integration behind one API, with a benchmark whose interventional
effects are known. The comparison row by row, and when not to use `chc`:
[why, and when not to](https://causaldyn.github.io/causal-hybrid-control/why/).

## Contributing and citation

[CONTRIBUTING.md](https://github.com/causaldyn/causal-hybrid-control/blob/main/CONTRIBUTING.md)
lists the gates a change has to pass: ruff, `ty`, the pytest suite, and `rocq compile` over
`proofs/*.v`. Machine-readable citation metadata is in
[CITATION.cff](https://github.com/causaldyn/causal-hybrid-control/blob/main/CITATION.cff); every
release is archived on Zenodo.

<!-- --8<-- [start:bibtex] -->
```bibtex
@software{gradina_causal_hybrid_control,
  author  = {Gradina, Ilia},
  title   = {causal-hybrid-control: physics-structured dynamics with a learned causal residual},
  year    = {2026},
  version = {0.14.2},
  doi     = {10.5281/zenodo.21737789},
  license = {Apache-2.0},
  url     = {https://github.com/causaldyn/causal-hybrid-control}
}
```

The `doi` is the *concept* DOI: it resolves to the newest release rather than freezing at the
`version` above, so a reader following the citation lands on current code.

<!-- --8<-- [end:bibtex] -->

## License

Apache-2.0 © Ilia Gradina, with a
[NOTICE](https://github.com/causaldyn/causal-hybrid-control/blob/main/NOTICE). Releases up to
0.10.0 stay under MIT.
