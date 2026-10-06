# Identification

**A low residual MSE is not causal identification.** Fitting `r_θ` by prediction error recovers the
*observational* control response, which is the wrong one whenever the logged action was chosen from
something that also moved the state. No amount of extra training fixes it, because it is not a
fitting problem: a controller that optimises against the observational response moves the state
the wrong way.

## What is identified, and how

[`chc.dynamics_id`](../api/dynamics_id.md) is the module that makes the *plant* causal. It lifts
Robinson partialling-out from a scalar effect to a state-dependent matrix. With nuisances
`g(x,z) = E[y | x,z]` and `m(x,z) = E[u | x,z]`, where `y = (x_next - x)/dt - f_known(t, x, u)` is
the part the residual must explain,

```text
y - g(x,z)  =  B_θ(x) (u - m(x,z))  +  eps,     E[eps | x, z, u - m] = 0
```

so `fit_causal_residual` regresses the **residualised state rate on the residualised action**, with
K-fold cross-fitting of both nuisances. Neyman orthogonality is what buys the guarantee: the channel
error is *second* order in nuisance error. When the confounder is never logged but an instrument
is, it estimates the channel by 2SLS instead — at a real variance premium when the instrument
explains little of the action, so an instrument is a weaker substitute than the word suggests.

Three limits, stated before the code is:

- Only the **channel** `B_θ` is interventional. The drift `a_θ` is fitted on the remainder, so it
  absorbs whatever the omitted confounder contributes in-sample: an *observational-conditional*
  drift. Planning is unbiased in the direction the optimiser moves; the predicted trajectory *level*
  still shifts if the confounder's distribution does.
- The residual must be **control-affine** (`ControlAffineResidual`). A general `r_θ(x, u)` has no
  partialling-out moment and gets no guarantee here.
- With no adjustment set and no instrument nothing in the log identifies the channel. The estimator
  reports `identified=False` rather than a confident wrong answer, and that case belongs to
  [`chc.sensitivity`](../api/sensitivity.md), which prices the radius instead of pretending it away.

## When the channel class misses the truth

Orthogonality protects the channel from the nuisances' error, not from the class. When `B_θ` cannot
represent the true channel, as when a line is fitted to a curved response, the fit is the
projection of the true channel under `s²(x) P(x)`. Here `s²(x) = E[(u - m)² | x]` is the variance
the adjustment set leaves in the action, and `P` is the log's law of states. So the fit is the best
line where the log was.

`weights=` moves the projection to `w s² P`. For a one-shot decision taken at states drawn from
`Q`, the weight `w = κ (dQ/dP) / s²` makes the fit the best line for that decision.

- **The weight is a function of the state alone.** A weight that reads the action breaks the
  orthogonality at first order, so the fit calls it with the states and nothing else.
- **A weight the caller estimates brings its own error at first order.** Nothing orthogonalises
  it.
- **No pointwise weight is right for a dynamic plan.** A dynamic plan reads the channel's slope
  along its path as well as its level.

The design and its numbers are in
[ADR 0013](https://github.com/causaldyn/causal-hybrid-control/blob/main/docs/adr/0013-a-weight-on-the-channel-moment.md).

## The adjustment set comes from a graph

Every effect estimator here can take its adjustment set as a literal tuple of column names. That
tuple is a causal claim typed by hand, and the two ways it goes wrong are silent: adjusting for a
collider *opens* a path that was closed, and adjusting for a mediator removes part of the effect
being estimated. Neither shows up as an error, a warning or a bad fit; both show up as a wrong
number.

[`CausalGraph`](../api/graph.md) takes the assumption where it belongs — in the graph — and derives
the tuple: the canonical set of Perković, Textor, Kalisch and Maathuis (2018), which is valid **iff
any set of observed variables is**. An empty result therefore means "no adjustment needed", and a
`not_identified` status means "no covariate adjustment identifies this effect" — a different
statement, and one worth failing on rather than guessing past. It is a graphical criterion, so it
inherits the graph's assumptions and nothing more: it does not test the DAG, and a missing edge is
still a missing edge.

## In the façade

[`prescribe`](../api/decision.md) makes the assumption a **required** argument: `adjustment=` takes
either a `CausalGraph`, from which the set is *derived* and can come back `not_identified`, or an
explicit sequence of column names, which *asserts* it. There is no default, because the default
would be "adjust for nothing", which is a causal claim this library exists to stop people making by
accident. An unidentified effect produces no schedule at all.

## See it

- [Tutorial 1](../tutorials/01_causal_vs_predictive_control.md) — the naive fit flips the sign, and
  the controller built on it drives the state the wrong way, as far as its actuator allows.
- [Tutorial 3](../tutorials/03_causal_inference_toolkit.md) — adjustment, IV/2SLS, Double ML,
  sensitivity and refutation, side by side.
- [Tutorial 7](../tutorials/07_real_data_lalonde.md) — real data with an experimental ground truth.
- `uv run python scripts/dynamics_id_demo.py` — where a prediction-error residual lands, and what
  identifying its channel is worth, scored on the true plant.
