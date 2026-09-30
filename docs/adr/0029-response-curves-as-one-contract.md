# ADR 0029 — Response curves as one contract

**Status:** proposed, 2026-09-30.

## Context

A media-mix plan reads a channel's response curve at the spend it proposes, usually beyond the
spend the log covers. CHC had one curve, the Hill inside `chc.mmm`'s plant, written inline. The
packages a user ports a model from ship a dozen or more families (Robyn, Meridian, PyMC-Marketing),
each with its own parameterisation, and a family that fits the logged range as well as another can
part from it where the plan goes.

An S-shaped curve also traps a planner. A Hill with slope above 1 has `h'(0) = 0`, so zero spend on
a channel is a stationary point, and a projected-gradient planner started at zero can stop there
and report convergence. Below the inflection `h'' > 0`, so the first-order conditions certify
nothing there either (Mangasarian's sufficiency fails; Weber, Prop. 3.7). The relaxed problem is
the curve's concave envelope: the tangent from the origin up to the spend `A` where it touches the
curve, `h(A) = A h'(A)`, and the curve beyond (Weber, Prop. 3.8). Its optimum bounds the true one.

## Decision

- **A module of its own, `chc.response`** (*experimental*), so that the plant, curves taken as
  known from another package, and the planner read the same objects. `chc.mmm` stays the plant and
  the benchmark.
- **One contract.** Each family is a standard shape of spend in units of its scale,
  `h(spend) = g(spend / K)`: `K` in currency, every other parameter dimensionless. A change of
  currency moves `K` alone, and every value, slope, inflection and tangency moves with it.
- **Seventeen bounded families** (`Saturation`): zero at zero spend, increasing, ceiling 1; the
  channel's coefficient carries the size.
  - Concave from zero: `MichaelisMenten`, `Exponential`, `Tanh`, `Arctan`, `Algebraic`,
    `HalfNormal`.
  - S-shaped over part of their range: `Hill`, `Weibull`, `Logistic` and `Gompertz` (normalised to
    0 at 0), `Richards`, `ChapmanRichards`, `GammaCDF`, `LogNormalCDF`, `BurrXII`, `BetaCDF`,
    `Kumaraswamy`.
- **Two unbounded baselines**, `Logarithmic` and `Power` (constant elasticity), **and `Ricker`**,
  the inverted U of ad fatigue, apart: not monotone, so no envelope.
- **Each bounded family states where it bends.** `inflection()` and `tangency()` are in spend, 0
  where the curve is concave from zero. Both are closed forms where one exists: the inflection for
  every family, the tangency for `Hill`, `(n - 1)^{1/n}`, and for `Weibull` and `ChapmanRichards`,
  through Lambert W's -1 branch. Elsewhere the tangency is the one root past the inflection of
  `g - z g'`, since `(g - z g')' = -z g''`. `validation/response_curves.mac` derives all of it and
  gives the roots to double precision as the tests' anchors.
- **`Envelope(curve)`** is itself a `Saturation`: the chord below the tangency, the curve beyond.
  The tangency is held fixed when it is differentiated, and that is exact, since at the tangency
  `g = z g'` and its own derivative drops out.
- **A floor is not a shape.** At zero spend a floor is the plant's base. So Janoschek is `Weibull`
  with a floor, and ADBUDG (Little 1970) and Morgan–Mercer–Flodin are `Hill` with one: names in
  the docs, not classes.
- **PyMC-Marketing's `LogisticSaturation` is `tanh(lam x / 2)`**, concave from zero, so `Tanh` with
  `K = 2 / lam`; the docs say so where a user porting a model looks for it by name.
- **Every family's slope at zero spend is its own.** JAX's gamma and beta CDFs return `nan` there at
  shape 1, where the curve is the exponential or `1 - (1 - z)^b`. Their slope in spend is a custom
  JVP on the density, written with `xlogy`. A flat zero in its place would have been as wrong, and
  a planner starts there.
- **`BetaCDF` has no slope in its shapes.** JAX's `betainc` has none, and a zero would be a wrong
  one, so asking raises and names `Kumaraswamy`, its closed-form counterpart.
- **`Tanh` is `(1 - e^{-2z}) / (1 + e^{-2z})`.** XLA's `tanh` falls by an ulp here and there on its
  way to 1, and the test that every curve rises caught it; this form rises.

## Consequences

- `tests/test_response.py` holds each family to the contract: zero at zero, increasing, under 1
  and tending to it; the curvature changing sign at the stated inflection; `h(A) = A h'(A)` at the
  tangency, and the tangencies against Maxima's roots to `1e-13`; the closed forms against the
  root they replace; the envelope concave and never below its curve; the envelope's slope in a
  curve parameter against a finite difference that moves the tangency; the nestings (Hill at slope
  1 is Michaelis–Menten, Burr XII at tail 1 is Hill, Weibull, gamma and Chapman–Richards at shape 1
  are the exponential, Richards at `nu = 1` is the logistic and tends to Gompertz as `nu -> 0`,
  the gap proportional to `nu`); a change of currency, as a property test; and a curve built inside
  a trace fitting by gradient.
- The module has no consumer yet. It is experimental until MM3's next steps use it.

## Not built

- **The planner's warm start and certificate.** `causal_plan` on every non-concave response starts
  from the envelope's plan and reports the relaxed value's gap. MM3's next step, with the S-curve
  counterexample from zero spend on every S-shaped family as its gate.
- **The adstock kernels** (geometric, delayed peak, Weibull), the ROI set (R21), and the plant's
  Hill replaced by a `Saturation`.
- **Fitting every family and planning against those that fit alike**, minimax regret across them
  (PL4), scored on Track M v2's worlds whose true curve is each family in turn.
- **The context-dependent Hill** (arXiv:2406.16728), after a check of the paper.
- **The mapping from PyMC-Marketing's and Meridian's forms**, checked against them in a throwaway
  environment (R17).

## Alternatives considered

- **One class with a `kind` field.** Rejected: every family has its own parameters, ranges,
  inflection and tangency, so a `kind` would switch on all of them, and a missing case would be a
  runtime error rather than an abstract method a new family cannot omit.
- **The tangency as a traced, differentiable root.** Not built. Its consumers, a warm start and a
  certificate, read it once from a fitted curve, and a bracketed root on concrete parameters is
  simpler and exact to rounding. A curve inside a trace still traces; only the tangency does not.
- **Clipping spend at zero.** Rejected: a clip is flat below zero, so a planner's gradient there is
  zero, and a planner that steps below zero is the caller's error to see, not the curve's to hide.
- **The curves in `chc.mmm`.** Rejected: the plant, the curves known from other packages, PL4 and
  MM6 all read them, and `chc.mmm` is one consumer.
- **PyMC-Marketing or Meridian as a dependency.** Rejected: each would pull a probabilistic
  programming stack for a dozen closed forms, and neither has the envelope.
