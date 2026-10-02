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
- **The planner never starts alone at zero on an S-curve.** `relax(model)` swaps every curve in a
  pytree that starts convex for its envelope, and returns `model` itself when there is none. When
  `causal_plan` is given no warm start and `relax(model) is not model`, it first plans the same
  problem on the relaxed model and starts from that plan; a `RecedingHorizon`'s cold start does
  the same. `CausalPlan.relaxed_cost` (*experimental*) is that problem's task cost. Where a larger
  response never costs more and the relaxed problem is convex, as a budget spread over curves of
  spend is, no plan costs less, so `task_cost - relaxed_cost` bounds the plan's distance from the
  best. The accepted steps of both descents are counted, and a descent that took no step from the
  relaxed plan reports the relaxed descent's status: the answer is where that one stopped.
- **A floor is not a shape.** At zero spend a floor is the plant's base. So Janoschek is `Weibull`
  with a floor, and ADBUDG (Little 1970) and Morgan–Mercer–Flodin are `Hill` with one: names in
  the docs, not classes.
- **PyMC-Marketing's `LogisticSaturation` is `tanh(lam x / 2)`**, concave from zero, so `Tanh` with
  `K = 2 / lam`; the docs say so where a user porting a model looks for it by name.
- **Every family's slope at zero spend is its own.** JAX's gamma and beta CDFs return `nan` there at
  shape 1, where the curve is the exponential or `1 - (1 - z)^b`. Their slope in spend is a custom
  JVP on the density, written with `xlogy` away from zero and through the power below at zero. A
  flat zero in its place would have been as wrong, and a planner starts there.
- **Where a slope at zero spend is infinite, it is read off zero.** A curve rising like `z^m` has
  its slopes of order above `m` infinite at zero unless `m` is whole: the slope itself below shape 1
  (`Hill`, `Weibull`, `ChapmanRichards`, `BurrXII`, `Kumaraswamy`, `GammaCDF`, `BetaCDF`, `Power`),
  the curvature below shape 2. The chain rule multiplies an infinite slope into `nan` wherever a
  zero meets it: a period after a plan that a kernel without carryover leaves unreached, and the
  slope of `spend / K` in `K` at zero spend, in any fit of a scale to a series with a dark period.
  JAX's own power is `nan` there at shape 1 too, at second order, `1 * 0 * 0^-1`. So each such power
  is `_power(base, exponent)`, a custom JVP whose slope in the base is the power one lower, itself
  the same function; a power not above 0 at a zero base, which only a slope meets, is read a machine
  epsilon off zero. Every slope at zero spend is then finite, every one finite there is exact, every
  slope in a parameter at zero spend is 0, as the curve is, and below shape 1 the slope at zero is
  steeper than anywhere past `eps K`. A number read through one, such as a decision weight in a
  parameter that turns that spend on, is large and finite.
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
- At zero spend, the same file holds every slope and curvature finite and a zero weight on either
  nothing, for seven families at shapes 0.5, 1, 1.5 and 2.5 and for `Power`; the curvature to
  Maxima's `g''(0)` at shapes 1 and 2 (`validation/response_curves.mac`, STEP 6); every slope in a
  parameter exactly 0, below shape 1 included; below shape 1 the slope steeper than anywhere from
  `1e-15 K` on; and at shape 1 the slope moved by the shape, `(f log z + r) / K` and infinite at
  zero, read at `z = eps`, with Maxima's `f` and `r` (STEP 7). `tests/test_allocation.py` plans a Hill of slope 0.5 behind a kernel without
  carryover against a bounded scalar search, and `tests/test_allocation_decision_weight.py` weighs
  a Hill of slope 1 behind one as Maxima's Michaelis–Menten.
- The planner's gate, the S-curve counterexample posed to `causal_plan` as a one-step plant (a
  budget of 300 over `1000 h(u_1) + 300 u_2 / (100 + u_2)`): from zero spend it reaches the best
  split, found by a grid and refined, to `1e-9` on each of eleven S-shaped families, where a
  descent from zeros stops at the greedy corner, 225, on the Hill with slope 3 and reports
  convergence. Where the best split spends past the curve's tangency the relaxed cost matches the
  plan's to `1e-12` of it; on the gamma CDF with shape 3 the tangency lies past the budget, and the
  gap is positive, a bound and not a certificate of optimality.
- Nothing in the library plans on these curves yet: `chc.mmm`'s plant keeps its own Hill on its
  adstock state until it takes a `Saturation`. It is experimental until the plant and Track M v2
  read it.

## Not built

- **The adstock kernels** (geometric, delayed peak, Weibull) and the return per unit, built by
  ADR 0030; the plant's Hill replaced by a `Saturation`.
- **Fitting every family and planning against those that fit alike**, minimax regret across them
  (PL4), scored on Track M v2's worlds whose true curve is each family in turn.
- **The context-dependent Hill** (arXiv:2406.16728), after a check of the paper.
- **The mapping from Meridian's forms**, checked against them in a throwaway environment;
  PyMC-Marketing's is built by ADR 0030.

## Alternatives considered

- **One class with a `kind` field.** Rejected: every family has its own parameters, ranges,
  inflection and tangency, so a `kind` would switch on all of them, and a missing case would be a
  runtime error rather than an abstract method a new family cannot omit.
- **Multi-start in place of the envelope.** Rejected: a start drawn at random lands in the corner's
  basin with a probability no one knows, and a multi-start's best plan certifies nothing, where the
  relaxed problem's value bounds every plan when it is convex.
- **The tangency as a traced, differentiable root.** Not built. Its consumers, a warm start and a
  certificate, read it once from a fitted curve, and a bracketed root on concrete parameters is
  simpler and exact to rounding. A curve inside a trace still traces; only the tangency does not.
- **Clipping spend at zero.** Rejected: a clip is flat below zero, so a planner's gradient there is
  zero, and a planner that steps below zero is the caller's error to see, not the curve's to hide.
- **A slope of 0 at zero spend where it is infinite, the double `where`.** Rejected: it tells a
  planner that a channel steepest at zero is flat there, and the planner leaves the channel dark.
- **Leaving the periods no spend reaches out of the planner's worth.** Rejected: it mends one
  consumer, and a fit of a scale through a dark period, a marginal return over one, and the
  curvature at shape 1 fail alike.
- **The curves in `chc.mmm`.** Rejected: the plant, the curves known from other packages, PL4 and
  MM6 all read them, and `chc.mmm` is one consumer.
- **PyMC-Marketing or Meridian as a dependency.** Rejected: each would pull a probabilistic
  programming stack for a dozen closed forms, and neither has the envelope.
