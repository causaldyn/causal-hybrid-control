# ADR 0005 — A bound on the steered state

**Status:** accepted, 2026-09-28.

## Context

`prescribe` builds its state vector as the target column followed by one coordinate per
constrained column, and refused a `Constraint` on the target column, since that would have been a
second coordinate for the same column. So the one state a decision is *about* could be steered but
not bounded.

On the BOPTEST heat pump (`docs/case-studies/boptest.md`, run L8.1) the target is the zone
temperature, and the comfort band bounds that same column. With no way to state the band, the
certificate had no barrier and no `gamma*` for comfort, the quantity the benchmark scores and the
naive loop's eight runaway weeks lost. Stating the band does not show that the audit would have
flagged those weeks. The runaway came from a fitted channel that changes sign, and the audit prices
the fitted model, under which more heat cools a zone above the crossing. Whether it would have is a
question for a rerun, not for this decision.

## Decision

- A `Constraint` may name the target column. It bounds the target's own coordinate, position 0,
  and adds no state. `_margins` already maps a constraint to its coordinate by name, so the
  barrier, the audit of every tied margin, `gamma_star` and `hold_constraints` apply unchanged.
- Two constraints on one column are refused, with a message that names the column: one
  `Constraint` carries both bounds. That was the same check before, reported as a clash between
  target and constraint.
- A target value outside its own bound is allowed. It asks for "as close as the bound permits",
  a legitimate request (steer a stock towards a level it may not exceed). Holding the bound is what
  keeps the plan from crossing it on the way; only pricing it reports where the plan would cross.
- The cost does not change. The target keeps its quadratic pull, and every other constrained state
  keeps weight zero, as `_cost` has always done.

## Evidence

The panel of `tests/test_decision.py` (200 units, 12 periods, seed 0; `horizon=15`, `dt=0.1`,
`tolerance=0.5`), steering `supply` to 1 with and without `Constraint("supply", hi=0.5)`, in
float64:

| arm | peak `supply` | first step above 0.5 | barrier steps | trustworthy | task cost | regret bound |
|---|---|---|---|---|---|---|
| no bound | 0.930 | 4 | — | 15 | 1.991 | 4.8e-8 |
| bound, priced | 0.930 | 4 | 0 | 0 | 1.991 | 4.8e-8 |
| bound, held | 0.397 | never | 15 | 15 | 4.654 | 6.48 |

- **Float32** gives the same three rows to three decimals. The one exception is the unheld plans'
  regret bound, `1.3e-5` against float64's `4.8e-8`.
- **`gamma*` is `inf` in both bounded arms.** Some action in the box keeps supply under the bound
  at every `Gamma`, so the priced plan fails its audit because of the action it chose, not because
  safety was out of reach.
- **The fit has one state in all three arms.**
- **A set point inside its band.** Steering `supply` to 0.25 inside `[0, 0.5]`, the band never
  binds: priced and held give the same plan, task cost 0.102, all 15 steps certified. The same
  holds for `[-0.5, 1]`.
- **Mutation check.** Restoring the refusal fails `test_a_bound_on_the_target_is_held_on_its_own_coordinate`
  with the old `DecisionError`.

## Consequences

- A comfort band around a set point is one call, and so is a stock steered towards a cap it may
  not cross.
- "As close as the bound permits" means as close as the barrier condition permits. `ḣ ≥ −α·h`,
  with the class-K gain `α = 1` that `prescribe` does not expose, binds at every step of the held
  plan above: its margin closes by the same factor, 0.903, each step, to 0.103 after 15. The plan
  approaches the bound geometrically and does not reach it in finite time. At the default
  `gamma = 1` the identification radius is zero, so none of that gap is a margin for error.
- A set point at the midpoint of its band is where the band's two margins tie. There the held
  solve reads the first tied margin's gradient, and the audit checks both (`_barrier` and
  `_certify` in `chc.decision`), so a plan leaving through the other edge is reported, not
  certified.
- The regret bound covers the price of holding. It is priced against the box alone, as for any
  held constraint, so the held plan's 4.654 is bounded against the box's 1.991, and it reads 6.48.
- A constraint on the target column, which raised `DecisionError`, now plans. No call that planned
  before changes: without one, the state tuple is what it was. A column constrained twice still
  raises, now naming the column.

## Alternatives

- **`Target(lo=, hi=)`.** Rejected: that would be two ways to state one bound. `Constraint`
  already carries the barrier, the tie handling, `gamma*` and `hold_constraints`, and a second
  spelling would have to duplicate all four or quietly differ from them.
- **Keep refusing, and have the caller copy the column under another name.** Rejected: the fit
  would regress on two identical columns, which identifies only their sum, and the caller would be
  working around the API to state an ordinary requirement.
- **A penalty on leaving the band, in the cost.** Rejected: `_cost` deliberately gives bounded
  states no weight. A penalty is a second objective nobody stated, and the certificate prices
  barriers, not penalties.
- **Refuse a target outside its own bound.** Rejected: "as close as permitted" is a real request,
  and refusing it would make the caller guess the reachable target.

## References

- Ames, A. D., Xu, X., Grizzle, J. W. & Tabuada, P. (2017). Control barrier function based
  quadratic programs for safety critical systems. *IEEE Trans. Automatic Control* 62(8),
  3861–3876.
