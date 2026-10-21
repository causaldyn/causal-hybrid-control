# ADR 0124 — A Koopman lift is centred, and an experiment is judged against acting well now

**Status:** accepted, 2026-10-21. Amends ADR 0011 (the experiment's actions and its allocation)
and ADR 0093 (the Koopman dictionary and the controller).

## Context

Two research-periphery tools gave a verdict that did not hold.

- **The Koopman closure verdict depended on where the state was logged.** The polynomial lift had
  no constant and lifted the raw state. The monomials of `x + c` span a different space for each
  level `c`, and none holds the constant that an affine map of the state needs. At degree 3 the
  plant `x' = 2x - x^3 + u/10` logged at levels 0, 1, 5 and 273.15 was refused, accepted,
  refused and refused. The tests' oscillator was accepted at every level, and its controller,
  regulating to the level's origin, left the plant 0.62 from the target after 200 steps at
  level 1 and overflowed at level 5. Nothing checked the loop the controller closes: it lifts
  the measured state afresh at every step, which is not the lifted model whose loop the LQR
  makes stable.
- **`design_experiment` judged an experiment against acting on the mean, and allocated on a model
  it had flagged.** Both arms acted on the certainty-equivalent action, which is not the best
  action under the prior. On one zone, `R = 0.1`, `b ~ N(0.5, 1)`, acting on the mean lost 1.01
  and the Bayes action 0.27, so an experiment worth 0.63 against the first lost 0.11 against
  the second, and the verdict recommended it. The allocation ran on the local regret model,
  and the verdict read Monte Carlo at that one allocation only. At a unit cost of 0.07 the model
  chose no units, the verdict was to abstain because "no experiment the budget buys both
  identifies it and pays", and one unit was worth 0.59 by Monte Carlo (0.033 against the Bayes
  action), while the same call warned that the model missed Monte Carlo by 75%.

## Decision

- **The Koopman lift is of the state less its centre, and it ends in a constant.** The centre is
  the midpoint of the logged states, not their mean, so a state the log holds at one value
  centres to exactly zero. `predict` adds the centre back. With the constant, a degree-`d` lift
  spans every polynomial of degree `d`, which no shift leaves; centred, every feature, residual
  and gain reads the same at every level. The bilinear products leave the constant out: its
  product is the action itself, already in `B`.
- **The LQR is solved on every feature but the constant**, which moves with no action and costs
  nothing; the constant's column of the gain is zero, and the controller's difference
  `phi(x) - phi(x_target)` cancels it.
- **`koopman_controller` refuses a loop that leaves the target.** Near the target the controller's
  loop moves a deviation by `J = [(A - B_t G) dphi/dx]`, the state's rows, `B_t` the model's
  channel there. Where `J`'s spectral radius is 1 or more the law is refused: then the loop
  leaves the target even in the model. On a linear plant with the degree-1 lift `J` is
  `A - B G` exactly.
- **Both arms of the experiment act on the Bayes action**: under the prior now, under the
  posterior after. It is the certainty-equivalent action with the action weight raised by
  `(coupling' Q coupling) o S` (`validation/experiment_bayes_action.mac`, step 1). The posterior's
  covariance does not depend on the reading, so its raise is computed once per allocation.
  `ZoneDecision.bayes_action(prior)` returns it, so a caller acts on the action the numbers are
  about. At a knife edge the Bayes action loses `e^2 q^3 s^4 / (8 r^2)`, two thirds of the
  certainty-equivalent action's `3 e^2 q^3 s^4 / (16 r^2)` (step 2).
- **Monte Carlo chooses the allocation where the model cannot.** Where the model misses Monte
  Carlo by more than 25%, or its own experiment does not pay more than its Monte Carlo error,
  the model only proposes: none, and at every spend from the budget down by halves to one
  unit's, its best experiment that spends it, in whole units. Monte Carlo scores each
  (`uses` times the regret after, plus the spend) on 1000 draws of its own stream,
  `(seed, 1)`, so the allocation chosen is not the one its own draws flatter, and the chosen
  one is reported on the main draws.
- **The reason says what was scored**: the experiments Monte Carlo scored, or the one the model
  chose and how far it was from Monte Carlo, or that the budget buys not one unit.

## Consequences

- The Koopman closure residuals move: the tests' oscillator under a degree-3 lift reads 0.149 at
  its worst feature, against 0.156 before, and still regulates. A gain has one more column than
  before, the constant's, which is zero. `channel` and `residuals` have one more row and column.
- `regret_now`, `regret_after`, `value` and `regret_bound` are the Bayes action's. The five
  lab scenarios keep their verdicts and their units. ADR 0011's bench numbers were measured with
  certainty-equivalent actions; `scripts/bench_experiment.py` now realises the Bayes action, now
  and after, as the design does.
- A design where the model is flagged, or its experiment does not pay, costs about `11 x 1000`
  more solves at a budget a thousand units deep.

## Alternatives considered

- **A constant feature alone.** It makes the span shift-invariant, but each feature `(x + c)^k`
  mixes the lower orders' errors with weights that move with `c`, so the verdict still moved.
- **Centring alone.** A linear plant `x' = A x + B u` is then affine in the centred state, which
  a lift without a constant cannot represent: the bilinear fit of the tests lost its exact
  closure.
- **Optimising the allocation on Monte Carlo in `K` dimensions.** A noisy objective; the search
  keeps the model's shape and lets Monte Carlo set only the scale.
- **Stating the certainty-equivalent premise.** It was stated, and the verdict still recommended
  an experiment that lost against acting well now.
