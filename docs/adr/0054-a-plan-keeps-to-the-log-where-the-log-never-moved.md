# ADR 0054 — A plan keeps to the log where the log never moved

**Status:** accepted, 2026-10-06. Amended 2026-10-07: the plan's cost and its decision weight read
a ruled lever at its rule's level, as its field does; until 0.14.3 they read its column of the
plan, which nothing else reads. Amended again 2026-10-07: a step holds a ruled lever at its rule's
level at the state it starts from, as the log held it over a period; until 0.14.3 the field read
the rule at every point RK4 reads inside a step. And `Prescription.policy()` sets it from the state
reached (ADR 0070). Amended a third time, 2026-10-07: which levers the log never moved, and the
drift's response to a move no transition tells apart, are read in the units the fit reads
`unmoved` in; until 0.15.0 they were read in raw units, and moved with the units of the state and
of the levers.

## Context

`fit_causal_residual` reads the channel off the Robinson moment, which has data only along the
directions of the channel that the action residuals move. `CausalDynamicsFit.unmoved` names the
rest: the directions along which the log's actions, less what the covariates predict, keep less
than the square root of the working precision of their raw size (ADR 0024). Since 0.13.0
`prescribe` held a lever whose whole channel is unmoved at its mean logged level, and planned every
other lever over its box. A review of 0.13.0 found three ways that fails, and fixing them found a
fourth.

- **A direction no single lever owns.** A log with `u2 = 2 u1` in every row leaves the direction
  `(2, -1)` unmoved on each feature of the channel, and no lever whole. The plan moved the levers
  apart, to `u1 = 2` and `u2 = 0.10`, where the log fixes only `B1 + 2 B2`, and the certificate
  read 3 trustworthy steps. Two channels the log cannot tell apart, `[0.8, 0.1]` and `[0.2, 0.4]`,
  run different paths under that plan.
- **A lever the log set from the state.** With `u1 = -0.3 y` in every row, the log fixes `A - 0.3
  B1`, not `A` and `B1` apart: `A = -0.74 + 0.3 k` and `B1 = k` make the same log for every `k`.
  Held at its mean, 0.0074, `u1` moves the rate by `k (0.0074 + 0.3 y)`, which depends on `k`.
- **A lever the log set from a column outside the state**, `u1 = 0.7 z`. No level reproduces the
  rule, and where `z` moves with the state, its mean does not keep the expected rate either.
- **The fit along the unmoved directions.** The moment's ridge set the channel there by the ratio
  of two roundings, which grows as the square of the actions' units, and, solved beside the moved
  directions, moved those too. A direction whose push on the log the drift's features cannot take
  up, `u1 y` above, which pushes `-0.3 y^2` past an affine drift, is fixed by the log's rates within
  the model class, and the fit did not take it from them. With the levers logged in units 1000 and
  1e6 times their own, the fit's rates missed the log's by 14.7 and 8.7e4, root mean square,
  against the noise's 1.47; the `rk4` fixed point did not converge, and under `euler` 0.13.0's
  plan predicted a path that reached 4e47 in three steps. On a log with `u = -0.3 x0` and an
  affine channel, a slope of 0.2 in `x0` added to the log moved the fit by 2e-11.

## Decision

**The fit keeps to the log where its moment has no data.**

- The moment is solved on the directions it has data on alone: the complement of `unmoved`'s
  span, orthogonal to it once each coefficient is scaled to the raw actions' size, the scale
  `unmoved` is read in, so the split reads the same in any units. The ridge keeps its meaning on
  the raw channel.
- Along a combination of `unmoved`'s directions whose push on the log's raw actions the drift
  regression takes up exactly, by least squares, no rate of the log tells the fits apart, and the
  channel is held at zero, in those scaled units.
- Along the rest, the log's rates rule out all but one value within the class, and the channel
  takes it, by least squares on the rate beside the drift's features. That value is what the log
  did, not an effect. The fit stays linear in its target, so the `rk4` fixed point, `influence`
  and the representer carry it.
- A log that moves every direction fits as before, bit for bit.

**A plan keeps to the log along the directions it never moved.** From the log's actions,
`prescribe` reads three nested spans, each to the precision `unmoved` is read to: the combinations
the log kept at one level, those the state alone predicts, and those the covariates predict.

- **Which levers the log never moved** is read where the fit read `unmoved`'s directions: each
  channel coefficient scaled to its column of the channel's design on the log's raw actions, a
  scale the fit records beside them. There the directions are orthogonal. In raw coefficient units
  the units of the state and of the levers set those columns' sizes apart, the directions lean
  together, and a span read there moves with the units.
- A lever whose whole channel is unmoved:
  - kept at one level, it is held at its mean, clipped to its box, as in 0.13.0;
  - set from the state alone, it follows the log's least-squares rule of the state, the nuisance's
    degree-2 polynomial, clipped to its box, inside the plan's field, and the plan's cost prices it
    at that level. A step sets it at the state the step starts from and holds it over the step, as
    the log held it over a period and as the fit reads a step. The schedule's column carries the
    rule read along the predicted path, and `InterventionSchedule.rules` names the lever;
    `Prescription.policy()` reads the rule in the state reached (ADR 0070);
  - set from a column outside the state, it gives no plan. The certificate reads `not_identified`
    and names the columns the rule reads.
- Among the other levers:
  - a combination kept at one level is held there by an equality row at every step. The level is
    clipped to what the boxes reach, and a level within rounding of zero is zero;
  - a combination set from the state, or from other columns, gives no plan, since no row of the
    plan's actions holds it.
- **The test.** The fits the log cannot tell apart are the fitted one moved along a combination
  the drift takes up, with the drift regression's response, by any amount. The field is linear in
  the parameters. So where such a move leaves the field as it was at every point RK4 reads in a
  step, the step lands where it did whatever the amount, not only to first order: there the test is
  exact. A step that holds a ruled lever is not such a step, and neither are the log's own: inside
  it the state moves and the lever does not, so a move does not cancel there. The test reads such
  a step at the state it starts from and the level it holds there, and a step that passes is one
  of the log's own: from that state, the level the log would have set. The move counts as zero
  where its terms cancel to the square root of the working precision of their size, and a nan
  does not count. A direction whose push on the log cancels to that precision has no drift
  response: the regression would read the rounding as one, which a plan that moves nothing could
  not cancel. The response is least squares in each column's own units (ADR 0057), as the fit's
  split reads it. The first step where a move is not zero is `first_loaded_step`, and the
  trustworthy prefix ends there.
- **The certificate** gains `estimability`: `estimable` where the log moved every direction,
  `held_to_log` where it did not and the plan keeps to it at every step, `not_estimable` for a
  refusal or a loaded step. It gains `identification_rank` and `unmoved_directions`, a state each,
  and `relations`, `rule_levers` and `first_loaded_step`. The report and `to_json` carry them all.
- **What a ruled lever does not take:** a cap on its steps, or a budget that prices it, since the
  rule moves it as the state moves; and `evaluate`, whose open-loop schedule cannot carry a rule.
  Each raises `DecisionError`. `max_levers` refuses a combination kept away from zero, which an
  unselected lever held at zero would leave.

## Consequences

- On the review's logs, one state and two levers over 100 units of 15 periods:
  - `u2 = 2 u1`: the plan keeps `u2 - 2 u1` at zero to 1e-9, and the two channels run one path, to
    1e-12;
  - `u2 = 2 u1 + 0.4`: the plan keeps 0.4; where the boxes reach no higher than -1, it holds -1,
    and reads `not_estimable` from its first step;
  - `u1 = -0.3 y` and `u1 = 0.1 y^2`: the schedule's `u1` is the rule along the predicted path, to
    1e-9, and the worlds `k = 0`, 0.8 and 2 run one path, to 1e-12. With `u1`'s box ending at
    -0.27, the rule leaves the box inside the second step, the third starts outside it, and
    `first_loaded_step` is 2 (1 until 0.14.3, which read the rule inside the step);
  - `u1 = 0.7 z` and `u2 - 2 u1 = -0.3 y` give no plan, and the reason names `z` and the
    combination;
  - a log that moves every direction gives 0.13.0's plan, bit for bit.
- The fit along unmoved directions:
  - logged in units 1 to 1e6 times their own, the channel times the units agrees to 2e-8, the fit's
    rates miss the log's by its noise in each, and `rk4` converges in each;
  - a slope added to the log is read back exactly;
  - on a log whose second action was always twice the first, a plan free to move the two apart
    lost 0.0014 against the truth, where it lost 0.55;
  - on a log whose policy the covariates determine whole, `u = 0.9 z - 0.3 x0`, the channel reads
    2.47 where the truth is 0.8, the confounder's push read as the action's; the ridge read 0.0004.
    Neither is an effect, and `unmoved` names every direction; only the first reproduces the log,
    which is what a plan that keeps to the log needs.
- Every fit with an unmoved direction moves, and every plan made on one. So does what
  `misspecification_cost` compares along those directions: ADR 0024's "the ridge sets it in both"
  becomes "both hold it at zero or read it off the log's rates".
- **In any units of the state.** With the state logged in 1e-15 to 1e15 of its units, and the
  target, the start, the tube's tolerance and the levers' prices with it, every log above reads as
  in the state's own units: the levers unmoved and ruled, the relations and the estimability
  exactly, the schedules to 6.7e-16. Until 0.15.0, read in raw units, at 1e-12 and at 1e12
  `u1 = -0.3 y`'s channel lay 2.5e-5 and 6.3e-6 off the unmoved span, past the square root of the
  precision: the log read as having set `1 u1 - 1.3e-18 u2` from the state, and gave no plan. At
  1e-9 `u1 = 0.7 z` was refused for such a combination, not for `u1`. With the levers read right,
  the drift's response read on the raw drift design lost the state's columns from 1e-13, and the
  plan read `not_estimable` from its first step.
- **In other units of a lever.** With `u1` logged in 1e-12 to 1e9 of its units, its box and its
  price with it, `u2 = 2 u1` and `u1 = -0.3 y` read as in its own units: the levers unmoved and
  ruled and the estimability exactly, the schedules to 3.2e-10 of `u1`'s own units. Until 0.15.0,
  at 1e9 the direction `u2 = 2 u1` never moved lay 2e-9 of the way along `u1`, under the
  precision: `u2` read as never moved and was held at its mean, and the schedule moved by 0.69. At
  1e-9 `u1` read so.
- **What this does not do.** The moment's ridge is still absolute in the actions' units, and
  shrinks a channel the log did move where the actions are small: logged at a millionth of their
  units, the review's moved lever read 5.1e-5 where it reads 0.0977. Fixing that moves every fit,
  so it is a change of its own. The spans of what the log kept, from the levers' own logged
  values, are still orthonormalised in the levers' raw units. With `u1` logged at 1e12 of its
  units, QR there reads `u2 = 2 u1`'s weight on `u1` as -1.99996e-12 where -2e-12, and its level
  as 4.4e-7 where 0: the plan holds that row, and reads `not_estimable` from its first step. And
  `u1 = -0.3 y` is refused there as set from outside the state. At 1e9 QR reads the weight of
  `u2 = 2 u1 + 0.4` on `u1` to 2.8e-8 of itself, and that plan moves by 5.5e-9 of `u1`'s units.

## Alternatives

- **The levers the log never moved read off `unmoved` as it is, in raw coefficient units**, as
  until 0.15.0. Rejected: no basis of a span reads well in both units, and the precision the fit
  reads the span to is the scaled units'. A direction that lies 2e-9 of the way along a lever in
  raw units lies under the square root of the precision there, whatever the basis.
- **The scale recomputed from the log by each reader**, as `reach` reads it (ADR 0069). Rejected:
  the fit computes it beside the directions, and carried with them it is one scale for every
  reader, one that holds the fit alone among them.
- **Refuse every plan along an unmoved direction**, the review's remedy for all but a fixed
  relation. Rejected for levers set from the state: the log does determine the plan's path there,
  as long as the plan keeps the rule.
- **The test along every unmoved direction, not only those the drift takes up.** Rejected: `u1 y`'s
  push, `-0.3 y^2`, is ruled out by the log within the class, though the moment has no data on it,
  and the test would refuse the right plan in the held case at an affine channel.
- **The rule read at every point RK4 reads**, as until 0.14.3. Rejected: the log set the lever
  once a period and held it, and the fit reads a step so. Read inside the step, the plan's path was
  not the path of the schedule it reports, each level held over its step: on the review's log
  `u1 = 0.1 y^2` they parted by 1.0e-3 in three steps, and the plan reported a task cost 1.5%
  above its schedule's. The test was exact on that field, which is not the one the log ran.
- **A rule of a column outside the state held at its mean**, with the column named and
  `held_to_log` scoped to it, the review's other answer. Rejected: the mean keeps the expected rate
  only where the column is independent of the state, which nothing here can check.
- **The ridge's value along the unmoved directions**, as before. Rejected: it moves with the
  actions' units, and it leaves the fit off the log along the directions the log fixes.
- **The covariates partialled out of the log's reading too**, which would keep the confounder out
  of it. Rejected: the plan follows the log's rule without the confounder, so the path it must
  predict is the log's given the state, as the drift's is.
