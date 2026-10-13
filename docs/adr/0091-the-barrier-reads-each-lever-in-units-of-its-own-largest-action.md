# ADR 0091 — The barrier reads each lever in units of its own largest action

**Status:** accepted, 2026-10-13. Amends the barrier's audit (ADR 0002, ADR 0085) and the rate
error of `prescribe`'s tube (ADR 0087). Amended by ADR 0096, 2026-10-13: the tube's rate error
adds the drift's own error, which part 2 left out. Amended in 0.19.0: under `prescribe`
`gamma*`'s best action is sought in the levers' box itself, not in the ellipsoid of part 1, which
holds actions an asymmetric box forbids (the last section).

## Context

ADR 0087 made the tube read the same in any units of each lever, and named three things it left
alone.

1. **The barrier's audit read the levers' units as one.** `certify_safety` prices a plan against an
   identification radius on the channel, `||B_hat - B||_op <= Delta`, with
   `Delta = (Γ-1)/(Γ+1) · cvar_gap · ||B_hat(x_0)||_2` (ADR 0085). The guaranteed barrier derivative
   at an action `u` is `a + <w, u> - Delta ||grad h|| ||u||`, and `Γ*` asks what the best action of
   the ball `||u|| <= u_max` can absorb. `prescribe` passed one `u_max`, the largest action of any
   lever's box. The ball, the operator norm and `||u||` each add the squares of numbers in the
   units of different levers. On the cross-module tests' panel (one state, two levers in
   `[-2, 2]`, three steps), bounded to `[0.2, 1.1]` at `Γ = 1.5`, the `free` and `together` logs
   cleared three steps of three. With `u1` logged in units 1e3 times its own, its box and price with
   it, or `u2` in thousandths, they cleared none, and the trusted steps fell from two to none. On
   the `state` log `Γ*` read 118 where it read 6.17. Held in the solve (`hold_constraints`), the
   `state` log's plan cleared no step of three in those units, where it clears all three.
2. **The tube priced each lever's error from the lever's zero.** `reach_j = max(|lo_j|, |hi_j|)`.
   An offset of a lever's origin, the log and the box shifted with it, moved the tube: on the
   `free` log, `u1` offset by 100 ended the tube at 0.713 where it ended at 0.538, and certified
   one step where it certified two.
3. **The states mix in the tube.** The tube is one norm over the states and its rate error a root
   mean over them, so states in units unlike each other mix in it, as the levers did.

## Decision

### 1. The audit reads the action set whose semi-axes are each lever's largest action

`certify_safety`'s `u_max` takes one number per lever, each lever's largest action `r_j` in its
own units. With `U = max_j r_j` and `D = diag(r / U)`, the audit reads:

- the action set `||D^+ u|| <= U`, the ellipsoid whose semi-axes are the `r_j`;
- the radius on `(B_hat - B) D`, sized by `||B_hat(x_0) D||_2`;
- a planned action's norm as `||D^+ u||`;
- a step's ceiling with the channel `D w` and the authority `U`.

`BarrierConstraint` takes the same `u_max`, so the held solve holds the condition the audit
prices, and `causal_plan`'s audit after the solve reads it. `prescribe` passes its box's `r_j`
(a lever it holds at its logged level is held there, and its `r_j` is that level's size). The
held solve now audits at the whole box's numbers, which the constraint carries, so `prescribe`'s
second audit of a held plan, which existed to replace the candidate box's `u_max`, is gone.

- **Why it reads no lever's units.** With lever `j` in units `c_j` times its own, `r_j` and `u_j`
  move as `c_j` and the channel's column `j` as `1 / c_j`. So `B_hat D` and `D w` move by `1 / U`'s
  factor alone, `D^+ u` by `U`'s, and the authority is `U`: the audit reads none of them, as it
  reads no common scaling of the levers (ADR 0085). `validation/barrier_lever_units.mac` checks the
  spread, the channel along the barrier and the size, and that the ball's spread moves with one
  lever.
- **One lever, or equal largest actions, keep the bits.** `D = D^+ = 1` exactly, since `x / x` is 1
  in binary floating point, and the audit is the ball of radius `U`. The quickstart, notebook 09,
  the BOPTEST and pendulum case studies, and a plan whose levers share one box read as before, to
  the bit.
- **The ellipsoid lies inside the box where the box is symmetric.** Its best action,
  `u_j = r_j^2 w_j / sqrt(sum_k (r_k w_k)^2)`, keeps each `|u_j| <= r_j`
  (`validation/barrier_lever_units.mac` (5)). The ball of radius `U` did not: in the tests'
  two-force oscillator, boxes of 5 and 1, its best action pushed the second force to 2.2. An
  asymmetric box is another matter: `|u_j| <= r_j` holds `-r_j` where the box is `[0, r_j]`, and
  the ellipsoid counted that action (amended below).
- **Edges.** A lever whose number is 0 has no action in the set. The audit refuses a plan that moves
  it, and the held solve a box that lets it move. A free side, `inf`, leaves no units to read its
  lever by: the audit then reads the levers' own units with `U = inf`, as one number `inf` did.

### 2. The tube prices each lever's error from the log's mean action

`reach_j` is now the largest distance lever `j`'s box allows from the log's mean action `ū_j`,
`max(|lo_j - ū_j|, |hi_j - ū_j|)`. The rest of ADR 0087's rate error is unchanged.

- **Why the log's mean action.** The fit takes the channel from the residualised actions,
  `u - E[u | covariates]`, and then regresses `y - B_hat(x) u` on the drift's features, a constant
  among them. The drift therefore absorbs `B_hat` times the log's action as those features predict
  it, and the field's error at an action is
  `(f_hat - f)(x) + (B_hat - B)(x) u = e_drift(x) + (B_hat - B)(x) (u - m(x))`, where `m(x)` is that
  prediction and `e_drift` is the drift's own error at it. The channel's error multiplies the
  action's distance from `m(x)`, not from zero. An origin shift of a lever is a relabelling: the
  fit on the shifted log is the same field, read at the shifted action, and its error at each
  physical action is the same.
- **The evidence.** On 120 logs of the cross-module tests' plant, its `free` policy, with `u1`
  logged about 3, the spread of the fitted rate's error over `u1` was least at `u1 = 3.19`; with
  `u1` logged about 0, at 0.60. Each is within the estimate's own scatter of the log's mean, and
  their difference, 2.59, of the offset, 3. Under a fit whose channel error multiplied the action
  itself, both would sit at 0.
- **Why the mean, and not `m(x)` along the plan or the plan's own action.** The tube carries one
  rate error for every step, and `channel_error` is already a root mean over the log's states, so
  its centre is the mean of `m(x)` over the same states: the log's mean action. `m(x)` at a
  planned state is an extrapolation of the logging policy and a different number at each step. The
  plan's own action is where the error is read, not where it vanishes.
- **A lever held at its mean logged level** is held at `ū_j` itself, read by the same expression,
  so it adds no error. Its column is what the drift absorbed; ADR 0087 priced it at `|ū_j|`.

### 3. The states are left mixed, and the docstring says so

A per-state tube needs the fit's error per state and lever, the field's rate in normalised
states, and a vector certificate; a per-state scale needs the same rate and errors read in those
scales. Neither is additive in the fit. The planner reads the problem the same in any units of
the state (ADR 0059), so a caller who logs each state in units of the error accepted on it, the
target's level, the bounds and the prices moved with it, gets the per-state scale with
`tolerance = 1`. `prescribe`'s `tolerance` says so. A test pins the reading: a common scaling of
the states moves the tube by its factor, and one state's scaling alone moves it otherwise.

## Consequences

- **The barrier reads the same in any units of each lever.** On the panel above, with `u1` or
  `u2` in units 1e3 or 1e-3 times its own, or the two apart, the barrier's certified steps and
  trusted steps are the same on all three logs, `gamma*` to 2e-14 of itself, bounded to
  `[0.2, 1.1]` or `[0.8, 1.1]`. Held, the `state` log's plan clears three steps of three in every
  units, at `gamma*` 6.509 to 3e-10, where the ball's held plan cleared none in two of them and
  read 6.36 to 130.7.
- **The held rounds step each lever in units of its largest action** where the levers' largest
  actions differ, `v_j = u_j / r_j`, through the descent's fixed metric with the box and the rows
  carried into it. Stepped in the caller's units, the tests' two-force oscillator held no step of
  20 with its forces in units `(1e3, 1)` or `(1, 1e-3)`. Now every units' rounds converge, hold
  all 20 steps and land on one cost to 4e-6. Where the largest actions are equal the rounds keep
  the caller's units, to the bit; the plan still moves with a scaling the levers share, as
  ADR 0085 records.
- **Where the levers' boxes differ in one unit, `gamma*` moves,** since the set now lies inside the
  box. On the `free` log with boxes 0.5 and 2, bounded to `[0.8, 1.1]`, it reads 2.30 where it read
  10.88: the ball's best action put `u1` at 1.98, nearly four times its box. With boxes 2 and 0.5
  it reads 10.79 where it read 10.88; on the `together` log 1.61 and 3.71 where both read 4.23. The
  certified steps do not move there. Held, the `state` log, whose `u1` is held at 0.028 beside
  `u2`'s box of 2, reads `gamma*` 6.509 where it read 6.359.
- **One number, one lever, or equal boxes: the same bits.** The quickstart and notebook 09, one
  lever each, keep their certificates to the bit under this part; so do the BOPTEST and pendulum
  case studies.
- **The tube reads the same wherever a lever's zero is.** On the panel above, `u1`'s zero moved by
  3 or 100, or `u2`'s by -50, the log and the box with it, leaves the `free` and `state` logs' tubes
  the same to 1.4e-5 of themselves, the fit's own rounding at an offset of 100, and their certified
  horizons. The `together` log's moves by up to 0.3 %: its levers moved together, and the fit
  splits the channel's error between them along the combination the log kept, a split that reads
  the levers' origin; the tube's centre does not.
- **Where a log's mean action is not zero, the tube moves.** On the panel above, whose logs act
  about zero, the tubes' ends move by 5e-5 of themselves at most (`free` 0.538331 to 0.538358);
  the certified horizons do not. The quickstart's ends at 2.37210 where it ended at 2.37191.
  Notebook 09's heater, logged about 0.97 kW in a box `[0, 3]`, is priced at 2.03 where it was
  priced at 3, and its tube ends at 0.641 where it ended at 0.784, its eight certified hours
  unchanged. The notebook's stored outputs and its prose quote 0.78, and need a rerun.
- **Coverage on the BOPTEST case's logs** (`scripts/bench_boptest_tube.py cover`, 100 logs a cell,
  200 draws each, the six default cells):

  | cell | tube one step ahead, before | after | plant inside at the first step, before | after | trusted steps, mean before | after | plant's error within tolerance at 95 % |
  |---|---|---|---|---|---|---|---|
  | one, AR(0) | 0.275 | 0.258 | 0.989 ± 0.002 | 0.982 ± 0.003 | 1.74 | 1.94 | 3 steps |
  | one, AR(0.5) | 0.286 | 0.264 | 0.993 ± 0.003 | 0.988 ± 0.004 | 1.13 | 1.56 | 2 |
  | one, AR(0.9) | 0.276 | 0.252 | 0.989 ± 0.009 | 0.983 ± 0.010 | 1.10 | 1.56 | 2 |
  | one, AR(0.99) | 0.212 | 0.196 | 0.988 ± 0.010 | 0.980 ± 0.012 | 1.96 | 2.04 | 2 |
  | slab, AR(0) | 0.248 | 0.232 | 0.972 ± 0.006 | 0.961 ± 0.007 | 2.08 | 2.54 | 3 |
  | slab, AR(0.9) | 0.226 | 0.206 | 0.994 ± 0.004 | 0.989 ± 0.005 | 1.93 | 2.00 | 3 |

  The heat pump's modulation is logged inside its box `[0, 1]`, away from its zero, so its error
  is priced at less than 1, and the noise's radius, most of the tube, does not move. Over the
  steps each log trusts, the plant stays inside the tube on 97.1 % to
  99.0 % of draws, where it did on 97.7 % to 99.5 %. Four logs of 100 in the `one, AR(0.99)` cell
  now trust 3 steps where the pooled 95 % error allows 2; on each, every draw stays inside the
  tube at the third step.
- **Not covered.** The drift's own error at the log's mean action is not in the tube, before or
  after. Centred, the tube prices the channel's part of the field's error alone; priced from zero,
  it overstated that part, which on some logs hid the omission. On that plant above, the fitted
  rate's error at the logged action is 0.043, where the channel's is 0.0075 per unit of `u1`.
  Carrying it is a new term of the tube and needs its own coverage evidence (Alternatives). The
  states still mix in the tube, as part 3 says.
- **The barrier still reads each lever from its zero.** Its spread, `Delta ||grad h|| ||D^+ u||`,
  and its action set are measured from the lever's zero, where the drift's absorbing the channel
  at the log's mean action, as part 2 argues, would measure them from there. On the `free` log
  bounded to `[0.2, 1.1]`, `u1`'s zero moved by 100, its log and box with it, clears no barrier
  step of three where it clears three, `gamma*` reads 1.04 where it reads `inf`, and the trusted
  steps fall from two to none. Centring the audit changes the action set and `Γ*`'s closed form
  (ADR 0002), and is not done here.
- `certify_safety`'s `u_max` and `BarrierConstraint`'s new `u_max` take one number per lever; a
  number, or `None`, reads as before. `_pessimistic_loop` gains a private `metric`. Tests:
  `tests/test_held_barrier.py`, `tests/test_cross_module_invariants.py` and
  `tests/test_decision.py`; `validation/barrier_lever_units.mac`.

## Alternatives

- **For the barrier, the box itself as the action set.** The worst case at an action stays
  `Delta ||grad h|| ||D^+ u||`, but the best action over a box against a Euclidean adversary has no
  closed form, and `Γ*` would need a search over the box's corners.
- **For the barrier, a radius per lever,** `||(B_hat - B)_j|| <= Delta_j`. The adversary's worst
  case becomes `sum_j Delta_j |u_j|`, up to `sqrt(m)` times the ball's, and levers that share units
  and boxes would move.
- **For the barrier, each lever in units of its spread on the log.** It reads no units either, but
  it moves levers that share units and boxes whenever their spreads differ, and it ties the action
  set to the logging policy rather than to what the plan may do.
- **For the tube, the box's half-width about its midpoint.** It reads no origin either, but it is
  not where the error vanishes: a log at one edge of its box would be priced at half the box where
  the far edge is a whole box away.
- **For the tube, carry the drift's own error at the log's action as well.** It is the error the
  centre leaves, and `drift_error` reads it. Adding it is a new term of the tube, which needs its
  own coverage evidence, and is not done here.
- **For the states, an additive `tolerance` vector.** Default-preserving, but the fit keeps no
  per-state lever errors and `_rate` bounds the field's slope in the caller's units; both would
  change for the vector alone, and the caller's rescaling already gives the reading.

## Amendment, 0.19.0: `gamma*` seeks its best action in the box itself

**Context.** Part 1 reads the action set as the ellipsoid `||D^+ u|| <= U` whose semi-axes are
each lever's largest action `r_j = max(|lo_j|, |hi_j|)`. It lies inside a symmetric box, but an
asymmetric one does not hold it: in `[0, 2]` the ellipsoid holds `-2`. On a log whose lever only
pushes the state up, at a start where the drift already crosses the state's upper bound, no action
of `[0, 2]` holds the first step at any level, and `gamma*` read 2.02, as it reads in `[-2, 2]`,
where `-2` is admissible. A lever `prescribe` holds at its logged level `c` was read as
`[-|c|, |c|]`. `gamma*` is the problem's ceiling over *admissible* actions; these were not.

**Decision.** `certify_safety` and `BarrierConstraint` take `box=(lo, hi)`, one pair per lever,
and `prescribe` passes its whole box, its held levers pinned at their levels. Given a box, a step's
threshold is `d* = sup (<w, u> - D) / ||D^+ u||` over the box's nonzero actions, `D` the deficit
the drift leaves and `w` the channel along the barrier, read in the coordinates `v = D^+ u` where
the box is `[U lo / r, U hi / r]` and the channel `D w`; `inf` where the box holds zero and
`D <= 0`, and `nan` where the supremum is negative, which certifies no level. The supremum lies on
the path `clip(s w, lo, hi)`, `s >= 0`, since for each radius a best action of
`<w, u> - d ||u||` is a best one of `<w, u>` within its own norm. On each linear piece of that path
the ratio has one stationary point, `s = (alpha b - beta a) / (beta b - alpha c)`, so the answer is
exact up to rounding (`chc.barrier._box_threshold`, `validation/barrier_box.mac`). `u_max` still
sets each lever's units, so `gamma*` reads the same in any units of each lever, and defaults to
each lever's largest action in the box. Without a box, `certify_safety` reads the ellipsoid as
before.

**Consequences.**

- No forbidden action is counted. In `[0, 2]`, `[0.5, 2]`, or held at 1, the pushing lever's step
  reads `nan` where the ellipsoid read 4.82, 4.82 and 1.91; the `prescribe` log above reads `nan`
  where it read 2.02, held or not.
- One lever in `[-U, U]` is the ball, and reads the closed form to rounding: the quickstart and
  notebook 09 (whose heater's box is `[0, 3]`; it reads 33.49) do not move but in the last digits.
- A symmetric box of several levers holds the ellipsoid and its corners: `gamma*` can only rise. On
  the cross-module tests' `free` log bounded to `[0.8, 1.1]`, levers in `[-2, 2]`, it reads 11.00
  where it read 10.88.
- Where the box does not hold zero, a step the drift alone keeps is no longer `inf`: every
  admissible action is moved by the effect's error, and the step's ceiling is finite.
- The spread is still read from each lever's zero, as part 1 says; only the set changed.
- Tests: `tests/test_held_barrier.py`.

**Alternatives.** An ellipsoid inside the box, centred at its centre: it reads the box's units
too, but it leaves the box's corners out, and its best action against a spread measured from zero
has no closed form either. The box's corners alone: the supremum can lie inside a face, where the
ratio's stationary point is.
