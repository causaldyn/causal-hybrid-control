# ADR 0087 — The tube prices each lever's error at its own largest action

**Status:** accepted, 2026-10-12. Amends the rate error of `prescribe`'s tube (ADR 0080). Amended by
ADR 0091, 2026-10-13: each lever's error is priced from the log's mean action, not from its zero.

## Context

`prescribe`'s tube carries two errors through the plan's RK4 steps (ADR 0080): the plant's own
noise, and a rate error `eps` of the fitted field. `eps` was `channel_error` times `u_max`.
`channel_error` is the root mean, over the channel's entries and the log's states, of the robust
standard error of `B_hat(x) - B(x)`. `u_max` is the largest action that any lever's box allows.

Both numbers mix the levers' units. Entry `(i, j)` of the channel is in state `i`'s units per unit
of lever `j`, so the root mean adds squares in the units of different levers. `u_max` is the
largest of numbers in the units of different levers. Their product reads the same in any units
that the levers share, and the docstring of `_model_error` restricted the claim to that case. It
does not read the same when one lever alone changes its units.

On the cross-module tests' panel (`tests/test_cross_module_invariants.py`: one state, two levers
in `[-2, 2]`, three steps, a tolerance of 0.5), the `free` log's tube ended at 0.538 and certified
two steps. With `u1` in units 1e3 times its own, its box and its price with it, `eps` read 10.36,
where it read 0.0135. The tube ended at 3.89 and certified no step. With `u1` in thousandths the
tube ended at 4.04. With `u1` in thousandths and `u2` in thousands it ended at 3503. The planner
reads the problem in any units (ADR 0059), but its certificate read the units of one lever.

## Decision

- **Each lever's error is read alone.** The fit reads `channel_error` off one covariance of the
  channel, or off three under two-way clustering, and takes the largest read (ADR 0061). It now
  also reads each covariance over each lever's column alone. `e_rj` is the root mean, over the
  entries of lever `j`'s column and the log's states, of the standard error of `B_hat(x) - B(x)`
  under covariance `r`. It is in the states' units per unit of lever `j`. The fit keeps the
  `(r, m)` array as the private field `CausalDynamicsFit._lever_errors`, beside
  `_parameter_size`. `channel_error` is computed as before, to the bit.
- **The tube prices each lever's error at that lever's largest action.** `reach_j` is the largest
  action that lever `j`'s box allows, `max(|lo_j|, |hi_j|)`, in lever `j`'s units, after
  `prescribe` holds each lever the log never moved, or set from the state alone, at its mean
  logged level. `e_rj reach_j` is in the states' units. The rate error is

  ```text
  eps = max_r sqrt( mean_j (e_rj reach_j)^2 ),
  ```

  the root mean, over the channel's entries, of each entry's error times its lever's largest
  action, read off each covariance, the largest read taken.
- **Why it reads no lever's units.** With lever `j` in units `c_j` times its own, its column of
  the channel and that column's standard error move as `1 / c_j`, and its box moves as `c_j`.
  Each product is the same number, so each read is, and so is the largest.
  `validation/tube_lever_units.mac` checks this, and that `channel_error` moves (1). It also
  checks that `eps` is the root mean over the levers of their reads times their actions (2), the
  two identities below (3, 4), the bound `eps <= channel_error * u_max` where the levers share
  units (5), and the factors of two of the alternatives (6).
- **The largest read is taken after each lever is priced.** Under two-way clustering the three
  reads can put the largest error on different levers. On the panel above they do: one read's
  largest error is `u1`'s, another's is `u2`'s. Which read is the largest then depends on the
  boxes. Read off the covariance that `channel_error` takes, it would depend on the units too
  (Alternatives). The tube is one more reader of the covariances, and like every other reader it
  takes the largest of its own reads.
- **One lever keeps its tube to the bit.** Its column is the whole channel, so `e_r1` is read `r`,
  computed by the same function on the same blocks, and the largest is `channel_error`. The root
  mean of one product is that product exactly, since `sqrt(x^2) = |x|` in binary floating point
  without overflow or underflow. Rounding preserves order, so the largest product is
  `channel_error` times the box's largest action, as before. The tests check this in float64 and
  float32, under Euler and RK4, read row by row, by clusters, two ways and across periods.
- **Levers in one unit whose boxes allow the same largest action keep their tube to rounding.**
  `mean_j e_rj^2` is the mean over all entries, so each read is `channel_error`'s read times that
  action.
- **What a lever adds.** A lever whose box allows no action but zero adds no error, but it counts
  in the mean, as an entry of zero error does. A lever that the log never moved, or set from the
  state alone, is held at its mean logged level `a`, so it adds its column's error times `|a|`.
  Before, that column's error entered `channel_error` and was priced at the largest action of any
  lever. A fit with no channel error
  has no lever errors: `eps` is 0 and the tube is not evaluated, as before. A free side gives
  `eps = inf`, and the tube is not evaluated, with the warning, as before.

## Consequences

- **The tube reads the same in any units of each lever.** On the panel above, with either lever
  in units 1e3 or 1e-3 times its own, or the two apart, the tube is the same to 2e-14 of itself
  on all three logs, and so are the certified horizon and the trusted steps. With no bounds on
  the state, the whole certificate on that panel now reads the same in any units of each lever.
- **Where the levers share units, `eps` can only shrink.** It is at most `channel_error * u_max`,
  and less where a lever's box allows less than another's, or where a lever is held at its logged
  level. The certified horizon moves nowhere on the panel, since the noise's radius is most of
  the tube there:

  | log | largest actions `(u1, u2)` | `eps` before | after | tube's end before | after |
  |---|---|---|---|---|---|
  | `free` | `(2, 2)` | 0.01355 | 0.01355 | 0.53833 | 0.53833 |
  | `free` | `(0.5, 2)` | 0.01355 | 0.01059 | 0.53752 | 0.53657 |
  | `free` | `(2, 0.5)` | 0.01355 | 0.01098 | 0.53797 | 0.53714 |
  | `free` | `(0, 2)` | 0.01355 | 0.01036 | 0.53725 | 0.53622 |
  | `together` | `(0.5, 2)` | 0.00592 | 0.00296 | 0.53412 | 0.53315 |
  | `together` | `(2, 0.5)` | 0.00592 | 0.00534 | 0.53416 | 0.53397 |
  | `state`, `u1` held at 0.028 | `(0.028, 2)` | 0.01445 | 0.00890 | 0.55084 | 0.54895 |

- **One lever, the same bits.** The quickstart, notebooks 00 and 09, the BOPTEST and pendulum case
  studies, and `scripts/bench_boptest_tube.py` plan one lever, so their tubes, certified horizons
  and trusted steps do not move. The media-mix plants plan levers in one currency with one box,
  so theirs move by rounding.
- `CausalDynamicsFit` gains the private field `_lever_errors`. No public field changes meaning.
- Tests: `tests/test_cross_module_invariants.py`, `tests/test_decision.py` and
  `tests/test_dynamics_id.py`.
- **Not covered.** `reach_j` is measured from lever `j`'s zero, so the tube reads where each
  lever's zero is: an offset of a lever's units moves it. The drift absorbs the channel times the
  log's mean action, so the error an action induces is the channel's error times the action's
  distance from what the log did, not from zero. Pricing that distance would read no zero, and
  would price a lever held at its logged level at nothing. But it moves the tube of every plan,
  one lever's included, and it needs its own coverage evidence. The states are not covered
  either: `eps` is still a mean over states, and the tube a norm over them, so states in units
  unlike each other mix as before. The barrier's audit, `gamma_star` and
  `barrier_certified_steps`, still reads one lever's units alone (ADR 0085). On the panel above,
  bounded to `[0.2, 1.1]` at `gamma = 1.5`, it certified three steps, and none with `u1` in
  thousands or `u2` in thousandths, so the trusted steps fell from two to none.

## Alternatives

- **Read the tube off the covariance that `channel_error` takes.** One field of errors, from the
  read that `channel_error` is the largest of. Rejected: under two-way clustering, which read is
  the largest depends on the units. On the panel above, with `u2` in units 1e3 times its own,
  `eps` would read 0.01334 where it reads 0.01355. On the `state` log it would read 0.00636 where
  it reads 0.00890.
- **Each lever's largest read, then the root mean over the levers.** It reads no units. But it
  combines reads of different covariances, which no covariance holds, and it moves levers that
  share units and boxes: 0.01497 on the `free` log, where `channel_error * 2` reads 0.01355.
- **The triangle inequality,** `sum_j c_j reach_j` with `c_j` the column's norm. It bounds
  `||(B_hat - B) u||` for each draw of the error. But `eps` is a one-sigma scale, not a bound
  (ADR 0080), and the sum moves levers that share units and boxes by up to `m` times: 1.99 times
  on the `free` log.
- **The root sum of squares over the levers,** `sqrt(sum_j (e_rj reach_j)^2)`. It is the scale
  of `(B_hat - B) u` on one state with every lever at its largest action, and a lever that cannot
  act adds nothing to it, where the root mean counts that lever in its denominator. But it moves
  every tube of `m` levers by `sqrt(m)`, 1.41 times on the panel, where the units were not at
  fault. It would sum over the levers and average over the states. And the tube's coverage
  (ADR 0080) was measured on one lever, where the three forms agree. A new normalisation of the
  tube needs its own coverage evidence.
- **The largest `u' S_i u` over the box's corners,** `S_i` the covariance of state `i`'s row of
  the channel. It reads the levers' covariance, which the entries' errors leave out, and no
  units. But it needs each state's row covariance kept, and a search over `2^m` corners. It also
  moves the scale by at least `sqrt(m)`, for the reason above.
- **A public field.** Not added. Its shape follows the covariance's reads, one or three, which is
  the fit's own business, and the tube is its only reader. `channel_error` stays the public scale.
