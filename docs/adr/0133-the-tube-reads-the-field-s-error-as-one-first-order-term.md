# ADR 0133 — The tube reads the field's error as one first-order term, under either integrator

**Status:** accepted, 2026-10-25. Amends ADR 0131 (what the Euler field inherits), ADR 0091 (the
distance the channel's error is priced at) and the drift's side of ADR 0104 (the field's error at
the plan's states); keeps ADR 0104's degrees of freedom. Amended by ADR 0141, 2026-10-27: under
`rk4` the field's error is read where each of RK4's four stages reads the field, each carried by its
own weight; read at the step's start alone it dropped a term of the order of `dt^2`. Euler fits
read as before.

## Context

The tube's field part states that the fitted field's miss at a planned state and any action of the
box is within `t` of its standard error. ADR 0131 read that error as two parts in quadrature: the
channel's error times the action's distance from the log's mean action `u_bar`, and the field's
error at `u_bar`, which is the drift's own error plus what the Euler field inherits there. Three
first-order terms were still missing.

- **The channel's part and the inherited part are one term.** A channel error `e` moves the field
  at an action `u` by `e' (g(x, u) - Gamma' phi(x))`. `m(x) = Gamma' phi(x)` is the log's
  projected action: what the drift's features predict of the logged actions. So the exact Euler
  term is `(u - m(x))' C (u - m(x))`. The quadrature drops `2 (u - u_bar)' C (u_bar - m(x))`.
  Beyond `u_bar`, away from `m(x)`, it reads up to `sqrt 2` too little in standard deviation;
  toward `m(x)` it reads too much (`validation/tube_projected_action.mac`).
- **Under `rk4` nothing was inherited.** The `rk4` fixed point moves the drift with the channel's
  error through the drift's block of its Newton matrix, by `-K_rr^-1 (Gamma - W Y_c)`. It also
  moves the channel with the drift's own error. So the field's variance is
  `V_cond + L^2 V_b + 2 L cross`, and the cross term is not zero, as Frisch and Waugh (1933) make
  it under Euler. At a rational point it takes back more than `L^2 V_b` adds
  (`validation/rk4_inherited.mac` (4)). In the scalar case the projected action's slope is
  `-S(z) Gamma / (R'(z) + g S'(z))`, `z = h a`, `g = h b Gamma` (`rk4_inherited.mac` (5)).
- **The push's mean was read from independent rows.** The intercept holds the adjustment set's
  push at its mean on the log. On a log whose push persists, the mean's variance is the long-run
  one: under AR(0.8), 8.8 times the independent rows' read at 200 rows, 9 on a long log
  (`validation/push_mean_variance.mac`). The drift's own error was pooled the same way. The
  channel's error was already read from the clusters' sums or the periods' cosines (ADR 0075).

The measurements below use plants stepped by their **exact flow**. On a plant stepped by the Euler
map, `rk4` reads `log(1 + a dt)/dt`, not `a` (`rk4_inherited.mac` (6)): at `a = -0.5`, `dt = 0.2`
it reads `-0.5268`, and `b = 1.0536`. A study that scores an `rk4` fit of an Euler-map plant
against the plant's law mixes that estimand bias with the error. The study behind ADR 0131 did
that in its `rk4` column: 40 % of that column's mean square was the bias. ADR 0131 states no `rk4`
number, and none of that column is used here.

## Decision

- **An identified fit keeps the field's terms** (`CausalDynamicsFit._field_terms`, under either
  integrator). They hold the log's mean action, the channel's covariances, each state's drift
  coefficients' move with the channel's error (`projection`), the drift's own covariance, its
  covariance with the channel (`cross`), and the push's mean variance. `_drift_inherited` and
  `_drift_inherited_error` are gone.
- **The tube reads one term from the projected action.** At a state `x` and an action `u`, state
  `s`'s variance is `phi' D_s phi + L_s' C L_s + 2 L_s' X_s phi + P_s`, with
  `L_s = e_s (x) g(x, u) - projection_s' phi`. Its sum over the states is a convex quadratic in
  `u`, so the tube reads its largest over the box's vertices. It takes the largest over the
  channel's covariances, as before. The guarantee stays "any action in the box". `prescribe`
  passes the box; a caller that passes only each lever's reach reads `u_bar +- reach`.
- **Under `rk4`** the projection and the drift's own move go through the drift's block of the
  fixed point's Newton matrix. The cross term reads the channel's weight on each row against the
  drift's, with the pooled noise.
- **Given `clusters` or `periods`**, the channel's scores, the drift's scores and the push are
  summed within each cluster or projected on the periods' cosines **together**. Each part is read
  at its own small-sample factor, and their cross-covariances come from the same sums. So
  `drift_error`, the drift's blocks and the push's mean are read as the channel's error is.
  `drift_error` keeps its meaning: the regression's own error, conditional on the fitted channel.
- **Given neither**, every part is read from the rows as before: the drift's own error and its
  cross term from the pooled residual, the push's mean as independent rows.
  `PersistenceCheck`'s docstring says so. `prescribe` always passes clusters or periods where a
  persistence test can be read.
- **The degrees of freedom stay the channel's** (ADR 0104). With clusters or periods every part
  now comes from the same independent terms, so that count is the right one for the whole error.

## Consequences

The field's miss at a state 2.5 of the log's spreads from its mean, scaled by the error the tube
reads there, on exact-flow plants (C1: `x' = -0.5 x + u + 0.8 z`, `dt = 0.2`; C2:
`x' = -0.7 x + 0.8 u + z`, `dt = 1`; the logged `u = -0.5 x + 0.8 z + 0.5 e`; 200 transitions;
linear nuisances, one fold). Root mean square / share within `+-1.96`, before and after:

| probe | integrator | C1 before | C1 after | C2 before | C2 after |
|---|---|---|---|---|---|
| `u_bar` (400 logs) | Euler | 1.027 / 0.938 | 1.016 / 0.950 | 1.021 / 0.932 | 1.015 / 0.945 |
| `u_bar` (400 logs) | `rk4` | 1.529 / 0.802 | 1.018 / 0.950 | 1.877 / 0.703 | 1.029 / 0.935 |
| far vertex of `u_bar +- 2` (200 logs) | Euler | 1.186 / 0.890 | 1.039 / 0.925 | 1.177 / 0.895 | 1.037 / 0.940 |
| far vertex of `u_bar +- 2` (200 logs) | `rk4` | 1.396 / 0.820 | 1.037 / 0.925 | 1.870 / 0.660 | 1.038 / 0.940 |

On C1 with an AR(0.8) push (300 logs) the fit given `periods`, at `u_bar`, read 1.886 / 0.697
under Euler and 2.733 / 0.523 under `rk4`, and `+-t_13` held 0.747 and 0.550. Now it reads 1.116 /
0.913 under both, and `+-t_13` holds 0.937. The rest of the gap is the cosines' own shortfall at a
fixed `nu`: at 200 periods and `nu = 13` their read's mean is 0.781 of the long-run variance
(`push_mean_variance.mac` (3)). Without `periods` the fit reads 1.870 / 0.690, as before.

- An `rk4` fit's tube widens where the log's actions lean on the state; at the mean action of C2
  the cross term is about −38 % of the field's variance, so adding Euler's term alone would
  over-widen it.
- An Euler fit's tube widens at the box's vertex away from the projected action, and narrows at
  the vertex toward it.
- The channel's error does not move. `drift_error` moves only for fits given clusters or periods.
- On `scripts/bench_short_log_tube.py`'s one-unit logs of 60 transitions, 100 logs each, the
  share inside the tube at the first and the last step read, before and after: one lever 0.952 –
  0.969 and 0.956 – 0.974; two levers 0.955 – 0.985 and 0.965 – 0.994; the bilinear plant (95
  logs) 0.999 – 0.972 and 0.999 – 0.976. These plants step by the Euler map and are fitted by
  `rk4`, so the fit's estimand bias rides in the field's miss. On short logs the tube widens, most
  where two levers lean on the state.
- ADR 0104's AR(0.8) alternative, a drift error robust to clusters and to persistence, is now
  taken, with the push's mean beside it. On the misspecification suite's AR(0.8) demand, 16 logs,
  the weakest step reads 0.9483, where it read 0.9463. Every other case of that suite moves by at
  most 0.010, upward, and no floor moves. In its well-specified case one of 16 logs now trusts 2
  steps where it trusted 3.

## Alternatives

- **Read the field's error as `grad_theta f' P grad_theta f`** from the joint covariance the `rk4`
  fixed point already builds. That is one form for both integrators. Its drift block reads the
  channel's robust spread, not the pooled noise, and so changes `drift_error`'s meaning and ADR
  0104's degrees of freedom. Not taken until its drift block is shown to equal the pooled form's.
- **Add Euler's `L^2 V_b` under `rk4` too**, without the cross term. On C2 that read 0.958 /
  0.958 at `u_bar`: too wide, since the cross term takes back 38 %.
- **One delta-method tube at the plan's own actions**, through the rollout's sensitivities. It
  keeps the cancellation across steps and is narrower when actions change sign, but it narrows the
  guarantee from any action in the box to the plan's actions. Later, opt-in.
- **Keep the quadrature.** It holds 0.89 of the logs at the far vertex under Euler and 0.66 under
  `rk4` where it states 0.95.
