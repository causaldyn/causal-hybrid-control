# ADR 0131 — The tube reads what the Euler field inherits at the log's mean action

**Status:** accepted, 2026-10-24. Amends ADR 0091 (the drift absorbs the channel at the log's mean
action) and ADR 0104 (the field's error at the plan's states), for Euler fits. Amended by ADR 0133,
2026-10-25: the inherited part and the channel's part are one term, priced from the log's projected
action, with no quadrature; `rk4` fits read theirs through the fixed point, the cross term with
it; and given clusters or periods the push's mean is read from them. `_drift_inherited` is gone.

## Context

The tube reads the fitted field's error at a planned state and action as two parts in quadrature:
the field's error at the log's mean action, which was the drift regression's own standard error,
and the channel's error times the action's distance from that mean (ADR 0091). The drift's own error
is the regression's, conditional on the fitted channel, with its noise the regression's residual.

ADR 0129 made the Euler fit hold the adjustment set's push in its drift. The push left the
regression's residual, and the drift's own error fell to what the noise alone leaves. Two errors it
had covered by accident were then read by nothing:

- **The push's mean on the log.** The push is held about its mean on the log, so the intercept
  carries that mean, which is off the plant's by the mean's own sampling error.
- **The channel's error, through the regression's target.** The regression reads its target at
  the fitted channel, so a channel error `e` moves the field at the mean action `u_bar` by
  `e' (g(x, u_bar) - Gamma' phi(x))`, with `g` the channel's regressors and `Gamma` the
  regression's coefficients on them. Where the log's actions are apart from the design, `Gamma`
  reads `u_bar` and the term is zero, which is ADR 0091's reading. Where the actions lean on the
  state or on the drivers, it is not zero.

On 40 logs of 200 transitions of `x' = -0.5 x + u + 0.8 z`, the logged `u` leaning on the state by
-0.5, fitted by Euler with `z` adjusted for, the drift's miss at a state 2.5 of the log's spreads
from its mean read 1.80 of its standard error in root mean square, and `+-1.96` of it held 0.72 of
the logs. The push's mean alone took that to 1.15 and 0.95, and the channel's term with it to 1.06
and 0.97. Before ADR 0129 the miss read 1.004, because the push's spread in the residual covered
both.

## Decision

- **An identified Euler fit keeps what its field inherits at the log's mean action**
  (`CausalDynamicsFit._drift_inherited`). It holds one block per channel covariance and per state,
  in the features of the larger of the drift's and the channel's degrees, then the drivers. Each
  block is the channel's covariance carried by `u_bar I - Gamma`, plus the push's mean square over
  `N^2` on the intercept. The push term counts the rows as independent, as the drift's own error
  does. `_drift_inherited_error` is the block's largest root mean at the log's rows.
- **The tube adds it to the drift's own error**, at each planned state (`_field_errors`) and over
  the log's rows (`_field_error`). It takes the largest over the channel's covariances, as the
  channel's own error does.
- `drift_error` keeps its meaning: the regression's own error, conditional on the fitted channel.
- The inherited part shares the channel's error with the channel's part at an action. The tube
  adds the two in quadrature, as it adds the rest, and does not read their cross term.

## Consequences

- An Euler fit's tube widens where the log's actions lean on the state or the drivers, and where
  the adjustment set's push is large against the noise. On the quickstart the trustworthy prefix
  stays 3 steps.
- A `"rk4"` fit reads as before. Its drift regression holds the push too (ADR 0120), so the same
  two terms are missing there. There they would go through the fixed point's own drift
  covariance, and that is not done.
- The channel's error and the drift's own error do not move.

## Alternatives

- **Read the drift's noise from the field's defect**, the push's spread included, as the Euler fit
  read it before ADR 0129. In the same test that read 0.53, with `+-1.96` holding every log. It is
  twice too wide wherever the push is large, and it does not read the channel's term.
- **Read the field's error from one robust covariance of the channel and the drift together.** It
  would read the cross term too. It changes the drift's noise from the pooled residual to the
  channel's robust spread, and the tube's degrees of freedom with it (ADR 0104). Not done here.
- **Keep the reading as it is.** Then the tube's field part holds 0.72 of the logs where it states
  0.95.
