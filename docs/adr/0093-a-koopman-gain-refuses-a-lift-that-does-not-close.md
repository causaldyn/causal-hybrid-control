# ADR 0093 — A Koopman gain refuses a lift that does not close

**Status:** accepted, 2026-10-13.

## Context

`KoopmanModel` fits `phi(x') ~ A phi(x) + B u` by least squares on a polynomial lift `phi`, and
`koopman_lqr_gain` solves the discrete LQR of `(A, B)`. Two things went unread.

- **The linear lift is misspecified for a control-affine plant.** For `x' = f(x) + g(x) u` each
  nonlinear feature of the lift moves by a term that is the action times a function of the state:
  `(x1')^2` holds `2 f1(x) g1(x) u`. Where that function is not constant, no dictionary of the
  state makes the term `B u` with a constant `B` (Strässer, Worthmann, Mezić, Berberich, Schaller
  and Allgöwer, Annual Reviews in Control 61, 2026). On the bilinear plant
  `x' = A x + b u + c x1 u`, the linear form's error at each transition is `c x1 u`.
- **Nothing said whether the lift closed.** The fit returned `A` and `B` whatever share of
  `phi(x')` they predicted, and the gain was computed on them. On the bilinear plant of the
  tests, the linear lift leaves 0.73 of the second state's root mean square unpredicted on
  transitions it was not fit on, and its gain was returned as for any other.

## Decision

- **`KoopmanModel(form="bilinear")`** fits `phi(x') ~ A phi(x) + B u + sum_j u_j H_j phi(x)`: the
  design `[phi(x), u]` gains the columns `u_j phi(x)`, each with its own ridge term in units of
  its mean square (ADR 0057), so the fit reads the same in any units. `predict`, `rollout` and the
  new `residuals` carry the bilinear term. The default `form="linear"` fits, predicts and returns
  its gain as before, bit for bit: the same design, Gram and solve.
- **`fit` reports `closure_residuals`.** For each feature of the lift: the root mean square of its
  one-step error on transitions held out of the fit, over the root mean square of the feature at
  the next state. The transitions are cut into five contiguous blocks, and each is predicted by a
  model fit on the other four. The model kept is fit on all of them, so the residual costs five
  more solves of the same size and moves nothing the model returns. It is ResDMD's residual
  (Colbrook and Townsend, CPAM 2024) for the whole dictionary at once, not per eigenpair.
  Read feature by feature, it is unitless: with the state or the action in 1e-9 to 1e6 of its
  units, no feature's residual on the tests' oscillator moves by more than 3e-8. A feature the
  next states never move reads 0 where it is predicted exactly and `inf` otherwise; one
  transition leaves none to hold out, and reads `nan`.
- **`KoopmanModel.residuals(x, u, x_next)`** returns each transition's error in the lift, for
  transitions the caller held out.
- **`koopman_lqr_gain(..., max_residual=0.5)`** raises `ValueError` where a feature's closure
  residual exceeds `max_residual` or is `nan`, naming the feature.
- **A bilinear model's gain needs `x_target`.** Its channel `B + [H_j phi(x)]_j` depends on the
  state; the gain is the LQR of the model linearised at the target, `KoopmanModel.channel`.
  Without a target it raises, since the origin's channel is the wrong one for any other target.

### Why 0.5

At 0.5 the model predicts three quarters of a feature's mean square on transitions it was not fit
on (its uncentred held-out `R^2` is 0.75), and its one-step error is half the feature's own size.
The tests' cubic oscillator under a degree-3 lift reads 0.156 at its worst feature (`x1 x2^2`)
and 1.7e-4 on the state itself, and its LQR drives the plant to its target: a margin of 3.2 under
the threshold, and no test or notebook moves. The linear lift of the bilinear test plant reads
`sqrt(10/19) = 0.73` and is refused; its bilinear lift reads 0 and is not. The threshold is a
convention, as every residual's is: it bounds an `L^2` error in the data's measure, not a
closed-loop guarantee. A caller who has one passes their own.

### Verification

`validation/koopman_bilinear_closure.mac` shows, for two states and one action:

1. on a sample closed under the sign flips of `x` and `u`, `x1 u` is orthogonal to the linear
   columns, so the linear residual is `c_k x1 u` at each transition;
2. the bilinear residual is zero;
3. under Gaussian `x` and `u`, the linear form's closure residual is
   `rho_k = |c_k| s / sqrt(|a_k|^2 + (b_k^2 + c_k^2) s^2)`;
4. its value on the tests' plant, `[sqrt(2/43), sqrt(10/19)]`;
5. the channel at `x` is `b + c x1`.

The tests check (1) and (2) to round-off without a ridge, (3) on 20 000 Gaussian transitions to
3 %, (5) against `dlqr` of the plant linearised at a target, and that a linear plant reads
round-off under both forms.

## Consequences

- A caller of `koopman_lqr_gain` whose lift does not close gets an error where they got a gain.
  `max_residual=np.inf` restores the old behaviour.
- The bilinear form has `d_u` times more coefficients, and its LQR is one linearisation: a target
  far from where the model is linearised needs another gain.
- In discrete time the bilinear form closes on a degree-1 lift of a bilinear plant, but not on a
  higher-degree lift: `(x1')^2` also holds `u^2`, which neither form carries. On the cubic
  oscillator the bilinear lift moves the worst residual from 0.156 to 0.150, since the degree-3
  dictionary does not hold `x1^4` either.

## Alternatives

- **Gate on the decoded state alone.** The LQR runs on the whole lifted `A`, and a feature that
  drifts feeds the state through it. ResDMD reads the dictionary; so does the gate.
- **The in-sample residual.** It reads the fit, not the closure: a rich dictionary fits noise-free
  data it does not close on.
- **Random folds.** On a logged trajectory neighbouring transitions are near copies, and a random
  split reads them across the cut. Contiguous blocks do not.
- **Fit on a share and hold the rest out.** The model kept would no longer be the default fit.
- **A centred ratio, one minus `R^2`.** The lift has no constant feature and the fit no intercept;
  the uncentred ratio is the error relative to what the fit predicts, as the ridge's mean squares
  are (ADR 0057).
- **`u^2` columns, or a gain schedule for the bilinear model.** Each is a larger change than a
  form that matches the plant's control-affine structure, and neither is needed for the refusal.
