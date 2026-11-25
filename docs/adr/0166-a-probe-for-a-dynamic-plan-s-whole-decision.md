# ADR 0166 — A probe for a dynamic plan's whole decision

**Status:** accepted, 2026-11-25. Experimental (`chc.experiment`). Amends ADR 0011: its "Not built"
dynamic decision is built, for an LQR gain. No default moves.

## Context

ADR 0011 sizes an experiment before a one-shot decision, and `dither_covariance` (ADR 0139)
designs a dither's covariance for a plan whose regret depends on its channel alone, the channel
read off the dither. Neither covers a dynamic plan. Its decision also depends on the drift, and in
a closed loop the probe moves the state the fit reads, so the information the fit gets is not the
probe's covariance. `docs/lifecycle.md` listed "an experiment design for a dynamic plan's whole
decision, beyond one lever's effect" as not built.

Wagenmaker, Simchowitz and Jamieson (*Task-optimal exploration in linear dynamical systems*, ICML
2021) give the criterion. A certainty-equivalent decision re-solved on an estimate loses, to
first order, the trace of the decision's Hessian in the model parameters against the inverse of
the information the exploration gathers. They design the exploration's spectrum under a power
constraint and show the design task-optimal.

For the linear plant `x' = A x + B u + w`, `w ~ N(0, S_w)`, the LQR gain of `E[x' Q x + u' R u]`,
and `T` periods of `u = -K x + eta`, `eta ~ N(0, U)` drawn afresh each period:

- **The weight.** By the cost-difference lemma the re-solved gain's regret is
  `Sigma_K (K - K*)' (R + B' P B) (K - K*)`, so to second order in the fit's error `e` it is
  `e' W e / 2`, `W = 2 J' (G (x) Sigma*) J`, `J` the gain's Jacobian. That is twice
  `ce_regret_law`'s second-order matrix, and `tr(W) / 2` is its `c2`: 2.3416 and 20.191 on the two
  plants its docstring quotes.
- **The information.** The least-squares fit of `[A B]` on `z = (x, u)` has, to first order, the
  error `N(0, S_w (x) Gamma^-1 / T)` row by row, `Gamma` the stationary covariance of `z`. So the
  expected regret is `Phi(U) / T` with `Phi(U) = tr(Wn Gamma(U)^-1) / 2`, `Wn` the weight traced
  against `S_w`. That is `ChannelMove.price`'s re-plan price at the fit's covariance.
- **Where the probe's information goes.** `Gamma = L diag(Sx, U) L'`, `L = [[I, 0], [-K, I]]`, so
  `Gamma^-1 = E Sx^-1 E' + F U^-1 F'`, `F = [K'; I]`. The fit reads `B` off the probe alone and the
  closed loop `A - B K` off the state's excitation `Sx(U)`, the Lyapunov solution driven by
  `B U B' + S_w`:

      Phi(U) = (tr(Wn_xx Sx(U)^-1) + tr(F' Wn F U^-1)) / 2.

  `Sx` is affine in `U`, so `Phi` is convex, and decreasing, so a cap on the power binds.
- **The static case.** A weight on `B` alone leaves the second term alone, whose optimum under
  `tr U = P` is `P M / tr M`, `M = (F' Wn F)^1/2`: `dither_covariance`'s matrix geometric mean with
  `C = I`. A cap on `tr(C U)` is the same cap on levers scaled by `C^1/2`.

`validation/dynamic_probe_design.mac` checks it in eight steps and 19 verdicts. All pass in
Maxima 5.50 and in CI's 5.49 (`localhost/chc-maxima:26.04`):

- the cost-difference lemma and `J(k*) = p s2` on a scalar plant, symbolic modulo the Riccati
  equation;
- the gain's implicit derivative against the Lyapunov form `chc.regret._gain_jacobian` computes,
  symbolic modulo the Riccati equation;
- the exact regret's Hessian against `2 G Sigma* g g'` at 40 digits, with `ce_regret_law`'s `c2`;
- the split of `Gamma^-1` and its lever block, symbolic on two states and two levers;
- `Phi` two ways on a scalar plant, 17.124638317676437346 at `U = 1/2`;
- the gradient, in its direct and adjoint-Lyapunov forms, and the second derivative, symbolic on
  one state and two levers;
- the cost-difference lemma off the optimum, and the optimum by Newton in 40 digits on the KKT
  conditions, on `a = 1.1`, `B = (1, 0.4)`, `R = [[1, 0.2], [0.2, 0.5]]` at power 1/2. `U*` is
  positive definite and the gradient there is `-nu I`, so it is the global optimum:
  `Phi* = 6.2217573451`, the white probe's 6.9785812211, ratio 0.89155;
- the static closed form and its gradient, symbolic.

## Decision

- **`chc.experiment.probe_design(a, b, q, r, *, noise, power, periods, weight=None, gain=None)`**
  returns a `ProbeDesign`:
  - `covariance`, the probe's `U`, of trace `power`;
  - `regret`, `Phi(U) / periods`, with `method` and `gap`, the Frank-Wolfe gap
    `tr(D U) - power lambda_min(D)` of the convex problem, `D` the gradient;
  - `white_regret` at `power / m` times the identity, and `ratio`;
  - `cost` and `white_cost`, what each probe adds to the average cost per period, exactly
    `tr((R + B' P_K B) U)`, `P_K` the cost-to-go of the gain the probe runs under;
  - `gain`, that gain: the LQR gain on `(a, b)`, or `gain` when given.
- **The weight** is the LQR's own by default, built from `chc.regret.dlqr` and
  `chc.regret._gain_jacobian`, which `ce_regret_law` uses. A caller may pass a weight over
  `(A.ravel(), B.ravel())`, or one over `B.ravel()` alone, as `CausalPlan.decision_weight` reads
  the one-step channel. The latter gives the static design.
- **The solve.** With one lever, or no state block in the weight, the optimum is in closed form.
  Otherwise L-BFGS runs over a Cholesky factor of `U` scaled to the cap, `U = P L L' / tr(L L')`.
  It starts from the white probe and from the second term's closed form, and the better end is
  kept. `Phi` and its gradient take two Lyapunov solves.
- **Refusals** (`ValueError`): a loop the probe runs in that is not stable, which has no stationary
  covariance; a plant with no stabilising LQR gain; a weight that is not symmetric positive
  semidefinite, of neither shape, or zero; a power that is not positive and finite; at most
  `n + m + 1` periods; and the shapes and definiteness of `q`, `r` and `noise`.
- **`probe_design` is exported at the top level**, since the lifecycle page's experiment section
  names it, and is experimental with its module.

## Consequences

`tests/test_probe_design.py` holds 32 tests:

- Maxima's numbers to `1e-10`, the static case against `dither_covariance` to `1e-12`, with and
  without a non-identity noise covariance and through `C^1/2`-scaled levers to the two-lever
  example's `sqrt(3 + sqrt 7)`;
- `Phi` read the long way, against the whole inverse of `Gamma` and a weight from central
  differences of the re-solved gain, and no probe on the cap nearby doing better;
- `cost` against the average cost read directly, under a gain off the optimum.

Each of eight planted errors fails it: the state term dropped, `F`'s sign flipped, the noise left
untraced, the weight halved, the white probe at full power, the cost at `P*`, the gradient's
`U^-1` term dropped, and `Sx` without the probe.

**The Monte Carlo.** On `A = [[0.9, 0.2], [-0.1, 0.7]]`, `B = [[1, 0.1], [0.3, 0.2]]`, `Q = I`,
`R = diag(1, 0.05)`, unit noise and power 1, the gain leans on the weak, cheap second lever. The
design gives that lever 0.83 of the power and predicts 0.753 of the white probe's regret. It adds
0.53 to the average cost per period, where the white probe adds 1.38.

400 runs of 2000 periods were made, and the gain re-solved on each fit. Both arms read the same
noise and the same standard normals for their probes (seed 20261125):

| | realised mean regret | first-order prediction |
|---|---|---|
| designed probe | 9.01e-3 ± 0.45e-3 | 9.10e-3 |
| white probe | 12.04e-3 ± 0.75e-3 | 12.09e-3 |
| ratio | 0.749 ± 0.018 | 0.753 |

The prediction is of the means. On a weaker lever, a channel of
0.05–0.1 at a cost of 0.01, the regret's third- and fourth-order terms are not small at these
lengths, and the first-order level reads low while the ratio still holds. `ce_regret_law`'s
`fourth_order_share` at the fit's covariance says when that is so.

**The whole decision against the channel's effect alone.** `Phi` over the white probe's:

| plant, power | static design on the weight's channel block | closed form of the probe term | design |
|---|---|---|---|
| above, 0.1 | 0.698 | 0.690 | 0.690 |
| above, 1 | 0.758 | 0.753 | 0.753 |
| above, 10 | 0.994 | 1.004 | 0.942 |
| `A = [[1.05, 0.5], [0, 0.3]]`, `B = [[1, 0], [0.5, 1]]`, `R = I`, 0.1 | 1.044 | 0.974 | 0.974 |

- At a power below the noise's, the state is excited mostly by the noise, and the probe term
  decides the design.
- At ten times the noise, the state's excitation is worth designing for: the design leaves 0.942
  where both static designs are no better than white.
- A design on the channel's block alone can lose to the white probe (1.044): it ignores what the
  closed loop's error costs.

## Not built

- **A probe correlated over time.** Wagenmaker et al. design the spectrum. At the same power, a
  coloured probe can excite the state more than a white one, and the state term is where the
  design gains at high power.
- **The probe's length and cost against the regret it removes.** The cap is the caller's.
  `cost` reports what the probe spends, but nothing trades the two.
- **A sequential design**, re-designed on each fit, as their algorithm does.
- **A box on the levers, or a fit that holds `A` known.** The latter reads `B` off `K x` as well.
- **A dynamic plan other than the LQR gain.** `CausalPlan`'s weight covers its one-step channel
  only, and gives the static design here.

## Alternatives

- **An SDP solver.** `cvxpy` is not in the lock, and the problem has a smooth convex objective on
  the spectraplex, which L-BFGS over a factor solves to a certified gap, as `dither_covariance`'s
  binding case already does.
- **A penalty in place of the cap**, `T tr(C U) + uses Phi(U) / T` as `dither_covariance`'s value.
  It needs the uses after the experiment, which the cap leaves to the caller. A cap on the probe's
  own cost is the cap on scaled levers.
- **`dither_covariance` on the weight's channel block.** It ignores the drift's share of the
  decision. It lost to the white probe on the second plant above.
- **Projected gradient on `U`.** It needs an eigendecomposition per step to stay on the cap. The
  factor keeps `U` positive semidefinite by construction.
