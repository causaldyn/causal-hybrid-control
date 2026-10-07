# ADR 0032 — An omitted-confounder bound for a fitted channel

**Status:** proposed, 2026-09-30.

## Context

CHC prices hidden confounding in one currency, the marginal sensitivity model's `Γ`, and it answers
what a plan's certificate asks: how far a confounder may move the odds of treatment before the
guarantee fails. A media-mix decision asks a narrower question of the fitted channel itself: is a
reallocation still worth making if the adjustment set missed a confounder, and how strong would
that confounder have to be to flip it? The question is linear in the channel. Its standard answer
is the omitted-variable bound of Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2022), in the
partial R² in which Cinelli and Hazlett's robustness value is read. CHC had that robustness value
for an OLS fit (`chc.causal.sensitivity_analysis`), not for `fit_causal_residual`'s cross-fitted
channel.

## Decision

- **`omitted_confounder_bound(fit, functional, *, cf_y, cf_d, rho=1, level=0.95, null=0)`** in
  `chc.dynamics_id`, returning an `OmittedConfounderBound`: the estimate, `bias_scale`, the sum
  over states of `σ_s ν_s`, the strength `|ρ| sqrt(cf_y cf_d / (1 - cf_d))`, the point bounds, the
  confidence bounds one-sided at `level`, and the robustness value of each. `functional` weighs
  the channel's coefficients in the channel's own shape, so a lever's effect and a move of spend
  between two levers are the same argument.
- **The representer is the moment's own weight on each row, with the cross-fitted nuisances
  held**: `N (D'D)^-1 D_i` under Euler, DoubleML's `v / mean(v^2)` for one lever. Under `rk4` it is
  carried through the fixed point to first order, as the fit's influence is, less the drift
  regression's own weight on the row, which reads the state alone. A fit made with
  `influence=True` keeps it, with the moment's residual.
- **Several states sum `σ_s ν_s`**, with the shares common to them.
- **The robustness value is a closed form**, `(sqrt(f^4 + 4 f^2) - f^2) / 2` at
  `f = |estimate - null| / (|ρ| bias_scale)`, Cinelli and Hazlett's. The confidence bound's is the
  root of that bound's distance to the null between 0 and it.
- **Refused**: a fit by instrument, whose moment's representer is the instrument's; a weighted fit;
  a fit made without `influence=True`; a functional of another shape; shares outside `[0, 1)`, `ρ`
  outside `[-1, 1]`, a level outside `[0.5, 1)`.

## Consequences

- On a linear Gaussian plant the bound is attained with one latent, for any functional of any
  number of levers, and strict by `(a1 b2 - a2 b1)^2 / (sv V)` in its square with two
  (`validation/omitted_confounder_bound.mac`). The tests therefore hold the bound's side on the
  true channel, not merely beyond it, and a bound that was valid but loose would fail them.
- Over 500 worlds with drawn loadings, half read by Euler and half by `rk4`, the confidence bounds
  covered the true reallocation in 0.940 of them (Clopper-Pearson 0.915-0.959) at a nominal 0.95,
  every miss on the side the bias points to (`scripts/bench_omitted_confounder.py`).
- DoubleML 0.11.4, fed the same cross-fitted predictions, gives the same bounds to 3.8e-10, the
  ridge on CHC's moment. The confidence bounds differ by 8e-5, since CHC's influence of the
  estimate is its own, robust to unequal noise; DoubleML's robustness values stop at its scalar
  minimiser's tolerance, 2.9e-5 from the null (`scripts/doubleml_bound_reference.py`).
- A fit made with `influence=True` carries `N n q` more numbers for the representer and `N n` for
  the residual.

## Alternatives

- **The whole fit's response to each row as the representer**, cross-fitted nuisances included,
  as the fit's influence is. Rejected: the nuisances' error enters its second moment with a
  positive sign, an upward bias of the order of the nuisance features over `N`, which grows with
  every covariate the adjustment set holds.
- **DoubleML as a dependency.** One function's worth of code, and its partially linear model fits
  one treatment at a time with the others as covariates, not a channel over states, and knows no
  `rk4` reading. It stays the oracle.
- **A bound on a plan's value.** A plan's value is not linear in the channel, so the bound would
  hold to first order only. The reallocation, which is linear, is the case asked about.

## Not built

- **Benchmarks for the shares**, Cinelli and Hazlett's "a confounder `k` times as strong as
  covariate `j`". They carry the caveat `benchmark_gamma` carries: nothing formal ties an
  unobserved confounder to an observed covariate (Ding 2024, p. 242).
