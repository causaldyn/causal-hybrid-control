# ADR 0039 — An observational channel checked against lift tests

**Status:** proposed, 2026-10-01.

## Context

ADR 0033 fitted a channel to geo tests and left the observational fit's side unbuilt: a check of
the channel an analyst reads off logged spend against the tests' gaps, and the gap between the two
used to calibrate the sensitivity of channels no test reached. On Heusch's generator (2026b), where
spend follows the business, an observational model reads paid shopping's return as 8.45 against a
true 4.20, with the controls an oracle would choose (Heusch 2026a). The tests are what can say so,
and by how much.

De Bartolomeis, Abad, Donhauser and Yang (AISTATS 2024) bound the marginal sensitivity model's `Γ`
from below with a trial: the least `Γ` whose sharp bounds on the observational effect reach the
trial's interval. A media-mix channel's spend is continuous and its effect a curve with carryover,
so the transfer here keeps their inversion and replaces the ATE by the lift the tests read.

## Decision

- **`check_observational(fit, observed)`** in `chc.lift`, returning an `ObservationalCheck`. It
  reads the tests `fit` was fitted to, which `LiftFit.tests` now carries, so a check cannot be
  handed other tests than the fit read.
- **An `F` test of the channel as a point of the fit's model**: the least squares of the gaps it
  predicts over the fit's, per parameter, over the fit's noise variance, against `F_{p, n-p}`. It
  is the extra-sum-of-squares test, the joint region whose profiles are the fit's intervals (Bates
  and Watts 1988). The channel must be of the fit's families, kernel length and form; one that
  fits the tests better than the fit, beyond what the fit tells apart, means the fit is not their
  least squares, and the check refuses rather than report a negative statistic.
- **The factor**: the least-squares multiple of the gaps the channel predicts that the tests read,
  `c = g·d / g·g`, with the `t` interval `c ± t s / |g|` from the fit's noise. It is the tested
  lift over the predicted one, each period weighted by what the channel predicts there, so tests
  of either sign add. It is linear in the gaps and unbiased for `g·m / g·g`, `m` the true gaps,
  whatever the channel's shape.
- **`ObservationalCheck.least_gamma(cvar_gap)`**: in units of the lift the channel predicts,
  `chc.sensitivity`'s identified set is `1 ± (Γ-1)/(Γ+1)·cvar_gap`, and the least `Γ` whose set
  reaches the factor's interval inverts the radius at the interval's distance from 1, as
  `barrier_gamma_star` does. `cvar_gap` is the analyst's, as everywhere `Γ` is spent; a
  reallocation's flip point (`docs/concepts/gamma.md`) reads it the same way. The `Γ` is the
  marginal sensitivity model's, as every `Γ` the library reports is.

## Consequences

- `tests/test_lift.py` holds the statistic and the factor to NumPy on the fixture's tests. The
  fit's own channel reads a factor of 1 and nothing to reject, since its coefficient is the
  least-squares multiple of its shape; `k` times it reads `1/k`, with the excess `(k-1)²|g|²` in
  closed form. A channel whose carryover lasts four times the truth's, scaled to the size the
  tests read, reads a factor of 1 and is rejected at `p < 1e-6`; the truth passes at 0.19.
  `least_gamma` is the first `Γ` on a grid whose set reaches the interval. Twelve mutations
  caught; a thirteenth rewrote the factor's arithmetic without changing it.
- The `F` is exact for a model linear in its parameters. Where the tests leave a parameter on a
  ridge, as Heusch's leave the scale, the fit has fewer effective parameters than it counts, and
  the test can be conservative, rejecting the truth less often than its level: power lost, not
  validity.
- A fit now holds its tests, so the check reads what the fit read and cannot be handed other
  tests.
- causaldyn-bench's `observational_check` track measures the check on Heusch's world, where both
  the truth and the observational fit are known.

## Alternatives

- **A ratio model**, the true lift within a factor `[1/Γ, Γ]` of the predicted one. Scale-free and
  read off the factor's interval with no analyst's gap, but a third meaning of one symbol, where
  the library's docs promise every `Γ` is the marginal sensitivity model's. The factor's interval
  carries the same information under its own name.
- **The tests' total lift, `Σ d`, as the moment.** Tests of opposite signs cancel in it, and it
  weights the periods where the channel predicts nothing as heavily as those where it predicts
  most.
- **A `χ²` against a known noise.** The noise is estimated, by the fit, so its reference is `F`.
- **Stacking the tests and the history into one fit.** On Heusch's world the observational moment
  is biased even with oracle controls, and a stacked fit would be a biased compromise; the check
  says whether the two may be combined, and this ADR does not combine them.

## Not built

- **The combination when the check passes**, and the plan from the tests alone over the tested
  range when it fails.
- **`Γ` carried to a channel no test reached.** The floor bounds the tested channel's confounding;
  a plan that reads it for another channel assumes the two are confounded alike, and still needs
  an assumed ceiling.
