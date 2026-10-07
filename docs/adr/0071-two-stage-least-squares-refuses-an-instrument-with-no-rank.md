# ADR 0071 — Two-stage least squares refuses an instrument whose moment has no rank

**Status:** accepted, 2026-10-07.

## Context

`estimate_effect_iv` regresses the action on the state, the instrument and a constant, then the
outcome on the state, the fitted action and a constant, and returns the fitted action's
coefficient. `IV2SLS` returns that coefficient as its effect. Nothing checked that the instrument
moves the action. Where it moves nothing beyond what the state and a constant explain, the fitted
action is collinear with them, the second stage's design is singular, and `jnp.linalg.lstsq`
returns its minimum-norm coefficient: a number, with no sign that it estimates nothing.

On 40,000 rows of `ConfoundedLinearSystem(gamma=1.0)`, where the truth is 1.0, the unfixed release
read:

| instrument | estimate |
|---|---|
| the log's own, which moves the action | 1.0085 |
| zeros | -0.0032 |
| a constant, 5 | -0.0032 |
| noise less its projection on a constant, the state and the action | -0.0032 |
| the state's affine copy, `2x + 3`, where the action moves with the state by `0.8x` | 0.63 |
| the same, the log in single precision | 2.61 |
| the same, the state alone in single precision and the copy in double | 0.15 |
| the log's own, where the state sets the action, `0.7x + 2` | 0.88 |
| zeros, beside a state of zeros that never moves | 5.1e-5 |
| noise drawn apart from the action | 0.263 |

The fifth row is the danger: the copy correlates with the action at 0.39, and 0.63 reads as an
estimate. ADR 0067 decides the same defect for the channel fit's instrument, in
`fit_causal_residual`, which returns a fit with an `identified` flag.

## Decision

- **The relevance.** ADR 0067's, for one action and one instrument: the canonical correlation
  between the first stage's push on the action and the action, each less its least-squares
  projection on a constant and the state. It is the partial correlation of the action and the
  instrument given the state, in absolute value: the root of the first stage's partial `R^2`. Each
  column is centred before its projection, and the state scaled to unit spread, so it reads the
  same in any units of the three columns and at any level of them.
- **Rounding.** The library's rule: a spread within 64 eps of its column's root mean square is
  rounding. Each column is known to that, so the action's rounding can move the product of the
  two columns beyond the state by 64 eps of the action's root mean square times the instrument's
  spread, and the instrument's rounding by 64 eps of its own times the action's spread. A product
  within the sum of the two reads a relevance of 0, a moment with no rank; so does a column with
  no spread beyond the state, or none beyond its rounding, whose product is within its own term.
  The eps is that of the coarsest precision the three columns come in: a column in single
  precision keeps single precision's rounding when the estimate computes in double. The state's
  copies above sat 0.35 and 0.20 single eps beyond the state, and read in double's eps their
  products came to 4.1e4 and 5.8e5 eps of the sum, a relevance.
- **A relevance of 0 raises `ValueError`**, naming the instrument. `estimate_effect_iv` returns a
  bare number, and `IV2SLS` an `EffectEstimate` whose `effect` a controller consumes as one;
  neither has a place for a flag. `chc.causal` refuses with `ValueError` wherever it has no answer
  to give. `IV2SLS` reads its estimate through the same function, so it refuses too.
- **`IV2SLS` reports the relevance**, experimental, as `diagnostics["instrument_relevance"]`, the
  reading ADR 0067 exposes on the channel fit. It grades an instrument that keeps the rank.
- **`solve_channel_moment` checks no rank.** It is public, in a stable module, and solves the
  moment for the caller's own `instrument_action`. It sees neither the raw shifter nor the
  covariates the nuisances took out, and the rank reads both. The rank of the moment it solves
  reads noise around zero, not zero: ADR 0067 measured its singular values at 1.9e-4 to 3.2e-3 a
  row with an instrument of zeros. A check there would refuse an instrument of exact zeros alone.
  And in a stable module a raise where a number was returned is a break, which the docs allow only
  where the number broke a promise; this one keeps its promise, to solve the moment it was handed.
  Its docstring says it checks no rank, and that the caller settles it.

## Consequences

- *Each row of the table but the first and the last raises.*
- *An instrument that moves the action estimates as before, bit for bit*: the check runs before
  the two stages and does not enter them. The plant's 1.0085126809414267 is pinned against the
  unfixed release's two stages, recomputed in the test on the machine that runs it.
- *The rank does not grade an instrument; the relevance does.* A column of noise drawn apart from
  the action keeps the rank and estimates 0.263, its relevance 0.0087 where the log's own
  instrument reads 0.53. No inference here is robust to a weak instrument, and the relevance is
  not a test.
- *Units and levels.* With the instrument, the action or the state in a millionth or a million
  millionth of its units, a million or a million million times them, or its sign flipped, the
  relevance of the plant's instrument and of the noise moved by 5.6e-16 of itself at most, and no
  decision moved. On the tests' logs short of rank, so read, with the instrument or the action a
  million above or below zero, and drawn from a state that far from it, the largest product read
  as rounding came to 0.018 eps of the sum in double precision and 2.4 single eps in single; over
  20 to 100,000 rows of the same plant, with constants, noise less its projection and the state's
  copy in any of those units, 0.47 eps. The plant's instrument reads 1.2e15 eps, and 2.2e6 single
  eps in single precision; noise drawn apart, 2.0e13 eps.
- *The two stages keep their limits.* Their least squares drops any direction of its design under
  eps times the rows of its largest. With the state a million from zero, or in a million millionth
  of its units, the plant's instrument estimates 1.0049, unadjusted for the state, where it reads
  1.0085 otherwise; with the instrument in such units, -0.0032 again, though its relevance reads
  0.53. Reading the stages in scaled columns would move every estimate's last bits, so it is not
  done here.
- *`estimate_effect_iv` no longer traces under `jax.jit`*: it reads the data to decide whether to
  raise. Nothing in the library or its notebooks traces it.
- *A nan in the log still reads a nan estimate*, as it did: a nan relevance is not 0.
- *Cost*: two least-squares projections on two columns, beside the two stages' own.
- *Tests*: an instrument of zeros, a constant one, noise less its projection on the state and the
  action, the state's affine copy where the action moves with the state, in single precision and
  beside the state in single precision, and an action the state sets, each refused, and again
  with the instrument, the action or the state in a millionth or a million millionth of its units,
  a million or a million million times them, or its sign flipped, with the instrument or the action
  a million above or below zero, and drawn from a state that far from it; a state that never
  moves, at 0 and at 4, beside the plant's instrument and beside zeros; the plant's instrument
  pinned bit for bit, and in a millionth or a million times its units to 1e-9; `IV2SLS` refusing
  under the caller's names, and its relevance against numpy's partial correlation to 1e-12, in the
  same units, for the plant's instrument and for noise.

## Alternatives

- **A result with an `identified` flag**, as `fit_causal_residual` returns. Rejected:
  `estimate_effect_iv` returns an `Array`, and every caller reads it as a number; a flag needs a
  new return type, a change of signature.
- **nan.** Rejected: it reads false in every comparison and passes into a plan unnoticed, and the
  library refuses a nan where it reads one.
- **`chc.decision.NotIdentifiedError`.** Rejected: it is a `DecisionError`, for a decision
  `prescribe` could not set up, in a module that imports `chc.causal`. Every refusal of
  `chc.causal` is a `ValueError`, which `NotIdentifiedError` subclasses, so a caller who catches
  that catches both; a type of its own for this one would be a new public name.
- **The second stage's rank as `lstsq` reads it**, its singular values against `eps * max(N, p)`
  of the largest. Rejected: that cutoff grows with the rows and moves with the columns' units, and
  it is the reading that returned the minimum-norm coefficient in the first place.
- **Two floors**, the instrument's spread beyond the state against 64 eps of its root mean square
  and its push on the action against 64 eps of the action's. Rejected: they count the action's
  rounding and not the instrument's, which far from zero fakes a push of its own. Noise less its
  projection, logged a million above zero, kept a relevance of 4.4e-13 and estimated. The sum
  holds both floors: a spread within its rounding leaves a product within its term.
- **A threshold on the first stage's F**, or Stock and Yogo's critical values. Rejected, as in
  ADR 0067: a test of weakness has a size, which is the inference this does not do, and it would
  refuse a log drawn with an instrument apart from the action, which `IV2SLS`'s frame test reads.
- **A rank check in `solve_channel_moment`.** Rejected above.
