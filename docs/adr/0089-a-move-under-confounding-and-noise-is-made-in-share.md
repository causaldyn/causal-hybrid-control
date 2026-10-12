# ADR 0089 — A move under confounding and noise is made in share

**Status:** accepted, 2026-10-12. Experimental.

## Context

A reallocation's value is linear in the channel, so its sign decides it. Two things leave that
sign unknown: confounding up to a level `Γ`, which puts the value anywhere in
`[β − k, β + k]` with `k = (Γ − 1)/(Γ + 1) · G`, and the estimate's noise, `β̂ ~ N(β, σ²)`.

The library priced each alone. The flip `Γ = (G + |d̂|)/(G − |d̂|)` (`docs/concepts/gamma.md`, "Where
a move flips") and `minimax_action` take the estimate as the value's centre and ignore `σ`. The fit's
confidence interval takes `σ` and ignores `Γ`. Nothing answered both at once, and the answer to both
at once is not either answer: past a width, every rule that minimises the worst regret over both
randomises.

## Decision

`chc.regret.minimax_regret_move(estimate, standard_error, gamma, cvar_gap=1.0, size=1.0)` returns
the minimax-regret rule of Montiel Olea, Qiu and Stoye (*Decision Theory for Treatment Choice
Problems with Partial Identification*, Review of Economic Studies 2026; arXiv 2312.17623 v3,
Section 4.2, equation (4.3), and Theorems 3 and 4, read at the source):

- `k ≤ √(π/2) σ`: the sign rule, the whole move where the estimate is not negative. It is
  essentially the only minimax-regret rule.
- Past it: the minimax regret is `k / 2`, every minimax-regret rule randomises, and the rule that
  randomises least makes `clip((β̂ + ρ)/(2ρ), 0, 1)` of the move, `ρ = k (1 − 2Φ(−ρ/σ))`.

The result carries the share, the rule, `ρ`, `k`, the rule's worst regret, the sign rule's worst
regret beside it, and `full_move_gamma`: the largest `Γ` at which the estimate still makes the
whole move, `k = β̂ / erf(β̂ / (σ√2))` inverted through `k(Γ)`.

- **`k` follows ADR 0085.** The gap is a share of `size`, the channel's size on the move's scale;
  `size = 1` reads `cvar_gap` in the move's own units. The rule then reads the same in any units of
  the value: share, rule and `full_move_gamma` to 1e-12, regrets and `ρ` scaled with the value.
- **A share is a probability, or a fraction.** Where the value is linear in the move's size, making
  a fraction `d` of it pays what making it with probability `d` pays in expectation.
- **`ρ` is found as the root of `t / erf(t/√2) = k/σ`,** which rises from `√(π/2)` at `t = 0`. The
  form `ρ − k erf(ρ/(σ√2))` cancels two near-equal terms near the threshold. The regime test reads
  the same constant, so a half-width one unit in the last place past the threshold still finds a
  root (share 1/2 at an estimate of 0).
- **Linear welfare only.** The rule takes a value, not a cost. `minimax_action`'s quadratic cost
  has another minimax rule, and this one is not offered for it.

## Consequences

- Maxima (`validation/minimax_regret_move.mac`) and the tests agree: `ρ` 1.87970281712 at `k = 2`
  and 4.99999713344 at `k = 5`, `σ = 1`. Over 48 001 values of `β` in `[−12, 12]`, with the rule's
  chance of moving in closed form, the rule's worst regret is `k/2` (1.000 and 2.5) to 1e-9. The sign
  rule's is 1.0509 and 3.3710, 5 % and 35 % more. A Monte Carlo of 200 000 estimates at each of 13
  values of `β` agrees within four standard errors.
- At the threshold both rules meet: the sign rule's worst regret is `k/2` there.
- **Noise moves the whole move's `Γ` up, not down.** The threshold `ρ` is below `k` wherever
  `σ > 0`, so an estimate stays above it past the noiseless flip. With `β̂ = 1.5`, `σ = 1` and
  `G = 4`, the move is whole up to `Γ = 2.526`, where the flip is 2.2. The noise costs a regret that
  hedging the confounding cannot buy back; below `√(π/2) σ` the sign rule moves whole on any
  estimate that is not negative, though the identified set then holds values below zero. As
  `σ → 0` the `Γ` falls to the flip, to 1e-12 at `σ = 1e-3`.
- Past the flip the advice is a share, not a hold: on the flip test's fitted move at twice its
  flip, the share is between 1/2 and 1. `minimax_action`'s minimax cost holds there.

## Alternatives

- **The flip alone.** It reads the estimate as known, and so says "hold" past the flip where the
  minimax-regret rule still moves, and "move" at any `Γ` below it however noisy the estimate.
- **A confidence interval widened by `k`.** It answers "is the sign identified at level α", a test,
  not a decision; its worst regret is not minimised.
- **Any minimax-regret rule.** Past the threshold there are infinitely many. The least-randomising
  one changes the advice least often from a whole move or a hold (Theorem 4).
- **A `criterion` of `minimax_action`.** Its arguments are a target and an effort, a quadratic
  cost's. The linear rule takes neither, so one signature would carry arguments each rule ignores.
