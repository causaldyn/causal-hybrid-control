# ADR 0172 — A plan's pessimism may be moved by the losses it realises

**Status:** accepted, 2026-11-25. Experimental (`ConformalPessimism`, `ConformalHorizon`, in
`chc.gate`). Beside ADR 0094 (the deployed loop's conformal radius is watched and adapted). No
default changes: `RecedingHorizon` is untouched, and a loop without the new classes plans as
before.

## Context

A plan's conservatism is a number the caller sets: `lam_unc`, the weight of the uncertainty
penalty in `causal_plan` and `RecedingHorizon`. Nothing in the library moved it after deployment.
`OnlineRadius` (ADR 0094) moves a radius so that a miss rate keeps near its target on any
sequence, but the radius is what the plan is scored against, not what it is planned with.

Lekeufack, Angelopoulos, Bajcsy, Jordan and Malik ("Conformal Decision Theory: Safe Autonomous
Decisions from Imperfect Predictions", ICRA 2024, arXiv:2310.05921) move the conservatism itself.
Their conformal controller, update (5), is `lambda_(t+1) = lambda_t + eta (eps - l_t)`, with
`l_t` in `[0, 1]` the loss the decision at `lambda_t` realised and a smaller `lambda` the more
conservative decision. Their Definition 1, eventual safety, asks for a `lambda_safe` such that on
any `K` consecutive decisions all at `lambda <= lambda_safe` the losses average at most
`eps_safe <= eps`. Their Theorem 1 then bounds the mean loss after `t >= K` decisions by
`eps + ((lambda_1 - lambda_safe) / eta + K) / t`, for `lambda_1 >= lambda_safe - eta`, on any
sequence of losses. The bound is one-sided, the weight is not clipped, and monotonicity of the
loss in `lambda` is assumed only in their batch setting, not in the online theorem.

`validation/conformal_decision.mac` checks the argument in the weight `w = -lambda`, which grows
with conservatism as `lam_unc` does:

- **STEP 1.** `w_(T+1) - w_1 = eta sum (l_t - eps)`, so the mean loss less `eps` is
  `(w_(T+1) - w_1) / (eta T)`, for symbolic losses at `T = 1..8`.
- **STEP 2.** With `cap = safe + K eta (B - eps)` and losses in `[0, B]`, `(cap - w_1) / (eta T)
  = ((safe - w_1) / eta + K (B - eps)) / T`. Their constant at `B = 1` exceeds it by `K eps / T`:
  their proof bounds a step's largest rise, `eta (B - eps)`, by `eta`.
- **STEP 3.** The one-step induction at `K = 1`: at or above `safe` a loss of at most `eps` does
  not raise the weight, and below it `cap - w_(t+1) = (safe - w_t) + eta (B - l_t) > 0`.
- **STEP 4.** Every adversary in exact rationals, losses in `{0, eps, B}`, `eta = 1/2`,
  `eps = 1/8`, `B = 1`, `safe = 1`, a floor at 0: over 10 decisions at `K = 1` and 8 at `K = 2`
  no weight passes the cap, and the worst play takes 22/23 and 14/15 of the bound.
- **STEP 5.** The adversary that plays `B` below `safe` and `eps` at or above it attains the bound
  up to `(safe - w_n) / (eta T)`, `w_n` the last weight below `safe`: as small as `safe` is close
  to `w_n`.
- **STEP 6.** Two-sided, as `OnlineRadius`'s bound is: where also `l >= eps` at every weight at
  most `unsafe`, the weight stays in `[unsafe - eta eps, safe + eta (B - eps)]`, of length
  `safe - unsafe + eta B`, and `|mean - eps| <= (safe - unsafe + eta B) / (eta T)`. Every such
  adversary over 10 decisions takes at most 5/8 of it.
- **STEP 7.** A loss that rises again with the weight, `B` below 1, 0 on `[1, 2)`, `B` from 2,
  at a claimed `safe = 3/2` and `eta = 2`: the weight jumps the band, reaches the cap after 7
  decisions and stays, and after 20 the mean less `eps` is 27/40 against a bound of 13/160.

It passes in Maxima 5.50 and in the CI image's 5.49.

## Decision

- **`ConformalPessimism(initial, *, target, step, bound, safe, lower=0.0, window=1)`**. Each
  decision is taken at `weight` and closes with `update(loss)`, a loss in `[0, bound]`, which
  moves the weight by `step (loss - target)` and holds it to `[lower, cap]`,
  `cap = safe + window step (bound - target)`. It reports the `path` of weights issued,
  `mean_loss`, `excess_bound = (cap - initial) / (step n)`, and how many updates the cap cut
  (`capped`) and the floor lifted (`floored`).
- **The bound needs no assumption while the cap has cut nothing.** By STEP 1, the weight is
  `initial` plus `step` times the losses' excess, plus the floor's lifts, less the cap's cuts. The
  floor lifts only after losses below the target. So with no cut the mean's excess is at most
  `(cap - initial) / (step n)`, whatever the losses were. Eventual safety at `safe` and `window`
  is what keeps the cap from cutting: their Lemma 1.1, in this sign, with `lower <= safe` and
  `initial <= safe + step (bound - target)`, their condition on `lambda_1`.
- **A cut refutes the assumption.** By the same lemma, under eventual safety the unclipped weight
  never passes the cap, so a cut says the losses were not safe at `safe` and `window`. The first
  cut logs a warning (`chc_event = "conformal_cap"`); the weight is still held to the cap, so a
  failed assumption cannot grow the pessimism without end. A non-monotone loss is documented and
  shown, not refused: no weight can test it in advance, and the cap is where it shows.
- **`ConformalHorizon(controller, pessimism)`**, the opt-in hook. `step` sets the controller's
  `lam_unc` to the weight and plans; `update` scores the plan's loss. They must alternate, so
  that each loss is the one its plan's weight realised. It refuses a controller with no
  uncertainty model, whose `lam_unc` weighs nothing, and a negative floor.
- **The weight goes in as a numpy `float64`.** `RecedingHorizon.lam_unc` is a public field of a
  mutable dataclass that each `step` reads, so the hook needs no change to `RecedingHorizon`. The
  penalised descent is compiled by `equinox.filter_jit`, which makes a Python float a constant of
  the program: each new weight compiled the descent again. A numpy scalar is a `float` and an
  input of the program, so after its first two steps a moving weight compiles nothing.
- **`lam_unc`, not the sensitivity `Gamma`.** `Gamma` reaches the plan through the radius of a
  `ConfoundingRobustPenalty` (`from_sensitivity`) or a barrier's `ChannelSensitivity`, a static
  field of a frozen object: a weight on it would need a new penalty object, and a new program, at
  every plan. The confounding penalty is `lam_unc * radius * sum ||u_t||`, so on it moving
  `lam_unc` moves the radius by the same factor. A weight on `Gamma` is not built.

## Consequences

`tests/test_conformal_decision.py`:

- The update is the published one, held to the floor and the cap, on a sequence that meets both.
- Every eventually safe adversary of STEP 4, replayed through the class in floating point, cuts
  nothing and takes 22/23 and 14/15 of the bound, Maxima's values to `1e-12`; the greedy one
  attains STEP 5's values exactly. The parameters are dyadic so that the replay takes the
  rationals' branches.
- Random losses whose law falls with the weight and meets the target from `safe` on, three seeds
  of 3000 decisions at three floors: the bound holds at every decision, the cap is never cut,
  while the floor lifted nothing the mean stays above `target - (initial - lower) / (step n)`,
  and it settles within 0.01 of the target.
- Losses of mean 1/2 whatever the weight reach the cap, log one warning, and pass the bound;
  STEP 7's loss gives Maxima's path, 14 cuts in 20 and the excess 27/40.
- A closed loop of 120 decisions on a one-state plant whose gain drifts from the model's at 0.05
  per unit of time, the loss a one-step prediction miss beyond 0.01: at a fixed `lam_unc = 0` the
  prediction missed on 70 of them, and on every one of the last 60; with `ConformalHorizon` at a
  target of 0.2, `step = 0.5` and `safe = 2`, it missed on 16, within the bound of 0.24, with no
  cut. The test asserts more than half and at most 0.24, not the counts.
- A moving weight compiles nothing after the first two steps; the same controller at a new
  Python float compiles. Its first plan is bitwise the plan a Python float gives.
- **Limits.** The bound is on the running mean of the sequence scored, not on any one decision.
  `safe` is the caller's statement; a cap never reached does not show it held, only that the bound
  did. On a plant with memory a loss reads earlier decisions too, and `window` must cover how many
  decisions at `safe` it takes to bring the loss to the target. The weight is one scalar for the
  whole plan, and the class does not tune `step`.

## Alternatives

- **Their update without a cap.** The bound is the same, but a loss with no safe weight grows the
  weight without end, and nothing says so. The cap is placed where eventual safety never reaches.
- **A cap at `safe`.** It cuts under the assumption itself: a weight just below `safe` after a
  loss of `B` passes it, and the cut discards excess loss the bound counts on.
- **An argument on `RecedingHorizon`** (`pessimism=`). A new field on a stable class, and a stable
  module reading an experimental one; a wrapper changes nothing there.
- **Their batch setting** (Corollary 2, conformal risk control over exchangeable decision
  functions). It needs exchangeability, which a closed loop breaks, and a monotone loss.
- **Moving `step` with time, as `OnlineRadius` does** (`eta_n = step n^(-decay)`). Their theorem
  is for a constant step; a decaying one slows the weight's response to a drift, which is what the
  controller is for.
