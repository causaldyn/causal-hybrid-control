# ADR 0094 — The deployed loop's conformal radius is watched and adapted

**Status:** proposed, 2026-10-13.

## Context

`SplitConformal` calibrates next-state intervals on held-out logs. They cover while the logs and
the loop are exchangeable, so a run of steps outside them says the model no longer describes the
plant. Nothing in the library watched for that run. `DriftAlarm` watches the channel through the
logged dither, which says nothing of the drift or the noise.

Two facts shape a monitor. First, the plant's noise is persistent: `prescribe`'s tube carries it
as AR(1) at `tube_persistence` (ADR 0080). With AR(1) noise, a step outside the interval makes
the next one likelier, so the misses are not independent, and a sequential test that treats them
as independent rejects a correct model too often. Second, a deployed plan steers its own states,
so the scores a radius meets are not exchangeable with the calibration's either. A guarantee that
holds on any sequence survives that feedback.

## Decision

- **`MissRateAlarm(calibration, *, persistence, alpha, delta)`** (*experimental*, in `chc.gate`
  beside `DriftAlarm`). It whitens consecutive held-out residuals, `w_t = r_t - persistence
  r_(t-1)`, scores each innovation by its norm, and takes split conformal's radius, the
  `k`-th smallest score with `k = ceil((n + 1)(1 - alpha))`. Given the calibration, the
  deployed innovations miss it at a fixed rate `p`, and over calibrations `p` is
  `Beta(n + 1 - k, k)` (Vovk 2012; Lin and Bansal, L4DC 2024). `bound` is that law's upper
  `delta / 2` quantile.
- **The test** is Podkopaev and Ramdas's (ICLR 2022, Proposition 1), with `delta_S = delta_T =
  delta / 2`, no tolerance, and the source risk's upper bound read off the exact Beta law rather
  than a concentration bound. The target side is a mixture, over a miss rate `q` uniform on
  `[bound, 1]`, of the likelihood ratios of `q` against `bound`. It has a closed form through the
  incomplete beta function, and it is a nonnegative supermartingale while each innovation misses
  with probability at most `bound` given the past. By Ville's inequality it ever reaches
  `2 / delta` with probability at most `delta / 2`; the alarm sounds there.
- **`persistence` is required**, with no default: 0 is a decision that the noise is not
  persistent, not an omission. The certificate's `tube_persistence` or a fit's lag-1 correlation
  supplies it.
- **`OnlineRadius(initial, *, alpha, bound, step, decay=0.6)`** (*experimental*, in `chc.gate`)
  keeps a radius per horizon and moves it by `eta_n (miss - alpha)`, `eta_n = step n^(-decay)`,
  after the `n`-th scored prediction of that horizon (Angelopoulos, Barber and Bates, ICML 2024,
  update (4)). A prediction is scored against the radius it was issued with, which the class
  remembers. Their Theorem 1 bounds the long-run miss rate within `(bound + eta_1) / (eta_n n)`
  of `alpha` on any sequence of scores in `[0, bound]`. A horizon `h` prediction waits `h - 1`
  updates for its score; the same induction as their Lemma 1 then confines the radius to
  `[-alpha h eta_1, bound + (1 - alpha) h eta_1]`, and their telescoping gives
  `(bound + h eta_1) / (eta_n n)`, which `miss_bound` reports.
- A score outside `[0, bound]` is refused: the theorem needs it, and a clipped score would count a
  miss the issued radius did not make.
- **Additive.** No existing default or result changes.

## Consequences

Measured by `tests/test_gate_conformal.py` and `validation/miss_rate_alarm.mac`.

- **The Beta law.** Maxima checks, for every `1 <= k <= n <= 8`, that `P(p > e)` is the binomial
  tail `sum_(i=0)^(n-k) C(n, i) e^i (1 - e)^(n - i)`, that the two Beta laws mirror each other,
  and that `E[p] = (n + 1 - k)/(n + 1) <= alpha`; at three sample points its bisection in bfloat
  matches scipy's `betaincinv` within `3e-16`. Over 4000 simulated calibrations at `n = 99`,
  `delta = 0.2`, the miss rate exceeded `bound` on `delta / 2` of them within four standard
  errors.
- **The mixture.** Maxima checks its closed form for `0 <= S, F <= 4`, and that its mean after
  `t <= 6` steps is exactly 1 at a miss rate of `bound` and below 1 under it; the test checks
  the same against quadrature and the binomial law at `t = 9`.
- **A correct plant with AR(1) noise at 0.8**, 300 calibration innovations, `alpha = delta = 0.1`,
  1000 paths of 1000 steps: the alarm sounded on 0.016 of the paths at the noise's persistence,
  0.020 at the lag-1 correlation fitted on each calibration log, and 0.209 unwhitened. A noise
  1.5 times the calibration's was caught on all 200 paths.
- **`OnlineRadius` against every adversary.** Whether a score misses depends only on whether it
  exceeds its issued radius, so the scores 0 and `bound` reach every sequence of misses an
  adversary can force. Over all `2^12` sequences on three horizons, at four step schedules, every
  horizon stayed within its bound at every period. At a constant step of 1, horizon 3 came within
  0.9 of `(bound + 3 eta_1) / (eta_1 n)`, past the `(bound + eta_1) / (eta_1 n)` of a horizon
  scored at once: the factor `h` is needed. Long uniform, constant, switching and heavy-tailed
  sequences, and an adversary chasing each issued radius, kept it too.
- **Limits.** The alarm's guarantee holds for the persistence the noise has; one fitted on the
  calibration log leaves a correlation of the order of its error. `OnlineRadius` bounds the
  long-run miss rate on the sequence scored, not the coverage at any one period, and not across
  horizons jointly.

## Alternatives considered

- **No whitening, an average-run-length alarm on raw misses.** Clustered misses break the
  supermartingale; the measured false-alarm rate was twice `delta`.
- **A conformal test martingale** (Vovk's exchangeability martingales). It tests exchangeability
  itself, which a closed loop breaks by design, and alarms on harmless shifts; the miss rate is
  what the tube's guarantee rests on.
- **A Hoeffding or empirical-Bernstein bound on the source side**, as Podkopaev and Ramdas use. The
  conformal miss rate's law is known exactly, so the exact quantile is tighter at no cost.
- **A grid of constant bets** in place of the uniform mixture. Equally valid; the mixture has a
  closed form that Maxima checks, and no grid to choose.
- **Running `h` interleaved copies of the one-step update per horizon**, which keeps their
  constant exactly. Each copy learns from `1/h` of the scores; the delayed update keeps one radius
  per horizon at the price of the factor `h`.
- **Oracles from third-party conformal packages.** Not used: every number here is checked against
  closed forms, enumeration, Monte Carlo and Maxima.
