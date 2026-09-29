# ADR 0021 — Watching a plan whose actions clip

**Status:** proposed, 2026-09-29. Amends ADR 0015 (`DecisionLog.dither` is the draw) and ADR 0018
(a clipped decision is read, not refused).

## Context

The channel-drift monitor of ADR 0018 refused a log with a clipped decision. A plan with a box on
its actions clips a dithered action whenever it runs near a bound, and a constrained plan's optimum
often sits on one. ADR 0018 recorded a candidate fix, skipping before the draw the decisions whose
nominal action lies near a bound, and did not build it.

Three ways to read a clipped decision were checked (`validation/dither_drift_evalue.mac` STEP 8):

- **Skip it after the draw, with an e-value of 1.** Not an e-value. On the radius's edge, with a
  lower clip, its mean is `1 + Phi(−m1) − Phi(−m1 − theta c)`, which tends to `1 + P(clip)`: 1.48
  with the nominal action on the bound and `theta c = 2`.
- **Set its e-values to 0.** Valid, because an e-value is non-negative, but each zero restarts
  every detector.
- **Read it with its draw:** the bet reads the draw `xi`, and the residual moves with the action as
  applied.

## Decision

- **A clipped decision is read with its draw.** With `h = clip(xi, −m1, m2)` the draw as the box
  cut it, in dither units, the residual at the action as applied is `r = c + k h`. The e-value's
  mean given `c` is `1 − (Phi(u2) − Phi(u1)) (1 − 1/s)`, with `s = 1 − theta k`,
  `u1 = −s m1 − theta c` and `u2 = s m2 − theta c` (STEP 8a–e). Since `u2 − u1 = s (m1 + m2) > 0`,
  it is:
  - at most 1 where `theta k <= 0`, which is every side an entry has not crossed (ADR 0018's
    STEP 4);
  - exactly 1 on the radius's edge, whatever the clip;
  - at least 1 on the side an entry has crossed, so a clip shrinks the power and never turns it
    against the alarm;
  - `1/s` without a clip, which is ADR 0018's identity.
- **The bound is one change of variables, and reaches past a box.** Pointwise,
  `e phi(xi) = phi(xi − theta r)`, so the mean is `∫ phi(g(xi) − theta c) dxi` with
  `g = xi − theta k h(xi)`. Where `theta k <= 0` and `h` is nondecreasing, `g' >= 1` and `g` is
  onto, so the mean is at most `∫ phi(v − theta c) dv = 1`. That covers an action applied as any
  nondecreasing function of its own draw: a box, a rate limit fixed before the draw, a saturating
  actuator (STEP 8h checks a `tanh`). It does not cover a projection that couples the actions,
  such as a budget shared across them, where one action's residual moves with another's draw.
- **`DecisionLog.dither` is the draw, before any clip.** This amends ADR 0015, which said "as
  applied". On an unclipped decision the two are the same number. On a clipped one only the draw
  makes the reading an e-value: the dither as applied, put in its place, has a mean of up to 1.151
  on the edge with the nominal action on a bound (STEP 8f). Version 1 of the record has not been
  released, so no stored log carries the earlier meaning, and the version stays 1.
- **`channel_drift_evalues` no longer refuses a clipped decision, and does not read
  `saturated`.** The flag stays for the readers that need it. The chi-square check now tests every
  draw, clipped or not, against exactly the stated law.
- **`DecisionLog.dither_draws` still refuses a clipped decision, and `channel_move` with it.** The
  product of the residual and the draw reads the move times the chance that the draw was not
  clipped, by Stein's lemma `E[h(xi) xi] = P(−m1 < xi < m2)`, so a clipped log would read the move
  short.
- **No margin rule.** Dithering only the decisions whose nominal action lies well inside the box is
  not needed for validity, and is not built.
- **Experimental,** like the monitor.

## Consequences

Measured by `scripts/bench_drift.py`, 300 paths, on the lab's plant of ADR 0018 with every action
clipped to `[−0.5, 0.5]` and every decision dithered. The box clipped 0.24 of the decisions, at the
channel 1.07 and at 1.4. Run lengths are censored at 80 000 decisions, so each mean is a lower
bound. No number here is a timing.

| | `A = 10³` | `A = 10⁴` |
|---|---|---|
| unchanged channel, boxed: run length / `A` | ≥ 2.84 | ≥ 4.61 (29% censored) |
| the same, unboxed (ADR 0018) | ≥ 3.23 | ≥ 5.60 (41% censored) |
| channel at 1.4, boxed: mean delay (median) | 438 (340) | 749 (615) |
| the same, unboxed (ADR 0018) | 259 (237) | 456 (411) |
| boxed, alarmed within 3000 decisions | 1.000 | 0.997 |
| boxed, clipped decisions' e-values set to 0: alarmed within 3000 | 0.000 | 0.000 |

- **The guarantee holds on the boxed plant.** Its run length is shorter than the unboxed plant's,
  because inside the radius a clip moves a column's mean from `1/s` toward 1: the reading is less
  conservative there, and still valid.
- **A box costs detection delay.** The move was caught in 1.69 and 1.64 times the unboxed delay.
  A clipped draw moves the action less, so it says less about the channel.
- **Setting the clipped decisions' e-values to 0 never alarmed.** With a quarter of the decisions
  clipped, every detector restarts about every fourth decision, and in 3000 decisions the alarm
  sounded on none of 300 paths at either target.
- **Nothing changed for an unclipped log.** The unboxed rows reproduce ADR 0018's to the digit.
- **Checked:**
  - `validation/dither_drift_evalue.mac` STEP 8: the residual's split, the three regions'
    integrals, the closed form and its limits, the dither as applied, skipping after the draw, and
    quadrature against the closed form and on a `tanh` actuator.
  - `proofs/dither_drift_evalue.v` (E): the region identities and the bounds on each side
    (9 lemmas, 26 in the file, on Stdlib's classical reals and nothing else).
  - `tests/test_gate.py`:
    - each column's mean on a boxed log against the closed form, by Monte Carlo, with the nominal
      action inside the box, on a bound and beyond one;
    - that the flag is not read;
    - the lab's plant, boxed, alarming before a move on no more than `H/A` of the paths and after
      it on at least 0.95.

## Alternatives considered

- **Skipping a clipped decision after the draw.** Rejected: it is not an e-value (Context).
- **Setting a clipped decision's e-values to 0.** Rejected: valid, but with a quarter of the
  decisions clipped it caught the move on none of 300 paths within 3000 decisions, where reading
  the draw caught it on 1.000 and 0.997 of them.
- **Dithering only well inside the box, decided before the draw.** Not built. It is valid, but the
  log would have to mark the decisions it did not dither, and an undithered decision tells the
  alarm nothing. Whether it would shorten the boxed delay was not measured.
- **Recording what the clip left of the draw, as ADR 0015 had it.** Rejected: no reader can use
  it. The monitor needs the draw, and the action as applied is logged already.
- **Correcting `channel_move` for the clip.** Not built: dividing by the chance that the draw was
  not clipped needs the nominal action and the box, and the log records neither.
