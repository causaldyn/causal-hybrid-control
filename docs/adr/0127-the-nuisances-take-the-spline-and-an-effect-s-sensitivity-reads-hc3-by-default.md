# ADR 0127 — The nuisances take the spline, and an effect's sensitivity reads HC3, by default

**Status:** accepted, 2026-10-21. Amends ADR 0106, ADR 0107 and ADR 0120.

## Context

0.19.0 shipped two defaults it said 0.20.0 would change. Each new reading was opt-in, and the old
default warned with a `FutureWarning`.

- **The nuisances' sieve.** `fit_causal_residual` and `prescribe` cross-fitted polynomial
  nuisances of `nuisance_degree` alone (`nuisance_knots=None`). A policy or a drift outside the
  polynomial's span leaves a bias that does not shrink with the log, and `channel_error` does not
  cover it: on the plant of ADR 0107 the channel came back 0.61 below its truth, on none of twelve
  logs within its interval. `nuisance_knots="auto"` adds a natural cubic spline at
  `ceil(N^(1/3))` knots (ADR 0107), and the sieve is chosen out of fold among the spline with the
  covariates' powers, the spline without them and the polynomial alone (ADR 0120). The default fit
  solved its moment once more on `"auto"`'s nuisances and warned where that moved its channel by
  more than three errors of the move.
- **The flip's condition.** It waited on the spline's cost where the polynomial is right. With the
  sieve chosen out of fold, `"auto"` scattered at most 1.01 times as far as the polynomial on every
  such design ADR 0120 measured, and less on heavy tails.
- **The sensitivity's error.** `sensitivity_analysis` read its `std_error` and `e_value_ci` from
  the classical OLS error, whose 95 % interval missed the effect on 22.5 % of logs whose noise grew
  with the action; HC3's missed 5.6 % (ADR 0106).

## Decision

- **`nuisance_knots="auto"` is the default** of `fit_causal_residual` and `prescribe`.
  `RunProvenance.nuisance_knots` records `"auto"`, the field's default too. `None` reads the
  polynomial alone, as up to 0.19.0; an integer, that many knots.
- **The `FutureWarning` is gone; its check stays on the polynomial fit.** At `None` an identified,
  unweighted fit without an instrument still solves its moment on `"auto"`'s nuisances and warns,
  now with a `UserWarning`, where the move passes its `t` limit (ADR 0120): the polynomial misses
  the log's nuisances, and its error does not cover the bias. That is still true of a fit that asks
  for the polynomial. At the default the check would read the fit against itself, so it does not
  run. `nuisance_shift` is unchanged.
- **The instrument's first stage reads the state in the nuisances' basis.** The first stage was
  the polynomial in the state and the instrument. Beside a spline sieve without a splined state's
  powers past its first, the polynomial's powers are outside the nuisances' span, so the first
  stage pushed the actions along directions the nuisances leave in their residual, whatever the
  instrument. An instrument of zeros on 200 rows at `nuisance_degree=3` read relevances 0.15,
  0.10 and 0.055 and an identified channel, set by the functional form alone. The first stage is
  now the nuisances' own sieve in the state and the polynomial's monomials that read the
  instrument. With the polynomial alone it is the polynomial, as before.
- **`sensitivity_analysis(robust=True)` is the default**, and `e_value_ci` reads its interval.
  `robust=False` keeps the classical error. The robustness value keeps the classical `t` either
  way, as ADR 0106 says why.
- **`nuisance_knots` leaves the experimental entries** of the API reference: it sets the default
  fit of a stable function.

## Consequences

- A default fit moves wherever the spline is chosen, and nowhere else: where the polynomial is
  chosen the fit is the polynomial's, bit for bit, less the check.
- Re-measured for the default, 0.19.0 → 0.20.0 (the polynomial's numbers stand for `None`):
  - the plant of ADR 0107 in the misspecification gate (40 logs): covered on none, bias 8.2
    errors, warned on all → covered on 39, bias 0.02 errors;
  - the gate's other four designs (well specified, heteroskedastic, persistent noise, persistent
    noise and lever): the same counts, 38, 38, 37 and 33 of 40;
  - a push `3 tanh(2 z)` on the plant of ADR 0120 (60 logs): the interval covered on 98 % under
    either integrator, its error 1.37 and 1.34 times the scatter → 97 % under either; the spline
    was chosen on 28 of the 60, and under `euler` its error came out 40 to 62 times smaller on
    the four of the first twelve logs that chose it;
  - the quadratic push of ADR 0120 under `rk4` (60 logs): 0.93, the same fits, the polynomial
    chosen on every log; a confounder held in the drift, leaning on the state or not: 0.967 both;
  - one unit of 200 transitions, AR(0.9) noise and an AR(0.7) lever (1000 logs, ADR 0075's
    design on fresh draws): the spline chosen on 238; coverage 0.724 → 0.740 row by row and
    0.941 → 0.938 by cosines;
  - ADR 0053's panels of 5 to 80 units whose noise persists within units (400 a cell,
    `scripts/bench_clustered_error.py`, section `size`): a 5 % test of a zero channel rejected
    12.5 % to 35.5 % row by row and 3.75 % to 14.75 % by unit → 12.25 % to 35 % and 4 % to 14 %;
    the spline was chosen on 1122 of the 11 200 fits;
  - the notebooks' panels: the polynomial was chosen, and every printed number is the same.
- On the media-mix plant of `chc.research.mmm`, over 40 logs fitted by Euler with `season`
  adjusted for, the channel's entries scattered a median 1.07 times as far as the polynomial's
  (0.87 to 1.31), and the fastest adstock row's own-state drift coefficient 0.030 about its
  integrator gap, against 0.021. Whether the polynomial holds that plant's nuisances was not
  measured; ADR 0120's 1.01 is a bound on the designs it names, not on every plant.
- Every default fit cross-fits three sieves. Up to 0.19.0 only an identified, unweighted
  orthogonal fit did so, in its check; instrumented, weighted and observational fits now do too.
- With one fold the sieve's choice reads its own rows, and the spline, a larger basis, is always
  chosen.
- `sensitivity_analysis`'s `std_error` and `e_value_ci` move to HC3's on every default call; the
  effect and the robustness value do not move.

## Alternatives

- **Keep the check under the default, against the polynomial.** Rejected: it would say the
  polynomial misses, of a fit that does not read the polynomial.
- **Drop the check.** Rejected: at `None` it still separates a log the polynomial misses from one
  it holds.
- **A first stage of the whole nuisance basis, the adjustment set's columns too.** Standard 2SLS
  practice, and nested in the nuisances as well. Rejected here: with the polynomial alone it would
  move every instrumented fit with an adjustment set, where the first stage was already nested.
- **Restrict an instrumented fit to the sieves with the powers.** Rejected: it hides the
  mismatch in the sieve's choice instead of removing it from the first stage.
