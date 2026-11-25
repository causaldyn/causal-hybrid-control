# ADR 0171 — A closed loop's exit probability is read apart from the barrier's audit

**Status:** accepted, 2026-11-25. Experimental (`chc.barrier`: `exit_probability`,
`ExitProbability`, `QuadraticBarrier`, `docs/api/index.md`). New: no default moves, and no value
another call returns moves.

## Context

The safety layer has three operations, and each reads a deterministic condition. `causal_plan`
holds `grad h . xdot >= -alpha h` in the solve, `certify_safety` prices it along the model's
predicted path, and `robust_safety_filter` clips one action into it at one state. None of them
says how likely the deployed loop is to leave the safe set under the plant's own noise. A filter
that lets a nominal controller push to the edge holds its loop's rest point on the boundary, and
any noise then takes the loop out with a probability that grows with the horizon.

Three results bound that probability for a discrete-time loop `x' = F(x, k(x)) + w`:

- **Ville's inequality on Kushner's supermartingale** (Cosner, Culbertson, Taylor and Ames, RSS
  2023, Thm 5). Premises: `h <= M` on the whole state space, `M > 0`; `alpha` in `(0, 1)` and
  `delta <= M (1 - alpha)` with `E[h(x') | x] >= alpha h(x) + delta` at every state; the noise
  independent and identically distributed. Then `P(min_{k <= K} h(x_k) < -gamma)` for `gamma >= 0`
  is at most, with `phi = M (1 - alpha) - delta`,
  `(M - h0)/(M + gamma) alpha^K + phi/(M + gamma) sum_{i=1}^K alpha^(i-1)` where
  `delta < -gamma (1 - alpha)`, and `1 - (h0 + gamma)/(M + gamma) ((M alpha + gamma + delta)/(M +
  gamma))^K` where `delta >= -gamma (1 - alpha)`. At `delta = gamma = 0` the second form is
  `1 - alpha^K h0/M`.
- **Freedman's inequality** (Cosner, Culbertson and Ames, L-CSS 2024, Thm 3). It needs no upper
  bound on `h`. It needs a bound that holds almost surely on each step's drop below its
  expectation, `E[h(x_k) | x_{k-1}] - h(x_k) <= delta`, and one on the conditional variance.
- **A conformal model-error radius** inside the robust condition (Hsu and Tsukamoto, L-CSS 2025).
  It bounds the chance that the model's error leaves its radius, not the chance that the noise
  takes the loop out.

The library's plants are Gaussian (`LinearGaussianPlant`). A Gaussian step has no almost-sure
bound on its drop, so Freedman's premise fails on every one of them. Ville's form needs two things
the caller cannot be trusted to supply: the bound `M`, and the mean of the next state's barrier at
every state. A quadratic barrier `h = M - (x - c)'P(x - c)` with `P >= 0` has both:

- `h <= M` everywhere, by construction;
- `E[h(m + w)] = h(m) - tr(P S)` for any noise of mean 0 and covariance `S`, whatever its law
  (`validation/exit_probability.mac`, STEP 6). So the drift condition at a state needs the loop's
  mean step there and the covariance, and no sampling.

An affine barrier has the same exact mean, since `E[h(m + w)] = h(m)`, but no `M`. Read with a
wrong one, the bound claims what is false: `h = 1 - x` on `x' = 0.6 x + N(0, 0.45^2)` meets the
condition with `alpha = 0.6` and `delta = 0.4` at every state, and with `M = h(x0) = 1` the second
form reads 0, while one step leaves `x <= 1` with probability 0.0131 (STEP 8).

On an affine loop `x' = F x + f + w` the margin `E[h(x') | x] - alpha h(x)` is a quadratic in `x`:
with `y = x - c` and `e = F c + f - c`, it is `r0 - 2 q'y + y'Q y`, where `Q = alpha P - F'P F`,
`q = F'P e` and `r0 = (1 - alpha) M - tr(P S) - e'P e`. It is bounded below iff `Q >= 0` and `q`
lies in `Q`'s range, and then its least value is `r0 - q'Q^+ q` (STEP 6). Its least value over
the whole state space is the theorem's premise, read exactly. A filter's loop is not affine: the
margin can be read at given states, and nothing between them.

## Decision

- **`QuadraticBarrier(upper, centre, shape)`**, `h(x) = upper - (x - centre)' shape (x -
  centre)`, refusing a `shape` that is not symmetric positive semidefinite (no upper bound) and an
  `upper` that is not positive. A positive definite `shape` gives an ellipsoid; a lower rank, a
  slab or a cylinder. A half-space is read through a slab inside it: the loop leaves the slab
  whenever it leaves the half-space, so the slab's bound covers the half-space too.
- **`exit_probability(barrier, plant, policy, x0, *, steps, alpha, level=0.0, states=None)`**
  returns an `ExitProbability`: `bound`, `steps`, `alpha`, `slack`, `level`, `initial` (`h(x0)`),
  `upper`, `branch` and `scope`.
  - Under an `AffinePolicy` the loop is `F = a + b K`, `f = b k + offset`, with the noise
    `plant.noise + b Sigma b'`. `slack` is `r0 - q'Q^+ q`, over every state
    (`scope="everywhere"`). Where `Q` has a negative direction, or a flat one `q` does not
    vanish on, the margin falls without bound, no slack holds, and the call raises, naming the
    direction and the eigenvalue.
  - Under a callable `u = policy(x)` the margin is read at `x0` and at each row of `states`, and
    `slack` is the least of them (`scope="states"`). `states` is required with a callable and
    refused with an affine policy.
  - `slack` is capped at `upper (1 - alpha)`, as the theorem asks. The bound is Thm 5's form for
    the branch: `"inward"` where `slack >= -level (1 - alpha)`, `"outward"` below. It is clipped
    at 1, so a start outside `{h >= -level}` reads 1.
- **A closed loop only.** A policy that is not a feedback law raises `TypeError`, naming the open
  loop: an `AffineSchedule` (a prescription's schedule is one), a `CausalPlan`, an array of
  actions. An `AffinePolicy` whose gain is zero raises `ValueError`. Theorem 5 would hold on such
  a loop, since a constant action is a trivial feedback; the refusal keeps the reading from ever
  being taken for a plan's.
- **Apart from the audit.** `certify_safety` neither reads nor returns the exit probability: no
  argument of it and no field of `SafetyCertificate` names it, and its output is the same before
  and after a call.
- `alpha` is the discrete-time factor of `E[h(x') | x] >= alpha h(x) + slack`, as the papers
  write it, not `certify_safety`'s continuous-time gain, which is a rate. The docstring says so.
- `exit_probability` and `QuadraticBarrier` are filed in the lifecycle's fifth stage, beside the
  filter, so the top level holds them; `ExitProbability`, which a call returns, stays at its
  module path. The three names are experimental entries of the stable `chc.barrier`.

## Consequences

The checks are in `tests/test_exit_probability.py` and `validation/exit_probability.mac`, which
derives both forms from the supermartingale (STEPs 1 to 4), checks at five sample points that no
other choice of the free parameter does better (STEP 5), and runs in the CI's Maxima 5.49.

- **Against the exact exit probability.** On `x' = 0.9 x + 0.5 u + w` under `u = -0.6 x`, `w ~
  N(0, 0.45^2)`, `h = 1 - x^2` and `x0 = 0`, at `alpha = 0.36` the margin is flat and the slack
  0.4375 at every state. The exit probability, by quadrature on the survival recursion (Simpson in
  Maxima, Gauss-Legendre in the test, to `1e-7`), and the bound:

  | K | exact | bound | bound / exact |
  |---|---|---|---|
  | 1 | 0.0263 | 0.2025 | 7.7 |
  | 2 | 0.0764 | 0.3640 | 4.8 |
  | 3 | 0.1280 | 0.4928 | 3.9 |
  | 5 | 0.2239 | 0.6774 | 3.0 |
  | 10 | 0.4202 | 0.8959 | 2.1 |

  From the centre, at one step, the bound is `tr(P S)/M`, Chebyshev's reading of the noise: the
  bound reads the noise through its covariance alone, and the Gaussian tail is far thinner. It is free of the
  dimension and loose, most of all at short horizons.
- **Against Monte Carlo**, 20 000 paths a loop, the seeds the tests use, exit frequency against
  the bound:
  - the scalar loop above at `alpha = 0.36`: 0.027, 0.225 and 0.416 at `K` = 1, 5 and 10, under
    the table's bounds;
  - the same loop from `x0 = 0.5` at `alpha = 0.8`, where the slack is -0.0025 and the branch
    outward: 0.062 against 0.403, 0.166 against 0.622 and 0.300 against 0.813 at `K` = 1, 3 and 6;
  - a double integrator, `dt = 0.1`, under `u = -4 x1 - 3 x2 + nu`, `nu ~ N(0, 25)`, on the
    ellipse of the loop's Lyapunov form centred at `(0.1, 0)`, at `alpha = 0.95`: the slack is
    `-2357033/2420101800` in exact rationals, outward, and the frequency 0.028 against 0.433 at
    `K = 10` and 0.073 against 0.668 at `K = 20`;
  - `robust_safety_filter` holding `x' = x + 0.1 (0.3 x + u) + N(0, 0.01)` at 1.6 against a
    nominal driven to 3, in the corridor `[-1, 2]`, read at 301 states across it at
    `alpha = 0.95`: 0.009 against 0.615 at `K = 20` and 0.139 against 0.833 at `K = 40`;
  - leaving `{h >= -0.2}` on the scalar loop at `alpha = 0.5`: 0.310 against 0.842, where leaving
    `{h >= 0}` is bounded by 0.896.
- **`alpha` decides less than it seems.** On a loop whose rest point is the barrier's centre,
  `e = 0`, the slack is `(1 - alpha) M - tr(P S)`, so `M alpha + slack = M - tr(P S)` at every
  `alpha` at which `Q >= 0`: there `alpha` decides whether the call refuses, not the bound. On a
  loop that meets the condition with no slack the bound is `1 - alpha^K h0/M`, and falls as
  `alpha` rises.
- **A callable is read where it is told.** States near the start alone read a larger slack, and a
  smaller bound, than states across the corridor: on the filter's loop the five states within 0.2
  of the start read a bound below the covering grid's. As `chc.reachability`'s verdicts are the
  grid's nodes' (ADR 0161), this bound rests on the states given.
- **Refused:** an unstable loop at every `alpha`; a stable loop read below its contraction
  (`alpha < 0.36` on the scalar one); and a slab on the position of the double integrator at every
  `alpha`, where `det Q = -alpha F_12^2 < 0` (STEP 6): the velocity moves the position's barrier,
  and the margin falls without bound along it.

## Alternatives

- **Freedman's form.** Not built: its premise is an almost-sure bound on each step's drop, and a
  Gaussian step has none. It would serve a plant whose noise has a bounded support, read off that
  support, and a barrier with no upper bound.
- **A Monte Carlo reading of the drift condition for any barrier.** Rejected: an estimated premise
  makes the bound an estimate, with an error the bound does not carry. The quadratic barrier's mean
  is exact.
- **Kushner's form of the premise, on `{h >= -level}` only.** Sound: the supermartingale stopped
  when the loop leaves needs the condition only at the states it occupies before, so it would
  accept loops the premise on every state refuses. It needs the margin's least value over an
  ellipsoid, a trust-region subproblem, in place of a closed form, and the paper states the
  premise on every state. Not built; the callable's `states` may already be taken across the set.
- **Choosing `alpha` inside the call.** Each `alpha` gives a valid bound, so their least over a
  grid is one too. Not built: the caller scans, and on a loop centred on the barrier `alpha`
  moves no bound (Consequences).
- **Attaching it to `certify_safety` or to `RecedingHorizon`.** Rejected: the audit reads a
  deterministic condition along a predicted path, and this reads a feedback law under noise at
  every state the noise reaches.
- **A conformal model-error radius** (Hsu and Tsukamoto). A different question, the model's error
  rather than the noise's exit. Not built.
