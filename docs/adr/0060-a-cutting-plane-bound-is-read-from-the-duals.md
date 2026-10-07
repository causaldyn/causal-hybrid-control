# ADR 0060 — A cutting-plane bound is read from the duals, rounded down

**Status:** accepted, 2026-10-07. Amends ADR 0038, ADR 0043 and ADR 0048 (how the bound of the
cutting planes is read), and ADR 0038 and ADR 0048 (the units their programs are written in).

## Context

`minimax_allocate`, `cvar_allocate`, `allocate_geos` and `budget_for_geos` bound their plans by
Kelley's cutting planes, and each read the bound off HiGHS's objective on its last program, which
is not a bound on the program's least:

- **HiGHS rounds its basis's value.** On the 87 programs of a search over 400 readings of six
  channels, its objective stood above the exact value of its basis, worked in PARI/GP, in 47, by
  up to 1.0e-7 of the program's units, and above the program's least, against an exactly feasible
  point, in 10, by up to 9.2e-8. On the least of `x` over `5 x >= 1` it reports 0.2, the double
  above 1/5.
- **HiGHS leaves out small entries.** It drops every entry of at most 1e-9, its
  `small_matrix_value`, from the program it solves (HiGHS #1965). On the least of `x1` over
  `x1 + 2^-33 x2 >= 1`, `x2` up to `2^20`, it reports 1, where the least is 8191/8192.
  `minimax_allocate` and `cvar_allocate` wrote a slope as a return a currency unit over the
  largest return, about one over the budget, so from a budget of 1e9 every slope was left out. On
  two linear readings returning 2 and 1 a unit on two channels, and 1 and 2, at a budget of 1e10,
  `minimax_allocate` closed on the split all on the first channel with a bound of half the best
  return, its worst regret, where the even split's is a quarter. At the level 1, on readings
  returning 2 and 1, and 1 and 1.5, against all on the second channel, `cvar_allocate` closed on
  that reference with a bound of 0, where all on the first gains a quarter of the budget.

The searches stop, and `cvar_allocate` prunes its boxes, on these numbers.

## Decision

- **The bound is read from HiGHS's duals on the program as written** (Neumaier and Shcherbina
  2004; Jansson 2004). For any row duals `y`, `c @ x = y @ (A @ x) + d @ x` with
  `d = c - A.T @ y`, so the least is at least each dual times the side its sign presses on, summed
  with each reduced cost times the end of its column's box it presses on. A dual that is not
  finite, or presses on an infinite side, is read as nothing; a reduced cost that presses on an
  infinite end gives `-inf`. The bound holds whatever the duals are, so HiGHS's errors in them
  only loosen it, and so do the entries it left out, which are in the matrix the bound is read on.
- **Free and one-sided columns are read in the box their planes imply.** `cvar_allocate`'s eta,
  `minimax_allocate`'s worst regret and `allocate_geos`' heights lie, at some least point, between
  the least and the most any plane reaches over the rates' box, and each of `cvar_allocate`'s
  excesses below their difference; these are read rounded outward. In the program's own bounds a
  reduced cost of 1e-17 on eta gives `-inf`.
- **The bound is worked exactly and rounded down.** Each product is split into two doubles that
  sum to it (Dekker 1971); each reduced cost's sign, and the bound, are read off `math.fsum` of the
  parts, which rounds their exact sum correctly, and where that lands above the sum the double
  below is taken. So the bound is its duals' exact bound rounded down, and a bound of nothing is
  nothing, which `allocate_geos`' stopping rule, a share of the bound, needs to close there. Where
  a product's factors are too small for the split to be exact, their exponents under -900
  together, its rounding is allowed for; past `2**300` there is no bound.
- **A bound scaled into currency is rounded outward**, a place past each rounding; nothing stays
  nothing. A scaling by a power of two that scaling back returns exactly is exact, and is not
  moved.
- **`minimax_allocate` and `cvar_allocate` write their programs in powers of two.** The rates are
  read in the power of two at or below the budget a period, each held at most the budget, which
  the budget's row implies, and the regrets or gains in the power of two at or below the largest
  best return, or the reference's largest return, and in units of 1 where that is nothing. A slope
  is then the return a budget's spend at it would bring over the largest, near 1 on a curve the
  budget bends; read a currency unit at a time it was about one over the budget a period, and
  HiGHS left it out from 1e9. A power of two scales each number exactly unless it leaves the
  normal doubles, so the program HiGHS solves is the planes', each limit rounded in currency as
  before, and its bound and split come back exactly; the split is then moved onto the budget's
  line in currency, as before. The bound is not exact where `cvar_allocate` divides it by its
  weight, `max(level n, 1)`, and that is not a power of two. The costs stay as they were, the
  least 1 (ADR 0052).
- **`cvar_allocate` also reads the bound from its duals evened over the readings**, and keeps the
  higher. A reading's mass past its excess's cost, or masses that miss eta's cost, cost the bound
  the reduced cost times its column's reach. HiGHS's masses stood off by up to 3e-7 on the 400
  readings, which cost 1.4e-9 of the reference's largest return, past the search's tolerance of
  1e-9. Each reading's mass is clamped to its excess's cost and the masses are brought to eta's, as
  far as the readings with a mass can take them; the bound then stood 2e-11 below the feasible
  point's value.
- **HiGHS's `small_matrix_value` stays at 1e-9.** `linprog` passes it only as an unrecognised
  option, with an `OptimizeWarning` on every call (SciPy 1.18.1), and HiGHS takes nothing below
  1e-12, so it would move the threshold three decades without closing it. Read in powers of two,
  the least entry other than nothing in the 4114 programs the tests that plan write at a fixed seed
  is 2.9e-5, past two whose returns underflow to nothing (below); read in currency it was 2.1e-7.
- **A program HiGHS does not solve is read as before.** SciPy returns no duals for it, and the
  planners keep the last program's bound.

## Consequences

- `tests/test_allocation_bound.py` holds the bound to its duals' exact bound rounded down, on random
  programs and duals, nan, infinite and wrong-signed among them; each product to two doubles that
  sum to it, or to its rounding within the miss where the split cannot be exact; the bound at or
  below its duals' where a product underflows, and nothing where a factor is nothing; at or below
  the least of small programs found at their vertices, and within 1e-9 of it from HiGHS's duals; the
  two programs above at 8191/8192 and under 1/5; a cvar program's implied box, `-inf` without it,
  and each planner's boxes around HiGHS's solution of every program it solved, at budgets under 1
  and with rates held above nothing among them; the evened duals back within 1e-12 of the least, on
  three cases of their own, and the higher of two bounds kept; and a plan with nothing to gain
  closing on its first program. Of 29 mutations of the bound's code, 28 are caught. The survivor
  writes the split's constant as `2**27` for `2**27 + 1`, and is equivalent: the product by `2**27`
  is exact, the rounding moves to the subtraction, and each half still has 26 bits; 8.5 million
  products agreed to the bit.
- The same file holds the two plans above to their best at budgets of 1e9, 1e10, 1e12 and 1e15,
  each search closed within 1e-9 of the largest return; a plan on two crossing readings of
  Michaelis-Menten curves, grown with its budget from 1 to 1e6 and 1e12, to its split and bound in
  shares of the budget, to the search's tolerance; each program's totals and rates' bounds to their
  powers of two, and the bound to the program's value times its power of two, exactly, below the
  worst regret on the crossing readings; the power of two to the amount's, a subnormal's among
  them; and a scaling by a power of two to the exact double where one holds it, and past the exact
  value where none does, or where the divisor is 3. All 28 mutations of the units' code are
  caught.
- At budgets of 1e9 to 1e15 the two plans above close on their first program, at the best:
  `minimax_allocate` on the even split, its worst regret and bound a quarter of the best return, and
  `cvar_allocate` on all on the first channel, its cvar and bound a quarter of the budget, each
  exactly. Read in currency their slopes fell under 1e-9: HiGHS's objective closed both on a false
  bound before this record, and the bound from the duals alone left them open, `minimax_allocate`
  after 500 rounds and `cvar_allocate` at its cap.
- On the tests that plan, at a fixed Hypothesis seed: reading the bound from the duals moved no
  search's rounds, boxes or status, and 107 bounds, each program's within 3e-12 of HiGHS's
  objective, relative to its size past 1. Writing the programs in powers of two moved no status;
  one search on S-curves cut 25 boxes for 23, in 609 programs for 607, to the same split within
  4e-16; 95 results moved, the worst regret or cvar of each by at most 0.016 of the larger of its
  two searches' gaps; and each program's bound stayed within 2.2e-11 of HiGHS's objective. At a
  budget of 1e18 the bounds of two plans came to their exact values, 5e19 and 2.5e19.
- Where every return underflows to nothing the returns are read in units of 1, and a slope's
  entry, times the budget's power of two, is as small as the budget. At budgets of 1e-160 and
  7.59e-310, on cubic Hill curves, HiGHS leaves the slopes out, and the bound read on the program
  as written holds, 1.0e-160 and 4.0e-310 over a cvar of nothing, where read in currency it was
  5.2e-161 and 1.8e-311. The search's tolerance there is 1e-9, and each closes on its first box.
- On searches over 400 readings, which no test runs: reading the bound from the duals cut two more
  boxes at six channels and the level 1, 411 for 409, and moved four others' bounds by at most
  2.5e-10, every box, status, split and cvar kept. Writing the programs in powers of two kept every
  status, and the boxes of four; at six channels and the level 1 it cut 403 boxes for 411. The
  splits moved by up to 0.06 of 600, and the cvars by up to 0.034 of the searches' gaps; the bounds
  moved by up to 2.4e-3, at the level 0.1, where the cap stopped the search with a gap of 0.034 for
  0.037.
- Reading the bound sorts the entries of the rows with a dual and sums them with `math.fsum`, one
  sum a column, and sums the terms twice more. No timing is quoted.

## Alternatives

- **Setting `small_matrix_value` to 1e-12.** Rejected: an `OptimizeWarning` on every program, and
  past a budget of 1e12 the slopes are left out again.
- **Refusing a program with an entry HiGHS leaves out.** Rejected: a saturated curve's slope is
  that small without harm, and the bound holds with it.
- **Bounding the rounding a priori,** each reduced cost enclosed by its count of terms times the
  unit roundoff of their magnitudes. Rejected after a trial: a bound of nothing came out 2.7e-14
  below it, and the share-of-the-bound stopping rule then ran four of `budget_for_geos`' tests to
  500 rounds where they had closed in one.
- **Rational arithmetic in the library.** Rejected: exact, but it works each entry as a fraction;
  the tests use it to check the doubles.
- **Stopping on HiGHS's objective and certifying with the dual bound.** Rejected: a search stopped,
  or a box pruned, on an objective above the least is the defect.
- **Refining the duals of HiGHS's basis in extended precision.** Rejected for now: SciPy does not
  return the basis, the bound from any duals is safe already, and evening `cvar_allocate`'s masses
  recovers what the 400-reading search needed.
- **The rates in units of the largest cap, as `allocate_geos` reads them.** Rejected: a cap far
  above the budget stretches every slope by their ratio, toward the 1e15 past which HiGHS refuses
  an entry. `allocate_geos` reads its worths at the caps; these programs read theirs at the budget,
  which also bounds every rate a split of it can take.
- **The budget a period itself as the unit.** Rejected: each number written, and the bound and the
  split read back, would be rounded; the power of two at or below it holds the rates under 2.
