# ADR 0060 — A cutting-plane bound is read from the duals, rounded down

**Status:** accepted, 2026-10-07. Amends ADR 0038, ADR 0043 and ADR 0048 (how the bound of the
cutting planes is read).

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
  `minimax_allocate` and `cvar_allocate` write a slope as a return a currency unit over the
  largest return, about one over the budget, so from a budget of 1e9 every slope is left out. On
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
  nothing.
- **`cvar_allocate` also reads the bound from its duals evened over the readings**, and keeps the
  higher. A reading's mass past its excess's cost, or masses that miss eta's cost, cost the bound
  the reduced cost times its column's reach. HiGHS's masses stood off by up to 3e-7 on the 400
  readings, which cost 1.4e-9 of the reference's largest return, past the search's tolerance of
  1e-9. Each reading's mass is clamped to its excess's cost and the masses are brought to eta's, as
  far as the readings with a mass can take them; the bound then stood 2e-11 below the feasible
  point's value.
- **HiGHS's `small_matrix_value` stays at 1e-9.** `linprog` passes it only as an unrecognised
  option, with an `OptimizeWarning` on every call (SciPy 1.18.1), and HiGHS takes nothing below
  1e-12, so it would move the threshold three decades without closing it. In two runs of the
  tests that plan, 4057 and 4112 programs, the least entry other than nothing is 2.1e-7.
- **A program HiGHS does not solve is read as before.** SciPy returns no duals for it, and the
  planners keep the last program's bound.

## Consequences

- `tests/test_allocation_bound.py` holds the bound to its duals' exact bound rounded down, on random
  programs and duals, nan, infinite and wrong-signed among them; each product to two doubles that
  sum to it, or to its rounding within the miss where the split cannot be exact; the bound at or
  below its duals' where a product underflows, and nothing where a factor is nothing; at or below
  the least of small programs found at their vertices, and within 1e-9 of it from HiGHS's duals; the
  two programs above at 8191/8192 and under 1/5; the two plans above at a bound on the safe side of
  the best; a cvar program's implied box, `-inf` without it, and each planner's boxes around HiGHS's
  solution of every program it solved; the evened duals back within 1e-12 of the least, on three
  cases of their own, and the higher of two bounds kept; and a plan with nothing to gain closing on
  its first program. Of 29 mutations of the new code, 28 are caught. The survivor writes the split's
  constant as `2**27` for `2**27 + 1`, and is equivalent: the product by `2**27` is exact, the
  rounding moves to the subtraction, and each half still has 26 bits; 8.5 million products agreed to
  the bit.
- On the tests that plan, at a fixed Hypothesis seed, no search's rounds, boxes or status moved; 107
  bounds moved, each program's within 3e-12 of HiGHS's objective, relative to its size past 1.
- On searches over 400 readings, which no test runs: at six channels and the level 1, two more boxes
  were cut, 411 for 409, and the bound rose by 9e-12 to 2678.484924293806, the split and its cvar
  unchanged; four others, at the levels 1, 0.5 and 0.1, kept every box, status, split and cvar,
  their bounds moved by at most 2.5e-10.
- A search whose slopes HiGHS leaves out no longer closes on HiGHS's objective: at budgets of 1e9
  and 1e10 the two plans above, which closed on one program, stop with their gap,
  `minimax_allocate`'s after 500 rounds and `cvar_allocate`'s at its cap. Read in units of the
  largest cap, as `allocate_geos` reads them, the rates would give HiGHS slopes it keeps; that
  changes every program HiGHS solves, and is left to its own decision.
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
