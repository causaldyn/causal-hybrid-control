# Theory

The papers that develop the results behind the library are not public yet, and this page will link
them once they are. What is public, and ships in every source distribution, is the machinery that
checks the claims the docstrings make: a docstring that rests on a proof or a derivation names its
file, and the file is in one of the two directories below.

## Formal proofs — [`proofs/`](https://github.com/causaldyn/causal-hybrid-control/tree/main/proofs)

The control and guarantee invariants are formally proved in Rocq, from the box-projection bounds
and idempotence behind `project_box`
([`box_projection.v`](https://github.com/causaldyn/causal-hybrid-control/blob/main/proofs/box_projection.v))
to the interference-aware regret certificate
([`interference_regret.v`](https://github.com/causaldyn/causal-hybrid-control/blob/main/proofs/interference_regret.v)).
Where Stdlib's lack of matrices limited a proof to its scalar or 2×2 shadow, the matrix statement is
proved separately in
[`proofs/mathcomp/`](https://github.com/causaldyn/causal-hybrid-control/tree/main/proofs/mathcomp)
with MathComp, at any dimension: the robust barrier margin over an operator or Frobenius ball, the
multivariate van Trees regret identity, and the explicit constant of the certainty-equivalence
regret bound.

Two things bound what "proved" means here:

- **Every lemma under `proofs/` is stated over Rocq's classical reals**, which are themselves
  axiomatic. The gate is therefore not "closed under the global context", which no such lemma can
  be; it is that nothing *outside* Stdlib's own classical-reals axioms gets in — no project
  `Axiom`, no admitted step, no `Hypothesis` leaking out of its `Section`. The MathComp lifts are
  stated over abstract fields instead, so for them "closed under the global context" is the gate.
- **A published statistical result a proof relies on is an input, not a theorem.** It enters the
  theorem's type as a named predicate carrying its citation, rather than as an axiom, so
  `Check <theorem>` shows the cited names and `Print <Name>` what was assumed under each.

```bash
just proofs                # compile every proof (CI does this on every push, under Rocq 9.2)
just assumptions           # Print Assumptions on every lemma; fails on anything beyond Stdlib's axioms
just proofs-mathcomp       # compile the MathComp lifts (MathComp 2.6, see CONTRIBUTING.md)
just assumptions-mathcomp  # fails unless every lift is closed under the global context
```

## Symbolic derivations — [`validation/`](https://github.com/causaldyn/causal-hybrid-control/tree/main/validation)

Correctness is cross-checked in independent tools, symbolic first. The derivations are Maxima,
which is authoritative; the Riccati equation and the matrix exponential are verified in exact and
high-precision arithmetic against PARI/GP and Octave, with SciPy used only as the fast float64
numeric. Where a bound is a finite first-order statement, z3 and cvc5 check its negation is
unsatisfiable.

```bash
just derivations   # run every Maxima derivation; fails if any did not complete
```

CI runs the same check on every push. It checks that each derivation *completes*, not that its
output matches a stored copy. Only the Maxima files run there: the PARI/GP, Octave, FriCAS, R,
Python and SMT files beside them (`*.gp`, `*.m`, `*.input`, `*.R`, `*.py`, `*.smt2`) are run by
hand, and CI would not notice one of them breaking.

## Result numbers in docstrings

Docstrings refer to results by number — `§40`, `Result 55` — and those numbers index a research
log that is not public. The proof and derivation files they cite are.
