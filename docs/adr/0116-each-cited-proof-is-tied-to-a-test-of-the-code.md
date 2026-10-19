# ADR 0116 — Each cited proof is tied to a test of the code

**Status:** accepted, 2026-10-19. Amended the same day, before any release: no expected failure
is left, and a statement whose docstring claimed more than its lemma is narrowed, not marked.

## Context

Docstrings cite Rocq proofs under `proofs/` and Maxima derivations under `validation/`. A proof
checks a statement over the reals. It does not check the Python that computes the statement, and
no test tied the two. A docstring could cite a lemma that its code no longer satisfies, and
nothing would fail. The theory page said so.

## Decision

- Each docstring that cites a file under `proofs/` or `validation/`, or a Rocq lemma by name, is a
  citation. `tests/test_formal_links.py` finds every citation in `src/chc` by reading the
  docstrings.
- Each `tests/test_formal_<area>.py` holds a `FORMAL_LINKS` literal. It maps each cited object to
  the files it cites and to the tests that check the cited statement on the float code. Where no
  float test can check the statement, the entry gives a reason instead.
- `test_formal_links.py` fails when a citation has no entry, when an entry lists other files than
  its docstring cites, when a listed test does not exist, or when a cited file is missing. A new
  citation therefore lands with its test, and a changed citation makes its test be read again.
- A test checks the statement independently of the code path where it can: a brute-force
  minimum, a series, a quadrature, a Monte Carlo risk with its standard error. Tolerances are what
  float64 needs, written as `rel=` with `abs=0.0` for a relative claim.
- Where the code does not satisfy the statement inside the lemma's hypotheses, the test is an
  explicit counterexample marked `xfail(strict=True)` with the numbers. The test is not loosened.
  When the code is fixed, the strict mark fails and must be removed.
- A strict mark is a state to leave, not to keep. Where the code computes the wrong number, the
  code is fixed. Where the docstring claims more than the lemma proves, the docstring is narrowed
  to the lemma's premise with the measured numbers, and the counterexample becomes a test of the
  narrowed statement and of the number that broke the old one.
- Closed forms are cross-checked in a second engine (Julia, PARI/GP or FriCAS) at sample points.
  These checks run by hand and are not part of CI.

## Consequences

- At the start, 117 objects cite 139 files. All 117 have tests; none needed a reason.
- 22 cited statements failed on the code. Nine had a local fix and were fixed. The other 13
  were strict expected failures, on eleven cited objects. By the amendment six of the 13 are fixed
  in the code and seven are read as narrowed docstrings; the theory page lists both.
- After the merges that followed, 123 objects cite 145 files, all with tests.
- A docstring's citation can no longer outlive the code it describes without a red test.
- The links name the tests, so the theory page can say which lemma is tied to which test.
- The suite gains the formal tests' run time.
- A test of a proof's statement is not a proof about the code. It samples the hypotheses' region.

## Alternatives

- **Extract the Rocq definitions to code.** Rejected: the proofs are over the classical reals,
  and the code is float64 NumPy and JAX. An extraction would test neither.
- **A pytest marker on each test.** Rejected: an unregistered marker warns, and a registry
  checked by reading source needs no collection step.
- **Name the tests in the docstrings.** Rejected: it puts test names in the public API's text.
