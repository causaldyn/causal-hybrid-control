# ADR 0016 — The top level is the lifecycle

**Status:** proposed, 2026-09-28.

## Context

Up to 0.7.0, `chc` bound 459 names at its top level, one per public routine of every module, from
the decision façade to the solvers, residual backends and certificates of single research results.
A reader of `chc.__all__` could not tell the names a decision goes through from the rest, and every
one of them was a promise that could only be kept or broken with a deprecation cycle.

The README's roadmap has said since 0.7.0 that 0.8.0 starts narrowing the top level, and that 1.0
keeps only the lifecycle's names there. `docs/lifecycle.md` files the library by the six stages of
a decision taken from logs, and says every name on it is importable from `chc`. Which names are
"the lifecycle's" needs a rule that a test can check, or the top level grows back one convenient
name at a time.

## Decision

- **The top level holds four kinds of name**, and no other:
  - every name `docs/lifecycle.md` files;
  - every class of the library that a call to one of them takes: a function's arguments, and the
    constructor's and public methods' arguments of a class on the page that no function on it
    returns. Protocols are left out, since a caller satisfies one with a class of its own;
  - the errors those calls raise for a caller to catch: `NotIdentifiedError`, `DecisionError`,
    `PanelError`, `CyclicGraphError`, `InfeasibleEvaluation`;
  - `__version__`.
  That is 60 names and `__version__`. What a call returns is read, not built, so it lives at its
  module path.
- **Every other name 0.7.0 bound, 421 of them, still imports from `chc` until 1.0**, served by a
  module `__getattr__` with a `DeprecationWarning` that names the module it lives in: `chc.rollout
  leaves the top level in 1.0: import it from chc.integrate`. The warning points at the caller's
  line. The name is not bound in the namespace, so `dir(chc)` and `from chc import *` no longer show
  it.
- **Module paths do not move.** Every name keeps the module it had, and the migration is a
  find-and-replace from the warning's text.
- **Type checkers keep seeing the 421 names** for as long as the warning lasts: they are bound under
  `TYPE_CHECKING`, each as `name as name`, so that a checked caller is warned at run time rather
  than broken at check time. A test holds that block equal to the names served at run time.
- **`import chc` still imports every module a top-level name came from**, the 46 it imported when
  the top level bound every name, so `import chc` followed by `chc.integrate.rollout` keeps working.
- **Names added since 0.7.0 that are not the lifecycle's** (the result types of evaluation, the
  gate, experiments and switchbacks, and `fit_logger`) were never released at the top level, and are
  not put there.
- **The library, its tests, scripts and notebooks, and the bench import nothing the top level only
  serves with a warning.** pytest turns that warning into an error in both repositories.

## Consequences

- `chc.__all__` held 61 entries when this was adopted, and reads as the lifecycle: the page's names,
  what their calls take, what they raise.
- A caller importing any of the 421 names from `chc` is warned at the import's line, and the warning
  names the module to import it from. Under Python's default filters a `DeprecationWarning` shows in
  `__main__` and under pytest, not in a library that imports the name, so a library depending on
  `chc` sees it in its own test suite.
- `from chc import *` binds the lifecycle's names instead of 459, without a warning. A star import
  cannot be warned per name without warning 421 times, and the README's promise of a warning before
  a removal is about names a caller writes.
- A name the lifecycle page gains has to be importable from `chc`, and so do the classes its call
  takes; the tests fail until it is.
- At 1.0, `_MOVED`, the `TYPE_CHECKING` block and `__getattr__` are deleted, and the 421 names stop
  importing from `chc`.

## Alternatives considered

- **Keep all 459 until 1.0 and cut them there without a warning** (D17 (b)). Rejected: the README
  promises a minor of `DeprecationWarning` before a removal.
- **Cut them in 0.8.0** (D17 (c)). Rejected for the same promise.
- **Only the page's names, about 30.** Rejected: a caller of `evaluate_plan` would leave the top
  level to build the plant and the policy it takes, and of the gate to build the batches it reads.
  The top level is where a decision's code should be able to stay.
- **Bind the moved names and warn on access through a module subclass.** Rejected: a bound name is
  found before any hook runs, and replacing the module's class to intercept every attribute read
  costs every access to every name, the lifecycle's included, for a warning 421 names need.
- **A `__getattr__` visible to type checkers, returning `Any`.** Rejected: it would type every moved
  name, and every misspelt one, as `Any` for the whole deprecation.
