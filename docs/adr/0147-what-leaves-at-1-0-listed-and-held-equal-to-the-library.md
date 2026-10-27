# ADR 0147 — What leaves at 1.0, listed, and held equal to the library

**Status:** accepted, 2026-10-27. Collects the removals that ADRs 0016, 0117, 0128 and 0130 date
to 1.0, and states the release's procedure. Changes no behaviour in 0.24.0. Amended 2026-10-27:
the list gains the kind `class`, a class renamed and served under its old name by its module's
`__getattr__` with the standard warning; its first entry is `chc.marketplace.DitherCheck`, renamed
`GradientDitherCheck` apart from `chc.decision.DitherCheck`, a different test (ADR 0140).

## Context

Every deprecated name in the library says that it leaves in 1.0, and 1.0 removes them together
(ADR 0130). What leaves was written down in four places, and not all of it was written down:

- **The top level's table.** ADR 0016 kept 421 names that 0.7.0 bound at the top level and the
  lifecycle does not hold, served from `chc._MOVED` with a `DeprecationWarning` since 0.8.0. ADR
  0117 moved 166 of them into `chc.research` in 0.19.0, and the top level serves those from there.
- **Renamed names.** ADR 0128 renamed three names in 0.20.0, and ADR 0130 four more in 0.21.0
  that keep an alias.
  A fourth 0.20.0 rename, `NonlinearDwrCurve.denominator_against_result_55`, followed ADR 0128's
  rule with no record of its own: only its changelog line and its warning name it.
- **Re-exports.** `chc.research.sensitivity` imports the two old regret names outside its
  `__all__` (ADR 0130). An import is not a warning: a scan of the source for warnings does not
  find it.
- **A record key.** `Prescription.to_json` writes the certificate's `identification_radius`
  beside `channel_error`, and at 1.0 stops, moving `SCHEMA_VERSION` from 2 to 3 (ADR 0130). A
  JSON key cannot warn.

A list made by hand when 1.0 is cut misses whatever has no warning, and a deprecation added after
the list was made is missed too.

## Decision

- **The list.** Each entry names what a caller writes today, what to write instead, the release
  that deprecated it and the records that did.

  | what leaves | write instead | since | ADR |
  |---|---|---|---|
  | `chc.<name>`, each of the 421 names of `chc._MOVED` | `<module>.<name>`, the module the warning names; 166 are in `chc.research` | 0.8.0 | 0016, 0117 |
  | `DecisionCertificate.identification_radius` | `channel_error` | 0.20.0 | 0128 |
  | `ConfoundingRobustLQRegretCurve.regret_bounds` | `second_order_regrets` | 0.20.0 | 0128 |
  | `highprob_regret_certificate` | `local_highprob_regret_certificate` | 0.20.0 | 0128 |
  | `NonlinearDwrCurve.denominator_against_result_55` | `denominator_against_closed_form` | 0.20.0 | 0128's rule |
  | `confounding_robust_lq_regret` | `confounding_robust_lq_second_order_regret` | 0.21.0 | 0130 |
  | `confounding_robust_lq_regret_matrix` | `confounding_robust_lq_second_order_regret_matrix` | 0.21.0 | 0130 |
  | `MultivariateActionFloor.floor` | `local_floor` | 0.21.0 | 0130 |
  | `MultivariateVanTreesCurve.floor` | `local_floor` | 0.21.0 | 0130 |
  | `chc.research.sensitivity.confounding_robust_lq_regret` and `..._matrix` | the `second_order` names in the same module | 0.21.0 | 0130 |
  | `to_json()["certificate"]["identification_radius"]` | `["channel_error"]`; `schema_version` 3 | 0.20.0 | 0128, 0130 |

  The functions and classes of the renamed rows live in `chc.research.regret`, except
  `DecisionCertificate` (`chc.decision`) and `NonlinearDwrCurve` (`chc.research.deep_galerkin`).
- **The list is machine-readable, in `src/chc/_removals.py`.** It is private, imports nothing from
  `chc`, and nothing in the library imports it. The top level's 421 names are one entry: they are
  listed once, in `chc._MOVED`, which serves them, and `_removals.TOP_LEVEL_NAMES` holds the
  count. It lives beside the code it lists, so ruff and ty check it as they check that code, and
  1.0 deletes it in the same change that deletes the code.
- **`tests/test_removals.py` holds the list equal to the library, both ways.**
  - Each entry still serves its value, and its warning names the replacement: each of the 421
    names from the module the warning names; each renamed function with a placeholder for every
    argument it requires and its new name replaced by a stub, so that what the old one returns is
    what the new one does; each renamed property on a stand-in that holds the new name alone. A
    re-export is the listed function, beside its new name, outside its module's `__all__`. The
    record's function writes the listed key and its new one from the same value, at
    `SCHEMA_VERSION` 2.
  - The source is read by its syntax tree. The set of functions that raise a
    `DeprecationWarning`, a `PendingDeprecationWarning` or a `FutureWarning`, or that a
    `deprecated` decorator marks, equals the list's warning sites; a `warn` whose class is not a
    plain name fails the test, since its class cannot be read. Every import of a listed function
    into another module, the top level aside, is a listed re-export. Every dictionary key that
    holds a renamed attribute's new value under its old name is a listed record key.
  - Each mutant fails it: an entry deleted (a renamed attribute, a re-export, the record key), a
    new `warnings.warn(..., category=FutureWarning)` or a `@deprecated` function, and a name
    taken out of `chc._MOVED`.
- **The 1.0 procedure**, in one change:
  1. `chc/__init__.py` loses `_MOVED`, the `TYPE_CHECKING` block, `__getattr__` and the imports
     only they used. It keeps importing every module a top-level name came from: ADR 0016
     promised `import chc` then `chc.<module>.<name>`, and deprecated nothing there.
  2. Each renamed property and function, and the two re-exports, are deleted.
  3. `to_json` stops writing `identification_radius`; `SCHEMA_VERSION` is 3, its docstring saying
     why.
  4. pytest's two filters that turn these warnings into errors are deleted.
  5. `_removals.py` and its test are deleted. A test holds that the top level binds its
     `__all__` and its modules and nothing else; a test holds each renamed name gone and its new
     one present; the tests that read `schema_version` 2 read 3.
  6. `docs/api/index.md` says the other names left the top level in 1.0.
  7. The changelog's 1.0 section carries a `### Removed` part with three entries: the 421
     top-level names, how to find each one's module (the 0.24.0 warning names it, and so does
     `chc._MOVED` in 0.24.0's source), and one example; the eight renamed names, each old → new;
     the two re-exports. Under `### Changed` it carries the record: `schema_version` 3, the key
     that left, and what a reader of version 2 does (read `channel_error`; a reader that checks
     the version refuses the record rather than find the key missing). It also says how to
     prepare: run the suite on 0.24.0 with `-W error::DeprecationWarning`.
- **What is not on the list.**
  - No `FutureWarning` is raised anywhere in the library. The opt-in modes stay opt-in through
    1.0 and none warns: `GateConfig.centred` (ADR 0135), `objective="expected"` (ADR 0137),
    `tube_actions="plan"` (ADR 0138), `prescribe(dither=...)` (ADR 0139),
    `payment="per_driver"` and the logged dither (ADR 0140), `inner="newton"` (ADR 0144). 1.0
    moves no default.
  - ADR 0049 announced a `DeprecationWarning` for one unit's windows and a refusal at 1.0. ADR
    0083 replaced both: the call reads one unit's windows across their time zeros and logs at
    `INFO`. No code warns, and nothing leaves.
  - ADR 0031 said a plan's prices would join the record "in its one change at 1.0". ADR 0055's
    rule, applied since (ADRs 0063, 0064, 0097), adds keys at any release without moving the
    version, so the prices owe nothing to 1.0. Version 3 is the key's removal alone.
  - Moving the classifier to Production/Stable and rewriting the reference's "What 0.x promises"
    belong to the 1.0 release, not to this list.

## Consequences

- 0.24.0 changes no behaviour: no warning, value, key or default moves.
- A deprecation added before 1.0 fails the test until it is listed, so 1.0 removes it with the
  rest, or the list says why it stays.
- The 1.0 change deletes about 1 080 lines of the library, nearly all of them the top level's
  table and its `TYPE_CHECKING` block.
- At 1.0, `chc.<name>` for any of the 421 is an `AttributeError` with Python's own message, and
  `from chc import <name>` an `ImportError`; neither names the module any more. Each old renamed
  name is an `AttributeError`.
- A record's digest over the whole of `to_json` moves at 1.0, as ADR 0130 said.
- Of the two packages built on the library, mixdyn uses none of the listed names. The bench
  annotates four functions with `chc.Prescription` in `graph_errors.py`: under
  `from __future__ import annotations` they never evaluate, so its tests run, but a type checker
  stops resolving them; `from chc.decision import Prescription` fixes it. Two of the library's
  own test files annotate the same way, and the 1.0 change moves them.

## Alternatives considered

- **Remove in 0.24.0.** Rejected: every warning says 1.0, and 0.24.0 moves nothing (ADR 0130).
- **The list as a test-data file.** Rejected: it would sit apart from the code it lists, and ruff
  and ty would not read it. The module costs some 130 lines in the wheel, unimported.
- **The list holds the 421 top-level names again.** Rejected: two copies of the table drift, and
  `chc._MOVED` is already the one the library serves from, under its own tests.
- **Find the deprecations by running the library under `-W error`.** Rejected: a test run reaches
  only the paths its tests take. The syntax tree reaches every warning, re-export and key.
- **Keep the 421 names one more minor after 1.0.** Rejected: ADR 0016 dated their end to 1.0, and
  they have warned since 0.8.0.
