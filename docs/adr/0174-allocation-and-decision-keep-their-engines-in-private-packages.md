# ADR 0174 — `chc.allocation` and `chc.decision` keep their engines in private packages

**Status:** accepted, 2026-11-25. No public name, import path, signature, record field or number
moves.

## Context

`chc/decision.py` had grown to 6 747 lines and `chc/allocation.py` to 4 296. A reader of either
looks for one of a few hundred lines among thousands. Allocation's seven public functions share
engines that know nothing of the public classes: the curves' worth and their concave envelopes,
the cutting-plane linear programs, the branch and bound over boxes of rates, the multi-period
schedule, the search over several readings, and the geo layout's search. Decision's tube reads,
832 lines in 32 definitions, read nothing of the rest of the module at run time: the three
constants they read are their own. The JSON form of the decision record reads nothing of it
either. The rest of `decision.py` is not like
that: `Prescription` and `prescribe` are 2 575 lines in two definitions, and most other helpers
raise `DecisionError` or build a public class.

A split must not change what any caller sees. The module's own names define what a caller reads:
`chc.decision.Lever`, a dataclass's `__module__` in a pickle or a traceback, the
`::: chc.decision` page, a test's `monkeypatch` of a module global, the logger a `caplog` reads,
the keys of the formal links, and the tier test's crossings.

## Decision

- **Public definitions never move; private modules never import the public one at run time.**
  Every name of `__all__` is defined where it was, so its `__module__`, its pickle path, its
  docs anchor and `inspect.getsource` stay true, and no class needs `__module__` set. A private
  module names a public class in an annotation only, under `TYPE_CHECKING`. A helper that reads a
  public name at run time therefore stays, and so does a helper that reads one that stays. The
  public module imports the helpers it calls, as before, and re-exports nothing new.
- **`chc._allocation`** holds `curves`, `planes`, `search`, `schedule`, `readings` and `geo`,
  layered so: `curves` first; `planes` and `search` on it; the three searches on those.
  `allocation.py` keeps the public classes and functions, the input checks, the plan on the
  envelopes, the goal seeks, the geo layout and the logger.
- **`chc._decision`** holds `tube` (the tube's reads) and `record` (the record's JSON form).
  The replay, `_replay_arguments`, stays: it builds `Lever`, `Target` and the other inputs and
  raises `DecisionError`. So do `_cost_bias`, the settings that tests patch, and every helper
  that reads them.
- **The move is verbatim.** Every line of code outside the import blocks is in exactly one file
  after the move, as many times as before (3 696 lines of allocation, 6 088 of decision). The new
  files add only their docstrings and their imports.
- **A test patches a name in every module that binds it.** After the split, one name can be bound
  in several modules: `linprog` in `planes` and `geo`, `_slope` in four. The conftest fixtures
  `allocation_patch` and `decision_patch` set a name in the public module and in every module of
  its private package that binds the same object. They refuse a name that no module binds. Before,
  the geo test patched `_slope` on `chc.allocation` alone. Against the split it read 0 calls and
  still passed `reads < 5_000`, a test that checked nothing.
- **A private package named after a module is that module's own** in `tests/test_tier_boundary.py`.
  `chc._decision`'s imports are read as `chc.decision`'s, and its crossings keep their entries.
  Imports between the two cross nothing.

## Consequences

- The replay gate, `test_a_record_replays_to_the_bit`, gives its records back to the bit on all
  three shapes after each step. After each of the two moves, 1 570 tests pass: the allocation and
  decision files, every file that patches either module, and the public-API, tier, docs-pages,
  `__all__`, formal-links and docs tests. So does the misspecification test that patches the
  noise read. Each of the 79 names those tests patch is called as many times in each test as
  before the move. The public surface of both modules, their names with `__module__`,
  signatures and fields and their loggers, reads the same.
- `import chc` loads 48 of the library's modules before and 51 after: the `chc._decision`
  package and its two modules. No other package's module is added. `chc.allocation` is not
  imported by `import chc`; importing it loads 7 more of the library's modules.
- `chc.allocation` binds 46 underscore names instead of 99. A test of a moved helper imports it
  from its module, for example `from chc._allocation.planes import _least`. The formal links of
  `_per_log_spread`, `_plan_field`, `_field_scale` and `_field_errors` are keyed under
  `chc._decision.tube`. The docstrings and the `validation/` files that cite those helpers name
  their new path.
- Git does not follow code moved between files: a branch that changed a moved helper replays its
  change in the new file.
- `decision.py` is 5 443 lines, and `Prescription` and `prescribe` are most of it.

## Alternatives

- **Move public classes too, with `__module__` set to the public module.** Rejected. A
  dataclass's generated methods, `typing.get_type_hints` and `inspect.getsource` then read the
  module that `__module__` names, not the one that defines the class. From Python 3.13,
  `getsource` returns another class's lines without an error. For the same reason
  `DecisionError` and `NotIdentifiedError` stay. That rule keeps about 950 lines of
  `DecisionError`-raising helpers in `decision.py`.
- **Turn `decision.py` into a package.** Rejected: the rename defeats merges of open branches,
  and the docs-pages and formal-links tests read modules by their file paths.
- **Private sibling modules** (`chc/_allocation_curves.py`, as with `_robust_cov.py`). Rejected:
  a package names its owner in one place, which the tier test reads, and keeps `src/chc/` short.
- **Function-level imports to let a private helper build a public class.** Rejected: they hide
  the layering and run on every call.
- **Cut `prescribe` into stages.** Not a move: it changes code that every record passes through,
  and it is left for its own change.
