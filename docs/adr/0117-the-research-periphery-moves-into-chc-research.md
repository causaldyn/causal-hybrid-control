# ADR 0117 — The research periphery moves into `chc.research`

**Status:** accepted, 2026-10-19. Supersedes the decision "Module paths do not move" of ADR 0016;
the rest of ADR 0016 stands.

## Context

0.20.0 freezes the library's surface. Sixteen of its modules each carry one research result — a
solver, a certificate, a benchmark or a case study's plant — and none is a stage of a decision
taken from logs: `benchmark`, `causal_bench`, `deep_galerkin`, `epidemic`, `flagship`,
`galerkin`, `koopman`, `lalonde`, `meanfield`, `mintime`, `mmm`, `sensitivity`, `spine`,
`surrogate`, `symbolic` and `transport`. They sat at the top of the package beside the spine, so a
freeze of the surface would have frozen them too.

Three modules mixed the two. `chc.regret` held the LQ solve, the certainty-equivalence gap, the
exploration certificates and policies and the minimax-regret actions, which the lifecycle and the
planners use, beside 103 certificates that each back one result. `chc.games` held the equilibrium
solver and three certificates that price it, and `chc.residual` the residual backends and the
evidence for one of them. `chc.lqr` imports `certainty_equivalence_gap`, so the decision façade
could not be imported without all 7,279 lines of `chc.regret`, and their import of
`chc.network_causal`.

ADR 0016 decided that module paths do not move, so that the top level's migration is a
find-and-replace from its warning's text. That decision left no way to separate the periphery
from the surface other than leaving it in place.

## Decision

- **The sixteen modules move whole to `chc.research.<module>`.** The research halves of `regret`,
  `games` and `residual` move to `chc.research.regret`, `chc.research.games` and
  `chc.research.residual`. The split follows the closure of the kept names over the module's own
  definitions, so no kept name reaches a moved one; the kept half of `chc.regret` imports nothing
  from `chc.network_causal`.
- **Each old path serves what moved until 0.20.0.** A module that moved whole stays importable at
  its old path, and each name read from it is served from the new path with a
  `DeprecationWarning`: `chc.mmm.run_marketing_mix leaves chc.mmm in 0.20.0: import it from
  chc.research.mmm`. A split module serves its moved names the same way. The warning points at the
  reader's line. A private name, and one that never existed, is an `AttributeError`.
- **An old path imports without a warning, and loads nothing from `chc.research`.** `import chc`
  imports the old paths it imported before, so `chc.mmm.<name>` keeps working, and stays silent for
  a caller that runs with warnings as errors. Nothing outside `chc.research` imports from it.
- **Type checkers see each moved name at its old path** while the warning lasts: a whole module by
  a star import of its new path under `TYPE_CHECKING`, a split module by `name as name` imports of
  exactly the names it serves. A test holds each set equal to what is served at run time.
- **The top level's 166 deprecated names that moved are served from `chc.research`**, so a caller
  is warned once, with the path to use. The 421 names of ADR 0016 and their removal at 1.0 are
  unchanged.
- **The library, its tests, scripts and notebooks, and the bench import nothing from an old path.**
  pytest turns the new warning into an error in both repositories, as it does the top level's.
- **`chc.research` is a stability tier of its own**, outside the surface 0.20.0 freezes: any of
  its modules may change or be withdrawn in any release.

## Consequences

- A caller of a moved name is warned at the line that reads it, and the warning names the path to
  use. The migration is a find-and-replace from the warning's text, as ADR 0016 wanted.
- `import chc.decision` loads nothing from `chc.research`: `chc.regret` falls from 7,279 lines to
  1,522, and the 6,020 that moved are not loaded. `chc.network_causal` is still imported, by
  `import chc` itself, as ADR 0016 keeps every module it imported then.
- An object pickled before 0.19.0 names its class's old module, and still unpickles through the
  served name, with the warning; from 0.20.0 it does not.
- The API reference files the research modules under `api/research/`, so the old pages' URLs stop
  resolving. The site has no redirects.
- In 0.20.0 the old paths, `chc._moved`, their `TYPE_CHECKING` blocks, the eager imports of the
  sixteen old paths in `chc/__init__.py` and the pytest filter are deleted. The top level's 166
  entries stay until 1.0, pointing at `chc.research`.
- The certificates that each back one research result inside modules of other tiers — the
  rollout, risk, ensemble and confounding certificates of `chc.uncertainty`, two in `chc.barrier`,
  the delay certificates, the reachability gaps and `linearized_regret_certificate` in `chc.lqr` —
  are filed in the research tier's terms but stay at their paths for now.

## Alternatives

- **Keep the paths and only file the modules in a research tier** (ADR 0016's rule). Rejected: the
  freeze is about what a caller can import, and a tier written on a page does not separate the
  import graph; `lqr` would still load all of `regret`.
- **Move them to a separate distribution.** Rejected for now: the proofs and derivations cite the
  routines by path, the tests share the library's fixtures, and a second package needs its own
  release. `chc.research` keeps the option open.
- **Drop the old paths without a warning.** Rejected: the README promises a minor of
  `DeprecationWarning` before a removal.
- **Warn on `import chc.mmm` itself.** Rejected: `import chc` imports it, so every caller would be
  warned, and a caller that runs with warnings as errors could not import the library.
