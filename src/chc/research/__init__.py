"""The research periphery: modules that each carry one research result, outside the public API.

A module here backs a result the documentation reports -- a solver, a certificate, a benchmark, a
case study's plant -- rather than a stage of a decision taken from logs. None of them is part of
the surface 0.20.0 freezes: any of them may change or be withdrawn in any release, and `import chc`
imports none of them. Import each from its own path, `chc.research.<module>`.

Until 0.20.0, each name here still imports from the path it had before 0.19.0 -- `chc.mmm`, or the
research half of `chc.regret`, `chc.games` and `chc.residual` -- with a `DeprecationWarning` that
names its path here (docs/adr/0117).
"""
