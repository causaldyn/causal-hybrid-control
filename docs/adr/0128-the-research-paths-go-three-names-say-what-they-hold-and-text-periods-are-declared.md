# ADR 0128 — The research paths go, three names say what they hold, and text periods are declared

**Status:** accepted, 2026-10-21. Carries out ADR 0117's removal; amends ADR 0058 (text periods).
Extended by ADR 0130: three more names renamed in 0.21.0, and `to_json` stops writing
`identification_radius` at 1.0.

## Context

0.20.0 freezes the library's surface. Three things 0.19.0 had announced, or left with a misleading
name, sat on it.

- **The paths the research modules left.** ADR 0117 moved sixteen modules into `chc.research`
  whole, and the research halves of `chc.regret`, `chc.games` and `chc.residual` beside them. Each
  old path served each moved name with a `DeprecationWarning`: `chc.mmm.run_marketing_mix leaves
  chc.mmm in 0.20.0: import it from chc.research.mmm`. Serving them took `chc._moved`, a
  `TYPE_CHECKING` block per old path, an `__all__` and a `__dir__` per old path so that a
  star-import reached the served names, eight eager imports in `chc/__init__.py` and a pytest
  filter.
- **Three names said more than their values.**
  - `DecisionCertificate.identification_radius` holds the channel's standard error, the value
    `CausalDynamicsFit.channel_error` holds, and the report already prints it as "channel standard
    error". The barrier's identification radius (`identification_radius_threshold`) is a different
    quantity: a radius on the channel, not an error of its estimate.
  - `ConfoundingRobustLQRegretCurve.regret_bounds` holds `L_reg (eps + Delta)^2`, the regret to
    second order. The exact regret's worst over the effect interval exceeds it, by 0.9 % at
    `Gamma = 1` and up to 8.4 % on the defaults: none of the values is a bound.
  - `highprob_regret_certificate` returns a band that holds with probability `1 - delta` only
    where the regret's curvature is constant over the confidence region. At `rr = 0.98`, `n = 50`
    it covers the exact regret in 0.725 of the draws at `delta = 0.01`.
- **Text periods.** 0.19.0 read undeclared text periods in their sorted order with a
  `FutureWarning` that a future release would refuse them: `"10"` sorts before `"2"`, and every
  month name out of the calendar's order.

## Decision

- **The old paths are deleted.** `import chc.mmm`, and each of the fifteen others, is a
  `ModuleNotFoundError`; `chc.regret.van_trees_certificate`, and each other moved name of the
  three split modules, an `AttributeError`. `chc._moved`, the `TYPE_CHECKING` blocks, the
  listings, the eager imports and the pytest filter go with them.
- **The top level's deprecated names stay until 1.0**, as ADR 0016 and their warning say: the 166
  that moved are served from `chc.research`, as since 0.19.0.
- **Each misleading name is renamed to what it holds, and the old one stays until 1.0** with a
  `DeprecationWarning` that names the new one: `<old> is renamed <new> and leaves in 1.0: <why>`.
  - `DecisionCertificate.identification_radius` → `channel_error`, the library's one name for the
    value. The old name is a property; `to_json` writes both keys, `identification_radius`
    repeating `channel_error`, so a reader of the record keeps working. The schema version does
    not move: a key is added, none changes its meaning.
  - `ConfoundingRobustLQRegretCurve.regret_bounds` → `second_order_regrets`, a property.
  - `highprob_regret_certificate` → `local_highprob_regret_certificate`; the old name is a function
    that warns and calls the new one. `HighProbRegretCurve` keeps its name.
- **pytest turns the renamed names' warning into an error**, as it does the top level's, so the
  library, its tests and notebooks use the new names.
- **Undeclared text periods are refused** with a `PanelError` that names the column and the two
  ways on: convert the periods to numbers or dates, or declare `frequency="observed"` to read them
  in their sorted order.

## Consequences

- A caller of an old research path fails at the import, with Python's own message; the 0.19.x
  warning named the path to use. An object pickled before 0.19.0 names its class's old module and
  no longer unpickles.
- `import chc` no longer binds `chc.mmm`, `chc.koopman` and the six other research modules it
  bound as attributes.
- A caller of a renamed name is warned at the line that reads it. A caller that builds a
  `DecisionCertificate` by hand passes `channel_error=`: the constructor does not take the old
  keyword. The library builds the certificate; nothing in it, its tests or notebooks did.
- A panel with text periods and no frequency, accepted with a warning in 0.19.0, is refused.
  `frequency="observed"` reads it as 0.19.0 did.

## Alternatives considered

- **Read text periods in a natural order**, numbers within text compared as numbers. Rejected:
  it orders `"2"` before `"10"`, but month names, `"Q1-2024"` against `"Q4-2023"`, or any label
  that is not a number with a prefix, still fall out of the calendar's order, and without a
  warning. ADR 0058 declined to infer a calendar for the same reason. The 0.19.0 warning promised
  a refusal.
- **Rename the JSON key without writing the old one.** Rejected: a stored record's reader would
  break with no warning, and a JSON reader cannot be warned.
- **Keep the old research paths until 1.0.** Rejected: their warning promised 0.20.0, and the
  freeze would otherwise carry sixteen modules whose only content is a forwarding table.
- **`channel_standard_error`** for the certificate's field. Rejected for one term per concept:
  `CausalDynamicsFit.channel_error` is the value it copies.
