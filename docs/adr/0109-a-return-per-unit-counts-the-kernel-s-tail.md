# ADR 0109 — A return per unit counts the kernel's tail

**Status:** accepted, 2026-10-19. Amends the carryover `roi` and `marginal_roi` count in ADR 0030.
Amended by ADR 0126, 2026-10-21: `tail=True` is the default, and the warning is gone.

## Context

ADR 0030 made `roi` and `marginal_roi` count the carryover that falls inside the spend series.
Spend in the last periods carries over past the series' end, and that return was cut. Through a
geometric kernel of retention 0.9 and length 30 on a Michaelis–Menten channel at 50 a period over
52 periods, the last period's `marginal_roi` read 0.0003. With 29 periods of no spend after the
series it read 0.0156, 52 times more. Its `roi` read 0.0003 against 0.0165. Over the last 12
periods the cut was 6–7 times, and over the whole series 1.3–1.6 times. A report that ranks
channels by these numbers ranks the long-carryover channel last for the reason that its return
comes later.

The allocation already counts the tail: a plan's worth runs each kernel's length after the plan
with nothing spent. The two readings of one channel disagreed.

## Decision

- `roi` and `marginal_roi` take `tail=`. `True` extends the series by the kernel's length less
  one periods of no spend, all of a `Channel`'s tail. A whole number extends it by that many
  periods, for a function of the series that is not a `Channel`. `False` keeps the cut.
- The default, `None`, keeps the cut and warns (`FutureWarning`) where the cut moves the number:
  on a `Channel` whose window spends in the kernel's last `length - 1` periods. A window that ends
  earlier has all its carryover inside the series, and gives the same number either way.
- A future release makes `True` the default. The warning names both options.
- Spend after the series is not counted. A caller who knows it extends the series with it.

## Consequences

- No number moves by default. A caller who reads the last periods of a series sees a warning
  until they choose.
- `tail=True` on a callable that is not a `Channel` raises `TypeError`: its kernel length is not
  known.
- `contribution` is unchanged. It reads the return inside the window's periods, which have no
  tail.
- `tests/test_response.py` holds a linear channel's last period to `b sum(w)` with the tail and to
  `b` without it, an unset tail to its warning where the cut moves the number and to no warning
  where it does not.

## Alternatives

- **Pad by default now.** Rejected: it moves every reading near a series' end without a warning,
  and the library ships behaviour changes opt-in first.
- **Pad by the series' own length or to a fixed horizon.** Rejected: the kernel's length is the
  whole of its carryover, and a longer pad adds zeros only.
- **Document the cut only**, as the docstring did. Rejected: a 52-fold cut in the reading a
  report ranks channels by needs a way out, not a note.
