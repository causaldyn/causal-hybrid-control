"""A validated long-format panel: one place where the shape assumptions are checked and recorded.

The estimators in this library take panel data in two incompatible shapes. The DiD and synthetic
control routines want a dense ``(n_units, n_periods)`` matrix of one outcome
(:mod:`chc.did`, :mod:`chc.scm`); everything else wants flat named columns
(:mod:`chc.frames`). Callers currently pivot between the two by hand, and the pivot is where the
quiet failures live: a duplicated ``(unit, time)`` row silently keeps whichever value NumPy wrote
last, a missing one silently becomes a zero, and neither shows up until an effect estimate is
already in a slide.

:class:`Panel` does that pivot once, refuses to do it when the data cannot support it, and says
which unit and which period were responsible. It holds a read-only copy of the columns it was
given, made once, and in an object column only values that cannot change in place, so that the
data cannot change under its hash; passing one around costs what passing a dict does.

A panel holds text, dates and times as well as numbers: a unit's name, a period's date. Which
columns are numbers is the reader's question, so a column is checked when it is read as numbers,
by :meth:`Panel.wide` and by every routine here that reads a panel's column as float64, and
refused there, naming the column, the unit and the time, unless it holds real numbers.

:class:`Provenance` travels with it. A number is reproducible only together with the bytes it came
from and the precision it was computed in, and this library has already been bitten by the second:
JAX's ``x64`` flag changes which sample a seed draws, so a seed alone does not name a dataset. The
hash is over the column bytes, so two panels that agree numerically but differ in dtype hash
differently --- which is the honest answer, because they will not produce the same numbers. An
object column's bytes are pointers, which no two runs share, so it is hashed by its values: each
value's type and text, each with its length.

A panel reads its periods on a calendar only where the caller declares one, as ``frequency``:
calendar months stamped as dates lie 28 to 31 days apart, on no grid a library could infer, and
read in the order logged a month no unit logged is not seen. The declared frequency is part of
the dataset's identity, so it enters the hash.
"""

from __future__ import annotations

import datetime
import hashlib
import math
import numbers
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from chc.frames import ColumnData, _not_numbers, as_columns

# The types an object column holds: none changes in place, and the text of each names its value,
# which is what the hash reads. A subclass counts, so a pandas Timestamp is a datetime; `int`
# covers `bool`, `date` covers `datetime`, and NumPy's scalars of text and numbers subclass `str`,
# `bytes` or `np.number`. `np.void`, a structured array's element, writes through to that array.
_HELD: tuple[type, ...] = (
    str,
    bytes,
    int,
    float,
    complex,
    Decimal,
    Fraction,
    UUID,
    datetime.date,
    datetime.time,
    datetime.timedelta,
    np.bool_,
    np.number,
    np.datetime64,
)
# The numbers among them that have a nan or an infinity.
_INEXACT: tuple[type, ...] = (float, complex, Decimal, np.inexact)

_CALENDAR = ("D", "W", "M", "Q", "Y")


class PanelError(ValueError):
    """A frame could not be read as a panel; the message names the column and the entity."""


@dataclass(frozen=True)
class Provenance:
    """What a number would need beside it to be reproduced: the bytes, the version, the precision.

    ``x64`` is not decoration. JAX's double-precision flag changes how many bits a threefry key
    spends per element, so the *same* seed draws a *different* sample at the two settings; a run
    recorded without it cannot be repeated. See ``docs/concepts/dtype-policy.md``.
    """

    # over column name, dtype, shape and bytes or values, in sorted name order, and the frequency
    # where one is declared
    data_sha256: str
    chc_version: str
    n_rows: int
    columns: tuple[str, ...]
    x64: bool  # jax.config.jax_enable_x64 at the moment the panel was built
    seed: int | None = None  # the generator's seed where the data is simulated; None for observed
    frequency: str | float | None = None  # the calendar the periods are read on, where declared

    def to_json(self) -> dict[str, Any]:
        """A plain dict, ready for ``json.dumps`` beside the result it describes."""
        return {
            "data_sha256": self.data_sha256,
            "chc_version": self.chc_version,
            "n_rows": self.n_rows,
            "columns": list(self.columns),
            "x64": self.x64,
            "seed": self.seed,
            "frequency": self.frequency,
        }


def _fingerprint(columns: Mapping[str, NDArray[Any]], frequency: str | float | None = None) -> str:
    digest = hashlib.sha256()
    for name in sorted(columns):
        array = np.ascontiguousarray(columns[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(_values(array) if array.dtype == object else array.tobytes())
    if frequency is not None:
        digest.update(f"frequency {frequency!r}".encode())
    return digest.hexdigest()


def _finite(number: Any) -> bool:
    """Whether a number is neither nan nor infinite; a Decimal answers by its own test, since
    ``float`` raises on its signalling nan."""
    return number.is_finite() if isinstance(number, Decimal) else bool(np.isfinite(number))


def _values(column: NDArray[Any]) -> bytes:
    """An object column's bytes by value, which its pointers are not: each value's type and text,
    each with its length, so that no two values run together."""
    out = bytearray()
    for value in column.tolist():
        for part in (type(value).__qualname__, str(value)):
            text = part.encode("utf-8", "surrogatepass")
            out += len(text).to_bytes(8, "little") + text
    return bytes(out)


class _Columns(Mapping[str, NDArray[Any]]):
    """A panel's columns, read-only, in a mapping that takes no new column.

    A copy of a ``dict`` would still take ``panel.columns[name] = ...``, and a ``MappingProxyType``
    does not pickle, so a panel could not reach a worker process.
    """

    __slots__ = ("_columns",)

    def __init__(self, columns: Mapping[str, NDArray[Any]]) -> None:
        for column in columns.values():
            column.setflags(write=False)
        self._columns = dict(columns)

    def __getitem__(self, name: str) -> NDArray[Any]:
        return self._columns[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)

    def __repr__(self) -> str:
        return repr(self._columns)

    def __reduce__(self) -> tuple[Any, ...]:
        # a deep copy rebuilds the arrays writable; rebuilding through __init__ sets the flag again
        return (_Columns, (self._columns,))


@dataclass(frozen=True, eq=False)
class Panel:
    """Long-format panel data whose ``(unit, time)`` index has been checked exactly once.

    ``eq=False`` because the fields hold arrays: a generated ``__eq__`` would compare them
    elementwise and return an array where a caller expects a bool, so ``panel_a == panel_b`` would
    raise inside any ``if``. Identity of a panel is :attr:`provenance` and its hash.
    """

    columns: Mapping[str, NDArray[Any]]
    unit: str
    time: str
    cluster: str | None
    provenance: Provenance

    @classmethod
    def from_frame(
        cls,
        data: ColumnData,
        *,
        unit: str,
        time: str,
        frequency: str | float | None = None,
        cluster: str | None = None,
        seed: int | None = None,
        require_balanced: bool = False,
    ) -> Panel:
        """Read a pandas/polars frame or a column mapping as a panel, or say exactly why not.

        An object column, which is how pandas hands over text, holds values of one type that
        cannot change in place and whose text names them: ``str``, ``bytes``, ``bool``, ``int``,
        ``float``, ``complex``, ``Decimal``, ``Fraction``, ``UUID``, ``date``, ``time``,
        ``datetime`` or ``timedelta``, a subclass of one (a pandas ``Timestamp`` is a
        ``datetime``), or NumPy's scalar of one of these kinds. Each value is hashed by its type and
        its text. A ``float``, a ``complex`` or a ``Decimal``, NumPy's among them, is refused when
        it is nan or infinite, as in a float column. A value that can change in place --- a dict, a
        list, a set, a bytearray, an array, an object of the caller's own class --- is refused
        rather than copied, since a copy would still change through ``panel[name][row]``. So are
        ``None`` and pandas' ``NA``, which is how a frame hands over a missing value, and any other
        type, a tuple or a pandas ``Period`` among them: fill or drop a missing value, and convert
        any other value to text, or a ``Period`` to a timestamp. A masked array's masked cell is a
        missing value too, refused by :func:`chc.frames.as_columns`.

        A column of text, dates, times, durations or complex numbers is held, as a label is, and
        so is a number past float64's range. Each is refused where it is read as numbers, by
        :meth:`wide` and by :func:`chc.decision.prescribe` among others: a column is read as
        numbers only where its dtype is boolean, integer or floating, or its values are ``bool``,
        ``int``, ``float``, ``Decimal`` or ``Fraction``, NumPy's among them, and each is finite in
        float64 where it is finite in its own type.

        Args:
            unit, time: the column names holding the entity and the period. Values may be any
                sortable type; periods are ranked, not assumed to be ``0..T-1``.
            frequency: the calendar the periods are read on, which says which two periods are
                consecutive. For dates, ``"D"``, ``"W"``, ``"M"``, ``"Q"`` or ``"Y"``: a row's
                period is the day, week, month, quarter or year its stamp falls in, by numpy's
                calendar units, a week running from a Thursday as 1970-01-01 did, a quarter being
                three months from January. For numbers, a positive step: a period is a whole
                number of steps from the first. Either way the rows of one period share one label,
                the earliest stamp logged in it. ``"observed"`` reads the periods in the order
                logged, as consecutive where no period lies between them. Undeclared, numbers and
                dates on a uniform grid are read on it, and other periods in the order logged;
                dates off a grid, calendar months among them, warn, and are refused from 0.16.
            cluster: an optional column naming the group cluster-robust inference should use ---
                declared here rather than at the call site, because it is a property of the sampling
                design and not of the estimator.
            seed: the generator seed, when the data is simulated. Recorded in :class:`Provenance`
                beside the ``x64`` flag, which is the other half of what a re-draw needs.
            require_balanced: fail at construction if some unit is missing some period, rather than
                at the first :meth:`wide` call. Off by default: unbalanced panels are ordinary, and
                the routines that cannot take one say so themselves.

        Raises:
            PanelError: for a missing index column, one column named as both the unit and the
                time, a non-1-D or ragged column, a non-finite number in a float, a complex or an
                object column, a missing date, a duplicated ``(unit, time)`` pair, an object column
                whose values are not of one type (a missing value among strings, say), whose values'
                text is their address in memory, or whose values' type a panel does not hold, a
                frequency the time column cannot be read in, a period off the declared step, a unit
                with two rows in one declared period, or --- under ``require_balanced`` --- a hole.
                Every message names the column and the offending entity.
            ValueError: from :func:`chc.frames.as_columns`, for two columns of one name or a
                masked cell, naming the column and its row.
        """
        raw = as_columns(data)
        # copied before it is checked, so that the bytes checked and hashed are the bytes held
        columns = {name: np.array(column, copy=True) for name, column in raw.items()}
        for role, name in (("unit", unit), ("time", time), ("cluster", cluster)):
            if name is not None and name not in columns:
                raise PanelError(
                    f"{role} column {name!r} is not in the frame; columns are {sorted(columns)}"
                )
        if unit == time:
            raise PanelError(
                f"unit and time are both {unit!r}; a panel is indexed by a unit and a period, two "
                "columns, and one column read as both gives each unit one period of its own"
            )
        if not columns:
            raise PanelError("the frame has no columns")

        n_rows = len(columns[unit])
        for name, column in columns.items():
            if column.ndim != 1:
                raise PanelError(f"column {name!r} is {column.ndim}-D; a panel column must be 1-D")
            if len(column) != n_rows:
                raise PanelError(
                    f"column {name!r} has {len(column)} rows but {unit!r} has {n_rows}"
                )

        units, times = columns[unit], columns[time]
        # the values before the index, whose check fails with a bare TypeError on a unit or a
        # period that is not hashable: a list, or a Decimal's signalling nan
        for name, column in columns.items():
            if column.dtype != object or not n_rows:
                continue
            values = column.tolist()
            kind = type(values[0])
            odd = next((row for row, value in enumerate(values) if type(value) is not kind), None)
            if odd is not None:
                label, period = units.tolist()[odd], times.tolist()[odd]
                raise PanelError(
                    f"column {name!r} is {values[odd]!r} for unit {label!r} at time {period!r}: "
                    f"type {type(values[odd]).__name__}, where row 0 is type {kind.__name__}; an "
                    "object column holds values of one type, so fill or drop a missing value"
                )
            if not issubclass(kind, _HELD):
                label, period = units.tolist()[0], times.tolist()[0]
                where = f"column {name!r} is {values[0]!r} for unit {label!r} at time {period!r}"
                if " at 0x" in str(values[0]):
                    raise PanelError(
                        f"{where}: {kind.__name__} values, whose text is their address in memory, "
                        "so no other run could hash them the same"
                    )
                raise PanelError(
                    f"{where}: type {kind.__name__}, which a panel does not hold; an object column "
                    "holds values that cannot change in place, of type str, bytes, bool, int, "
                    "float, complex, Decimal, Fraction, UUID, date, time, datetime or timedelta, "
                    "or NumPy's scalars of these: fill or drop a missing value, and convert any "
                    "other value to one of them"
                )

        for name, column in columns.items():
            if np.issubdtype(column.dtype, np.inexact) or column.dtype.kind in "mM":
                finite = np.isfinite(column)
            elif column.dtype == object and n_rows and isinstance(column[0], _INEXACT):
                finite = np.array([_finite(value) for value in column.tolist()], dtype=bool)
            else:
                continue
            bad = np.flatnonzero(~finite)
            if bad.size:
                row = int(bad[0])
                raise PanelError(
                    f"column {name!r} is {column[row]} for unit {units[row]!r} at time "
                    f"{times[row]!r} ({bad.size} of {n_rows} rows are not finite)"
                )

        seen: dict[tuple[Any, Any], int] = {}
        for row, key in enumerate(zip(units.tolist(), times.tolist(), strict=True)):
            if key in seen:
                raise PanelError(
                    f"unit {key[0]!r} appears twice at time {key[1]!r} (rows {seen[key]} and "
                    f"{row}); a panel is indexed by (unit, time), so one of them would be lost"
                )
            seen[key] = row

        frequency = _frequency(times, frequency, time)
        if frequency is not None and frequency != "observed":
            first: dict[tuple[Any, int], int] = {}
            periods = _ordinals(times, frequency).tolist()
            for row, key in enumerate(zip(units.tolist(), periods, strict=True)):
                if key in first:
                    raise PanelError(
                        f"unit {key[0]!r} has rows {first[key]} and {row} in one period of "
                        f"frequency {frequency!r}, at {times[first[key]]} and {times[row]}; a "
                        "panel is indexed by (unit, period), so one of them would be lost"
                    )
                first[key] = row
        elif frequency is None and times.dtype.kind == "M" and _grid_steps(times) is None:
            warnings.warn(
                f"time column {time!r} holds dates on no uniform grid, so its periods are read in "
                "the order logged and a period no unit logged is not seen; declare their calendar "
                "with frequency='D', 'W', 'M', 'Q' or 'Y', or frequency='observed' to keep this "
                "reading. Undeclared, such dates are refused from chc 0.16",
                FutureWarning,
                stacklevel=2,
            )

        panel = cls(
            columns=_Columns(columns),
            unit=unit,
            time=time,
            cluster=cluster,
            provenance=Provenance(
                data_sha256=_fingerprint(columns, frequency),
                chc_version=installed_version(),
                n_rows=n_rows,
                columns=tuple(sorted(columns)),
                x64=_x64_enabled(),
                seed=seed,
                frequency=frequency,
            ),
        )
        if require_balanced and not panel.is_balanced:
            missing = panel._first_hole()
            raise PanelError(
                f"panel is unbalanced: unit {missing[0]!r} has no row at time {missing[1]!r} "
                f"({panel.n_units} units x {panel.n_periods} periods needs "
                f"{panel.n_units * panel.n_periods} rows, got {panel.provenance.n_rows})"
            )
        return panel

    # ---- shape ----------------------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        """Every column name, in the frame's own order."""
        return tuple(self.columns)

    @property
    def units(self) -> tuple[Any, ...]:
        """The distinct unit labels, sorted --- the row order of :meth:`wide`."""
        return tuple(sorted(set(self.columns[self.unit].tolist())))

    @property
    def periods(self) -> tuple[Any, ...]:
        """The distinct period labels, sorted --- the column order of :meth:`wide`."""
        return tuple(sorted(set(self._labels())))

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    @property
    def is_balanced(self) -> bool:
        """Whether every unit is observed in every period, which :meth:`wide` requires."""
        return self.provenance.n_rows == self.n_units * self.n_periods

    def __getitem__(self, name: str) -> NDArray[Any]:
        if name not in self.columns:
            raise KeyError(f"no column {name!r}; columns are {sorted(self.columns)}")
        return self.columns[name]

    # ---- conversion -----------------------------------------------------------------------

    def codes(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Row-aligned integer codes into :attr:`units` and :attr:`periods`.

        What the estimators that group by entity want: ``chc.regret``'s fold builders and the
        cluster-robust variance routines index by position, and deriving the positions here means
        one ranking convention rather than one per caller.
        """
        unit_rank = {label: index for index, label in enumerate(self.units)}
        time_rank = {label: index for index, label in enumerate(self.periods)}
        units = np.array([unit_rank[u] for u in self.columns[self.unit].tolist()], dtype=np.int64)
        times = np.array([time_rank[t] for t in self._labels()], dtype=np.int64)
        return units, times

    def wide(self, name: str) -> NDArray[np.float64]:
        """Column ``name`` as the dense ``(n_units, n_periods)`` float64 matrix DiD and SCM take.

        Float64 regardless of the input dtype and of JAX's ``x64`` flag, matching
        :mod:`chc.did` and :mod:`chc.scm`, which are NumPy-float64 by contract so that an estimate
        does not move with a global flag.

        Raises:
            PanelError: if the column does not hold real numbers --- text, dates, times,
                durations, complex numbers, any other object --- or holds one that is finite in its
                own type and an infinity in float64, naming the first such value's unit and time;
                or if the panel is unbalanced, naming the first missing ``(unit, time)``.
        """
        values = self._numbers(name)
        if not self.is_balanced:
            missing = self._first_hole()
            raise PanelError(
                f"cannot reshape {name!r} to (n_units, n_periods): unit {missing[0]!r} has no row "
                f"at time {missing[1]!r}. Fill or drop the gaps, or use the long columns directly"
            )
        units, times = self.codes()
        out = np.empty((self.n_units, self.n_periods), dtype=np.float64)
        out[units, times] = values
        return out

    def _numbers(self, name: str) -> NDArray[np.float64]:
        """Column ``name`` as float64 in row order, where it holds real numbers
        (:func:`chc.frames._not_numbers`): the one way the library reads a panel's column as
        numbers.

        Raises:
            PanelError: naming the column, the value refused, its unit and its time.
        """
        column = self[name]
        problem = _not_numbers(column)
        if problem is not None:
            row, why = problem
            raise PanelError(
                f"column {name!r} is {column[row]!r} for unit {self.columns[self.unit][row]!r} at "
                f"time {self.columns[self.time][row]!r}: {why}"
            )
        return np.asarray(column, dtype=np.float64)

    def _labels(self) -> list[Any]:
        """Each row's period label: its stamp, or under a declared calendar or step the earliest
        stamp logged in its period, so that the rows of one period share one label."""
        column = self.columns[self.time]
        frequency = self.provenance.frequency
        if frequency is None or frequency == "observed":
            return column.tolist()
        periods = _ordinals(column, frequency)
        _, inverse = np.unique(periods, return_inverse=True)
        order = np.lexsort((column, periods))  # by period, then by stamp
        earliest = order[np.unique(periods[order], return_index=True)[1]]
        return column[earliest][inverse].tolist()

    def _first_hole(self) -> tuple[Any, Any]:
        present = set(zip(self.columns[self.unit].tolist(), self._labels(), strict=True))
        for label in self.units:
            for period in self.periods:
                if (label, period) not in present:
                    return label, period
        raise AssertionError("panel is balanced; _first_hole must not be called")


def _frequency(column: NDArray[Any], frequency: object, name: str) -> str | float | None:
    """``frequency`` as :class:`Provenance` records it, a step as a float, once the time column
    can be read in it."""
    if frequency is None:
        return None
    if frequency == "observed":
        return "observed"
    if isinstance(frequency, str):
        if frequency not in _CALENDAR:
            raise PanelError(
                f"frequency {frequency!r} is none of {', '.join(map(repr, _CALENDAR))}, a positive "
                "step or 'observed'"
            )
        if column.dtype.kind != "M":
            raise PanelError(
                f"frequency {frequency!r} reads dates, and time column {name!r} holds "
                f"{column.dtype}; numbers take a positive step"
            )
        return frequency
    step = (
        float(frequency)
        if isinstance(frequency, numbers.Real) and not isinstance(frequency, bool | np.bool_)
        else math.nan
    )
    if not 0.0 < step < math.inf:
        raise PanelError(
            f"frequency {frequency!r} is no positive step, nor one of "
            f"{', '.join(map(repr, _CALENDAR))} or 'observed'"
        )
    if column.dtype.kind not in "iuf":
        raise PanelError(
            f"a step of {frequency} reads numbers, and time column {name!r} holds {column.dtype}; "
            f"dates take one of {', '.join(map(repr, _CALENDAR))}"
        )
    if column.size:
        off = np.flatnonzero(_whole_steps(column - column.min(), step)[1])
        if off.size:
            row = int(off[0])
            raise PanelError(
                f"time column {name!r} is {column[row]} at row {row}, which is no whole number of "
                f"steps of {step:g} from its first period, {column.min()}"
            )
    return step


def _whole_steps(offsets: NDArray[Any], step: float) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Each offset as a whole number of steps, and where it is not one: exactly for whole numbers
    over a whole step, to a part in 1e9 otherwise."""
    if offsets.dtype.kind in "iu" and float(step).is_integer():
        whole, rest = np.divmod(offsets, int(step))
        return whole.astype(np.int64), rest != 0
    ratio = offsets / step
    whole = np.rint(ratio)
    return whole.astype(np.int64), ~np.isclose(ratio, whole, rtol=1e-9, atol=1e-9)


def _ordinals(column: NDArray[Any], frequency: str | float) -> NDArray[np.int64]:
    """Each row's period under a declared calendar or step: a date's count of days, weeks, months,
    quarters or years since 1970, a number's count of steps from the first period."""
    if isinstance(frequency, str):
        unit = "M" if frequency == "Q" else frequency
        counted = column.astype(f"datetime64[{unit}]").astype(np.int64)
        return counted // 3 if frequency == "Q" else counted
    if not column.size:
        return np.zeros(0, dtype=np.int64)
    return _whole_steps(column - column.min(), frequency)[0]


def _grid_steps(column: NDArray[Any]) -> NDArray[np.int64] | None:
    """Each row's period as a whole number of the least spacing between the periods logged, where
    every period is one; None for text and for periods off such a grid."""
    if column.dtype.kind == "M":  # any resolution: `periods` holds ints at ns, datetimes above
        column = column.astype(np.int64)
    if column.dtype.kind not in "iuf":
        return None
    distinct = np.unique(column)
    if distinct.size < 2:
        return np.zeros(column.size, dtype=np.int64)
    whole, off = _whole_steps(column - distinct[0], np.min(np.diff(distinct)))
    return None if off.any() else whole


def installed_version() -> str:
    """The version of the *installed* distribution, which is the only one that can be reproduced.

    Read from package metadata rather than from a literal in the source, because a literal is a
    second copy of `pyproject.toml`'s `version` and second copies drift: `chc.__version__` sat at
    `0.3.0` through the 0.4.0 and 0.5.0 releases, so every `Provenance` written by a caller reading
    it named a version that did not produce the numbers beside it.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("causal-hybrid-control")
    except PackageNotFoundError:  # running from a source tree with no installed distribution
        return "unknown"


def _x64_enabled() -> bool:
    """Whether JAX is in double precision, asked in a way a type checker can resolve.

    ``jax.config.jax_enable_x64`` is injected onto ``Config`` at import time rather than declared on
    it, so whether a checker sees it depends on the jax version the lockfile resolves -- it does on
    Python 3.14 here and does not on 3.11, which is a red CI job for a green local run.
    ``canonicalize_dtype`` is a declared public function and answers the same question by
    construction: with x64 off, ``float64`` canonicalises down to ``float32``.
    """
    import jax
    import numpy as onp

    return jax.dtypes.canonicalize_dtype(onp.float64) == onp.float64
