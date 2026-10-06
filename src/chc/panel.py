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
given, made once, so that the data cannot change under its hash; passing one around costs what
passing a dict does.

:class:`Provenance` travels with it. A number is reproducible only together with the bytes it came
from and the precision it was computed in, and this library has already been bitten by the second:
JAX's ``x64`` flag changes which sample a seed draws, so a seed alone does not name a dataset. The
hash is over the column bytes, so two panels that agree numerically but differ in dtype hash
differently --- which is the honest answer, because they will not produce the same numbers. An
object column's bytes are pointers, which no two runs share, so it is hashed by its values: each
value's type and text, each with its length.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chc.frames import ColumnData, as_columns


class PanelError(ValueError):
    """A frame could not be read as a panel; the message names the column and the entity."""


@dataclass(frozen=True)
class Provenance:
    """What a number would need beside it to be reproduced: the bytes, the version, the precision.

    ``x64`` is not decoration. JAX's double-precision flag changes how many bits a threefry key
    spends per element, so the *same* seed draws a *different* sample at the two settings; a run
    recorded without it cannot be repeated. See ``docs/concepts/dtype-policy.md``.
    """

    data_sha256: str  # over column name, dtype, shape and bytes or values, in sorted name order
    chc_version: str
    n_rows: int
    columns: tuple[str, ...]
    x64: bool  # jax.config.jax_enable_x64 at the moment the panel was built
    seed: int | None = None  # the generator's seed where the data is simulated; None for observed

    def to_json(self) -> dict[str, Any]:
        """A plain dict, ready for ``json.dumps`` beside the result it describes."""
        return {
            "data_sha256": self.data_sha256,
            "chc_version": self.chc_version,
            "n_rows": self.n_rows,
            "columns": list(self.columns),
            "x64": self.x64,
            "seed": self.seed,
        }


def _fingerprint(columns: Mapping[str, NDArray[Any]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(columns):
        array = np.ascontiguousarray(columns[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(_values(array) if array.dtype == object else array.tobytes())
    return digest.hexdigest()


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
        cluster: str | None = None,
        seed: int | None = None,
        require_balanced: bool = False,
    ) -> Panel:
        """Read a pandas/polars frame or a column mapping as a panel, or say exactly why not.

        Args:
            unit, time: the column names holding the entity and the period. Values may be any
                sortable type; periods are ranked, not assumed to be ``0..T-1``.
            cluster: an optional column naming the group cluster-robust inference should use ---
                declared here rather than at the call site, because it is a property of the sampling
                design and not of the estimator.
            seed: the generator seed, when the data is simulated. Recorded in :class:`Provenance`
                beside the ``x64`` flag, which is the other half of what a re-draw needs.
            require_balanced: fail at construction if some unit is missing some period, rather than
                at the first :meth:`wide` call. Off by default: unbalanced panels are ordinary, and
                the routines that cannot take one say so themselves.

        Raises:
            PanelError: for a missing index column, a non-1-D or ragged column, a non-finite value,
                a duplicated ``(unit, time)`` pair, an object column whose values are not of one
                type (a missing value among strings, say) or whose values' text is their address in
                memory, or --- under ``require_balanced`` --- a hole. Every message names the
                column and the offending entity.
        """
        raw = as_columns(data)
        # copied before it is checked, so that the bytes checked and hashed are the bytes held
        columns = {name: np.array(column, copy=True) for name, column in raw.items()}
        for role, name in (("unit", unit), ("time", time), ("cluster", cluster)):
            if name is not None and name not in columns:
                raise PanelError(
                    f"{role} column {name!r} is not in the frame; columns are {sorted(columns)}"
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
        seen: dict[tuple[Any, Any], int] = {}
        for row, key in enumerate(zip(units.tolist(), times.tolist(), strict=True)):
            if key in seen:
                raise PanelError(
                    f"unit {key[0]!r} appears twice at time {key[1]!r} (rows {seen[key]} and "
                    f"{row}); a panel is indexed by (unit, time), so one of them would be lost"
                )
            seen[key] = row

        for name, column in columns.items():
            if not np.issubdtype(column.dtype, np.floating):
                continue
            bad = np.flatnonzero(~np.isfinite(column))
            if bad.size:
                row = int(bad[0])
                raise PanelError(
                    f"column {name!r} is {column[row]} for unit {units[row]!r} at time "
                    f"{times[row]!r} ({bad.size} of {n_rows} rows are not finite)"
                )

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
            if kind is not str and " at 0x" in str(values[0]):
                raise PanelError(
                    f"column {name!r} holds {kind.__name__} values, whose text is their address in "
                    f"memory ({values[0]}), so no other run could hash them the same"
                )

        panel = cls(
            columns=_Columns(columns),
            unit=unit,
            time=time,
            cluster=cluster,
            provenance=Provenance(
                data_sha256=_fingerprint(columns),
                chc_version=installed_version(),
                n_rows=n_rows,
                columns=tuple(sorted(columns)),
                x64=_x64_enabled(),
                seed=seed,
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
        return tuple(sorted(set(self.columns[self.time].tolist())))

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
        times = np.array([time_rank[t] for t in self.columns[self.time].tolist()], dtype=np.int64)
        return units, times

    def wide(self, name: str) -> NDArray[np.float64]:
        """Column ``name`` as the dense ``(n_units, n_periods)`` float64 matrix DiD and SCM take.

        Float64 regardless of the input dtype and of JAX's ``x64`` flag, matching
        :mod:`chc.did` and :mod:`chc.scm`, which are NumPy-float64 by contract so that an estimate
        does not move with a global flag.

        Raises:
            PanelError: if the panel is unbalanced, naming the first missing ``(unit, time)``.
        """
        column = self[name]
        if not self.is_balanced:
            missing = self._first_hole()
            raise PanelError(
                f"cannot reshape {name!r} to (n_units, n_periods): unit {missing[0]!r} has no row "
                f"at time {missing[1]!r}. Fill or drop the gaps, or use the long columns directly"
            )
        units, times = self.codes()
        out = np.empty((self.n_units, self.n_periods), dtype=np.float64)
        out[units, times] = np.asarray(column, dtype=np.float64)
        return out

    def _first_hole(self) -> tuple[Any, Any]:
        present = set(
            zip(self.columns[self.unit].tolist(), self.columns[self.time].tolist(), strict=True)
        )
        for label in self.units:
            for period in self.periods:
                if (label, period) not in present:
                    return label, period
        raise AssertionError("panel is balanced; _first_hole must not be called")


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
