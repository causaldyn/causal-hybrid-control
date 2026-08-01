"""Accept a columnar frame -- pandas, polars, or a plain mapping -- wherever chc takes named data.

Neither pandas nor polars is a chc dependency, so a frame is recognised *structurally* rather than
by import: both expose ``.columns`` and ``frame[name]``, and every column converts through
``np.asarray``. Estimators call :func:`as_columns` once on entry and index a plain dict afterwards.

The mapping branch is guarded by an ``isinstance`` rather than by duck-typing, because a frame that
is not a ``Mapping`` still survives the mapping idioms with the wrong answer instead of an error:
``dict(polars_frame)`` and ``for name in polars_frame`` both iterate columns *as values*, handing a
caller column data where it asked for column names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike


class ColumnFrame(Protocol):
    """A frame with named columns, indexable by name: pandas, polars, pyarrow, and friends."""

    @property
    def columns(self) -> Sequence[str]: ...

    def __getitem__(self, name: str, /) -> Any: ...


ColumnData = Mapping[str, Any] | ColumnFrame
"""What every chc entry point that takes named data accepts."""


def as_columns(data: ColumnData) -> dict[str, ArrayLike]:
    """Normalise ``data`` to ``{name: column}`` from a mapping or any columnar frame.

    A mapping passes through **unconverted**, which is what keeps this cheap enough to sit on every
    entry point: the library's own callers hand over JAX arrays, and `np.asarray` on those would
    force a device-to-host copy and raise outright on a tracer inside `jax.jit`. Only the frame
    branch materialises, and it stops at NumPy -- precision is the caller's decision, and the two
    consumers disagree on purpose (:mod:`chc.gmethods` is float64 by contract, the estimators follow
    JAX's x64 flag).
    """
    if isinstance(data, Mapping):
        return {str(name): column for name, column in data.items()}
    names = getattr(data, "columns", None)
    if names is None:
        msg = (
            "expected a mapping of column name -> array, or a frame exposing `.columns` "
            f"(pandas / polars), got {type(data).__name__}"
        )
        raise TypeError(msg)
    return {str(name): np.asarray(data[name]) for name in names}
