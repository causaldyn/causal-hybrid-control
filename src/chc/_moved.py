"""Serve a name from the path it had before 0.19.0 moved it into `chc.research`, with a warning.

An old path still imports without a warning, so `import chc`, which imports it, stays silent and
does not load `chc.research`. Reading a moved name from it imports the new path and warns at the
reader's line, naming the path to import the name from instead. The old paths, this module and
the warning go in 0.20.0 (docs/adr/0117).
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable, Collection

MODULES: dict[str, str] = {
    f"chc.{name}": f"chc.research.{name}"
    for name in (
        "benchmark",
        "causal_bench",
        "deep_galerkin",
        "epidemic",
        "flagship",
        "galerkin",
        "koopman",
        "lalonde",
        "meanfield",
        "mintime",
        "mmm",
        "sensitivity",
        "spine",
        "surrogate",
        "symbolic",
        "transport",
    )
}
"""Each module that moved whole, from its old path to its new one."""


def _served(old: str, new: str, name: str) -> object:
    warnings.warn(
        f"{old}.{name} leaves {old} in 0.20.0: import it from {new}",
        DeprecationWarning,
        stacklevel=3,
    )
    return getattr(importlib.import_module(new), name)


def serve_module(old: str) -> Callable[[str], object]:
    """The module ``__getattr__`` of ``old``, a module that moved whole: its public names."""
    new = MODULES[old]

    def __getattr__(name: str) -> object:
        if name.startswith("_") or not hasattr(importlib.import_module(new), name):
            raise AttributeError(f"module {old!r} has no attribute {name!r}")
        return _served(old, new, name)

    return __getattr__


def serve_names(old: str, new: str, names: Collection[str]) -> Callable[[str], object]:
    """The module ``__getattr__`` of ``old``, a module that kept its other names: ``names``."""

    def __getattr__(name: str) -> object:
        if name not in names:
            raise AttributeError(f"module {old!r} has no attribute {name!r}")
        return _served(old, new, name)

    return __getattr__
