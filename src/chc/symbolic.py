"""Moved to `chc.research.symbolic` in 0.19.0, and served here with a warning until 0.20.0."""
# Each name in __all__ comes from the star-import that only type checkers read.
# ruff: noqa: F405

from typing import TYPE_CHECKING

# Listed, so that a star-import reads each name through __getattr__, and warns, and dir()
# shows them.
__all__ = [
    "LIBRARY",
    "SymbolicEdge",
    "SymbolicExtractionCurve",
    "Vector",
    "extract_symbolic",
    "extract_symbolic_edge",
    "kan_edge",
    "symbolic_extraction_certificate",
]

if TYPE_CHECKING:
    from chc.research.symbolic import *  # noqa: F403
else:
    from chc import _moved

    __getattr__ = _moved.serve_module(__name__)


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
