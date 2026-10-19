"""Moved to `chc.research.deep_galerkin` in 0.19.0, and served here with a warning until 0.20.0."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chc.research.deep_galerkin import *  # noqa: F403
else:
    from chc import _moved

    __getattr__ = _moved.serve_module(__name__)
