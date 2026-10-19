"""Moved to `chc.research.mmm` in 0.19.0, and served here with a warning until 0.20.0."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chc.research.mmm import *  # noqa: F403
else:
    from chc import _moved

    __getattr__ = _moved.serve_module(__name__)
