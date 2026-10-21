"""Moved to `chc.research.galerkin` in 0.19.0, and served here with a warning until 0.20.0."""
# Each name in __all__ comes from the star-import that only type checkers read.
# ruff: noqa: F405

from typing import TYPE_CHECKING

# Listed, so that a star-import reads each name through __getattr__, and warns, and dir()
# shows them.
__all__ = [
    "ConvectionDiffusionCurve",
    "convection_diffusion_1d",
    "convection_diffusion_certificate",
    "convection_diffusion_exact",
    "optimal_upwind",
    "poisson_1d",
    "poisson_2d",
    "thomas_solve",
]

if TYPE_CHECKING:
    from chc.research.galerkin import *  # noqa: F403
else:
    from chc import _moved

    __getattr__ = _moved.serve_module(__name__)


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
