"""Moved to `chc.research.deep_galerkin` in 0.19.0, and served here with a warning until 0.20.0."""
# Each name in __all__ comes from the star-import that only type checkers read.
# ruff: noqa: F405

from typing import TYPE_CHECKING

# Listed, so that a star-import reads each name through __getattr__, and warns, and dir()
# shows them.
__all__ = [
    "BlindnessSweep",
    "CongestedMeanFieldGame",
    "FieldMLP",
    "FinitePopulationGap",
    "LQMeanFieldGame",
    "MeanFieldCurve",
    "MeanFieldDGM",
    "MeanFieldOptimizer",
    "MeanFieldSolution",
    "NonlinearDwrCurve",
    "PopulationNoise",
    "ScalarMLP",
    "adjoint_weighted_error",
    "dual_weighted_error_estimate",
    "finite_population_gap_certificate",
    "lq_mean_field_certificate",
    "nonlinear_dwr_certificate",
    "residual_blindness_sweep",
    "solve_mfg_dgm",
    "solve_poisson_dgm",
]

if TYPE_CHECKING:
    from chc.research.deep_galerkin import *  # noqa: F403
else:
    from chc import _moved

    __getattr__ = _moved.serve_module(__name__)


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
