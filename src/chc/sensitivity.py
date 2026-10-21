"""Moved to `chc.research.sensitivity` in 0.19.0, and served here with a warning until 0.20.0."""
# Each name in __all__ comes from the star-import that only type checkers read.
# ruff: noqa: F405

from typing import TYPE_CHECKING

# Listed, so that a star-import reads each name through __getattr__, and warns, and dir()
# shows them.
__all__ = [
    "BarrierConfoundingCurve",
    "BarrierReachabilityGap",
    "ConfoundingRegretFloorCurve",
    "ConfoundingRobustCertificate",
    "ConfoundingRobustClosedLoopCertificate",
    "ConfoundingRobustControlCurve",
    "ConfoundingRobustLQRegretCurve",
    "ConfoundingRobustPenalty",
    "ConfoundingRobustTask",
    "DynamicConfoundingCurve",
    "GammaBenchmark",
    "GammaBenchmarkCertificate",
    "MarketplaceControlCurve",
    "SafetyCertificate",
    "SafetyFilterBenchmark",
    "admissible_action_interval",
    "asymmetric_control_improvement",
    "barrier_confounding_certificate",
    "barrier_gamma_star",
    "barrier_reachability_gap",
    "benchmark_gamma",
    "certainty_equivalence_control",
    "certify_safety",
    "confounding_regret_floor_certificate",
    "confounding_robust_certificate",
    "confounding_robust_closed_loop_bound",
    "confounding_robust_closed_loop_certificate",
    "confounding_robust_control",
    "confounding_robust_control_benchmark",
    "confounding_robust_control_certificate",
    "confounding_robust_inflation",
    "confounding_robust_lq_regret",
    "confounding_robust_lq_regret_matrix",
    "confounding_robust_radius",
    "confounding_robust_tracking_benchmark",
    "confounding_robust_tracking_loop",
    "control_channel",
    "gamma_benchmark_certificate",
    "identification_radius_threshold",
    "lq_regret_sensitivity",
    "msm_worst_case_mean",
    "negative_control_gamma",
    "robust_barrier_margin",
    "robust_safe_action",
    "robust_safety_filter",
    "safety_filter_benchmark",
    "worst_case_asymmetric_loss",
]

if TYPE_CHECKING:
    from chc.research.sensitivity import *  # noqa: F403
else:
    from chc import _moved

    __getattr__ = _moved.serve_module(__name__)


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
