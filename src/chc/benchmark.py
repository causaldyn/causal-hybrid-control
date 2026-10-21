"""Moved to `chc.research.benchmark` in 0.19.0, and served here with a warning until 0.20.0."""
# Each name in __all__ comes from the star-import that only type checkers read.
# ruff: noqa: F405

from typing import TYPE_CHECKING

# Listed, so that a star-import reads each name through __getattr__, and warns, and dir()
# shows them.
__all__ = [
    "BenchmarkTask",
    "CausalDynamicsTask",
    "ConfoundingRobustTask",
    "DelayOscillationTask",
    "InventoryTask",
    "ModelUncertaintyTask",
    "MultiSeedResult",
    "PricingTask",
    "SupportShiftTask",
    "TaskResult",
    "leaderboard",
    "leaderboard_multiseed",
    "run_multiseed",
]

if TYPE_CHECKING:
    from chc.research.benchmark import *  # noqa: F403
else:
    from chc import _moved

    __getattr__ = _moved.serve_module(__name__)


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
