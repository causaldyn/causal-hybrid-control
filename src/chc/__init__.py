"""Causal Hybrid Control: hybrid dynamics + a learned causal residual + constrained control.

Current spine (v0.0.1): hybrid dynamics, RK4 rollout, a hand-written discrete adjoint (verified
against autodiff and finite differences), and projected-gradient optimal control. Causal
identification, pessimism, MPC, and the benchmark are on the roadmap.
"""

from __future__ import annotations

from chc.adjoint import control_gradient_adjoint
from chc.control import project_box, projected_gradient_control
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import DampedOscillator, Dynamics, HybridDynamics
from chc.integrate import rk4_step, rollout
from chc.lqr import (
    continuous_lqr,
    dlqr_feedback_controls,
    finite_horizon_dlqr,
    linearize_continuous,
    linearize_discrete,
)
from chc.residual import MLPResidual, ZeroResidual

__version__ = "0.0.1"

__all__ = [
    "DampedOscillator",
    "Dynamics",
    "HybridDynamics",
    "MLPResidual",
    "QuadraticCost",
    "ZeroResidual",
    "__version__",
    "continuous_lqr",
    "control_gradient_adjoint",
    "dlqr_feedback_controls",
    "finite_horizon_dlqr",
    "linearize_continuous",
    "linearize_discrete",
    "project_box",
    "projected_gradient_control",
    "rk4_step",
    "rollout",
    "total_cost",
]
