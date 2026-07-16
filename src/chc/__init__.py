"""Causal Hybrid Control: hybrid dynamics + a learned causal residual + constrained control.

Current spine (v0.0.1): hybrid dynamics, RK4 rollout, a hand-written discrete adjoint (verified
against autodiff and finite differences), and projected-gradient optimal control. Causal
identification, pessimism, MPC, and the benchmark are on the roadmap.
"""

from __future__ import annotations

from chc.adjoint import control_gradient_adjoint
from chc.causal import ConfoundedLinearSystem, estimate_control_effect
from chc.control import project_box, projected_gradient_control
from chc.cost import QuadraticCost, total_cost
from chc.dynamics import DampedOscillator, Dynamics, HybridDynamics, LinearDynamics
from chc.galerkin import poisson_1d, thomas_solve
from chc.integrate import rk4_step, rollout
from chc.lqr import (
    continuous_lqr,
    dlqr_feedback_controls,
    finite_horizon_dlqr,
    linearize_continuous,
    linearize_discrete,
)
from chc.mpc import mpc_control
from chc.offpolicy import GaussianPolicy, fit_behavior_policy, off_policy_value
from chc.residual import KANResidual, MLPResidual, ZeroResidual
from chc.splitting import (
    exact_linear_flow,
    lie_trotter_step,
    residual_flow,
    strang_marchuk_step,
)
from chc.support import SupportModel, pessimistic_control
from chc.train import fit_residual, one_step_mse

__version__ = "0.0.1"

__all__ = [
    "ConfoundedLinearSystem",
    "DampedOscillator",
    "Dynamics",
    "GaussianPolicy",
    "HybridDynamics",
    "KANResidual",
    "LinearDynamics",
    "MLPResidual",
    "QuadraticCost",
    "SupportModel",
    "ZeroResidual",
    "__version__",
    "continuous_lqr",
    "control_gradient_adjoint",
    "dlqr_feedback_controls",
    "estimate_control_effect",
    "exact_linear_flow",
    "finite_horizon_dlqr",
    "fit_behavior_policy",
    "fit_residual",
    "lie_trotter_step",
    "linearize_continuous",
    "linearize_discrete",
    "mpc_control",
    "off_policy_value",
    "one_step_mse",
    "pessimistic_control",
    "poisson_1d",
    "project_box",
    "projected_gradient_control",
    "residual_flow",
    "rk4_step",
    "rollout",
    "strang_marchuk_step",
    "thomas_solve",
    "total_cost",
]
