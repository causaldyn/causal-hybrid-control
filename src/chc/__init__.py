"""Causal Hybrid Control: hybrid dynamics + a learned causal residual + constrained control.

Current spine (v0.0.1): hybrid dynamics, RK4 rollout, a hand-written discrete adjoint (verified
against autodiff and finite differences), projected-gradient optimal control, MPC, causal
identification behind a pluggable estimator interface, pessimism/support, off-policy evaluation,
and the oracle-regret benchmark.
"""

from __future__ import annotations

from chc.adjoint import control_gradient_adjoint
from chc.causal import (
    ConfoundedLinearSystem,
    estimate_control_effect,
    estimate_effect_dml,
    estimate_effect_iv,
    refute_effect,
    sensitivity_analysis,
)
from chc.control import project_box, projected_gradient_control
from chc.cost import QuadraticCost, total_cost
from chc.deep_galerkin import ScalarMLP, solve_poisson_dgm
from chc.discovery import LaggedGraph, TigramiteDiscovery, discover_lagged_parents
from chc.dynamics import DampedOscillator, Dynamics, HybridDynamics, LinearDynamics
from chc.estimators import (
    IV2SLS,
    BackdoorOLS,
    CausalEffectEstimator,
    DoubleML,
    DoWhyEstimator,
    EconMLDoubleML,
    EffectEstimate,
)
from chc.galerkin import poisson_1d, poisson_2d, thomas_solve
from chc.games import project_simplex, softmax_congestion_equilibrium, stackelberg_allocation
from chc.independence import partial_corr_test
from chc.integrate import rk4_step, rollout
from chc.irf import innovations, local_projection_irf, structured_irf
from chc.koopman import KoopmanModel, koopman_controller, koopman_lqr_gain
from chc.lqr import (
    continuous_lqr,
    dlqr_feedback_controls,
    finite_horizon_dlqr,
    linearize_continuous,
    linearize_discrete,
)
from chc.meanfield import MeanFieldControl
from chc.mpc import mpc_control
from chc.network_causal import ConfoundedNetworkSystem, estimate_network_effects
from chc.offpolicy import GaussianPolicy, fit_behavior_policy, off_policy_value
from chc.residual import GraphResidual, KANResidual, MLPResidual, ZeroResidual
from chc.splitting import (
    exact_linear_flow,
    lie_trotter_step,
    residual_flow,
    strang_marchuk_step,
)
from chc.support import SupportModel, pessimistic_control
from chc.toeplitz import (
    gohberg_semencul_apply,
    gohberg_semencul_generators,
    levinson_durbin,
    sample_autocorrelation,
    solve_toeplitz,
    toeplitz_matvec,
)
from chc.train import fit_residual, fit_residual_multistep, one_step_mse, rollout_mse
from chc.transport import MeanFieldTransport, solve_transport, transport_step

__version__ = "0.0.1"

__all__ = [
    "IV2SLS",
    "BackdoorOLS",
    "CausalEffectEstimator",
    "ConfoundedLinearSystem",
    "ConfoundedNetworkSystem",
    "DampedOscillator",
    "DoWhyEstimator",
    "DoubleML",
    "Dynamics",
    "EconMLDoubleML",
    "EffectEstimate",
    "GaussianPolicy",
    "GraphResidual",
    "HybridDynamics",
    "KANResidual",
    "KoopmanModel",
    "LaggedGraph",
    "LinearDynamics",
    "MLPResidual",
    "MeanFieldControl",
    "MeanFieldTransport",
    "QuadraticCost",
    "ScalarMLP",
    "SupportModel",
    "TigramiteDiscovery",
    "ZeroResidual",
    "__version__",
    "continuous_lqr",
    "control_gradient_adjoint",
    "discover_lagged_parents",
    "dlqr_feedback_controls",
    "estimate_control_effect",
    "estimate_effect_dml",
    "estimate_effect_iv",
    "estimate_network_effects",
    "exact_linear_flow",
    "finite_horizon_dlqr",
    "fit_behavior_policy",
    "fit_residual",
    "fit_residual_multistep",
    "gohberg_semencul_apply",
    "gohberg_semencul_generators",
    "innovations",
    "koopman_controller",
    "koopman_lqr_gain",
    "levinson_durbin",
    "lie_trotter_step",
    "linearize_continuous",
    "linearize_discrete",
    "local_projection_irf",
    "mpc_control",
    "off_policy_value",
    "one_step_mse",
    "partial_corr_test",
    "pessimistic_control",
    "poisson_1d",
    "poisson_2d",
    "project_box",
    "project_simplex",
    "projected_gradient_control",
    "refute_effect",
    "residual_flow",
    "rk4_step",
    "rollout",
    "rollout_mse",
    "sample_autocorrelation",
    "sensitivity_analysis",
    "softmax_congestion_equilibrium",
    "solve_poisson_dgm",
    "solve_toeplitz",
    "solve_transport",
    "stackelberg_allocation",
    "strang_marchuk_step",
    "structured_irf",
    "thomas_solve",
    "toeplitz_matvec",
    "total_cost",
    "transport_step",
]
