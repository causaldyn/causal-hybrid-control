"""Causal Hybrid Control: hybrid dynamics + a learned causal residual + constrained control.

Current spine (v0.1.0): hybrid dynamics, RK4 rollout, a hand-written discrete adjoint (verified
against autodiff and finite differences), projected-gradient optimal control, MPC, causal
identification behind a pluggable estimator interface, pessimism/support, off-policy evaluation,
and the oracle-regret benchmark.
"""

from __future__ import annotations

from chc.adjoint import control_gradient_adjoint, control_gradient_diffrax, total_cost_diffrax
from chc.causal import (
    ConfoundedLinearSystem,
    e_value,
    estimate_control_effect,
    estimate_effect_dml,
    estimate_effect_iv,
    refute_effect,
    sensitivity_analysis,
)
from chc.control import project_box, projected_gradient_control
from chc.cost import QuadraticCost, total_cost
from chc.deep_galerkin import ScalarMLP, solve_poisson_dgm
from chc.did import GroupTimeATT, callaway_santanna, de_chaisemartin, twoway_fixed_effects_att
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
    RLearner,
)
from chc.galerkin import poisson_1d, poisson_2d, thomas_solve
from chc.games import project_simplex, softmax_congestion_equilibrium, stackelberg_allocation
from chc.gmethods import naive_pooled_effect, sequential_g_formula
from chc.independence import partial_corr_test
from chc.integrate import rk4_step, rollout
from chc.irf import (
    innovations,
    irf_control_sequence,
    local_projection_irf,
    structured_irf,
)
from chc.koopman import KoopmanModel, koopman_controller, koopman_lqr_gain
from chc.lqr import (
    continuous_lqr,
    dlqr_feedback_controls,
    finite_horizon_dlqr,
    linearize_continuous,
    linearize_discrete,
    linearized_regret_certificate,
)
from chc.matching import MarketplaceMatching, SinkhornResult, marketplace_report, sinkhorn
from chc.meanfield import MeanFieldControl
from chc.metrics import overshoot, rise_time, settling_time, steady_state_error
from chc.mintime import (
    BangBangResult,
    bang_bang_control,
    bang_bang_rollout,
    double_integrator_min_time,
    switching_function,
)
from chc.mpc import mpc_control
from chc.network_causal import (
    ConfoundedNetworkSystem,
    NeighbourMessagePassing,
    estimate_network_effects,
    estimate_network_effects_gnn,
)
from chc.offpolicy import GaussianPolicy, fit_behavior_policy, off_policy_value
from chc.regret import (
    RegretCurve,
    certainty_equivalence_gap,
    closed_loop_cost,
    dlqr,
    regret_scaling,
)
from chc.residual import GraphResidual, KANResidual, MLPResidual, ZeroResidual
from chc.scm import SyntheticControlResult, augmented_synthetic_control, synthetic_control
from chc.splitting import (
    exact_linear_flow,
    lie_trotter_step,
    residual_flow,
    strang_marchuk_step,
)
from chc.support import SupportModel, pessimistic_control
from chc.toeplitz import (
    gohberg_semencul_apply,
    gohberg_semencul_covariance,
    gohberg_semencul_generators,
    levinson_durbin,
    sample_autocorrelation,
    solve_toeplitz,
    toeplitz_matvec,
)
from chc.train import fit_residual, fit_residual_multistep, one_step_mse, rollout_mse
from chc.transport import MeanFieldTransport, solve_transport, transport_step
from chc.uncertainty import (
    EnsembleResidual,
    EnsembleUncertainty,
    SplitConformal,
    WassersteinPenalty,
    fit_ensemble,
)

__version__ = "0.1.0"

__all__ = [
    "IV2SLS",
    "BackdoorOLS",
    "BangBangResult",
    "CausalEffectEstimator",
    "ConfoundedLinearSystem",
    "ConfoundedNetworkSystem",
    "DampedOscillator",
    "DoWhyEstimator",
    "DoubleML",
    "Dynamics",
    "EconMLDoubleML",
    "EffectEstimate",
    "EnsembleResidual",
    "EnsembleUncertainty",
    "GaussianPolicy",
    "GraphResidual",
    "GroupTimeATT",
    "HybridDynamics",
    "KANResidual",
    "KoopmanModel",
    "LaggedGraph",
    "LinearDynamics",
    "MLPResidual",
    "MarketplaceMatching",
    "MeanFieldControl",
    "MeanFieldTransport",
    "NeighbourMessagePassing",
    "QuadraticCost",
    "RLearner",
    "RegretCurve",
    "ScalarMLP",
    "SinkhornResult",
    "SplitConformal",
    "SupportModel",
    "SyntheticControlResult",
    "TigramiteDiscovery",
    "WassersteinPenalty",
    "ZeroResidual",
    "__version__",
    "augmented_synthetic_control",
    "bang_bang_control",
    "bang_bang_rollout",
    "callaway_santanna",
    "certainty_equivalence_gap",
    "closed_loop_cost",
    "continuous_lqr",
    "control_gradient_adjoint",
    "control_gradient_diffrax",
    "de_chaisemartin",
    "discover_lagged_parents",
    "dlqr",
    "dlqr_feedback_controls",
    "double_integrator_min_time",
    "e_value",
    "estimate_control_effect",
    "estimate_effect_dml",
    "estimate_effect_iv",
    "estimate_network_effects",
    "estimate_network_effects_gnn",
    "exact_linear_flow",
    "finite_horizon_dlqr",
    "fit_behavior_policy",
    "fit_ensemble",
    "fit_residual",
    "fit_residual_multistep",
    "gohberg_semencul_apply",
    "gohberg_semencul_covariance",
    "gohberg_semencul_generators",
    "innovations",
    "irf_control_sequence",
    "koopman_controller",
    "koopman_lqr_gain",
    "levinson_durbin",
    "lie_trotter_step",
    "linearize_continuous",
    "linearize_discrete",
    "linearized_regret_certificate",
    "local_projection_irf",
    "marketplace_report",
    "mpc_control",
    "naive_pooled_effect",
    "off_policy_value",
    "one_step_mse",
    "overshoot",
    "partial_corr_test",
    "pessimistic_control",
    "poisson_1d",
    "poisson_2d",
    "project_box",
    "project_simplex",
    "projected_gradient_control",
    "refute_effect",
    "regret_scaling",
    "residual_flow",
    "rise_time",
    "rk4_step",
    "rollout",
    "rollout_mse",
    "sample_autocorrelation",
    "sensitivity_analysis",
    "sequential_g_formula",
    "settling_time",
    "sinkhorn",
    "softmax_congestion_equilibrium",
    "solve_poisson_dgm",
    "solve_toeplitz",
    "solve_transport",
    "stackelberg_allocation",
    "steady_state_error",
    "strang_marchuk_step",
    "structured_irf",
    "switching_function",
    "synthetic_control",
    "thomas_solve",
    "toeplitz_matvec",
    "total_cost",
    "total_cost_diffrax",
    "transport_step",
    "twoway_fixed_effects_att",
]
