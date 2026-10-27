"""Causal Hybrid Control: hybrid dynamics + a learned causal residual + constrained control.

The spine: hybrid dynamics, RK4 rollout, a hand-written discrete adjoint (verified
against autodiff and finite differences), projected-gradient optimal control, MPC, causal
identification behind a pluggable estimator interface, pessimism/support, off-policy evaluation,
and the oracle-regret benchmark. On top of it: control under hidden confounding, the safety
layer that spends the identification radius on a constraint, and `chc.research.spine`, which runs
the whole chain -- confounded logs to certified plan -- as one decision.

Two distinctions the API deliberately keeps visible rather than smoothing over. Fitting a residual
by prediction error is not identifying it: under a confounded logging policy that recovers the
observational control response, and `chc.dynamics_id` is the module that recovers the interventional
one. And "safety" names three different operations -- **plan** (`causal_plan`, which holds a
barrier in the solve only when handed one), **audit** (`certify_safety`, read-only), **filter**
(`robust_safety_filter`, one action at a time) -- and the audit's verdict is the one to read, even
on a plan solved under the barrier; see `chc.plan` for what each does and does not promise.

The top level holds the lifecycle's names (`docs/lifecycle.md`), the classes a call to one of them
takes, and the errors they raise. Every other name lives at its module path, and imports from
there alone: the 421 others that 0.7.0 bound here left the top level in 1.0 (docs/adr/0016).
"""

from __future__ import annotations

# Every module a top-level name came from is imported, as it was when the top level bound every
# name, so that `import chc` followed by `chc.<module>.<name>` keeps working. A module that moved
# into `chc.research` in 0.19.0 is not: its old path was deleted in 0.20.0 (docs/adr/0128).
# isort: off
from chc import (
    adjoint as adjoint,
    barrier as barrier,
    causal as causal,
    control as control,
    cost as cost,
    decision as decision,
    delay as delay,
    did as did,
    discovery as discovery,
    dynamics as dynamics,
    dynamics_id as dynamics_id,
    estimators as estimators,
    evaluation as evaluation,
    experiment as experiment,
    frames as frames,
    games as games,
    gate as gate,
    gmethods as gmethods,
    graph as graph,
    independence as independence,
    integrate as integrate,
    irf as irf,
    lqr as lqr,
    marketplace as marketplace,
    matching as matching,
    metrics as metrics,
    mpc as mpc,
    network_causal as network_causal,
    offpolicy as offpolicy,
    panel as panel,
    pathway as pathway,
    plan as plan,
    reachability as reachability,
    regret as regret,
    residual as residual,
    scm as scm,
    splitting as splitting,
    support as support,
    switchback as switchback,
    toeplitz as toeplitz,
    train as train,
    uncertainty as uncertainty,
)

# isort: on
from chc.barrier import robust_safety_filter
from chc.causal import e_value, estimate_effect_dml, refute_effect, sensitivity_analysis
from chc.control import LinearConstraint
from chc.cost import QuadraticCost
from chc.decision import (
    Constraint,
    DecisionError,
    Driver,
    Lever,
    NotIdentifiedError,
    Target,
    prescribe,
)
from chc.dynamics import HybridDynamics
from chc.dynamics_id import fit_causal_residual
from chc.evaluation import (
    AffinePolicy,
    AffineSchedule,
    InfeasibleEvaluation,
    InitialLaw,
    LinearGaussianPlant,
    certify_evaluation,
    evaluate_plan,
)
from chc.experiment import ChannelPrior, ZoneDecision, ZoneExperiment, design_experiment
from chc.gate import (
    BlockBatch,
    DecisionLog,
    DeploymentGate,
    DriftAlarm,
    GateConfig,
    MissRateAlarm,
    OnlineRadius,
    ZoneBatch,
    ZonePlan,
    channel_drift_evalues,
    channel_move,
)
from chc.graph import CausalGraph, CyclicGraphError
from chc.irf import delay_estimate
from chc.matching import shadow_price_effect, shadow_price_interval
from chc.mpc import PeriodBudget, RecedingHorizon
from chc.offpolicy import GaussianPolicy, off_policy_value
from chc.panel import Panel, PanelError, Provenance
from chc.panel import installed_version as _installed_version
from chc.plan import (
    BarrierConstraint,
    CausalPlan,
    ChannelSensitivity,
    causal_plan,
    certify_safety,
    plan_regret_bound,
)
from chc.regret import (
    adaptive_exploration_certificate,
    capped_exploration_policy,
    minimax_action,
    optimal_exploration_certificate,
)
from chc.support import SupportModel
from chc.switchback import (
    BlockDesign,
    Horizon,
    MarkovDesign,
    PersistencePrior,
    SwitchbackPlan,
    TargetMDE,
    design_switchback,
    randomisation_interval,
    randomisation_test,
    read_switchback,
    restate_mde,
)
from chc.uncertainty import SplitConformal

__version__ = _installed_version()
"""The installed distribution's version. See :func:`chc.panel.installed_version`."""

__all__ = [
    "AffinePolicy",
    "AffineSchedule",
    "BarrierConstraint",
    "BlockBatch",
    "BlockDesign",
    "CausalGraph",
    "CausalPlan",
    "ChannelPrior",
    "ChannelSensitivity",
    "Constraint",
    "CyclicGraphError",
    "DecisionError",
    "DecisionLog",
    "DeploymentGate",
    "DriftAlarm",
    "Driver",
    "GateConfig",
    "GaussianPolicy",
    "Horizon",
    "HybridDynamics",
    "InfeasibleEvaluation",
    "InitialLaw",
    "Lever",
    "LinearConstraint",
    "LinearGaussianPlant",
    "MarkovDesign",
    "MissRateAlarm",
    "NotIdentifiedError",
    "OnlineRadius",
    "Panel",
    "PanelError",
    "PeriodBudget",
    "PersistencePrior",
    "Provenance",
    "QuadraticCost",
    "RecedingHorizon",
    "SplitConformal",
    "SupportModel",
    "SwitchbackPlan",
    "Target",
    "TargetMDE",
    "ZoneBatch",
    "ZoneDecision",
    "ZoneExperiment",
    "ZonePlan",
    "__version__",
    "adaptive_exploration_certificate",
    "capped_exploration_policy",
    "causal_plan",
    "certify_evaluation",
    "certify_safety",
    "channel_drift_evalues",
    "channel_move",
    "delay_estimate",
    "design_experiment",
    "design_switchback",
    "e_value",
    "estimate_effect_dml",
    "evaluate_plan",
    "fit_causal_residual",
    "minimax_action",
    "off_policy_value",
    "optimal_exploration_certificate",
    "plan_regret_bound",
    "prescribe",
    "randomisation_interval",
    "randomisation_test",
    "read_switchback",
    "refute_effect",
    "restate_mde",
    "robust_safety_filter",
    "sensitivity_analysis",
    "shadow_price_effect",
    "shadow_price_interval",
]
