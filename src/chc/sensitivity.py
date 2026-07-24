"""Sensitivity-aware robust control under HIDDEN CONFOUNDING -- a facade over the §32-§37 line.

Offline pessimism (``chc.support``, ``chc.uncertainty``) assumes the observed transitions identify
the causal effect. Under hidden confounding they do not: the effect is only *partially* identified
in a sensitivity interval. This module gathers the one "sensitivity-aware robust control" story
whose primitives are otherwise split across two modules by their numerics (``chc.regret`` is
NumPy/SciPy, ``chc.uncertainty`` is JAX). Nothing is moved -- this is a discovery/API surface only.

The pipeline, estimate -> sensitivity radius -> robust control -> validation:

1. **Sensitivity radius (§32, bounded density-ratio / MSM).** Given a treated-outcome sample and a
   sensitivity ``Gamma >= 1`` (the analyst's unfalsifiable input, ``[1/Gamma, Gamma]`` density-ratio
   box), the sharp worst-case effect is a CVaR mixture; :func:`confounding_robust_inflation` /
   :func:`msm_worst_case_mean` give the radius inflation, :func:`confounding_robust_radius` widens a
   pessimism radius (never optimistic, tight at ``Gamma=1``, monotone).
2. **Regret floor (§33).** :func:`confounding_robust_lq_regret` -- the confounding effect-bias
   becomes a control-regret floor SECOND order in the bias (:func:`lq_regret_sensitivity` for the
   toy ``L_reg``; :func:`confounding_robust_lq_regret_matrix` for the multivariate Frobenius lift).
3. **Robust controller (§35).** :func:`confounding_robust_control` -- under an ASYMMETRIC loss
   (over/under-shoot) the radius shifts the gain (sign dichotomy) and beats certainty-equivalence
   (:func:`certainty_equivalence_control`); :func:`asymmetric_control_improvement` is the piecewise
   worst-case-loss gain, :func:`worst_case_asymmetric_loss` the loss itself.
4. **Closed loop (§34).** :func:`confounding_robust_closed_loop_bound` -- the radius feeds the
   replan-tube (``L_x + L_u*L_pi``), the effect error scaled by the control magnitude.
5. **Validation.** :func:`confounding_robust_control_benchmark` grounds it on a synthetic
   observational confounded marketplace; the ``*_certificate`` functions are the machine-checked
   evidence.

Worked example (the calibration is explicit -- NOT baked in)::

    from chc.sensitivity import (
        confounding_robust_inflation, certainty_equivalence_control, confounding_robust_control,
    )

    b_hat, target, gamma = 1.3, 1.0, 2.5   # biased effect estimate, service target, sensitivity
    cvar_gap = b_hat            # <- YOUR effect-scale CVaR-gap calibration; keeps D < b_hat
    D = confounding_robust_inflation(cvar_gap, 0.0, gamma)   # effect half-width Delta(Gamma)
    u_ce = certainty_equivalence_control(b_hat, target)      # trusts the biased estimate
    u_robust = confounding_robust_control(b_hat, D, target, 1.0, 4.0)   # undershoot 4x -> push up

HONEST: ``Gamma`` and the CVaR-gap calibration are the analyst's inputs; the sign-identification
constraint ``b_hat > D > 0`` must hold. These robustify pessimism, they do NOT test for confounding.
See ``discoveries/theorems.md`` §32-§37 for the proofs and scope.
"""

from __future__ import annotations

from chc.regret import (
    ConfoundingRegretFloorCurve,
    ConfoundingRobustControlCurve,
    ConfoundingRobustLQRegretCurve,
    DynamicConfoundingCurve,
    MarketplaceControlCurve,
    asymmetric_control_improvement,
    certainty_equivalence_control,
    confounding_regret_floor_certificate,
    confounding_robust_control,
    confounding_robust_control_benchmark,
    confounding_robust_control_certificate,
    confounding_robust_lq_regret,
    confounding_robust_lq_regret_matrix,
    confounding_robust_tracking_benchmark,
    confounding_robust_tracking_loop,
    lq_regret_sensitivity,
    worst_case_asymmetric_loss,
)
from chc.uncertainty import (
    ConfoundingRobustCertificate,
    ConfoundingRobustClosedLoopCertificate,
    ConfoundingRobustPenalty,
    confounding_robust_certificate,
    confounding_robust_closed_loop_bound,
    confounding_robust_closed_loop_certificate,
    confounding_robust_inflation,
    confounding_robust_radius,
    msm_worst_case_mean,
)

__all__ = [
    "ConfoundingRegretFloorCurve",
    "ConfoundingRobustCertificate",
    "ConfoundingRobustClosedLoopCertificate",
    "ConfoundingRobustControlCurve",
    "ConfoundingRobustLQRegretCurve",
    "ConfoundingRobustPenalty",
    "DynamicConfoundingCurve",
    "MarketplaceControlCurve",
    "asymmetric_control_improvement",
    "certainty_equivalence_control",
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
    "lq_regret_sensitivity",
    "msm_worst_case_mean",
    "worst_case_asymmetric_loss",
]
