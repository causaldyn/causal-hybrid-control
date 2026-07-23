"""chc.sensitivity: the confounding-robust control facade -- re-exports + a runnable pipeline."""

import chc.regret as regret
import chc.uncertainty as uncertainty
from chc import sensitivity


def test_facade_reexports_are_the_same_objects_not_reimplementations() -> None:
    # the facade must be a pure re-export surface (no divergent copies)
    assert sensitivity.confounding_robust_control is regret.confounding_robust_control
    assert sensitivity.confounding_robust_inflation is uncertainty.confounding_robust_inflation
    assert sensitivity.msm_worst_case_mean is uncertainty.msm_worst_case_mean
    assert sensitivity.asymmetric_control_improvement is regret.asymmetric_control_improvement


def test_all_advertised_symbols_are_importable() -> None:
    for name in sensitivity.__all__:
        assert hasattr(sensitivity, name), name


def test_estimate_to_radius_to_control_pipeline_runs() -> None:
    # the documented chain: effect estimate -> §32 half-width -> §35 robust gain vs CE
    b_hat, target, gamma = 1.3, 1.0, 2.5
    cvar_gap = (
        b_hat  # effect-scale calibration (as in Result 37); keeps D < b_hat (identified sign)
    )
    halfwidth = sensitivity.confounding_robust_inflation(cvar_gap, 0.0, gamma)
    u_ce = sensitivity.certainty_equivalence_control(b_hat, target)
    u_robust = sensitivity.confounding_robust_control(b_hat, halfwidth, target, 1.0, 4.0)
    assert (
        0.0 < halfwidth < b_hat
    )  # Gamma>1 inflates the radius but the effect sign stays identified
    assert (
        u_robust > u_ce
    )  # undershoot (churn) costlier -> the robust controller pushes the gain up
