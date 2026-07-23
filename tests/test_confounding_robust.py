"""chc.uncertainty: confounding-robust (MSM -> CVaR) pessimism -- sharp, tight, monotone.

Closes the Gamma -> CVaR -> control-radius gap: the marginal-sensitivity-model worst-case effect is
a CVaR mixture, and its inflation feeds the pessimism radius (Rocq ``confounding_robust_cvar.v``).
"""

import numpy as np
import pytest

from chc.uncertainty import (
    closed_loop_rollout_bound,
    confounding_robust_certificate,
    confounding_robust_closed_loop_bound,
    confounding_robust_closed_loop_certificate,
    confounding_robust_inflation,
    confounding_robust_radius,
    msm_worst_case_mean,
)


def test_closed_form_matches_the_sharp_brute_force_bound() -> None:
    cert = confounding_robust_certificate(seed=0)
    assert cert.ok
    assert cert.closed_form == pytest.approx(cert.brute_force, abs=1e-12)  # CVaR form == box-LP opt
    assert cert.at_gamma_one == pytest.approx(cert.sample_mean, abs=1e-12)  # tight at Gamma=1
    assert cert.monotone


def test_brute_force_agreement_holds_across_seeds() -> None:
    for seed in range(6):
        cert = confounding_robust_certificate(seed=seed)
        assert cert.closed_form == pytest.approx(cert.brute_force, abs=1e-12)  # not seed-luck


def test_worst_case_reduces_to_the_mean_under_no_confounding() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert msm_worst_case_mean(y, 1.0) == pytest.approx(float(np.mean(y)))  # Gamma=1 -> point ID


def test_worst_case_is_monotone_and_never_optimistic_in_gamma() -> None:
    rng = np.random.default_rng(1)
    y = rng.standard_normal(200)
    mu = float(np.mean(y))
    vals = [msm_worst_case_mean(y, g) for g in (1.0, 1.5, 2.0, 4.0, 8.0)]
    assert all(v >= mu - 1e-12 for v in vals)  # pessimistic: worst-case never below the mean
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))  # wider as Gamma grows


def test_inflation_is_the_gamma_scaled_cvar_gap() -> None:
    # closed form (Gamma-1)/(Gamma+1)*(mhi-mlo); at Gamma=3 the factor is 1/2
    assert confounding_robust_inflation(4.0, 2.0, 3.0) == pytest.approx(0.5 * (4.0 - 2.0))
    assert confounding_robust_inflation(4.0, 2.0, 1.0) == 0.0  # no confounding -> no inflation


def test_inflation_rejects_gamma_below_one() -> None:
    with pytest.raises(ValueError, match="Gamma must be >= 1"):
        confounding_robust_inflation(4.0, 2.0, 0.5)


def test_robust_radius_grows_from_the_nominal_and_is_monotone() -> None:
    rng = np.random.default_rng(2)
    y = rng.standard_normal(200)
    base = 0.3
    r1 = confounding_robust_radius(base, y, 1.0)
    r2 = confounding_robust_radius(base, y, 2.0)
    r3 = confounding_robust_radius(base, y, 5.0)
    assert r1 == pytest.approx(base)  # Gamma=1: radius unchanged
    assert base <= r2 <= r3  # never below nominal; wider with more assumed confounding


# --- Result 34: confounding-robust CLOSED LOOP (§32 radius x §31 replan-tube) ---


def test_closed_loop_bound_recovers_the_plain_tube_at_gamma_one() -> None:
    # Gamma=1: no confounding inflation, so it equals the plain §31 closed-loop tube exactly
    args = (1.0, 0.5, 1.5, 0.05, 0.05, 25)  # state_lip, control_lip, policy_lip, error, dt, horizon
    # (state_lip, control_lip, policy_lip, base_error, control_mag, cvar_gap, gamma, dt, horizon)
    robust = confounding_robust_closed_loop_bound(1.0, 0.5, 1.5, 0.05, 0.8, 0.4, 1.0, 0.05, 25)
    assert robust == pytest.approx(closed_loop_rollout_bound(*args))


def test_closed_loop_bound_widens_monotonically_with_gamma() -> None:
    def bound(g: float) -> float:
        return confounding_robust_closed_loop_bound(1.0, 0.5, 1.5, 0.05, 0.8, 0.4, g, 0.05, 25)

    vals = [bound(g) for g in (1.0, 1.5, 2.0, 3.0, 5.0)]
    assert all(
        vals[i] <= vals[i + 1] for i in range(len(vals) - 1)
    )  # more confounding -> wider tube
    assert bound(3.0) > bound(1.0)  # strictly wider under real confounding


def test_certificate_confirms_shrinking_safe_horizon() -> None:
    cert = confounding_robust_closed_loop_certificate(gamma=3.0)
    assert cert.ok
    assert cert.robust_radius >= cert.nominal_radius  # Gamma=1 recovers §31; Gamma>1 is wider
    assert cert.robust_safe_step <= cert.nominal_safe_step  # confounding erodes the safe horizon
    assert cert.radius_monotone  # tube nondecreasing over the Gamma grid
    assert cert.horizon_monotone  # safe horizon nonincreasing over the Gamma grid
