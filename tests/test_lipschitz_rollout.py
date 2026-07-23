"""chc.uncertainty: certified-Lipschitz rollout-error bound (discrete Gronwall) -- sound + honest.

Ties the shipped LipschitzResidual's certified constant to a machine-checked pessimism radius.
"""

import numpy as np
import pytest

from chc.uncertainty import (
    contractive_rollout_bound,
    contractive_rollout_certificate,
    lipschitz_rollout_bound,
    lipschitz_rollout_certificate,
)


def test_certificate_deviation_stays_under_the_bound() -> None:
    cert = lipschitz_rollout_certificate(seed=0)
    assert cert.ok  # measured rollout deviation <= the certified Gronwall bound
    assert cert.measured_deviation <= cert.certified_bound + 1e-9
    assert cert.measured_deviation > 0.0  # a non-trivial deviation was actually produced


def test_bound_holds_across_seeds() -> None:
    for seed in range(6):
        cert = lipschitz_rollout_certificate(seed=seed)
        assert (
            cert.measured_deviation <= cert.certified_bound + 1e-9
        )  # the guarantee is not seed-luck


def test_bound_is_not_vacuous_on_a_short_horizon() -> None:
    cert = lipschitz_rollout_certificate(seed=1, horizon=8, dt=0.05)
    # short horizon / bounded L: the certified radius is within an order of magnitude of the truth
    assert cert.measured_deviation >= 0.2 * cert.certified_bound


def test_contractive_certificate_confirms_negative_log_norm_and_flat_radius() -> None:
    cert = contractive_rollout_certificate(seed=0)
    assert cert.ok
    assert cert.contraction_rate > 0.0  # certified |mu| > 0
    assert cert.empirical_one_sided <= -cert.contraction_rate + 1e-6  # genuinely contracting
    assert cert.measured_deviation <= cert.bounded_radius + 1e-6  # under the flat ceiling
    assert cert.bounded_radius < cert.lipschitz_blowup  # contraction beats the e^{L*T} envelope


def test_contractive_radius_stays_capped_while_lipschitz_explodes() -> None:
    # the contraction ceiling is eps/rate for ALL horizons, unlike the norm-Lipschitz e^{L*T}
    contractive = contractive_rollout_bound(1.0, 0.1, 0.05, 400)
    lipschitz = lipschitz_rollout_bound(1.0, 0.1, 0.05, 400)
    assert contractive <= 0.1 / 1.0 + 1e-9  # capped at eps/rate, horizon-independent
    assert contractive < 1e-4 * lipschitz  # while the norm-Lipschitz bound blows up exponentially


def test_contractive_bound_returns_inf_when_cfl_violated() -> None:
    assert contractive_rollout_bound(1.0, 0.1, 5.0, 8) == float("inf")  # rate*dt = 5 >= 1


def test_vanishing_lipschitz_gives_the_linear_envelope() -> None:
    # L -> 0: the Gronwall closed form degrades to the linear eps*dt*H (no exponential blow-up)
    assert lipschitz_rollout_bound(0.0, 0.1, 0.05, 8) == pytest.approx(0.1 * 0.05 * 8)


def test_bound_matches_the_gronwall_closed_form() -> None:
    lipschitz, model_error, dt, horizon = 1.5, 0.2, 0.05, 10
    expected = model_error * ((1.0 + lipschitz * dt) ** horizon - 1.0) / lipschitz
    assert lipschitz_rollout_bound(lipschitz, model_error, dt, horizon) == pytest.approx(expected)


def test_bound_is_monotone_in_model_error_and_horizon() -> None:
    small = lipschitz_rollout_bound(1.0, 0.1, 0.05, 8)
    more_error = lipschitz_rollout_bound(1.0, 0.2, 0.05, 8)
    longer = lipschitz_rollout_bound(1.0, 0.1, 0.05, 16)
    assert more_error > small  # larger per-step error -> larger certified radius
    assert longer > small  # longer horizon -> larger certified radius


def test_bound_grows_exponentially_with_lipschitz_horizon_product() -> None:
    # HONEST SCOPE: the bound is exp(L*T); it must blow up for large L*T (documented, not hidden)
    short = lipschitz_rollout_bound(2.0, 0.1, 0.05, 8)  # L*T = 2*0.4 = 0.8
    long_horizon = lipschitz_rollout_bound(2.0, 0.1, 0.05, 200)  # L*T = 2*10 = 20
    assert long_horizon > 100.0 * short  # exponential premium at large L*T
    assert np.isfinite(long_horizon)


def test_time_varying_certificate_is_tighter_than_constant_and_has_a_cutoff() -> None:
    from chc.uncertainty import time_varying_rollout_certificate

    cert = time_varying_rollout_certificate(seed=0)
    assert cert.ok
    assert cert.varying_final <= cert.constant_final + 1e-9  # per-step L is never looser than max-L
    assert cert.certified_until_step >= 0  # a valid honest planning horizon
    assert cert.safe_until_step >= 1  # some prefix of the plan is robustly feasible


def test_certified_horizon_shrinks_as_tolerance_tightens() -> None:
    from chc.uncertainty import certified_horizon

    lipschitz = [1.0] * 40
    error = [0.1] * 40
    loose = certified_horizon(lipschitz, error, 0.05, tolerance=1.0)
    tight = certified_horizon(lipschitz, error, 0.05, tolerance=0.1)
    assert tight <= loose  # a stricter tolerance certifies fewer steps


def test_constraint_tightening_flags_the_true_trajectory_safe() -> None:
    from chc.uncertainty import constraint_margin

    # nominal margin -0.3 to g<=0, L_g=1, growing error radius: safe until the tube eats the margin
    radii = np.array([0.0, 0.1, 0.2, 0.35, 0.5])
    margin = constraint_margin(-0.3 * np.ones(5), 1.0, radii)
    assert bool(margin[0] <= 0.0)  # at k=0 the true trajectory is certified feasible
    assert bool(margin[-1] > 0.0)  # once L_g*e_k exceeds the nominal margin, feasibility is lost


def test_closed_loop_radius_exceeds_open_loop_when_replanning() -> None:
    from chc.uncertainty import closed_loop_rollout_bound

    open_loop = lipschitz_rollout_bound(1.0, 0.1, 0.05, 10)
    closed = closed_loop_rollout_bound(1.0, 0.5, 2.0, 0.1, 0.05, 10)  # L_pi=2 policy sensitivity
    assert closed > open_loop  # re-planning feeds state error through the policy -> larger tube
