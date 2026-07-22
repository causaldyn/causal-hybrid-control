"""chc.uncertainty: certified-Lipschitz rollout-error bound (discrete Gronwall) -- sound + honest.

Ties the shipped LipschitzResidual's certified constant to a machine-checked pessimism radius.
"""

import numpy as np
import pytest

from chc.uncertainty import lipschitz_rollout_bound, lipschitz_rollout_certificate


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
