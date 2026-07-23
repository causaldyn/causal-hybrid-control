"""chc.regret: §35 minimax confounding-robust controller under asymmetric loss.

Under a symmetric loss CE is already minimax (pessimism = centre, §33). Under an ASYMMETRIC loss the
pessimism radius shifts the gain and strictly beats CE (Rocq ``confounding_robust_control.v``).
"""

import pytest

from chc.regret import (
    certainty_equivalence_control,
    confounding_robust_control,
    confounding_robust_control_certificate,
    worst_case_asymmetric_loss,
)


def test_certificate_confirms_shift_and_strict_improvement() -> None:
    cert = confounding_robust_control_certificate()
    assert cert.ok
    assert cert.u_robust == pytest.approx(cert.numeric_argmin, abs=1e-3)  # closed form == minimax
    assert cert.u_robust < cert.u_certainty_equivalence  # overshoot costlier -> gain shifted down
    assert cert.worst_case_loss_robust < cert.worst_case_loss_ce  # strictly better worst case
    assert cert.symmetric_equals_ce


def test_symmetric_loss_recovers_certainty_equivalence() -> None:
    # alpha=beta: the minimax control is the CE centre (recovers §33's "pessimism = centre")
    u_ce = certainty_equivalence_control(1.4, 1.0)
    u_rob = confounding_robust_control(1.4, 0.3, 1.0, 2.0, 2.0)
    assert u_rob == pytest.approx(u_ce)


def test_gain_shifts_further_with_more_confounding_and_more_asymmetry() -> None:
    u_ce = certainty_equivalence_control(1.3, 1.0)
    small = confounding_robust_control(1.3, 0.1, 1.0, 3.0, 1.0)
    wide = confounding_robust_control(1.3, 0.4, 1.0, 3.0, 1.0)
    skewed = confounding_robust_control(1.3, 0.1, 1.0, 9.0, 1.0)
    assert small < u_ce  # any asymmetry + confounding shifts the gain down
    assert wide < small  # more confounding -> more conservative
    assert skewed < small  # more asymmetry -> more conservative


def test_undershoot_costlier_shifts_the_gain_up() -> None:
    # beta > alpha (undershoot worse): the robust controller becomes MORE aggressive than CE
    u_ce = certainty_equivalence_control(1.3, 1.0)
    u_rob = confounding_robust_control(1.3, 0.25, 1.0, 1.0, 3.0)
    assert u_rob > u_ce  # push harder to avoid the costlier undershoot


def test_worst_case_loss_is_the_max_of_the_weighted_tails() -> None:
    # a control that overshoots badly: the alpha-weighted overshoot tail dominates
    w = worst_case_asymmetric_loss(1.0, 1.3, 0.25, 1.0, 4.0, 1.0)
    over = 4.0 * ((1.3 + 0.25) * 1.0 - 1.0)  # alpha * overshoot at b_hat+D
    assert w == pytest.approx(over)
