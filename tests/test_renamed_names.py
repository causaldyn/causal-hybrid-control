"""A name 0.20.0 or 0.21.0 renamed still reads under its old one, warning that it leaves in 1.0.

Each old name said more than its value is: a channel's standard error read as an
"identification radius", a regret to second order read as "regret bounds", and a band that holds
only where the regret's curvature is constant read as a high-probability certificate (0.20.0); the
same second-order regret, scalar and matrix, read as the regret itself, and a van Trees floor read
at the plant supplied as the floor (0.21.0). Each is renamed to what it is, and its old name serves
the same value with a `DeprecationWarning` naming the new one.
"""

from __future__ import annotations

import numpy as np
import pytest

import chc
from chc.decision import DecisionCertificate
from chc.graph import AdjustmentSet
from chc.research import regret as research_regret
from chc.research import sensitivity
from chc.research.regret import (
    HighProbRegretCurve,
    confounding_robust_lq_regret_certificate,
    confounding_robust_lq_second_order_regret,
    confounding_robust_lq_second_order_regret_matrix,
    local_highprob_regret_certificate,
    multivariate_action_floor,
)


def test_the_certificate_s_channel_error_reads_under_its_old_name_with_a_warning() -> None:
    certificate = DecisionCertificate(
        identification="identified",
        adjustment=AdjustmentSet(("demand",), "identified", "a graph's backdoor set"),
        channel_error=0.25,
        overlap=1.0,
        certificate_status="certified",
        certified_horizon=3,
        barrier_certified_steps=None,
        gamma_star=None,
        solver_status="converged",
        solver_iterations=1,
        regret_bound=None,
    )
    with pytest.warns(
        DeprecationWarning,
        match=r"^DecisionCertificate\.identification_radius is renamed channel_error and leaves "
        r"in 1\.0: it is the channel's standard error, not a radius$",
    ):
        assert certificate.identification_radius == 0.25


def test_the_second_order_regrets_read_under_their_old_name_with_a_warning() -> None:
    curve = confounding_robust_lq_regret_certificate()
    with pytest.warns(
        DeprecationWarning,
        match=r"^ConfoundingRobustLQRegretCurve\.regret_bounds is renamed second_order_regrets "
        r"and leaves in 1\.0",
    ):
        old = curve.regret_bounds
    assert old is curve.second_order_regrets


def test_the_local_band_reads_under_its_old_name_with_a_warning() -> None:
    arguments = {"n": 40, "deltas": (0.1, 0.01), "n_trials": 50}
    with pytest.warns(
        DeprecationWarning,
        match=r"^highprob_regret_certificate is renamed local_highprob_regret_certificate and "
        r"leaves in 1\.0",
    ):
        old = research_regret.highprob_regret_certificate(**arguments)
    new = local_highprob_regret_certificate(**arguments)
    assert isinstance(old, HighProbRegretCurve)
    for field in ("deltas", "highprob_bands", "empirical_coverage", "confounded_bands"):
        np.testing.assert_array_equal(getattr(old, field), getattr(new, field))
    assert old.cr_floor == new.cr_floor


def test_the_second_order_regret_reads_under_its_old_name_with_a_warning() -> None:
    with pytest.warns(
        DeprecationWarning,
        match=r"^confounding_robust_lq_regret is renamed confounding_robust_lq_second_order_regret "
        r"and leaves in 1\.0: it is the regret to second order, not a bound on the worst regret$",
    ):
        old = research_regret.confounding_robust_lq_regret(2.0, 0.1, 0.3)
    assert old == confounding_robust_lq_second_order_regret(2.0, 0.1, 0.3)


def test_the_matrix_second_order_regret_reads_under_its_old_name_with_a_warning() -> None:
    curvature = np.array([[2.0, 0.5], [0.5, 3.0]])
    stat, confounding = np.array([[0.1, 0.0], [0.2, 0.1]]), np.array([[0.3, 0.1], [0.0, 0.2]])
    with pytest.warns(
        DeprecationWarning,
        match=r"^confounding_robust_lq_regret_matrix is renamed "
        r"confounding_robust_lq_second_order_regret_matrix and leaves in 1\.0: it is the regret to "
        r"second order, not a bound on the worst regret$",
    ):
        old = research_regret.confounding_robust_lq_regret_matrix(curvature, stat, confounding)
    assert old == confounding_robust_lq_second_order_regret_matrix(curvature, stat, confounding)


@pytest.mark.parametrize(
    ("old", "new", "arguments"),
    [
        (
            "confounding_robust_lq_regret",
            "confounding_robust_lq_second_order_regret",
            (2.0, 0.1, 0.3),
        ),
        (
            "confounding_robust_lq_regret_matrix",
            "confounding_robust_lq_second_order_regret_matrix",
            (np.eye(2), np.full((2, 2), 0.1), np.full((2, 2), 0.3)),
        ),
    ],
)
def test_an_old_regret_name_still_serves_at_the_top_level_and_in_the_facade(
    old: str, new: str, arguments: tuple
) -> None:
    """The top level keeps the old name among its deprecated ones, served from
    ``chc.research.regret``, and does not take the new one: a research name lives there alone. The
    facade ``chc.research.sensitivity`` imports the old name too, outside its ``__all__``."""
    with pytest.warns(
        DeprecationWarning,
        match=rf"^chc\.{old} leaves the top level in 1\.0: import it from chc\.research\.regret$",
    ):
        served = getattr(chc, old)
    assert served is getattr(research_regret, old) is getattr(sensitivity, old)
    with pytest.warns(DeprecationWarning, match=rf"^{old} is renamed {new} and leaves in 1\.0"):
        assert served(*arguments) == getattr(research_regret, new)(*arguments)
    with pytest.raises(AttributeError):
        getattr(chc, new)
    assert old not in sensitivity.__all__
    assert getattr(sensitivity, new) is getattr(research_regret, new)
    assert new in sensitivity.__all__


def test_the_local_floor_reads_under_its_old_name_with_a_warning() -> None:
    floor = multivariate_action_floor(
        effect=np.zeros((1, 1)),
        state_weight=np.eye(1),
        action_weight=np.array([[0.5]]),
        target=np.ones(1),
        information=np.array([[6.0]]),
    )
    with pytest.warns(
        DeprecationWarning,
        match=r"^MultivariateActionFloor\.floor is renamed local_floor and leaves in 1\.0: it is "
        r"read at the plant supplied, and a Bayes action can go below it$",
    ):
        old = floor.floor
    assert old == floor.local_floor == pytest.approx(1.0 / 3.0, rel=1e-12, abs=0.0)
