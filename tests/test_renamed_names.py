"""A name 0.20.0 renamed still reads under its old one, warning that it leaves in 1.0.

Each old name said more than its value is: a channel's standard error read as an
"identification radius", a regret to second order read as "regret bounds", and a band that holds
only where the regret's curvature is constant read as a high-probability certificate. Each is
renamed to what it is, and its old name serves the same value with a `DeprecationWarning` naming
the new one.
"""

from __future__ import annotations

import numpy as np
import pytest

from chc.decision import DecisionCertificate
from chc.graph import AdjustmentSet
from chc.research import regret as research_regret
from chc.research.regret import (
    HighProbRegretCurve,
    confounding_robust_lq_regret_certificate,
    local_highprob_regret_certificate,
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
