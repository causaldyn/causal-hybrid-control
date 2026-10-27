"""The names 0.20.0, 0.21.0 and 0.24.0 renamed are gone in 1.0, under their old names.

Each old name said more than its value is: a channel's standard error read as an
"identification radius", a regret to second order read as "regret bounds", and a band that holds
only where the regret's curvature is constant read as a high-probability certificate (0.20.0); the
same second-order regret, scalar and matrix, read as the regret itself, and a van Trees floor read
at one plant as the floor, on the action floor and on the van Trees curve (0.21.0); and the
marketplace's gradient test under the name of :class:`chc.decision.DitherCheck`, another test
(0.24.0). Each served its value under the old name with a `DeprecationWarning` naming the new one
until 1.0, which removed them together (docs/adr/0147).
"""

from __future__ import annotations

import dataclasses

import pytest

from chc import marketplace
from chc.decision import DecisionCertificate
from chc.research import deep_galerkin, regret, sensitivity

GONE = [
    (DecisionCertificate, "identification_radius", "channel_error"),
    (regret.ConfoundingRobustLQRegretCurve, "regret_bounds", "second_order_regrets"),
    (regret, "highprob_regret_certificate", "local_highprob_regret_certificate"),
    (
        deep_galerkin.NonlinearDwrCurve,
        "denominator_against_result_55",
        "denominator_against_closed_form",
    ),
    (regret, "confounding_robust_lq_regret", "confounding_robust_lq_second_order_regret"),
    (sensitivity, "confounding_robust_lq_regret", "confounding_robust_lq_second_order_regret"),
    (
        regret,
        "confounding_robust_lq_regret_matrix",
        "confounding_robust_lq_second_order_regret_matrix",
    ),
    (
        sensitivity,
        "confounding_robust_lq_regret_matrix",
        "confounding_robust_lq_second_order_regret_matrix",
    ),
    (regret.MultivariateActionFloor, "floor", "local_floor"),
    (regret.MultivariateVanTreesCurve, "floor", "local_floor"),
    (marketplace, "DitherCheck", "GradientDitherCheck"),
]


@pytest.mark.parametrize(
    ("owner", "old", "new"), GONE, ids=lambda part: getattr(part, "__name__", part)
)
def test_an_old_name_is_gone_and_its_new_one_stays(owner: object, old: str, new: str) -> None:
    assert not hasattr(owner, old)
    if isinstance(owner, type):
        assert new in {field.name for field in dataclasses.fields(owner)}
    else:
        assert callable(getattr(owner, new))
