"""What leaves at 1.0: each name the library serves only with a ``DeprecationWarning``, each
re-export of one, and each record key kept for a stored record's reader, with its replacement.

Every deprecated name says that it leaves in 1.0, and 1.0 removes them together (docs/adr/0147).
``tests/test_removals.py`` holds this list equal to the library: each entry still serves its value,
and its warning names the replacement; each deprecation warning in the source is raised at a site
listed here; each re-export of a listed function, and each record key that repeats a renamed name,
is listed. The top level's deprecated names are listed once, in ``chc._MOVED``, which serves them.
1.0 deletes each entry's code, and this module with it. This module imports nothing from ``chc``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["top-level name", "function", "attribute", "re-export", "record key"]


@dataclass(frozen=True)
class Removal:
    """One thing that leaves at 1.0, and what a caller writes instead."""

    # What a caller writes today, as a dotted path. ``chc.<name>`` stands for each name of
    # ``chc._MOVED``; a record key is its path in the record, ``certificate.identification_radius``.
    old: str
    # What to write instead. A top-level name's module is the one ``chc._MOVED`` names; a record
    # key's replacement is the key beside it.
    new: str
    kind: Kind
    # The module and qualified name of the code that warns or, for a record key, that writes it.
    # A re-export's site is its function's, which warns when it is called.
    site: str
    since: str  # the release that deprecated it
    adrs: tuple[str, ...]  # the decision records that deprecated it and dated its removal


TOP_LEVEL_NAMES = 421
"""The size of ``chc._MOVED``: every name 0.7.0 bound at the top level that the lifecycle does not
hold, 166 of them served from ``chc.research`` since 0.19.0 (ADRs 0016 and 0117)."""

SCHEMA_VERSION_AT_1_0 = 3
"""``chc.decision.SCHEMA_VERSION`` once ``to_json`` stops writing ``identification_radius``: a key
that leaves changes what a reader of it finds, so its removal moves the version (ADR 0130)."""

_REGRET = "chc.research.regret"

REMOVALS: tuple[Removal, ...] = (
    Removal(
        "chc.<name>",
        "<module>.<name>",
        "top-level name",
        "chc.__getattr__",
        "0.8.0",
        ("0016", "0117"),
    ),
    Removal(
        "chc.decision.DecisionCertificate.identification_radius",
        "chc.decision.DecisionCertificate.channel_error",
        "attribute",
        "chc.decision.DecisionCertificate.identification_radius",
        "0.20.0",
        ("0128",),
    ),
    Removal(
        "certificate.identification_radius",
        "certificate.channel_error",
        "record key",
        "chc.decision.Prescription.to_json",
        "0.20.0",
        ("0128", "0130"),
    ),
    Removal(
        f"{_REGRET}.ConfoundingRobustLQRegretCurve.regret_bounds",
        f"{_REGRET}.ConfoundingRobustLQRegretCurve.second_order_regrets",
        "attribute",
        f"{_REGRET}.ConfoundingRobustLQRegretCurve.regret_bounds",
        "0.20.0",
        ("0128",),
    ),
    Removal(
        f"{_REGRET}.highprob_regret_certificate",
        f"{_REGRET}.local_highprob_regret_certificate",
        "function",
        f"{_REGRET}.highprob_regret_certificate",
        "0.20.0",
        ("0128",),
    ),
    # Renamed in 0.20.0 under ADR 0128's rule, with no record of its own.
    Removal(
        "chc.research.deep_galerkin.NonlinearDwrCurve.denominator_against_result_55",
        "chc.research.deep_galerkin.NonlinearDwrCurve.denominator_against_closed_form",
        "attribute",
        "chc.research.deep_galerkin.NonlinearDwrCurve.denominator_against_result_55",
        "0.20.0",
        ("0128",),
    ),
    Removal(
        f"{_REGRET}.confounding_robust_lq_regret",
        f"{_REGRET}.confounding_robust_lq_second_order_regret",
        "function",
        f"{_REGRET}.confounding_robust_lq_regret",
        "0.21.0",
        ("0130",),
    ),
    Removal(
        "chc.research.sensitivity.confounding_robust_lq_regret",
        "chc.research.sensitivity.confounding_robust_lq_second_order_regret",
        "re-export",
        f"{_REGRET}.confounding_robust_lq_regret",
        "0.21.0",
        ("0130",),
    ),
    Removal(
        f"{_REGRET}.confounding_robust_lq_regret_matrix",
        f"{_REGRET}.confounding_robust_lq_second_order_regret_matrix",
        "function",
        f"{_REGRET}.confounding_robust_lq_regret_matrix",
        "0.21.0",
        ("0130",),
    ),
    Removal(
        "chc.research.sensitivity.confounding_robust_lq_regret_matrix",
        "chc.research.sensitivity.confounding_robust_lq_second_order_regret_matrix",
        "re-export",
        f"{_REGRET}.confounding_robust_lq_regret_matrix",
        "0.21.0",
        ("0130",),
    ),
    Removal(
        f"{_REGRET}.MultivariateActionFloor.floor",
        f"{_REGRET}.MultivariateActionFloor.local_floor",
        "attribute",
        f"{_REGRET}.MultivariateActionFloor.floor",
        "0.21.0",
        ("0130",),
    ),
    Removal(
        f"{_REGRET}.MultivariateVanTreesCurve.floor",
        f"{_REGRET}.MultivariateVanTreesCurve.local_floor",
        "attribute",
        f"{_REGRET}.MultivariateVanTreesCurve.floor",
        "0.21.0",
        ("0130",),
    ),
)
