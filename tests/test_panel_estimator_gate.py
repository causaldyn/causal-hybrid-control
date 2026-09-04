"""chc.regret: Result 51's Psi put against a real cross-fitted DML fit on a panel (§60).

Result 51 shipped Psi with a scope note saying it is a functional of the process, not a re-derived
estimator. These run the comparison that settles it, and the answer is not the flattering one: the
functional RANKS the two fold partitions correctly at every cluster count and its ``O(1/g)`` washout
law holds, but its point prediction of the variance ratio is conservative -- the real estimator
gains more from the good partition than Psi says.

The washout law is the reason the experiment has to sweep the cluster count at all: the partition
enters Psi once, linearly, and with a block-diagonal covariance over ``g`` independent clusters the
whole design effect is ``O(1/g)``. Measure at one ``g`` and the result is a number with no scope.
"""

import numpy as np
import pytest

from chc.network_causal import cycle_shells, panel_covariance
from chc.regret import _psi_ratio_closed_form, panel_estimator_certificate

GAMMAS = (1.0, 0.7, 0.4)


def _psi(sigma: np.ndarray, fold: np.ndarray, g: int, m: int, p: int, k: int = 2) -> float:
    r = k / (k - 1)
    rows = g * m * p
    trace, total = g * float(np.trace(sigma)), g * float(sigma.sum())
    same = g * float(sigma[fold[:, None] == fold[None, :]].sum())
    trace_a = rows - r**2 + (r**2 - 1) * k
    return (
        rows**2
        * (trace - r**4 * total / rows + (r**4 - 1) * (k / rows) * same)
        / (trace_a**2 * trace)
    )


def _partitions(m: int, p: int) -> tuple[np.ndarray, np.ndarray]:
    unit = np.repeat(np.arange(m), p)
    return (unit % 2).astype(np.int_), (unit * 2 // m).astype(np.int_)


def test_the_closed_form_ratio_is_the_direct_computation() -> None:
    # validation/panel_estimator_gate.mac (4): every partition-free factor cancels
    m, p = 12, 12
    sigma = panel_covariance(cycle_shells(m, 2), GAMMAS, 0.9, 1, p)
    parity, block = _partitions(m, p)
    for g in (2, 6, 20, 100):
        direct = _psi(sigma, parity, g, m, p) / _psi(sigma, block, g, m, p)
        assert direct == pytest.approx(
            _psi_ratio_closed_form(sigma, parity, block, g, 2), abs=1e-12
        )


def test_the_design_effect_washes_out_in_the_cluster_count() -> None:
    m, p = 12, 12
    sigma = panel_covariance(cycle_shells(m, 2), GAMMAS, 0.9, 1, p)
    parity, block = _partitions(m, p)
    ratios = [_psi_ratio_closed_form(sigma, parity, block, g, 2) for g in (2, 6, 20, 100, 1000)]
    assert ratios == sorted(ratios)
    assert ratios[0] < 0.5  # a two-cluster panel: the design halves the variance
    assert ratios[-1] > 0.99  # a thousand: it is gone
    # the gap closes like 1/g, so g * (1 - ratio) settles rather than growing
    scaled = [g * (1.0 - r) for g, r in zip((20, 100, 1000), ratios[2:], strict=True)]
    assert max(scaled) / min(scaled) < 1.2


def test_a_smaller_same_fold_mass_is_always_better() -> None:
    m, p = 12, 12
    sigma = panel_covariance(cycle_shells(m, 2), GAMMAS, 0.9, 1, p)
    parity, block = _partitions(m, p)
    mass = {
        name: float(sigma[fold[:, None] == fold[None, :]].sum())
        for name, fold in (("parity", parity), ("block", block))
    }
    assert mass["parity"] < mass["block"]
    for g in (2, 20):
        assert _psi(sigma, parity, g, m, p) < _psi(sigma, block, g, m, p)


def test_the_ordering_does_not_flip_with_the_cluster_count() -> None:
    m, p = 12, 12
    sigma = panel_covariance(cycle_shells(m, 2), GAMMAS, 0.6, 1, p)
    parity, block = _partitions(m, p)
    assert all(
        _psi_ratio_closed_form(sigma, parity, block, g, 2) < 1.0 for g in (1, 2, 5, 50, 5000)
    )


def test_the_gate_against_a_real_estimator() -> None:
    # 120 draws, not 40: at 40 the sample variance ratio is biased toward 1 and the conservatism
    # finding flips on a lucky sample -- measured under both JAX precisions, which draw DIFFERENT
    # panels because the sampler derives its NumPy seed from a JAX key
    gate = panel_estimator_certificate(cluster_counts=(2, 20), draws=120, bootstrap=400)
    assert gate.signs_agree
    assert gate.predicted_washout
    assert gate.measured_washout
    assert gate.closed_form_error < 1e-10
    # the finding, recorded rather than gated: Psi understates the design gain
    assert gate.functional_is_conservative
    assert gate.worst_shortfall > 0.0
