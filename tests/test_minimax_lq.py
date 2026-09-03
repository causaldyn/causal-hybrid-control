"""chc.regret: the minimax LQ controller over a partially identified effect interval (§58).

§33 showed by counterexample that certainty equivalence is not minimax for the LQ loss and left the
robust controller unbuilt. These pin the closed form against brute force, against CE, and against
the structural claim that makes the horizon case work -- the adversary never switches endpoints.
"""

import numpy as np
import pytest

from chc.regret import (
    _closed_loop_cost,
    minimax_action,
    minimax_lq_certificate,
    minimax_lq_policy,
)


def test_the_static_closed_form_reproduces_section_33s_counterexample() -> None:
    robust = minimax_action(target=1.0, b_lo=0.5, b_hi=1.5, effort=1.0)
    assert robust.action == pytest.approx(0.4)
    assert robust.worst_case == pytest.approx(0.8)
    assert robust.binding == "lower"
    ce_worst = max((0.5 * 0.5 - 1.0) ** 2, (1.5 * 0.5 - 1.0) ** 2) + 0.5**2
    assert robust.worst_case < ce_worst == pytest.approx(0.8125)


def test_the_closed_form_matches_a_brute_force_search_across_regimes() -> None:
    rng = np.random.default_rng(11)
    for _ in range(60):
        b_hat = float(rng.uniform(-2.0, 2.0))
        halfwidth = float(rng.uniform(0.0, 1.5))
        target = float(rng.uniform(-2.0, 2.0))
        effort = float(10 ** rng.uniform(-3, 1))
        curvature = float(10 ** rng.uniform(-2, 1))
        robust = minimax_action(target, b_hat - halfwidth, b_hat + halfwidth, effort, curvature)
        grid = np.linspace(robust.action - 2.0, robust.action + 2.0, 200_001)
        lower = curvature * ((b_hat - halfwidth) * grid - target) ** 2 + effort * grid**2
        upper = curvature * ((b_hat + halfwidth) * grid - target) ** 2 + effort * grid**2
        assert robust.worst_case <= float(np.min(np.maximum(lower, upper))) + 1e-9


def test_the_robust_action_is_not_always_smaller_than_certainty_equivalence() -> None:
    # "Be robust, act less" is a statement about expensive effort, not about pessimism: the sign of
    # the correction flips at curvature*b_lo*b_hat = effort.
    timid = minimax_action(1.0, 0.5, 1.5, effort=1.0)  # b_lo*b_hat = 0.5 < 1
    bold = minimax_action(1.0, 1.5, 2.5, effort=1.0)  # b_lo*b_hat = 3.0 > 1
    assert timid.action < 1.0 / (1.0 + 1.0)
    assert bold.action > 2.0 / (4.0 + 1.0)


def test_an_unidentified_effect_sign_makes_doing_nothing_optimal() -> None:
    robust = minimax_action(target=1.0, b_lo=-0.5, b_hi=0.5, effort=1.0)
    assert robust.action == 0.0
    assert robust.binding == "zero"


def test_an_empty_interval_and_a_free_action_are_rejected_not_answered() -> None:
    with pytest.raises(ValueError, match="empty identified interval"):
        minimax_action(1.0, 1.5, 0.5, effort=1.0)
    with pytest.raises(ValueError, match="effort must be positive"):
        minimax_action(1.0, 0.5, 1.5, effort=0.0)


def test_the_horizon_policy_beats_certainty_equivalences_worst_case() -> None:
    curve = minimax_lq_certificate()
    assert curve.ok
    assert curve.static_action == pytest.approx(curve.grid_action, abs=1e-4)
    assert all(v <= c + 1e-12 for v, c in zip(curve.values, curve.ce_values, strict=True))
    assert curve.values[0] == pytest.approx(curve.ce_values[0])  # no interval, no difference


def test_the_per_step_adversary_buys_nothing_over_a_constant_unknown_effect() -> None:
    # The result that turns the DP relaxation from a caveat into a statement: the robust action
    # always leaves the LOWER endpoint worst, so a constant b_lo realises the per-step optimum.
    rng = np.random.default_rng(3)
    for _ in range(40):
        state_gain = float(rng.uniform(-2.0, 2.0))
        b_hat = float(rng.uniform(0.2, 2.0))
        halfwidth = float(rng.uniform(0.0, 0.95) * b_hat)
        state_cost = float(10 ** rng.uniform(-2, 1))
        effort = float(10 ** rng.uniform(-3, 1))
        terminal = float(10 ** rng.uniform(-2, 1))
        horizon = int(rng.integers(2, 10))
        policy = minimax_lq_policy(
            state_gain,
            b_hat - halfwidth,
            b_hat + halfwidth,
            state_cost,
            effort,
            terminal,
            horizon,
        )
        effects = np.linspace(b_hat - halfwidth, b_hat + halfwidth, 501)
        realised = max(
            _closed_loop_cost(policy.gains, state_gain, float(b), state_cost, effort, terminal)
            for b in effects
        )
        assert policy.value == pytest.approx(realised, rel=1e-9)
        at_lower = _closed_loop_cost(
            policy.gains, state_gain, b_hat - halfwidth, state_cost, effort, terminal
        )
        assert at_lower == pytest.approx(policy.value, rel=1e-9)


def test_a_zero_width_interval_recovers_the_ordinary_riccati_gains() -> None:
    policy = minimax_lq_policy(1.0, 0.8, 0.8, 1.0, 0.3, 2.0, 6)
    assert policy.gains == pytest.approx(policy.ce_gains)
    assert policy.value == pytest.approx(policy.ce_value)
