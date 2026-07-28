"""End-to-end gate for the four-layer spine: fit -> plan -> certify -> audit on the true plant."""

import jax.numpy as jnp
import pytest

from chc.spine import run_spine, two_zone_market

_REPORT = run_spine()
_HORIZON = len(_REPORT.arm("causal").certificate.planned_certified)


def test_the_backdoor_adjustment_recovers_the_gain_the_naive_fit_gets_wrong() -> None:
    naive, causal = _REPORT.arm("naive"), _REPORT.arm("causal")
    assert naive.effect < 0.0 < causal.effect  # the confounding sign flip
    assert abs(causal.effect - _REPORT.effect_true) < 0.05
    assert abs(naive.effect - _REPORT.effect_true) > 1.0


def test_a_plan_built_on_the_confounded_gain_costs_more_than_it_promises() -> None:
    naive, causal = _REPORT.arm("naive"), _REPORT.arm("causal")
    # the causal arm's model IS the plant, so its planned cost is what it pays
    assert causal.true_cost == pytest.approx(causal.plan.task_cost, rel=1e-2)
    assert naive.true_cost > 2.0 * naive.plan.task_cost  # it believes its own wrong model
    assert naive.true_cost > causal.true_cost


def test_gamma_star_prices_the_problem_while_the_certified_prefix_prices_the_assumption() -> None:
    tight, loose = run_spine(gamma=1.05), run_spine(gamma=3.0)
    for name in ("naive", "causal"):
        assert tight.arm(name).certificate.gamma_star == loose.arm(name).certificate.gamma_star
        assert tight.arm(name).certificate.certified_steps > (
            loose.arm(name).certificate.certified_steps
        )


def test_the_certificate_is_informative_rather_than_certifying_everything() -> None:
    # a sweep where every step certifies proves nothing; so does one where none does
    for arm in run_spine(gamma=1.0).arms:
        assert 0 < arm.certificate.certified_steps < _HORIZON


def test_a_confounded_model_reports_a_fragile_problem_without_seeing_the_truth() -> None:
    naive, causal = _REPORT.arm("naive"), _REPORT.arm("causal")
    assert naive.certificate.gamma_star < 2.0 < causal.certificate.gamma_star


def test_the_true_plant_stays_in_the_safe_set_over_the_certified_prefix() -> None:
    causal = _REPORT.arm("causal")
    assert causal.certificate.certified_steps > 0
    assert causal.true_barrier_min > 0.0
    # and the plan does leave the safe set later, so the prefix is a real restriction
    assert float(jnp.min(causal.true_trajectory[:, 1])) < _REPORT.supply_floor


def test_the_incentive_column_conserves_drivers() -> None:
    # an incentive moves drivers between zones, it does not create them -- the interference channel
    market = two_zone_market(1.0)
    x = jnp.array([0.3, -0.2])
    response = market(0.0, x, jnp.array([1.0])) - market(0.0, x, jnp.zeros(1))
    assert abs(float(jnp.sum(response))) < 1e-6
