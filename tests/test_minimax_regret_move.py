"""How much of a move to make when its value is linear, its estimate noisy, and confounding may
bias it: the minimax-regret rule of Montiel Olea, Qiu and Stoye (Review of Economic Studies 2026;
arXiv 2312.17623 v3, Section 4.2).

The numbers are Maxima's (``validation/minimax_regret_move.mac``). The worst regret over the
values the data allow is read here two ways: the rule's probability of moving in closed form,
``E (Y - c)_+ = (m - c) Phi((m - c)/s) + s phi((m - c)/s)``, and by Monte Carlo.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from chc.dynamics_id import fit_causal_residual, omitted_confounder_bound
from chc.regret import minimax_regret_move
from chc.sensitivity import barrier_gamma_star

ROOT_HALF_PI = math.sqrt(math.pi / 2.0)


def _at_gap(k: float, sigma: float = 1.0, estimate: float = 0.0):
    """The move at half-width ``k``: Gamma = 3 halves the gap, so the gap is ``2 k``."""
    return minimax_regret_move(estimate, sigma, gamma=3.0, cvar_gap=2.0 * k)


def _moves(beta: np.ndarray, sigma: float, rho: float) -> np.ndarray:
    """``E clip((Y + rho) / (2 rho), 0, 1)`` for ``Y ~ N(beta, sigma^2)``, in closed form."""

    def plus(c: float) -> np.ndarray:
        z = (beta - c) / sigma
        return (beta - c) * norm.cdf(z) + sigma * norm.pdf(z)

    return (plus(-rho) - plus(rho)) / (2.0 * rho)


def _worst(beta: np.ndarray, k: float, moved: np.ndarray) -> np.ndarray:
    """The worst regret over values in ``[beta - k, beta + k]`` of moving with probability
    ``moved``: the largest positive value times the chance of holding, or the most negative times
    the chance of moving."""
    return np.maximum(np.maximum(beta + k, 0.0) * (1.0 - moved), np.maximum(k - beta, 0.0) * moved)


GRID = np.linspace(-12.0, 12.0, 48_001)


@pytest.mark.parametrize(
    ("k", "rho", "sign_regret"),
    [(2.0, 1.87970281712, 1.05093240493), (5.0, 4.99999713344, 3.37098113946)],
)
def test_past_the_threshold_the_rule_randomises_and_its_worst_regret_is_half_the_width(
    k: float, rho: float, sign_regret: float
) -> None:
    move = _at_gap(k)
    assert move.rule == "linear"
    assert move.halfwidth == pytest.approx(k, rel=1e-15, abs=0.0)
    assert move.threshold == pytest.approx(rho, rel=1e-10, abs=0.0)
    assert move.worst_regret == k / 2.0
    assert move.sign_regret == pytest.approx(sign_regret, rel=1e-10, abs=0.0)
    # the fixed point
    assert move.threshold == pytest.approx(
        k * (1.0 - 2.0 * norm.cdf(-move.threshold)), rel=1e-13, abs=0.0
    )
    # over every reduced form, the rule's worst regret reaches k/2 and no more; the sign rule's
    # reaches its own, which is more
    linear = _worst(GRID, k, _moves(GRID, 1.0, move.threshold))
    assert linear.max() == pytest.approx(k / 2.0, rel=1e-9, abs=0.0)
    sign = _worst(GRID, k, norm.cdf(GRID))
    assert sign.max() == pytest.approx(sign_regret, rel=1e-7, abs=0.0)
    assert sign.max() > linear.max() + 0.05


def test_the_worst_regret_holds_by_monte_carlo() -> None:
    """The rule's chance of moving, drawn: 200 000 estimates at each of 13 reduced forms."""
    k = 2.0
    rho = _at_gap(k).threshold
    rng = np.random.default_rng(20261012)
    betas = np.linspace(-3.0, 3.0, 13)
    draws = betas[:, None] + rng.standard_normal((betas.size, 200_000))
    moved = np.clip((draws + rho) / (2.0 * rho), 0.0, 1.0)
    shares = moved.mean(axis=1)
    errors = moved.std(axis=1) / math.sqrt(moved.shape[1])
    assert np.all(np.abs(shares - _moves(betas, 1.0, rho)) < 4.0 * errors)
    regret = _worst(betas, k, shares)
    assert regret.max() == pytest.approx(1.0, abs=4.0 * k * errors.max())
    assert regret.max() < _worst(betas, k, (draws >= 0.0).mean(axis=1)).max()


@pytest.mark.parametrize(("k", "sigma"), [(0.5, 1.0), (1.2, 1.0), (3.0, 4.0), (0.0, 2.0)])
def test_below_the_threshold_the_sign_rule_is_minimax(k: float, sigma: float) -> None:
    assert k <= ROOT_HALF_PI * sigma
    move = minimax_regret_move(0.0, sigma, gamma=3.0, cvar_gap=2.0 * k)
    assert (move.rule, move.share, move.threshold) == ("sign", 1.0, 0.0)
    assert minimax_regret_move(-1e-300, sigma, gamma=3.0, cvar_gap=2.0 * k).share == 0.0
    sign = _worst(GRID * sigma, k, norm.cdf(GRID))
    assert move.worst_regret == move.sign_regret
    assert move.worst_regret == pytest.approx(sign.max(), rel=1e-7, abs=0.0)
    assert move.worst_regret >= k / 2.0


def test_the_two_rules_meet_at_the_threshold() -> None:
    """At ``k = sqrt(pi/2) sigma`` the sign rule's worst regret is ``k / 2``, and just past it the
    band the share rises across is narrow: the linear rule is the sign rule in the limit."""
    sigma = 1.7
    at = minimax_regret_move(0.3, sigma, gamma=3.0, cvar_gap=2.0 * ROOT_HALF_PI * sigma)
    assert at.rule == "sign"
    assert at.worst_regret == pytest.approx(ROOT_HALF_PI * sigma / 2.0, rel=1e-12, abs=0.0)
    past = minimax_regret_move(
        0.3, sigma, gamma=3.0, cvar_gap=2.0 * ROOT_HALF_PI * sigma * (1.0 + 1e-6)
    )
    assert past.rule == "linear"
    assert 0.0 < past.threshold < 0.01 * sigma
    assert past.share == 1.0
    assert past.worst_regret == pytest.approx(at.worst_regret, rel=2e-6, abs=0.0)
    # a hair past the threshold in floating point still finds its root
    hair = minimax_regret_move(0.0, 1.0, gamma=3.0, cvar_gap=2.0 * np.nextafter(ROOT_HALF_PI, 2.0))
    assert hair.rule == "linear"
    assert 0.0 < hair.threshold < 1e-6
    assert hair.share == 0.5


@pytest.mark.parametrize(
    ("estimate", "share"),
    [(-3.0, 0.0), (-1.87970281712, 0.0), (-0.5, 0.36700), (0.0, 0.5), (0.9, 0.73940), (2.5, 1.0)],
)
def test_the_share_rises_linearly_across_the_band(estimate: float, share: float) -> None:
    move = _at_gap(2.0, estimate=estimate)
    expected = min(max((estimate + move.threshold) / (2.0 * move.threshold), 0.0), 1.0)
    assert move.share == pytest.approx(expected, rel=1e-15, abs=1e-15)
    assert move.share == pytest.approx(share, abs=1e-5)


def test_with_no_noise_the_rule_is_the_identified_set_s_own() -> None:
    """``sigma = 0``: the value is known to ``[d - k, d + k]``, and the least worst regret over it
    is had at the share ``(d + k) / (2 k)``, where both sides lose ``(k^2 - d^2) / (2 k)``."""
    move = minimax_regret_move(0.4, 0.0, gamma=3.0, cvar_gap=4.0)
    assert (move.rule, move.threshold) == ("linear", 2.0)
    assert move.share == pytest.approx(0.6, rel=1e-15, abs=0.0)
    assert move.sign_regret == 2.0
    held = minimax_regret_move(0.0, 0.0, gamma=1.0, cvar_gap=4.0)
    assert (held.rule, held.share, held.worst_regret) == ("sign", 1.0, 0.0)


def test_the_whole_move_holds_to_its_gamma_and_noise_raises_it() -> None:
    estimate, sigma, gap = 1.5, 1.0, 4.0
    move = minimax_regret_move(estimate, sigma, gamma=1.0, cvar_gap=gap)
    full = move.full_move_gamma
    reach = estimate / math.erf(estimate / math.sqrt(2.0))
    assert reach == pytest.approx(1.73133071971, rel=1e-10, abs=0.0)  # Maxima's STEP 6
    assert full == pytest.approx((gap + reach) / (gap - reach), rel=1e-12, abs=0.0)
    assert minimax_regret_move(estimate, sigma, gamma=full * (1 - 1e-9), cvar_gap=gap).share == 1.0
    assert minimax_regret_move(estimate, sigma, gamma=full * (1 + 1e-6), cvar_gap=gap).share < 1.0
    flip = barrier_gamma_star(estimate, gap, 1.0)  # the noiseless identified set's flip
    assert 1.0 < flip < full
    quiet = minimax_regret_move(estimate, 1e-3, gamma=1.0, cvar_gap=gap).full_move_gamma
    assert quiet == pytest.approx(flip, rel=1e-12, abs=0.0)
    assert minimax_regret_move(estimate, 0.0, gamma=1.0, cvar_gap=gap).full_move_gamma == flip
    assert math.isnan(minimax_regret_move(-0.2, sigma, gamma=1.0, cvar_gap=gap).full_move_gamma)
    # a gap no Gamma can stretch to the estimate's reach never takes the move away
    assert minimax_regret_move(estimate, sigma, gamma=1.0, cvar_gap=1.7).full_move_gamma == math.inf


def test_the_gap_is_a_share_of_the_channel_s_size() -> None:
    """ADR 0085's convention: the half-width is ``(G - 1)/(G + 1) cvar_gap size``, so the move
    reads the same in any units of its value."""
    base = minimax_regret_move(0.7, 0.9, gamma=4.0, cvar_gap=0.5, size=6.0)
    assert base.halfwidth == pytest.approx(3.0 / 5.0 * 0.5 * 6.0, rel=1e-15, abs=0.0)
    for unit in (1e-6, 1e3):
        moved = minimax_regret_move(
            0.7 * unit, 0.9 * unit, gamma=4.0, cvar_gap=0.5, size=6.0 * unit
        )
        assert moved.rule == base.rule
        assert moved.share == pytest.approx(base.share, rel=1e-12, abs=0.0)
        assert moved.threshold == pytest.approx(base.threshold * unit, rel=1e-12, abs=0.0)
        assert moved.worst_regret == pytest.approx(base.worst_regret * unit, rel=1e-12, abs=0.0)
        assert moved.sign_regret == pytest.approx(base.sign_regret * unit, rel=1e-12, abs=0.0)
        assert moved.full_move_gamma == pytest.approx(base.full_move_gamma, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"estimate": math.nan}, "estimate must be finite"),
        ({"estimate": math.inf}, "estimate must be finite"),
        ({"standard_error": -1.0}, "standard_error"),
        ({"standard_error": math.inf}, "standard_error"),
        ({"gamma": 0.99}, "gamma"),
        ({"gamma": math.nan}, "gamma"),
        ({"cvar_gap": -0.1}, "cvar_gap"),
        ({"size": math.nan}, "size"),
    ],
)
def test_it_refuses_what_is_not_a_move(kwargs: dict[str, float], match: str) -> None:
    arguments = {"estimate": 0.5, "standard_error": 1.0, "gamma": 2.0} | kwargs
    with pytest.raises(ValueError, match=match):
        minimax_regret_move(**arguments)


def test_a_fitted_move_reads_its_influence_error() -> None:
    """On the flip test's log, the move from lever a to lever b: its estimate and its influence
    standard error. The whole move holds at least to the noiseless flip, and past it the share
    falls but the move is not held."""
    rng = np.random.default_rng(0)
    n = 4000
    x, observed, latent = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
    u = 0.8 * observed[:, None] + np.outer(latent, [0.0, 0.6]) + 0.5 * rng.standard_normal((n, 2))
    push = u @ np.array([1.0, 1.3]) + 0.7 * observed + 0.4 * latent
    after = x + 0.05 * (-0.5 * x + push) + 0.01 * rng.standard_normal(n)
    data = {
        "x": jnp.asarray(x)[:, None],
        "u": jnp.asarray(u),
        "x_next": jnp.asarray(after)[:, None],
        "observed": jnp.asarray(observed)[:, None],
    }
    fit = fit_causal_residual(
        lambda t, x, u: jnp.zeros_like(x),
        data,
        0.05,
        adjust_for=("observed",),
        degree=1,
        channel_degree=0,
        influence=True,
    )
    move = np.array([-1.0, 1.0]).reshape(1, 2, 1)
    bound = omitted_confounder_bound(fit, move, cf_y=0.0, cf_d=0.0)
    error = (bound.estimate - bound.ci_lower) / norm.ppf(0.975)
    assert 0.0 < error < bound.estimate
    gap = 2.0
    flip = barrier_gamma_star(abs(bound.estimate), gap, 1.0)
    full = minimax_regret_move(bound.estimate, error, gamma=1.0, cvar_gap=gap).full_move_gamma
    assert 1.0 < flip <= full
    past = minimax_regret_move(bound.estimate, error, gamma=2.0 * flip, cvar_gap=gap)
    assert past.rule == "linear"
    assert 0.5 < past.share < 1.0
