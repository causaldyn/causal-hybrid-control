"""The tube's levels: one conformal scale for the whole horizon, and a region for the channel.

The plan's tube holds the plant at 0.95 step by step (ADR 0080, ADR 0096), and every step of a
horizon at once at less: on AR(1) noise two steps at 0.95 each hold together 0.915. The field's
error in it is a standard error read at its studentised quantile. ``HorizonConformal`` reads the
tube's profile as the weights of one conformal score over the horizon, so the band
``scale * r_k`` holds every step at once at ``1 - delta`` on windows of the logging policy;
``channel_region`` states the self-normalised ellipsoid of the fitted channel (ADR 0099). Neither
moves a number of the tube.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from chc.dynamics import LinearDynamics
from chc.dynamics_id import ChannelRegion, channel_region, fit_causal_residual
from chc.integrate import rk4_step
from chc.uncertainty import (
    HorizonConformal,
    _conformal_rank,
    _kl_floor,
    _noise_tube,
    _shift_gap,
    gaussian_mixing,
    horizon_errors,
    mixing_penalty,
)

_DT = 0.2
_RATE = -0.5
_SIGMA = 0.05
# RK4's step of x' = -0.5 x at dt = 0.2, the AR(1) coefficient of the gap
_STEP = 1.0 - 0.1 + 0.1**2 / 2.0 - 0.1**3 / 6.0 + 0.1**4 / 24.0
_DRAWS = 20_000


def _ar1_errors(rng: np.random.Generator, windows: int, horizon: int) -> np.ndarray:
    """``(windows, horizon, 1)``: the gap ``e_{k+1} = M e_k + w_k`` from ``e_0 = 0``."""
    noise = rng.normal(0.0, _SIGMA, (windows, horizon))
    gap = np.zeros((windows, horizon))
    carried = np.zeros(windows)
    for k in range(horizon):
        carried = _STEP * carried + noise[:, k]
        gap[:, k] = carried
    return gap[:, :, None]


def _profile(horizon: int) -> list[float]:
    """The exact tube's noise radii ``r_1..r_H`` on that plant, at 0.95 a step."""
    tube = _noise_tube(np.array([[_RATE]]), np.array([[_SIGMA**2]]), 0.0, _DT, horizon)
    return [float(radius) for radius in tube[1:]]


@pytest.mark.parametrize("n", range(1, 13))
def test_the_scale_covers_rank_over_n_plus_one_of_every_exchangeable_draw(n: int) -> None:
    """Exhaustively, for ``n <= 12``: given the ``n + 1`` scores, each is the test window with
    probability ``1 / (n + 1)`` under exchangeability, and the other ``n`` calibrate. The share of
    them the band covers is ``k / (n + 1)``, ``k = ceil((n + 1)(1 - delta))``, which lies in
    ``[1 - delta, 1 - delta + 1 / (n + 1))``; with ties it covers at least that."""
    rng = np.random.default_rng(n)
    for delta_twentieths in range(1, 11):
        delta = delta_twentieths / 20.0
        rank = _conformal_rank(n, delta)
        for scores in (rng.permutation(n + 1) + 1.0, rng.integers(1, 4, n + 1).astype(float)):
            if rank > n:
                with pytest.raises(ValueError, match="cannot certify a joint level"):
                    HorizonConformal.calibrate(scores[1:, None, None], [1.0], delta)
                continue
            covered = 0
            for test in range(n + 1):
                rest = np.delete(scores, test)[:, None, None]
                band = HorizonConformal.calibrate(rest, [1.0], delta)
                covered += int(band.covers(scores[test : test + 1, None, None])[0])
            share = Fraction(covered, n + 1)
            exact = Fraction(rank, n + 1)
            low = 1 - Fraction(delta_twentieths, 20)
            assert low <= exact < low + Fraction(1, n + 1)
            if len(set(scores.tolist())) == n + 1:
                assert share == exact
            else:
                assert share >= exact


def test_every_ordering_of_seven_scores_gives_the_same_share() -> None:
    """The exchangeability argument written out: over all ``8!`` orderings of eight distinct
    scores, the last one, the test window, is covered on ``k / 8`` of them."""
    scores = np.arange(1.0, 9.0)
    covered = total = 0
    for ordering in itertools.permutations(range(8)):
        drawn = scores[list(ordering)]
        band = HorizonConformal.calibrate(drawn[:7, None, None], [1.0], 0.25)
        covered += int(band.covers(drawn[7:, None, None])[0])
        total += 1
    assert Fraction(covered, total) == Fraction(_conformal_rank(7, 0.25), 8) == Fraction(6, 8)


def test_the_band_holds_every_step_at_once_at_its_level_on_an_ar1_plant() -> None:
    """20,000 draws of 39 calibration windows and one test window, 8 steps each, on AR(1) noise,
    scored against the exact tube's own profile: the band holds the whole test window on
    ``36 / 40 = 0.9`` of draws, within four Monte Carlo standard errors, inside
    ``[1 - delta, 1 - delta + 1 / (n + 1)]``."""
    rng = np.random.default_rng(93)
    horizon, n, delta = 8, 39, 0.1
    profile = _profile(horizon)
    errors = _ar1_errors(rng, _DRAWS * (n + 1), horizon).reshape(_DRAWS, n + 1, horizon, 1)
    held = np.array(
        [
            HorizonConformal.calibrate(draw[:n], profile, delta).covers(draw[n:])[0]
            for draw in errors
        ]
    )
    share = float(np.mean(held))
    se = math.sqrt(0.9 * 0.1 / _DRAWS)
    assert abs(share - 36 / 40) <= 4.0 * se
    assert 1.0 - delta - 4.0 * se <= share <= 1.0 - delta + 1.0 / (n + 1) + 4.0 * se


def test_the_tube_at_095_a_step_holds_two_steps_together_at_0915() -> None:
    """Each step of the exact tube holds the AR(1) gap at 0.95; both steps of two together hold it
    at 0.9152476 (``validation/tube_levels.mac`` (4)), within four Monte Carlo standard errors."""
    rng = np.random.default_rng(7)
    errors = _ar1_errors(rng, 4 * _DRAWS, 2)[:, :, 0]
    radii = np.array(_profile(2))
    per_step = np.mean(np.abs(errors) <= radii, axis=0)
    joint = float(np.mean(np.all(np.abs(errors) <= radii, axis=1)))
    se = math.sqrt(0.95 * 0.05 / (4 * _DRAWS))
    assert np.all(np.abs(per_step - 0.95) <= 4.0 * se)
    assert abs(joint - 0.9152476) <= 4.0 * math.sqrt(0.915 * 0.085 / (4 * _DRAWS))


def test_the_errors_are_the_plant_s_noise_carried_by_the_model_s_steps() -> None:
    """On a log of a linear plant the model is, the window's error at step ``k`` is the noise
    carried by RK4's step: ``sum_j M^(k - 1 - j) w_{t + j}``."""
    rng = np.random.default_rng(3)
    model = LinearDynamics(jnp.array([[_RATE]]), jnp.array([[1.0]]))
    steps = 40
    actions = rng.normal(size=(steps, 1))
    noise = rng.normal(0.0, _SIGMA, steps)
    states = np.zeros((steps + 1, 1))
    for t in range(steps):
        clean = rk4_step(model, 0.0, jnp.asarray(states[t]), jnp.asarray(actions[t]), _DT)
        states[t + 1] = np.asarray(clean) + noise[t]
    errors = horizon_errors(model, states, actions, _DT, 5, stride=3)
    starts = list(range(0, steps - 5 + 1, 3))
    assert errors.shape == (len(starts), 5, 1)
    expected = np.array(
        [
            [sum(_STEP ** (k - 1 - j) * noise[start + j] for j in range(k)) for k in range(1, 6)]
            for start in starts
        ]
    )
    np.testing.assert_allclose(errors[:, :, 0], expected, rtol=1e-10, atol=1e-14)
    assert horizon_errors(model, states, actions, _DT, 5).shape == (8, 5, 1)


def test_horizon_errors_refuse_a_log_that_does_not_pair_or_hold_a_window() -> None:
    model = LinearDynamics(jnp.array([[_RATE]]), jnp.array([[1.0]]))
    with pytest.raises(ValueError, match="one action"):
        horizon_errors(model, np.zeros((5, 1)), np.zeros((5, 1)), _DT, 2)
    with pytest.raises(ValueError, match="holds no window"):
        horizon_errors(model, np.zeros((3, 1)), np.zeros((2, 1)), _DT, 3)
    with pytest.raises(ValueError, match="at least 1"):
        horizon_errors(model, np.zeros((5, 1)), np.zeros((4, 1)), _DT, 2, stride=0)
    with pytest.raises(ValueError, match="finite"):
        horizon_errors(model, np.full((5, 1), np.nan), np.zeros((4, 1)), _DT, 2)


def test_the_scale_is_the_ranked_score_and_the_band_the_profile_times_it() -> None:
    errors = np.array([[[3.0], [1.0]], [[1.0], [8.0]], [[2.0], [2.0]], [[0.5], [0.5]]])
    profile = [1.0, 4.0]  # scores: max(3, 1/4) = 3, max(1, 2) = 2, 2, 0.5
    band = HorizonConformal.calibrate(np.tile(errors, (5, 1, 1)), profile, 0.2)
    assert band.windows == 20
    assert band.rank == 17  # ceil(21 * 0.8)
    assert band.scale == 3.0  # sorted: 0.5 x5, 2 x10, 3 x5; the 17th is 3
    assert band.band == (3.0, 12.0)
    assert band.level == pytest.approx(0.8, abs=0.0, rel=1e-15)
    assert band.eta is None
    assert band.covers(errors).tolist() == [True, True, True, True]
    assert band.covers(np.array([[[3.0], [12.5]]])).tolist() == [False]


def test_calibrate_refuses_a_profile_that_does_not_pair_or_has_a_zero() -> None:
    errors = np.ones((30, 3, 2))
    with pytest.raises(ValueError, match="one row per step"):
        HorizonConformal.calibrate(errors, [1.0, 1.0], 0.1)
    with pytest.raises(ValueError, match="positive and finite"):
        HorizonConformal.calibrate(errors, [1.0, 0.0, 1.0], 0.1)
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        HorizonConformal.calibrate(errors, [1.0, 1.0, 1.0], 1.0)
    with pytest.raises(ValueError, match="with mixing only"):
        HorizonConformal.calibrate(errors, [1.0, 1.0, 1.0], 0.1, train_lag=3)


def test_one_state_s_mixing_bound_is_its_closed_form_and_windows_read_it_past_their_span() -> None:
    beta = gaussian_mixing([[0.8]], [[1.0]])
    for k in (1, 2, 5, 30):
        assert beta(k) == pytest.approx(0.5 * math.sqrt(-math.log(1.0 - 0.64**k)), rel=1e-12)
    windows = gaussian_mixing([[0.8]], [[1.0]], stride=4, span=3)
    assert windows(1) == pytest.approx(beta(1), rel=1e-12)
    assert windows(2) == pytest.approx(beta(5), rel=1e-12)
    assert gaussian_mixing([[0.8]], [[1.0]], stride=3, span=3)(1) == 1.0


def test_two_states_mixing_bound_is_the_mean_kl_s_closed_form() -> None:
    """``beta(k) <= sqrt(-log det(I - S^-1 F^k S F^k')) / 2`` (``validation/tube_levels.mac``
    (2)), computed here by the determinant, which the library reads by eigenvalues."""
    from scipy.linalg import solve_discrete_lyapunov

    transition = np.array([[0.7, 0.3], [-0.2, 0.5]])
    covariance = np.array([[1.0, 0.3], [0.3, 0.5]])
    stationary = solve_discrete_lyapunov(transition, covariance)
    beta = gaussian_mixing(transition, covariance)
    for k in (1, 3, 8):
        carried = np.linalg.matrix_power(transition, k)
        inner = np.eye(2) - np.linalg.solve(stationary, carried @ stationary @ carried.T)
        assert beta(k) == pytest.approx(0.5 * math.sqrt(-math.log(np.linalg.det(inner))), rel=1e-9)
    with pytest.raises(ValueError, match="not stable"):
        gaussian_mixing([[1.0]], [[1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        gaussian_mixing([[0.5]], [[0.0]])


def _penalty_by_hand(beta: list[float], n: int, slacks: np.ndarray) -> float:
    """Oliveira et al.'s ``eps_cal + delta_cal``, written out loop by loop."""
    best = math.inf
    for a in range(1, n + 1):
        spread = math.sqrt(0.25 + 2.0 / a * sum((a - j) * beta[j] for j in range(1, a)))
        for m in range(1, n + 1):
            r = n + 1 - 2 * m * a
            if r < 1:
                break
            floor = 4.0 * (m - 1) * beta[a] + beta[r]
            for slack in slacks:
                log_term = math.log(4.0 / slack)
                value = (
                    spread * math.sqrt(4.0 / (n - r + 1) * log_term)
                    + log_term / (3.0 * m)
                    + (r - 1) / n
                    + floor
                    + slack
                )
                best = min(best, value)
    return best


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_the_mixing_penalty_is_the_theorem_s_least_over_its_blocks(rho: float) -> None:
    n = 40
    chain = (lambda k: 0.0) if rho == 0.0 else gaussian_mixing([[rho]], [[1.0]])
    beta = [1.0] + [chain(j) for j in range(1, n + 1)]
    expected = _penalty_by_hand(beta, n, np.geomspace(1e-9, 1.0, 241))
    assert mixing_penalty(chain, n) == pytest.approx(expected, rel=1e-12)
    assert mixing_penalty(chain, n, train_lag=4) == pytest.approx(expected + chain(4), rel=1e-12)


def test_the_penalty_falls_with_the_windows_and_rises_with_the_dependence() -> None:
    fast, slow = gaussian_mixing([[0.3]], [[1.0]]), gaussian_mixing([[0.95]], [[1.0]])
    assert mixing_penalty(fast, 2000) < mixing_penalty(fast, 200)
    assert mixing_penalty(fast, 500) < mixing_penalty(slow, 500)


def test_one_series_states_its_level_less_eta_and_refuses_where_eta_reaches_delta() -> None:
    rng = np.random.default_rng(11)
    profile = [1.0, 1.0]
    beta = gaussian_mixing([[0.5]], [[1.0]], stride=3, span=2)
    small = rng.normal(size=(200, 2, 1))
    with pytest.raises(ValueError, match="no joint level is left"):
        HorizonConformal.calibrate(small, profile, 0.05, mixing=beta)
    large = rng.normal(size=(20_000, 2, 1))
    band = HorizonConformal.calibrate(large, profile, 0.2, mixing=beta, train_lag=50)
    eta = mixing_penalty(beta, 20_000, 50)
    assert band.eta == eta
    assert 0.0 < eta < 0.2
    assert band.level == pytest.approx(0.8 - eta, abs=0.0, rel=1e-15)
    assert band.scale == HorizonConformal.calibrate(large, profile, 0.2).scale


# --- the shift budget -----------------------------------------------------------------------

# conformal_shift_budget.mac (3): (delta, rho) -> 1 - beta*, the KL root's gap, to 40 digits
_KL_GAPS = {
    (0.1, 0.05): 3.127839627227936540300086934407307430606e-2,
    (0.05, 0.01): 2.49811448166510254755891669892780834939e-2,
    (0.2, 0.5): 6.913622143754781967040144623535897546016e-3,
    (0.1, 0.01): 6.291062984725830877598039935399332922351e-2,
    (0.01, 0.1): 1.678597852949661221102213745746450938664e-7,
}


def _uniform_draws(rng: np.random.Generator, n: int) -> np.ndarray:
    """``(_DRAWS, n, 1, 1)`` calibration scores drawn from the uniform law on ``[0, 1)``."""
    return rng.uniform(size=(_DRAWS, n, 1, 1))


def _shares(
    calibration: np.ndarray, fresh: np.ndarray, delta: float, shift: tuple[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """The share of the fresh scores, ``(laws, _DRAWS)``, one a draw for each law, that the
    shifted band and the unshifted one cover, each calibrated on the draw's own scores."""
    shifted = np.zeros(fresh.shape[0])
    unshifted = np.zeros(fresh.shape[0])
    for draw, scores in zip(calibration, fresh.T, strict=True):
        tests = scores[:, None, None]
        shifted += HorizonConformal.calibrate(draw, [1.0], delta, shift=shift).covers(tests)
        unshifted += HorizonConformal.calibrate(draw, [1.0], delta).covers(tests)
    return shifted / _DRAWS, unshifted / _DRAWS


@pytest.mark.parametrize("kind", ["tv", "kl"])
def test_a_shift_of_radius_zero_reads_the_band_without_one(kind: str) -> None:
    """At a radius of 0 the rank, the scale and the level are the unshifted band's, and too few
    windows are refused alike, over every ``n <= 40`` and ``delta`` in twentieths."""
    rng = np.random.default_rng(5)
    for n in range(1, 41):
        errors = rng.normal(size=(n, 3, 2))
        for twentieths in range(1, 20):
            delta = twentieths / 20.0
            if _conformal_rank(n, delta) > n:
                for shift in (None, (kind, 0.0)):
                    with pytest.raises(ValueError, match="cannot certify a joint level"):
                        HorizonConformal.calibrate(errors, [1.0, 2.0, 0.5], delta, shift=shift)
                continue
            plain = HorizonConformal.calibrate(errors, [1.0, 2.0, 0.5], delta)
            zero = HorizonConformal.calibrate(errors, [1.0, 2.0, 0.5], delta, shift=(kind, 0.0))
            assert plain.rank == _conformal_rank(n, delta)
            assert zero.shift == (kind, 0.0)
            assert dataclasses.replace(zero, shift=None) == plain
            assert zero.level == plain.level


def test_the_tv_rank_reads_one_less_delta_plus_eps_in_exact_arithmetic() -> None:
    """conformal_shift_budget.mac (1): the rank is ``ceil((n + 1)(1 - delta + eps))`` as written.
    At ``n = 59``, ``delta = 0.1`` and ``eps = 0.05`` it is 57, where the floats' ``60 * 0.95``
    reads 57.00000000000001 and asks for the 58th; 19 windows certify that level, 18 do not."""
    rng = np.random.default_rng(9)
    for n in range(24, 80):
        for delta, eps in ((0.1, 0.05), (0.2, 0.15), (0.05, 0.01), (0.3, 0.0)):
            band = HorizonConformal.calibrate(
                rng.normal(size=(n, 1, 1)), [1.0], delta, shift=("tv", eps)
            )
            exact = (n + 1) * (1 - Fraction(str(delta)) + Fraction(str(eps)))
            assert band.rank == math.ceil(exact)
            assert band.level == pytest.approx(1.0 - delta, abs=0.0, rel=1e-15)
    assert math.ceil(60 * (1 - 0.1 + 0.05)) == 58
    scores = np.arange(1.0, 60.0)[:, None, None]
    assert HorizonConformal.calibrate(scores, [1.0], 0.1, shift=("tv", 0.05)).scale == 57.0
    with pytest.raises(ValueError, match="needs at least 19 of them"):
        HorizonConformal.calibrate(scores[:18], [1.0], 0.1, shift=("tv", 0.05))
    assert HorizonConformal.calibrate(scores[:19], [1.0], 0.1, shift=("tv", 0.05)).rank == 19


def test_the_band_holds_its_level_where_a_tv_shift_moves_eps_of_the_windows_beyond_it() -> None:
    """conformal_shift_budget.mac (1): 20,000 draws of 39 calibration windows and one fresh one,
    scores uniform on ``[0, 1)``. Two laws of the fresh score are at a total variation of exactly
    ``eps = 0.05``: the scores below ``eps`` moved to 2, beyond every band, the worst case, and a
    share ``eps`` of the windows at 2, the mixture. The shifted band, the 38th score, holds the
    first on ``38/40 - eps = 0.9 = 1 - delta`` of draws in expectation and the second on 0.9025;
    the unshifted one, the 36th, on 0.85 and 0.855: below its level, by more than ten standard
    errors."""
    rng = np.random.default_rng(31)
    n, delta, eps = 39, 0.1, 0.05
    calibration = _uniform_draws(rng, n)
    fresh = rng.uniform(size=_DRAWS)
    moved = np.where(fresh < eps, 2.0, fresh)
    mixed = np.where(rng.uniform(size=_DRAWS) < eps, 2.0, rng.uniform(size=_DRAWS))
    se = math.sqrt(0.9 * 0.1 / _DRAWS)
    shifted, unshifted = _shares(calibration, np.stack([moved, mixed]), delta, ("tv", eps))
    assert np.all(np.abs(shifted - [0.9, 0.9025]) <= 4.0 * se)
    assert np.all(np.abs(unshifted - [0.85, 0.855]) <= 4.0 * se)
    assert np.all(shifted >= 1.0 - delta - 4.0 * se)
    assert np.all(unshifted < 1.0 - delta - 10.0 * se)


def test_the_kl_level_is_maxima_s_root_and_grows_with_the_radius() -> None:
    """conformal_shift_budget.mac (3) and (4): the gap ``1 - beta*`` is Maxima's to 1e-12 of
    itself, ``beta*`` to 1e-10, and puts the divergence back at ``rho``; ``beta*`` and the rank
    grow with ``rho`` from ``1 - delta`` at 0, by ``sqrt(2 delta (1 - delta) rho)`` near it. At
    ``delta = 0.1``, ``rho = 0.05`` 31 windows certify the level, 30 do not."""
    for (delta, rho), gap in _KL_GAPS.items():
        found = float(_shift_gap(delta, ("kl", rho)))
        assert found == pytest.approx(gap, rel=1e-12, abs=0.0)
        assert abs((1.0 - found) - (1.0 - gap)) <= 1e-10
        level = 1.0 - found
        divergence = (1 - delta) * math.log((1 - delta) / level) + delta * math.log(delta / found)
        assert divergence == pytest.approx(rho, rel=1e-9, abs=0.0)
    radii = [0.0, 1e-6, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    gaps = [float(_shift_gap(0.1, ("kl", rho))) for rho in radii]
    assert gaps[0] == 0.1
    assert all(later < earlier for earlier, later in itertools.pairwise(gaps))
    # (4): near 0 the gap falls by sqrt(2 delta (1 - delta) rho), to a relative O(sqrt(rho))
    first_order = math.sqrt(2.0 * 0.1 * 0.9 * 1e-6)
    assert (0.1 - gaps[1]) / first_order == pytest.approx(0.9987428172926519, rel=1e-9, abs=0.0)
    errors = np.random.default_rng(2).normal(size=(3000, 1, 1))
    ranks = [HorizonConformal.calibrate(errors, [1.0], 0.1, shift=("kl", r)).rank for r in radii]
    assert ranks[0] == _conformal_rank(3000, 0.1)
    assert all(later >= earlier for earlier, later in itertools.pairwise(ranks))
    assert ranks[-1] > ranks[0]
    with pytest.raises(ValueError, match="needs at least 31 of them"):
        HorizonConformal.calibrate(errors[:30], [1.0], 0.1, shift=("kl", 0.05))
    assert HorizonConformal.calibrate(errors[:31], [1.0], 0.1, shift=("kl", 0.05)).rank == 31


def test_the_band_holds_its_level_under_the_kl_tilt_at_the_radius() -> None:
    """conformal_shift_budget.mac (6): 20,000 draws of 99 calibration windows and one fresh one,
    scores uniform on ``[0, 1)``. The fresh score's law puts ``1 - delta`` on ``[0, beta*)`` and
    the rest above, a divergence of exactly ``rho = 0.05``, ``beta* = 0.96872``. The shifted band,
    the 97th score, holds it on 0.91791 of draws in expectation; the unshifted one, the 90th, on
    0.83616."""
    rng = np.random.default_rng(37)
    n, delta, rho = 99, 0.1, 0.05
    level = 1.0 - float(_shift_gap(delta, ("kl", rho)))
    divergence = (1 - delta) * math.log((1 - delta) / level) + delta * math.log(
        delta / (1.0 - level)
    )
    assert divergence == pytest.approx(rho, rel=1e-12, abs=0.0)
    calibration = _uniform_draws(rng, n)
    below = rng.uniform(size=_DRAWS) < 1.0 - delta
    fresh = np.where(below, 0.0, level) + rng.uniform(size=_DRAWS) * np.where(
        below, level, 1.0 - level
    )
    [shifted], [unshifted] = _shares(calibration, fresh[None], delta, ("kl", rho))
    se = math.sqrt(0.9 * 0.1 / _DRAWS)
    assert abs(shifted - 0.9179050763393374) <= 4.0 * se
    assert abs(unshifted - 0.8361625394282504) <= 4.0 * se
    assert shifted >= 1.0 - delta
    assert unshifted < 1.0 - delta - 10.0 * se


def test_one_series_under_a_shift_states_what_the_ball_leaves_of_beta_less_eta() -> None:
    """On one series the level is ``g(beta* - eta)``: ``1 - delta - eta`` under total variation;
    under KL the root ``z < beta* - eta`` of ``KL(z || beta* - eta) = rho``, below
    ``1 - delta - eta`` (conformal_shift_budget.mac (5): 0.86573 and 0.81989 at ``eta`` of 0.02
    and 0.05, ``delta = 0.1``, ``rho = 0.05``). Where the ball can take all that is left, as at
    ``delta = 0.4``, ``rho = 1`` and the ``eta = 0.356`` of 64 windows, it refuses."""
    level = 1.0 - float(_shift_gap(0.1, ("kl", 0.05)))
    assert _kl_floor(level - 0.02, 0.05) == pytest.approx(0.8657253624004547, rel=1e-12, abs=0.0)
    assert _kl_floor(level - 0.05, 0.05) == pytest.approx(0.8198868926952535, rel=1e-12, abs=0.0)
    rng = np.random.default_rng(11)
    profile = [1.0, 1.0]
    beta = gaussian_mixing([[0.5]], [[1.0]], stride=3, span=2)
    large = rng.normal(size=(20_000, 2, 1))
    tv = HorizonConformal.calibrate(
        large, profile, 0.2, mixing=beta, train_lag=50, shift=("tv", 0.05)
    )
    eta = tv.eta
    assert eta is not None
    assert 0.0 < eta < 0.2
    assert tv.level == pytest.approx(0.8 - eta, abs=0.0, rel=1e-15)
    assert tv.scale == HorizonConformal.calibrate(large, profile, 0.2, shift=("tv", 0.05)).scale
    kl = HorizonConformal.calibrate(
        large, profile, 0.2, mixing=beta, train_lag=50, shift=("kl", 0.05)
    )
    assert kl.eta == eta
    held = 1.0 - float(_shift_gap(0.2, ("kl", 0.05))) - eta
    z = kl.level
    assert 0.0 < z < 0.8 - eta
    divergence = z * math.log(z / held) + (1 - z) * math.log((1 - z) / (1 - held))
    assert divergence == pytest.approx(0.05, rel=1e-9, abs=0.0)
    independent = rng.normal(size=(64, 1, 1))
    assert HorizonConformal.calibrate(independent, [1.0], 0.4, mixing=lambda j: 0.0).level > 0.2
    with pytest.raises(ValueError, match=r"can take all of the 0\.628"):
        HorizonConformal.calibrate(independent, [1.0], 0.4, mixing=lambda j: 0.0, shift=("kl", 1.0))


def test_calibrate_refuses_a_shift_it_cannot_read_or_hold() -> None:
    """A ball it does not know, a radius it cannot read, a total variation of ``delta`` or more,
    and a KL radius past the windows: 568,532 hold ``rho = 1`` at ``delta = 0.1``
    (conformal_shift_budget.mac (3)), and past 1e15 the floats no longer count them."""
    errors = np.ones((400, 2, 1))
    for shift, message in (
        (("wasserstein", 0.1), "the ball is 'tv' or 'kl'"),
        (("tv", -0.01), "finite and at least 0"),
        (("kl", math.nan), "finite and at least 0"),
        (("kl", math.inf), "finite and at least 0"),
        (("tv", 0.1), "no number of calibration windows certifies"),
        (("tv", 0.5), "the band is infinite"),
        (("kl", 1.0), "needs at least 568532 of them"),
        (("kl", 5.0), "needs more than 1e15 of them"),
    ):
        with pytest.raises(ValueError, match=message):
            HorizonConformal.calibrate(errors, [1.0, 1.0], 0.1, shift=shift)  # type: ignore[arg-type]


# --- the channel's region -------------------------------------------------------------------

_TRUTH = np.array([[[1.0, 0.0]]])  # x' = -0.5 x + 1.0 u + 0.8 z: the channel, constant in x


def _closed_loop_log(seed: int, rows: int = 1500) -> dict[str, jnp.ndarray]:
    """One unit under a feedback policy, ``u = -0.5 x + 0.8 z + own draw``, Euler steps of
    ``x' = -0.5 x + u + 0.8 z`` with noise: ``z`` is logged; ``decoy`` is a column of noise."""
    rng = np.random.default_rng(seed)
    x = 0.0
    cols: dict[str, list[list[float]]] = {name: [] for name in ("x", "u", "x_next", "z", "decoy")}
    for _ in range(rows):
        z = rng.normal()
        u = -0.5 * x + 0.8 * z + 0.5 * rng.normal()
        x_next = x + _DT * (-0.5 * x + u + 0.8 * z) + _SIGMA * rng.normal()
        for name, value in (("x", x), ("u", u), ("x_next", x_next), ("z", z)):
            cols[name].append([value])
        cols["decoy"].append([rng.normal()])
        x = x_next
    return {name: jnp.asarray(values) for name, values in cols.items()}


_KNOWN = LinearDynamics(jnp.zeros((1, 1)), jnp.zeros((1, 1)))


def _region(seed: int, adjust_for: str, level: float = 0.9) -> ChannelRegion:
    fit = fit_causal_residual(
        _KNOWN, _closed_loop_log(seed), _DT, adjust_for=(adjust_for,), integrator="euler"
    )
    return channel_region(fit, level)


def test_the_region_holds_the_channel_at_its_level_on_closed_loop_logs() -> None:
    """60 closed-loop logs, the confounder adjusted for: the region at 0.9 holds the channel on
    every one. The theorem's region is wide, since it holds at every sample size at once."""
    held = [_region(seed, "z").contains(_TRUTH) for seed in range(60)]
    assert np.mean(held) >= 0.9


def test_the_region_misses_the_channel_under_a_hidden_confounder() -> None:
    """The same logs adjusted for a column of noise in place of ``z``: the confounder pushes the
    rate with the action, the fit reads about 1.7 for 1.0, and the region holds it on none."""
    held = [_region(seed, "decoy").contains(_TRUTH) for seed in range(30)]
    assert np.mean(held) <= 0.05


def test_the_region_is_the_ellipsoid_written_out() -> None:
    fit = fit_causal_residual(
        _KNOWN, _closed_loop_log(0), _DT, adjust_for=("z",), integrator="euler"
    )
    region = channel_region(fit, 0.9)
    assert fit._channel_gram is not None
    assert fit.moment_residual is not None
    gram, ridge = fit._channel_gram.matrix, fit._channel_gram.ridge
    residual = np.asarray(fit.moment_residual)
    noise = math.sqrt(float(np.sum(residual**2)) / (residual.shape[0] - 2))
    log_ratio = np.linalg.slogdet(gram)[1] - np.sum(np.log(ridge))
    coefficients = np.asarray(fit.residual.channel).ravel()
    prior = math.sqrt(float(np.sum(ridge * coefficients**2)))
    expected = noise * math.sqrt(2.0 * math.log(1.0 / 0.1) + log_ratio) + prior
    assert region.radius[0] == pytest.approx(expected, rel=1e-12)
    assert np.array_equal(region.channel, np.asarray(fit.residual.channel))
    widened = channel_region(fit, 0.9, bound=1.0)
    assert widened.radius[0] == pytest.approx(expected - prior + 1.0, rel=1e-12)
    # a channel off the centre by the radius in the Gram's norm sits on the edge
    vector = np.linalg.eigh(gram)[1][:, 0]
    edge = vector * region.radius[0] / math.sqrt(vector @ gram @ vector)
    assert region.contains(region.channel + 0.999 * edge.reshape(1, 1, 2))
    assert not region.contains(region.channel + 1.001 * edge.reshape(1, 1, 2))


def test_the_operator_radius_and_the_rate_error_read_the_same_ellipsoid() -> None:
    region = _region(1, "z")
    state = np.array([0.4])
    radius = region.operator_radius(state)
    assert region.rate_error(state[None], [2.5]) == pytest.approx(2.5 * radius, rel=1e-12)
    assert region.rate_error(np.array([[0.4], [0.0]]), [1.0]) >= radius
    # the stability level inverts the radius: the level at which the radius is rho is rho's level
    assert region.stability_level(radius, state) == pytest.approx(0.9, rel=1e-9)
    assert (
        region.stability_level(0.5 * radius, state)
        < 0.9
        < region.stability_level(2.0 * radius, state)
    )
    assert region.stability_level(0.0, state) == 0.0


@pytest.mark.parametrize("flip", [1.0, -1.0])
def test_two_levers_rate_error_is_the_largest_over_the_box_s_corners(flip: float) -> None:
    """The form is even in the action, so two of the four corners are the others' mirrors; which
    pair holds the largest turns on the levers' coupling, so the Gram is read with the second
    lever's sign flipped as well."""
    rng = np.random.default_rng(2)
    root = rng.normal(size=(4, 4))
    sign = np.diag([1.0, 1.0, flip, flip])
    gram = sign @ (root @ root.T + np.eye(4)) @ sign
    region = ChannelRegion(
        level=0.9,
        channel=np.zeros((1, 2, 2)),
        gram=gram,
        radius=np.array([1.3]),
        noise=np.array([1.0]),
        prior=np.array([0.0]),
        log_det_ratio=3.0,
        channel_degree=1,
    )
    state, features, reach = np.array([0.7]), np.array([1.0, 0.7]), np.array([0.5, 2.0])

    def bound_at(action: np.ndarray) -> float:
        row = np.kron(action, features)
        return 1.3 * math.sqrt(float(row @ np.linalg.solve(gram, row)))

    corners = [np.array([a, b]) * reach for a in (-1.0, 1.0) for b in (-1.0, 1.0)]
    inside = [bound_at(rng.uniform(-1.0, 1.0, 2) * reach) for _ in range(2000)]
    found = region.rate_error(state[None], reach)
    assert found == pytest.approx(max(bound_at(corner) for corner in corners), rel=1e-12)
    assert max(inside) <= found
    assert bound_at(reach) != pytest.approx(bound_at(reach * np.array([1.0, -1.0])), rel=1e-3)


def test_channel_region_refuses_the_fits_whose_estimate_is_not_least_squares() -> None:
    data = _closed_loop_log(0, rows=400)
    rk4 = fit_causal_residual(_KNOWN, data, _DT, adjust_for=("z",), integrator="rk4")
    assert rk4._channel_gram is None
    with pytest.raises(ValueError, match="integrator='euler'"):
        channel_region(rk4)
    iv = fit_causal_residual(_KNOWN, data, _DT, instrument="z", integrator="euler")
    assert iv._channel_gram is None
    with pytest.raises(ValueError, match="instrument's moment"):
        channel_region(iv)
    unidentified = fit_causal_residual(_KNOWN, data, _DT, integrator="euler")
    with pytest.raises(ValueError, match="does not identify"):
        channel_region(unidentified)
    weighted = fit_causal_residual(
        _KNOWN,
        data,
        _DT,
        adjust_for=("z",),
        integrator="euler",
        weights=lambda x: 1.0 + x[:, 0] ** 2,
    )
    assert weighted._channel_gram is None
    with pytest.raises(ValueError, match="weighed its moment"):
        channel_region(weighted)
    fit = fit_causal_residual(_KNOWN, data, _DT, adjust_for=("z",), integrator="euler")
    for level in (0.0, 1.0):
        with pytest.raises(ValueError, match="in \\(0, 1\\)"):
            channel_region(fit, level)
    with pytest.raises(ValueError, match="non-negative"):
        channel_region(fit, bound=-1.0)


def test_two_states_share_the_level_and_add_their_radii_in_quadrature() -> None:
    """Each state's ellipsoid holds at ``delta / n``, so the joint holds at ``1 - delta``; an
    action's rate error adds the states' radii in quadrature."""
    gram = np.array([[4.0, 1.0], [1.0, 2.0]])
    region = ChannelRegion(
        level=0.9,
        channel=np.zeros((2, 1, 2)),
        gram=gram,
        radius=np.zeros(2),
        noise=np.array([0.3, 0.7]),
        prior=np.array([0.01, 0.02]),
        log_det_ratio=5.0,
        channel_degree=1,
    )
    radii = region._radii(0.1)
    expected = np.array([0.3, 0.7]) * math.sqrt(2.0 * math.log(2.0 / 0.1) + 5.0) + [0.01, 0.02]
    np.testing.assert_allclose(radii, expected, rtol=1e-14)
    region = dataclasses.replace(region, radius=radii)
    state = np.array([0.5])
    row = np.array([1.0, 0.5])
    unit = math.sqrt(float(row @ np.linalg.solve(gram, row)))
    assert region.operator_radius(state) == pytest.approx(
        math.sqrt(float(np.sum(radii**2))) * unit, rel=1e-12
    )
    assert region.stability_level(region.operator_radius(state), state) == pytest.approx(
        0.9, rel=1e-12
    )


def _period_shock_panel(seed: int) -> tuple[dict[str, jnp.ndarray], np.ndarray, np.ndarray]:
    """30 units over 20 periods, ``x' = x + 0.1 (-0.5 x + 0.8 u + 0.3 w + e)``, ``u = 0.5 w + a``:
    each period a shock every unit shares moves the levers' ``a`` and the noise ``e``, each half
    of its variance."""
    rng = np.random.default_rng(seed)
    units, periods = 30, 20
    w = rng.normal(size=(units, periods))
    lever_shock = np.repeat(rng.normal(size=(1, periods)), units, axis=0)
    noise_shock = np.repeat(rng.normal(size=(1, periods)), units, axis=0)
    a = math.sqrt(0.5) * (lever_shock + rng.normal(size=(units, periods)))
    e = math.sqrt(0.5) * (noise_shock + rng.normal(size=(units, periods)))
    x = np.empty((units, periods + 1))
    x[:, 0] = rng.normal(size=units)
    u = 0.5 * w + a
    for t in range(periods):
        x[:, t + 1] = x[:, t] + 0.1 * (-0.5 * x[:, t] + 0.8 * u[:, t] + 0.3 * w[:, t] + e[:, t])
    data = {
        "x": jnp.asarray(x[:, :-1].reshape(-1, 1)),
        "u": jnp.asarray(u.reshape(-1, 1)),
        "w": jnp.asarray(w.reshape(-1, 1)),
        "x_next": jnp.asarray(x[:, 1:].reshape(-1, 1)),
    }
    return data, np.repeat(np.arange(units), periods), np.tile(np.arange(periods), units)


_SHOCK_TRUTH = np.full((1, 1, 1), 0.8)


@pytest.mark.parametrize("seed", [0, 25, 60])
def test_a_fit_told_of_a_shared_shock_holds_the_channel_the_rows_region_misses(seed: int) -> None:
    """Read row by row, a shock every unit shares in a period adds up across its rows beyond each
    row's noise: on 1000 such logs the 0.95 region missed the channel on 6.9 %. On these three it
    misses; the fit given the units and the periods reads the noise within them, and holds it."""
    data, unit, period = _period_shock_panel(seed)
    rows = fit_causal_residual(_KNOWN, data, 0.1, adjust_for=("w",), channel_degree=0)
    assert not channel_region(rows).contains(_SHOCK_TRUTH)
    told = fit_causal_residual(
        _KNOWN,
        data,
        0.1,
        adjust_for=("w",),
        channel_degree=0,
        clusters=np.column_stack([unit, period]),
    )
    assert channel_region(told).contains(_SHOCK_TRUTH)


def _clustered_noise_by_hand(fit, labels: np.ndarray) -> float:
    """``R^2`` of a one-coefficient channel under ``labels``, ``(N,)`` or ``(N, 2)``: the scores
    ``d_i e_i`` summed within the clusters and squared at CR1's factor, two ways the largest of
    the two ways' sums less their cells' and each way's alone at its own factor, over ``V``; the
    channel's design ``d`` read off the representer, ``N V^-1 d_i``."""
    gram = float(fit._channel_gram.matrix[0, 0])
    rows = fit.moment_residual.shape[0]
    design = np.asarray(fit.representer)[:, 0, 0] * gram / rows
    scores = design * np.asarray(fit.moment_residual)[:, 0]

    def summed(codes: np.ndarray) -> tuple[float, int]:
        sums = np.bincount(np.unique(codes, return_inverse=True)[1], weights=scores)
        return float(np.sum(sums**2)), sums.size

    def factor(count: int) -> float:
        return count / (count - 1) * (rows - 1) / (rows - 1)

    if labels.ndim == 1:
        square, count = summed(labels)
        return factor(count) * square / gram
    (first, g1), (second, g2) = summed(labels[:, 0]), summed(labels[:, 1])
    cells, _ = summed(labels[:, 0] * 1000 + labels[:, 1])
    smaller = min(g1, g2)
    both = factor(smaller) * max(first + second - cells, 0.0)
    return max(both, factor(g1) * first, factor(g2) * second) / gram


@pytest.mark.parametrize("two_ways", [False, True])
def test_a_clustered_fit_reads_the_region_s_noise_within_its_clusters(two_ways: bool) -> None:
    """``R^2`` is the larger of the rows' mean square and the clustered scores' spread in ``V``'s
    norm, written out here for one coefficient; a fit without clusters reads the rows' alone."""
    data, unit, period = _period_shock_panel(3)
    labels = np.column_stack([unit, period]) if two_ways else period
    fit = fit_causal_residual(
        _KNOWN, data, 0.1, adjust_for=("w",), channel_degree=0, clusters=labels, influence=True
    )
    rows = fit_causal_residual(_KNOWN, data, 0.1, adjust_for=("w",), channel_degree=0)
    residual = np.asarray(fit.moment_residual)
    own = float(np.sum(residual**2)) / (residual.shape[0] - 1)
    shared = _clustered_noise_by_hand(fit, labels)
    assert shared > 4.0 * own  # the shock adds up across a period's 30 units
    assert channel_region(fit).noise[0] == pytest.approx(math.sqrt(shared), rel=1e-9)
    assert channel_region(rows).noise[0] == pytest.approx(math.sqrt(own), rel=1e-12)
    assert rows._channel_gram is not None
    assert fit._channel_gram == rows._channel_gram  # the region reads the clusters, not the Gram


def test_a_two_way_region_reads_the_same_whichever_dimension_comes_first() -> None:
    """Two ways, the region reads the largest of the two ways' sums and each way's alone, as the
    fit's error does: the order the fit was given its two dimensions in does not move it, and on a
    panel whose shock is the period's it reads above the rows' noise either way."""
    data, unit, period = _period_shock_panel(3)
    rows = channel_region(
        fit_causal_residual(_KNOWN, data, 0.1, adjust_for=("w",), channel_degree=0)
    ).noise[0]
    noise = [
        channel_region(
            fit_causal_residual(
                _KNOWN,
                data,
                0.1,
                adjust_for=("w",),
                channel_degree=0,
                clusters=np.column_stack(order),
            )
        ).noise[0]
        for order in ((unit, period), (period, unit))
    ]
    assert noise[0] > 2.0 * rows
    assert noise[1] == pytest.approx(noise[0], rel=1e-12)


def test_few_clusters_never_read_the_region_s_noise_below_the_rows() -> None:
    """Read from five clusters whose noise is independent, the clustered spread is noisy and can
    fall below each row's own scale, which the theorem needs whatever the clusters say."""
    data, unit, _ = _period_shock_panel(3)
    rng = np.random.default_rng(1)
    data = {
        **data,
        "x_next": data["x"]
        + 0.1 * (-0.5 * data["x"] + 0.8 * data["u"])
        + 0.1 * jnp.asarray(rng.normal(size=data["x"].shape)),
    }
    found = []
    for groups in (5, 6, 7, 8):
        labels = unit % groups
        fit = fit_causal_residual(
            _KNOWN, data, 0.1, adjust_for=("w",), channel_degree=0, clusters=labels, influence=True
        )
        residual = np.asarray(fit.moment_residual)
        own = float(np.sum(residual**2)) / (residual.shape[0] - 1)
        shared = _clustered_noise_by_hand(fit, labels)
        found.append(shared < own)
        expected = math.sqrt(max(own, shared))
        assert channel_region(fit).noise[0] == pytest.approx(expected, rel=1e-9)
    assert any(found)  # a clustering whose spread reads below the rows' is among them


def test_a_channel_affine_in_the_state_reads_its_noise_where_the_clusters_add_up_most() -> None:
    """Two coefficients a state: ``R^2`` is the largest generalised eigenvalue of the clustered
    scores' spread ``M`` against ``V``, the most ``a' M a`` reaches over ``a' V a = 1``."""
    data, _, period = _period_shock_panel(4)
    fit = fit_causal_residual(
        _KNOWN, data, 0.1, adjust_for=("w",), channel_degree=1, clusters=period, influence=True
    )
    assert fit._channel_gram is not None
    gram = fit._channel_gram.matrix
    rows = fit.moment_residual.shape[0]
    design = np.asarray(fit.representer)[:, 0, :] @ gram / rows
    scores = design * np.asarray(fit.moment_residual)[:, :1]
    sums = np.zeros((20, 2))
    np.add.at(sums, period, scores)
    spread = 20 / 19 * (rows - 1) / (rows - 2) * sums.T @ sums
    largest = float(scipy.linalg.eigh(spread, gram, eigvals_only=True)[-1])
    residual = np.asarray(fit.moment_residual)
    own = float(np.sum(residual**2)) / (rows - 2)
    assert largest > own
    smallest = float(scipy.linalg.eigh(spread, gram, eigvals_only=True)[0])
    assert smallest < 0.5 * largest  # the direction matters
    assert channel_region(fit).noise[0] == pytest.approx(math.sqrt(largest), rel=1e-9)
