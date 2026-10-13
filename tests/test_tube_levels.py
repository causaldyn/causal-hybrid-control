"""The tube's levels: one conformal scale for the whole horizon, and a region for the channel.

The plan's tube holds the plant at 0.95 step by step (ADR 0080, ADR 0096), and every step of a
horizon at once at less: on AR(1) noise two steps at 0.95 each hold together 0.915. The channel's
error in it is a one-standard-error scale, with no level. ``HorizonConformal`` reads the tube's
profile as the weights of one conformal score over the horizon, so the band ``scale * r_k`` holds
every step at once at ``1 - delta`` on windows of the logging policy; ``channel_region`` states the
self-normalised ellipsoid of the fitted channel (ADR 0099). Neither moves a number of the tube.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from chc.dynamics import LinearDynamics
from chc.dynamics_id import ChannelRegion, channel_region, fit_causal_residual
from chc.integrate import rk4_step
from chc.uncertainty import (
    HorizonConformal,
    _conformal_rank,
    _noise_tube,
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
