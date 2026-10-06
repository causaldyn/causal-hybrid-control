"""chc.dynamics_id.omitted_confounder_bound: a latent the adjustment set left out, in partial R^2s.

The plant is linear with one observed confounder, adjusted for, and latents that are not. With the
shares the latents actually have, the bound on a linear Gaussian plant is attained when one latent
is left out (``validation/omitted_confounder_bound.mac``, STEPs 1-2) and strict by a known amount
when two pull the lever and the rate apart (STEP 3). So the bound's side must land on the true
channel, not merely beyond it: a bound that is valid but loose passes a coverage test and fails
these.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from chc.dynamics_id import _cross_fit_residuals, fit_causal_residual, omitted_confounder_bound

DRIFT, NOISE, ETA = -0.5, 0.01, 0.5
TO_ACTION, TO_RATE = 0.8, 0.7  # the observed confounder's loadings


def _known(t: float, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros_like(x)


def _t4(z: float) -> float:
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


def _s4(z: float) -> float:
    return 1.0 + z / 2.0 + z**2 / 6.0 + z**3 / 24.0


def _log(
    channel: np.ndarray,
    latent_to_action: np.ndarray,
    latent_to_rate: np.ndarray,
    *,
    dt: float,
    scheme: str,
    n: int = 10_000,
    seed: int = 0,
    walk: float = 0.0,
) -> dict[str, jnp.ndarray]:
    """One state, ``m`` levers, standard latents in the columns of the loadings: ``(m, k)`` on the
    levers and ``(k,)`` on the rate. Stepped the way ``scheme`` reads it, inputs held over a step.
    ``walk > 0`` adds a second state, a random walk with steps of that size that nothing moves.
    """
    rng = np.random.default_rng(seed)
    levers, latents = latent_to_action.shape
    x, observed = rng.standard_normal(n), rng.standard_normal(n)
    latent = rng.standard_normal((n, latents))
    u = TO_ACTION * observed[:, None] + latent @ latent_to_action.T
    u = u + ETA * rng.standard_normal((n, levers))
    push = u @ channel + TO_RATE * observed + latent @ latent_to_rate
    if scheme == "euler":
        after = x + dt * (DRIFT * x + push)
    else:
        after = _t4(DRIFT * dt) * x + dt * _s4(DRIFT * dt) * push
    after = after + NOISE * rng.standard_normal(n)
    states, following = x[:, None], after[:, None]
    if walk > 0.0:
        other = rng.standard_normal(n)
        states = np.column_stack([x, other])
        following = np.column_stack([after, other + walk * rng.standard_normal(n)])
    return {
        "x": jnp.asarray(states),
        "u": jnp.asarray(u),
        "x_next": jnp.asarray(following),
        "observed": jnp.asarray(observed)[:, None],
    }


def _shares(
    latent_to_action: np.ndarray,
    latent_to_rate: np.ndarray,
    functional: np.ndarray,
    *,
    dt: float,
    scheme: str,
) -> tuple[float, float, float, float]:
    """The population's bias on the functional, its shares ``cf_y`` and ``cf_d``, and the bound's
    excess over the bias, all in the rate's units, which the ``rk4`` reading divides the step's
    noise into by ``dt S4(A dt)``."""
    levers = latent_to_action.shape[0]
    short = latent_to_action @ latent_to_action.T + ETA**2 * np.eye(levers)
    solved = np.linalg.solve(short, latent_to_action)  # (m, k)
    bias = float(functional @ solved @ latent_to_rate)
    nu2 = float(functional @ np.linalg.solve(short, functional))
    cf_d = 1.0 - nu2 * ETA**2 / float(functional @ functional)
    explained = float(
        latent_to_rate
        @ (np.eye(len(latent_to_rate)) - latent_to_action.T @ solved)
        @ latent_to_rate
    )
    noise = (NOISE / (dt * (1.0 if scheme == "euler" else _s4(DRIFT * dt)))) ** 2
    cf_y = explained / (explained + noise)
    bound = math.sqrt((explained + noise) * nu2 * cf_y * cf_d / (1.0 - cf_d))
    return bias, cf_y, cf_d, bound - abs(bias)


def _fit(data: dict[str, jnp.ndarray], dt: float, scheme: str):
    return fit_causal_residual(
        _known,
        data,
        dt,
        adjust_for=("observed",),
        degree=1,
        channel_degree=0,
        integrator=scheme,
        influence=True,
    )


def test_the_representer_is_the_moment_s_weight_with_the_nuisances_held() -> None:
    """``v / mean(v^2)``, DoubleML's, to the ridge. Letting the cross-fitted nuisances move with
    the row as well would add their error to it, and grow its second moment."""
    channel, to_action, to_rate = np.array([1.0]), np.array([[1.0]]), np.array([1.0])
    data = _log(channel, to_action, to_rate, dt=0.05, scheme="euler", n=2000)
    fit = _fit(data, 0.05, "euler")
    _, lever, _, _ = _cross_fit_residuals(
        (data["x_next"] - data["x"]) / 0.05,
        data["u"],
        jnp.concatenate([data["x"], data["observed"]], axis=1),
        degree=2,
        folds=2,
        ridge=1e-6,
        seed=0,
    )
    v = np.asarray(lever)[:, 0]
    np.testing.assert_allclose(np.asarray(fit.representer)[:, 0, 0], v / np.mean(v**2), rtol=1e-8)


@pytest.mark.parametrize(("scheme", "dt"), [("euler", 0.05), ("rk4", 0.05), ("rk4", 1.4)])
def test_one_latent_left_out_puts_the_bound_on_the_true_channel(scheme: str, dt: float) -> None:
    """``rk4`` at ``A dt = -0.7``: the fixed point carries the representer, and the Euler reading
    of the same log, which does not, lands far from the channel."""
    channel, to_action, to_rate = np.array([1.0]), np.array([[1.0]]), np.array([1.0])
    one = np.ones(1)
    bias, cf_y, cf_d, excess = _shares(to_action, to_rate, one, dt=dt, scheme=scheme)
    assert excess == pytest.approx(0.0, abs=1e-12)  # attained
    data = _log(channel, to_action, to_rate, dt=dt, scheme=scheme)
    fit = _fit(data, dt, scheme)
    bound = omitted_confounder_bound(fit, np.ones((1, 1, 1)), cf_y=cf_y, cf_d=cf_d)
    assert bound.estimate - 1.0 == pytest.approx(bias, abs=4.0 * fit.channel_error)
    assert bound.lower == pytest.approx(1.0, abs=4.0 * fit.channel_error)
    if dt > 1.0:
        euler = omitted_confounder_bound(
            _fit(data, dt, "euler"), np.ones((1, 1, 1)), cf_y=cf_y, cf_d=cf_d
        )
        assert abs(euler.lower - 1.0) > 0.2


def test_two_latents_pulling_the_lever_and_the_rate_apart_leave_the_predicted_slack() -> None:
    """STEP 3: the bound exceeds the bias by ``(a1 b2 - a2 b1)^2 / (sv V)`` in its square."""
    channel, to_action, to_rate = np.array([1.0]), np.array([[1.0, 0.6]]), np.array([0.4, 1.2])
    bias, cf_y, cf_d, excess = _shares(to_action, to_rate, np.ones(1), dt=0.05, scheme="euler")
    assert excess > 0.1
    fit = _fit(_log(channel, to_action, to_rate, dt=0.05, scheme="euler"), 0.05, "euler")
    bound = omitted_confounder_bound(fit, np.ones((1, 1, 1)), cf_y=cf_y, cf_d=cf_d)
    assert bound.strength * bound.bias_scale == pytest.approx(abs(bias) + excess, rel=0.02)
    assert bound.lower < 1.0 - 0.5 * excess


def test_a_move_between_two_levers_is_bounded_like_one_lever() -> None:
    """STEP 2: one latent, two levers, the functional ``b - a``: attained again."""
    channel, to_action, to_rate = np.array([1.0, 0.4]), np.array([[1.0], [-0.5]]), np.array([1.0])
    move = np.array([-1.0, 1.0])
    bias, cf_y, cf_d, excess = _shares(to_action, to_rate, move, dt=0.05, scheme="euler")
    assert excess == pytest.approx(0.0, abs=1e-12)
    fit = _fit(_log(channel, to_action, to_rate, dt=0.05, scheme="euler"), 0.05, "euler")
    bound = omitted_confounder_bound(fit, move.reshape(1, 2, 1), cf_y=cf_y, cf_d=cf_d)
    truth = float(move @ channel)
    side = bound.lower if bias > 0.0 else bound.upper
    assert bound.estimate - truth == pytest.approx(bias, abs=0.02)
    assert side == pytest.approx(truth, abs=0.02)


def test_a_functional_of_one_state_reads_that_state_alone() -> None:
    """Under Euler a channel coefficient reads its own state's rate, so a second state's noise,
    here a hundred times the first's residual variance and more, leaves the bound where it was."""
    channel, to_action, to_rate = np.array([1.0]), np.array([[1.0]]), np.array([1.0])
    _, cf_y, cf_d, _ = _shares(to_action, to_rate, np.ones(1), dt=0.05, scheme="euler")
    data = _log(channel, to_action, to_rate, dt=0.05, scheme="euler", walk=0.5)
    fit = _fit(data, 0.05, "euler")
    first = np.zeros((2, 1, 1))
    first[0] = 1.0
    bound = omitted_confounder_bound(fit, first, cf_y=cf_y, cf_d=cf_d)
    assert bound.lower == pytest.approx(1.0, abs=4.0 * fit.channel_error)


def test_the_confidence_bound_spreads_as_the_bound_does_over_logs() -> None:
    """The spread each log reports, ``(lower - ci_lower) / z``, against the spread of ``lower``
    itself over sixty logs whose noise grows with the lever's own noise, so that the estimate's
    influence and ``bias_scale``'s are not independent. Leaving ``bias_scale``'s out reads 0.7 of
    the spread."""
    quantile = float(norm.ppf(0.95))
    lowers, spreads = [], []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        n = 2000
        x, observed, latent = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
        own = ETA * rng.standard_normal(n)
        u = TO_ACTION * observed + latent + own
        after = x + 0.05 * (DRIFT * x + u + TO_RATE * observed + latent)
        after = after + 0.05 * (0.2 + (own / ETA) ** 2) * rng.standard_normal(n)
        data = {
            "x": jnp.asarray(x)[:, None],
            "u": jnp.asarray(u)[:, None],
            "x_next": jnp.asarray(after)[:, None],
            "observed": jnp.asarray(observed)[:, None],
        }
        bound = omitted_confounder_bound(
            _fit(data, 0.05, "euler"), np.ones((1, 1, 1)), cf_y=0.3, cf_d=0.3
        )
        lowers.append(bound.lower)
        spreads.append((bound.lower - bound.ci_lower) / quantile)
    assert 0.8 < float(np.std(lowers, ddof=1)) / float(np.mean(spreads)) < 1.3


@pytest.mark.parametrize("null", [0.0, 1.5])
def test_the_robustness_values_put_the_bounds_on_the_null(null: float) -> None:
    channel, to_action, to_rate = np.array([1.0]), np.array([[1.0]]), np.array([1.0])
    fit = _fit(_log(channel, to_action, to_rate, dt=0.05, scheme="euler", n=2000), 0.05, "euler")
    one = np.ones((1, 1, 1))
    first = omitted_confounder_bound(fit, one, cf_y=0.1, cf_d=0.1, null=null)
    point, confident = first.robustness_value, first.robustness_value_ci
    assert 0.0 < confident < point < 1.0
    at_point = omitted_confounder_bound(fit, one, cf_y=point, cf_d=point, null=null)
    at_confident = omitted_confounder_bound(fit, one, cf_y=confident, cf_d=confident, null=null)
    assert at_point.lower == pytest.approx(null, abs=1e-9)
    assert at_confident.ci_lower == pytest.approx(null, abs=1e-8)


def test_the_bound_refuses_what_it_is_not_derived_for() -> None:
    channel, to_action, to_rate = np.array([1.0]), np.array([[1.0]]), np.array([1.0])
    data = _log(channel, to_action, to_rate, dt=0.05, scheme="euler", n=500)
    fit = _fit(data, 0.05, "euler")
    one = np.ones((1, 1, 1))
    shares = {"cf_y": 0.1, "cf_d": 0.1}
    bare = fit_causal_residual(_known, data, 0.05, adjust_for=("observed",), channel_degree=0)
    weighted = fit_causal_residual(
        _known,
        data,
        0.05,
        adjust_for=("observed",),
        channel_degree=0,
        influence=True,
        weights=lambda states: jnp.exp(states[:, 0]),
    )
    unadjusted = fit_causal_residual(_known, data, 0.05, channel_degree=0, influence=True)
    for other, message in (
        (bare, "influence=True"),
        (weighted, "weighted"),
        (unadjusted, "method"),
    ):
        with pytest.raises(ValueError, match=message):
            omitted_confounder_bound(other, one, **shares)
    with pytest.raises(ValueError, match="shape"):
        omitted_confounder_bound(fit, np.ones((1, 2, 1)), **shares)
    for name, value in (("cf_y", 1.0), ("cf_d", -0.1)):
        with pytest.raises(ValueError, match=name):
            omitted_confounder_bound(fit, one, **{**shares, name: value})
    with pytest.raises(ValueError, match="rho"):
        omitted_confounder_bound(fit, one, rho=1.5, **shares)
    with pytest.raises(ValueError, match="level"):
        omitted_confounder_bound(fit, one, level=0.4, **shares)


def test_repeated_rows_clustered_by_their_original_bound_as_the_original_does() -> None:
    """A log of 400 rows each repeated 16 times: summed within each row's copies, both influences
    are the original's, so the confidence bounds are; summed row by row they narrow fourfold."""
    channel, to_action, to_rate = np.array([1.0]), np.array([[0.5]]), np.array([0.5])
    data = _log(channel, to_action, to_rate, dt=0.05, scheme="euler", n=400, seed=3)
    options = {
        "adjust_for": ("observed",),
        "channel_degree": 0,
        "nuisance_degree": 1,
        "folds": 1,
        "influence": True,
    }
    copies = {name: jnp.repeat(column, 16, axis=0) for name, column in data.items()}
    original = fit_causal_residual(_known, data, 0.05, **options)
    rows = fit_causal_residual(_known, copies, 0.05, **options)
    clustered = fit_causal_residual(
        _known, copies, 0.05, **options, clusters=np.repeat(np.arange(400), 16)
    )
    one, shares = np.ones((1, 1, 1)), {"cf_y": 0.05, "cf_d": 0.05}
    expected = omitted_confounder_bound(original, one, **shares)
    got = omitted_confounder_bound(clustered, one, **shares)
    for name in ("estimate", "bias_scale", "lower", "upper"):
        assert getattr(got, name) == pytest.approx(getattr(expected, name), rel=1e-7), name
    for name in ("ci_lower", "ci_upper", "robustness_value_ci"):
        assert getattr(got, name) == pytest.approx(getattr(expected, name), rel=1e-6), name
    narrow = omitted_confounder_bound(rows, one, **shares)
    width = expected.ci_upper - expected.upper
    assert narrow.ci_upper - narrow.upper == pytest.approx(width / 4.0, rel=0.01)
