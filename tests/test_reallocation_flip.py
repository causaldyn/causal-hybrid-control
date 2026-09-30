"""MM7: how much confounding flips a reallocation, in Gamma and in variance shares, and what to do
past the flip.

Moving spend from lever a to lever b is worth ``d = b_b - b_a`` a unit, which is linear in the
channel. In variance shares the move's flip point is the robustness value
:func:`omitted_confounder_bound` reports for it. Under the marginal sensitivity model at level
Gamma the identified set of ``d`` is ``d_hat +- (Gamma - 1)/(Gamma + 1) G``, ``G`` the analyst's
CVaR gap on the move's scale, so the move keeps its sign until ``Gamma = (G + |d_hat|)/(G -
|d_hat|)``: the inversion :func:`barrier_gamma_star` makes for a certificate, with the value's
distance from zero as the radius the decision absorbs. Past it the minimax-cost action holds, and
the minimax-regret action still moves.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chc import minimax_action
from chc.dynamics_id import fit_causal_residual, omitted_confounder_bound
from chc.sensitivity import barrier_gamma_star, confounding_robust_inflation

MOVE = np.array([-1.0, 1.0]).reshape(1, 2, 1)  # lever a's spend moved to lever b
GAP = 2.0  # the analyst's CVaR gap, on the move's scale
TARGET, EFFORT = 1.0, 0.1  # the lift the move is for, and the cost of each unit moved


def _known(t: float, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros_like(x)


@pytest.fixture(scope="module")
def fit():
    """Lever b is worth 0.3 more a unit than lever a. The fit adjusts for an observed confounder
    of both, and a latent pulling lever b and the rate reads the move higher than it is."""
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
    return fit_causal_residual(
        _known, data, 0.05, adjust_for=("observed",), degree=1, channel_degree=0, influence=True
    )


def _identified(value: float, gamma: float) -> tuple[float, float]:
    radius = confounding_robust_inflation(GAP, 0.0, gamma)
    return value - radius, value + radius


def test_the_gamma_flip_is_where_the_robust_move_stops(fit) -> None:
    """Over a grid of Gamma, the minimax-cost move holds from the first point past the flip on,
    and at no point before it."""
    value = omitted_confounder_bound(fit, MOVE, cf_y=0.0, cf_d=0.0).estimate
    flip = barrier_gamma_star(abs(value), GAP, 1.0)
    assert 1.0 < flip < np.inf
    gammas = np.geomspace(1.0, 4.0 * flip, 4001)
    held = [minimax_action(TARGET, *_identified(value, g), EFFORT).action == 0.0 for g in gammas]
    first = held.index(True)
    assert all(held[first:])
    assert gammas[first - 1] < flip <= gammas[first]


def test_the_share_flip_is_the_robustness_value(fit) -> None:
    bound = omitted_confounder_bound(fit, MOVE, cf_y=0.1, cf_d=0.1)
    assert bound.estimate > 0.0
    for share, side in ((0.999, 1.0), (1.001, -1.0)):
        moved = bound.robustness_value * share
        at = omitted_confounder_bound(fit, MOVE, cf_y=moved, cf_d=moved)
        assert side * at.lower > 0.0


def test_past_the_flip_the_cost_minimax_holds_and_the_regret_minimax_moves(fit) -> None:
    """The regret of an action ``u`` at an effect ``b`` is ``(b^2 + effort) (u - u*(b))^2``, over
    what the action that knew ``b`` would have paid. By brute force over the identified set and
    over actions, the hedge has the least worst regret, and holding has more."""
    value = omitted_confounder_bound(fit, MOVE, cf_y=0.0, cf_d=0.0).estimate
    low, high = _identified(value, 2.0 * barrier_gamma_star(abs(value), GAP, 1.0))
    assert low < 0.0 < high
    held = minimax_action(TARGET, low, high, EFFORT)
    hedged = minimax_action(TARGET, low, high, EFFORT, criterion="regret")
    assert held.binding == "zero"
    assert hedged.action * value > 0.0

    effects = np.linspace(low, high, 20_001)
    spread = effects**2 + EFFORT

    def worst_regret(action: float) -> float:
        return float(np.max(spread * (action - effects * TARGET / spread) ** 2))

    assert hedged.worst_case == pytest.approx(worst_regret(hedged.action), rel=1e-6)
    best = min(worst_regret(action) for action in np.linspace(-2.0, 2.0, 4001))
    assert best >= worst_regret(hedged.action) - 1e-6
    assert worst_regret(hedged.action) < worst_regret(0.0)
