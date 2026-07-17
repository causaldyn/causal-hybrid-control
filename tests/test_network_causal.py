"""Network causal gate: naive DML is blind to spillover; network DML recovers direct + spillover."""

import jax

from chc.estimators import DoubleML
from chc.network_causal import ConfoundedNetworkSystem, estimate_network_effects


def _data() -> dict[str, jax.Array]:
    return ConfoundedNetworkSystem().sample(jax.random.key(0))


def test_network_dml_recovers_direct_and_spillover() -> None:
    effects = estimate_network_effects(_data())
    assert abs(effects["direct"] - 1.0) < 0.1  # true direct = 1.0
    assert abs(effects["spillover"] - 0.6) < 0.1  # true spillover = 0.6


def test_naive_dml_is_blind_to_spillover() -> None:
    """The whole point: naive DML estimates a single effect and misses the interference channel."""
    data = _data()
    naive_direct = float(DoubleML().estimate(data, covariates=("x", "z")).effect)
    effects = estimate_network_effects(data)
    # naive's per-unit read (direct only) undershoots the true total effect direct + spillover:
    assert naive_direct < effects["direct"] + effects["spillover"] - 0.2
