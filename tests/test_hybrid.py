"""HybridDynamics invariants: zero residual == known dynamics; MLP residual perturbs it."""

import jax
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from chc import DampedOscillator, HybridDynamics, MLPResidual, ZeroResidual

finite = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)


@given(pos=finite, vel=finite, force=finite)
@settings(max_examples=50, deadline=None)
def test_zero_residual_recovers_known(pos: float, vel: float, force: float) -> None:
    known = DampedOscillator(omega=1.3, zeta=0.2)
    hybrid = HybridDynamics(known=known, residual=ZeroResidual(out_dim=2))
    x = jnp.array([pos, vel])
    u = jnp.array([force])
    assert jnp.allclose(hybrid(0.0, x, u), known(0.0, x, u))


def test_mlp_residual_shapes_and_effect() -> None:
    known = DampedOscillator(omega=1.0, zeta=0.1)
    residual = MLPResidual(state_dim=2, control_dim=1, out_dim=2, key=jax.random.key(0))
    hybrid = HybridDynamics(known=known, residual=residual)
    x = jnp.array([0.5, -0.3])
    u = jnp.array([0.2])
    out = hybrid(0.0, x, u)
    assert out.shape == (2,)
    assert not jnp.allclose(out, known(0.0, x, u))
