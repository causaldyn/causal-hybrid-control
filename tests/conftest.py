"""Enable float64 so the finite-difference gradient gate has clean numerical headroom."""

import jax

jax.config.update("jax_enable_x64", True)
