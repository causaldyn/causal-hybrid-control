"""Enable float64 so the finite-difference gradient gate has clean numerical headroom.

On a GPU, two more defaults hold the suite to what it asserts on the CPU: without preallocation
the pytest-xdist workers share the card rather than the first claiming 75% of it, and float32
matmuls run in float32 rather than TF32, whose 10-bit mantissa no tolerance here was set for."""

import os

# before jax is imported, so that no backend can have started without it
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
