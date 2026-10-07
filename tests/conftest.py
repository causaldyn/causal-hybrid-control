"""Enable float64 so the finite-difference gradient gate has clean numerical headroom.

On a GPU, two more defaults hold the suite to what it asserts on the CPU: without preallocation
the pytest-xdist workers share the card rather than the first claiming 75% of it, and float32
matmuls run in float32 rather than TF32, whose 10-bit mantissa no tolerance here was set for."""

import os

import pytest
from scipy.optimize import OptimizeResult

# before jax is imported, so that no backend can have started without it
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
# jax 0.5.0 made this the default, and a key draws other numbers without it: the leg on the
# dependency floors would otherwise test other data than every leg on the lockfile does
jax.config.update("jax_threefry_partitionable", True)


@pytest.fixture
def stall(monkeypatch):
    """Arms ``chc.allocation``'s linear programs so that the call numbered ``at`` returns a program
    HiGHS left unsolved: status 1, its iteration limit reached, as on planes it would pivot on
    without end, or 4, ended with its optimality conditions unmet after half as many iterations.
    The list returned records each call's iteration limit."""
    import chc.allocation  # here, not above: the module must load after float64 is enabled

    def arm(at: int, status: int = 1) -> list[int]:
        real, limits = chc.allocation.linprog, []

        def linprog(*args, **kwargs):
            limits.append(kwargs["options"]["maxiter"])
            if len(limits) == at:
                return OptimizeResult(
                    status=status,
                    nit=limits[-1] if status == 1 else limits[-1] // 2,
                    message=f"HiGHS left it unsolved ({status}).",
                )
            return real(*args, **kwargs)

        monkeypatch.setattr(chc.allocation, "linprog", linprog)
        return limits

    return arm
