"""What a barrier certificate is worth: the pointwise §40 check against the true reachable tube.

Run: uv run python scripts/reachability_demo.py
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from chc.reachability import BarrierReachabilityGap, barrier_reachability_gap

# The plant `chc.spine` plans on, with the same supply floor. Relative degree 1: the incentive
# column moves the constrained zone directly, so `B^T grad h != 0` and the barrier can see it.
ZONE_A = jnp.array([[-0.6, 0.3], [0.3, 0.25]])
ZONE_B = jnp.array([[1.0], [-1.0]])
DI_B = jnp.array([[0.0], [1.0]])


def _row(name: str, gap: BarrierReachabilityGap) -> str:
    cbf = "yes" if gap.valid_cbf else "no"
    return (
        f"{name:22}{gap.radius:>8.2f}{gap.safe_fraction:>8.3f}{gap.reachable_fraction:>8.3f}"
        f"{gap.barrier_fraction:>9.3f}{cbf:>11}{gap.certified_but_unreachable:>7.3f}"
    )


def _sweep(
    barrier: Callable[[Array], Array],
    drift: Callable[[Array], Array],
    b_matrix: Array,
    radii: tuple[float, ...],
    **grid: object,
) -> list[BarrierReachabilityGap]:
    return [
        barrier_reachability_gap(barrier, drift, b_matrix, radius=r, **grid)  # type: ignore[arg-type]
        for r in radii
    ]


def main() -> None:
    zone = _sweep(
        lambda x: x[1] + 0.4,
        lambda x: ZONE_A @ x,
        ZONE_B,
        (0.0, 0.6, 1.2),
        lower=(-1.5, -1.5),
        upper=(1.5, 1.5),
        resolution=(61, 61),
        horizon=2.0,
        steps=1200,
        u_max=3.0,
    )
    integrator = _sweep(
        lambda x: x[0],
        lambda x: jnp.array([x[1], 0.0]),
        DI_B,
        (0.0, 0.5),
        lower=(-1.0, -2.0),
        upper=(3.0, 2.0),
        resolution=(61, 61),
        horizon=2.0,
        steps=500,
        u_max=1.0,
    )

    header = (
        f"{'plant':22}{'radius':>8}{'safe':>8}{'reach':>8}"
        f"{'barrier':>9}{'valid CBF':>11}{'trap':>7}"
    )
    print(header)
    print("-" * len(header))
    for i, gap in enumerate(zone):
        print(_row("two-zone supply floor" if i == 0 else "", gap))
    print()
    for i, gap in enumerate(integrator):
        print(_row("double integrator" if i == 0 else "", gap))

    identified, unidentified = zone[1], zone[2]
    print(
        f"\nWith the robust condition holding at every point of the safe set (valid CBF up to"
        f" radius {identified.radius:.2f}), the reachable tube IS the safe set --"
        f" {identified.reachable_fraction:.3f} against {identified.safe_fraction:.3f}. That is the"
        f" invariance the safety layer assumes, checked rather than assumed."
    )
    print(
        f"Past the zero-action threshold the certificate dies and the truth follows: at radius"
        f" {unidentified.radius:.2f} only {unidentified.reachable_fraction:.3f} of the grid is"
        f" really safe for the horizon, and {unidentified.certified_but_unreachable:.3f} of it"
        f" passes the pointwise check anyway."
    )

    tight, loose = integrator
    print(
        f"\nThe double integrator is the sharp case: h has relative degree 2, so B^T grad h == 0"
        f" and the pointwise condition never sees the actuator. Its verdict is IDENTICAL at both"
        f" radii ({tight.barrier_fraction:.3f} vs {loose.barrier_fraction:.3f}) while the true tube"
        f" shrinks from {tight.reachable_fraction:.3f} to {loose.reachable_fraction:.3f}."
    )
    print(
        f"So {loose.certified_but_unreachable:.1%} of the grid is certified by the §40 check and"
        f" unreachable in truth. A per-step certified prefix is a filter, not a proof --"
        f" the proof needs the condition on the whole set, which is what `valid_cbf` reports."
    )


if __name__ == "__main__":
    main()
