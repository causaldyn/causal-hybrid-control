"""Where a prediction-error residual lands, and what identifying its channel is worth.

Run: uv run python scripts/dynamics_id_demo.py
"""

from __future__ import annotations

import jax

# The test suite runs in float64 and every number quoted in the docs comes from it. In float32 the
# IV row lands ~7x closer to the truth by cancellation, which would make this demo disagree with
# the gated numbers -- so match the suite rather than the default.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax import Array  # noqa: E402

from chc.benchmark import CausalDynamicsTask, leaderboard  # noqa: E402
from chc.dynamics import HybridDynamics  # noqa: E402
from chc.dynamics_id import ConfoundedControlAffineSystem, fit_causal_residual  # noqa: E402
from chc.residual import ControlAffineResidual  # noqa: E402
from chc.train import fit_residual  # noqa: E402

CHANNEL = jnp.array([[1.0], [0.5]])


def known(t: float | Array, x: Array, u: Array) -> Array:
    return jnp.zeros_like(x)


def main() -> None:
    system = ConfoundedControlAffineSystem(
        drift=jnp.array([[-0.5, 0.1], [0.0, -0.3]]),
        channel=CHANNEL,
        confounder_to_rate=jnp.array([[2.0], [1.0]]),
        confounder_to_action=jnp.array([[-1.5]]),
        instrument_to_action=jnp.array([[0.8]]),
    )
    data = system.sample(4000, jax.random.key(0), known)

    trained, _ = fit_residual(
        HybridDynamics(
            known=known,
            residual=ControlAffineResidual(drift=jnp.zeros((2, 3)), channel=jnp.zeros((2, 1, 3))),
        ),
        data,
        system.dt,
        steps=800,
        lr=1e-1,
    )
    fits = {
        "mse (chc.train)": trained.residual.channel[:, :, 0],
        "unadjusted moment": fit_causal_residual(known, data, system.dt).residual.channel[:, :, 0],
        "orthogonal (z adjusted)": fit_causal_residual(
            known, data, system.dt, adjust_for=("z",)
        ).residual.channel[:, :, 0],
        "iv (z latent)": fit_causal_residual(
            known, data, system.dt, instrument="w"
        ).residual.channel[:, :, 0],
    }

    print(f"== control channel, true B = {CHANNEL.ravel()} ==\n")
    header = f"{'fit':26}{'B_hat':>24}{'error':>9}"
    print(header)
    print("-" * len(header))
    for name, got in fits.items():
        estimate = "[" + " ".join(f"{v:+.3f}" for v in got.ravel()) + "]"
        print(f"{name:26}{estimate:>24}{float(jnp.linalg.norm(got - CHANNEL)):>9.3f}")

    print(
        "\nthe prediction-error fit and the unadjusted moment agree on the same wrong channel:"
        "\nminimising rollout error harder cannot fix it, because it is not a fitting problem."
        "\n\nthe two identified routes are not equals. The shifter explains only ~18% of the"
        "\naction's variance, so the IV estimate is consistent but ~40x noisier than adjusting"
        "\nfor a logged confounder -- an instrument is a weaker substitute than the word suggests."
    )
    print("\n== the same fits used to plan, scored on the true plant ==\n")
    print(leaderboard(CausalDynamicsTask().run()))
    print(
        "\nnote the viol and ood columns: the biased planner prices the actuator as useless and"
        "\nunder-commands, so it never leaves the logged region. Only regret sees the failure."
    )


if __name__ == "__main__":
    main()
