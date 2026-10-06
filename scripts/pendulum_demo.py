"""A pendulum raised to an angle and held there, planned from a log whose torque was confounded.

Run: uv run python scripts/pendulum_demo.py

The control-audience case study of :func:`chc.prescribe`, on ``Pendulum-v1``'s dynamics:
``theta`` from upright and ``omega' = (3g/2l) sin(theta) + 3/(m l^2) u`` at ``g = 10, m = l = 1``,
so a gravity coefficient of 15 and a torque channel of 3. Gymnasium steps it semi-implicitly; this
log is stepped with RK4, the planner's own integrator, so no discretisation gap sits between the
truth and the model class fitted to it. Nothing imports this plant, so it lives here rather than
in a module.

**The confounder sits on the channel.** A wind torque ``w``, AR(1), adds to the commanded one, and
the logging operator cancels part of the wind it measures, ``u = -k w + e``. The torque column is
then correlated with a disturbance that enters ``omega'`` exactly where the torque does, and the
channel fitted without the wind is ``b (k(k-1) s_w^2 + s_e^2) / (k^2 s_w^2 + s_e^2)`` for white
wind (:func:`omitted_wind_channel`) -- negative for every ``k`` strictly between the roots of
``k^2 - k + s_e^2 / s_w^2``, so an operator who cancels half the wind teaches the log that torque
pushes backwards. That is a bias of the control channel, which cross-fit Robinson partialling-out
removes; a confounder acting through the drift would show what it does not remove
(:mod:`chc.dynamics_id`). Gravity is first principles and goes in as ``known=``, so what the log
has to identify is the actuator. The design constants are the ones
``causaldyn_bench.pendulum_causal`` (Track J) logs Gymnasium's pendulum with.

Three readings of one log by the same planner, each audited by running its schedule on the *true*
pendulum at zero wind -- the plan is made for the expected disturbance, as :mod:`chc.mmm` audits at
an average season:

* ``adjusted`` --- the graph derives the adjustment set ``{wind}``;
* ``asserted`` --- an empty set asserted, which is what fitting the log as it stands amounts to;
* ``latent`` --- the same graph with the wind declared unobserved: nothing identifies the channel,
  and :func:`chc.prescribe` returns no schedule at all.

The task: from hanging at rest, raise the pendulum ``LIFT`` radians and hold it there for
``HORIZON`` steps, the torque inside Pendulum-v1's own box and ``|omega| <= SPEED_LIMIT`` held in
the solve as the barrier (``hold_constraints=True``). Holding costs ``(15/3) sin(LIFT)`` of the
box's 2.

HONEST SCOPE:

* The schedule is open loop on a frictionless plant, and it lands because the fitted model is
  right. Deployed, it would be re-planned from each measured state (:class:`chc.RecedingHorizon`).
* The barrier is held at the library's class-K gain of 1 per second, so the approach to the speed
  limit is what binds, not the limit: every schedule here stays well inside it.
* The certificate's tube compounds at the norm of the field's slope in the state where the plan
  starts, which for a hanging pendulum is 15 per second: a ``local`` rate, since gravity bends the
  field away from there. It is fed ``channel_error``, the channel's standard error at the states
  the log visited, and compounds it at that rate, so the trusted prefix is short. The audit is what
  shows the rest of the schedule holding; the certificate does not claim it.

Precision: nothing here sets ``jax_enable_x64``. The header names the precision the numbers were
computed at, and in float32 a warning says what moves: the fits do not, the barrier-held solves do.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from chc import CausalGraph, Constraint, Lever, Panel, Target, prescribe
from chc.decision import Prescription
from chc.integrate import rollout

THETA, OMEGA, TORQUE, WIND = "theta", "omega", "torque", "wind"
GRAVITY = 15.0  # 3 g / (2 l) at Pendulum-v1's g = 10, l = 1
GAIN = 3.0  # 3 / (m l^2) at m = l = 1: the torque channel, and the estimand
DT = 0.05  # Pendulum-v1's step
MAX_TORQUE = 2.0  # Pendulum-v1's torque box

COMPENSATION = 0.5  # the share of the measured wind the operator cancels: the confounding
EXPLORE = 0.06  # the operator's own torque noise: the overlap that identifies the channel
WIND_SCALE, WIND_RHO, WIND_CAP = 0.175, 0.6, 0.5  # stationary std, AR(1) coefficient, clip
SWING = 1.6  # each logged episode starts within this angle of hanging

HANGING = (math.pi, 0.0)  # at rest at the bottom: where every plan starts
LIFT = 0.3
SPEED_LIMIT = 1.0
HORIZON = 40
TOLERANCE = 0.1  # the forecast error the certificate's tube may reach, a third of the lift

EDGES = (
    (WIND, TORQUE),  # the operator reacts to the measured wind: this is the confounding
    (WIND, OMEGA),  # and the wind pushes the pendulum
    (TORQUE, OMEGA),
    (OMEGA, THETA),
)

_log = logging.getLogger(__name__)


class Pendulum(eqx.Module):
    """Pendulum-v1's field, ``theta`` from upright, driven by the torque actually applied."""

    gain: float = GAIN

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return jnp.stack([x[1], GRAVITY * jnp.sin(x[0]) + self.gain * u[0]])


KNOWN = Pendulum(gain=0.0)  # gravity, from first principles; the actuator is left to the log


def sample_log(*, episodes: int, steps: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Swing the pendulum about hanging while an operator cancels ``COMPENSATION`` of the wind.

    Each episode starts from a fresh swing, with the wind already stationary. Both the applied
    torque and the speed stay well inside Pendulum-v1's clips, which would otherwise make the
    logged torque stop being the applied one.
    """
    rng = np.random.default_rng(seed)
    innovation = WIND_SCALE * math.sqrt(1.0 - WIND_RHO**2)
    wind = np.empty((episodes, steps))
    wind[:, 0] = np.clip(rng.normal(0.0, WIND_SCALE, episodes), -WIND_CAP, WIND_CAP)
    for step in range(1, steps):
        drawn = WIND_RHO * wind[:, step - 1] + innovation * rng.normal(size=episodes)
        wind[:, step] = np.clip(drawn, -WIND_CAP, WIND_CAP)
    torque = -COMPENSATION * wind + EXPLORE * rng.normal(size=(episodes, steps))
    start = np.column_stack(
        [math.pi + rng.uniform(-SWING, SWING, episodes), rng.uniform(-1.0, 1.0, episodes)]
    )
    applied = jnp.asarray(torque + wind)[..., None]
    states = jax.vmap(lambda x0, us: rollout(Pendulum(), x0, us, DT))(jnp.asarray(start), applied)
    states = np.asarray(states)[:, :-1]  # the last torque's outcome falls outside the log
    return {
        "episode": np.repeat(np.arange(episodes), steps),
        "step": np.tile(np.arange(steps), episodes),
        THETA: states[..., 0].ravel(),
        OMEGA: states[..., 1].ravel(),
        TORQUE: torque.ravel(),
        WIND: wind.ravel(),
    }


def omitted_wind_channel(wind: np.ndarray) -> float:
    """The torque channel a fit without the wind returns, for white wind: the bias as a formula."""
    variance, k = float(np.var(wind)), COMPENSATION
    return GAIN * (k * (k - 1.0) * variance + EXPLORE**2) / (k**2 * variance + EXPLORE**2)


@dataclass(frozen=True)
class Audit:
    """A schedule run on the TRUE pendulum at zero wind, scored against the target angle."""

    trajectory: Array  # (horizon + 1, 2)
    final_error: float  # theta minus the target at the last step: where the hold ended
    rms_error: float  # over the planned steps, the start excluded
    peak_speed: float  # the largest |omega| on the way, against SPEED_LIMIT


def audit(actions: Array) -> Audit:
    trajectory = rollout(Pendulum(), jnp.asarray(HANGING), actions, DT)
    error = trajectory[1:, 0] - (HANGING[0] - LIFT)
    return Audit(
        trajectory=trajectory,
        final_error=float(error[-1]),
        rms_error=float(jnp.sqrt(jnp.mean(error**2))),
        peak_speed=float(jnp.max(jnp.abs(trajectory[:, 1]))),
    )


@dataclass(frozen=True)
class Reading:
    """One causal claim about the log: what :func:`chc.prescribe` made of it, and what that did."""

    name: str
    prescription: Prescription
    audit: Audit | None  # None exactly when the reading produced no schedule

    @property
    def channel(self) -> float:
        """The fitted torque channel on the ``omega`` row at hanging, where every plan starts.

        Read at the start state rather than as the constant term: the channel is affine in the
        state, and its constant is the channel at ``theta = 0``, which the log never visits.
        """
        residual = self.prescription.model_fit.residual
        return float(residual.control_channel(jnp.asarray(HANGING))[1, 0])


@dataclass(frozen=True)
class PendulumCase:
    """The three readings of one log, the idle pendulum, and the prediction for the biased one."""

    readings: tuple[Reading, ...]
    idle: Audit  # no torque at all: the pendulum stays where it hangs, LIFT from the target
    omitted_wind_channel: float

    def reading(self, name: str) -> Reading:
        return next(reading for reading in self.readings if reading.name == name)


def graph(*, latent: bool = False) -> CausalGraph:
    return CausalGraph.from_edges(EDGES, latent=(WIND,) if latent else ())


def run_pendulum(*, episodes: int = 20, steps: int = 100, seed: int = 0) -> PendulumCase:
    """Read one log three ways through :func:`chc.prescribe`; audit each schedule on the truth."""
    logs = sample_log(episodes=episodes, steps=steps, seed=seed)
    panel = Panel.from_frame(logs, unit="episode", time="step", seed=seed)

    def read(name: str, adjustment: CausalGraph | tuple[str, ...]) -> Reading:
        prescription = prescribe(
            panel,
            levers=[Lever(TORQUE, lo=-MAX_TORQUE, hi=MAX_TORQUE)],
            target=Target(THETA, value=HANGING[0] - LIFT),
            constraints=[Constraint(OMEGA, lo=-SPEED_LIMIT, hi=SPEED_LIMIT)],
            hold_constraints=True,
            adjustment=adjustment,
            known=KNOWN,
            horizon=HORIZON,
            dt=DT,
            tolerance=TOLERANCE,
            x0=jnp.asarray(HANGING),
            seed=seed,
        )
        schedule = None if prescription.plan is None else prescription.schedule.magnitudes
        return Reading(name, prescription, None if schedule is None else audit(schedule))

    return PendulumCase(
        readings=(
            read("adjusted", graph()),
            read("asserted", ()),
            read("latent", graph(latent=True)),
        ),
        idle=audit(jnp.zeros((HORIZON, 1))),
        omitted_wind_channel=omitted_wind_channel(logs[WIND]),
    )


def main() -> None:
    case = run_pendulum()
    x64 = case.readings[0].prescription.provenance.x64
    precision = "float64" if x64 else "float32"
    if not x64:
        _log.warning(
            "float32: the fits match float64 to the digits printed, the barrier-held solves do not"
            " -- peak omega and the asserted schedule's errors move; export JAX_ENABLE_X64=1 for"
            " float64"
        )
    print(
        f"== Pendulum-v1, logged while an operator cancelled half the measured wind; true torque"
        f" channel {GAIN:.3f}; computed in {precision} ==\n"
    )
    print(
        f"task: from hanging, raise the pendulum {LIFT:.2f} rad and hold it for {HORIZON} steps of"
        f" {DT} s, |torque| <= {MAX_TORQUE}, |omega| <= {SPEED_LIMIT} rad/s held as the barrier\n"
    )
    print(
        "| reading | identification | adjusted for | torque channel | trusted steps "
        "| final angle error | rms angle error | peak omega |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for reading in case.readings:
        certificate = reading.prescription.certificate
        covariates = ", ".join(certificate.adjustment.covariates) or "nothing"
        head = f"| {reading.name} | {certificate.identification} |"
        if reading.audit is None:
            print(f"{head} - | - | {certificate.trustworthy_steps} | no schedule | - | - |")
            continue
        run = reading.audit
        print(
            f"{head} {covariates} | {reading.channel:+.3f} | {certificate.trustworthy_steps} |"
            f" {run.final_error:+.3f} | {run.rms_error:.3f} | {run.peak_speed:.3f} |"
        )
    idle = case.idle
    print(
        f"| none | - | - | - | - | {idle.final_error:+.3f} | {idle.rms_error:.3f} |"
        f" {idle.peak_speed:.3f} |"
    )

    adjusted, asserted = case.reading("adjusted"), case.reading("asserted")
    assert adjusted.audit is not None
    assert asserted.audit is not None
    print(
        f"\nAdjusted for the wind, the torque channel is {adjusted.channel:+.3f} against"
        f" {GAIN:.3f}, and its schedule, run on the true pendulum, ends"
        f" {abs(adjusted.audit.final_error):.3f} rad from the target."
    )
    print(
        f"Asserting that nothing needs adjusting returns {asserted.channel:+.3f} -- the"
        f" white-wind formula predicts {case.omitted_wind_channel:+.3f} -- so the plan pushes"
        f" the wrong way and ends {asserted.audit.final_error:+.3f} rad from the target,"
        f" {asserted.audit.final_error / idle.final_error:.1f}x as far as never acting."
    )
    print(
        f"The certificate trusts the adjusted schedule for its first"
        f" {adjusted.prescription.certificate.trustworthy_steps} steps and the asserted one for"
        f" {asserted.prescription.certificate.trustworthy_steps}: a channel with no standard error"
        " bounds no trajectory, whatever the barrier says. Declared latent, the wind leaves"
        " nothing to adjust for, and there is no schedule to trust."
    )


if __name__ == "__main__":
    main()
