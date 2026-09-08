"""Marketing-mix budget scheduling: which channel to spend on, when, from confounded logs.

The case study the facade was built for, in a domain where the confounding is not hypothetical.
Media spend is *planned against demand*: a marketer raises budget in the weeks a product sells
anyway, so a model fitted on the log credits the channel with the season. Fit that model and
optimise against it and the plan over-spends on whichever channel the planner favoured, which is
the observational-response failure :mod:`chc.dynamics_id` exists to stop --- here with the channel
being an incremental return and the plant a saturating carryover system.

The plant, control-affine by construction so identification and safety read the same object:

    d(sales)/dt   = -decay*(sales - base) + sum_c beta_c * hill(adstock_c) + sum_c gamma_c * spend_c
    d(adstock_c)/dt = -theta_c * adstock_c + spend_c

Each channel acts twice: ``gamma_c`` is the immediate incremental return and is the **control
channel**, ``beta_c`` the carried-over return through a saturating adstock and is part of the
**drift**. That split is the point rather than a modelling convenience. Under a policy that chases
seasonality it is ``gamma_c`` that is confounded, and ``gamma_c`` is exactly what cross-fit
Robinson partialling-out identifies; the adstock rows are mechanical and are handed to
:func:`chc.prescribe` as ``known=``, so the fit has only the sales row to learn.

HONEST SCOPE, and it bounds what the numbers below mean:

* ``hill`` is saturating and :class:`chc.residual.ControlAffineResidual`'s drift is a polynomial in
  the state, so the fitted plant is a **local surrogate** of the truth. Every arm is therefore
  audited by rolling its schedule out on the true plant, as :mod:`chc.spine` does, and the reported
  sales are the audited ones rather than the planner's own forecast.
* There is no hard total-budget constraint, because the solver constrains a box and not a
  half-space; spend is priced through ``Lever.unit_cost`` instead. Arms are compared at matched
  total spend and by return per unit spent, which is the comparison that does not depend on the
  budget anyway.
* ``theta_c`` is taken as known. In practice it is fitted (Robyn, Meridian); treating it as known
  here isolates the question this module is about, which is the incremental return and not the
  carryover rate.
* **``known=`` was exact only up to the integrator, and this module is where that was found.**
  :func:`chc.dynamics_id.fit_causal_residual` reads the state rate as a forward difference
  ``(x_next - x)/dt`` while :func:`chc.plan.causal_plan` rolls out with RK4, so at a coarse ``dt``
  the residual quietly absorbed the gap between the two. It was not small and it was not noise: on
  the fastest-decaying channel (``theta*dt = 0.7``) the fitted decay came back ``-0.503`` against
  the ``-0.7`` handed over as known, and the control channel with it. :func:`chc.decision.prescribe`
  now defaults to ``integrator="rk4"``, which closes the gap by defect correction; a test asserts
  both halves --- that the known rows come back empty under RK4, and that the same log read with
  ``integrator="euler"`` still leaves exactly the closed-form amplification difference. Every
  headline number here is audited on the true plant either way, so none of them ever depended on
  the gap; a schedule read off the *planner's own forecast* did.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from chc.decision import Constraint, Lever, Prescription, Target, prescribe
from chc.dynamics import Dynamics, LinearDynamics
from chc.dynamics_id import Integrator
from chc.graph import CausalGraph
from chc.integrate import rk4_step, rollout
from chc.panel import Panel

SEASON = "season"
SALES = "sales"


class MarketingMixPlant(eqx.Module):
    """The true continuous plant: saturating carryover plus an immediate incremental return.

    State is ``[sales, adstock_0, ..., adstock_{C-1}]`` and control is spend per channel. The
    seasonality term is a *state* here rather than an exogenous input, so that the plant stays a
    plain :class:`~chc.dynamics.Dynamics` that :func:`chc.integrate.rollout` can audit; the logging
    simulation drives it week by week and records the season it drove with.
    """

    decay: float
    base: float
    beta: Array  # (C,) carried-over return, through the saturating adstock
    gamma: Array  # (C,) immediate incremental return -- the control channel, and what is confounded
    theta: Array  # (C,) adstock decay rates
    half: Array  # (C,) half-saturation points of the Hill response
    season_gain: float
    season: float = 0.0  # the exogenous demand level the plant is currently sitting at

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        sales, adstock = x[0], x[1:]
        carried = jnp.sum(self.beta * adstock / (self.half + adstock))
        immediate = jnp.sum(self.gamma * u)
        d_sales = (
            -self.decay * (sales - self.base) + carried + immediate + self.season_gain * self.season
        )
        return jnp.concatenate([jnp.array([d_sales]), -self.theta * adstock + u])


def adstock_dynamics(theta: Array) -> LinearDynamics:
    """The mechanical half of the plant, exactly: carryover decays, spend adds to it.

    Handed to :func:`chc.prescribe` as ``known=`` so the fit has only the sales row left to learn.
    A test asserts the learned residual is near zero on these rows, which is what "known" is
    supposed to mean and is not otherwise checked anywhere.
    """
    n_channels = int(theta.shape[0])
    size = n_channels + 1
    a_matrix = jnp.zeros((size, size)).at[1:, 1:].set(jnp.diag(-theta))
    b_matrix = jnp.zeros((size, n_channels)).at[1:, :].set(jnp.eye(n_channels))
    return LinearDynamics(a_matrix, b_matrix)


@dataclass(frozen=True)
class MarketingMixSystem:
    """A synthetic multi-region weekly media log whose planner chased the season.

    Nothing here is anyone's data: the channel names are generic and the parameters are chosen to
    put the confounding in the incremental return, which is the mechanism under study.
    """

    channels: tuple[str, ...] = ("search", "social", "video")
    beta: tuple[float, ...] = (0.9, 0.6, 0.4)
    gamma: tuple[float, ...] = (0.5, 0.2, 0.35)  # true incremental returns: search > video > social
    theta: tuple[float, ...] = (0.7, 0.4, 0.25)  # search decays fastest, video carries longest
    half: tuple[float, ...] = (1.0, 1.0, 1.0)
    decay: float = 0.5
    base: float = 2.0
    season_gain: float = 1.6
    policy_gain: float = 0.9  # how hard the logged planner chased the season: the confounding
    spend_floor: float = 0.05
    spend_ceiling: float = 2.0
    noise: float = 0.05

    def plant(self, season: float = 0.0) -> MarketingMixPlant:
        """The true vector field at a given exogenous demand level."""
        return MarketingMixPlant(
            decay=self.decay,
            base=self.base,
            beta=jnp.array(self.beta),
            gamma=jnp.array(self.gamma),
            theta=jnp.array(self.theta),
            half=jnp.array(self.half),
            season_gain=self.season_gain,
            season=season,
        )

    @property
    def spend_columns(self) -> tuple[str, ...]:
        return tuple(f"spend_{name}" for name in self.channels)

    @property
    def adstock_columns(self) -> tuple[str, ...]:
        return tuple(f"adstock_{name}" for name in self.channels)

    def sample(
        self, *, n_regions: int = 40, n_weeks: int = 26, dt: float = 1.0, seed: int = 0
    ) -> dict[str, np.ndarray]:
        """Simulate the log week by week, with spend chosen from the season it is chasing.

        Integrated with :func:`chc.integrate.rk4_step` --- the same integrator the planner rolls
        out with --- so the demo carries no hidden discretisation gap between the truth and the
        model class fitted to it.
        """
        rng = np.random.default_rng(seed)
        n_channels = len(self.channels)
        names = ("region", "week", SALES, SEASON, *self.adstock_columns, *self.spend_columns)
        columns: dict[str, list[float]] = {name: [] for name in names}
        for region in range(n_regions):
            state = jnp.concatenate(
                [jnp.array([self.base + rng.normal(0.0, 0.2)]), jnp.zeros(n_channels)]
            )
            for week in range(n_weeks):
                season = float(rng.normal(0.0, 1.0))
                spend = np.clip(
                    0.5 + self.policy_gain * season + rng.normal(0.0, 0.35, n_channels),
                    self.spend_floor,
                    self.spend_ceiling,
                )
                columns["region"].append(region)
                columns["week"].append(week)
                columns[SEASON].append(season)
                columns[SALES].append(float(state[0]))
                for index, name in enumerate(self.adstock_columns):
                    columns[name].append(float(state[1 + index]))
                for index, name in enumerate(self.spend_columns):
                    columns[name].append(float(spend[index]))
                nxt = rk4_step(self.plant(season), 0.0, state, jnp.asarray(spend), dt)
                state = nxt + jnp.asarray(rng.normal(0.0, self.noise, nxt.shape))
        return {name: np.asarray(values) for name, values in columns.items()}

    def graph(self) -> CausalGraph:
        """Season confounds every channel; each channel drives sales; adstock carries it forward."""
        edges: list[tuple[str, str]] = []
        for channel, spend, adstock in zip(
            self.channels, self.spend_columns, self.adstock_columns, strict=True
        ):
            del channel
            edges += [(SEASON, spend), (spend, SALES), (spend, adstock), (adstock, SALES)]
        edges.append((SEASON, SALES))
        return CausalGraph.from_edges(edges)

    def levers(self, unit_cost: float = 0.08) -> list[Lever]:
        return [
            Lever(name, lo=self.spend_floor, hi=self.spend_ceiling, unit_cost=unit_cost)
            for name in self.spend_columns
        ]


@dataclass(frozen=True)
class MmmArm:
    """One allocation rule, audited on the true plant rather than on its own forecast."""

    name: str
    spend: Array  # (horizon, C) what it actually spent
    audited_sales: Array  # (horizon + 1,) sales on the TRUE plant under that spend
    prescription: Prescription | None  # None for the fixed-rule baseline

    @property
    def total_spend(self) -> float:
        return float(jnp.sum(self.spend))

    @property
    def cumulative_sales(self) -> float:
        """Sales summed over the planned weeks --- the area under the curve, which is the revenue.

        Deliberately not the terminal value. An optimiser that understands carryover front-loads
        spend and then tapers, which maximises the area and *lowers* the endpoint; scoring on the
        endpoint would rank that behaviour last for doing the right thing. The first draft of this
        module did exactly that and ranked a flat split above the prescribed one.
        """
        return float(jnp.sum(self.audited_sales[1:]))


@dataclass(frozen=True)
class MmmReport:
    """Three allocation rules on one log, and the truth none of them was allowed to see."""

    arms: tuple[MmmArm, ...]
    true_gamma: tuple[float, ...]
    channels: tuple[str, ...]
    baseline: str = "none"  # the do-nothing arm every lift is measured against

    def arm(self, name: str) -> MmmArm:
        return next(arm for arm in self.arms if arm.name == name)

    def lift(self, name: str) -> float:
        """Cumulative sales above the do-nothing arm --- what the incremental spend bought."""
        return self.arm(name).cumulative_sales - self.arm(self.baseline).cumulative_sales

    def efficiency(self, name: str) -> float:
        """Lift per unit of *incremental* spend; ``nan`` for the baseline, which spends no extra."""
        extra = self.arm(name).total_spend - self.arm(self.baseline).total_spend
        return self.lift(name) / extra if extra > 0 else float("nan")

    def table(self) -> str:
        """A Markdown comparison, deterministic and assertable."""
        lines = [
            "| arm | total spend | cumulative sales | lift | lift / extra spend |",
            "|---|---|---|---|---|",
        ]
        lines.extend(
            f"| {arm.name} | {arm.total_spend:.3f} | {arm.cumulative_sales:.3f} | "
            f"{self.lift(arm.name):+.3f} | {self.efficiency(arm.name):+.4f} |"
            for arm in self.arms
        )
        return "\n".join(lines)


def run_marketing_mix(
    system: MarketingMixSystem | None = None,
    *,
    horizon: int = 12,
    dt: float = 1.0,
    n_regions: int = 40,
    n_weeks: int = 26,
    unit_cost: float = 0.08,
    sales_target: float = 8.0,
    seed: int = 0,
    integrator: Integrator = "rk4",
) -> MmmReport:
    """Plan the next ``horizon`` weeks three ways and audit all three on the true plant.

    The arms are the same optimiser reading the same log through three different causal claims:

    * ``adjusted`` --- the graph, so ``season`` is adjusted for and the incremental returns are
      identified;
    * ``confounded`` --- an empty adjustment set *asserted*, which is what fitting the log directly
      amounts to and is the arm that credits the channel with the season;
    * ``flat`` --- an equal split across channels held constant, at the ``adjusted`` arm's realised
      total spend, so the head-to-head is at matched budget and differs only in allocation;
    * ``none`` --- spend held at the floor, which is the do-nothing counterfactual every lift is
      measured against.

    Audited at ``season = 0``: the plan is for an average week, and letting an arm be scored on a
    season it did not know about would measure the draw rather than the allocation.
    """
    system = system or MarketingMixSystem()
    logs = system.sample(n_regions=n_regions, n_weeks=n_weeks, dt=dt, seed=seed)
    panel = Panel.from_frame(logs, unit="region", time="week", seed=seed)
    states = (SALES, *system.adstock_columns)
    known = adstock_dynamics(jnp.array(system.theta))
    truth = system.plant(season=0.0)

    def plan(adjustment: CausalGraph | tuple[str, ...]) -> Prescription:
        return prescribe(
            panel,
            levers=system.levers(unit_cost),
            target=Target(SALES, value=sales_target),
            constraints=[Constraint(name, lo=0.0) for name in system.adstock_columns],
            adjustment=adjustment,
            known=known,
            horizon=horizon,
            dt=dt,
            tolerance=1.0,
            seed=seed,
            integrator=integrator,
        )

    adjusted, confounded = plan(system.graph()), plan(())
    x0 = _start_state(panel, states)

    arms = [
        _audit(name, jnp.asarray(result.schedule.magnitudes), truth, x0, dt, result)
        for name, result in (("adjusted", adjusted), ("confounded", confounded))
    ]
    n_channels = len(system.channels)
    per_step = arms[0].total_spend / (horizon * n_channels)
    arms.append(_audit("flat", jnp.full((horizon, n_channels), per_step), truth, x0, dt, None))
    floor = jnp.full((horizon, n_channels), system.spend_floor)
    arms.append(_audit("none", floor, truth, x0, dt, None))
    return MmmReport(arms=tuple(arms), true_gamma=system.gamma, channels=system.channels)


def _start_state(panel: Panel, states: tuple[str, ...]) -> Array:
    """The pooled last observed state --- the same "where we are now" ``prescribe`` plans from."""
    units, times = panel.codes()
    last = {}
    for row, (unit, period) in enumerate(zip(units, times, strict=True)):
        if int(unit) not in last or period > times[last[int(unit)]]:
            last[int(unit)] = row
    rows = np.array(sorted(last.values()), dtype=np.int64)
    return jnp.mean(
        jnp.stack([jnp.asarray(np.asarray(panel[name], dtype=float)[rows]) for name in states], 1),
        axis=0,
    )


def _audit(
    name: str,
    spend: Array,
    truth: Dynamics,
    x0: Array,
    dt: float,
    prescription: Prescription | None,
) -> MmmArm:
    trajectory = rollout(truth, x0, spend, dt)
    return MmmArm(
        name=name,
        spend=spend,
        audited_sales=jax.vmap(lambda state: state[0])(trajectory),
        prescription=prescription,
    )
