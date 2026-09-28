"""A city's zones over time: idle supply and open requests per zone, an incentive and a price per
zone as levers, recruits drawn from neighbouring zones, and supply that answers an incentive late.

The marketplace plant the loop runs on. :mod:`chc.marketplace`'s softmax equilibrium is not
control-affine, so it stays a validation target; this plant is control-affine by construction, so
identification and safety read the same object. Per zone ``i``, ``K`` zones on a ring::

    d(supply)/dt = gamma (sigma - supply) + (I - P)(recruit incentive + carry stock) - m - eta xi
    d(queue)/dt  = demand (1 + xi - elasticity price) - m - alpha queue
    d(stock)/dt  = incentive - theta stock

``m`` is the zone's rate of trips (:data:`Matching`) and ``xi`` its demand shock. The levers are
moves from the do-nothing point, both on ``[0, 1]``: the incentive in units of the budget, the price
a rise over the list price as a share of it. An incentive recruits ``recruit`` drivers a period at
once and ``carry`` a period per unit of its stock, the incentive exposure decaying at ``theta``, so
part of the response arrives late, ``1/theta`` periods on average. A share ``spill`` of the recruits
come from the neighbouring zones, ``P[j, i] = spill / 2`` for each neighbour ``j`` of ``i``, so the
city gains ``1 - spill`` of what the zone does. A full price rise turns away ``elasticity`` of the
zone's demand.

The do-nothing point is the parameterisation: ``supply`` and ``queue`` are each zone's idle supply
and open requests with no lever moved and no shock, and ``sigma`` and ``demand`` are derived so that
the point is the steady state::

    sigma = supply + m0 / gamma,   demand = m0 + alpha queue,
    m0 = mu supply queue / (supply + queue)

HONEST SCOPE, and it bounds what a number read off this plant means:

* **The logged operator chased the shock.** Both levers rose with the zone's shock, which raises
  demand and takes supply off the road. A fit that ignores it reads a price rise as raising demand
  and an incentive as recruiting a fraction of what it does. The shock is logged (``shock_<i>``), so
  it can be adjusted for, and :meth:`ZoneMarketSystem.graph` derives that adjustment. Shocks are
  independent from period to period, so the stock's drift is not confounded; a persistent shock
  would confound it too.
* **The log's actions share its noise.** Under the logged operator an action and the plant's noise
  both move with the period's shock, and an off-policy evaluation that weights actions by ``u | x``
  alone assumes they do not. :meth:`ZoneMarketSystem.linear_gaussian` is the one-period law under a
  policy that does not see the shock.
* **A switchback on one zone reads the zone's recruits, not the city's:** the zone gains ``recruit``
  and its neighbours lose ``spill * recruit`` between them.
* **Under harmonic matching the drift is nonlinear**, and a fit whose drift is a polynomial in the
  state is a local surrogate of it, as in :mod:`chc.mmm`: audit a plan on this plant, not on the
  fit's forecast. Under linear matching every closed form here is exact.
* ``theta`` is taken as known, as :mod:`chc.mmm` takes its adstock rates:
  :meth:`ZoneMarketSystem.stock_dynamics` hands the stock rows to a fit as ``known=``.
* Nothing here is anyone's data. The parameters are generic, chosen to put the confounding in the
  levers' effects, which is the mechanism under study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike, NDArray

from chc.decision import Lever
from chc.dynamics import LinearDynamics
from chc.evaluation import LinearGaussianPlant
from chc.experiment import ZoneDecision
from chc.graph import CausalGraph
from chc.integrate import rk4_step

Matching = Literal["harmonic", "linear"]
"""How a zone's idle supply and open requests turn into trips:

* ``"harmonic"``: ``mu supply queue / (supply + queue)``, never more than the smaller side;
* ``"linear"``: its tangent at the do-nothing point, which passes through zero because the harmonic
  law is homogeneous of degree one, so the two laws share the do-nothing steady state.
"""

_PER_ZONE = ("supply", "queue", "recruit", "carry", "elasticity")
_LOGGED_LEVEL = 0.3  # where the logged operator set both levers in a period with no shock


class ZoneMarketPlant(eqx.Module):
    """The true continuous plant: ``x = [supply, queue, stock]`` and ``u = [incentive, price]``,
    each block one entry per zone.

    Control-affine with a constant channel: an incentive recruits ``(I - P) diag(recruit)`` at once
    and a price rise turns away ``diag(demand * elasticity)``; everything else is drift. ``shock``
    is the demand shock each zone is sitting at, a field rather than an input so that the plant
    stays a plain :class:`~chc.dynamics.Dynamics`.
    """

    sigma: Array  # (K,) the supply level drivers relax to with no trips
    demand: Array  # (K,) requests a period with no price rise and no shock
    recruit: Array  # (K,) drivers a period one unit of incentive recruits at once: the channel
    carry: Array  # (K,) drivers a period one unit of incentive stock recruits
    elasticity: Array  # (K,) the share of demand a full price rise turns away
    transfer: Array  # (K, K) I - P: column i is where zone i's recruits land and whom they left
    match_supply: Array  # (K,) the linear law's weight on supply
    match_queue: Array  # (K,) the linear law's weight on the queue
    shock: Array  # (K,)
    mu: float
    gamma: float
    alpha: float
    theta: float
    shock_supply: float
    harmonic: bool = eqx.field(static=True)

    def trips(self, x: Array) -> Array:
        """Each zone's rate of trips at state ``x``."""
        k = self.sigma.shape[0]
        supply, queue = x[:k], x[k : 2 * k]
        if self.harmonic:
            return self.mu * supply * queue / (supply + queue)
        return self.match_supply * supply + self.match_queue * queue

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        k = self.sigma.shape[0]
        supply, queue, stock = x[:k], x[k : 2 * k], x[2 * k :]
        incentive, price = u[:k], u[k:]
        trips = self.trips(x)
        recruited = self.transfer @ (self.recruit * incentive + self.carry * stock)
        d_supply = (
            self.gamma * (self.sigma - supply) + recruited - trips - self.shock_supply * self.shock
        )
        d_queue = (
            self.demand * (1.0 + self.shock - self.elasticity * price) - trips - self.alpha * queue
        )
        return jnp.concatenate([d_supply, d_queue, incentive - self.theta * stock])


@dataclass(frozen=True)
class ZoneMarketSystem:
    """A synthetic city log, one row per day and period, whose operator chased each zone's shock.

    The per-zone tuples set the number of zones and must agree in length. Zone 0 is the centre, with
    the most supply and demand and the strongest recruitment.

    Raises:
        ValueError: on per-zone tuples of different lengths, a do-nothing point or a rate that is
            not positive, a negative channel, shock or noise, an ``elasticity`` outside ``[0, 1)``,
            a ``spill`` outside ``[0, 1]`` or with one zone to spill into, or an unknown matching.
    """

    supply: tuple[float, ...] = (8.0, 6.0, 6.0, 4.0)  # idle drivers with no lever moved
    queue: tuple[float, ...] = (4.0, 3.0, 3.0, 2.0)  # open requests with no lever moved
    recruit: tuple[float, ...] = (2.0, 1.5, 1.5, 1.0)
    carry: tuple[float, ...] = (0.6, 0.5, 0.5, 0.4)
    elasticity: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5)
    mu: float = 0.5  # trips a period per unit of the harmonic mean of supply and queue
    gamma: float = 0.2  # how fast idle supply relaxes to sigma
    alpha: float = 0.3  # the share of open requests abandoned a period
    theta: float = 0.3  # how fast the incentive stock decays
    spill: float = 0.5
    shock_sd: float = 0.3
    shock_supply: float = 2.0  # drivers a period one unit of shock takes off the road
    policy_gain: float = 1.0  # how hard the logged operator moved both levers with the shock
    dither: float = 0.1  # the logged levers' own noise, which is what identifies their effects
    noise: float = 0.05  # supply and queue noise a period
    matching: Matching = "harmonic"

    def __post_init__(self) -> None:
        lengths = {name: len(getattr(self, name)) for name in _PER_ZONE}
        if len(set(lengths.values())) != 1 or 0 in lengths.values():
            raise ValueError(f"the per-zone tuples must share one non-zero length, got {lengths}")
        for name in ("supply", "queue", "mu", "gamma", "alpha", "theta"):
            if not np.all(np.asarray(getattr(self, name)) > 0.0):
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("recruit", "carry", "shock_sd", "shock_supply", "dither", "noise"):
            if not np.all(np.asarray(getattr(self, name)) >= 0.0):
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if not all(0.0 <= share < 1.0 for share in self.elasticity):
            raise ValueError(
                "elasticity must lie in [0, 1): a full price rise that turns away all demand or "
                f"more leaves a negative demand, got {self.elasticity}"
            )
        if not 0.0 <= self.spill <= 1.0:
            raise ValueError(f"spill must lie in [0, 1], got {self.spill}")
        if self.spill > 0.0 and self.zones == 1:
            raise ValueError("one zone has no neighbour to draw recruits from; set spill=0")
        if self.matching not in ("harmonic", "linear"):
            raise ValueError(f"matching must be 'harmonic' or 'linear', got {self.matching!r}")

    @property
    def zones(self) -> int:
        return len(self.supply)

    def _do_nothing_trips(self) -> NDArray[np.float64]:
        supply, queue = np.asarray(self.supply), np.asarray(self.queue)
        return self.mu * supply * queue / (supply + queue)

    def _transfer(self) -> NDArray[np.float64]:
        k = self.zones
        drawn = np.zeros((k, k))
        for i in range(k):
            neighbours = {(i - 1) % k, (i + 1) % k} - {i}
            for j in neighbours:
                drawn[j, i] = self.spill / len(neighbours)
        return np.eye(k) - drawn

    def plant(self, shock: ArrayLike | None = None) -> ZoneMarketPlant:
        """The true vector field at a demand shock per zone, none by default."""
        supply, queue = np.asarray(self.supply), np.asarray(self.queue)
        trips = self._do_nothing_trips()
        return ZoneMarketPlant(
            sigma=jnp.asarray(supply + trips / self.gamma),
            demand=jnp.asarray(trips + self.alpha * queue),
            recruit=jnp.asarray(self.recruit),
            carry=jnp.asarray(self.carry),
            elasticity=jnp.asarray(self.elasticity),
            transfer=jnp.asarray(self._transfer()),
            match_supply=jnp.asarray(self.mu * queue**2 / (supply + queue) ** 2),
            match_queue=jnp.asarray(self.mu * supply**2 / (supply + queue) ** 2),
            shock=jnp.zeros(self.zones) if shock is None else jnp.asarray(shock, dtype=float),
            mu=self.mu,
            gamma=self.gamma,
            alpha=self.alpha,
            theta=self.theta,
            shock_supply=self.shock_supply,
            harmonic=self.matching == "harmonic",
        )

    def _names(self, kind: str) -> tuple[str, ...]:
        return tuple(f"{kind}_{i}" for i in range(self.zones))

    @property
    def state_columns(self) -> tuple[str, ...]:
        """The plant's state order: every zone's supply, then every queue, then every stock."""
        return (*self._names("supply"), *self._names("queue"), *self._names("stock"))

    @property
    def lever_columns(self) -> tuple[str, ...]:
        """The plant's action order: every zone's incentive, then every price."""
        return (*self._names("incentive"), *self._names("price"))

    @property
    def shock_columns(self) -> tuple[str, ...]:
        return self._names("shock")

    @property
    def do_nothing(self) -> Array:
        """The steady state with no lever moved and no shock, in :attr:`state_columns` order."""
        return jnp.concatenate(
            [jnp.asarray(self.supply), jnp.asarray(self.queue), jnp.zeros(self.zones)]
        )

    def sample(
        self, *, n_days: int = 40, n_periods: int = 48, dt: float = 1.0, seed: int = 0
    ) -> dict[str, np.ndarray]:
        """Simulate the log period by period, each zone's levers moved with its own shock.

        Every day starts at the do-nothing point, with supply and queue jittered by ``noise``. A
        row holds the state at the start of its period and the levers and shock over it. Integrated
        with :func:`chc.integrate.rk4_step`, which the planner rolls out with, so the log carries no
        discretisation gap from the model class fitted to it.
        """
        rng = np.random.default_rng(seed)
        k = self.zones
        shocks = rng.normal(0.0, self.shock_sd, (n_periods, n_days, k))
        levers = np.clip(
            _LOGGED_LEVEL
            + self.policy_gain * shocks[:, :, None, :]
            + rng.normal(0.0, self.dither, (n_periods, n_days, 2, k)),
            0.0,
            1.0,
        ).reshape(n_periods, n_days, 2 * k)
        jitter = np.concatenate([np.full(2 * k, self.noise), np.zeros(k)])

        @jax.jit
        def advance(x: Array, u: Array, shock: Array) -> Array:
            step = lambda xi, ui, si: rk4_step(self.plant(si), 0.0, xi, ui, dt)  # noqa: E731
            return jax.vmap(step)(x, u, shock)

        state = np.asarray(self.do_nothing) + jitter * rng.normal(size=(n_days, 3 * k))
        states = np.empty((n_periods, n_days, 3 * k))
        for period in range(n_periods):
            states[period] = state
            moved = advance(jnp.asarray(state), jnp.asarray(levers[period]), shocks[period])
            state = np.asarray(moved) + jitter * rng.normal(size=(n_days, 3 * k))

        columns = {
            "day": np.repeat(np.arange(n_days), n_periods),
            "period": np.tile(np.arange(n_periods), n_days),
        }
        for values, names in (
            (states, self.state_columns),
            (levers, self.lever_columns),
            (shocks, self.shock_columns),
        ):
            for index, name in enumerate(names):
                columns[name] = values[:, :, index].T.reshape(-1)  # day by day
        return columns

    def graph(self) -> CausalGraph:
        """Each zone's shock moves its levers and both its states; an incentive moves its stock and
        the supply of every zone it recruits in or from, and so does the stock; a price moves its
        queue; supply moves its queue.

        Supply and queue drain each other through the trips within a period, and only supply's
        edge is drawn: a cycle has no adjustment set, and the reverse edge opens no back-door path
        from a lever, whose every confounder is a zone's shock either way.
        """
        transfer = self._transfer()
        supply, queue, stock = self._names("supply"), self._names("queue"), self._names("stock")
        incentive, price, shock = self._names("incentive"), self._names("price"), self.shock_columns
        edges: list[tuple[str, str]] = []
        for i in range(self.zones):
            edges += [
                (shock[i], incentive[i]),
                (shock[i], price[i]),
                (shock[i], supply[i]),
                (shock[i], queue[i]),
                (price[i], queue[i]),
                (incentive[i], stock[i]),
                (supply[i], queue[i]),
            ]
            for j in np.flatnonzero(transfer[:, i]):
                edges += [(incentive[i], supply[j]), (stock[i], supply[j])]
        return CausalGraph.from_edges(edges)

    def levers(self, *, incentive_cost: float = 0.0, price_cost: float = 0.0) -> list[Lever]:
        """Every lever on ``[0, 1]``, in :attr:`lever_columns` order."""
        costs = (incentive_cost,) * self.zones + (price_cost,) * self.zones
        return [
            Lever(name, lo=0.0, hi=1.0, unit_cost=cost)
            for name, cost in zip(self.lever_columns, costs, strict=True)
        ]

    def stock_dynamics(self) -> LinearDynamics:
        """The mechanical rows, exactly: the stock decays at ``theta`` and the incentive adds to it.

        In :attr:`state_columns` and :attr:`lever_columns` order and zero on every other row;
        handed to a fit as ``known=``, it leaves only supply and the queue to learn.
        """
        k = self.zones
        a = jnp.zeros((3 * k, 3 * k)).at[2 * k :, 2 * k :].set(-self.theta * jnp.eye(k))
        b = jnp.zeros((3 * k, 2 * k)).at[2 * k :, :k].set(jnp.eye(k))
        return LinearDynamics(a, b)

    @property
    def incentive_channel(self) -> NDArray[np.float64]:
        """Drivers a period an incentive held at one recruits once settled, per zone:
        ``recruit + carry / theta``. The channel :meth:`zone_decision`'s ``b`` stands for."""
        return np.asarray(self.recruit) + np.asarray(self.carry) / self.theta

    def zone_decision(
        self, *, target: ArrayLike, state_weight: ArrayLike, action_weight: ArrayLike
    ) -> ZoneDecision:
        """The incentives to hold, as a one-shot :class:`~chc.experiment.ZoneDecision`, prices at
        the list price.

        Its state is each zone's trips a period once the incentives have settled, its lever each
        zone's incentive on ``[0, 1]``, its channel :attr:`incentive_channel`::

            trips = m0 + diag(alpha mu_s / D)(I - P) diag(channel) incentive,
            D = gamma mu_q + gamma alpha + mu_s alpha,

        ``mu_s`` and ``mu_q`` the linear law's weights on supply and the queue (derived in
        ``validation/zone_market.mac``). Exact under linear matching; under harmonic matching, the
        settled trips' first-order response at the do-nothing point, where the two laws touch.
        """
        plant = self.plant()
        mu_s, mu_q = np.asarray(plant.match_supply), np.asarray(plant.match_queue)
        d = self.gamma * mu_q + self.gamma * self.alpha + mu_s * self.alpha
        return ZoneDecision(
            coupling=(self.alpha * mu_s / d)[:, None] * self._transfer(),
            baseline=self._do_nothing_trips(),
            target=np.asarray(target, dtype=np.float64),
            state_weight=np.asarray(state_weight, dtype=np.float64),
            action_weight=np.asarray(action_weight, dtype=np.float64),
            lo=np.zeros(self.zones),
            hi=np.ones(self.zones),
        )

    def linear_gaussian(self, dt: float = 1.0) -> LinearGaussianPlant:
        """One period of :meth:`sample`'s log as ``x' = a x + b u + offset + w``, under a policy
        that does not see the shock.

        The Jacobians of one RK4 period at the do-nothing point, so exact under linear matching.
        ``w`` carries the period's shock through the step and the supply and queue noise, with
        covariance ``shock_sd^2 e e'`` plus ``noise^2`` on those rows, ``e`` the step's response to
        a shock. The logged operator did see the shock, so its actions and ``w`` are correlated.
        """
        x0, u0 = self.do_nothing, jnp.zeros(2 * self.zones)

        def step(x: Array, u: Array, shock: Array) -> Array:
            return rk4_step(self.plant(shock), 0.0, x, u, dt)

        jacobians = jax.jacfwd(step, argnums=(0, 1, 2))(x0, u0, jnp.zeros(self.zones))
        a, b, e = (np.asarray(m, dtype=np.float64) for m in jacobians)
        rest = np.asarray(step(x0, u0, jnp.zeros(self.zones)), dtype=np.float64)
        noise = np.concatenate([np.full(2 * self.zones, self.noise**2), np.zeros(self.zones)])
        return LinearGaussianPlant(
            a=a,
            b=b,
            offset=rest - a @ np.asarray(x0, dtype=np.float64),
            noise=self.shock_sd**2 * e @ e.T + np.diag(noise),
        )
