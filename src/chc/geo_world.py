"""A geo media-mix world: each geo's KPI made by its own media, its effects drawn from a stated
hierarchy, so a model fitted to it can be scored against the truth it was drawn from.

``geos`` geos over ``weeks`` weeks. A geo's KPI a week is its base, moved by a season shared by
every geo, plus each channel's lift, plus what its neighbours' media lift in it, plus noise:

    y_{g,t} = P_g b_g (1 + A sin(2 pi t / 52)) + sum_c m_{g,c,t} + o_{g,t} + e_{g,t}

* **Per head.** A channel's lift is the geo's population ``P_g`` times a lift per head,
  ``m_{g,c,t} = P_g beta_{g,c} D_{c,t} h_c(a_{g,c,t} / P_g)``: ``a`` the geo's spend on the channel
  through its normalised geometric adstock, ``h_c`` the channel's saturation curve with its scale in
  spend per head a week, and ``beta_{g,c}`` the KPI per head a week at the curve's ceiling. A geo
  twice the size spending twice as much lifts twice as much. As a :class:`chc.response.Channel`
  the lift is the curve with its scale times ``P_g`` and the coefficient ``P_g beta_{g,c} D_{c,t}``,
  which :meth:`GeoMediaWorld.cells` returns for :func:`chc.allocation.allocate_geos` to plan on.
* **The hierarchy.** ``log beta_{g,c} = log beta_c + tau_c z_{g,c}``, the ``z`` standard normal and
  independent across geos and channels: ``beta_c`` is the channel's national median and ``tau_c``
  the spread of the geos about it. A geo's base per head ``b_g`` and its population are log-normal
  about the mix's in the same way.
* **National media** is bought nationally and allotted to the geos by population, so every geo
  sees the same spend per head; its lift per head still differs by geo through ``beta_{g,c}``.
* **Spillover**, an option: a geo's lift on the channels it buys itself also lifts each of its two
  neighbours on a ring of the geos, in their order, by ``spillover / 2`` of what it lifts at home.
  National media has no neighbour to spill to.
* **Drift**, an option: each channel's lift moves with the weeks by a national factor
  ``D_{c,t} = exp(d_{c,t})``, ``d`` a random walk from 0 with steps of standard deviation
  ``drift`` a week.
* **The log's spend** puts the confounding of :mod:`chc.mmm` in: a geo's spend per head on a
  channel is the channel's mean times an intensity of the geo's, log-normal with log-SD
  ``spend_spread``, times ``1 + policy sin(2 pi t / 52)``, a planner raising spend in the weeks the
  season raises the KPI, times a week's log-normal jitter of mean 1 and log-SD ``spend_noise``.
* **Noise**: each head's KPI a week independent with standard deviation ``noise``, so a geo's is
  ``noise sqrt(P_g)``.

Every variate is drawn whatever the parameters, in one order, so two mixes that differ only in a
parameter read the same variates at the same seed: common random numbers for a paired comparison.

HONEST SCOPE:

* The hierarchy is log-normal and independent across channels: no correlation between a geo's
  effects, no trend, one annual season that every geo shares.
* The neighbours are a ring of the geos in their order, not a geography.
* National media's spend per head is the same in every geo, so the geos hold no contrast in it: a
  model reads its effect from the weeks alone, as a model of the whole country does, and its spread
  over the geos from how each geo follows the same series. Only the channels a geo buys itself
  vary across the geos.
* The curves are JAX's. In double precision (``jax_enable_x64``) the KPI is exact to rounding; in
  single, JAX's default, each lift is good to single precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import equinox as eqx
import numpy as np
from numpy.typing import ArrayLike

from chc.response import Channel, GeometricAdstock, Saturation

__all__ = ["GeoMediaMix", "GeoMediaWorld", "MediaChannel"]

_YEAR = 52.0


def _finite(name: str, value: float, *, least: float, strict: bool = False) -> None:
    if not (math.isfinite(value) and (value > least if strict else value >= least)):
        bound = f"above {least}" if strict else f"{least} or more"
        raise ValueError(f"{name} must be a finite number {bound}, got {value!r}")


@dataclass(frozen=True, eq=False)
class MediaChannel:
    """One channel of a :class:`GeoMediaMix`, its parameters national.

    Attributes:
        name: the channel's name.
        curve: its saturation per head, the scale in spend per head a week.
        effect: the KPI per head a week at the curve's ceiling, the median over the geos.
        spread: the log-SD of a geo's effect about the median.
        retention: the share of its adstock a week keeps; the kernel is normalised.
        spend: the log's mean spend per head a week.
        national: bought nationally and allotted to the geos by population.

    Raises:
        TypeError: a curve that is not a :class:`chc.response.Saturation`.
        ValueError: an effect or a spend not above 0, a negative spread, or a retention outside
            ``[0, 1)``.
    """

    name: str
    curve: Saturation
    effect: float
    spread: float = 0.0
    retention: float = 0.0
    spend: float = 1.0
    national: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.curve, Saturation):
            raise TypeError(
                f"channel {self.name!r}'s curve is a {type(self.curve).__name__}, "
                "not a chc.response.Saturation"
            )
        _finite(f"channel {self.name!r}'s effect", self.effect, least=0.0, strict=True)
        _finite(f"channel {self.name!r}'s spread", self.spread, least=0.0)
        _finite(f"channel {self.name!r}'s spend", self.spend, least=0.0, strict=True)
        if not 0.0 <= self.retention < 1.0:
            raise ValueError(
                f"channel {self.name!r}'s retention must be in [0, 1), got {self.retention!r}"
            )


@dataclass(frozen=True, eq=False)
class GeoMediaMix:
    """The parameters a world is drawn from; :meth:`draw` draws one.

    Attributes:
        channels: the mix's channels.
        geos: how many geos.
        weeks: how many weeks of history.
        population: a geo's median population.
        population_spread: the log-SD of the populations about it.
        base: the median geo's KPI per head a week before media and season.
        base_spread: the log-SD of the geos' bases per head about it.
        season: the annual season's amplitude, a share of the base, below 1.
        policy: how far the log's spend rises with the season, between -1 and 1: the confounding.
        spend_spread: the log-SD of a geo's spend per head on a channel about the channel's mean.
        spend_noise: the log-SD of a week's spend about its plan.
        noise: the standard deviation of one head's KPI a week.
        spillover: what a geo's media lifts in its two neighbours together, a share of its lift at
            home; above 0 it needs three geos for the ring.
        drift: the standard deviation a week of each channel's national random walk in the log of
            its lift.
        length: each adstock kernel's length in weeks.

    Raises:
        ValueError: on any parameter outside the ranges above, no channels, or two with one name.
    """

    channels: tuple[MediaChannel, ...]
    geos: int = 20
    weeks: int = 104
    population: float = 1e5
    population_spread: float = 1.0
    base: float = 1.0
    base_spread: float = 0.2
    season: float = 0.2
    policy: float = 0.0
    spend_spread: float = 0.5
    spend_noise: float = 0.3
    noise: float = 1.0
    spillover: float = 0.0
    drift: float = 0.0
    length: int = 8

    def __post_init__(self) -> None:
        if not self.channels or not all(isinstance(c, MediaChannel) for c in self.channels):
            raise ValueError("a mix needs one MediaChannel or more")
        names = [c.name for c in self.channels]
        if len(set(names)) < len(names):
            raise ValueError(f"the channels' names must differ, got {names}")
        for name in ("geos", "weeks", "length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a whole number, at least 1, got {value!r}")
        _finite("population", self.population, least=0.0, strict=True)
        _finite("base", self.base, least=0.0, strict=True)
        for name in (
            "population_spread",
            "base_spread",
            "spend_spread",
            "spend_noise",
            "noise",
            "spillover",
            "drift",
        ):
            _finite(name, getattr(self, name), least=0.0)
        if not 0.0 <= self.season < 1.0:
            raise ValueError(f"season must be in [0, 1), got {self.season!r}")
        if not -1.0 < self.policy < 1.0:
            raise ValueError(f"policy must be between -1 and 1, got {self.policy!r}")
        if self.spillover > 0.0 and self.geos < 3:
            raise ValueError(
                f"spillover runs on a ring of the geos, which needs three, got {self.geos}"
            )

    def draw(self, seed: int) -> GeoMediaWorld:
        """One world: its geos' populations and effects, the log's spend, and the KPI it made."""
        rng = np.random.default_rng(seed)
        geos, weeks, width = self.geos, self.weeks, len(self.channels)
        population = self.population * np.exp(self.population_spread * rng.standard_normal(geos))
        per_head = self.base * np.exp(self.base_spread * rng.standard_normal(geos))
        median = np.array([c.effect for c in self.channels])
        spread = np.array([c.spread for c in self.channels])
        effect = median * np.exp(spread * rng.standard_normal((geos, width)))
        intensity = np.exp(self.spend_spread * rng.standard_normal((geos, width)))
        steps = rng.standard_normal((weeks - 1, width))
        jitter = rng.standard_normal((geos, weeks, width))
        national_jitter = rng.standard_normal((weeks, width))
        shocks = rng.standard_normal((geos, weeks))

        season = np.sin(2.0 * np.pi * np.arange(weeks) / _YEAR)
        walk = np.vstack([np.zeros((1, width)), np.cumsum(self.drift * steps, axis=0)])
        planned = 1.0 + self.policy * season
        mean = np.array([c.spend for c in self.channels])
        sigma = self.spend_noise
        spend = (
            population[:, None, None]
            * mean
            * intensity[:, None, :]
            * planned[None, :, None]
            * np.exp(sigma * jitter - sigma**2 / 2.0)
        )
        shares = population / population.sum()
        for c, channel in enumerate(self.channels):
            if channel.national:
                bought = population.sum() * channel.spend * planned
                bought = bought * np.exp(sigma * national_jitter[:, c] - sigma**2 / 2.0)
                spend[:, :, c] = shares[:, None] * bought
        base = (population * per_head)[:, None] * (1.0 + self.season * season)
        drift = np.exp(walk)
        media, spilled = _lifts(self, population, effect, drift, spend)
        noise = self.noise * np.sqrt(population)[:, None] * shocks
        return GeoMediaWorld(self, population, effect, drift, spend, base, media, spilled, noise)


@dataclass(frozen=True, eq=False)
class GeoMediaWorld:
    """One draw of a :class:`GeoMediaMix`: its geos, the media they ran and the KPI it made.

    Arrays run geo by week by channel, the channels in the mix's order.

    Attributes:
        mix: what the world was drawn from.
        population: ``(geos,)`` each geo's population.
        effect: ``(geos, channels)`` each geo's KPI per head a week at each curve's ceiling, before
            drift.
        drift: ``(weeks, channels)`` each channel's national factor on its lift, 1 in week 0.
        spend: ``(geos, weeks, channels)`` the log's spend, national channels allotted by
            population.
        base: ``(geos, weeks)`` the KPI before media, its season in.
        media: ``(geos, weeks, channels)`` each channel's lift at home.
        spilled: ``(geos, weeks)`` what the neighbours' media lift in each geo.
        noise: ``(geos, weeks)``.
    """

    mix: GeoMediaMix
    population: np.ndarray
    effect: np.ndarray
    drift: np.ndarray
    spend: np.ndarray
    base: np.ndarray
    media: np.ndarray
    spilled: np.ndarray
    noise: np.ndarray

    @property
    def kpi(self) -> np.ndarray:
        """``(geos, weeks)``: the base, every channel's lift, what spills in, and the noise."""
        return self.base + self.media.sum(axis=2) + self.spilled + self.noise

    @property
    def shares(self) -> np.ndarray:
        """``(geos,)`` each geo's share of the population: what national media is allotted by."""
        return self.population / self.population.sum()

    def cells(self, week: int = -1) -> tuple[tuple[Channel, ...], ...]:
        """Each geo's channels as :class:`chc.response.Channel`, their lift at ``week``'s drift:
        a row a geo, as :func:`chc.allocation.allocate_geos` takes them. Spillover is not in
        them."""
        rows = []
        for g, people in enumerate(self.population):
            row = []
            for c, channel in enumerate(self.mix.channels):
                kernel = GeometricAdstock(
                    channel.retention, length=self.mix.length, normalized=True
                )
                curve = eqx.tree_at(lambda h: h.scale, channel.curve, channel.curve.scale * people)
                size = people * self.effect[g, c] * self.drift[week, c]
                row.append(Channel(kernel, curve, size))
            rows.append(tuple(row))
        return tuple(rows)

    def lift(self, spend: ArrayLike) -> np.ndarray:
        """``(geos, weeks)``: what ``spend``, geo by week by channel, lifts in each geo, at home and
        from its neighbours, with nothing spent before the first week."""
        spend = np.asarray(spend, dtype=float)
        if spend.shape != self.spend.shape:
            raise ValueError(
                f"spend must be {self.spend.shape}, geo by week by channel, got {spend.shape}"
            )
        media, spilled = _lifts(self.mix, self.population, self.effect, self.drift, spend)
        return media.sum(axis=2) + spilled


def _lifts(
    mix: GeoMediaMix,
    population: np.ndarray,
    effect: np.ndarray,
    drift: np.ndarray,
    spend: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Each channel's lift at home, ``(geos, weeks, channels)``, and what spills into each geo."""
    weeks = spend.shape[1]
    media = np.empty_like(spend)
    for c, channel in enumerate(mix.channels):
        weights = np.asarray(
            GeometricAdstock(channel.retention, length=mix.length, normalized=True).weights()
        )
        adstock = np.zeros_like(spend[:, :, c])
        for lag, weight in enumerate(weights[:weeks]):
            adstock[:, lag:] += weight * spend[:, : weeks - lag, c]
        head = np.asarray(channel.curve(adstock / population[:, None]), dtype=float)
        media[:, :, c] = population[:, None] * effect[:, c][:, None] * drift[:, c] * head
    local = [c for c, channel in enumerate(mix.channels) if not channel.national]
    home = media[:, :, local].sum(axis=2)
    spilled = mix.spillover / 2.0 * (np.roll(home, 1, axis=0) + np.roll(home, -1, axis=0))
    return media, spilled
