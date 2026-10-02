# ADR 0045 — A geo world drawn from a stated hierarchy

**Status:** proposed, 2026-10-02. Experimental.

## Context

A model of geos is scored on whether it recovers each geo's effect and how far the geos spread
about the national one, and a plan over geos on what it returns on the truth. Neither can be read
off data, whose truth is unknown. `chc.mmm`'s `MarketingMixSystem` has regions, but they replicate
one plant: every region the same returns, a state of sales whose decay carries a lift on, and an
immediate return with no saturation. A plan over geos (`allocate_geos`, ADR 0043) and a model of a
channel's contribution read a channel as `chc.response` does: a week's spend through a carryover
kernel and a saturation curve, times a coefficient, added to the KPI.

## Decision

- **`chc.geo_world`**: `MediaChannel(name, curve, effect, spread=0, retention=0, spend=1,
  national=False)` states a channel nationally; `GeoMediaMix(channels, geos=20, weeks=104, ...)`
  states the geos, the log and the options; `GeoMediaMix.draw(seed)` returns a `GeoMediaWorld`, its
  arrays geo by week by channel.
- **The truth is in the planner's units.** A geo's lift on a channel is its population times a lift
  per head, the curve per head of the spend per head through a normalised geometric adstock, times
  the geo's effect and the week's drift. `GeoMediaWorld.cells(week)` returns each geo's channels as
  `chc.response.Channel`, the curve's scale times the population and the coefficient the
  population times the effect and the drift, so a plan made on the cells is read on the world with
  no model between them.
- **Per head**, so geos of any size share a curve: a geo twice the size spending twice as much lifts
  twice as much, and the effect is the KPI per head a week at the curve's ceiling.
- **A log-normal hierarchy.** A geo's effect on a channel is log-normal about the channel's national
  median with the channel's log-SD, independent across geos and channels; populations and bases per
  head are log-normal about the mix's. An effect is positive, and a hierarchical model with a log
  link reads the spread in logs.
- **National media is allotted by population**, so every geo sees the same spend per head on it and
  its lift per head differs by geo through the effect alone.
- **The confounding is `chc.mmm`'s**: the log's spend rises with the season by `policy`, a planner
  spending in the weeks the season raises the KPI. A geo's intensity on a channel, log-normal, and a
  week's jitter of mean 1 move the spend about that plan.
- **Spillover on a ring**, an option: the lift of a geo's own channels lifts each of its two
  neighbours by `spillover / 2` of itself; national media has no neighbour to spill to. The cells
  leave it out, so a plan on them is the plan that ignores it, and `GeoMediaWorld.lift` reads what
  any spend lifts, spillover included.
- **Drift**, an option: each channel's lift moves by a national factor, the exponential of a random
  walk from 0.
- **Common random numbers.** Every variate is drawn whatever the parameters, in one order, so two
  mixes that differ only in a parameter read the same variates at a seed.
- **The world is numpy; the curves are `chc.response`'s**, called on numpy arrays, so the KPI is the
  curves' to rounding in double precision and to single precision in JAX's default.

## Consequences

- `tests/test_geo_world.py` holds:
  - one seed, one world; another seed, another;
  - each geo's cells lifting what its media lift at home, through `chc.response`'s own kernel and
    curve, at three weeks of a drifting world, to `1e-12`;
  - a lift per head computed apart, Michaelis-Menten through the normalised kernel, to `1e-12`;
  - over 4000 geos, each channel's effects, the populations, the bases per head and the spend per
    head about the stated medians with the stated log-SDs, within five standard errors;
  - a week's jitter of mean 1 and its log-SD, over 5000 weeks;
  - the base's season, the allotment by population, the spend's rise with the season by the policy;
  - the spillover's share, its two neighbours, and that a geo's national spend moves only its own
    lift;
  - the drift's steps and the noise per head, each to its stated SD;
  - the same variates across a change of spillover and of policy;
  - a quarter planned on the world's cells by `allocate_geos`, national media held, returning at
    least the quarter held at its spend;
  - the refusals.
- All 25 mutations of the module tried fail a test.
- The hierarchy is one level, independent across channels: no correlation between a geo's effects,
  no trend, one season every geo shares. The neighbours are a ring of the geos in their order, not a
  geography. National media is allotted by population, not by exposure.
- No timing is quoted.

## Alternatives

- **`chc.mmm`'s system with a set of parameters for each region.** Its sales state carries a lift on
  by a decay no `chc.response.Channel` has, and its immediate return has no saturation, so a plan or
  a fit in the cells' units would be scored against a truth in other units, and the score would
  carry the approximation between them.
- **Effects normal about the median.** At the spreads a geo model meets, a share of the geos would
  draw a negative effect.
- **A graph of neighbours for spillover.** A ring is the least structure in which every geo has two
  neighbours and a geo's spend lifts only them; a graph is a parameter every test would have to
  choose.
