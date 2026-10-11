# ADR 0079 — A national channel is one rate over its geos

**Status:** accepted, 2026-10-11. Extends ADR 0043 and ADR 0044: `allocate_geos` and
`budget_for_geos` take channels bought once for every geo. Their S-shaped cells are searched in
boxes as ADR 0076 searches a cell.

## Context

`allocate_geos` plans every cell, one geo's channel, at its own rate. A channel bought once for
every geo, national TV for one, has one spend a period, and each geo receives a fixed share of it,
its share of the population for one. Its cells move together. The planners had no way to say so,
and a per-geo plan of such a channel is a buy nobody can make.

On the tests' grid of three geos and three channels with carryover, at a budget of 400 a period,
the per-geo plan gains 25,039.2. It buys channel 1 at 43.9, 22.7 and 10.0 in the three geos: no
national buy at the population shares 0.2, 0.5 and 0.3 lands that way. The same total of 76.67
spread at those shares, every other cell as planned, gains 24,418.2. The best plan with channel 1
bought at those shares gains 24,504.5. It buys 52.05 of channel 1 and moves the rest to the other
cells. The per-geo plan's 534.7 more is not reachable with a national buy.

## Decision

- **`National(shares, lower, upper)`**, a frozen dataclass in `chc.allocation`. `shares` holds
  each geo's share of the national spend. Each share is positive and finite, and the shares sum to
  1 within as many epsilons as there are shares, the rounding that shares divided by a population
  total leave. `lower` and `upper` bound the national spend a period. The shares are held as
  given, in a read-only copy.
- **`national: Mapping[int, National] | None`** in `allocate_geos` and `budget_for_geos`, keyed
  by the channel's column in the grid. `cells`, `history` and the totals keep their shapes and
  meanings: a national channel still has a cell in every geo, with that geo's curve, carryover and
  history. A mapping by column holds each channel once, so a channel cannot be declared national
  twice.
- **A national channel's cells have no box of their own.** Its column of `lower` and `upper` holds
  `nan`, and the box is the national one. A finite value there, in either bound or in any geo, is
  a channel both national and per-geo, and is refused: the two boxes would say different things
  about one buy.
- **The national spend a period is one rate of the plan.** The geos' own cells are rates as
  before, in the grid's order, and each national channel's rate follows them. The national rate's
  worth is its cells' worths summed, each cell reading its share of the rate: its adstock's reach
  is scaled by the share. Every geo's cell worth is concave where its curve is, so the sum is too.
  The national rate counts whole in the budget and in its channel's total. It counts in each
  geo's total by the geo's share: a total is now a weighted sum of rates, and every other weight
  is 1. The linear programs, the Newton steps on the binding totals' prices, and the budgets'
  reach all read the weights.
- **One price per constraint, as before.** Inside its box the national rate returns, a currency
  unit, the budget's price plus its channel's price plus each geo's price weighted by that geo's
  share. Where every geo's total is fixed the budget's price passes to the geos, as before: the
  shares sum to 1, so the sum each rate meets does not move.
- **On S-shaped cells the national rate is searched in boxes as a cell is.** Over a box of the
  national rate, each cell's envelope over its share of the box (ADR 0051) is concave in the
  rate, above the cell's worth, and equal to it at both ends. Their sum has the same three
  properties. It bounds every plan in the box, and the search cuts the box at the plan's
  national rate. The sum can stand above the sum's own concave envelope, which the search closes
  by cutting.
- **The plan comes back as before.** `GeoAllocation.spend` keeps its `(geos, channels)` shape. A
  national channel's column holds each geo's share of the national rate, and sums to it to
  rounding. `channel_prices` prices the national spend over the plan.

## Consequences

- `tests/test_geo_national.py` holds:
  - two geos of two channels, channel 1 national, geo 0's total capped. Every plan whose national
    spend and geo 0's own cell are whole steps of a 400th of the budget is read on the curves. On
    concave curves at a budget of 500, with a cap of 200 that binds, the best of those plans
    returns 1727.135 and the plan returns 1727.201. On Hill curves at a budget of 3.2 the cap of
    0.8 binds: the plan and the lattice's best both return 1.51651, in 3 boxes. Under a cap of
    1.2 the plan returns 1.567857 against the lattice's 1.567855, in 5 boxes, and leaves geo 0 at
    0.90. With no cap, the search puts all the budget on the national channel in 9 boxes. The
    concave plan also solves its optimality conditions, solved apart in Maxima in 40-digit
    bigfloat: its rates, its worth and its prices agree to 2e-15 of each;
  - over one geo, each of the three channels national with a share of 1: `allocate`'s plan, to
    8e-15 in the rates and 2e-15 in the worth and the price;
  - two geos planned cell by cell, at a budget of 250 a period: channel 1 runs inside both boxes,
    at 48.98 and 27.13. Bought nationally at those ratios, within a national box that keeps each
    cell in its own box, the plan is the cell-by-cell plan to 7e-16 in the rates. At even shares
    it returns 69.4 less;
  - on the three geos, a cap on geo 0 and a floor on channel 2 that bind: every geo's own cell
    and the national channel meet their prices. Every geo's total fixed at the free plan's passes
    the budget's price to the geos. A cap 5% above what geo 1 spends does not bind, though geo 1
    would break it if the national spend counted whole there. A budget `1e-9` to `1e-3` below the
    most a cap on geo 0 allows is planned exactly;
  - a national box, and a channel total on the national channel, that hold its spend below the
    free plan's;
  - the budgets a goal runs over, worked out by hand under a floor and a cap on geo 0 that read
    the national spend by geo 0's share. A return target, and a marginal target at which the
    national channel returns the target a unit;
  - the refusals: shares that do not sum to 1, are not positive or are not one a geo; a national
    box that is not ordered or not finite; a channel both national and given a box in either
    bound; a column the grid does not have, a key that is not a column, a value that is not a
    `National`, and a `national` that is not a mapping; and a geo total outside what its cells'
    boxes spend, read with the national channel's share.
- Of 15 mutations of the change, 15 fail a test: each weight dropped from the programs, the
  Newton steps, the budgets' reach and the totals' check; the geos' weights set to 1; the share
  dropped from the plan's spend
  or from a cell's reach; a cell dropped from the national worth; the national bound left unbuilt,
  or rebuilt on concave cells, which turns on the search's guarded Newton steps; the channel
  total's weight read as a share; and the refusals of the shares, of a box of its own and of a
  key that is a boolean.
- With no national channel, every plan is the one before, bit for bit. Twenty-two plans were
  compared in double precision with the tree before this change, every field but the seconds:
  `allocate_geos` free, under totals, under fixed totals of geos and of channels, a hair below a
  cap and on a linear cell; the grid of S-curves free, under totals, under fixed channel totals,
  with carryover and at a cap of one box; each goal over the grid and over S-curves; and
  `allocate`, `budget_for` and `minimax_allocate`. All 322 fields are equal. In single precision,
  JAX's default, the same plans but the three goals over the grid with carryover, 19 plans and 277
  fields, are equal too.
- A national channel adds one rate and removes as many as there are geos, so its programs are
  smaller. On S-shaped cells its bound is the sum of its cells' envelopes, and a search may take
  more boxes than one cell's would. No timing is quoted.
- The shares are fixed. The plan chooses how much to buy, not where it lands.

## Alternatives

- **The cells as rates, tied by equality rows.** The rows would hold each national cell at its
  share of one rate, a row for each geo after the first. The box search would cut each cell
  apart. A cut of one cell leaves the envelopes of the others as loose as before, so the search
  would cut every cell of the channel to close it. Rejected for the one rate.
- **A box for each cell of a national channel, as well as the national box.** It would bound the
  national rate at each geo's cap over its share. It is a second box for one buy, and a column of
  boxes that disagree with the national one would plan a buy the caller did not set. Refused.
- **A channel total in place of the national box.** A channel's total bounds its spend over the
  plan, not a period's, and is optional. The search needs a finite box for each rate.
- **Shares divided by their sum.** That would change the caller's split silently. Shares that sum
  to 1 only to rounding are read as given; others are refused, with a message that says to divide
  them by their sum.
- **`Sequence[National]`, each naming its column.** A sequence can name a column twice. Rejected
  for the mapping.
- **A new field for the national spend on `GeoAllocation`.** The column of `spend` sums to it.
  Not added.
