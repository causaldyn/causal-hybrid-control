# ADR 0122 — A graph is read over the period a transition spans

**Status:** accepted, 2026-10-21.

## Context

`prescribe` fits each state's step, `x_next - x`, on the levers, the states at the step's start
and the adjustment set. The states at the start are in that regression whatever the set holds.
In 0.19.0 a graph's set was derived on the graph over the columns, for the levers' effect on every
state together. Read so, a state is one node: the value the logger reads and the value the lever
moves are the same variable.

Two defects followed. A logger that reads a constrained state, `c -> u` beside `c -> y`, made `c`
an outcome that is its own treatment's parent, so no set identified the effect, and `prescribe`
refused a log that 0.18.0 had fitted (channel 0.822 into `y`, truth 0.8; -0.008 into `c`, truth
0). And a logger that reads the state its lever moves, `y -> u` beside `u -> y`, could not be
drawn at all: the two edges are a cycle, and `CausalGraph` refuses cycles. The docstring still
promised an identified empty set "for one set by the states alone".

## Decision

- A graph is read over one period: the levers at its start (`X`), the states a period on (`Y`),
  and the states at its start held (`I`), since the fit conditions on them.
- `CausalGraph.adjustment_set` takes `given=`, nodes every set holds. The set is
  `(an(X u Y u I) n O) \ forb(X, Y)`; it holds `I` and is valid iff some observed set that holds
  `I` is (van der Zander, Liskiewicz and Textor 2019, adjustment sets `I <= Z <= O`). A given
  node in `forb(X, Y)` leaves the effect not identified. A brute-force test checks the claim
  against every observed superset of `I` on random DAGs.
- A graph over the columns is read so: an edge into a state moves that state a period on, every
  other edge acts within the period, and each state's next value follows from its own. Edges
  into other columns are not copied into the second period: there they are descendants of the
  outcome or roots, and no path between `X` and `Y` passes through them unblocked.
- A logger that reads a state its lever moves is drawn with `CausalGraph.unrolled(lags=1)`: the
  logger's edges at lag 0, each move at lag 1, each state's own persistence at lag 1. The
  graph records its `lags`; `prescribe` reads it from slice `t-1` to slice `t`. Nodes at any
  other slice, and the columns at `t` other than the states, are latent there, because the fit
  adjusts for no column except at the transition's start.
- The set `prescribe` reports names the covariates. The states at the start are left out of it,
  as they are already in the fit.

## Consequences

- The constrained-state logger is identified again with `("w",)`. Its channel is the same, to the
  last bit, as the one from the graph without the logger's edge: 0.828 into `y` and -0.008 into
  `c` on the reproducing log, under RK4. A state-feedback logger drawn unrolled gives 0.830 and
  -0.397 (truth 0.8 and -0.4).
- A latent that confounds the lever with a constrained state is still refused, by both readings.
- The reading assumes that a latent drawn over the columns acts within one period. A latent that
  persists across periods makes the state at the start a collider, and has to be drawn unrolled
  at every lag it acts.

## Alternatives

- **Add the states to the conditioning set of the one-node reading.** Rejected: a state that is
  both the outcome and a node the set conditions on is no adjustment set. The two values have to
  be two nodes.
- **Accept cycles through a state in `CausalGraph.from_edges`.** Rejected: every method on the
  class assumes a DAG, and a cycle says nothing about which value of the state is read.
- **A `states=` argument on `CausalGraph`.** Rejected: `unrolled` already draws time, and one
  more constructor would be a second way to state the same lags.
