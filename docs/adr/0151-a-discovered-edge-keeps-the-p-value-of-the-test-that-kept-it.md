# ADR 0151 — A discovered edge keeps the p-value of the test that kept it

**Status:** accepted, 2026-10-30. Experimental (`chc.discovery`). Adds two fields to
`LaggedGraph` and a method, `LaggedGraph.causal_graph`. Changes no parent, edge or pathway that
1.0.0 discovers, and neither `chc.graph` nor the decision record: `SCHEMA_VERSION` stays 3.

## Context

ADR 0145 gave a `CausalGraph`'s edges a source (`EdgeProvenance`): `asserted`, `mechanism`, or
`discovered` with the name of the test that kept the edge and that test's p-value, which the
decision record writes and replays. Nothing in the library produced a discovered edge.
`discover_lagged_parents` runs a partial-correlation test (`chc.independence.partial_corr_test`,
Fisher z, two-sided) for every candidate and every kept parent, and returned a boolean
`LaggedGraph`: every p-value it computed was dropped. A caller who wanted the record to say which
edges the data were asked about had to retype them by hand.

The selection is grow-shrink (Margaritis and Thrun, NIPS 1999; the IAMB of Tsamardinos, Aliferis
and Statnikov, FLAIRS 2003). A kept parent has been tested more than once: in the grow phase, given
the parents selected before it, and in each pass of the shrink, given the other parents kept at
that pass. Which of those p-values the edge carries is the decision.

## Decision

- **The p-value is the shrink's last pass.** The last pass tests each kept parent given every
  other kept parent of its target, and is the test that keeps it: it is the pass whose worst
  p-value falls below `alpha`. A target with one kept parent has it tested given none.
  `_select_parents` returns these p-values beside the columns; no test runs that the selection
  did not already run, and the selection is unchanged.
- **`LaggedGraph.state_p_values` and `control_p_values`**, after `max_lag` and default None, so a
  graph built positionally, by a caller or by `TigramiteDiscovery`, is built as before.
  `discover_lagged_parents` always fills them. They are float64 numpy arrays of the parents'
  shapes, nan exactly where there is no edge, held as a read-only copy and checked on
  construction. They are not jax arrays: without `jax_enable_x64` a jax array is float32, whose
  least positive number is about 1.4e-45, and the p-values of strong edges sit far below it; on
  the 600-step series the tests use, kept p-values reach 1.8e-246.
- **`LaggedGraph.causal_graph(state_names, control_names=None)`** returns
  `CausalGraph.unrolled` of the `(source, target, lag)` triples over `lags=max_lag`, and is equal
  to it. Each unrolled copy of an edge carries `EdgeProvenance("discovered", "partial_corr_test",
  p)`: the test ran once, on the lag-aligned design, and the edge it kept is time-homogeneous, so
  every slice's copy is that one edge. It refuses a count of names other than the components', a
  name given twice, and a graph without p-values, since a discovered edge carries its test's
  p-value. `chc.decision.prescribe` takes the graph as `adjustment=` and its record writes each
  edge discovered, with the p-value bit for bit.
- **The p-value is nominal.** The parents were chosen by the same data, so under the null of an
  edge the p-value read after the selection is not uniform. It is not a p-value valid given the
  selection (Berk, Brown, Buja, Zhang and Zhao, "Valid post-selection inference", Annals of
  Statistics 2013), and it is not corrected for the multiplicity the module already states. On a
  long series it is 0.0 where `erfc` underflows. The docstrings say each of these.

## Consequences

- An unrolled graph's edges are discovered; a decision fitted on it records which test kept each
  edge and at what p-value, and `Prescription.from_json` replays the record.
- Every parent, edge and pathway is what 1.0.0 produced: a test digests the parents and edges of
  two series at two levels and with a parent cap, and a pathway's impulse responses, as 1.0.0
  produced them.
- Two edges whose p-values underflowed to 0.0 are no longer ranked by them. The selection itself
  breaks such ties by the partial correlation for that reason; the record does not carry the
  correlation (below).
- `TigramiteDiscovery` returns a graph without p-values, so `causal_graph` refuses it.

## Alternatives

- **The grow phase's p-value.** Rejected: it was read given the parents selected before the
  candidate, and the conditioning set has changed since. A parent that entered as a proxy for one
  selected later is read again by the shrink, and the shrink's verdict is what kept the edge.
- **A statistic field on `EdgeProvenance`**, such as the partial correlation, which ranks edges
  whose p-values underflowed. Not added: it would change the stable `chc.graph` and the record's
  edges. Were it added, the rule would be an optional field, default None, read with `.get` so
  that every version-3 record still replays, with the schema version unchanged and a test of it.
- **Selective p-values.** Not built: a p-value valid given the selection is a different
  procedure, with its own conditioning on the selection event, not a field on this one.
- **Filling the p-values from tigramite's own matrix.** Not built: tigramite is not installable
  where this library is tested, and a path that no test runs is not shipped. A caller who uses
  `TigramiteDiscovery` can build the `CausalGraph` with `from_edges(..., provenance=)` from
  PCMCI's results.
- **`CausalGraph.unrolled(..., provenance=)`**, keyed by the triples, would build the same graph in
  one call. Not added here: it changes the stable `chc.graph`, and `from_edges(..., provenance=)` on
  the unrolled pairs gives the same graph from what exists.
