# ADR 0161 — A reachability gap reads the barrier condition at the grid's nodes

**Status:** accepted, 2026-11-24. Research certificate in another tier's module
(`chc.reachability`, `docs/api/index.md`). Restates what `BarrierReachabilityGap.valid_cbf`,
`HigherOrderBarrierGap.valid_cbf` and the two `ok` fields claim. No value moves.

## Context

`barrier_reachability_gap` evaluates the robust barrier condition

    r(x) = grad h . f + U (||B' grad h|| - radius ||grad h||)_+ + alpha h >= 0

at the nodes of a uniform 2-D grid and reads `valid_cbf = all(h < 0 or r >= 0)` over them. The
tube is solved by Lax-Friedrichs on the same nodes. `higher_order_barrier_gap` does the same with
`psi1` on the working set. Both fields said the condition holds "at EVERY point" of the set, "so
the theorem applies": a claim about the continuum, which the code never evaluates.

The proposed repair keeps the claim and makes it true: require `r >= L * delta` at each node,
`delta` a cell's half-diagonal and `L` a Lipschitz bound of `r` on the cell. That is sound when
`L` is a bound. The question is whether the module can compute one. `r`'s slope on a cell needs
the supremum over the cell of `grad^2 h` and the drift's Jacobian, beside `grad h` and `f`: the
continuum again. The module takes `barrier` and `drift` as black-box callables and evaluates
them at the nodes only.

A measurement settles it (`validation/reachability_grid_blind.mac`,
`tests/test_reachability.py::test_valid_cbf_reads_the_condition_at_the_grid_nodes_only`). On a
21-node grid of `[-1, 1]^2` (spacing 1/10), with `f = (1, 0)`, `B = 0`, `alpha = 1` and
`h = 1 - 4 phi`, where `phi` is a smooth bump of radius 1/25 about the centre `c` of the cell
`[0, 1/10]^2`:

- the bump's support is 1/20 from the cell's edges and `sqrt(2)/20` from its corners, so at every
  node `h = 1` and its gradient and Hessian are exactly zero, in float64 as in exact arithmetic.
  Every quantity read at the nodes is that of `h = 1`, whose residual is 1 everywhere;
- both gaps read `valid_cbf = True` and `ok = True`, with the safe, reachable and certified
  fractions all 1;
- between the nodes, at `c - (1/50, 0)`, the residual is `1 - (1636/9) e^(-4/3) = -46.9`;
- `h(c) = 1 - 4/e = -0.47`: the state `c - (1/25, 0)`, where `h = 1`, reaches `c` after 0.04
  under the drift. `{h >= 0}` is not forward invariant, and neither is the working set.

So no rule on the grid's data separates this barrier from `h = 1`. An `L` read off the nodes, by
differences of the residual or by its gradient there, is 0 on both, and the margin test passes
both.

## Decision

- **`valid_cbf` is the condition at every node of the set**: the nodes with `h >= 0`, or the
  working set's nodes for the higher-order gap. It is the theorem's hypothesis on the grid, not on
  the set. `ok` checks the tube at the same nodes, so a False `ok` is the grid's condition and the
  grid's tube disagreeing.
- **The documents say so.** The module, both dataclasses and both functions state that every
  verdict is read at the grid's nodes, with the bump as the reason a node margin cannot repair it.
  The same statement is in `chc.research.sensitivity`'s docstring, `docs/modules.md`,
  `docs/concepts/certificates.md` and `scripts/reachability_demo.py`'s text.
- No field, signature or number moves.

## Consequences

- A `valid_cbf` of True is evidence about the grid, the same evidence the tube gives. A caller
  who needs invariance of the set needs a bound the grid does not hold.
- The test pins both halves: on the bump, both gaps read valid while the residual is negative
  between nodes and the flow leaves the set; and both dataclasses' docstrings name the grid's
  nodes and no longer say "EVERY". Before this change the second half fails.

## Alternatives

- **A node margin with `L` read from the grid.** Rejected: it is not sound. On the bump every
  such `L` is 0.
- **A node margin with `L` from the caller**, as `ChannelSensitivity.curvature` takes a bound on
  how far a barrier bends. Sound wherever the bound is. Not taken now: it adds an argument to a
  certificate that exists to calibrate the barrier on small problems, and no caller asks for a
  continuum claim. A continuum claim also has to cover the cells whose nodes all have `h < 0` but
  which hold a sliver of `{h >= 0}`, so it needs a bound on `h`'s slope as well as `r`'s.
- **Interval arithmetic over each cell.** Sound for any traced callable. Rejected: it needs a
  dependency outside the runtime set.
- **Rename the field**, for example to `valid_cbf_on_grid`. Rejected: a public name change for a
  statement the docstring carries.
