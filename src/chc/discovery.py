"""Lagged causal-parent discovery for the dynamics -- a minimal, control-focused grow-shrink
selection (not PC, not PCMCI).

For each next-state component ``x^j_t`` this finds which lagged variables
``{x^i_{t-tau}, u^k_{t-tau}}`` are its direct parents with the MCI-calibrated partial-correlation
test (:func:`chc.independence.partial_corr_test`), in two phases (the grow-shrink of Margaritis
and Thrun, NIPS 1999; the IAMB of Tsamardinos, Aliferis and Statnikov, FLAIRS 2003). **Grow**: add
the candidate with the strongest association *given the already-selected parents*, until none is
significant. Conditioning on the
selected parents (dominant autoregressive lags are picked first) removes indirect-path false
positives -- ``x^i_{t-2} -> x^i_{t-1} -> x^j_t`` makes ``x^i_{t-2}`` look like a parent to a naive
test, but not once ``x^i_{t-1}`` is conditioned on. **Shrink**: drop, one at a time and weakest
first, every selected parent that is not significant given the others. A candidate picked early as
a proxy for a parent not yet selected -- a child's lag that tracks its parent's -- is read again
once that parent is in, and leaves; without this phase it stayed for good. The strongest candidate
is the one with the largest partial correlation, not the least p-value: on a long series both
p-values underflow to 0, and the tie fell to the column order.

``alpha`` is each test's level, with no correction for multiplicity: the step that stops tests
every candidate left at ``alpha``, so a target with ``m`` of them gains a false parent with
probability up to about ``m * alpha``. For a family-wise level, pass ``alpha`` over the candidates,
``alpha / ((d_state + d_control) * max_lag)``.

Each kept edge carries a p-value (:attr:`LaggedGraph.state_p_values`): the partial-correlation
test of the parent with its target given every *other* kept parent of that target, the shrink
phase's last pass, which kept it (one kept parent is tested given none). The test ran once, on the
lag-aligned design, so the p-value belongs to the time-homogeneous edge, not to a slice. It is
*nominal*: the parents were chosen by these same data, and a p-value read after a data-driven
selection is not uniform under the null. It is not a selective p-value, valid given the selection
(Berk, Brown, Buja, Zhang and Zhao, "Valid post-selection inference", Annals of Statistics 2013),
and it is not corrected for the multiplicity above. On a long series it reads 0.0 where ``erfc``
underflows, and two such edges are no longer ranked by it.

Deliberately minimal: lag-capped, next-state targets only, no contemporaneous-link search. The heavy
algorithms (PCMCI+/LPCMCI) stay a lazy, opt-in tigramite adapter; the point here is to feed
``GraphResidual``'s adjacency and the effect estimators' adjustment set from data, not by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from chc.frames import _real_numbers
from chc.graph import CausalGraph, EdgeProvenance
from chc.independence import partial_corr_test

__all__ = [
    "LaggedGraph",
    "TigramiteDiscovery",
    "discover_lagged_parents",
]


_TEST = "partial_corr_test"  # the test a discovered edge names (chc.independence)


@dataclass(frozen=True)
class LaggedGraph:
    """Discovered lagged parents. ``state_parents[j, i, tau-1]`` = ``x^i_{t-tau} -> x^j_t``.

    ``state_p_values`` and ``control_p_values`` hold, at each edge, the p-value of the test that
    kept it (see the module docstring for which test, on which conditioning set, and why it is
    nominal), and nan where there is no edge; :func:`discover_lagged_parents` fills them, and
    :class:`TigramiteDiscovery` and a graph built by hand leave them None. They are float64 numpy
    arrays, read-only, and not jax ones: without ``jax_enable_x64`` a jax array rounds to float32,
    which reads a p-value of 1e-300 as 0.0.

    Raises:
        ValueError: on p-values of another shape than their parents, not nan exactly where there is
            no edge, outside ``[0, 1]`` at an edge, or given for one kind of parent and not the
            other.
    """

    state_parents: Array  # (d_state, d_state, max_lag) bool
    control_parents: Array | None  # (d_state, d_control, max_lag) bool, or None if no controls
    max_lag: int
    state_p_values: np.ndarray | None = None  # (d_state, d_state, max_lag) float64, nan: no edge
    control_p_values: np.ndarray | None = None  # (d_state, d_control, max_lag), or None

    def __post_init__(self) -> None:
        given = self.state_p_values is not None
        if given and (self.control_parents is None) != (self.control_p_values is None):
            raise ValueError(
                "a graph with p-values holds one for each kind of parent it has: control_p_values "
                "is given exactly when control_parents is"
            )
        if not given and self.control_p_values is not None:
            raise ValueError("control_p_values is given without state_p_values")
        for name, parents in (
            ("state_p_values", self.state_parents),
            ("control_p_values", self.control_parents),
        ):
            p_values = getattr(self, name)
            if p_values is None:
                continue
            edge = np.asarray(parents, dtype=bool)
            held = np.array(p_values, dtype=np.float64)  # a copy, so the caller's array may move
            if held.shape != edge.shape:
                raise ValueError(f"{name} has shape {held.shape}, its parents {edge.shape}")
            if not np.array_equal(np.isnan(held), ~edge):
                raise ValueError(f"{name} is nan exactly where there is no edge")
            if not np.all((held[edge] >= 0.0) & (held[edge] <= 1.0)):
                raise ValueError(f"{name} holds a p-value outside [0, 1]")
            held.flags.writeable = False
            object.__setattr__(self, name, held)

    def adjacency(self) -> Array:
        """State-to-state coupling collapsed over lags -- a ``GraphResidual`` adjacency (0/1)."""
        return jnp.any(self.state_parents, axis=2).astype(float)

    def node_adjacency(self, node_dim: int) -> Array:
        """Collapse component parents to a node-level adjacency (nodes = blocks of ``node_dim``).

        ``adjacency[j, i] = 1`` iff any component of node ``i`` is a discovered lagged parent of any
        component of node ``j`` -- the adjacency ``GraphResidual`` consumes when the flat state is
        ``n_nodes`` blocks of ``node_dim``.
        """
        coupling = jnp.any(self.state_parents, axis=2)  # (d_state, d_state): row j <- source i
        n_nodes = coupling.shape[0] // node_dim
        blocks = coupling.reshape(n_nodes, node_dim, n_nodes, node_dim)
        return jnp.any(blocks, axis=(1, 3)).astype(float)

    def edges(self) -> list[tuple[int, int, int, str]]:
        """Every discovered edge as ``(target, source, lag, kind)`` (kind ``state``/``control``)."""
        found: list[tuple[int, int, int, str]] = []
        for kind, tensor in (("state", self.state_parents), ("control", self.control_parents)):
            if tensor is None:
                continue
            targets, sources, lags = np.nonzero(np.asarray(tensor))
            for target, source, lag in zip(targets, sources, lags, strict=True):
                found.append((int(target), int(source), int(lag) + 1, kind))
        return found

    def causal_graph(
        self, state_names: Sequence[str], control_names: Sequence[str] | None = None
    ) -> CausalGraph:
        """The edges as a :class:`~chc.graph.CausalGraph` unrolled over ``max_lag``, each edge
        ``discovered`` by ``partial_corr_test`` with its p-value.

        The graph is :meth:`CausalGraph.unrolled` of the ``(source, target, lag)`` triples over
        ``lags=max_lag``, and is equal to it. Every unrolled copy of an edge carries the edge's
        p-value: the test ran once, on the lag-aligned design, and the edge holds at every slice.
        :func:`chc.decision.prescribe` takes the graph as ``adjustment=``, and its record writes
        each edge's source, test and p-value.

        Args:
            state_names: one distinct name per state component, in the series' column order.
            control_names: one per control column; None or empty for a graph without controls.

        Raises:
            ValueError: on a count of names other than the components', a name given twice, or a
                graph without p-values (built by hand or by :class:`TigramiteDiscovery`): a
                discovered edge carries its test's p-value.
        """
        if self.state_p_values is None:
            raise ValueError(
                "this graph holds no p-values (it was built by hand or by TigramiteDiscovery), "
                "and a discovered edge carries its test's p-value; discover it with "
                "discover_lagged_parents"
            )
        if isinstance(state_names, str) or isinstance(control_names, str):
            raise ValueError("names are a sequence of names, not one string")
        names = {"state": tuple(state_names), "control": tuple(control_names or ())}
        counts = {
            "state": self.state_parents.shape[1],
            "control": 0 if self.control_parents is None else self.control_parents.shape[1],
        }
        for kind, count in counts.items():
            if len(names[kind]) != count:
                raise ValueError(
                    f"the graph has {count} {kind} components, and {len(names[kind])} "
                    f"{kind} names were given"
                )
        every = names["state"] + names["control"]
        repeated = sorted({name for name in every if every.count(name) > 1})
        if repeated:
            raise ValueError(f"each component has its own name; {repeated} are given twice")

        p_values = {"state": self.state_p_values, "control": self.control_p_values}
        triples = []
        provenance: dict[tuple[str, str], EdgeProvenance] = {}
        for target, source, lag, kind in self.edges():
            parent, child = names[kind][source], names["state"][target]
            triples.append((parent, child, lag))
            held = p_values[kind]
            assert held is not None  # __post_init__: p-values for each kind of parent there is
            found = EdgeProvenance("discovered", _TEST, float(held[target, source, lag - 1]))
            for step in range(self.max_lag - lag + 1):
                pair = (
                    CausalGraph.lagged_name(parent, step + lag),
                    CausalGraph.lagged_name(child, step),
                )
                provenance[pair] = found
        unrolled = CausalGraph.unrolled(triples, lags=self.max_lag)
        graph = CausalGraph.from_edges(
            unrolled.edges, latent=unrolled.latent, provenance=provenance
        )
        return replace(graph, lags=unrolled.lags)


def _lagged_design(
    series: np.ndarray, controls: np.ndarray | None, max_lag: int
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Stack every candidate ``x^i_{t-tau}`` / ``u^k_{t-tau}`` as columns aligned to ``x_t``.

    Returns the ``(n_rows, n_candidates)`` matrix and a list of ``(kind, index, lag)`` tags.
    """
    n = series.shape[0]
    columns: list[np.ndarray] = []
    tags: list[tuple[str, int, int]] = []
    blocks = [("state", series)] + ([("control", controls)] if controls is not None else [])
    for kind, data in blocks:
        for index in range(data.shape[1]):
            for lag in range(1, max_lag + 1):
                columns.append(data[max_lag - lag : n - lag, index])
                tags.append((kind, index, lag))
    return np.column_stack(columns), tags


def _scored(
    target: np.ndarray, candidates: np.ndarray, columns: list[int], given: list[int]
) -> list[tuple[float, float, int]]:
    """``(p_value, -|partial correlation|, column)`` for each column, given the ``given`` columns.

    Every column is tested on the same conditioning set, so its p-value falls as its ``|partial
    correlation|`` rises; the correlation breaks the ties of p-values that underflowed to 0.
    """
    conditioning = candidates[:, given] if given else None
    scored = []
    for c in columns:
        rho, p_value = partial_corr_test(candidates[:, c], target, conditioning)
        scored.append((float(p_value), -abs(float(rho)), c))
    return scored


def _select_parents(
    target: np.ndarray, candidates: np.ndarray, alpha: float, max_parents: int
) -> dict[int, float]:
    """Grow-shrink selection: add the strongest candidate given the current parents while it is
    significant, then drop the weakest parent given the rest while it is not.

    Returns each kept column and its p-value given the other kept columns, from the shrink's last
    pass, in the order the columns were selected. The grow phase's p-value of a column was read
    given the parents selected before it, a set that has changed since.
    """
    selected: list[int] = []
    remaining = list(range(candidates.shape[1]))
    while remaining and len(selected) < max_parents:
        best_p, _, best_c = min(_scored(target, candidates, remaining, selected))
        if not best_p < alpha:
            break  # nothing left is a significant parent given what we already have
        selected.append(best_c)
        remaining.remove(best_c)
    while selected:
        scores = [
            _scored(target, candidates, [c], [s for s in selected if s != c])[0] for c in selected
        ]
        worst_p, _, worst_c = max(scores)
        if worst_p < alpha:  # every parent left is significant given the others
            return {c: p_value for p_value, _, c in scores}
        selected.remove(worst_c)
    return {}


def discover_lagged_parents(
    series: np.ndarray,
    controls: np.ndarray | None = None,
    max_lag: int = 3,
    alpha: float = 0.01,
    max_parents: int | None = None,
) -> LaggedGraph:
    """Discover each state component's lagged parents from a trajectory; see the module docstring.

    ``series`` is ``(T, d_state)``; ``controls`` an optional ``(T, d_control)``. ``max_parents``
    caps the forward selection per target (default: the candidate count). Returns a ``LaggedGraph``.

    Each edge's p-value is the shrink's last test of it, given the target's other kept parents
    (``state_p_values``, ``control_p_values``); no test runs for it beyond the selection's own. It
    is nominal, not selective, and 0.0 where ``erfc`` underflows (see the module docstring).

    ``alpha`` is each test's level, not the graph's: a candidate that is not a parent enters at
    about that rate, so a graph of many candidates holds a false edge far more often; on three
    states and one control over three lags, 31 of whose 36 candidate edges are not parents, about
    1 % of those 31 entered at ``alpha=0.01``, and 26 % of 500 graphs of 300 steps and 27 % of 500
    of 1000 steps held at least one false edge.
    """
    series = np.asarray(_real_numbers(series, "series"), dtype=float)
    if series.ndim != 2:
        raise ValueError(f"series must be (T, d_state); got shape {series.shape}")
    controls = (
        None if controls is None else np.asarray(_real_numbers(controls, "controls"), dtype=float)
    )
    # a nan reads a nan p-value, which no test rejects and the selection's minimum may still pick
    if not (np.isfinite(series).all() and (controls is None or np.isfinite(controls).all())):
        raise ValueError("series and controls must be finite")
    d_state = series.shape[1]
    d_control = 0 if controls is None else controls.shape[1]
    design, tags = _lagged_design(series, controls, max_lag)
    cap = design.shape[1] if max_parents is None else max_parents

    state_parents = np.zeros((d_state, d_state, max_lag), dtype=bool)
    control_parents = np.zeros((d_state, d_control, max_lag), dtype=bool) if d_control else None
    state_p_values = np.full(state_parents.shape, np.nan)
    control_p_values = None if control_parents is None else np.full(control_parents.shape, np.nan)
    n = series.shape[0]
    for target_index in range(d_state):
        target = series[max_lag:n, target_index]
        for candidate, p_value in _select_parents(target, design, alpha, cap).items():
            kind, source_index, lag = tags[candidate]
            if kind == "state":
                state_parents[target_index, source_index, lag - 1] = True
                state_p_values[target_index, source_index, lag - 1] = p_value
            else:
                assert control_parents is not None  # a control parent implies d_control > 0
                assert control_p_values is not None
                control_parents[target_index, source_index, lag - 1] = True
                control_p_values[target_index, source_index, lag - 1] = p_value

    control_out = None if control_parents is None else jnp.asarray(control_parents)
    return LaggedGraph(
        jnp.asarray(state_parents), control_out, max_lag, state_p_values, control_p_values
    )


_TIGRAMITE_HINT = (
    "TigramiteDiscovery requires 'tigramite' (and 'joblib'), which is NOT a chc dependency: "
    "tigramite is GPL-3.0, so it can never be pinned here. Install it in your own environment: "
    "pip install tigramite joblib."
)


@dataclass(frozen=True)
class TigramiteDiscovery:
    """Adapter over tigramite's PCMCI (lazy import; requires tigramite installed).

    A drop-in for :func:`discover_lagged_parents` returning the same :class:`LaggedGraph`, but with
    PCMCI + a tigramite conditional-independence test doing the work. Reach for it when the native
    minimal grow-shrink is not enough -- dense coupling, or nonlinear dependencies via a
    nonparametric test (``cond_ind_test=GPDC()``/``CMIknn()``). tigramite is GPL-3.0, so it stays
    bring-your-own-env and is never a chc dependency.

    The graph holds no p-values (``state_p_values`` is None), so
    :meth:`LaggedGraph.causal_graph` refuses it: PCMCI's own p-value matrix is not read into it,
    since tigramite is not installed where this library is tested.
    """

    pc_alpha: float = 0.01
    cond_ind_test: Any = None  # a tigramite CI-test instance; defaults to ParCorr

    def discover(
        self, series: np.ndarray, controls: np.ndarray | None = None, max_lag: int = 3
    ) -> LaggedGraph:
        """Run PCMCI on ``[series, controls]`` and return the state components' lagged parents."""
        series = np.asarray(_real_numbers(series, "series"), dtype=float)
        controls = (
            None
            if controls is None
            else np.asarray(_real_numbers(controls, "controls"), dtype=float)
        )
        try:
            from tigramite import data_processing as pp
            from tigramite.independence_tests.parcorr import ParCorr
            from tigramite.pcmci import PCMCI
        except ImportError as exc:  # pragma: no cover - exercised only without tigramite
            raise ImportError(_TIGRAMITE_HINT) from exc

        d_state = series.shape[1]
        stacked = series if controls is None else np.column_stack([series, controls])
        test = self.cond_ind_test if self.cond_ind_test is not None else ParCorr()
        pcmci = PCMCI(dataframe=pp.DataFrame(stacked), cond_ind_test=test, verbosity=0)
        graph = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=self.pc_alpha)["graph"]

        d_control = stacked.shape[1] - d_state
        state_parents = np.zeros((d_state, d_state, max_lag), dtype=bool)
        control_parents = np.zeros((d_state, d_control, max_lag), dtype=bool) if d_control else None
        for target in range(d_state):  # only state components are targets; controls are exogenous
            for source in range(stacked.shape[1]):
                for lag in range(1, max_lag + 1):
                    if graph[source, target, lag] != "-->":  # x^source_{t-lag} -> x^target_t
                        continue
                    if source < d_state:
                        state_parents[target, source, lag - 1] = True
                    else:
                        assert control_parents is not None  # a control parent implies d_control > 0
                        control_parents[target, source - d_state, lag - 1] = True

        control_out = None if control_parents is None else jnp.asarray(control_parents)
        return LaggedGraph(jnp.asarray(state_parents), control_out, max_lag)
