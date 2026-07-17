"""Lagged causal-parent discovery for the dynamics -- a minimal, control-focused PC1 (not PCMCI).

For each next-state component ``x^j_t`` this finds which lagged variables
``{x^i_{t-tau}, u^k_{t-tau}}`` are its direct parents, by greedy forward selection with the
MCI-calibrated partial-correlation test (:func:`chc.independence.partial_corr_test`): at each step
add the candidate with the strongest association *given the already-selected parents*, until none is
significant. Conditioning on the selected parents (dominant autoregressive lags are picked first)
removes indirect-path false positives -- ``x^i_{t-2} -> x^i_{t-1} -> x^j_t`` makes ``x^i_{t-2}``
look like a parent to a naive test, but not once ``x^i_{t-1}`` is conditioned on. See ``plans/17``.

Deliberately minimal: lag-capped, next-state targets only, no contemporaneous-link search. The heavy
algorithms (PCMCI+/LPCMCI) stay a lazy, opt-in tigramite adapter; the point here is to feed
``GraphResidual``'s adjacency and the effect estimators' adjustment set from data, not by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

from chc.independence import partial_corr_test


@dataclass(frozen=True)
class LaggedGraph:
    """Discovered lagged parents. ``state_parents[j, i, tau-1]`` = ``x^i_{t-tau} -> x^j_t``."""

    state_parents: Array  # (d_state, d_state, max_lag) bool
    control_parents: Array | None  # (d_state, d_control, max_lag) bool, or None if no controls
    max_lag: int

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


def _select_parents(
    target: np.ndarray, candidates: np.ndarray, alpha: float, max_parents: int
) -> list[int]:
    """Greedy forward selection: add the most-significant candidate given the current parents."""
    selected: list[int] = []
    remaining = list(range(candidates.shape[1]))
    while remaining and len(selected) < max_parents:
        conditioning = candidates[:, selected] if selected else None
        scored = [
            (float(partial_corr_test(candidates[:, c], target, conditioning)[1]), c)
            for c in remaining
        ]
        best_p, best_c = min(scored)
        if best_p >= alpha:
            break  # nothing left is a significant parent given what we already have
        selected.append(best_c)
        remaining.remove(best_c)
    return selected


def discover_lagged_parents(
    series: Array,
    controls: Array | None = None,
    max_lag: int = 3,
    alpha: float = 0.01,
    max_parents: int | None = None,
) -> LaggedGraph:
    """Discover each state component's lagged parents from a trajectory; see the module docstring.

    ``series`` is ``(T, d_state)``; ``controls`` an optional ``(T, d_control)``. ``max_parents``
    caps the forward selection per target (default: the candidate count). Returns a ``LaggedGraph``.
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 2:
        raise ValueError(f"series must be (T, d_state); got shape {series.shape}")
    controls = None if controls is None else np.asarray(controls, dtype=float)
    d_state = series.shape[1]
    d_control = 0 if controls is None else controls.shape[1]
    design, tags = _lagged_design(series, controls, max_lag)
    cap = design.shape[1] if max_parents is None else max_parents

    state_parents = np.zeros((d_state, d_state, max_lag), dtype=bool)
    control_parents = np.zeros((d_state, d_control, max_lag), dtype=bool) if d_control else None
    n = series.shape[0]
    for target_index in range(d_state):
        target = series[max_lag:n, target_index]
        for candidate in _select_parents(target, design, alpha, cap):
            kind, source_index, lag = tags[candidate]
            if kind == "state":
                state_parents[target_index, source_index, lag - 1] = True
            else:
                control_parents[target_index, source_index, lag - 1] = True  # type: ignore[index]

    control_out = None if control_parents is None else jnp.asarray(control_parents)
    return LaggedGraph(jnp.asarray(state_parents), control_out, max_lag)
