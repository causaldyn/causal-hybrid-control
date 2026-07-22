"""Temporal causal pathways: which set of variables, over which lags, drives a target (plans/17-18).

A one-step causal tool answers "does X move the target, and in which direction?". CHC's dynamics
layer answers the *temporal* question: **which variables, at which lags and along which multi-step
chains, sequentially drive a target over time -- and which are actionable levers?**
:func:`causal_pathway` composes the two dynamics-causal primitives to answer it end to end:

1. **Discover** the lagged graph (:func:`chc.discovery.discover_lagged_parents`) -- the direct
   ``x^i_{t-tau} -> x^j_t`` edges, cleaned by conditioning on already-selected parents.
2. **Reach** every *ancestor* of the target by walking the graph backward (transitive closure over
   the directed lag-edges), recording each source's shortest onset lag -- the walk-sum *candidate*
   support (L1; distinct path products can cancel, so this bounds, not equals, the nonzero support).
3. **Estimate** each ancestor's signed *total* dynamic effect via Jorda local projections
   (:func:`chc.irf.local_projection_irf`) -- a signed impulse response, not a presence flag; the
   total effect does not condition on mediators, so an indirect lever's whole chain counts.
4. **Rank + truncate**: order by cumulative ``|effect|`` over the horizon; a near-zero link caps a
   path (weakest-link, L3), and truncating at a finite horizon loses geometrically little (L2).

The three laws behind steps 2-4 are machine-checked in ``proofs/causal_pathway.v`` and
``validation/causal_pathway.mac``: L1 IRF walk-sum (Lutkepohl 2005 eq. 2.1.22), L2 geometric
horizon-truncation (tail ``C*rho^{H+1}/(1-rho)``), L3 weakest-link multiplicative bottleneck.

HONESTY -- what the signs mean. This is regime (b) of the sign-credibility ladder: the sign is
*estimated conditional on an explicit, untestable causal model*, between SVAR sign-restrictions
(sign imposed) and LP-IV (sign instrument-identified). Each effect is backdoor-adjusted for the
target's own state only; that identifies the total effect of an exogenous / randomised lever and of
direct parents (Jorda conditional exogeneity), but a hidden confounder, a contemporaneous common
cause, or a discovery error that admits a collider can silently flip a sign -- add the confounder to
the trajectory and it is picked up, omit it and it stays confounded. LP and a suitably-specified VAR
target the *same* population impulse responses (Plagborg-Moller & Wolf 2021); finite-lag
implementations differ by truncation and finite-sample regularisation, so read horizons well beyond
``max_lag`` as increasingly variance-dominated extrapolation. A composed estimator -- no new
dependency, no new estimator.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

from chc.discovery import discover_lagged_parents
from chc.irf import local_projection_irf, structured_irf

_SIGN_TOL = 1e-8


@dataclass(frozen=True)
class PathwayEdge:
    """One source on the pathway to the target: its lag, signed effect, and actionability."""

    source: int  # source variable index (into ``series`` columns, or ``controls`` if actionable)
    kind: str  # "state" or "control"
    lag: int  # shortest onset lag: fewest steps from this source forward to the target (Law L1)
    sign: int  # +1 / -1 / 0 : sign of the peak (h>=1) dynamic effect on the target
    contribution: float  # cumulative |IRF| over h=1..horizon -- the ranking magnitude (Laws L2/L3)
    peak_horizon: int  # the horizon h at which the |effect| is largest
    actionable: bool  # kind == "control": a lever the controller can set directly
    irf: np.ndarray  # the full signed impulse response g_0..g_horizon of this source on the target


@dataclass(frozen=True)
class CausalPathway:
    """The ranked pathway to a target: each ancestor, its lag, signed effect, and lever flag."""

    target: int
    edges: tuple[PathwayEdge, ...]  # ranked, strongest cumulative |effect| first
    horizon: int

    def actionable(self) -> tuple[PathwayEdge, ...]:
        """The control levers on the pathway -- the sources the controller can actually set."""
        return tuple(edge for edge in self.edges if edge.actionable)

    def summary(self) -> str:
        """A ranked table: rank, variable, onset lag, sign, ``|effect|``, actionability."""
        lines = [f"causal pathway -> x{self.target} (horizon {self.horizon}):"]
        for rank, edge in enumerate(self.edges, start=1):
            var = f"{'u' if edge.kind == 'control' else 'x'}{edge.source}"
            arrow = {1: "+", -1: "-", 0: "0"}[edge.sign]
            tag = "actionable" if edge.actionable else "state"
            lines.append(
                f"  {rank}. {var:<5} onset lag {edge.lag}  sign {arrow}  "
                f"|effect| {edge.contribution:.3f}  peak h={edge.peak_horizon}  [{tag}]"
            )
        return "\n".join(lines)


def _name(kind: str, index: int) -> str:
    return f"{'c' if kind == 'control' else 's'}{index}"


def _ancestor_onsets(
    edges: list[tuple[int, int, int, str]], target: int
) -> dict[tuple[str, int], int]:
    """Backward BFS over the lag-graph: each ancestor of ``target`` -> its shortest onset lag.

    ``edges`` are ``(target, source, lag, kind)`` direct edges. Walking backward from the target and
    summing lags gives the *candidate* support of the IRF walk-sum (Law L1) -- path products can
    cancel, so it bounds, not equals, the nonzero support; the shortest cumulative lag is the
    onset -- the first horizon a source's influence can reach the target. Controls are exogenous
    leaves (no parents), so the walk terminates; the strict ``<`` guard makes any lag-cycle converge
    (lags are positive, so a node is revisited only while its onset strictly decreases).
    """
    parents_of: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    for edge_target, source, lag, kind in edges:
        parents_of[edge_target].append((kind, source, lag))

    onset: dict[tuple[str, int], int] = {}
    queue: deque[tuple[tuple[str, int], int]] = deque([(("state", target), 0)])
    while queue:
        (node_kind, node_index), cumulative = queue.popleft()
        if node_kind != "state":
            continue  # a control node is exogenous -- it has no lagged parents
        for source_kind, source_index, lag in parents_of.get(node_index, ()):
            node = (source_kind, source_index)
            reached = cumulative + lag
            if node not in onset or reached < onset[node]:
                onset[node] = reached
                queue.append((node, reached))
    return onset


def causal_pathway(
    series: np.ndarray,
    target: int,
    controls: np.ndarray | None = None,
    horizon: int = 8,
    max_lag: int = 3,
    alpha: float = 0.01,
) -> CausalPathway:
    """Discover the ranked temporal pathway driving ``series[:, target]``; see the module docstring.

    ``series`` is ``(T, d_state)``; ``controls`` an optional ``(T, d_control)`` of levers. Discovers
    the lagged graph, reaches every ancestor of ``target``, estimates each ancestor's signed total
    dynamic effect over ``horizon`` steps, and returns them ranked by cumulative ``|effect|``.
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 2:
        raise ValueError(f"series must be (T, d_state); got shape {series.shape}")
    controls = None if controls is None else np.asarray(controls, dtype=float)

    graph = discover_lagged_parents(series, controls, max_lag=max_lag, alpha=alpha)
    onsets = _ancestor_onsets(graph.edges(), target)

    data: dict[str, Array] = {
        _name("state", i): jnp.asarray(series[:, i]) for i in range(series.shape[1])
    }
    if controls is not None:
        data |= {_name("control", k): jnp.asarray(controls[:, k]) for k in range(controls.shape[1])}

    target_name = _name("state", target)
    found: list[PathwayEdge] = []
    for (kind, index), lag in onsets.items():
        source_name = _name(kind, index)
        adjust = () if source_name == target_name else (target_name,)
        irf = np.asarray(
            local_projection_irf(
                data, horizon, treatment=source_name, outcome=target_name, adjust_for=adjust
            )
        )
        dynamic = irf[1:]  # drop g_0: the h>=1 propagation, not the impact regression
        if dynamic.size and float(np.max(np.abs(dynamic))) > _SIGN_TOL:
            peak = int(np.argmax(np.abs(dynamic)))
            sign, peak_horizon = int(np.sign(dynamic[peak])), peak + 1
        else:
            sign, peak_horizon = 0, 0
        found.append(
            PathwayEdge(
                source=index,
                kind=kind,
                lag=lag,
                sign=sign,
                contribution=float(np.sum(np.abs(dynamic))),
                peak_horizon=peak_horizon,
                actionable=kind == "control",
                irf=irf,
            )
        )

    found.sort(key=lambda edge: edge.contribution, reverse=True)
    return CausalPathway(target=target, edges=tuple(found), horizon=horizon)


@dataclass(frozen=True)
class PathwayCertificate:
    """Numeric evidence that :func:`causal_pathway` recovers a known signed chain and the laws."""

    ranked_sources: tuple[str, ...]  # variables strongest-first (e.g. ("x1", "u0"))
    mediator_sign: int  # recovered sign of the direct mediator m -> x effect (truth: negative)
    control_sign: int  # recovered sign of the indirect lever u -> m -> x chain (truth: negative)
    control_onset_lag: int  # onset lag of the indirect lever u (truth: 2, via one mediator step)
    mediator_contribution: float  # cumulative |effect| of the true driver m (Law L3 reference)
    decoy_contribution: float  # cumulative |effect| of the unlinked decoy -- must be ~0 (Law L3)
    truncation_tail: float  # denoised IRF mass beyond the horizon (Law L2)
    geometric_bound: float  # the geometric majorant that must dominate the tail (Law L2)
    ok: bool


def causal_pathway_certificate(
    seed: int = 0, n: int = 6000, horizon: int = 8
) -> PathwayCertificate:
    """Recover a known lever -> mediator -> target chain: signs, onset lag, weakest-link, L2 tail.

    Ground truth: exogenous lever ``u`` drives mediator ``m`` (``+0.8``), ``m`` drives target ``x``
    (``-0.9``), plus an independent white-noise decoy with no path to ``x``. A correct pathway
    recovers ``m`` at onset lag 1 (negative sign), the indirect lever ``u`` at onset lag 2 (negative
    sign, the weakest-link product ``+0.8 * -0.9``, L3), keeps the unlinked decoy's contribution
    near zero (L3/L1), and the denoised mediator IRF's mass beyond the horizon stays under the
    geometric majorant ``|g_H| * r/(1-r)`` (L2, ``r`` the measured local decay ratio).
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n)  # exogenous, randomised lever -> its total effect is identified
    decoy = rng.standard_normal(n)  # white-noise decoy: no autocorrelation, no spurious regression
    m = np.zeros(n)
    x = np.zeros(n)
    noise = 0.1 * rng.standard_normal((n, 2))
    for t in range(1, n):
        m[t] = 0.5 * m[t - 1] + 0.8 * u[t - 1] + noise[t, 0]
        x[t] = 0.6 * x[t - 1] - 0.9 * m[t - 1] + noise[t, 1]

    series = np.column_stack([x, m, decoy])  # target = 0 (x), mediator = 1 (m), decoy = 2
    controls = u.reshape(-1, 1)  # control 0 = the actionable lever u
    path = causal_pathway(series, target=0, controls=controls, horizon=horizon, max_lag=3)

    by_key = {(edge.kind, edge.source): edge for edge in path.edges}
    mediator = by_key.get(("state", 1))
    lever = by_key.get(("control", 0))
    decoy_edge = by_key.get(("state", 2))
    mediator_contribution = mediator.contribution if mediator is not None else 0.0
    decoy_contribution = decoy_edge.contribution if decoy_edge is not None else 0.0

    # Law L2 via the DENOISED structured (AR-propagated) IRF: LP tails are too noisy to read decay
    # off directly, so use the AR geometric propagation itself for the truncation bound.
    struct = np.abs(
        structured_irf(
            {"s0": jnp.asarray(x), "s1": jnp.asarray(m)},
            2 * horizon,
            order=4,
            treatment="s1",
            outcome="s0",
            adjust_for=("s0",),
        )
    )
    tail_terms = struct[horizon + 1 :]  # |g_{H+1}|, |g_{H+2}|, ...
    ratio = float(struct[horizon + 1] / struct[horizon]) if struct[horizon] > 0 else 0.0
    truncation_tail = float(np.sum(tail_terms))
    geometric_bound = (
        float(struct[horizon]) * ratio / (1.0 - ratio) if 0.0 < ratio < 1.0 else float("inf")
    )

    ranked_sources = tuple(
        f"{'u' if edge.kind == 'control' else 'x'}{edge.source}" for edge in path.edges
    )
    ok = (
        (mediator.sign if mediator is not None else 0) == -1
        and (lever.sign if lever is not None else 0) == -1
        and (lever.lag if lever is not None else -1) == 2
        and decoy_contribution <= 0.1 * mediator_contribution
        and truncation_tail <= geometric_bound
    )
    return PathwayCertificate(
        ranked_sources=ranked_sources,
        mediator_sign=mediator.sign if mediator is not None else 0,
        control_sign=lever.sign if lever is not None else 0,
        control_onset_lag=lever.lag if lever is not None else -1,
        mediator_contribution=mediator_contribution,
        decoy_contribution=decoy_contribution,
        truncation_tail=truncation_tail,
        geometric_bound=geometric_bound,
        ok=ok,
    )
