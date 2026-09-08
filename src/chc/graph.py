"""Named causal DAGs: which covariates to adjust for, decided by the graph instead of by hand.

Every effect estimator in this library takes its adjustment set as a literal tuple of column names
--- ``BackdoorOLS().estimate(data, covariates=("x", "z"))``. That tuple is a causal claim typed by
hand, and the two ways it goes wrong are silent: adjusting for a collider *opens* a path that was
closed (M-bias), and adjusting for a mediator removes part of the effect being estimated. Neither
shows up as an error, a warning, or a bad fit; both show up as a wrong number.

:class:`CausalGraph` takes the assumption where it belongs --- in the graph --- and derives the
tuple. :meth:`CausalGraph.adjustment_set` returns the canonical set of Perkovic, Textor, Kalisch and
Maathuis (2018), ``Adjust(X, Y, O) = (an(X u Y) n O) \\ forb(X, Y)``, which is valid **iff any set
of observed variables is**: an empty result therefore means "no adjustment needed", and a
``not_identified`` status means "no covariate adjustment identifies this effect", which is a
different statement and is worth failing on rather than guessing past.

This is a graphical criterion, so it inherits the graph's assumptions and nothing more. It says
which set to adjust for *given* the DAG; it does not test the DAG, and a missing edge is still a
missing edge. Discovery of the lagged graph from data is :mod:`chc.discovery`, whose
:class:`~chc.discovery.LaggedGraph` is a different object on purpose: that one is indexed, boolean
and time-lagged, for the dynamics; this one is named, static and cross-sectional, for estimators.

Pure Python over names and sets --- no NumPy, no JAX, nothing to trace. Graphs here are tens of
nodes, and the algorithms are linear in the edge count.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

AdjustmentStatus = Literal["identified", "not_identified"]
"""Whether *some* set of observed covariates identifies the effect by adjustment."""


class CyclicGraphError(ValueError):
    """A ``CausalGraph`` was built from edges containing a directed cycle."""


@dataclass(frozen=True)
class AdjustmentSet:
    """The covariates to adjust for, or the statement that no such set exists.

    ``covariates`` is empty in two entirely different situations, which is why :attr:`status` is
    reported beside it and not derived from emptiness: an *identified* effect needing no adjustment
    (a randomised lever) and a *not identified* one (confounded through a latent) both have nothing
    to condition on. Passing the first into an estimator is correct; passing the second is a wrong
    number with no symptom.
    """

    covariates: tuple[str, ...]
    status: AdjustmentStatus
    reason: str  # why, in words, naming the nodes responsible -- for logs and error messages

    def __bool__(self) -> bool:
        """True iff the effect is identified by adjustment, regardless of set size.

        Deliberately *not* ``len(self.covariates) > 0``: the truthiness a caller wants from this
        object is "may I proceed", and the empty-but-identified case must answer yes.
        """
        return self.status == "identified"


@dataclass(frozen=True)
class CausalGraph:
    """A directed acyclic graph over named variables, some of which may be unobserved.

    Nodes are column names, so a graph and a :class:`~chc.panel.Panel` share one vocabulary and a
    typo is caught by :meth:`require_columns` rather than by a silent ``KeyError`` three layers
    down. Latent nodes carry the confounding a dataset cannot adjust for; they are what makes
    ``not_identified`` reachable, and a graph with none of them can always adjust for something.
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]  # (parent, child)
    latent: frozenset[str]

    @classmethod
    def from_edges(
        cls, edges: Iterable[tuple[str, str]], *, latent: Iterable[str] = ()
    ) -> CausalGraph:
        """Build from ``(parent, child)`` pairs, rejecting cycles and unknown latent names.

        Nodes are collected from the edges, so an isolated node needs no declaration and cannot be
        declared; that is the right trade for a graph whose only use is path reasoning, where a node
        with no edges changes no answer.

        Raises:
            CyclicGraphError: if the edges contain a directed cycle, naming one of its nodes.
            ValueError: if a latent name is not a node, or an edge is a self-loop.
        """
        pairs = tuple((str(parent), str(child)) for parent, child in edges)
        for parent, child in pairs:
            if parent == child:
                raise ValueError(f"self-loop on {parent!r}: a variable cannot cause itself")
        seen: dict[str, None] = {}
        for parent, child in pairs:
            seen.setdefault(parent, None)
            seen.setdefault(child, None)
        nodes = tuple(seen)
        latent_set = frozenset(str(name) for name in latent)
        unknown = sorted(latent_set - set(nodes))
        if unknown:
            raise ValueError(
                f"latent nodes {unknown} do not appear in any edge; a latent variable that causes "
                "nothing and is caused by nothing cannot confound anything"
            )
        graph = cls(nodes=nodes, edges=tuple(dict.fromkeys(pairs)), latent=latent_set)
        graph._require_acyclic()
        return graph

    # ---- structure ------------------------------------------------------------------------

    @property
    def observed(self) -> tuple[str, ...]:
        """Nodes a dataset could contain a column for, in insertion order."""
        return tuple(node for node in self.nodes if node not in self.latent)

    def parents(self, node: str) -> tuple[str, ...]:
        return tuple(p for p, c in self.edges if c == node)

    def children(self, node: str) -> tuple[str, ...]:
        return tuple(c for p, c in self.edges if p == node)

    def ancestors(self, nodes: str | Sequence[str], *, inclusive: bool = True) -> frozenset[str]:
        """Every node with a directed path into ``nodes``; ``inclusive`` adds ``nodes`` itself."""
        return self._walk(nodes, self._parent_map(), inclusive=inclusive)

    def descendants(self, nodes: str | Sequence[str], *, inclusive: bool = True) -> frozenset[str]:
        """Every node reachable from ``nodes`` by a directed path; ``inclusive`` adds the source."""
        return self._walk(nodes, self._child_map(), inclusive=inclusive)

    def require_columns(self, available: Iterable[str]) -> None:
        """Check that every observed node has a column, naming all the missing ones at once.

        Raises:
            KeyError: listing every observed node absent from ``available``. Reporting them
                together rather than one per run is the difference between one fix and five.
        """
        missing = sorted(set(self.observed) - set(available))
        if missing:
            raise KeyError(
                f"graph names observed variables with no column: {missing}; "
                f"declare them latent or add the columns"
            )

    # ---- separation -----------------------------------------------------------------------

    def d_separated(
        self, first: str | Sequence[str], second: str | Sequence[str], given: Iterable[str] = ()
    ) -> bool:
        """Whether every path between the two sets is blocked by ``given`` (Bayes-ball).

        Koller and Friedman's reachability algorithm: walk ``(node, direction)`` states outward from
        the first set, where a collider passes the walk on only if it or one of its descendants is
        conditioned on. Linear in the edge count, and the primitive every other method here is
        written in terms of.
        """
        return not (self._reachable(first, given) & set(_as_tuple(second)))

    # ---- adjustment -----------------------------------------------------------------------

    def is_valid_adjustment_set(
        self,
        covariates: Iterable[str],
        *,
        treatment: str | Sequence[str],
        outcome: str | Sequence[str],
    ) -> bool:
        """The generalised adjustment criterion for ``covariates`` relative to ``(X, Y)``.

        Two conditions, and both are necessary: no covariate lies in ``forb(X, Y)`` (which is what
        rules out mediators and their descendants), and the covariates d-separate ``X`` from ``Y``
        in the *proper back-door graph* --- the DAG with the first edge of every proper causal path
        deleted, so that only the non-causal paths remain to be blocked.
        """
        given = set(covariates)
        unknown = sorted(given - set(self.nodes))
        if unknown:
            raise KeyError(f"covariates {unknown} are not nodes of this graph")
        if given & self.latent:
            return False
        if given & self._forbidden(treatment, outcome):
            return False
        return self._proper_backdoor(treatment, outcome).d_separated(treatment, outcome, given)

    def adjustment_set(
        self, *, treatment: str | Sequence[str], outcome: str | Sequence[str]
    ) -> AdjustmentSet:
        """The canonical adjustment set over the observed nodes, or why there is none.

        ``Adjust(X, Y, O) = (an(X u Y) n O) \\ forb(X, Y)`` is valid **iff some subset of the
        observed nodes is** (Perkovic et al. 2018, Theorem 3.4 and the ``O``-restricted corollary),
        so one construction answers both questions: build it, check it, and a failed check is a
        proof that nothing else would have worked either.
        """
        exposure, response = _as_tuple(treatment), _as_tuple(outcome)
        self._require_nodes(exposure + response)
        overlap = set(exposure) & set(response)
        if overlap:
            raise ValueError(f"{sorted(overlap)} is both treatment and outcome")
        latent_exposure = sorted((set(exposure) | set(response)) & self.latent)
        if latent_exposure:
            return AdjustmentSet(
                (),
                "not_identified",
                f"{latent_exposure} is latent, so no column carries it",
            )

        candidate = tuple(
            node
            for node in self.nodes
            if node in self.ancestors(exposure + response)
            and node not in self.latent
            and node not in self._forbidden(exposure, response)
        )
        if self.is_valid_adjustment_set(candidate, treatment=exposure, outcome=response):
            reason = (
                "no adjustment needed: no unblocked non-causal path"
                if not candidate
                else f"blocks every non-causal path from {list(exposure)} to {list(response)}"
            )
            return AdjustmentSet(candidate, "identified", reason)

        blame = sorted(self._open_confounders(exposure, response))
        return AdjustmentSet(
            (),
            "not_identified",
            f"no observed set blocks every non-causal path; latent {blame} confounds it"
            if blame
            else "no observed set blocks every non-causal path",
        )

    # ---- internals ------------------------------------------------------------------------

    def _parent_map(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {node: [] for node in self.nodes}
        for parent, child in self.edges:
            out[child].append(parent)
        return out

    def _child_map(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {node: [] for node in self.nodes}
        for parent, child in self.edges:
            out[parent].append(child)
        return out

    def _walk(
        self, nodes: str | Sequence[str], links: dict[str, list[str]], *, inclusive: bool
    ) -> frozenset[str]:
        sources = _as_tuple(nodes)
        self._require_nodes(sources)
        seen: set[str] = set()
        queue = deque(sources)
        while queue:
            node = queue.popleft()
            for other in links[node]:
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        return frozenset(seen | set(sources)) if inclusive else frozenset(seen - set(sources))

    def _require_nodes(self, names: Sequence[str]) -> None:
        unknown = sorted(set(names) - set(self.nodes))
        if unknown:
            raise KeyError(f"{unknown} are not nodes of this graph; nodes are {list(self.nodes)}")

    def _require_acyclic(self) -> None:
        indegree = dict.fromkeys(self.nodes, 0)
        for _, child in self.edges:
            indegree[child] += 1
        children = self._child_map()
        queue = deque(node for node, degree in indegree.items() if degree == 0)
        removed = 0
        while queue:
            node = queue.popleft()
            removed += 1
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if removed != len(self.nodes):
            stuck = sorted(node for node, degree in indegree.items() if degree > 0)
            raise CyclicGraphError(f"edges contain a directed cycle through one of {stuck}")

    def _reachable(self, sources: str | Sequence[str], given: Iterable[str]) -> set[str]:
        """Nodes connected to ``sources`` by a path active given ``given`` (Koller-Friedman 3.1)."""
        starts = _as_tuple(sources)
        self._require_nodes(starts)
        condition = set(given)
        self._require_nodes(tuple(condition))
        ancestral = self.ancestors(tuple(condition)) if condition else frozenset()
        parents, children = self._parent_map(), self._child_map()

        visited: set[tuple[str, str]] = set()
        found: set[str] = set()
        queue = deque((node, "up") for node in starts)
        while queue:
            node, direction = queue.popleft()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))
            if node not in condition:
                found.add(node)
            if direction == "up" and node not in condition:
                queue.extend((parent, "up") for parent in parents[node])
                queue.extend((child, "down") for child in children[node])
            elif direction == "down":
                if node not in condition:
                    queue.extend((child, "down") for child in children[node])
                if node in ancestral:  # a collider whose descendant is conditioned on
                    queue.extend((parent, "up") for parent in parents[node])
        return found

    def _causal_nodes(self, exposure: Sequence[str], response: Sequence[str]) -> frozenset[str]:
        """``cn(X, Y)``: the nodes lying on proper causal paths from ``X`` to ``Y``."""
        blocked = set(exposure)
        children = self._child_map()

        forward: set[str] = set()
        queue = deque(
            child for node in exposure for child in children[node] if child not in blocked
        )
        forward.update(queue)
        while queue:
            node = queue.popleft()
            for child in children[node]:
                if child not in blocked and child not in forward:
                    forward.add(child)
                    queue.append(child)

        backward = {node for node in self.ancestors(response) if node not in blocked}
        interior = forward & backward
        heads = {node for node in exposure if set(children[node]) & interior}
        return frozenset(interior | heads)

    def _forbidden(
        self, treatment: str | Sequence[str], outcome: str | Sequence[str]
    ) -> frozenset[str]:
        """``forb(X, Y) = de(cn(X, Y)) u X u Y`` --- the nodes adjustment must never touch.

        ``Y`` is in the literature's ``forb`` already whenever a proper causal path exists, since it
        is then in ``cn``. It is named explicitly because of the case where none does: the effect is
        zero, ``cn`` is empty, and the canonical construction would otherwise offer to adjust for
        the outcome itself --- which d-separates ``X`` from ``Y`` trivially and certifies nothing.
        """
        exposure, response = _as_tuple(treatment), _as_tuple(outcome)
        causal = self._causal_nodes(exposure, response)
        below = self.descendants(tuple(causal)) if causal else frozenset()
        return frozenset(below | set(exposure) | set(response))

    def _proper_backdoor(
        self, treatment: str | Sequence[str], outcome: str | Sequence[str]
    ) -> CausalGraph:
        """The DAG with the first edge of every proper causal path from ``X`` to ``Y`` removed."""
        exposure, response = _as_tuple(treatment), _as_tuple(outcome)
        causal = self._causal_nodes(exposure, response)
        kept = tuple(
            (parent, child)
            for parent, child in self.edges
            if not (parent in set(exposure) and child in causal)
        )
        return CausalGraph(nodes=self.nodes, edges=kept, latent=self.latent)

    def _open_confounders(self, exposure: Sequence[str], response: Sequence[str]) -> frozenset[str]:
        """Latent nodes whose measurement alone would identify the effect --- the actionable blame.

        Deliberately *not* "latents on some open path": that set contains nodes a caller can do
        nothing with, since measuring one of them need not close the others. This answers the
        question a caller can act on --- which single column, if it existed, would flip the status.
        """
        return frozenset(
            node
            for node in self.latent
            if CausalGraph(self.nodes, self.edges, self.latent - {node}).adjustment_set(
                treatment=exposure, outcome=response
            )
        )


def _as_tuple(names: str | Sequence[str]) -> tuple[str, ...]:
    return (names,) if isinstance(names, str) else tuple(names)
