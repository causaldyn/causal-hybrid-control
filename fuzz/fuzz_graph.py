"""Fuzz :mod:`chc.graph` against an oracle that shares none of its code.

    uv run --with-requirements fuzz/requirements.txt python fuzz/fuzz_graph.py -max_total_time=60

Each input decodes to an edge list over at most seven nodes -- self-loops, cycles and repeated edges
included -- a set of latent nodes, one or two treatments, an outcome and a conditioning set. The
graph either refuses the input with the error its docstrings name, or answers as brute force over
the skeleton's simple paths does:

* ``from_edges`` refuses a self-loop, a directed cycle and a latent name no edge carries, and
  accepts every other edge list;
* ``d_separated`` agrees with the path-blocking rule;
* ``is_valid_adjustment_set`` agrees with the generalised adjustment criterion of Perkovic et al.
  (2018), written over the proper paths, those that meet the treatments only where they start;
* ``adjustment_set`` returns a valid set when it says identified, and when it says not identified
  no subset of the observed nodes is valid, the completeness Perkovic et al. (2018) give it.

Anything else, a disagreement or another exception, is a crash with the input that caused it.
"""

from __future__ import annotations

import itertools
import sys

import atheris

with atheris.instrument_imports(include=["chc.graph"]):
    from chc.graph import CausalGraph, CyclicGraphError

NODES = tuple(f"v{index}" for index in range(7))
Edge = tuple[str, str]


def _cyclic(nodes: set[str], edges: set[Edge]) -> bool:
    """Whether repeatedly removing the nodes no edge enters leaves any behind."""
    remaining, live = set(nodes), set(edges)
    while True:
        sources = {node for node in remaining if not any(child == node for _, child in live)}
        if not sources:
            return bool(remaining)
        remaining -= sources
        live = {(parent, child) for parent, child in live if parent not in sources}


def _descendants(edges: set[Edge], node: str) -> set[str]:
    found, frontier = {node}, [node]
    while frontier:
        current = frontier.pop()
        for parent, child in edges:
            if parent == current and child not in found:
                found.add(child)
                frontier.append(child)
    return found


def _paths(edges: set[Edge], start: str, end: str) -> list[tuple[str, ...]]:
    """Every simple path from ``start`` to ``end`` in the skeleton, whichever way edges point."""
    linked = {(parent, child) for parent, child in edges} | {
        (child, parent) for parent, child in edges
    }
    found: list[tuple[str, ...]] = []

    def extend(path: tuple[str, ...]) -> None:
        if path[-1] == end:
            found.append(path)
            return
        for first, second in sorted(linked):
            if first == path[-1] and second not in path:
                extend((*path, second))

    extend((start,))
    return found


def _blocked(path: tuple[str, ...], edges: set[Edge], given: set[str]) -> bool:
    """A path is blocked by a conditioned non-collider, or by a collider with no conditioned
    descendant (itself included)."""
    for before, node, after in zip(path, path[1:], path[2:], strict=False):
        if (before, node) in edges and (after, node) in edges:
            if not _descendants(edges, node) & given:
                return True
        elif node in given:
            return True
    return False


def _causal(path: tuple[str, ...], edges: set[Edge]) -> bool:
    return all(step in edges for step in itertools.pairwise(path))


def _valid_adjustment(
    edges: set[Edge], latent: set[str], given: set[str], treatments: set[str], outcome: str
) -> bool:
    """No covariate latent, none a descendant of a node a proper causal path passes after its
    treatment, and every proper non-causal path blocked."""
    if given & latent:
        return False
    paths = [
        path
        for start in sorted(treatments)
        for path in _paths(edges, start, outcome)
        if not set(path[1:]) & treatments
    ]
    on_causal = {node for path in paths if _causal(path, edges) for node in path[1:]}
    forbidden = treatments.union(*(_descendants(edges, node) for node in on_causal))
    if given & forbidden:
        return False
    return all(_blocked(path, edges, given) for path in paths if not _causal(path, edges))


def _subset(provider: atheris.FuzzedDataProvider, pool: list[str]) -> set[str]:
    return {node for node in pool if provider.ConsumeBool()}


def check(data: bytes) -> None:
    provider = atheris.FuzzedDataProvider(data)
    size = provider.ConsumeIntInRange(2, len(NODES))
    edges = [
        (
            NODES[provider.ConsumeIntInRange(0, size - 1)],
            NODES[provider.ConsumeIntInRange(0, size - 1)],
        )
        for _ in range(provider.ConsumeIntInRange(0, 2 * size))
    ]
    present = {node for edge in edges for node in edge}
    latent = _subset(provider, list(NODES[:size]))
    unique = set(edges)

    if any(parent == child for parent, child in edges):
        try:
            CausalGraph.from_edges(edges, latent=latent)
        except ValueError:
            return
        raise AssertionError(f"a self-loop was accepted: {edges}")
    if latent - present:
        try:
            CausalGraph.from_edges(edges, latent=latent)
        except ValueError:
            return
        raise AssertionError(f"latent {sorted(latent - present)} was accepted with no edge")
    if _cyclic(present, unique):
        try:
            CausalGraph.from_edges(edges, latent=latent)
        except CyclicGraphError:
            return
        raise AssertionError(f"a cycle was accepted: {edges}")
    graph = CausalGraph.from_edges(edges, latent=latent)
    if set(graph.nodes) != present or set(graph.edges) != unique:
        raise AssertionError(f"{edges} became nodes {graph.nodes} and edges {graph.edges}")
    if len(present) < 2:
        return

    order = sorted(present)
    outcome = order[provider.ConsumeIntInRange(0, len(order) - 1)]
    treatments = {
        order[provider.ConsumeIntInRange(0, len(order) - 1)]
        for _ in range(provider.ConsumeIntInRange(1, 2))
    }
    if outcome in treatments:
        return
    treatment = tuple(sorted(treatments))
    others = [node for node in order if node != outcome and node not in treatments]
    given = _subset(provider, others)
    case = f"{edges}, latent {sorted(latent)}, {list(treatment)} -> {outcome}"

    separated = all(_blocked(path, unique, given) for path in _paths(unique, treatment[0], outcome))
    if graph.d_separated(treatment[0], outcome, given) != separated:
        raise AssertionError(f"{case}: d_separated({treatment[0]} | {sorted(given)}) is wrong")

    expected = _valid_adjustment(unique, latent, given, treatments, outcome)
    if graph.is_valid_adjustment_set(given, treatment=treatment, outcome=outcome) != expected:
        raise AssertionError(f"{case}: is_valid_adjustment_set({sorted(given)}) is not {expected}")

    answer = graph.adjustment_set(treatment=treatment, outcome=outcome)
    if answer.status == "identified":
        if not _valid_adjustment(unique, latent, set(answer.covariates), treatments, outcome):
            raise AssertionError(f"{case}: {answer.covariates} is not a valid adjustment set")
        return
    if (treatments | {outcome}) & latent:
        return
    observed = [node for node in others if node not in latent]
    for count in range(len(observed) + 1):
        for candidate in itertools.combinations(observed, count):
            if _valid_adjustment(unique, latent, set(candidate), treatments, outcome):
                raise AssertionError(f"{case}: not identified, yet {candidate} adjusts")


def main() -> None:
    atheris.Setup(sys.argv, check)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
