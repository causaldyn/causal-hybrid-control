"""The adjustment criterion on the DAGs it is famous for getting wrong, then on random ones.

The named cases are the ones a hand-typed ``covariates=`` tuple gets wrong: a collider that must
not be adjusted for, a mediator that must not be adjusted for, and a latent that makes the effect
unidentifiable however hard one tries. The random cases are the load-bearing check, and both of
them compare against an implementation that shares no code with the one under test:
d-separation against textbook path enumeration, and the canonical adjustment set against an
exhaustive search over every subset of the observed variables.
"""

from __future__ import annotations

import itertools
import random

import pytest

from chc.graph import CausalGraph, CyclicGraphError

CONFOUNDER = [("z", "x"), ("z", "y"), ("x", "y")]
MEDIATOR = [("x", "m"), ("m", "y")]
M_BIAS = [("u1", "z"), ("u2", "z"), ("u1", "x"), ("u2", "y"), ("x", "y")]
FRONT_DOOR = [("u", "x"), ("u", "y"), ("x", "m"), ("m", "y")]
BUTTERFLY = [("u1", "z"), ("u2", "z"), ("u1", "x"), ("u2", "y"), ("z", "x"), ("z", "y"), ("x", "y")]


def test_a_confounder_is_adjusted_for_and_a_mediator_is_not() -> None:
    confounded = CausalGraph.from_edges(CONFOUNDER)
    assert confounded.adjustment_set(treatment="x", outcome="y").covariates == ("z",)

    mediated = CausalGraph.from_edges(MEDIATOR)
    result = mediated.adjustment_set(treatment="x", outcome="y")
    assert result.status == "identified"
    assert result.covariates == ()  # adjusting for m would remove the very effect being estimated
    assert not mediated.is_valid_adjustment_set(("m",), treatment="x", outcome="y")


def test_the_m_bias_collider_must_not_be_adjusted_for() -> None:
    """The case a "control for everything you measured" rule gets exactly backwards."""
    graph = CausalGraph.from_edges(M_BIAS, latent=("u1", "u2"))
    result = graph.adjustment_set(treatment="x", outcome="y")
    assert result.status == "identified"
    assert result.covariates == ()  # x and y are already d-separated by the collider at z
    assert not graph.is_valid_adjustment_set(("z",), treatment="x", outcome="y")
    assert graph.is_valid_adjustment_set((), treatment="x", outcome="y")

    # the mechanism, on the back-door graph alone: with the direct edge gone the path through the
    # collider is the only one left, and it is closed until z is conditioned on.
    backdoor = CausalGraph.from_edges([e for e in M_BIAS if e != ("x", "y")], latent=("u1", "u2"))
    assert backdoor.d_separated("x", "y", ())
    assert not backdoor.d_separated("x", "y", ("z",))  # conditioning OPENS the path


def test_a_latent_confounder_is_reported_as_not_identified_and_blamed_by_name() -> None:
    graph = CausalGraph.from_edges(FRONT_DOOR, latent=("u",))
    result = graph.adjustment_set(treatment="x", outcome="y")
    assert result.status == "not_identified"
    assert not result  # the whole point of __bool__: empty-and-identified must not read the same
    assert "'u'" in result.reason or "u" in result.reason
    # m is on the causal path, so it is forbidden even though adjusting for it "closes" nothing
    assert not graph.is_valid_adjustment_set(("m",), treatment="x", outcome="y")


def test_the_butterfly_needs_the_collider_it_also_forbids() -> None:
    """z is simultaneously a confounder (z->x, z->y) and a collider (u1->z<-u2)."""
    graph = CausalGraph.from_edges(BUTTERFLY, latent=("u1", "u2"))
    assert graph.adjustment_set(treatment="x", outcome="y").status == "not_identified"
    observed = CausalGraph.from_edges(BUTTERFLY)  # measure u1 and u2 and it becomes identifiable
    assert observed.adjustment_set(treatment="x", outcome="y").status == "identified"


def test_an_identified_effect_with_nothing_to_adjust_for_is_still_truthy() -> None:
    randomised = CausalGraph.from_edges([("x", "y")])
    result = randomised.adjustment_set(treatment="x", outcome="y")
    assert result.covariates == ()
    assert bool(result) is True


def test_the_outcome_is_never_offered_as_a_covariate() -> None:
    """With no causal path X -> Y the canonical set would include Y and then certify nothing."""
    backwards = CausalGraph.from_edges([("y", "x"), ("z", "x"), ("z", "y")])
    result = backwards.adjustment_set(treatment="x", outcome="y")
    assert "y" not in result.covariates
    assert not backwards.is_valid_adjustment_set(("y",), treatment="x", outcome="y")


def test_multiple_treatments_are_handled_as_a_set() -> None:
    graph = CausalGraph.from_edges([("z", "x1"), ("z", "x2"), ("z", "y"), ("x1", "y"), ("x2", "y")])
    result = graph.adjustment_set(treatment=("x1", "x2"), outcome="y")
    assert result.covariates == ("z",)


def test_a_cycle_and_a_self_loop_are_refused_at_construction() -> None:
    with pytest.raises(CyclicGraphError):
        CausalGraph.from_edges([("a", "b"), ("b", "c"), ("c", "a")])
    with pytest.raises(ValueError, match="self-loop"):
        CausalGraph.from_edges([("a", "a")])
    with pytest.raises(ValueError, match="latent"):
        CausalGraph.from_edges([("a", "b")], latent=("ghost",))


def test_require_columns_names_every_missing_column_at_once() -> None:
    graph = CausalGraph.from_edges(CONFOUNDER, latent=())
    with pytest.raises(KeyError) as caught:
        graph.require_columns(["x"])
    assert "'y'" in str(caught.value)
    assert "'z'" in str(caught.value)


# ---- cross-checks against independent implementations ---------------------------------------


def _undirected_paths(
    nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...], src: str, dst: str
) -> list[list[tuple[str, str, str]]]:
    """Every simple path from ``src`` to ``dst``, each step tagged ``out`` (->) or ``in`` (<-)."""
    links: dict[str, list[tuple[str, str]]] = {node: [] for node in nodes}
    for parent, child in edges:
        links[parent].append((child, "out"))
        links[child].append((parent, "in"))
    found: list[list[tuple[str, str, str]]] = []

    def walk(node: str, seen: set[str], path: list[tuple[str, str, str]]) -> None:
        if node == dst and path:
            found.append(list(path))
            return
        for other, direction in links[node]:
            if other not in seen:
                walk(other, seen | {other}, [*path, (node, direction, other)])

    walk(src, {src}, [])
    return found


def _blocked(
    path: list[tuple[str, str, str]], given: set[str], descendants: dict[str, set[str]]
) -> bool:
    """The textbook rule node by node: a non-collider in Z, or a collider with no descendant."""
    sequence = [path[0][0]] + [step[2] for step in path]
    directions = [step[1] for step in path]
    for i in range(1, len(sequence) - 1):
        collider = directions[i - 1] == "out" and directions[i] == "in"
        if collider:
            if not descendants[sequence[i]] & given:
                return True
        elif sequence[i] in given:
            return True
    return False


def _d_separated_by_enumeration(graph: CausalGraph, x: str, y: str, given: tuple[str, ...]) -> bool:
    conditioned = set(given)
    descendants = {node: set(graph.descendants(node)) for node in graph.nodes}
    return all(
        _blocked(path, conditioned, descendants)
        for path in _undirected_paths(graph.nodes, graph.edges, x, y)
    )


def _random_dag(size: int, density: float, rng: random.Random) -> list[tuple[str, str]]:
    order = [f"v{i}" for i in range(size)]
    return [
        (order[i], order[j])
        for i in range(size)
        for j in range(i + 1, size)
        if rng.random() < density
    ]


def test_bayes_ball_agrees_with_path_enumeration_on_random_dags() -> None:
    rng = random.Random(7)
    checked = 0
    for _ in range(60):
        edges = _random_dag(rng.randint(4, 7), 0.5, rng)
        if not edges:
            continue
        graph = CausalGraph.from_edges(edges)
        x, y = rng.sample(list(graph.nodes), 2)
        rest = [node for node in graph.nodes if node not in (x, y)]
        for size in range(min(3, len(rest)) + 1):
            for given in itertools.combinations(rest, size):
                assert graph.d_separated(x, y, given) == _d_separated_by_enumeration(
                    graph, x, y, given
                ), (edges, x, y, given)
                checked += 1
    assert checked > 500  # the loop actually ran; a silently empty sweep proves nothing


def test_the_canonical_set_is_valid_exactly_when_some_observed_set_is() -> None:
    """Perkovic et al.'s completeness claim, checked by brute force over every observed subset."""
    rng = random.Random(23)
    identified = unidentified = 0
    for _ in range(60):
        edges = _random_dag(rng.randint(4, 7), 0.5, rng)
        if not edges:
            continue
        nodes = list(CausalGraph.from_edges(edges).nodes)
        x, y = rng.sample(nodes, 2)
        rest = [node for node in nodes if node not in (x, y)]
        latent = tuple(node for node in rest if rng.random() < 0.35)
        graph = CausalGraph.from_edges(edges, latent=latent)
        result = graph.adjustment_set(treatment=x, outcome=y)
        candidates = [node for node in graph.observed if node not in (x, y)]
        exhaustive = any(
            graph.is_valid_adjustment_set(subset, treatment=x, outcome=y)
            for size in range(len(candidates) + 1)
            for subset in itertools.combinations(candidates, size)
        )
        assert bool(result) == exhaustive, (edges, latent, x, y, result)
        if result:
            assert graph.is_valid_adjustment_set(result.covariates, treatment=x, outcome=y)
            identified += 1
        else:
            unidentified += 1
    assert identified > 0  # both branches of the completeness claim were exercised
    assert unidentified > 0
