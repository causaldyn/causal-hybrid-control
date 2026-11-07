"""The tiers ``docs/api/index.md`` promises, checked on the library's syntax tree.

The page sorts every module of ``chc`` into four tiers and lists the experimental entries that sit
in modules of other tiers. The stable tier follows Semantic Versioning, and a stable module hands
on to its callers what every module it reads does: one that imports an experimental name, or an
underscore name another tier keeps for itself, moves when that module moves. So each such import
is listed here, in :data:`CROSSINGS`, with the entry it serves. An experimental module's name may
serve only one of the page's experimental entries, the one whose result it moves; an underscore
name of an evolving module, or any name of a module private to the package, names the stable entry
a change of it moves.

An import the list does not hold fails, and so does a listed one the source no longer makes.
Nothing outside ``chc.research`` imports it (ADR 0117). The page's table and its entry list are
parsed, and every module and entry they name resolves; the README's counts of the stable tier are
the library's.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

import chc

ROOT = Path(__file__).resolve().parent.parent
# The package under test: the source tree in a checkout, site-packages where a wheel is tested.
LIBRARY = Path(chc.__file__).resolve().parent
INDEX = ROOT / "docs" / "api" / "index.md"

# (stable module, the module it imports from, the name) -> the entry the name serves, written as
# the page writes one: `module: entry`. Where the source module is experimental, that entry must be
# one of the page's experimental entries.
CROSSINGS: dict[tuple[str, str, str], str] = {
    ("decision", "_robust_cov", "robust_cov"): "decision: prescribe",
    ("decision", "causal", "_polynomial_features"): "decision: prescribe",
    ("decision", "evaluation", "_evaluate_across_time"): "decision: Prescription.evaluate",
    ("decision", "evaluation", "_evaluate_by_unit"): "decision: Prescription.evaluate",
    ("decision", "evaluation", "_logger_check_json"): "decision: Prescription.logger_check",
    ("decision", "evaluation", "_policy_json"): "decision: Prescription.evaluate",
    ("decision", "frames", "_not_numbers"): "decision: prescribe",
    ("decision", "frames", "_real_numbers"): "decision: prescribe",
    ("decision", "misspecification", "MisspecificationCost"): "decision: DitherCheck",
    ("decision", "misspecification", "_contrast"): "decision: prescribe(dither=)",
    (
        "decision",
        "misspecification",
        "planned_cost_bias",
    ): "decision: prescribe(planned_cost_bias=)",
    ("decision", "uncertainty", "_TUBE_COVERAGE"): "decision: prescribe",
    ("decision", "uncertainty", "_chi_square_mixture_survival"): (
        "decision: DecisionCertificate.tube_per_log_spread"
    ),
    ("decision", "uncertainty", "_linear_tube"): "decision: prescribe",
    ("decision", "uncertainty", "_noise_covariance"): "decision: prescribe",
    ("decision", "uncertainty", "_noise_covariances"): (
        "decision: DecisionCertificate.tube_per_log_spread"
    ),
    ("decision", "uncertainty", "_noise_tube"): "decision: prescribe",
    ("decision", "uncertainty", "_NoiseRead"): "decision: prescribe",
    ("decision", "uncertainty", "_read_covariances"): "decision: prescribe",
    ("decision", "uncertainty", "_step_parts"): "decision: DecisionCertificate.tube_per_log_spread",
    (
        "decision",
        "uncertainty",
        "_studentised",
    ): "decision: DecisionCertificate.tube_per_log_spread",
    ("dynamics_id", "_robust_cov", "_cluster_count"): "dynamics_id: fit_causal_residual",
    ("dynamics_id", "_robust_cov", "_cluster_factor"): "dynamics_id: persistence_check",
    ("dynamics_id", "_robust_cov", "_clustered_squares"): "dynamics_id: omitted_confounder_bound",
    ("dynamics_id", "_robust_cov", "_cosines"): "dynamics_id: persistence_check",
    ("dynamics_id", "_robust_cov", "_degrees_of_freedom"): "dynamics_id: fit_causal_residual",
    ("dynamics_id", "_robust_cov", "_period_cosines"): "dynamics_id: fit_causal_residual",
    ("dynamics_id", "_robust_cov", "_small_sample"): "dynamics_id: fit_causal_residual",
    ("dynamics_id", "_robust_cov", "robust_cov"): "dynamics_id: fit_causal_residual",
    ("dynamics_id", "_robust_cov", "sandwich"): "dynamics_id: lag_check",
    ("dynamics_id", "causal", "_ACTION_AND_OUTCOME"): "dynamics_id: fit_causal_residual",
    ("dynamics_id", "causal", "_TRANSITION"): "dynamics_id: fit_causal_residual",
    ("dynamics_id", "causal", "_polynomial_features"): "dynamics_id: fit_causal_residual",
    ("dynamics_id", "causal", "_refuse_reread"): "dynamics_id: fit_causal_residual",
    ("dynamics_id", "causal", "_ridge_predict"): "dynamics_id: fit_causal_residual",
    ("dynamics_id", "causal", "_stream_key"): "dynamics_id: fit_causal_residual",
    ("dynamics_id", "frames", "_at"): "dynamics_id: fit_causal_residual",
    ("dynamics_id", "frames", "_real_entries"): "dynamics_id: fit_causal_residual",
    ("dynamics_id", "frames", "_real_numbers"): "dynamics_id: fit_causal_residual",
    ("dynamics_id", "frames", "_unmasked"): "dynamics_id: fit_causal_residual",
    ("lqr", "_action_units", "action_units"): "lqr: finite_horizon_dlqr",
    ("panel", "frames", "_NOT_READ"): "panel: Panel",
    ("panel", "frames", "_not_numbers"): "panel: Panel",
    ("plan", "causal", "_polynomial_features"): "plan: causal_plan",
    ("plan", "response", "relax"): "plan: CausalPlan.relaxed_cost",
    ("plan", "support", "_augmented"): "plan: causal_plan",
    ("plan", "support", "_penalised_gradient"): "plan: causal_plan",
    ("plan", "support", "_pessimistic_loop"): "plan: causal_plan",
    ("plan", "support", "_radius_norms"): "plan: causal_plan",
    ("plan", "support", "_smooth"): "plan: causal_plan",
    ("plan", "uncertainty", "_check_tolerance"): "plan: causal_plan",
    ("plan", "uncertainty", "_linear_tube"): "plan: causal_plan",
    ("plan", "uncertainty", "_noise_covariance"): "plan: causal_plan",
    ("plan", "uncertainty", "_noise_persistence"): "plan: causal_plan",
    ("plan", "uncertainty", "_noise_tube"): "plan: causal_plan",
    ("plan", "uncertainty", "_NoiseRead"): "plan: causal_plan",
    ("plan", "uncertainty", "_tube"): "plan: causal_plan",
    ("plan", "uncertainty", "_within"): "plan: causal_plan",
    ("residual", "frames", "_real_numbers"): "residual: fit_spectral_residual",
    ("train", "frames", "_real_entries"): "train: fit_residual",
}

# `name`, `Class.member`, `function(argument=)`, `Class(argument="value")`, `Class.member["key"]`
ENTRY = re.compile(
    r'(?P<name>\w+)(?:\.(?P<member>\w+))?(?:\((?P<argument>\w+)=(?P<value>"[^"]*")?\))?'
    r'(?:\[(?P<key>"[^"]*")\])?'
)


def _table() -> dict[str, list[str]]:
    """Each tier's row of the page's table: the module names in backticks in its middle column."""
    rows = {}
    for line in INDEX.read_text().splitlines():
        found = re.match(r"\| \*\*(\w+)\*\* \| (.+?) \| [^|]+ \|$", line)
        if found:
            names = re.findall(r"`([\w.]+)`", found[2])
            rows[found[1]] = [name for name in names if name != "chc.research"]
    return rows


def _entries() -> dict[str, list[str]]:
    """The page's experimental entries in modules of other tiers, by module."""
    text = INDEX.read_text()
    block = text[text.index("entries in modules of other tiers are experimental") :]
    block = block[: block.index("\n## ")]
    return {
        found[1]: re.findall(r"`([^`]+)`", found[2])
        for found in re.finditer(
            r"^- `chc\.(\w+)`: (.+?)(?=^- `chc\.|\Z)", block, re.MULTILINE | re.DOTALL
        )
    }


def _tier(module: str, table: dict[str, list[str]]) -> str:
    """The tier of ``module``, a dotted name below ``chc``: the page names the stable and the
    experimental modules one by one and every other one is evolving; an underscore module is
    private to the package."""
    head = module.split(".")[0]
    if head == "research":
        return "research"
    if head.startswith("_"):
        return "private"
    if head in table["stable"]:
        return "stable"
    return "experimental" if head in table["experimental"] else "evolving"


def _module_name(path: Path) -> str:
    """``path``'s dotted name below ``chc``: ``plan`` for ``chc/plan.py``, ``research`` for
    ``chc/research/__init__.py``."""
    parts = path.relative_to(LIBRARY).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(module: str, source: str, package: bool = False) -> list[tuple[str, str]]:
    """``(imported module, name)`` for every import of ``chc`` in ``source``, at any depth: a
    module imported whole as ``import chc.x`` gives the name ``*``, and a relative import is read
    from ``module``'s package, ``module`` itself where it is one."""
    found = []
    here = ["chc", *filter(None, module.split("."))][: None if package else -1]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [
                (a.name.removeprefix("chc."), "*") for a in node.names if a.name[:4] == "chc."
            ]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = here[: len(here) - node.level + 1]
                target = ".".join([*base, *([node.module] if node.module else [])])
            else:
                target = node.module or ""
            if target.startswith("chc."):
                found += [(target.removeprefix("chc."), a.name) for a in node.names]
    return found


def _crossings(module: str, source: str, table: dict[str, list[str]]) -> set[tuple[str, str, str]]:
    """The imports of a stable ``module`` that cross a tier: any name of an experimental module,
    and any underscore name, or name of a private module, that another tier holds."""
    crossings = set()
    for target, name in _imports(module, source):
        tier = _tier(target, table)
        if tier in ("stable", "research"):
            continue
        if tier == "experimental" or tier == "private" or name.startswith("_"):
            crossings.add((module, target, name))
    return crossings


def _sources() -> dict[str, str]:
    return {_module_name(path): path.read_text() for path in sorted(LIBRARY.rglob("*.py"))}


def _packages() -> set[str]:
    return {_module_name(path) for path in LIBRARY.rglob("__init__.py")}


def test_the_tier_table_names_modules_that_exist_each_in_one_tier() -> None:
    table = _table()
    assert set(table) == {"stable", "evolving", "experimental", "research"}
    modules = set(_sources())
    for tier in ("stable", "evolving", "experimental"):
        missing = [name for name in table[tier] if name not in modules]
        assert not missing, (tier, missing)
    missing = [name for name in table["research"] if f"research.{name}" not in modules]
    assert not missing, ("research", missing)
    named = table["stable"] + table["evolving"] + table["experimental"]
    assert len(named) == len(set(named)), named


def test_every_experimental_entry_the_page_lists_resolves() -> None:
    """Each entry names an object of its module: a member its class has, an argument its
    function takes, and a value or key its module's source spells. The count the page states is
    the count it lists."""
    entries = _entries()
    stated = re.search(r"(\d+) entries in modules of other tiers", INDEX.read_text())
    assert stated is not None
    assert sum(len(names) for names in entries.values()) == int(stated[1])
    unresolved = [
        f"chc.{module}: {entry}"
        for module, names in entries.items()
        for entry in names
        if not _resolves(module, entry)
    ]
    assert not unresolved, unresolved


def _resolves(module: str, entry: str) -> bool:
    parsed = ENTRY.fullmatch(entry)
    if parsed is None:
        return False
    owner = importlib.import_module(f"chc.{module}")
    target = getattr(owner, parsed["name"], None)
    if target is None:
        return False
    if parsed["member"]:
        member = parsed["member"]
        fields = getattr(target, "__dataclass_fields__", {})
        if not (hasattr(target, member) or member in fields):
            return False
        target = getattr(target, member, None)
    if parsed["argument"] and (
        target is None or parsed["argument"] not in inspect.signature(target).parameters
    ):
        return False
    source = inspect.getsource(owner)
    return all(text is None or text in source for text in (parsed["value"], parsed["key"]))


def test_every_crossing_of_a_tier_is_listed_with_the_entry_it_serves() -> None:
    table = _table()
    entries = _entries()
    made = set()
    for module, source in _sources().items():
        if _tier(module, table) == "stable":
            made |= _crossings(module, source, table)
    assert sorted(made - set(CROSSINGS)) == [], "a stable module crosses a tier unlisted"
    assert sorted(set(CROSSINGS) - made) == [], "a listed crossing the source no longer makes"
    for (module, target, name), served in CROSSINGS.items():
        owner, _, entry = served.partition(": ")
        assert owner == module, (module, target, name, served)
        assert _resolves(owner, entry), served
        if _tier(target, table) == "experimental":
            assert entry in entries.get(owner, []), (
                f"chc.{module} reads {name} from chc.{target}, which only an experimental entry "
                f"may, and {entry} is not one"
            )


def test_a_new_crossing_is_seen_at_any_depth_and_by_any_form_of_import() -> None:
    table = _table()
    source = (
        "import chc.dlm\n"
        "from chc.causal import fit_backdoor, _ridge_predict\n"
        "def f():\n"
        "    from chc.lift import LiftFit\n"
        "    from ._action_units import action_units\n"
        "    from .uncertainty import _tube\n"
        "    from chc.plan import _plan\n"
        "    from ..research.spine import spine\n"
    )
    assert _crossings("plan", source, table) == {
        ("plan", "dlm", "*"),
        ("plan", "causal", "_ridge_predict"),
        ("plan", "lift", "LiftFit"),
        ("plan", "_action_units", "action_units"),
        ("plan", "uncertainty", "_tube"),
    }


def test_nothing_outside_chc_research_imports_it() -> None:
    """``import chc`` loads no research module (ADR 0117), and no other module may either."""
    table = _table()
    packages = _packages()
    reaching = sorted(
        (module, target)
        for module, source in _sources().items()
        if _tier(module, table) != "research"
        for target, _ in _imports(module, source, module in packages)
        if _tier(target, table) == "research"
    )
    assert reaching == []


def _exported(path: Path) -> list[str]:
    """The names in the ``__all__`` a module assigns, read from its syntax tree."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            assert isinstance(node.value, ast.List | ast.Tuple), path
            return [ast.literal_eval(element) for element in node.value.elts]
    return []


def test_the_readme_states_the_stable_tier_s_size_as_the_library_has_it() -> None:
    """The README's first screen says that stable means the stable tier, with its counts: the
    names the stable modules' ``__all__`` lists, of every module's, and the stable modules."""
    table = _table()
    paths = sorted(LIBRARY.rglob("*.py"))
    stable = sum(len(_exported(p)) for p in paths if _module_name(p) in table["stable"])
    every = sum(len(_exported(p)) for p in paths)
    text = (ROOT / "README.md").read_text()
    first = text[: text.index("\n## ")]
    stated = re.search(r"(\d+) of the (\d+) names .+? in (\d+) modules", first, re.DOTALL)
    assert stated is not None, "the README's first screen does not state the stable tier's size"
    assert tuple(map(int, stated.groups())) == (stable, every, len(table["stable"]))
