"""The research periphery lives in `chc.research`, and every path it left still serves it, warning.

0.19.0 moved sixteen modules into `chc.research` whole, and the research halves of `chc.regret`,
`chc.games` and `chc.residual` beside them. Each old path imports without a warning, so that
`import chc` stays silent for a caller that turns warnings into errors, and serves each moved name
with a `DeprecationWarning` naming the new path, until 0.20.0 deletes it. These tests hold the
boundary itself: nothing outside `chc.research` imports from it, so the lifecycle loads none of it.
"""

from __future__ import annotations

import ast
import importlib
import os
import pkgutil
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import chc
import chc.research
from chc import _moved

SRC = Path(chc.__file__).resolve().parent
SPLIT = {
    "chc.regret": "chc.research.regret",
    "chc.games": "chc.research.games",
    "chc.residual": "chc.research.residual",
}


def _defined(module: str) -> set[str]:
    """The public names a module's own top level defines, imports aside."""
    path = SRC.joinpath(*module.split(".")[1:]).with_suffix(".py")
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")}


def _served_names(new: str) -> list[str]:
    """A moved module's public names: its ``__all__``, or the names its own top level defines."""
    return sorted(getattr(importlib.import_module(new), "__all__", None) or _defined(new))


def test_the_research_package_holds_the_modules_that_moved_whole_and_the_three_halves() -> None:
    found = {f"chc.research.{m.name}" for m in pkgutil.iter_modules(chc.research.__path__)}
    assert found == set(_moved.MODULES.values()) | set(SPLIT.values())
    assert len(_moved.MODULES) == 16


@pytest.mark.parametrize("old", sorted(_moved.MODULES))
def test_a_module_that_moved_whole_serves_each_name_from_its_new_path_with_a_warning(
    old: str,
) -> None:
    new = _moved.MODULES[old]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shim = importlib.import_module(old)
    names = _served_names(new)
    assert _defined(new) <= set(names)
    for name in names:
        with pytest.warns(
            DeprecationWarning,
            match=rf"^{re.escape(old)}\.{name} leaves {re.escape(old)} in 0\.20\.0: "
            rf"import it from {re.escape(new)}$",
        ):
            found = getattr(shim, name)
        assert found is getattr(importlib.import_module(new), name), name


@pytest.mark.parametrize("old", sorted(SPLIT))
def test_a_split_module_serves_exactly_its_moved_half_with_a_warning(old: str) -> None:
    new = SPLIT[old]
    module = importlib.import_module(old)
    assert set(module._MOVED) == _defined(new)
    assert not set(module._MOVED) & set(vars(module))
    for name in sorted(module._MOVED):
        with pytest.warns(
            DeprecationWarning,
            match=rf"^{old}\.{name} leaves {old} in 0\.20\.0: import it from {new}$",
        ):
            found = getattr(module, name)
        assert found is getattr(importlib.import_module(new), name), name


@pytest.mark.parametrize("old", sorted(SPLIT))
def test_type_checkers_see_each_moved_half_at_its_old_path(old: str) -> None:
    tree = ast.parse(SRC.joinpath(old.split(".")[1]).with_suffix(".py").read_text())
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    )
    bound = {
        (statement.module, alias.name)
        for statement in block.body
        if isinstance(statement, ast.ImportFrom)
        for alias in statement.names
        if alias.asname == alias.name
    }
    assert bound == {(SPLIT[old], name) for name in importlib.import_module(old)._MOVED}


@pytest.mark.parametrize("old", sorted(_moved.MODULES))
def test_type_checkers_see_a_whole_module_at_its_old_path(old: str) -> None:
    tree = ast.parse(SRC.joinpath(old.split(".")[1]).with_suffix(".py").read_text())
    stars = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and [a.name for a in node.names] == ["*"]
    ]
    assert stars == [_moved.MODULES[old]]


def test_an_old_path_serves_no_private_and_no_unknown_name() -> None:
    with pytest.raises(AttributeError, match=r"module 'chc\.mmm' has no attribute 'no_such_name'"):
        _ = chc.mmm.no_such_name
    # what the new module imports is not a name that moved: it served `chc.mmm.np`, warning
    for imported in ("np", "jax", "dataclass", "annotations", "LinearDynamics"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(AttributeError, match=f"has no attribute '{imported}'"):
                getattr(chc.mmm, imported)
    with pytest.raises(AttributeError, match="has no attribute '_DEFAULT_ERRORS'"):
        _ = chc.regret._DEFAULT_ERRORS
    with pytest.raises(ImportError):
        from chc.regret import no_such_name  # noqa: F401


def test_the_warning_points_at_the_line_that_read_the_name() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from chc.mmm import run_marketing_mix  # noqa: F401
        from chc.regret import van_trees_certificate  # noqa: F401
    assert [(w.category, w.filename) for w in caught] == [(DeprecationWarning, __file__)] * 2


def test_the_top_level_serves_each_name_that_moved_from_its_chc_research_path() -> None:
    """Served from its old path, a moved name would warn twice and name a deprecated module."""
    research = {name: m for name, m in chc._MOVED.items() if m.startswith("chc.research.")}
    assert len(research) == 166
    assert [name for name, m in research.items() if name not in _defined(m)] == []
    assert [
        name
        for name, m in chc._MOVED.items()
        if m in SPLIT and name in importlib.import_module(m)._MOVED
    ] == []


@pytest.mark.parametrize("statement", ["import chc", "import chc.decision"])
def test_importing_the_library_loads_nothing_from_research_and_warns_nothing(
    statement: str,
) -> None:
    """A caller running with warnings as errors must be able to import the library, and the
    lifecycle must not load the research periphery: `lqr` used to need all of `chc.regret`."""
    probe = (
        f"{statement}; import sys; "
        "print(sorted(m for m in sys.modules if m.startswith('chc.research')))"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    out = subprocess.run(
        [sys.executable, "-W", "error", "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=600,
    )
    assert out.stdout.strip() == "[]"


def _runtime_imports(tree: ast.Module) -> list[str]:
    """Every module a file imports outside an `if TYPE_CHECKING:` block, functions included."""
    found: list[str] = []
    pending: list[ast.AST] = [tree]
    while pending:
        node = pending.pop()
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            pending += node.orelse
            continue
        if isinstance(node, ast.ImportFrom) and node.module:
            found += [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        pending += ast.iter_child_nodes(node)
    return found


def test_no_module_outside_chc_research_imports_from_it() -> None:
    crossing = {
        str(path.relative_to(SRC)): sorted(
            {m for m in _runtime_imports(ast.parse(path.read_text())) if "research" in m}
        )
        for path in sorted(SRC.glob("*.py"))
    }
    assert {path: found for path, found in crossing.items() if found} == {}


def test_the_kept_half_of_regret_is_numpy_and_scipy_only() -> None:
    imported = _runtime_imports(ast.parse((SRC / "regret.py").read_text()))
    assert not [m for m in imported if m.startswith(("chc.network_causal", "jax"))]
