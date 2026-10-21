"""The research periphery lives in `chc.research`, and the paths it left are gone.

0.19.0 moved sixteen modules into `chc.research` whole, and the research halves of `chc.regret`,
`chc.games` and `chc.residual` beside them. Their old paths served each moved name with a
`DeprecationWarning` through 0.19.x, and 0.20.0 deleted them: a whole module's old path no longer
imports, and a split module no longer has its moved half. These tests hold the boundary itself:
nothing outside `chc.research` imports from it, so the lifecycle loads none of it.
"""

from __future__ import annotations

import ast
import importlib
import os
import pkgutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import chc
import chc.research

SRC = Path(chc.__file__).resolve().parent
WHOLE = (
    "benchmark",
    "causal_bench",
    "deep_galerkin",
    "epidemic",
    "flagship",
    "galerkin",
    "koopman",
    "lalonde",
    "meanfield",
    "mintime",
    "mmm",
    "sensitivity",
    "spine",
    "surrogate",
    "symbolic",
    "transport",
)
SPLIT = ("regret", "games", "residual")


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


def test_the_research_package_holds_the_modules_that_moved_whole_and_the_three_halves() -> None:
    found = {m.name for m in pkgutil.iter_modules(chc.research.__path__)}
    assert found == {*WHOLE, *SPLIT}


@pytest.mark.parametrize("name", WHOLE)
def test_the_old_path_of_a_module_that_moved_whole_is_gone(name: str) -> None:
    assert not SRC.joinpath(f"{name}.py").exists()
    sys.modules.pop(f"chc.{name}", None)
    with pytest.raises(ModuleNotFoundError, match=rf"No module named 'chc\.{name}'"):
        importlib.import_module(f"chc.{name}")
    with pytest.raises(AttributeError, match=rf"module 'chc' has no attribute '{name}'"):
        getattr(chc, name)


@pytest.mark.parametrize("name", SPLIT)
def test_a_split_module_no_longer_has_its_moved_half(name: str) -> None:
    old = importlib.import_module(f"chc.{name}")
    moved = _defined(f"chc.research.{name}")
    assert moved
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert [n for n in sorted(moved) if hasattr(old, n)] == []
    assert not hasattr(old, "_MOVED")


def test_the_top_level_serves_each_name_that_moved_from_its_chc_research_path() -> None:
    """The top level's deprecated names stay until 1.0: each must still be served from a module
    that defines it, now that the paths the research modules left are gone."""
    research = {name: m for name, m in chc._MOVED.items() if m.startswith("chc.research.")}
    assert len(research) == 166
    assert [name for name, m in research.items() if name not in _defined(m)] == []
    assert [name for name, m in chc._MOVED.items() if name not in _defined(m)] == []


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
