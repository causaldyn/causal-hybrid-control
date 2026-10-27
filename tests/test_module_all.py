"""Every public module declares its surface in ``__all__``: its own names, and every name its API
page renders.

Without ``__all__`` a module's surface is everything it binds, imports included: ``from
chc.decision import *`` handed out ``CausalGraph``, ``Literal`` and ``np`` beside the module's own
names, and a type checker reading the library could not tell an export from an import. The
declared surface narrows only that; ``chc.decision.CausalGraph`` still resolves as an attribute.

The rules, read from each module's source:

- ``__all__`` resolves, holds no duplicate, and holds only names the module defines itself -- a
  ``def``, a ``class``, a constant or a type alias. A name it imports from elsewhere is in it only
  where the module is a documented re-export point, :data:`RE_EXPORTS`.
- It holds every name the module's API page renders. ``docs/api/<module>.md`` renders the module
  with mkdocstrings' defaults: each top-level function, class and attribute that is not private
  and has a docstring (a string literal right below an assignment is an attribute's docstring).
- Each name the top level still serves from another module, with a warning naming that module, is
  in that module's ``__all__``: the warning sends a caller to an export.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from types import ModuleType

import pytest

import chc
import chc.research

MODULES = sorted(
    [f"chc.{m.name}" for m in pkgutil.iter_modules(chc.__path__) if not m.name.startswith("_")]
    + [f"chc.research.{m.name}" for m in pkgutil.iter_modules(chc.research.__path__)]
)

# The modules whose ``__all__`` lists names they import, and why.
RE_EXPORTS = {
    "chc.research.sensitivity": "a facade gathering the confounding-robust tools that live in "
    "chc.barrier, chc.uncertainty and chc.research.regret; it defines nothing itself",
}


def _bindings(module: ModuleType) -> tuple[set[str], set[str], set[str]]:
    """The names a module's top level defines, the names it imports, and the defined names an API
    page renders (each public, with a docstring)."""
    defined: set[str] = set()
    imported: set[str] = set()
    rendered: set[str] = set()

    def visit(body: list[ast.stmt]) -> None:
        for index, node in enumerate(body):
            after = body[index + 1] if index + 1 < len(body) else None
            documented = (
                isinstance(after, ast.Expr)
                and isinstance(after.value, ast.Constant)
                and isinstance(after.value.value, str)
            )
            names: list[str] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
                documented = ast.get_docstring(node) is not None
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imported.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, (ast.If, ast.Try)):
                visit(node.body)
                visit(node.orelse)
                for handler in getattr(node, "handlers", ()):
                    visit(handler.body)
            defined.update(names)
            rendered.update(n for n in names if documented and not n.startswith("_"))

    visit(ast.parse(inspect.getsource(module)).body)
    return defined, imported, rendered


@pytest.mark.parametrize("name", MODULES)
def test_each_public_module_declares_its_own_names(name: str) -> None:
    module = importlib.import_module(name)
    declared = getattr(module, "__all__", None)
    assert isinstance(declared, list), f"{name} defines no __all__"
    assert len(declared) == len(set(declared))
    assert [item for item in declared if not hasattr(module, item)] == []
    defined, imported, _ = _bindings(module)
    allowed = defined | imported if name in RE_EXPORTS else defined - imported
    assert sorted(set(declared) - allowed) == []


@pytest.mark.parametrize("name", MODULES)
def test_each_public_module_declares_every_name_its_api_page_renders(name: str) -> None:
    module = importlib.import_module(name)
    _, _, rendered = _bindings(module)
    assert sorted(rendered - set(getattr(module, "__all__", ()))) == []


def test_the_re_export_points_are_public_modules_that_re_export() -> None:
    for name in RE_EXPORTS:
        module = importlib.import_module(name)
        _, imported, _ = _bindings(module)
        assert name in MODULES
        assert set(module.__all__) & imported


def test_each_moved_name_is_an_export_of_the_module_its_warning_names() -> None:
    absent = {
        name: owner
        for name, owner in chc._MOVED.items()
        if name not in importlib.import_module(owner).__all__
    }
    assert absent == {}
