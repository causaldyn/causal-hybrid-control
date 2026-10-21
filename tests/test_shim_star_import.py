"""A star-import from an old path of the research periphery imports its moved names, warning.

Each old path served its moved names through a module ``__getattr__`` alone, which a star-import
never calls without an ``__all__``: ``from chc.koopman import *`` imported nothing and warned
nothing, and ``dir()`` hid every moved name. Each old path now lists them.
"""

from __future__ import annotations

import ast
import importlib
import warnings
from pathlib import Path

import pytest

import chc
from chc import _moved

SRC = Path(chc.__file__).resolve().parent
SPLIT = {
    "chc.regret": "chc.research.regret",
    "chc.games": "chc.research.games",
    "chc.residual": "chc.research.residual",
}


def _defined(module: str) -> set[str]:
    """A module's ``__all__``, or else the public names its own top level defines."""
    path = SRC.joinpath(*module.split(".")[1:]).with_suffix(".py")
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets == {"__all__"} and module.startswith("chc.research."):
                return set(ast.literal_eval(node.value))
            names |= targets
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")}


def _moved_names(old: str) -> set[str]:
    if old in _moved.MODULES:
        return _defined(_moved.MODULES[old])
    return set(importlib.import_module(old)._MOVED)


OLD_PATHS = sorted([*_moved.MODULES, *SPLIT])


@pytest.mark.parametrize("old", OLD_PATHS)
def test_an_old_path_lists_every_name_it_serves(old: str) -> None:
    module = importlib.import_module(old)
    moved = _moved_names(old)
    if old in _moved.MODULES:
        assert set(module.__all__) == moved
    else:
        assert set(module.__all__) == _defined(old) | moved
    assert len(module.__all__) == len(set(module.__all__))
    assert moved <= set(dir(module))


@pytest.mark.parametrize("old", OLD_PATHS)
def test_a_star_import_from_an_old_path_imports_the_moved_names_and_warns(old: str) -> None:
    moved = _moved_names(old)
    new = _moved.MODULES.get(old) or SPLIT[old]
    namespace: dict[str, object] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec(f"from {old} import *", namespace)
    target = importlib.import_module(new)
    for name in moved:
        assert namespace[name] is getattr(target, name)
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert sorted(messages) == sorted(
        f"{old}.{name} leaves {old} in 0.20.0: import it from {new}" for name in moved
    )
