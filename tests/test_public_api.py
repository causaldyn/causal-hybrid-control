"""Invariants of the top-level namespace itself, not of any one routine in it.

Both of these are cheap and both have already caught something. The shadowing check is the reason
the prescribe facade lives in ``chc.decision``: a module named after the function it exports is
overwritten by that function during ``chc/__init__``, so ``import chc.prescribe`` then
``chc.prescribe.Lever`` raises ``AttributeError`` -- a failure with no symptom until a user writes
the most natural line there is.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
import typing
import warnings
from pathlib import Path

import pytest

import chc


def test_no_export_shadows_a_submodule() -> None:
    modules = {found.name for found in pkgutil.iter_modules(chc.__path__)}
    assert sorted(modules & set(chc.__all__)) == []


def test_every_advertised_symbol_exists() -> None:
    missing = [name for name in chc.__all__ if not hasattr(chc, name)]
    assert missing == []


def test_the_namespace_advertises_no_duplicates() -> None:
    assert len(chc.__all__) == len(set(chc.__all__))


def test_the_advertised_version_is_the_installed_one() -> None:
    """`chc.__version__` was a literal, and it sat at 0.3.0 through two releases.

    A second copy of `pyproject.toml`'s `version` drifts the moment a release forgets it, and the
    thing that reads it is `Provenance` -- so a stale literal names a version that did not produce
    the numbers beside it. Pinned against package metadata, which is what `uv build` writes.
    """
    from importlib.metadata import version

    assert chc.__version__ == version("causal-hybrid-control")
    assert chc.__version__ != "unknown"  # the installed-from-a-source-tree fallback, not this


LIFECYCLE_PAGE = Path(__file__).resolve().parent.parent / "docs" / "lifecycle.md"


def _lifecycle_names() -> list[str]:
    """Every name `docs/lifecycle.md` files that some module of the library defines, in page
    order; the page's other code spans are arguments and symbols."""
    defined = {
        name
        for found in pkgutil.iter_modules(chc.__path__)
        for name in vars(importlib.import_module(f"chc.{found.name}"))
        if not name.startswith("_")
    }
    names = [
        ticked.split(".")[0] for ticked in re.findall(r"`(\w[\w.]*)`", LIFECYCLE_PAGE.read_text())
    ]
    return list(dict.fromkeys(name for name in names if name in defined))


def _chc_classes(annotation: object) -> set[type]:
    """The classes the library defines that an annotation names, protocols aside: a caller
    satisfies a protocol with its own class, and needs nothing from `chc` to do it."""
    found: set[type] = set()
    for part in (annotation, typing.get_origin(annotation), *typing.get_args(annotation)):
        if isinstance(part, type) and part.__module__.startswith("chc."):
            if not getattr(part, "_is_protocol", False):
                found.add(part)
        elif part is not annotation and typing.get_args(part):
            found |= _chc_classes(part)
    return found


def test_the_top_level_holds_every_name_the_lifecycle_files() -> None:
    names = _lifecycle_names()
    assert len(names) >= 30  # the page was read, not an empty match
    assert [name for name in names if name not in chc.__all__] == []


def test_the_top_level_holds_what_a_lifecycle_call_takes() -> None:
    """A caller of a name on the page must not have to leave the top level to build its
    arguments; what a call returns is read, not built, so it lives at its module path."""
    entries = [getattr(chc, name) for name in _lifecycle_names()]
    functions = [entry for entry in entries if inspect.isfunction(entry)]
    returned = set().union(
        *(_chc_classes(typing.get_type_hints(f).get("return")) for f in functions)
    )
    built = [
        entry
        for entry in entries
        if isinstance(entry, type)
        and entry not in returned
        and not issubclass(entry, BaseException)
    ]
    callables = functions + [cls.__init__ for cls in built]
    callables += [
        method
        for cls in built
        for name, method in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    ]
    taken = set()
    for target in callables:
        hints = typing.get_type_hints(target)
        hints.pop("return", None)
        for annotation in hints.values():
            taken |= _chc_classes(annotation)
    assert chc.ZoneBatch in taken  # a method's argument counts: the gate is fed batches
    assert sorted(cls.__name__ for cls in taken if cls.__name__ not in chc.__all__) == []


def test_a_name_that_left_the_top_level_still_resolves_with_a_warning_naming_its_module() -> None:
    """0.7.0 bound 459 names here: `__version__`, 37 the lifecycle files, and these 421."""
    assert len(chc._MOVED) == 421
    for name, module in chc._MOVED.items():
        with pytest.warns(
            DeprecationWarning, match=rf"^chc\.{name} leaves the top level in 1\.0: "
        ):
            found = getattr(chc, name)
        assert found is getattr(importlib.import_module(module), name), name


def test_the_warning_points_at_the_line_that_imported_the_name() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from chc import ConfoundedLinearSystem  # noqa: F401
    assert caught
    assert {(w.category, w.filename) for w in caught} == {(DeprecationWarning, __file__)}
    assert all("import it from chc.causal" in str(w.message) for w in caught)


def test_a_name_that_left_the_top_level_is_neither_bound_nor_advertised() -> None:
    moved = set(chc._MOVED)
    assert sorted(moved & set(vars(chc))) == []
    assert sorted(moved & set(chc.__all__)) == []


def test_a_name_the_library_never_had_is_still_an_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'no_such_name'"):
        _ = chc.no_such_name
    with pytest.raises(ImportError):
        from chc import no_such_name  # noqa: F401


def test_type_checkers_see_every_name_that_left_the_top_level() -> None:
    """The names bound under `TYPE_CHECKING` are what a checked caller's imports resolve to, so
    they must be exactly the ones served at run time, each from the same module."""
    tree = ast.parse(Path(chc.__file__).read_text())
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    )
    bound = {
        alias.name: statement.module
        for statement in block.body
        if isinstance(statement, ast.ImportFrom)
        for alias in statement.names
        if alias.asname == alias.name
    }
    assert bound == chc._MOVED
