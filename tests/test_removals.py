"""The list of what leaves at 1.0 is the library's own: nothing on it is gone, nothing off it warns.

Every deprecated name says that it leaves in 1.0, and 1.0 removes them together from
``chc._removals``. A list that misses one name leaves a warning that promised a removal behind;
a list that holds a name the library no longer serves sends 1.0 looking for code that is not
there. These tests read both ways: each entry still serves its value and names its replacement,
and each deprecation warning in the source, each re-export of a listed function and each record
key that repeats a renamed value under its old name is listed.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import re
import textwrap
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import chc
from chc import _removals, decision
from chc._removals import REMOVALS, Removal

SRC = Path(chc.__file__).resolve().parent
DEPRECATIONS = {"DeprecationWarning", "PendingDeprecationWarning", "FutureWarning"}
OTHER_WARNINGS = {
    name
    for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, Warning) and name not in DEPRECATIONS
}


def _of(kind: str) -> list[Removal]:
    return [entry for entry in REMOVALS if entry.kind == kind]


def _module_of(path: Path) -> str:
    parts = path.relative_to(SRC.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _sources() -> Iterator[tuple[str, ast.Module]]:
    for path in sorted(SRC.rglob("*.py")):
        yield _module_of(path), ast.parse(path.read_text(), filename=str(path))


def _scoped(module: str, tree: ast.AST) -> Iterator[tuple[str, ast.AST]]:
    """Every node with the qualified name of the function or class it sits in, after the module's
    own name; a node at the module's top level sits in the module."""

    def walk(node: ast.AST, scope: str) -> Iterator[tuple[str, ast.AST]]:
        for child in ast.iter_child_nodes(node):
            yield scope, child
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield from walk(child, f"{scope}.{child.name}")
            else:
                yield from walk(child, scope)

    return walk(tree, module)


def _category(call: ast.Call) -> str:
    """The warning class a ``warn`` call names: ``UserWarning`` where it names none."""
    given = call.args[1] if len(call.args) > 1 else None
    for keyword in call.keywords:
        if keyword.arg == "category":
            given = keyword.value
    if given is None:
        return "UserWarning"
    assert isinstance(given, ast.Name), f"a warning's class must be named: {ast.unparse(call)}"
    return given.id


def _deprecation_sites() -> set[str]:
    """The qualified name of every function in the source that raises a deprecation warning, or
    that a ``deprecated`` decorator marks."""
    sites = set()
    for module, tree in _sources():
        for scope, node in _scoped(module, tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    target = decorator.func if isinstance(decorator, ast.Call) else decorator
                    if ast.unparse(target).split(".")[-1] == "deprecated":
                        sites.add(f"{scope}.{node.name}")
            if not isinstance(node, ast.Call):
                continue
            called = ast.unparse(node.func)
            if called.split(".")[-1] != "warn" or not called.startswith(("warn", "_warn")):
                continue
            category = _category(node)
            assert category in DEPRECATIONS | OTHER_WARNINGS, (
                f"{scope}: a warning of class {category}, which this test cannot read"
            )
            if category in DEPRECATIONS:
                sites.add(scope)
    return sites


def _resolve(path: str) -> Any:
    """The object a dotted path names, read without its property's getter."""
    parts = path.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            found: Any = importlib.import_module(".".join(parts[:cut]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[cut:]:
            found = inspect.getattr_static(found, attribute)
        return found
    raise LookupError(path)


def _owner(path: str) -> tuple[str, str]:
    """The dotted path of what holds a name, and the name."""
    owner, _, name = path.rpartition(".")
    return owner, name


def _renamed(entry: Removal) -> str:
    """The start of a renamed name's warning: its old name after its module, and its new one."""
    old_owner = _owner(entry.old)[0]
    module = old_owner if entry.kind == "function" else _owner(old_owner)[0]
    return (
        rf"^{re.escape(entry.old.removeprefix(module + '.'))} is renamed "
        rf"{re.escape(_owner(entry.new)[1])} and leaves in 1\.0: \S"
    )


def test_the_list_holds_each_kind_and_no_entry_twice() -> None:
    assert len({entry.old for entry in REMOVALS}) == len(REMOVALS)
    assert {entry.kind for entry in REMOVALS} == {
        "top-level name",
        "function",
        "attribute",
        "re-export",
        "record key",
    }
    assert all(re.fullmatch(r"0\.\d+\.\d+", entry.since) for entry in REMOVALS)
    adrs = {path.name[:4] for path in (SRC.parent.parent / "docs" / "adr").glob("[0-9]*.md")}
    assert sorted({adr for entry in REMOVALS for adr in entry.adrs} - adrs) == []


def test_each_top_level_name_still_imports_with_a_warning_naming_its_module() -> None:
    (entry,) = _of("top-level name")
    assert entry.site == "chc.__getattr__"
    assert len(chc._MOVED) == _removals.TOP_LEVEL_NAMES
    for name, module in chc._MOVED.items():
        with pytest.warns(
            DeprecationWarning,
            match=rf"^chc\.{name} leaves the top level in 1\.0: import it from "
            rf"{re.escape(module)}$",
        ):
            found = getattr(chc, name)
        assert found is getattr(importlib.import_module(module), name), name


@pytest.mark.parametrize("entry", _of("function"), ids=lambda entry: entry.old)
def test_each_renamed_function_still_calls_its_new_name_with_a_warning_naming_it(
    entry: Removal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old function is called with a placeholder for each argument it requires, and its new
    name is replaced by a stub: what the old one returns must be what the new one does."""
    assert entry.site == entry.old
    old, (module, new) = _resolve(entry.old), _owner(entry.new)
    assert _owner(entry.old)[0] == module
    assert callable(_resolve(entry.new))
    returned = object()
    monkeypatch.setattr(importlib.import_module(module), new, lambda *_, **__: returned)
    required = [
        parameter.name
        for parameter in inspect.signature(old).parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ]
    with pytest.warns(DeprecationWarning, match=_renamed(entry)):
        assert old(**dict.fromkeys(required)) is returned


@pytest.mark.parametrize("entry", _of("attribute"), ids=lambda entry: entry.old)
def test_each_renamed_attribute_still_reads_its_new_name_with_a_warning_naming_it(
    entry: Removal,
) -> None:
    """The old property is read on a stand-in that holds the new name alone."""
    assert entry.site == entry.old
    old, (owner, new) = _resolve(entry.old), _owner(entry.new)
    assert owner == _owner(entry.old)[0]
    assert isinstance(old, property)
    assert new in inspect.signature(_resolve(owner)).parameters
    held = object()
    with pytest.warns(DeprecationWarning, match=_renamed(entry)):
        assert old.fget(SimpleNamespace(**{new: held})) is held


@pytest.mark.parametrize("entry", _of("re-export"), ids=lambda entry: entry.old)
def test_each_re_export_is_the_listed_function_and_sits_beside_its_new_name(
    entry: Removal,
) -> None:
    (function,) = [listed for listed in _of("function") if listed.site == entry.site]
    assert _resolve(entry.old) is _resolve(function.old)
    assert _resolve(entry.new) is _resolve(function.new)
    owner, name = _owner(entry.old)
    assert name not in importlib.import_module(owner).__all__


def test_the_record_still_writes_each_listed_key_beside_its_new_name() -> None:
    (entry,) = _of("record key")
    assert entry.site == "chc.decision.Prescription.to_json"
    assert decision.SCHEMA_VERSION == _removals.SCHEMA_VERSION_AT_1_0 - 1
    source = textwrap.dedent(inspect.getsource(decision.Prescription.to_json))
    keys = {
        ast.literal_eval(key): ast.unparse(value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant)
    }
    old, new = _owner(entry.old)[1], _owner(entry.new)[1]
    assert keys[old] == keys[new] == f"certificate.{new}"


def test_every_deprecation_warning_the_source_raises_is_listed() -> None:
    """A new deprecation that is not listed fails here, and so does an entry whose warning has
    gone: the warning sites and the list's are the same set."""
    listed = {entry.site for entry in REMOVALS if entry.kind not in ("re-export", "record key")}
    assert _deprecation_sites() == listed


def test_every_re_export_of_a_listed_function_is_listed() -> None:
    """An old function imported into another module outside the top level, which ``chc._MOVED``
    serves, leaves at 1.0 with it."""
    functions = {_owner(entry.old) for entry in _of("function")}
    found = {
        f"{module}.{alias.asname or alias.name}"
        for module, tree in _sources()
        if module != "chc"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if (node.module, alias.name) in functions
    }
    assert found == {entry.old for entry in _of("re-export")}


def test_every_record_key_that_repeats_a_renamed_value_under_its_old_name_is_listed() -> None:
    """A key named after a renamed attribute's old name, holding the attribute's new one, is the
    old name kept for a reader of the record, and leaves with it."""
    renamed = {(_owner(entry.old)[1], _owner(entry.new)[1]) for entry in _of("attribute")}
    found = set()
    for module, tree in _sources():
        for scope, node in _scoped(module, tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(value, ast.Attribute)
                    and (key.value, value.attr) in renamed
                ):
                    found.add((scope, key.value))
    listed = {(entry.site, _owner(entry.old)[1]) for entry in _of("record key")}
    assert found == listed
