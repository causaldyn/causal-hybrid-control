"""Every docstring that cites a proof or a derivation is tied to a test of the code it describes.

A docstring that names a file under ``proofs/`` or ``validation/``, or a Rocq lemma by name, is a
citation. Each ``tests/test_formal_*.py`` holds a ``FORMAL_LINKS`` literal that maps a cited
object (``chc.<module>.<qualname>``) to the files it cites and to the tests that check the cited
statement on the float code, or to a ``reason`` the statement cannot be checked there. This file
fails when a citation has no entry, when an entry lists other files than the docstring cites, when
a listed test does not exist, or when a cited file is missing.
"""

from __future__ import annotations

import ast
import fnmatch
import importlib
import inspect
import re
from pathlib import Path
from types import ModuleType

import chc

ROOT = Path(__file__).resolve().parents[1]
# The package under test: the source tree in a checkout, site-packages where a wheel is tested.
SRC = Path(chc.__file__).resolve().parent
CITED_FILE = re.compile(
    r"(?:proofs/(?:mathcomp/)?|validation/)([A-Za-z0-9_]+\.(?:v|mac|gp|smt2|m|py|input|R))"
    r"|``([A-Za-z0-9_]+\.(?:v|mac|gp|smt2))``"
)
TICKED = re.compile(r"``([A-Za-z0-9_*']+)``")
LEMMA = re.compile(
    r"^\s*(?:Lemma|Theorem|Corollary|Proposition|Fact|Remark)\s+([A-Za-z0-9_']+)", re.MULTILINE
)


def _lemmas() -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for proof in (ROOT / "proofs").rglob("*.v"):
        for name in LEMMA.findall(proof.read_text()):
            names.setdefault(name, set()).add(proof.name)
    return names


def _cited(doc: str, lemmas: dict[str, set[str]]) -> set[str]:
    files = {a or b for a, b in CITED_FILE.findall(doc)}
    if "Rocq" in doc or "proofs" in doc:
        for tick in TICKED.findall(doc):
            # ``u*`` is an optimum, ``msm_inflation_*`` a family of lemmas
            matched = fnmatch.filter(lemmas, tick) if "_" in tick.rstrip("*") else [tick]
            for name in matched:
                files |= lemmas.get(name, set())
    return files


def citations() -> dict[str, set[str]]:
    """``chc.<module>.<qualname>``, or ``chc.<module>`` for a module docstring, to its files."""
    lemmas = _lemmas()
    found: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        module = ".".join(("chc", *path.relative_to(SRC).with_suffix("").parts))
        tree = ast.parse(path.read_text())

        def visit(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = f"{prefix}.{child.name}"
                    files = _cited(ast.get_docstring(child) or "", lemmas)
                    if files:
                        found[name] = files
                    visit(child, name)

        files = _cited(ast.get_docstring(tree) or "", lemmas)
        if files:
            found[module] = files
        visit(tree, module)
    return found


def links() -> dict[str, tuple[Path, dict]]:
    """Every ``FORMAL_LINKS`` entry, with the test file that holds it."""
    out: dict[str, tuple[Path, dict]] = {}
    for path in sorted((ROOT / "tests").glob("test_formal_*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            else:
                continue
            if "FORMAL_LINKS" in names and node.value is not None:
                for target, entry in ast.literal_eval(node.value).items():
                    assert target not in out, f"{target} is linked twice"
                    out[target] = (path, entry)
    return out


def _functions(path: Path) -> dict[str, str]:
    """Every function of a test file, by name, to its source."""
    text = path.read_text()
    return {
        n.name: ast.get_source_segment(text, n) or ""
        for n in ast.walk(ast.parse(text))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _test_names(path: Path) -> set[str]:
    return {name for name in _functions(path) if name.startswith("test_")}


def _resolve(path: Path, test: str) -> tuple[Path, str]:
    """A linked test is a name in the linking file, or ``<file>.py::<name>`` under ``tests/``."""
    if "::" in test:
        file, name = test.split("::")
        return ROOT / "tests" / file, name
    return path, test


def _reach(path: Path, name: str) -> str:
    """The test's source and that of the file's private helpers it calls, three calls deep."""
    functions, seen, frontier, text = _functions(path), {name}, [name], ""
    for _ in range(4):
        bodies = [functions.get(f, "") for f in frontier]
        text += "".join(bodies)
        frontier = [
            f
            for f in functions
            if f.startswith("_")
            and f not in seen
            and any(re.search(rf"\b{f}\b", b) for b in bodies)
        ]
        seen.update(frontier)
    return text


def _aliases(path: Path, name: str) -> set[str]:
    """``name`` and whatever the test file imports it as."""
    found = {name}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            found |= {a.asname for a in node.names if a.name == name and a.asname}
    return found


def _module_of(target: str) -> ModuleType:
    """The longest importable prefix of ``chc.<module>.<qualname>``."""
    parts = target.split(".")
    for end in range(len(parts), 1, -1):
        try:
            return importlib.import_module(".".join(parts[:end]))
        except ImportError:
            continue
    raise AssertionError(f"{target}: no importable module")


def _reaches(path: Path, test: str, names: list[str]) -> bool:
    file, name = _resolve(path, test)
    body = _reach(file, name.split("[")[0])
    spelled = set().union(*(_aliases(file, n) for n in names))
    return any(re.search(rf"\b{re.escape(n)}\b", body) for n in spelled)


def _defined(module: ModuleType) -> list[str]:
    return [
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and getattr(value, "__module__", None) == module.__name__
    ]


def test_every_cited_file_exists() -> None:
    present = {p.name for p in (ROOT / "proofs").rglob("*.v")}
    present |= {p.name for p in (ROOT / "validation").iterdir()}
    missing = {
        target: sorted(files - present) for target, files in citations().items() if files - present
    }
    assert not missing, missing


def test_every_citation_has_a_link() -> None:
    cited, linked = citations(), links()
    unlinked = sorted(set(cited) - set(linked))
    assert not unlinked, f"cited without a linking test: {unlinked}"
    stale = sorted(set(linked) - set(cited))
    assert not stale, f"linked but no longer cites a proof or derivation: {stale}"
    moved = {
        target: (sorted(cited[target]), sorted(entry["cites"]))
        for target, (_, entry) in linked.items()
        if set(entry["cites"]) != cited[target]
    }
    assert not moved, f"the cited files changed; revisit the link: {moved}"


def test_every_link_names_tests_that_exist_or_a_reason() -> None:
    for target, (path, entry) in links().items():
        tests = entry.get("tests", ())
        if not tests:
            assert entry.get("reason"), f"{target}: neither tests nor a reason"
            continue
        absent = [
            test
            for test in tests
            if _resolve(path, test)[1] not in _test_names(_resolve(path, test)[0])
        ]
        assert not absent, f"{target}: {absent} not in {path.name}"


def test_every_linked_test_reaches_its_object() -> None:
    """A link is worth its name only where a test drives the cited code: one of the listed tests,
    with the private helpers of its file it calls, names the object (or imports it under another
    name). A module is reached through any name of its ``__all__``; a class or helper through
    ``via``, a function of its module whose source names it (the function that returns the class,
    the caller of the helper)."""
    unreached: dict[str, tuple[str, ...]] = {}
    for target, (path, entry) in links().items():
        tests = entry.get("tests", ())
        module = _module_of(target)
        leaf = target.rsplit(".", 1)[-1]
        if module.__name__ == target:
            names = list(getattr(module, "__all__", None) or _defined(module))
        else:
            names = [leaf]
        if "via" in entry:
            via = getattr(module, entry["via"])
            assert re.search(rf"\b{re.escape(leaf)}\b", inspect.getsource(via)), (
                f"{target}: {entry['via']} does not name {leaf}"
            )
            names.append(entry["via"])
        if tests and not any(_reaches(path, test, names) for test in tests):
            unreached[target] = tests
    assert not unreached, unreached
