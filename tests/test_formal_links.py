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
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "chc"
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


def _test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
    }


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
        absent = set(tests) - _test_names(path)
        assert not absent, f"{target}: {sorted(absent)} not in {path.name}"
