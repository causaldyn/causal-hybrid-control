"""Every public module has an API page, and every API page renders a module that exists.

The reference is static Markdown -- one `docs/api/<module>.md` per module, holding a single
`::: chc.<module>` directive -- so it drifts in both directions without a sound: a new module ships
with no page, or a renamed one leaves a page behind that fails only when someone builds the site.
Each page also states its stability tier, and `mkdocs.yml` files it under one; both are copies of
the tier table on the reference's index page, and a page promising "stable" for a module the table
calls experimental is a promise nobody made. The modules of `chc.research` have theirs under
`docs/api/research/`, in the research tier.

The index page also lists the experimental entries of the stable and evolving modules, which a
docstring marks *Experimental*: the list holds every marked one, each entry names something that
exists, and the count the page states is the list's.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import chc
import chc.research

ROOT = Path(__file__).resolve().parent.parent
API = ROOT / "docs" / "api"
MODULES = sorted(
    m.name
    for m in pkgutil.iter_modules(chc.__path__)
    if not m.name.startswith("_") and m.name != "research"
)
PAGES = sorted(path.stem for path in API.glob("*.md") if path.stem != "index")
RESEARCH = sorted(m.name for m in pkgutil.iter_modules(chc.research.__path__))
RESEARCH_PAGES = sorted(p.stem for p in (API / "research").glob("*.md") if p.stem != "index")


def _declared_tiers() -> dict[str, str]:
    """The index page names the stable and the experimental modules; the rest are evolving."""
    text = (API / "index.md").read_text()
    named: dict[str, str] = {}
    for tier in ("stable", "experimental"):
        row = re.search(rf"^\| \*\*{tier}\*\* \| (.+?) \|", text, re.MULTILINE)
        assert row is not None, f"docs/api/index.md's tier table has no `{tier}` row"
        named |= dict.fromkeys(re.findall(r"`(\w+)`", row.group(1)), tier)
    return {module: named.get(module, "evolving") for module in MODULES}


def _nav_tiers() -> dict[str, str | None]:
    """The tier group each API page is filed under in `mkdocs.yml`'s nav."""
    tiers: dict[str, str | None] = {}
    group: str | None = None
    for line in (ROOT / "mkdocs.yml").read_text().splitlines():
        if heading := re.fullmatch(r"\s*- (Stable|Evolving|Experimental|Research):\s*", line):
            group = heading.group(1).lower()
        elif page := re.fullmatch(r"\s*- api/((?:research/)?\w+)\.md\s*", line):
            tiers[page.group(1)] = group
    return tiers


def test_every_public_module_has_an_api_page() -> None:
    assert sorted(set(MODULES) - set(PAGES)) == []


def test_every_api_page_renders_the_module_it_is_named_after() -> None:
    wrong = {}
    for page in PAGES:
        targets = re.findall(r"^::: chc\.(\S+)\s*$", (API / f"{page}.md").read_text(), re.MULTILINE)
        if targets != [page] or page not in MODULES:
            wrong[page] = targets
    assert wrong == {}


def test_each_page_states_the_declared_tier_and_is_filed_under_it() -> None:
    declared, nav = _declared_tiers(), _nav_tiers()
    stated = {}
    for page in PAGES:
        text = (API / f"{page}.md").read_text()
        found = re.search(r"\*\*Stability tier: \[(\w+)\]\(index\.md#\1\)\.\*\*", text)
        stated[page] = found.group(1) if found else None
    wrong = {
        module: (declared[module], stated.get(module), nav.get(module))
        for module in MODULES
        if not declared[module] == stated.get(module) == nav.get(module)
    }
    assert wrong == {}


def test_every_research_module_has_a_page_that_renders_it_in_the_research_tier() -> None:
    nav = _nav_tiers()
    wrong = {}
    for module in sorted(set(RESEARCH) | set(RESEARCH_PAGES)):
        page = API / "research" / f"{module}.md"
        text = page.read_text() if page.exists() else ""
        targets = re.findall(r"^::: chc\.research\.(\S+)\s*$", text, re.MULTILINE)
        tier = re.search(r"\*\*Stability tier: \[(\w+)\]\(\.\./index\.md#\1\)\.\*\*", text)
        found = (targets, tier.group(1) if tier else None, nav.get(f"research/{module}"))
        if found != ([module], "research", "research") or module not in RESEARCH:
            wrong[module] = found
    assert wrong == {}
    assert "research/index" in nav


# A docstring's mark, *experimental* in a sentence or "Experimental" opening one; "the
# experimental one", a comparison with an experiment's arm, is no mark.
_MARK = re.compile(r"\*experimental\*|\bExperimental\b", re.IGNORECASE)
_SENTENCE_MARK = re.compile(r"\*experimental\*|\bExperimental\b")
_SECTION = re.compile(r"^(Args|Returns|Yields|Raises|Attributes|Examples?|Notes?):\s*$")
_FIELD_MARK = re.compile(r"``(\w+)``,? \(\*experimental\*\)")
_ENTRY = re.compile(r"^(\w+)(?:\.(\w+))?(?:\((\w+)=(?:\"\w+\")?\))?(?:\[\"\w+\"\])?$")
# A mark that closes a paragraph about a part of what the entry returns, listed under that part.
_MARKS_A_PART = {
    ("decision", "Prescription.evaluate"): ("evaluation", "PlanEvaluation.logger_check")
}


def _split(doc: str) -> tuple[str, dict[str, str]]:
    """A docstring's prose, and its Args and Attributes entries by name."""
    prose: list[str] = []
    entries: dict[str, str] = {}
    section, name = None, None
    for line in inspect.cleandoc(doc).splitlines():
        if heading := _SECTION.match(line):
            section, name = heading.group(1), None
        elif section is None:
            prose.append(line)
        elif section in ("Args", "Attributes"):
            if entry := re.match(r"^    \**(\w+)(?: \([^)]*\))?:\s?(.*)$", line):
                name = entry.group(1)
                entries[name] = entry.group(2)
            elif name is not None:
                entries[name] += " " + line.strip()
    return "\n".join(prose), entries


def _marked(module: str) -> set[tuple[str, str]]:
    """The entries of ``chc.<module>`` its docstrings and field comments mark experimental, each
    with the module it is listed under."""
    path = ROOT / "src" / "chc" / f"{module}.py"
    lines = path.read_text().splitlines()
    marked: set[str] = set()

    def visit(body: list[ast.stmt], owner: str) -> None:
        for node in body:
            if isinstance(node, ast.AnnAssign) and owner and isinstance(node.target, ast.Name):
                above = node.lineno - 2
                comment = [lines[node.lineno - 1].partition("#")[2]]
                while above >= 0 and lines[above].lstrip().startswith("#"):
                    comment.append(lines[above])
                    above -= 1
                if _SENTENCE_MARK.search(" ".join(comment)) and not node.target.id.startswith("_"):
                    marked.add(f"{owner}.{node.target.id}")
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) or node.name.startswith("_"):
                continue
            name = f"{owner}.{node.name}" if owner else node.name
            prose, entries = _split(ast.get_docstring(node) or "")
            if _SENTENCE_MARK.search(prose.replace("(*experimental*)", "")):
                marked.add(name)
            marked.update(f"{name}.{field}" for field in _FIELD_MARK.findall(prose))
            for argument, text in entries.items():
                if _MARK.search(text):
                    kind = isinstance(node, ast.ClassDef)
                    marked.add(f"{name}.{argument}" if kind else f"{name}({argument}=)")
            if isinstance(node, ast.ClassDef):
                visit(node.body, name)

    visit(ast.parse(path.read_text()).body, "")
    return {_MARKS_A_PART.get((module, entry), (module, entry)) for entry in marked}


def _listed() -> tuple[int, dict[str, list[str]]]:
    """The count the index page states, and its experimental entries by module."""
    text = (API / "index.md").read_text()
    head = re.search(r"^(\d+) entries in modules of other tiers", text, re.MULTILINE)
    assert head is not None, "docs/api/index.md states no count of experimental entries"
    listed: dict[str, list[str]] = {}
    body = text[head.end() : text.index("\n## ", head.end())]
    for item in re.split(r"\n- ", body)[1:]:
        module = re.match(r"`chc\.(\w+)`:", item)
        assert module is not None, f"an entry list names no module: {item[:40]!r}"
        listed[module.group(1)] = re.findall(r"`([^`]+)`", item[module.end() :])
    return int(head.group(1)), listed


def _exists(module: object, entry: str) -> bool:
    match = _ENTRY.match(entry)
    if match is None:
        return False
    name, member, argument = match.groups()
    owner = getattr(module, name, None)
    if owner is None:
        return False
    if member is not None:
        fields = (
            {f.name for f in dataclasses.fields(owner)}
            if dataclasses.is_dataclass(owner)
            else set()
        )
        if member not in fields and not hasattr(owner, member):
            return False
        owner = getattr(owner, member, None)
    if argument is not None:
        return argument in inspect.signature(owner).parameters
    return True


def test_the_index_lists_every_experimental_entry_of_the_other_tiers() -> None:
    count, listed = _listed()
    other_tiers = [m for m, tier in _declared_tiers().items() if tier != "experimental"]
    pairs = {(module, entry) for module, entries in listed.items() for entry in entries}
    assert sorted(set().union(*map(_marked, other_tiers)) - pairs) == []
    assert sorted(set(listed) - set(other_tiers)) == []
    absent = {
        module: [e for e in entries if not _exists(importlib.import_module(f"chc.{module}"), e)]
        for module, entries in listed.items()
    }
    assert {m: e for m, e in absent.items() if e} == {}
    assert count == sum(len(entries) for entries in listed.values())
