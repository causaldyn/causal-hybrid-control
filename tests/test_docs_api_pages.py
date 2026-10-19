"""Every public module has an API page, and every API page renders a module that exists.

The reference is static Markdown -- one `docs/api/<module>.md` per module, holding a single
`::: chc.<module>` directive -- so it drifts in both directions without a sound: a new module ships
with no page, or a renamed one leaves a page behind that fails only when someone builds the site.
Each page also states its stability tier, and `mkdocs.yml` files it under one; both are copies of
the tier table on the reference's index page, and a page promising "stable" for a module the table
calls experimental is a promise nobody made. The modules of `chc.research` have theirs under
`docs/api/research/`, in the research tier; the paths they left are not modules of their own and
have no page.
"""

from __future__ import annotations

import pkgutil
import re
from pathlib import Path

import chc
import chc.research
from chc import _moved

ROOT = Path(__file__).resolve().parent.parent
API = ROOT / "docs" / "api"
LEFT = {old.removeprefix("chc.") for old in _moved.MODULES}
MODULES = sorted(
    m.name
    for m in pkgutil.iter_modules(chc.__path__)
    if not m.name.startswith("_") and m.name not in LEFT and m.name != "research"
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
