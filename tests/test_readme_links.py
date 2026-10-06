"""Every link in the README resolves wherever the README is read.

The README is the package's long description, so PyPI renders it too, and PyPI resolves no relative
link: `docs/quickstart.md` is a 404 there. So each link is absolute or an anchor, and an absolute
link into this project names something that exists: a page of the site, built from `docs/` and
the notebooks, with the heading a fragment names, or a file of the repository.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SITE = "https://causaldyn.github.io/causal-hybrid-control/"
REPOSITORY = re.compile(r"https://github\.com/causaldyn/causal-hybrid-control/(?:blob|tree)/main/")
FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _links(markdown: str) -> list[str]:
    prose = FENCE.sub("", markdown)
    return re.findall(r"\]\(([^)\s]+)\)", prose) + re.findall(r'(?:href|src)="([^"]+)"', prose)


LINKS = _links((ROOT / "README.md").read_text(encoding="utf-8"))


def _slug(heading: str) -> str:
    """Python-Markdown's default heading id, which mkdocs gives each heading."""
    text = unicodedata.normalize("NFKD", heading).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[-\s]+", "-", re.sub(r"[^\w\s-]", "", text).strip().lower())


def _page(path: str) -> str | None:
    """The markdown a site path is built from, or None where no page builds there."""
    stem = path.strip("/")
    if stem.startswith("tutorials/") and stem != "tutorials":
        notebook = ROOT / "notebooks" / f"{stem.removeprefix('tutorials/')}.ipynb"
        if not notebook.is_file():
            return None
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    for source in (ROOT / "docs" / f"{stem}.md", ROOT / "docs" / stem / "index.md"):
        if stem and source.is_file():
            return source.read_text(encoding="utf-8")
    return (ROOT / "docs" / "index.md").read_text(encoding="utf-8") if not stem else None


def _ids(markdown: str) -> set[str]:
    return {_slug(heading) for heading in HEADING.findall(FENCE.sub("", markdown))}


def test_the_readme_has_links_to_check() -> None:
    assert any(link.startswith(SITE) for link in LINKS)
    assert any(REPOSITORY.match(link) for link in LINKS)


@pytest.mark.parametrize("link", LINKS)
def test_each_link_is_absolute_or_an_anchor(link: str) -> None:
    assert link.startswith(("https://", "#"))


@pytest.mark.parametrize("link", [link for link in LINKS if link.startswith(SITE)])
def test_each_site_link_names_a_page_and_a_heading_on_it(link: str) -> None:
    path, _, fragment = link.removeprefix(SITE).partition("#")
    page = _page(path)
    assert page is not None, f"no page builds at {path!r}"
    assert not fragment or fragment in _ids(page), f"{path!r} has no heading #{fragment}"


@pytest.mark.parametrize("link", [link for link in LINKS if REPOSITORY.match(link)])
def test_each_repository_link_names_a_file_on_main(link: str) -> None:
    assert (ROOT / REPOSITORY.sub("", link)).exists()


@pytest.mark.parametrize("link", [link for link in LINKS if link.startswith("#")])
def test_each_anchor_names_a_heading_of_the_readme(link: str) -> None:
    assert link.removeprefix("#") in _ids((ROOT / "README.md").read_text(encoding="utf-8"))
