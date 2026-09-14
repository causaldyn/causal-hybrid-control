"""Every hand-maintained copy of the version, pinned to the one that ships.

`chc.__version__` was a literal and sat at 0.3.0 through two releases; looking for its siblings
found `CITATION.cff` and the README's BibTeX block stale at exactly the same value, and
`SECURITY.md`'s supported-versions table is a fourth copy waiting to do it again. None of them could
drift *detectably* before, because each agreed with itself.

`pyproject.toml` is the source of truth here rather than the installed metadata, deliberately: these
are files in the repository, and what they must agree with is the version this checkout would build
-- which is what a release commit edits, and what an editable install may lag behind.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def declared_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def test_the_citation_file_names_the_version_that_ships(declared_version: str) -> None:
    """`CITATION.cff` is what Zenodo and GitHub's "Cite this repository" read."""
    text = (ROOT / "CITATION.cff").read_text()
    found = re.search(r"^version:\s*(\S+)\s*$", text, re.MULTILINE)
    assert found is not None, "CITATION.cff has no `version:` key"
    assert found.group(1) == declared_version


def test_the_readme_bibtex_names_the_version_that_ships(declared_version: str) -> None:
    """The BibTeX block is what a paper citing this library copies."""
    text = (ROOT / "README.md").read_text()
    found = re.search(r"^\s*version\s*=\s*\{([^}]+)\},\s*$", text, re.MULTILINE)
    assert found is not None, "README.md has no BibTeX `version = {...}` line"
    assert found.group(1) == declared_version


def test_the_security_policy_supports_the_current_minor(declared_version: str) -> None:
    """A supported-versions table that does not list the current release supports nothing."""
    major, minor, *_ = declared_version.split(".")
    text = (ROOT / "SECURITY.md").read_text()
    assert f"| {major}.{minor}.x | yes |" in text
