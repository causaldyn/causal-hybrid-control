"""Nothing the library publishes points into the research repository it was developed in.

The docstrings under `src/chc/` render into the API reference, the pages under `docs/` are the
site, and `docs/build.sh` executes the notebooks into its tutorials. The sdist also ships
`proofs/`, `validation/`, `scripts/`, `tests/` and `fuzz/`, and the docstrings and pages send a
reader into them by path. `plans/<n>` and `discoveries` are paths in the author's research
repository, which is not public: a reader who follows one finds nothing, and a task's number in one
of its plans, `plan <n>'s P<k>`, points there too. A pointer that carries a result names the public
file that holds it instead -- a proof under `proofs/`, a derivation under `validation/`, a page of
the site -- and a pointer that only records where an idea came from has no reader to serve.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The scan of tests/ reads this file too. `[/]` keeps the pattern's own text from matching it;
# `plans/[0-9]` and a task's number need no such help, since a bracket is not a digit.
PRIVATE_PATH = re.compile(rb"plans/[0-9]|discoveries[/]|plan [0-9]+'s P[0-9]")


def _files(directory: Path) -> list[Path]:
    """Every file below ``directory``, less the bytecode an import may have left behind."""
    return [p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts]


def _pointers(paths: Iterable[Path]) -> list[str]:
    """``file:line`` of every line naming a private path; bytes, since a built page has images."""
    return [
        f"{path.relative_to(ROOT)}:{number}"
        for path in sorted(paths)
        for number, line in enumerate(path.read_bytes().splitlines(), start=1)
        if PRIVATE_PATH.search(line)
    ]


def test_no_library_file_points_into_the_research_repository() -> None:
    assert _pointers(_files(ROOT / "src" / "chc")) == []


def test_no_docs_page_points_into_the_research_repository() -> None:
    # The tutorial pages exist only once the site is built, so the notebooks stand in for them.
    notebooks = list((ROOT / "notebooks").glob("*.ipynb"))
    assert _pointers(_files(ROOT / "docs") + notebooks) == []


def test_no_proof_derivation_script_or_test_points_into_the_research_repository() -> None:
    # The sdist ships these beside the library, and the docstrings and pages cite them by path.
    shipped = ("proofs", "validation", "scripts", "tests", "fuzz")
    assert _pointers([path for name in shipped for path in _files(ROOT / name)]) == []
