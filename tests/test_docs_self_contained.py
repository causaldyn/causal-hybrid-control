"""Nothing the library publishes points into the research repository it was developed in.

The docstrings under `src/chc/` render into the API reference, the pages under `docs/` are the
site, and `docs/build.sh` executes the notebooks into its tutorials. The sdist also ships
`proofs/`, `validation/`, `scripts/`, `tests/` and `fuzz/`, and the docstrings and pages send a
reader into them by path. `plans/<n>` and `discoveries` are paths in the author's research
repository, which is not public: a reader who follows one finds nothing, and a task's number in one
of its plans, `plan <n>'s P<k>`, points there too. A pointer that carries a result names the public
file that holds it instead -- a proof under `proofs/`, a derivation under `validation/`, a page of
the site -- and a pointer that only records where an idea came from has no reader to serve.

A result's number in the research log points there as well: `Result 55`, `§40`, `CONTRIBUTION 2`,
an item's `(A20)` or `(L3.2)`. The text says what the result states, or names the public object
that holds it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import chc

ROOT = Path(__file__).resolve().parent.parent
# The package under test: the source tree in a checkout, site-packages where a wheel is tested.
LIBRARY = Path(chc.__file__).resolve().parent
# The scan of tests/ reads this file too. `[/]` keeps the pattern's own text from matching it;
# `plans/[0-9]` and a task's number need no such help, since a bracket is not a digit.
PRIVATE_PATH = re.compile(rb"plans/[0-9]|discoveries[/]|plan [0-9]+'s P[0-9]")


def _files(directory: Path) -> list[Path]:
    """Every file below ``directory``, less the bytecode an import may have left behind."""
    return [p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts]


def _pointers(paths: Iterable[Path], base: Path = ROOT) -> list[str]:
    """``file:line`` of every line naming a private path; bytes, since a built page has images."""
    return [
        f"{path.relative_to(base)}:{number}"
        for path in sorted(paths)
        for number, line in enumerate(path.read_bytes().splitlines(), start=1)
        if PRIVATE_PATH.search(line)
    ]


def test_no_library_file_points_into_the_research_repository() -> None:
    assert _pointers(_files(LIBRARY), LIBRARY.parent) == []


def test_no_docs_page_points_into_the_research_repository() -> None:
    # The tutorial pages exist only once the site is built, so the notebooks stand in for them.
    notebooks = list((ROOT / "notebooks").glob("*.ipynb"))
    assert _pointers(_files(ROOT / "docs") + notebooks) == []


def test_no_proof_derivation_script_or_test_points_into_the_research_repository() -> None:
    # The sdist ships these beside the library, and the docstrings and pages cite them by path.
    shipped = ("proofs", "validation", "scripts", "tests", "fuzz")
    assert _pointers([path for name in shipped for path in _files(ROOT / name)]) == []


# A result's number, an item's label and a task's number in the research log. A capital `L` or `P`
# label carries a dot, so the laws `(L1)`-`(L3)` a docstring defines for itself do not match, and
# `D(A1)` is a function of a variable, not a label. A name carries the number as well, in digits or
# in words: `result_55`, `result_six_s_bound`.
NUMBER_WORD = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    "sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
)
PRIVATE_LABEL = re.compile(
    r"(?i:\b(?:results?|contributions?) #?[0-9]+[a-z]?\b(?!\.[0-9]))"
    rf"|(?i:(?<![a-z0-9])results?_(?:[0-9]+|{NUMBER_WORD})(?![a-z0-9]))"
    r"|(?<!\w)\((?:A[0-9]+|L[0-9]+\.[0-9]+)\)"
    r"|\bP[0-9]+\.[0-9]+\b"
    r"|§[0-9]"
)
# A section sign is public where it cites a dated source: a year or an arXiv number shortly before
# it, or a page right after it, as in `Kendall (1954) §3` or `(§3.4, p. 501)`.
DATED_SOURCE = re.compile(r"(?:19|20)[0-9]{2}|arXiv")
PAGE = re.compile(r"§[0-9.]+,\s+pp?\. [0-9]")
# Old names that still serve their value under a DeprecationWarning until 1.0.
RENAMED = ("denominator_against_result_55",)


def _labels_in(text: str) -> list[tuple[int, str]]:
    """``(offset, label)`` of every private label in ``text``, less the cited published sections."""
    for name in RENAMED:
        text = text.replace(name, " " * len(name))
    return [
        (m.start(), m.group())
        for m in PRIVATE_LABEL.finditer(text)
        if not (
            m.group().startswith("§")
            and (
                DATED_SOURCE.search(text, max(0, m.start() - 120), m.start())
                or PAGE.match(text, m.start())
            )
        )
    ]


def _labels(paths: Iterable[Path], base: Path = ROOT) -> list[str]:
    """``file:line: label`` of every private label; text, since a citation can span a line."""
    found = []
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8")
        for start, label in _labels_in(text):
            line = text.count("\n", 0, start) + 1
            found.append(f"{path.relative_to(base)}:{line}: {label}")
    return found


def _prose(directory: Path) -> list[Path]:
    """The files below ``directory`` a reader reads: sources, pages and derivations, no images."""
    return [p for p in _files(directory) if p.suffix not in {".png", ".svg", ".ico", ".pdf"}]


def test_no_library_docstring_or_comment_cites_a_research_log_result() -> None:
    assert _labels(_prose(LIBRARY), LIBRARY.parent) == []


def test_no_docs_page_or_derivation_cites_a_research_log_result() -> None:
    pages = _prose(ROOT / "docs") + _prose(ROOT / "validation") + [ROOT / "README.md"]
    assert _labels(pages) == []


def test_no_proof_script_or_test_cites_a_research_log_result() -> None:
    # This file is left out: its docstring and its pattern's own examples spell the labels.
    shipped = [
        path for name in ("proofs", "scripts", "tests", "fuzz") for path in _prose(ROOT / name)
    ]
    assert _labels(path for path in shipped if path != Path(__file__).resolve()) == []


def test_the_label_pattern_tells_a_research_log_result_from_a_published_section() -> None:
    private = (
        "the bound of Result 55 (d)",
        "RESULT 68 (A20) -- the port-Hamiltonian residual",
        "exactly Results 18 (order transfer)",
        "CONTRIBUTION 3 -- adaptive exploration",
        "a self-certifying bound (result #3)",
        "the safety analogue of §11's sign threshold",
        "a bound inside a box (L3.2)",
        "Plan 24's P2.6",
        "def result_56(horizon):",
        "Lemma the_one_by_one_floor_is_result_ten_s_constant :",
        "Corollary results_56_and_66_follow_at_every_horizon :",
    )
    public = (
        "Kendall (1954) §3 gives the variance",
        "(Chernozhukov, Wuthrich and Zhu, JASA 2021, §2.4.2 and Lemma 7)",
        "moved California into the donor pool (§3.4,\n  p. 501)",
        "a near-zero link caps a path (weakest-link, L3), and the truncation loses little (L2).",
        "loses ``q (A1 - A1*)^2 / D(A1)`` of it",
        "the result 0.5 is returned",
        "result_bounds = solve(results_by_seed)",
    )
    assert [text for text in private if not _labels_in(text)] == []
    assert [text for text in public if _labels_in(text)] == []
