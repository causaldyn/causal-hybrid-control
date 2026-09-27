"""Notebook prose carries no TeX, because the documentation site loads no math renderer.

The site sends a reader's browser to no third party -- `mkdocs.yml` turns Google Fonts off for the
same reason -- and a MathJax or KaTeX script would have to come from one or be vendored in. The
tutorials were the only pages with TeX, three spans in all, so they are written the way every other
page is: inline code with Unicode, `ẋ = f_known(x, u) + r_θ(x, u)`. A TeX span in a markdown cell
reaches the site as raw source.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
# A dollar-delimited span holding TeX markup; two prices in one sentence hold none.
TEX = re.compile(r"\$\$.+?\$\$|\$[^$\n]*[\\_^{][^$\n]*\$", re.DOTALL)


def tex_spans(notebook: Path) -> list[str]:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    prose = ("".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown")
    return [match.group(0) for text in prose for match in TEX.finditer(text)]


def test_there_are_notebooks_to_check() -> None:
    assert NOTEBOOKS, "no notebooks found under notebooks/"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_prose_carries_no_tex(notebook: Path) -> None:
    assert tex_spans(notebook) == [], f"{notebook.name}: TeX the site cannot render"
