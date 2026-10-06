"""The committed notebook outputs are what a reader of the README sees on GitHub.

GitHub renders each `.ipynb` with the outputs it was committed with, so those outputs come from one
clean run, top to bottom, with no error in them. A notebook run where an NVIDIA GPU sits beside a
CPU-only jaxlib printed JAX's "Falling back to cpu" warning into its first output; each notebook
pins the CPU, so the outputs do not depend on the machine that made them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
PIN = 'jax.config.update("jax_platforms", "cpu")'


def code_cells(notebook: Path) -> list[dict]:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return [cell for cell in cells if cell["cell_type"] == "code"]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_the_outputs_come_from_one_run_top_to_bottom(notebook: Path) -> None:
    counts = [cell["execution_count"] for cell in code_cells(notebook)]
    assert counts == list(range(1, len(counts) + 1)), (
        f"{notebook.name}: re-run it, `just notebooks`"
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_no_output_is_an_error_or_jax_falling_back_to_the_cpu(notebook: Path) -> None:
    outputs = [output for cell in code_cells(notebook) for output in cell.get("outputs", [])]
    assert [output["ename"] for output in outputs if output["output_type"] == "error"] == []
    streams = "".join("".join(output.get("text", "")) for output in outputs)
    assert "CUDA-enabled jaxlib" not in streams


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_each_notebook_pins_the_cpu(notebook: Path) -> None:
    # every notebook imports chc, which imports jax, whether or not the notebook names it
    source = "\n".join("".join(cell["source"]) for cell in code_cells(notebook))
    assert PIN in source, f"{notebook.name} does not pin the CPU"
