"""The accelerator extras forward to JAX's own, name for name."""

from __future__ import annotations

import tomllib
from importlib.metadata import metadata, version
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent


def test_every_extra_that_forwards_to_jax_names_one_jax_provides() -> None:
    """pip and uv install an extra jax does not provide with a warning and nothing else, so a
    renamed or retired JAX build would leave `pip install "causal-hybrid-control[cuda13]"` on the
    CPU. The oneAPI extra's marker skips it where the lock resolves a jax without one."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        extras = tomllib.load(handle)["project"]["optional-dependencies"]
    forwarded = {
        name: requirement
        for name, specs in extras.items()
        for requirement in map(Requirement, specs)
        if requirement.name == "jax"
    }
    provided = set(metadata("jax").get_all("Provides-Extra") or ())
    applicable = {
        name: requirement
        for name, requirement in forwarded.items()
        if requirement.marker is None or requirement.marker.evaluate()
    }
    assert {"cpu", "cuda13", "cuda12"} <= set(applicable)
    for name, requirement in applicable.items():
        assert requirement.extras == {name}
        assert name in provided, f"jax {version('jax')} provides no extra {name!r}"
