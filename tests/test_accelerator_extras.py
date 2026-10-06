"""The accelerator extras forward to JAX's own, name for name."""

from __future__ import annotations

import tomllib
from importlib.metadata import metadata, version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent


def test_every_extra_that_forwards_to_jax_names_one_jax_provides() -> None:
    """pip and uv install an extra jax does not provide with a warning and nothing else, so a
    renamed or retired JAX build would leave `pip install "causal-hybrid-control[cuda13]"` on the
    CPU. An extra whose floor or marker excludes the jax installed here is skipped: jax 0.4.30, the
    floor, predates the CUDA 13, ROCm 7 and oneAPI builds."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        extras = tomllib.load(handle)["project"]["optional-dependencies"]
    forwarded = {
        name: requirement
        for name, specs in extras.items()
        for requirement in map(Requirement, specs)
        if requirement.name == "jax"
    }
    provided = {
        canonicalize_name(extra) for extra in metadata("jax").get_all("Provides-Extra") or ()
    }
    installed = version("jax")
    applicable = {
        name: requirement
        for name, requirement in forwarded.items()
        if (requirement.marker is None or requirement.marker.evaluate())
        and requirement.specifier.contains(installed, prereleases=True)
    }
    assert {"cpu", "cuda12"} <= set(applicable)
    for name, requirement in applicable.items():
        assert requirement.extras == {name}
        assert name in provided, f"jax {version('jax')} provides no extra {name!r}"
