"""Named data enters chc as a mapping, a pandas frame or a polars frame, with the same answer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc.causal import ConfoundedLinearSystem
from chc.estimators import IV2SLS, BackdoorOLS, DoubleML, RLearner
from chc.frames import as_columns
from chc.gmethods import naive_pooled_effect, sequential_g_formula

pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")

G_SPEC = {"treatments": ("a0", "a1"), "confounders": (("l0",), ("l1",)), "outcome": "y"}


def _effect_data() -> dict[str, jax.Array]:
    return ConfoundedLinearSystem().sample(4_000, jax.random.key(0))


def _time_varying_data(n: int = 4_000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    l0 = rng.normal(0.0, 1.0, n)
    a0 = 0.9 * l0 + rng.normal(0.0, 1.0, n)
    l1 = a0 + 0.7 * l0 + rng.normal(0.0, 1.0, n)
    a1 = 1.1 * l1 + 0.3 * a0 + rng.normal(0.0, 1.0, n)
    y = a0 + 1.5 * a1 + 0.5 * l0 + 0.8 * l1 + rng.normal(0.0, 0.3, n)
    return {"a0": a0, "a1": a1, "l0": l0, "l1": l1, "y": y}


def _as_numpy(data: dict[str, jax.Array] | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(column) for name, column in data.items()}


@pytest.mark.parametrize("estimator", [BackdoorOLS(), IV2SLS(), DoubleML(), RLearner()])
def test_estimators_agree_across_frame_backends(estimator) -> None:
    data = _effect_data()
    columns = _as_numpy(data)
    from_mapping = estimator.estimate(data).effect
    from_pandas = estimator.estimate(pd.DataFrame(columns)).effect
    from_polars = estimator.estimate(pl.DataFrame(columns)).effect
    assert from_pandas == pytest.approx(from_mapping, rel=1e-6)
    assert from_polars == pytest.approx(from_mapping, rel=1e-6)


def test_g_methods_agree_across_frame_backends() -> None:
    data = _time_varying_data()
    regime = {"regime": (1.0, 1.0), "baseline": (0.0, 0.0), **G_SPEC}
    for frame in (pd.DataFrame(data), pl.DataFrame(data)):
        assert sequential_g_formula(frame, **regime) == pytest.approx(
            sequential_g_formula(data, **regime), rel=1e-9
        )
        assert naive_pooled_effect(frame, **G_SPEC) == pytest.approx(
            naive_pooled_effect(data, **G_SPEC), rel=1e-9
        )


def test_a_polars_frame_yields_column_names_not_column_values() -> None:
    """The regression this module exists for: polars survives the mapping idioms with wrong data.

    ``dict(frame)`` and ``for name in frame`` iterate a polars frame's columns *as values*, so the
    pre-normalisation code would have keyed on data and raised nothing.
    """
    frame = pl.DataFrame({"u": [1.0, 2.0], "x": [3.0, 4.0]})
    assert list(as_columns(frame)) == ["u", "x"]
    assert np.asarray(as_columns(frame)["x"]).tolist() == [3.0, 4.0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_round_trip_through_either_frame_preserves_dtype(dtype) -> None:
    """A frame that silently widened float32 would move every fitted number in this library."""
    column = np.arange(4, dtype=dtype)
    for frame in (pd.DataFrame({"u": column}), pl.DataFrame({"u": column})):
        assert np.asarray(as_columns(frame)["u"]).dtype == dtype


def test_a_mapping_passes_through_without_a_host_copy() -> None:
    """Identity, not equality: converting the library's own JAX arrays would copy off-device."""
    column = jnp.asarray([1.0, 2.0])
    assert as_columns({"u": column})["u"] is column


def test_as_columns_names_the_type_it_cannot_read() -> None:
    with pytest.raises(TypeError, match="got list"):
        as_columns([1.0, 2.0])  # type: ignore[arg-type]
