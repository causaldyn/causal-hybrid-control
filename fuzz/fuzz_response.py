"""Fuzz :mod:`chc.response`'s saturation curves where their shapes turn: near 1, and far from it.

    uv run --with-requirements fuzz/requirements.txt python fuzz/fuzz_response.py -max_total_time=60

Each input picks a family and its parameters: log-uniform over six decades, within ``1e-15`` above
or below 1, exactly 1, or anywhere in ``[-1, 3]`` so that zero and negative values are tried as
well. A curve refuses its parameters with ``ValueError``, or it keeps every promise of its
docstrings, in float64:

* it runs from 0, to rounding, to its ceiling of 1 and never falls, and its slope is finite and
  never negative, which a planner's gradient reads;
* its inflection and tangency are finite, the inflection is not past the tangency, and at the
  tangency the line from the origin supports the curve: a tangent where the curve is smooth, and
  between its slopes on either side at the corner where a curve convex up to the end of its
  support meets its ceiling;
* its ``nonconvexity()`` is in ``[0, 1]``, and its ``Envelope`` lies on or above it, below 1, and
  never further above it than ``nonconvexity()`` says.

Anything else, a broken promise or another exception, is a crash with the input that caused it.
"""

from __future__ import annotations

import math
import sys

import atheris
import jax

jax.config.update("jax_enable_x64", True)

with atheris.instrument_imports(include=["chc.response"]):
    import jax.numpy as jnp

    from chc import response

FAMILIES = {
    response.MichaelisMenten: (),
    response.Exponential: (),
    response.Tanh: (),
    response.Arctan: (),
    response.Algebraic: (),
    response.HalfNormal: (),
    response.Hill: ("slope",),
    response.Weibull: ("shape",),
    response.Logistic: ("steepness",),
    response.Gompertz: ("displacement",),
    response.Richards: ("steepness", "asymmetry"),
    response.ChapmanRichards: ("power",),
    response.GammaCDF: ("shape",),
    response.LogNormalCDF: ("sigma",),
    response.BurrXII: ("slope", "tail"),
    response.BetaCDF: ("a", "b"),
    response.Kumaraswamy: ("a", "b"),
}
ORDER = list(FAMILIES)
TOLERANCE = 1e-9
ROUNDING = 1e-15  # a curve's value at zero spend, and its least, are 0 to rounding


def _parameter(provider: atheris.FuzzedDataProvider) -> float:
    kind = provider.ConsumeIntInRange(0, 4)
    if kind == 0:
        return 10.0 ** provider.ConsumeFloatInRange(-3.0, 3.0)
    if kind in (1, 2):
        offset = 10.0 ** -provider.ConsumeFloatInRange(0.5, 15.0)
        return 1.0 + offset if kind == 1 else 1.0 - offset
    if kind == 3:
        return 1.0
    return provider.ConsumeFloatInRange(-1.0, 3.0)


def check(data: bytes) -> None:
    provider = atheris.FuzzedDataProvider(data)
    family = ORDER[provider.ConsumeIntInRange(0, len(ORDER) - 1)]
    values = {name: _parameter(provider) for name in FAMILIES[family]}
    try:
        curve = family(scale=1.0, **values)
    except ValueError:
        return
    case = f"{family.__name__}({values})"

    inflection, touch = curve.inflection(), curve.tangency()
    if not (math.isfinite(inflection) and math.isfinite(touch) and inflection >= 0.0):
        raise AssertionError(f"{case}: inflection {inflection}, tangency {touch}")
    if touch > 0.0 and inflection > touch * (1.0 + TOLERANCE):
        raise AssertionError(f"{case}: inflection {inflection} past the tangency {touch}")

    reach = max(4.0 * touch, 4.0 * inflection, 8.0)
    grid = jnp.concatenate(
        [jnp.linspace(0.0, max(touch, 1e-12), 2049), jnp.linspace(0.0, reach, 2049)]
    )
    grid = jnp.sort(grid)
    values_on = curve.standard(grid)
    if not bool(jnp.all(jnp.isfinite(values_on))):
        raise AssertionError(f"{case}: not finite on [0, {reach}]")
    if abs(float(values_on[0])) > ROUNDING or float(jnp.min(values_on)) < -ROUNDING:
        raise AssertionError(
            f"{case}: starts at {float(values_on[0])}, falls to {float(jnp.min(values_on))}"
        )
    if float(jnp.max(values_on)) > 1.0 + TOLERANCE:
        raise AssertionError(f"{case}: rises to {float(jnp.max(values_on))}, past its ceiling")
    if float(jnp.min(jnp.diff(values_on))) < -TOLERANCE:
        raise AssertionError(f"{case}: falls by {-float(jnp.min(jnp.diff(values_on)))}")
    slopes = jax.vmap(jax.grad(curve.standard))(grid)
    if not bool(jnp.all(jnp.isfinite(slopes))) or float(jnp.min(slopes)) < -TOLERANCE:
        worst = jnp.argmin(jnp.where(jnp.isfinite(slopes), slopes, -jnp.inf))
        raise AssertionError(f"{case}: slope {float(slopes[worst])} at {float(grid[worst])}")

    if touch > 0.0:
        chord = float(curve.standard(jnp.asarray(touch))) / touch
        slope = jax.grad(curve.standard)
        left = float(slope(jnp.asarray(math.nextafter(touch, 0.0))))
        right = float(slope(jnp.asarray(touch)))
        if not right * (1.0 - 1e-7) - 1e-12 <= chord <= left * (1.0 + 1e-7) + 1e-12:
            raise AssertionError(
                f"{case}: at the tangency {touch} the chord's slope {chord} is not between the "
                f"curve's {left} on the left and {right} on the right"
            )

    gap = curve.nonconvexity()
    if not 0.0 <= gap <= 1.0:
        raise AssertionError(f"{case}: nonconvexity {gap}")
    envelope = response.Envelope(curve).standard(grid)
    rise = envelope - values_on
    if float(jnp.min(rise)) < -TOLERANCE or float(jnp.max(envelope)) > 1.0 + TOLERANCE:
        raise AssertionError(f"{case}: envelope {float(jnp.min(rise))} below the curve or past 1")
    if float(jnp.max(rise)) > gap + 1e-7:
        raise AssertionError(
            f"{case}: envelope rises {float(jnp.max(rise))}, nonconvexity says {gap}"
        )


def main() -> None:
    atheris.Setup(sys.argv, check)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
