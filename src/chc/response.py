"""A channel's response to its spend: how its return rises with what is spent, how long the spend
carries over, and what the return is per unit spent.

Every family is a standard shape ``g`` of spend in units of its own scale ``K``,
``h(spend) = g(spend / K)``, with ``K`` in currency and every other parameter dimensionless. A
change of currency therefore moves ``K`` and nothing else, and every number read off a curve -- a
value, a slope per unit of spend, an inflection or a tangency in spend -- moves with it exactly.

**Bounded families** (:class:`Saturation`) start at ``h(0) = 0``, increase and rise to a ceiling of
1; a channel's coefficient carries the size of the return, the curve only its shape.

* Concave from zero spend: :class:`MichaelisMenten`, :class:`Exponential` (Mitscherlich's law, the
  monomolecular curve), :class:`Tanh`, :class:`Arctan`, :class:`Algebraic` and :class:`HalfNormal`.
* A shape parameter, S-shaped over part of its range: :class:`Hill` (the log-logistic),
  :class:`Weibull`, :class:`Logistic` and :class:`Gompertz`, both normalised to 0 at 0,
  :class:`Richards`, :class:`ChapmanRichards`, :class:`GammaCDF`, :class:`LogNormalCDF`,
  :class:`BurrXII`, and :class:`BetaCDF` and :class:`Kumaraswamy`, which reach the ceiling at ``K``.

**Unbounded baselines** (:class:`Response`): :class:`Logarithmic` and :class:`Power`, the constant
elasticity. **Not monotone, apart:** :class:`Ricker`, the inverted U of ad fatigue; nothing below
about envelopes applies to it.

A floor is not a shape. At zero spend a floor is the plant's base, so Janoschek's curve is
:class:`Weibull` with a floor, and ADBUDG (Little 1970) and Morgan--Mercer--Flodin are
:class:`Hill` with one.

**Why the envelope.** On an S-shaped curve ``h'(0) = 0`` wherever the curve starts convex, so zero
spend on a channel is a stationary point, and a planner started at zero can stop there and report
convergence. The concave envelope (:class:`Envelope`) is the tangent from the origin up to the
spend ``A`` where it touches the curve, ``h(A) = A h'(A)``, and the curve beyond it. Planned
against, it has no such trap, its plan is a warm start for the true curve, and its value bounds the
true optimum, so the gap between the two certifies the plan. :func:`relax` swaps every such curve in
a model for its envelope, and :func:`chc.plan.causal_plan` plans the relaxed model first whenever it
is given no warm start. The inflection and the tangency are in closed form where one exists and
found as a root otherwise (``validation/response_curves.mac``).

**Carryover** (:class:`Adstock`). A period's adstock is the spend of the ``length`` periods up to
it, weighted by lag, ``a_t = sum_{l < L} w_l s_{t - l}``, with nothing spent before the series:
:class:`GeometricAdstock`, :class:`DelayedAdstock`, whose carryover peaks after the spend, and
:class:`WeibullAdstock`. The length is the kernel's own, never a share of the series, so logging
another period moves no earlier value; ``normalized`` has no default, since packages differ on it
and every coefficient fitted through the kernel moves with it. :class:`Channel` is a channel's
return, ``coefficient * curve(kernel(spend))``, adstock first.

**The return**, in the outcome's units unless ``revenue_per_kpi`` turns a KPI into revenue:
:func:`contribution`, the channel's return in a window's periods; and per currency unit spent,
:func:`roi`, the incremental return on a window's spend, carryover included;
:func:`marginal_roi`, the return on one more unit spread over the window; and
:func:`steady_state_marginal_roi`, the slope of a period's return once the adstock of a constant
spend has settled. A coefficient is none of them: through a long geometric kernel of
retention ``r`` a unit of spend on a linear channel returns ``1 / (1 - r)`` times what it returns
in its own period, and on a saturating curve the return per unit moves with the spend.

**PyMC-Marketing, mapped** (``scripts/pymc_marketing_reference.py`` runs its transforms beside
these). ``geometric_adstock`` and ``delayed_adstock`` with ``l_max`` are :class:`GeometricAdstock`
and :class:`DelayedAdstock` with ``length = l_max``; the CDF ``weibull_adstock`` with ``l_max``
carries one weight more, :class:`WeibullAdstock` with ``length = l_max + 1`` and its ``lam`` as
``scale``. ``LogisticSaturation``, ``(1 - e^{-lam x}) / (1 + e^{-lam x})``, is
``tanh(lam x / 2)``: concave from zero, so :class:`Tanh` with ``K = 2 / lam``, not
:class:`Logistic`. ``tanh_saturation`` with ``b`` and ``c`` is ``b`` times :class:`Tanh` with
``K = b c``, ``michaelis_menten`` with ``alpha`` and ``lam`` is ``alpha`` times
:class:`MichaelisMenten` with ``K = lam``, ``hill_function`` is :class:`Hill`,
``hill_saturation_sigmoid`` is a multiple of :class:`Logistic`, and ``root_saturation`` is
:class:`Power`. Neither PyMC-Marketing nor Meridian is a dependency: each would bring a
probabilistic-programming stack for a few closed forms, and neither has the envelope.

HONEST SCOPE:

* :meth:`Saturation.inflection`, :meth:`Saturation.tangency` and :class:`Envelope` read concrete
  parameters: a fitted curve, not one inside a trace. Every curve itself traces, differentiates and
  compiles, and its parameters are leaves a fit can move. So does every kernel.
* :class:`BetaCDF` has no derivative in its shapes ``a`` and ``b``, since JAX's ``betainc`` has
  none; asking for one raises. :class:`Kumaraswamy` is its closed-form counterpart.
* Spend is nonnegative. Below zero a family's value is not defined, and several return ``nan``.
* A curve that rises from zero spend like ``z^m`` has its slopes of order above ``m`` infinite
  there unless ``m`` is whole: the slope itself below shape 1 (:class:`Hill`, :class:`Weibull`,
  :class:`ChapmanRichards`, :class:`BurrXII`, :class:`Kumaraswamy`, :class:`GammaCDF`,
  :class:`BetaCDF`, and :class:`Power`), the curvature below shape 2. Each is read a machine epsilon
  of the scale off zero instead: finite, so a zero weight on it, as in a period no spend reaches, is
  nothing rather than ``nan``, and the slope still steeper there than anywhere past it. A number
  read through one, such as a decision weight in a parameter that turns that spend on, is large
  and finite, not infinite. Every slope finite at zero spend is the curve's own.
* In single precision the numeric tangencies are good to single precision.
* The readings of the return take concrete numbers: they report on a fitted channel, not on one
  inside a trace. The carryover they count is what falls inside the series; to count the rest,
  extend the series with what is spent after it.
* Meridian's forms are not mapped yet, nor run beside these.
"""

from __future__ import annotations

import abc
import functools
import math
from collections.abc import Callable
from typing import ClassVar, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.core import Tracer
from jax.custom_derivatives import SymbolicZero
from jax.scipy.special import betainc, betaln, erf, gammainc, gammaln, ndtr, xlog1py, xlogy
from jax.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import lambertw

Model = TypeVar("Model")


def _real(value: ArrayLike) -> Array:
    return jnp.asarray(value, dtype=float)


def _require(
    name: str, value: Array, low: float, high: float = math.inf, *, low_included: bool = False
) -> None:
    """Refuse a parameter outside ``(low, high]``, or ``[low, high]`` when ``low_included``."""
    if jnp.ndim(value) != 0:
        raise ValueError(
            f"{name} has shape {jnp.shape(value)}; each parameter is one number, so several "
            "channels take several curves and kernels"
        )
    if isinstance(value, Tracer):
        return  # a fit's own parameterisation keeps a traced value in range
    number = float(value)
    above = number >= low if low_included else number > low
    if not (above and number <= high and math.isfinite(number)):
        interval = f"{'[' if low_included else '('}{low}, {high}{']' if high < math.inf else ')'}"
        raise ValueError(f"{name}={number} is outside {interval}")


def _lambert_root(rate: float) -> float:
    """The positive root of ``e^u - 1 = rate * u``, for ``rate > 1``.

    With ``t = u + 1/rate`` the equation is ``t e^{-t} = e^{-1/rate} / rate``, so ``-t`` is Lambert
    W of ``-e^{-1/rate} / rate``: the 0 branch gives ``u = 0``, the -1 branch the positive root.
    """
    return -1.0 / rate - float(lambertw(-math.exp(-1.0 / rate) / rate, k=-1).real)


def _off_zero(base: Array, exponent: Array) -> Array:
    """``base``, but a machine epsilon where it is 0 and the power of it is not above 0."""
    epsilon = jnp.finfo(jnp.result_type(base, exponent)).eps
    return jnp.where((base == 0.0) & (exponent <= 0.0), epsilon, base)


@jax.custom_jvp
def _power(base: Array, exponent: Array) -> Array:
    """``base ** exponent`` for ``base >= 0``, with every slope of it finite at a zero base.

    A power below 1 rises from zero with an infinite slope, and the chain rule multiplies that by
    whatever zero meets it into ``nan``: a period no spend reaches, the slope of ``spend / K`` in
    ``K`` at zero spend. Each slope is a power one lower, so a power not above 0 at a zero base,
    which only a slope meets, is read a machine epsilon off zero, where it is finite. A positive
    power is exact at zero, and so is every slope that is finite there.
    """
    return _off_zero(base, exponent) ** exponent


@_power.defjvp
def _power_jvp(primals: tuple[Array, Array], tangents: tuple[Array, Array]):
    base, exponent = primals
    base_dot, exponent_dot = tangents
    value = _power(base, exponent)
    at = _off_zero(base, exponent)
    # the slope in the exponent is the value times log(base), so 0 where the value is
    log = jnp.log(jnp.where(at > 0.0, at, 1.0))
    return value, exponent * _power(base, exponent - 1.0) * base_dot + value * log * exponent_dot


def _density(power: Array, rest: Callable[[Array], Array], z: Array) -> Array:
    """``z^power exp(rest(z))``: through logarithms away from zero, so that neither factor
    overflows, and through :func:`_power` at zero, so that its slopes there are finite."""
    zero = z == 0.0
    at, away = jnp.where(zero, z, 0.0), jnp.where(zero, 0.5, z)
    return jnp.where(
        zero, _power(at, power) * jnp.exp(rest(at)), jnp.exp(xlogy(power, away) + rest(away))
    )


@jax.custom_jvp
def _gamma_cdf(shape: Array, z: Array) -> Array:
    return gammainc(shape, z)


@functools.partial(_gamma_cdf.defjvp, symbolic_zeros=True)
def _gamma_cdf_jvp(primals: tuple[Array, Array], tangents: tuple[Array, Array]):
    # JAX's own slope in z is exp((a - 1) log z - z - lgamma(a)), which is 0 * -inf = nan at zero
    # spend for a = 1, where the curve is the exponential and its slope 1
    shape, z = primals
    shape_dot, z_dot = tangents
    value = _gamma_cdf(shape, z)
    slope = jnp.zeros_like(value)
    if not isinstance(shape_dot, SymbolicZero):
        slope = slope + jax.jvp(lambda a: gammainc(a, z), (shape,), (shape_dot,))[1]
    if not isinstance(z_dot, SymbolicZero):
        slope = slope + _density(shape - 1.0, lambda at: -at - gammaln(shape), z) * z_dot
    return value, slope


@jax.custom_jvp
def _beta_cdf(a: Array, b: Array, z: Array) -> Array:
    return betainc(a, b, z)


@functools.partial(_beta_cdf.defjvp, symbolic_zeros=True)
def _beta_cdf_jvp(primals: tuple[Array, Array, Array], tangents: tuple[Array, Array, Array]):
    # the slope in z as for the gamma CDF; JAX has none in a and b, and a zero would be a wrong one
    a, b, z = primals
    a_dot, b_dot, z_dot = tangents
    if not (isinstance(a_dot, SymbolicZero) and isinstance(b_dot, SymbolicZero)):
        raise ValueError(
            "BetaCDF has no derivative in its shapes a and b (JAX's betainc has none); fit them "
            "without one, or use Kumaraswamy, its closed-form counterpart"
        )
    value = _beta_cdf(a, b, z)
    if isinstance(z_dot, SymbolicZero):
        return value, jnp.zeros_like(value)
    density = _density(a - 1.0, lambda at: xlog1py(b - 1.0, -at) - betaln(a, b), z)
    return value, density * z_dot


class Response(eqx.Module):
    """A response curve: the return on a channel as a function of its spend, zero at zero spend.

    ``h(spend) = standard(spend / scale)``, with ``scale`` in currency.
    """

    scale: eqx.AbstractVar[Array]

    @abc.abstractmethod
    def standard(self, z: Array) -> Array:
        """The curve with spend in units of ``scale``."""

    def __call__(self, spend: ArrayLike) -> Array:
        return self.standard(jnp.asarray(spend, dtype=float) / self.scale)

    def __check_init__(self) -> None:
        _require("scale", self.scale, 0.0)


class Saturation(Response):
    """A bounded response: zero at zero spend, increasing, and rising to a ceiling of 1."""

    _support: ClassVar[float] = math.inf  # where the curve reaches its ceiling, in scales

    @abc.abstractmethod
    def _standard_inflection(self) -> float:
        """Where the curve turns from convex to concave, in scales; 0 if it is concave from 0."""

    def _standard_tangency(self) -> float:
        """Where the tangent from the origin touches the curve, ``g(z) = z g'(z)``, in scales.

        ``(g - z g')' = -z g''``, so ``g - z g'`` falls from 0 while the curve is convex and rises
        once it is concave: the touching point is its one root past the inflection.
        """
        start = self._standard_inflection()
        if start == 0.0:
            return 0.0
        slope = jax.grad(self.standard)

        def gap(z: float) -> float:
            at = _real(z)
            return float(self.standard(at) - at * slope(at))

        if gap(start) >= 0.0:
            return start  # the convex stretch is within rounding of none
        end = start
        for _ in range(64):
            # past its support a curve is flat at 1, where the gap is 1
            end = min(2.0 * end, self._support)
            if gap(end) > 0.0:
                return brentq(gap, start, end, xtol=1e-15 * start)
        raise RuntimeError(f"{self!r}: no tangency within {end} scales")

    def inflection(self) -> float:
        """Where the curve turns from convex to concave, in spend; 0 if it is concave from zero.

        Below it ``h'' > 0``, so a plan's first-order conditions do not certify it there.
        """
        return float(self.scale) * self._standard_inflection()

    def tangency(self) -> float:
        """The spend where the tangent from the origin touches the curve; 0 if it is concave."""
        return float(self.scale) * self._standard_tangency()

    def nonconvexity(self) -> float:
        """The most the envelope rises above the curve, ``max (envelope - curve)``; 0 if concave.

        A number of the shape alone, on the curve's scale of a ceiling of 1, so a change of
        currency leaves it. A plan made on envelopes loses at most a channel's coefficient times
        this where it leaves the channel inside its chord (:mod:`chc.allocation`). Hill's at slope
        2, 3 and 5 is 0.0674, 0.1547 and 0.2920, and it rises to 1, a step's, as the slope grows
        (``validation/envelope_nonconvexity.mac``).
        """
        touch = self._standard_tangency()
        if touch == 0.0:
            return 0.0
        chord = float(self.standard(_real(touch))) / touch
        slope = jax.grad(self.standard)

        def rise(z: float) -> float:
            return float(slope(_real(z))) - chord

        # the excess chord z - g rises while g' < chord and falls from where g' = chord, below the
        # inflection, to the tangency, where g' = chord again: the first crossing is its peak
        bend = self._standard_inflection()
        if rise(0.0) >= 0.0 or rise(bend) <= 0.0:
            return 0.0  # the convex stretch is within rounding of none
        peak = brentq(rise, 0.0, bend, xtol=1e-15 * bend)
        return chord * peak - float(self.standard(_real(peak)))


class Envelope(Saturation):
    """The concave envelope of a saturation curve: the least concave curve on or above it.

    Linear from the origin to the tangency ``A``, where ``h(A) = A h'(A)``, and the curve beyond
    it; the curve itself when it is concave. Planned against, it bounds the return any plan on the
    curve can reach, and its plan is the warm start that keeps a planner off zero spend.
    """

    curve: Saturation
    # the tangency, in scales: a leaf, so that envelopes of different curves of one family share a
    # tree structure, and one compiled program serves every draw of a posterior
    touch: Array

    def __init__(self, curve: Saturation) -> None:
        self.curve = curve
        self.touch = _real(curve._standard_tangency())

    @property
    def scale(self) -> Array:
        return self.curve.scale

    def standard(self, z: Array) -> Array:
        # the touching point is held fixed: there g = z g', so its own derivative drops out of the
        # chord's, and a slope in the curve's parameters is exact. A concave curve touches at 0
        # and its chord is never read; 1 stands in for it so that the division is defined
        bent = self.touch > 0.0
        at = jnp.where(bent, self.touch, 1.0)
        chord = self.curve.standard(at) / at
        return jnp.where(bent & (z < self.touch), chord * z, self.curve.standard(z))

    def _standard_inflection(self) -> float:
        return 0.0


def relax(model: Model) -> Model:
    """``model`` with every saturation curve in it that starts convex replaced by its envelope.

    ``model`` is any pytree holding curves: a plant, a tuple of curves, one curve. A curve concave
    from zero is left as it is, and when no curve starts convex the answer is ``model`` itself, so
    ``relax(model) is model`` says there is nothing to relax. :func:`chc.plan.causal_plan` reads it
    that way, and starts its descent from the relaxed problem's plan.
    """
    relaxed = False

    def swap(node: object) -> object:
        nonlocal relaxed
        if isinstance(node, Saturation):
            envelope = Envelope(node)
            if envelope.touch > 0.0:
                relaxed = True
                return envelope
        return node

    answer = jax.tree_util.tree_map(swap, model, is_leaf=lambda node: isinstance(node, Saturation))
    return answer if relaxed else model


class MichaelisMenten(Saturation):
    """``z / (1 + z)``: half the ceiling at ``K``. :class:`Hill` at slope 1."""

    scale: Array = eqx.field(converter=_real)

    def standard(self, z: Array) -> Array:
        return z / (1.0 + z)

    def _standard_inflection(self) -> float:
        return 0.0


class Exponential(Saturation):
    """``1 - e^{-z}``: Mitscherlich's law, the monomolecular curve; ``1 - 1/e`` of the ceiling at
    ``K``."""

    scale: Array = eqx.field(converter=_real)

    def standard(self, z: Array) -> Array:
        return -jnp.expm1(-z)

    def _standard_inflection(self) -> float:
        return 0.0


class Tanh(Saturation):
    """``tanh(z)``. PyMC-Marketing's ``LogisticSaturation`` with ``lam`` is this with
    ``K = 2 / lam``."""

    scale: Array = eqx.field(converter=_real)

    def standard(self, z: Array) -> Array:
        # XLA's tanh falls by an ulp here and there on its way to 1; this form rises, and is closer
        return -jnp.expm1(-2.0 * z) / (1.0 + jnp.exp(-2.0 * z))

    def _standard_inflection(self) -> float:
        return 0.0


class Arctan(Saturation):
    """``(2/pi) arctan(z)``: half the ceiling at ``K``, and the slowest approach to it here."""

    scale: Array = eqx.field(converter=_real)

    def standard(self, z: Array) -> Array:
        return 2.0 / jnp.pi * jnp.arctan(z)

    def _standard_inflection(self) -> float:
        return 0.0


class Algebraic(Saturation):
    """``z / sqrt(1 + z^2)``, which is ``x / sqrt(K^2 + x^2)`` in spend."""

    scale: Array = eqx.field(converter=_real)

    def standard(self, z: Array) -> Array:
        return z / jnp.hypot(1.0, z)

    def _standard_inflection(self) -> float:
        return 0.0


class HalfNormal(Saturation):
    """``erf(z)``: the half-normal CDF with standard deviation ``K / sqrt(2)``."""

    scale: Array = eqx.field(converter=_real)

    def standard(self, z: Array) -> Array:
        return erf(z)

    def _standard_inflection(self) -> float:
        return 0.0


class Hill(Saturation):
    """``z^n / (1 + z^n)``: half the ceiling at ``K``, the log-logistic CDF (Hill 1910).

    S-shaped for ``slope > 1``, with inflection ``((n - 1)/(n + 1))^{1/n}`` and tangency
    ``(n - 1)^{1/n}``; :class:`MichaelisMenten` at ``slope = 1``. With a floor it is ADBUDG and
    Morgan--Mercer--Flodin.
    """

    scale: Array = eqx.field(converter=_real)
    slope: Array = eqx.field(converter=_real)

    def __check_init__(self) -> None:
        _require("slope", self.slope, 0.0)

    def standard(self, z: Array) -> Array:
        below = z <= 1.0
        # z^n below 1 and z^-n above it, so no power overflows; each branch sees 1 in the other's
        # range, where its slope is finite
        rising = _power(jnp.where(below, z, 1.0), self.slope)
        falling = jnp.where(below, 1.0, z) ** -self.slope
        return jnp.where(below, rising / (1.0 + rising), 1.0 / (1.0 + falling))

    def _standard_inflection(self) -> float:
        n = float(self.slope)
        return ((n - 1.0) / (n + 1.0)) ** (1.0 / n) if n > 1.0 else 0.0

    def _standard_tangency(self) -> float:
        n = float(self.slope)
        return (n - 1.0) ** (1.0 / n) if n > 1.0 else 0.0


class Weibull(Saturation):
    """``1 - exp(-z^k)``: the Weibull CDF, ``1 - 1/e`` of the ceiling at ``K``.

    S-shaped for ``shape > 1``, with inflection ``((k - 1)/k)^{1/k}``; the tangency is ``u^{1/k}``
    at the positive root of ``e^u - 1 = k u``. :class:`Exponential` at ``shape = 1``; with a floor,
    Janoschek's curve (1957).
    """

    scale: Array = eqx.field(converter=_real)
    shape: Array = eqx.field(converter=_real)

    def __check_init__(self) -> None:
        _require("shape", self.shape, 0.0)

    def standard(self, z: Array) -> Array:
        return -jnp.expm1(-_power(z, self.shape))

    def _standard_inflection(self) -> float:
        k = float(self.shape)
        return ((k - 1.0) / k) ** (1.0 / k) if k > 1.0 else 0.0

    def _standard_tangency(self) -> float:
        k = float(self.shape)
        return _lambert_root(k) ** (1.0 / k) if k > 1.0 else 0.0


class Logistic(Saturation):
    """The logistic ``1 / (1 + e^{-s (z - 1)})`` moved and stretched to run from 0 at zero spend to
    1, with its midpoint and inflection at ``K``. Always S-shaped."""

    scale: Array = eqx.field(converter=_real)
    steepness: Array = eqx.field(converter=_real)

    def __check_init__(self) -> None:
        _require("steepness", self.steepness, 0.0)

    def standard(self, z: Array) -> Array:
        s = self.steepness
        return (jax.nn.sigmoid(s * (z - 1.0)) - jax.nn.sigmoid(-s)) / jax.nn.sigmoid(s)

    def _standard_inflection(self) -> float:
        return 1.0


class Gompertz(Saturation):
    """Gompertz's ``exp(-b e^{-z})`` moved and stretched to run from 0 at zero spend to 1.

    ``b`` is the displacement: the inflection is at ``log b``, so the curve is S-shaped for
    ``b > 1`` and concave from zero otherwise.
    """

    scale: Array = eqx.field(converter=_real)
    displacement: Array = eqx.field(converter=_real)

    def __check_init__(self) -> None:
        _require("displacement", self.displacement, 0.0)

    def standard(self, z: Array) -> Array:
        b = self.displacement
        # exp(-b e^-z) - e^-b, written so that neither a small b nor a large z cancels
        return jnp.exp(-b * jnp.exp(-z)) * -jnp.expm1(b * jnp.expm1(-z)) / -jnp.expm1(-b)

    def _standard_inflection(self) -> float:
        return max(math.log(float(self.displacement)), 0.0)


class Richards(Saturation):
    """Richards' generalised logistic ``(1 + nu e^{-s (z - 1)})^{-1/nu}``, normalised to 0 at 0.

    The inflection is at ``K`` for every asymmetry ``nu``. :class:`Logistic` at ``nu = 1``; as
    ``nu -> 0`` it tends to :class:`Gompertz` with ``b = e^s`` and scale ``K / s``.
    """

    scale: Array = eqx.field(converter=_real)
    steepness: Array = eqx.field(converter=_real)
    asymmetry: Array = eqx.field(converter=_real)

    def __check_init__(self) -> None:
        _require("steepness", self.steepness, 0.0)
        _require("asymmetry", self.asymmetry, 0.0)

    def standard(self, z: Array) -> Array:
        s, nu = self.steepness, self.asymmetry

        def rising(at: Array) -> Array:
            return jnp.exp(-jnp.log1p(nu * jnp.exp(-s * (at - 1.0))) / nu)

        floor = rising(_real(0.0))
        return (rising(z) - floor) / (1.0 - floor)

    def _standard_inflection(self) -> float:
        return 1.0


class ChapmanRichards(Saturation):
    """``(1 - e^{-z})^p``, the Chapman--Richards growth curve.

    S-shaped for ``power > 1``, with inflection ``log p``; the tangency is the positive root of
    ``e^z - 1 = p z``. :class:`Exponential` at ``power = 1``.
    """

    scale: Array = eqx.field(converter=_real)
    power: Array = eqx.field(converter=_real)

    def __check_init__(self) -> None:
        _require("power", self.power, 0.0)

    def standard(self, z: Array) -> Array:
        return _power(-jnp.expm1(-z), self.power)

    def _standard_inflection(self) -> float:
        return max(math.log(float(self.power)), 0.0)

    def _standard_tangency(self) -> float:
        p = float(self.power)
        return _lambert_root(p) if p > 1.0 else 0.0


class GammaCDF(Saturation):
    """The gamma CDF with unit rate, ``P(a, z)``. S-shaped for ``shape > 1``, with inflection at
    the mode ``a - 1``; :class:`Exponential` at ``shape = 1``."""

    scale: Array = eqx.field(converter=_real)
    shape: Array = eqx.field(converter=_real)

    def __check_init__(self) -> None:
        _require("shape", self.shape, 0.0)

    def standard(self, z: Array) -> Array:
        return _gamma_cdf(self.shape, z)

    def _standard_inflection(self) -> float:
        return max(float(self.shape) - 1.0, 0.0)


class LogNormalCDF(Saturation):
    """``Phi(log z / sigma)``: the log-normal CDF with median ``K``. Always S-shaped, with
    inflection at the mode ``e^{-sigma^2}``."""

    scale: Array = eqx.field(converter=_real)
    sigma: Array = eqx.field(converter=_real)

    def __check_init__(self) -> None:
        _require("sigma", self.sigma, 0.0)

    def standard(self, z: Array) -> Array:
        positive = z > 0.0
        # the log at zero spend is -inf; the branch that reads it sees 1 instead
        return jnp.where(positive, ndtr(jnp.log(jnp.where(positive, z, 1.0)) / self.sigma), 0.0)

    def _standard_inflection(self) -> float:
        return math.exp(-(float(self.sigma) ** 2))


class BurrXII(Saturation):
    """``1 - (1 + z^c)^{-k}``, the Burr XII CDF, with ``slope`` ``c`` and ``tail`` ``k``.

    S-shaped for ``slope > 1``, with inflection ``((c - 1)/(c k + 1))^{1/c}``. :class:`Hill` at
    ``tail = 1``; the tail sets how slowly the last of the ceiling arrives, as ``z^{-c k}``.
    """

    scale: Array = eqx.field(converter=_real)
    slope: Array = eqx.field(converter=_real)
    tail: Array = eqx.field(converter=_real)

    def __check_init__(self) -> None:
        _require("slope", self.slope, 0.0)
        _require("tail", self.tail, 0.0)

    def standard(self, z: Array) -> Array:
        return -jnp.expm1(-self.tail * jnp.log1p(_power(z, self.slope)))

    def _standard_inflection(self) -> float:
        c, k = float(self.slope), float(self.tail)
        return ((c - 1.0) / (c * k + 1.0)) ** (1.0 / c) if c > 1.0 else 0.0


class BetaCDF(Saturation):
    """The beta CDF ``I_z(a, b)`` on ``[0, K]``, and the ceiling beyond ``K``.

    S-shaped for ``a > 1``, with inflection at the mode ``(a - 1)/(a + b - 2)``; at ``b = 1`` it is
    ``z^a``, convex up to ``K``, where it touches its envelope. ``b >= 1``, since below 1 the curve
    would rise ever faster into its ceiling. No derivative in ``a`` or ``b`` (JAX's ``betainc``
    has none); :class:`Kumaraswamy` has them.
    """

    scale: Array = eqx.field(converter=_real)
    a: Array = eqx.field(converter=_real)
    b: Array = eqx.field(converter=_real)

    _support: ClassVar[float] = 1.0

    def __check_init__(self) -> None:
        _require("a", self.a, 0.0)
        _require("b", self.b, 1.0, low_included=True)

    def standard(self, z: Array) -> Array:
        inside = z < 1.0
        return jnp.where(inside, _beta_cdf(self.a, self.b, jnp.where(inside, z, 0.5)), 1.0)

    def _standard_inflection(self) -> float:
        a, b = float(self.a), float(self.b)
        return (a - 1.0) / (a + b - 2.0) if a > 1.0 else 0.0


class Kumaraswamy(Saturation):
    """``1 - (1 - z^a)^b`` on ``[0, K]``, and the ceiling beyond ``K``: Kumaraswamy's closed-form
    counterpart of :class:`BetaCDF`.

    S-shaped for ``a > 1``, with inflection ``((a - 1)/(a b - 1))^{1/a}``; ``b >= 1``, as for the
    beta CDF.
    """

    scale: Array = eqx.field(converter=_real)
    a: Array = eqx.field(converter=_real)
    b: Array = eqx.field(converter=_real)

    _support: ClassVar[float] = 1.0

    def __check_init__(self) -> None:
        _require("a", self.a, 0.0)
        _require("b", self.b, 1.0, low_included=True)

    def standard(self, z: Array) -> Array:
        inside = z < 1.0
        at = jnp.where(inside, z, 0.5)
        return jnp.where(inside, -jnp.expm1(self.b * jnp.log1p(-_power(at, self.a))), 1.0)

    def _standard_inflection(self) -> float:
        a, b = float(self.a), float(self.b)
        return ((a - 1.0) / (a * b - 1.0)) ** (1.0 / a) if a > 1.0 else 0.0


class Logarithmic(Response):
    """``log(1 + z)``: concave and unbounded, a baseline."""

    scale: Array = eqx.field(converter=_real)

    def standard(self, z: Array) -> Array:
        return jnp.log1p(z)


class Power(Response):
    """``z^rho``, ``0 < rho <= 1``: constant elasticity ``rho``, concave and unbounded.
    PyMC-Marketing's ``RootSaturation``."""

    scale: Array = eqx.field(converter=_real)
    exponent: Array = eqx.field(converter=_real)

    def __check_init__(self) -> None:
        _require("exponent", self.exponent, 0.0, 1.0)

    def standard(self, z: Array) -> Array:
        return _power(z, self.exponent)


class Ricker(Response):
    """``z e^{1 - z}``: rises to its peak of 1 at ``K`` and falls after it, the inverted U of ad
    fatigue. Not monotone, so no envelope, and a plan's concavity arguments do not reach it."""

    scale: Array = eqx.field(converter=_real)

    def standard(self, z: Array) -> Array:
        return z * jnp.exp(1.0 - z)


def _length(length: int) -> None:
    if not (isinstance(length, int) and length >= 1):
        raise ValueError(f"length={length!r} is not a whole number of periods, at least 1")


class Adstock(eqx.Module):
    """A carryover kernel: ``a_t = sum_{l < L} w_l s_{t - l}``, the spend of the ``length`` periods
    up to a period weighted by lag, with nothing spent before the series.

    A period's adstock reads those ``L`` periods and no others, so appending periods to a series
    moves no earlier value, and ``normalized`` divides by the kernel's own sum, never the series'.
    It has no default: packages differ on it, and every coefficient fitted through the kernel moves
    with it.
    """

    length: eqx.AbstractVar[int]
    normalized: eqx.AbstractVar[bool]

    @abc.abstractmethod
    def _unnormalized(self) -> Array:
        """The ``(length,)`` weights before normalising, lag 0 first."""

    def weights(self) -> Array:
        """The ``(length,)`` weights, lag 0 first."""
        weights = self._unnormalized()
        return weights / jnp.sum(weights) if self.normalized else weights

    def __call__(self, spend: ArrayLike) -> Array:
        """The adstock of a ``(T,)`` spend series, or of each column of a ``(T, C)`` one."""
        series = jnp.asarray(spend, dtype=float)
        if series.ndim not in (1, 2):
            raise ValueError(f"spend has shape {series.shape}; a kernel reads a (T,) or (T, C) one")
        weights = self.weights()

        def carry(column: Array) -> Array:
            return jnp.convolve(column, weights)[: column.shape[0]]

        return carry(series) if series.ndim == 1 else jax.vmap(carry, 1, 1)(series)

    def __check_init__(self) -> None:
        _length(self.length)


class GeometricAdstock(Adstock):
    """``w_l = r^l``: each period keeps ``retention`` of the carryover it received."""

    retention: Array = eqx.field(converter=_real)
    length: int = eqx.field(static=True, kw_only=True)
    normalized: bool = eqx.field(static=True, kw_only=True)

    def __check_init__(self) -> None:
        _require("retention", self.retention, 0.0, 1.0, low_included=True)

    def _unnormalized(self) -> Array:
        # a running product, since the slope of r ** 0 at r = 0 is 0 * inf
        return jnp.cumprod(jnp.full(self.length, self.retention).at[0].set(1.0))


class DelayedAdstock(Adstock):
    """``w_l = r^{(l - d)^2}``: the carryover peaks ``delay`` periods after the spend (Jin, Wang,
    Sun, Chan and Koehler 2017)."""

    retention: Array = eqx.field(converter=_real)
    delay: Array = eqx.field(converter=_real)
    length: int = eqx.field(static=True, kw_only=True)
    normalized: bool = eqx.field(static=True, kw_only=True)

    def __check_init__(self) -> None:
        _require("retention", self.retention, 0.0, 1.0)
        _require("delay", self.delay, 0.0, low_included=True)

    def _unnormalized(self) -> Array:
        return self.retention ** ((jnp.arange(self.length, dtype=float) - self.delay) ** 2)


class WeibullAdstock(Adstock):
    """``w_0 = 1`` and ``w_l = prod_{j=1}^{l} (1 - F(j))``, ``F`` the Weibull CDF of ``shape`` and
    ``scale``: the share of the carryover that survives each lag. The scale is in periods, not a
    share of the series."""

    shape: Array = eqx.field(converter=_real)
    scale: Array = eqx.field(converter=_real)
    length: int = eqx.field(static=True, kw_only=True)
    normalized: bool = eqx.field(static=True, kw_only=True)

    def __check_init__(self) -> None:
        _require("shape", self.shape, 0.0)
        _require("scale", self.scale, 0.0)

    def _unnormalized(self) -> Array:
        # 1 - F(j) = exp(-(j / scale)^shape), so the product is the exponential of a running sum
        lags = jnp.arange(1, self.length, dtype=float)
        hazard = jnp.cumsum((lags / self.scale) ** self.shape)
        return jnp.exp(-jnp.concatenate([jnp.zeros(1), hazard]))


class Channel(eqx.Module):
    """A channel's return on its spend, ``coefficient * curve(kernel(spend))``: adstock first.

    ``coefficient`` is in the outcome's units per unit of the curve, so the return is in the
    outcome's units, revenue or a KPI.
    """

    kernel: Adstock
    curve: Response
    coefficient: Array = eqx.field(converter=_real)

    def __check_init__(self) -> None:
        _require("coefficient", self.coefficient, -math.inf)

    def __call__(self, spend: ArrayLike) -> Array:
        return self.coefficient * self.curve(self.kernel(spend))


def _series(spend: ArrayLike) -> Array:
    series = jnp.asarray(spend, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"spend has shape {series.shape}; a return is read one channel at a time")
    return series


def _window(spend: ArrayLike, window: slice) -> tuple[Array, Array, float]:
    series = _series(spend)
    inside = jnp.zeros_like(series).at[window].set(1.0)
    spent = float(jnp.sum(series * inside))
    if not spent > 0.0:
        raise ValueError(f"the window {window} spends {spent}; its return per unit is undefined")
    return series, inside, spent


def contribution(
    channel: Callable[[Array], Array],
    spend: ArrayLike,
    window: slice = slice(None),
    *,
    revenue_per_kpi: float = 1.0,
) -> float:
    """The channel's return in the window's periods, carried over from all the spend before them.

    What a decomposition attributes to the channel in those periods, in the outcome's units, or in
    revenue through ``revenue_per_kpi``. The return the window's own spend causes, wherever it
    falls, is instead :func:`roi` times what the window spent.
    """
    return revenue_per_kpi * float(jnp.sum(channel(_series(spend))[window]))


def roi(
    channel: Callable[[Array], Array],
    spend: ArrayLike,
    window: slice = slice(None),
    *,
    revenue_per_kpi: float = 1.0,
) -> float:
    """The incremental return on the window's spend, per currency unit, carryover included.

    The channel's return over the whole series, less its return with the window's spend removed,
    over what the window spent. ``channel`` is any function of a ``(T,)`` spend series returning a
    ``(T,)`` series of returns. The carryover counted is what falls inside the series: to count all
    of it, extend the series by the kernel's length with what is spent after.
    ``revenue_per_kpi`` turns a KPI into revenue; at 1 the outcome is revenue.
    """
    series, inside, spent = _window(spend, window)
    removed = series * (1.0 - inside)
    return revenue_per_kpi * float(jnp.sum(channel(series) - channel(removed))) / spent


def marginal_roi(
    channel: Callable[[Array], Array],
    spend: ArrayLike,
    window: slice = slice(None),
    *,
    revenue_per_kpi: float = 1.0,
) -> float:
    """The return on one more currency unit, spread over the window in proportion to its spend.

    The derivative of the channel's return over the whole series along the window's spend, over
    what the window spent: the limit of scaling the window's spend by ``1 + e``, where a step of
    finite size would move the number on a curved response. Carryover is counted as in
    :func:`roi`.
    """
    series, inside, spent = _window(spend, window)
    _, slope = jax.jvp(lambda path: jnp.sum(channel(path)), (series,), (series * inside,))
    return revenue_per_kpi * float(slope) / spent


def steady_state_marginal_roi(
    channel: Channel, level: float, *, revenue_per_kpi: float = 1.0
) -> float:
    """The return on one more currency unit a period, once ``level`` a period has been spent for
    longer than the kernel remembers.

    The adstock settles at ``level * sum(w)``, so a period's return is ``b h(level * sum(w))`` and
    its slope in the level ``b h'(level * sum(w)) sum(w)``: through a long unnormalised geometric
    kernel of retention ``r``, ``b h'(level / (1 - r)) / (1 - r)``.
    """
    _require("level", _real(level), 0.0, low_included=True)
    total = jnp.sum(channel.kernel.weights())
    slope = jax.grad(channel.curve)(_real(level) * total)
    return revenue_per_kpi * float(channel.coefficient * slope * total)
