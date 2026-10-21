"""Lift tests read as structure: a geo test's gap between its groups, period by period, identifies a
channel's carryover, the shape of its curve and its size, over the spend the test covered.
*Experimental.*

A geo test changes one channel's spend in one group of regions and leaves another group spending
as planned. The groups share everything else -- the season, the other channels, promotions, every
demand shock nobody observed -- so the difference of their outcomes is the channel's own::

    d_t = M(s_treated)_t - M(s_control)_t + nu_t,

``M`` the channel's return on a spend path (a :class:`chc.response.Channel`) and ``nu_t`` the
groups' own noise (Heusch 2026a). Every parameter of ``M`` is in it: the gap's rise when the spend
changes and its decay after carry the carryover, tests at different spend levels carry the curve's
shape, and the gap's size carries the coefficient. A test reduced to one number, its total lift
over its total spend, keeps only the last.

:func:`fit_lift` fits the equation by least squares over every period of every test, with one noise
scale across them, and gives each parameter a profile-likelihood interval with an ``F`` cutoff
(Bates and Watts 1988): exact for a model linear in its parameters and, unlike a Wald interval, it
follows a likelihood that curves. A parameter the tests do not pin runs to its bound, and its
interval says so: a curve the tests never bent has no upper scale.

The curve is identified over the adstock the tests covered, :attr:`LiftFit.tested_adstock`, and
nowhere else. Past it a plan reads the family's shape, not the experiment.

:func:`check_observational` reads a channel fitted to observational data, where spend follows the
business, against the tests a fit read: an ``F`` test of whether the tests' gaps could be the
channel's, and the factor of the lift it predicts that the tests read. Through an analyst's
effect-scale gap the factor gives the least marginal-sensitivity ``Gamma`` the tests leave the
observational channel, a floor under the confounding a plan assumes (De Bartolomeis et al. 2024
bound ``Gamma`` from below with a trial in the same way).

HONEST SCOPE:

* The noise is one scale, independent across tests, and by default across periods. A geo gap's
  noise persists from week to week, and read as independent it makes the intervals too narrow and
  the check reject too often: at an AR(1) coefficient of 0.8 the check rejected the true channel in
  30% of 200 worlds at 5%. ``serial="ar1"`` reads each test's noise as an AR(1) instead, and
  rejected 4.5% there. A gap read through a synthetic control or a regression carries an error
  common to its periods, which neither models.
* The groups spend alike on every other channel and answer alike, so a group's spend and outcome
  divided by its share of the market are the market's (Heusch's scaling).
* The check's ``F`` is the region the intervals are profiles of: exact for a model linear in its
  parameters, approximate on a curve. Its ``Gamma`` bounds the tested channel's confounding; read
  for a channel no test reached, it assumes the two are confounded alike.
* The arithmetic is JAX's, as for the library's other JAX estimators: float64 under
  ``jax_enable_x64``, and float32, good to about seven digits, without it.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.tree_util import PyTreeDef
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from chc.barrier import barrier_gamma_star
from chc.frames import _real_numbers
from chc.response import Channel

_Array = NDArray[np.float64]

_log = logging.getLogger(__name__)

# where a parameter lives, by its name, and whether it is fitted as its log; every other parameter
# of a curve or a kernel is a scale or a shape, positive, and is fitted as its log. A retention is
# too: a carryover that fades within a period trades with a curve that saturates at once, the two
# falling together, and in logs that ridge is a line a solver follows rather than a curve it crawls
_BOXES: dict[str, tuple[float, float, bool]] = {
    "retention": (0.0, 1.0, True),
    "delay": (0.0, math.inf, False),
    "exponent": (0.0, 1.0, False),
    "b": (1.0, math.inf, False),
    "coefficient": (-math.inf, math.inf, False),
}
# a logged parameter's profile stops this far from its estimate, a factor of e^20, and reads the
# rest as unbounded; the fit keeps it within twice that of where it started, so a ridge the gap
# cannot see along, a line's scale, never overflows
_REACH = 20.0
_WIDEN = 60  # doublings of the first step before a profile gives up on crossing the cutoff
_EVALUATIONS, _CONTINUATIONS = 2000, 5  # a solve's budget, and how often it may go on
# a point cheaper than the fit by less than this share of the cutoff moves no interval's end
# visibly, and is how a fit reads a least cost that lies on the way to a bound
_SLACK = 1e-6
# serial="ar1": the coefficient is re-read from the residuals and the gaps refitted until it moves
# less than this, at most this many rounds; and kept this far inside 1, where the transform holds
_AR_TOLERANCE, _AR_ROUNDS, _AR_BOUND = 1e-4, 20, 0.99


def _series(value: ArrayLike, name: str) -> _Array:
    array = np.array(_real_numbers(value, name), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty vector, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} has a non-finite entry")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class GeoArm:
    """One group of a geo test: the tested channel's spend and the group's outcome.
    *Experimental.*

    ``spend`` runs over the periods before the readout and then the readout, ``outcome`` over the
    readout. A period's adstock reads the spend of the kernel's length up to it, so the readout's
    first periods need the spend before them: a window cut from a longer series, with nothing
    before it, understates the adstock at its start (Heusch 2026b). ``share`` is the group's share
    of the market the channel is modelled at; its spend and outcome are divided by it.

    Raises:
        ValueError: on an empty or non-finite series, a negative spend, a spend shorter than the
            readout, or a share outside ``(0, 1]``.
    """

    spend: _Array
    outcome: _Array
    share: float = 1.0

    def __post_init__(self) -> None:
        spend = _series(self.spend, "spend")
        outcome = _series(self.outcome, "outcome")
        if np.any(spend < 0.0):
            raise ValueError("spend has a negative entry; a curve is defined on spend >= 0")
        if spend.size < outcome.size:
            raise ValueError(
                f"spend runs {spend.size} periods and the outcome {outcome.size}; the spend covers "
                "the readout and the periods before it"
            )
        if not (math.isfinite(self.share) and 0.0 < self.share <= 1.0):
            raise ValueError(f"share={self.share!r} is not a share of the market in (0, 1]")
        object.__setattr__(self, "spend", spend)
        object.__setattr__(self, "outcome", outcome)

    @property
    def history(self) -> int:
        """The periods of spend before the readout."""
        return self.spend.size - self.outcome.size


@dataclass(frozen=True)
class LiftTest:
    """A geo test of one channel: a treated group whose spend on it changed and a control group
    that spent as planned, alike on every other channel. *Experimental.*

    Raises:
        ValueError: when the groups' spend or readout cover different numbers of periods.
    """

    treated: GeoArm
    control: GeoArm

    def __post_init__(self) -> None:
        treated, control = self.treated, self.control
        if (treated.spend.size, treated.outcome.size) != (control.spend.size, control.outcome.size):
            raise ValueError(
                f"the treated group covers {treated.spend.size} periods of spend and "
                f"{treated.outcome.size} of readout, the control {control.spend.size} and "
                f"{control.outcome.size}; a test's groups cover the same periods"
            )

    @property
    def difference(self) -> _Array:
        """The treated group's outcome less the control's, each at the market's scale."""
        return self.treated.outcome / self.treated.share - self.control.outcome / self.control.share


@dataclass(frozen=True)
class LiftFit:
    """A channel fitted to lift tests' gaps, with an interval for each parameter.
    *Experimental.* See :func:`fit_lift`."""

    channel: Channel  # the fitted channel, of the template's families
    parameters: tuple[str, ...]  # each fitted number's place: "kernel.retention", "coefficient"
    estimate: _Array  # (p,)
    # (p,) each interval's ends; a parameter's bound where the tests do not close its interval, and
    # 0 or inf for a scale or a shape the profile could not bound within a factor of e^20
    lower: _Array
    upper: _Array
    level: float
    noise_sd: float  # of the gap in one period, at the market's scale
    dof: int  # periods read less parameters fitted
    # the adstock the readouts covered, both groups, at the fitted kernel: where the curve is
    # identified; past it a plan reads the family's shape
    tested_adstock: tuple[float, float]
    tests: tuple[LiftTest, ...] = field(repr=False)  # the tests fitted, which a check reads again
    # the AR(1) coefficient of each test's noise the fit read with serial="ar1", estimated; 0 with
    # serial="independent". noise_sd is then its innovations' sd
    ar_coefficient: float = 0.0
    # with serial="independent", the two-sided p-value of the residuals' lag-1 correlation within
    # each test (Breusch-Godfrey's t on the fit's Jacobian); below 0.05 the fit logged a WARNING
    # that names serial="ar1". None with serial="ar1", and nan where nothing is left to read
    serial_p_value: float | None = None

    def interval(self, parameter: str) -> tuple[float, float]:
        """``parameter``'s interval, by its name in :attr:`parameters`."""
        if parameter not in self.parameters:
            raise KeyError(f"{parameter!r} is not one of {self.parameters}")
        index = self.parameters.index(parameter)
        return float(self.lower[index]), float(self.upper[index])


@dataclass(frozen=True)
class _Layout:
    """A channel's parameters as one vector, each in the coordinate it is fitted in."""

    static: Channel  # the template with its parameters taken out
    treedef: PyTreeDef
    names: tuple[str, ...]
    logged: tuple[bool, ...]
    lower: _Array  # the box, in the fitted coordinates
    upper: _Array
    coefficient: int  # the coefficient's place, which the gap is linear in

    @classmethod
    def of(cls, channel: Channel) -> _Layout:
        parameters, static = eqx.partition(channel, eqx.is_inexact_array)
        paths, treedef = jax.tree_util.tree_flatten_with_path(parameters)
        names = tuple(".".join(str(getattr(key, "name", key)) for key in path) for path, _ in paths)
        boxes = [_BOXES.get(name.rsplit(".", 1)[-1], (0.0, math.inf, True)) for name in names]
        logged = tuple(box[2] for box in boxes)

        def fitted(end: float, log: bool) -> float:
            return (math.log(end) if end > 0.0 else -math.inf) if log else end

        lower = np.array([fitted(low, log) for low, _, log in boxes])
        upper = np.array([fitted(high, log) for _, high, log in boxes])
        return cls(static, treedef, names, logged, lower, upper, names.index("coefficient"))

    def coordinates(self, channel: Channel) -> _Array:
        """The template's values in the fitted coordinates."""
        values = np.array(
            [
                float(leaf)
                for leaf in jax.tree_util.tree_leaves(eqx.filter(channel, eqx.is_inexact_array))
            ]
        )
        for name, value, log in zip(self.names, values, self.logged, strict=True):
            if log and not value > 0.0:
                raise ValueError(
                    f"the template's {name} is {value}; it is fitted as its log, so the fit "
                    "cannot start there: give a positive value"
                )
        return np.where(self.logged, np.log(np.where(self.logged, values, 1.0)), values)

    def natural(self, z: _Array) -> _Array:
        values = np.array(z, dtype=np.float64)
        logged = np.array(self.logged)
        values[logged] = np.exp(values[logged])
        return values

    def channel(self, z: ArrayLike) -> Channel:
        return _channel(jnp.asarray(z), self.static, self.treedef, self.logged)


def _channel(z: Array, static: Channel, treedef: PyTreeDef, logged: tuple[bool, ...]) -> Channel:
    values = jnp.where(jnp.array(logged), jnp.exp(z), z)
    parameters = jax.tree_util.tree_unflatten(treedef, [values[i] for i in range(len(logged))])
    return eqx.combine(parameters, static)


Readouts = tuple[tuple[Array, Array, Array], ...]  # (treated spend, control spend, gap) per test


def _shape_of(
    z: Array,
    static: Channel,
    treedef: PyTreeDef,
    logged: tuple[bool, ...],
    readouts: Readouts,
    coefficient: int,
    rho: Array,
) -> Array:
    """The gaps a coefficient of 1 predicts, each test's whitened at ``rho``, stacked."""
    channel = _channel(z.at[coefficient].set(1.0), static, treedef, logged)
    return jnp.concatenate(
        [
            _whitened((channel(treated) - channel(control))[treated.shape[0] - gap.shape[0] :], rho)
            for treated, control, gap in readouts
        ]
    )


def _whitened(segment: Array, rho: Array) -> Array:
    """One test's periods through the Prais-Winsten transform: an AR(1) of ``rho`` to its
    innovations, the first period scaled to their variance. The identity at ``rho = 0``."""
    return jnp.concatenate([jnp.sqrt(1.0 - rho**2) * segment[:1], segment[1:] - rho * segment[:-1]])


def _gaps(readouts: Readouts, rho: Array) -> Array:
    return jnp.concatenate([_whitened(gap, rho) for *_, gap in readouts])


def _projected(shape: Array, gap: Array) -> Array:
    """The least-squares coefficient of ``gap`` on ``shape``, 0 where the shape is flat."""
    size = shape @ shape
    positive = size > 0.0
    return jnp.where(positive, (shape @ gap) / jnp.where(positive, size, 1.0), 0.0)


def _residual_of(
    z: Array,
    static: Channel,
    treedef: PyTreeDef,
    logged: tuple[bool, ...],
    readouts: Readouts,
    coefficient: int,
    rho: Array,
    project: bool,
) -> Array:
    """The gaps less the channel's, whitened at ``rho``, the coefficient at its least-squares value
    when ``project``."""
    shape = _shape_of(z, static, treedef, logged, readouts, coefficient, rho)
    gap = _gaps(readouts, rho)
    size = _projected(shape, gap) if project else z[coefficient]
    return gap - size * shape


def _coefficient_of(
    z: Array,
    static: Channel,
    treedef: PyTreeDef,
    logged: tuple[bool, ...],
    readouts: Readouts,
    coefficient: int,
    rho: Array,
) -> Array:
    shape = _shape_of(z, static, treedef, logged, readouts, coefficient, rho)
    return _projected(shape, _gaps(readouts, rho))


_residual = eqx.filter_jit(_residual_of)
_jacobian = eqx.filter_jit(jax.jacfwd(_residual_of))
_coefficient = eqx.filter_jit(_coefficient_of)


class _Problem:
    """The stacked least squares, and its profile in one parameter.

    The coefficient is never searched: the gap is linear in it, so wherever it is free it takes its
    least-squares value given the rest (variable projection, Golub and Pereyra 1973), and a line's
    ridge, where scale and coefficient grow together, is no ridge in what is left.
    """

    def __init__(
        self, layout: _Layout, readouts: Readouts, origin: _Array, rho: float = 0.0
    ) -> None:
        self.layout = layout
        self.readouts = readouts
        self.rho = jnp.asarray(rho, dtype=readouts[0][2].dtype)
        self.tolerance = 1e3 * float(jnp.finfo(readouts[0][2].dtype).eps)
        logged = np.array(layout.logged)
        self.lower = np.where(logged, np.maximum(origin - 2.0 * _REACH, layout.lower), layout.lower)
        self.upper = np.where(logged, np.minimum(origin + 2.0 * _REACH, layout.upper), layout.upper)

    def _arguments(self) -> tuple[Channel, PyTreeDef, tuple[bool, ...], Readouts, int, Array]:
        layout = self.layout
        return (
            layout.static,
            layout.treedef,
            layout.logged,
            self.readouts,
            layout.coefficient,
            self.rho,
        )

    def residual(self, z: _Array, project: bool = False) -> _Array:
        value = _residual(jnp.asarray(z), *self._arguments(), project)
        return np.asarray(value, dtype=np.float64)

    def jacobian(self, z: _Array, project: bool = False) -> _Array:
        jacobian = np.asarray(_jacobian(jnp.asarray(z), *self._arguments(), project), np.float64)
        if not np.all(np.isfinite(jacobian)):
            bad = [self.layout.names[j] for j in np.flatnonzero(~np.isfinite(jacobian).all(axis=0))]
            raise ValueError(
                f"the gap's slope in {bad} is not finite at {self.layout.natural(z)}; the family's "
                "derivative fails at a spend the tests read"
            )
        return jacobian

    def solve(self, start: _Array, fixed: int | None = None, value: float = 0.0) -> _Array:
        """The least-squares point from ``start``, with parameter ``fixed`` held at ``value``."""
        coefficient = self.layout.coefficient
        project = fixed != coefficient
        held = {fixed, coefficient} if project else {fixed}
        free = np.array([j for j in range(start.size) if j not in held], dtype=int)

        def full(rest: _Array) -> _Array:
            z = start.copy()
            z[free] = rest
            if fixed is not None:
                z[fixed] = value
            return z

        z = full(np.clip(start[free], self.lower[free], self.upper[free]))
        # a solve that spends its evaluations goes on from where it stopped: where the least cost
        # lies at a coordinate's infinity, the Jacobian's scaling holds the trust region small on
        # the way there, and a fresh start resets it
        for _ in range(_CONTINUATIONS if free.size else 0):
            solution = optimize.least_squares(
                lambda r: self.residual(full(r), project),
                z[free],
                jac=lambda r: self.jacobian(full(r), project)[:, free],
                bounds=(self.lower[free], self.upper[free]),
                method="trf",
                x_scale="jac",
                ftol=self.tolerance,
                xtol=self.tolerance,
                gtol=self.tolerance,
                max_nfev=_EVALUATIONS,
            )
            z = full(solution.x)
            if solution.status != 0:
                break
        else:
            if free.size:
                held_at = "" if fixed is None else f" with {self.layout.names[fixed]} held"
                raise RuntimeError(
                    f"the least squares{held_at} spent {_CONTINUATIONS * _EVALUATIONS} evaluations "
                    f"without converging, at {self.layout.natural(z)}"
                )
        if project:
            z[coefficient] = float(_coefficient(jnp.asarray(z), *self._arguments()))
        return z

    def cost(self, z: _Array) -> float:
        return float(np.sum(self.residual(z) ** 2))


def _end(
    problem: _Problem, best: _Array, index: int, side: float, cutoff: float, step: float
) -> tuple[float, _Array | None]:
    """How far parameter ``index`` goes on ``side`` before the profile's cost rises by ``cutoff``,
    or a cheaper point than ``best`` when the walk finds one."""
    floor = problem.cost(best)
    own = problem.layout.upper[index] if side > 0 else problem.layout.lower[index]
    if math.isfinite(own):  # the parameter's own end, which an interval reaching it ends at
        bound, unbounded = own, False
    elif problem.layout.logged[index]:  # a factor of e^20, read as unbounded
        edge = problem.upper[index] if side > 0 else problem.lower[index]
        bound, unbounded = best[index] + side * min(_REACH, abs(edge - best[index])), True
    else:
        bound, unbounded = side * math.inf, True
    inside, start = best[index], best
    for doubling in range(_WIDEN):
        value = best[index] + side * step * 2.0**doubling
        at_bound = (value - bound) * side >= 0.0
        value = bound if at_bound else value
        point = problem.solve(start, index, value)
        rise = problem.cost(point) - floor
        if rise < -_SLACK * cutoff:
            return math.nan, point
        if rise > cutoff:
            break
        if at_bound:
            return (side * math.inf if unbounded else bound), None
        inside, start = value, point
    else:
        return side * math.inf, None

    def excess(position: float) -> float:
        return problem.cost(problem.solve(start, index, position)) - floor - cutoff

    # relative to the end itself: a coefficient's walk can bracket it from far away
    tolerance = problem.tolerance
    return optimize.brentq(excess, inside, value, xtol=tolerance, rtol=tolerance), None


def fit_lift(
    tests: Sequence[LiftTest],
    channel: Channel,
    *,
    level: float = 0.95,
    serial: Literal["independent", "ar1"] | None = None,
) -> LiftFit:
    """Fit ``channel`` to the tests' gaps by least squares, each parameter with a profile-likelihood
    interval. *Experimental.*

    ``channel`` names the families, its kernel and its curve, and its kernel's and curve's values
    are where the fit starts, so give a scale near the spend the tests saw. Every parameter is
    fitted: a coefficient of either sign, so a channel that loses sales and one that does nothing
    are both answers; a retention in ``(0, 1]`` and each scale and shape as its logarithm, within
    a factor of ``e^40`` of its start. The coefficient is never searched: the gap is linear in it,
    so wherever the rest moves it takes its least-squares value there (variable projection, Golub
    and Pereyra 1973), and the template's own is ignored. The periods of every test are stacked,
    with one noise variance, estimated.

    ``serial`` is how a test's noise runs over its periods. ``"independent"``, the default, reads
    each period's alone. A geo gap's noise persists from week to week, and read as independent it
    makes every interval too narrow: on go-dark tests whose noise was an AR(1) of coefficient 0.8,
    the coefficient's 95% interval held the truth in 88.5% of 200 worlds, and
    :func:`check_observational` rejected the true channel in 30% at 5%. ``"ar1"`` reads each test's
    noise as ``nu_t = rho nu_{t-1} + e_t``, ``e`` i.i.d. and one ``rho`` for every test, and fits
    the gaps through the Prais-Winsten transform, which at the true ``rho`` is the independent
    problem above (generalised least squares; Prais and Winsten 1954). ``rho`` is the least-squares
    AR(1) coefficient of the residuals, each test's periods on their previous one, read again after
    each refit until it settles (iterated Cochrane-Orcutt); it costs a degree of freedom, and the
    intervals and the check take it as known. On those worlds it read 0.78 at 0.8 (sd 0.08), the
    interval held the truth in 95.5% and the check rejected 4.5%; at 0.5, 98.5% and 5.0%; on
    independent noise, 96.5% and 6.0%. The estimate reads low, as a residual's lag-1 correlation
    does: the fit has spent degrees of freedom on the residuals, and a test's run is short. Over
    another 200 such worlds it read -0.046, 0.286, 0.475 and 0.777 at 0, 0.3, 0.5 and 0.8 (Monte
    Carlo SE 0.006 to 0.010), and the coefficient's interval held the truth in 95.0 to 98.0%. It
    is not corrected: Kendall's ``(1 + 3 rho) / T`` for a series' own mean would add 0.21 at 0.8
    where it reads 0.023 low, and the intervals above already carry the bias. Each test needs a
    run of periods for this to read anything: those tests had 16. :attr:`LiftFit.ar_coefficient`
    is the estimate and :attr:`LiftFit.noise_sd` the innovations'.

    With ``"independent"`` the fit tests the premise it reads: the residuals' lag-1 correlation
    within each test, by Breusch and Godfrey's regression of the residuals on the fit's Jacobian
    and their own value a period earlier (Breusch 1978; Godfrey 1978), its ``t`` two-sided against
    ``t(n - p - 1)``. :attr:`LiftFit.serial_p_value` holds the p-value, and below 0.05 the fit
    logs a WARNING that names ``serial="ar1"``. On 200 worlds of four go-dark tests of 16 periods it
    rejected 5.5 % of independent noise, and 51 %, 93.5 % and 100 % of AR(1) noise at 0.3, 0.5 and
    0.8, where the independent reading's check rejected the true channel on 11 %, 17 % and 30 %.

    ``serial`` left at None reads ``"independent"`` and warns (``FutureWarning``) that 0.20.0
    reads ``"ar1"``: on those worlds' independent noise, ``"ar1"``'s check rejected the true
    channel on 6.0 % and the coefficient's interval covered 96.5 %, so it holds where the noise
    does not persist, at a degree of freedom. Pass ``serial`` to keep either reading quietly.

    Each interval holds the values whose profile, the least cost with that value held, lies within
    ``s^2 F_{1, n - p}(level)`` of the least cost, ``s^2`` the noise variance over ``n - p`` degrees
    of freedom. A walk out from the estimate finds each end and a root-finder sharpens it; an end
    the tests leave open is the parameter's bound, or ``0`` or ``inf`` for a retention, a scale or
    a shape that moves a factor of ``e^20`` without reaching the cutoff. Should the walk find a
    point cheaper than the fit by a millionth of the cutoff, the fit restarts from there.

    Where the least cost lies on the way to a bound, the estimate is where the fit stopped on that
    way. Tests that never bent the curve are fitted best by a line, which a saturating family
    reaches only as its scale and coefficient grow without end; noise the size of the effect can be
    fitted best by a step, a curve saturated at any spend with a carryover gone within a period,
    which the family reaches as the scale and the retention fall together. The estimate predicts
    the gaps as the limit does, and its interval, open on the limit's side, is the answer about the
    parameter.

    Raises:
        TypeError: when ``channel`` is not a :class:`chc.response.Channel`.
        ValueError: on no tests, ``level`` outside ``(0, 1)``, ``serial`` neither
            ``"independent"`` nor ``"ar1"``, a test with fewer periods of spend
            before its readout than the kernel reads, fewer periods than parameters, a template
            whose retention, scale or shape is not positive, or a gap whose slope in a parameter
            is not finite.
        RuntimeError: when a solve does not converge within 10 000 evaluations, or the profile
            keeps finding points cheaper than the fit.
    """
    if not isinstance(channel, Channel):
        raise TypeError(
            f"channel is a {type(channel).__name__}; fit_lift fits a chc.response.Channel"
        )
    tests = tuple(tests)
    if not tests:
        raise ValueError("fit_lift needs at least one test")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    if serial is None:
        warnings.warn(
            "fit_lift reads each test's noise as independent until 0.20.0, which reads it as an "
            'AR(1) (serial="ar1"); pass serial="independent" to keep this reading, or "ar1" now',
            FutureWarning,
            stacklevel=2,
        )
        serial = "independent"
    if serial not in ("independent", "ar1"):
        raise ValueError(f"serial must be 'independent' or 'ar1', got {serial!r}")
    reach = channel.kernel.length - 1
    for position, test in enumerate(tests):
        if test.treated.history < reach:
            raise ValueError(
                f"test {position} carries {test.treated.history} periods of spend before its "
                f"readout and the kernel reads {reach}; its first periods' adstock would miss "
                "spend it never saw"
            )
    layout = _Layout.of(channel)
    readouts = tuple(
        (
            jnp.asarray(test.treated.spend / test.treated.share),
            jnp.asarray(test.control.spend / test.control.share),
            jnp.asarray(test.difference),
        )
        for test in tests
    )
    start = np.clip(layout.coordinates(channel), layout.lower, layout.upper)
    problem = _Problem(layout, readouts, start)
    periods = sum(test.treated.outcome.size for test in tests)
    dof = periods - len(layout.names) - (serial == "ar1")
    if dof < 1:
        raise ValueError(
            f"{periods} periods cannot fit {len(layout.names)} parameters and a noise variance"
            + (" and an AR(1) coefficient" if serial == "ar1" else "")
        )
    best = problem.solve(start)
    rho = 0.0
    if serial == "ar1":
        for _ in range(_AR_ROUNDS):
            estimate = _ar_coefficient(layout.channel(best), tests)
            if abs(estimate - rho) <= _AR_TOLERANCE:
                break
            rho = estimate
            problem = _Problem(layout, readouts, start, rho)
            # the last fit can sit on a ridge, a line's, that is a least cost of the independent
            # gaps and not of the whitened ones, which a solve from it never leaves
            best = min(problem.solve(start), problem.solve(best), key=problem.cost)
    for _ in range(8):
        variance = problem.cost(best) / dof
        cutoff = variance * float(stats.f.ppf(level, 1, dof))
        steps = _steps(problem, best, variance)
        ends, cheaper = [], None
        for index in range(best.size):
            for side in (-1.0, 1.0):
                end, cheaper = _end(problem, best, index, side, cutoff, steps[index])
                if cheaper is not None:
                    break
                ends.append(end)
            if cheaper is not None:
                break
        if cheaper is None:
            break
        best = problem.solve(cheaper)
    else:
        raise RuntimeError(
            "the profile kept finding cheaper points than the fit; it did not settle"
        )
    lower = layout.natural(np.array(ends[0::2]))
    upper = layout.natural(np.array(ends[1::2]))
    fitted = layout.channel(best)
    serial_p = None
    if serial == "independent":
        serial_p = _serial_p_value(problem, best, tests)
        if serial_p < 0.05:
            _log.warning(
                "fit_lift read each test's noise as independent, and its residuals are correlated "
                "from one period to the next (p = %.3g): the intervals are too narrow and "
                'check_observational rejects too often; refit with serial="ar1"',
                serial_p,
            )
    return LiftFit(
        channel=fitted,
        parameters=layout.names,
        estimate=layout.natural(best),
        lower=lower,
        upper=upper,
        level=level,
        noise_sd=math.sqrt(variance),
        dof=dof,
        tested_adstock=_tested_adstock(fitted, tests),
        tests=tests,
        ar_coefficient=rho,
        serial_p_value=serial_p,
    )


def _serial_p_value(problem: _Problem, best: _Array, tests: tuple[LiftTest, ...]) -> float:
    """Breusch-Godfrey's test of the residuals' lag-1 correlation within each test: the ``t`` of
    the lagged residual, 0 at a test's first period, in the least squares of the residuals on it
    and the fit's Jacobian, two-sided against ``t(n - p - 1)``; nan where nothing is left to
    read."""
    residual = problem.residual(best)
    lagged, at = np.zeros_like(residual), 0
    for test in tests:
        size = test.treated.outcome.size
        lagged[at + 1 : at + size] = residual[at : at + size - 1]
        at += size
    design = np.column_stack([problem.jacobian(best), lagged])
    inverse = np.linalg.pinv(design.T @ design)
    slope = (inverse @ design.T @ residual)[-1]
    left = residual - design @ (inverse @ design.T @ residual)
    dof = residual.size - np.linalg.matrix_rank(design)
    spread = float(left @ left) / dof * inverse[-1, -1] if dof > 0 else 0.0
    if not spread > 0.0:
        return math.nan
    return float(2.0 * stats.t.sf(abs(slope) / math.sqrt(spread), dof))


def _steps(problem: _Problem, best: _Array, variance: float) -> _Array:
    """Each parameter's first step out: its Wald standard error where the Jacobian gives one, at
    most a factor of ``e`` for a logged parameter and its box's width for a boxed one."""
    layout = problem.layout
    jacobian = problem.jacobian(best)
    try:
        errors = np.sqrt(
            np.clip(np.diag(np.linalg.pinv(jacobian.T @ jacobian)) * variance, 0.0, None)
        )
    except np.linalg.LinAlgError:
        errors = np.full(best.size, math.nan)
    width = layout.upper - layout.lower
    ceiling = np.where(layout.logged, 1.0, np.where(np.isfinite(width), width, math.inf))
    fallback = np.where(np.isfinite(ceiling), 0.1 * ceiling, 0.1 * np.maximum(np.abs(best), 1.0))
    usable = np.isfinite(errors) & (errors > 0.0)
    return np.where(usable, np.minimum(errors, ceiling), fallback)


def _ar_coefficient(channel: Channel, tests: tuple[LiftTest, ...]) -> float:
    """The least-squares AR(1) coefficient of ``channel``'s residuals, each test's periods on their
    previous one, pooled over the tests; kept within ``_AR_BOUND`` of 0."""
    residuals = [
        test.difference - segment
        for test, segment in zip(tests, _predicted_by_test(channel, tests), strict=True)
    ]
    cross = sum(float(r[1:] @ r[:-1]) for r in residuals)
    lagged = sum(float(r[:-1] @ r[:-1]) for r in residuals)
    if not lagged > 0.0:
        return 0.0
    return min(max(cross / lagged, -_AR_BOUND), _AR_BOUND)


def _tested_adstock(channel: Channel, tests: tuple[LiftTest, ...]) -> tuple[float, float]:
    reached = [
        np.asarray(channel.kernel(arm.spend / arm.share))[arm.history :]
        for test in tests
        for arm in (test.treated, test.control)
    ]
    covered = np.concatenate(reached)
    return float(np.min(covered)), float(np.max(covered))


@dataclass(frozen=True)
class ObservationalCheck:
    """An observational channel read against the lift tests a fit read. *Experimental.* See
    :func:`check_observational`."""

    statistic: float  # F: the channel's least squares over the fit's, per parameter, over s^2
    p_value: float
    dof: tuple[int, int]  # the F's: the fit's parameters, and its periods less them
    # the multiple of the gaps the channel predicts that the tests read, and its t interval at
    # level; nan where the channel predicts no gap
    factor: float
    factor_interval: tuple[float, float]
    level: float

    @property
    def rejected(self) -> bool:
        """Whether the tests reject the channel at :attr:`level`."""
        return self.p_value < 1.0 - self.level

    def least_gamma(self, cvar_gap: float) -> float:
        """The least marginal-sensitivity ``Gamma`` whose identified set reaches the tests.

        In units of the lift the channel predicts, the set is ``1 ± (Gamma-1)/(Gamma+1) cvar_gap``,
        ``chc.research.sensitivity``'s radius, ``cvar_gap`` the analyst's effect-scale CVaR gap as a
        share of that lift. The least ``Gamma`` whose set reaches :attr:`factor_interval` inverts
        the radius at the interval's distance from 1, as :func:`chc.barrier.barrier_gamma_star`
        does. Less confounding than this the tests refute at :attr:`level`, as a known-null outcome
        refutes it in :func:`chc.uncertainty.negative_control_gamma` given a ``level``: a floor, not
        a ceiling.

        ``1.0`` when the interval holds 1; ``inf`` when it lies ``cvar_gap`` or further from 1, so
        that no level reconciles the two and the tests refute the model rather than calibrate it;
        ``nan`` when the channel predicts no gap.

        Raises:
            ValueError: when ``cvar_gap`` is not positive and finite.
        """
        if not (math.isfinite(cvar_gap) and cvar_gap > 0.0):
            raise ValueError(f"cvar_gap must be positive and finite, got {cvar_gap}")
        if math.isnan(self.factor):
            return math.nan
        lower, upper = self.factor_interval
        return barrier_gamma_star(max(lower - 1.0, 1.0 - upper, 0.0), cvar_gap, 1.0)


def check_observational(fit: LiftFit, observed: Channel) -> ObservationalCheck:
    """Read ``observed``, a channel fitted to observational data, against the tests ``fit`` read.
    *Experimental.*

    Two readings of the tests' gaps ``d``, at the fit's level and with its noise variance ``s^2``
    over ``n - p`` degrees of freedom; of a fit read with ``serial="ar1"``, the gaps, and the ones
    ``observed`` predicts, through the fit's Prais-Winsten transform, at its
    :attr:`LiftFit.ar_coefficient`:

    * **An ``F`` test** of ``observed`` as a point of the fit's model: the least squares of the gaps
      it predicts over the fit's, per parameter, over ``s^2``, against ``F_{p, n-p}`` (Bates and
      Watts 1988), the joint region the fit's intervals are profiles of.
    * **The factor**, the least-squares multiple of the gaps it predicts, ``g``, that the tests
      read: ``c = g.d / g.g``, with the ``t`` interval ``c ± t s / |g|``. It is the tested lift over
      the lift ``observed`` predicts, each period weighted by what it predicts there, so tests of
      either sign add rather than cancel; ``1`` when the lift is the channel's.

    The ``F`` rejects a channel whose shape the tests contradict even where its size is right; the
    factor says how far its size is off, and :meth:`ObservationalCheck.least_gamma` turns that into
    a floor under the confounding. Where spend follows the business an observational fit reads the
    business's decisions as the channel's effect, and the tests are what can say so: on Heusch's
    generator it reads paid shopping's return at 8.45 against a true 4.20, with an oracle's
    controls (Heusch 2026a).

    Raises:
        TypeError: when ``observed`` is not a :class:`chc.response.Channel`.
        ValueError: when ``observed`` is not of the fit's families, kernel length and form, so the
            fit's model does not hold it; when it predicts a gap that is not finite; when the fit
            reproduces the tests exactly, leaving no noise to test against; or when ``observed``
            fits the tests better than the fit by more than the fit tells apart, so the fit is not
            their least squares, and a fit from it, ``fit_lift(fit.tests, observed)``, is due.
    """
    if not isinstance(observed, Channel):
        raise TypeError(
            f"observed is a {type(observed).__name__}; the check reads a chc.response.Channel"
        )
    fitted, fitted_static = eqx.partition(fit.channel, eqx.is_inexact_array)
    parameters, static = eqx.partition(observed, eqx.is_inexact_array)
    same = jax.tree_util.tree_structure(parameters) == jax.tree_util.tree_structure(fitted)
    if not (same and eqx.tree_equal(static, fitted_static)):
        raise ValueError(
            "the observed channel is not of the fit's families, kernel length and form; the check "
            "tests it as a point of the fit's model, so fit the tests from its template"
        )
    rho = fit.ar_coefficient
    gap = np.concatenate([_whitened_numpy(test.difference, rho) for test in fit.tests])
    predicted = _predicted(observed, fit.tests, rho)
    if not np.all(np.isfinite(predicted)):
        raise ValueError("the observed channel predicts a gap that is not finite on the tests")
    least = float(np.sum((gap - _predicted(fit.channel, fit.tests, rho)) ** 2))
    cost = float(np.sum((gap - predicted) ** 2))
    parameter_count, dof = len(fit.parameters), fit.dof
    variance = least / dof
    if not variance > 0.0:
        raise ValueError("the fit reproduces the tests exactly; there is no noise to test against")
    excess = cost - least
    if excess < -_SLACK * variance * float(stats.f.ppf(fit.level, 1, dof)):
        raise ValueError(
            f"the observed channel fits the tests better than the fit, by {-excess:.6g} against "
            f"its least squares {least:.6g}, so the fit is not their least squares: fit from it, "
            "fit_lift(fit.tests, observed)"
        )
    statistic = max(excess, 0.0) / parameter_count / variance
    size = float(predicted @ predicted)
    if size > 0.0:
        factor = float(predicted @ gap) / size
        half = float(stats.t.ppf((1.0 + fit.level) / 2.0, dof)) * math.sqrt(variance / size)
        interval = (factor - half, factor + half)
    else:
        factor, interval = math.nan, (math.nan, math.nan)
    return ObservationalCheck(
        statistic=statistic,
        p_value=float(stats.f.sf(statistic, parameter_count, dof)),
        dof=(parameter_count, dof),
        factor=factor,
        factor_interval=interval,
        level=fit.level,
    )


def _predicted_by_test(channel: Channel, tests: tuple[LiftTest, ...]) -> list[_Array]:
    """The gaps ``channel`` predicts, test by test, at the market's scale."""
    return [
        np.asarray(
            channel(test.treated.spend / test.treated.share)
            - channel(test.control.spend / test.control.share),
            dtype=np.float64,
        )[test.treated.history :]
        for test in tests
    ]


def _predicted(channel: Channel, tests: tuple[LiftTest, ...], rho: float = 0.0) -> _Array:
    """The gaps ``channel`` predicts, each test's whitened at ``rho``, stacked over the tests."""
    return np.concatenate(
        [_whitened_numpy(segment, rho) for segment in _predicted_by_test(channel, tests)]
    )


def _whitened_numpy(segment: _Array, rho: float) -> _Array:
    """:func:`_whitened` in NumPy, for the check's arithmetic."""
    return np.concatenate([math.sqrt(1.0 - rho**2) * segment[:1], segment[1:] - rho * segment[:-1]])
