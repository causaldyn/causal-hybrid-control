"""Whether a plan pays for its model class being wrong: two fits of one class on one log, their
difference priced in the plan's own regret. *Experimental.*

Under a class that holds the truth, every weight on the state estimates the same parameters, and
an instrument estimates what an adjustment set does; only the variance moves. Under a class that
misses the truth, each weight estimates its own projection (see ``weights`` in
:func:`chc.dynamics_id.fit_causal_residual`), so how far two fits land apart is a symptom of the
miss. Whether it matters is the plan's question: :func:`misspecification_cost` prices the
difference in the regret of planning on one fit when the other's model holds, and tests it against
the law it has when the class holds -- a Hausman test in the plan's metric, whose power lies where
the plan's cost does.

**The price is over every parameter, the drift's included.** The drift is fitted on what the
channel leaves of the rate, so two fits whose channels differ by ``dB`` have drifts that differ by
the log's projection of ``dB u`` on the drift's features: their models differ by
``dB (u - uhat(x))``, with ``uhat`` the log's action projected on those features, and not by
``dB u``. :meth:`chc.plan.CausalPlan.decision_weight` prices the channel alone. On logs whose
actions averaged -1, 0 and +1 it read 0.14, 0.44 and 2.8 of the regret between the two fits'
plans, where this module's quadratic read 0.49, 0.54 and 0.49
(``scripts/bench_misspecification.py``, section ``metric``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from jax import Array
from numpy.typing import NDArray
from scipy import integrate

from chc.cost import total_cost
from chc.dynamics import HybridDynamics
from chc.dynamics_id import CausalDynamicsFit, _clustered_squares
from chc.plan import CausalPlan, _regret_curvature, _weighable
from chc.residual import ControlAffineResidual


@dataclass(frozen=True)
class MisspecificationCost:
    """What the choice between two fits of one class costs a plan. *Experimental.*

    See :func:`misspecification_cost`.
    """

    # (d' W d - tr(W S)) / 2: planning on the reference fit when the alternative's model holds, to
    # second order, the noise's share taken out; negative when the noise outweighs the difference
    cost: float
    cost_error: float  # its standard error: a scale, not coverage
    noise: float  # tr(W S) / 2: what d' W d / 2 reads on average when the class holds the truth
    p_value: float  # the chance d' W d reads at least this large when the class holds the truth
    # directions the log's actions never moved (CausalDynamicsFit.unmoved) that the plan's regret
    # weighs: neither fit's moment has data there, so neither the cost nor the p-value can see a
    # miss along them, and a pass says nothing about them
    unseen: int


def misspecification_cost(
    plan: CausalPlan,
    reference: CausalDynamicsFit,
    alternative: CausalDynamicsFit,
    *,
    tolerance: float | None = None,
) -> MisspecificationCost:
    """What choosing between two fits of one class of one log costs ``plan``, made on
    ``reference``'s model: estimated, with a standard error, and tested against the class holding
    the truth. *Experimental.*

    With ``d`` the fits' difference in their parameters, the channel's and the drift's,
    ``alternative``'s less ``reference``'s, ``S`` its covariance and ``W`` the regret's curvature in
    the parameters at the plan, planning on ``reference`` when ``alternative``'s model holds loses
    ``d' W d / 2`` to second order. ``W`` comes from :meth:`chc.plan.CausalPlan.decision_weight`'s
    envelope, with the model perturbed by a residual of the fits' class inside its field, rolled
    out by RK4 as the plan is. Since ``E[dh' W dh] = d' W d + tr(W S)``, the cost is read as
    ``(dh' W dh - tr(W S)) / 2``, with :meth:`chc.gate.ChannelMove.price`'s standard error. When
    the class holds the truth, ``d = 0`` and ``dh' W dh`` is a sum of chi-squares weighted by the
    eigenvalues of ``W S``, whose tail Imhof's inversion gives: that is ``p_value``.

    ``S`` sums, row by row and state by state, the outer products of the difference between the
    two fits' influences, each with its own score, so it holds whether the class does or not, and
    whichever folds each fit drew; for fits given ``clusters``, it sums each cluster's difference
    first, and two-way it is whichever of the three sums the channel's error reads off has the
    largest ``tr(W S)``. That needs the fits to read one log's rows, which the function cannot
    check. Over 200 logs of 4000 rows each (section ``calibration``), when the class held, the
    test rejected 1.0%, 6.0% and 11.5% of them at 1, 5 and 10%, and ``cost_error`` was twice
    the cost's spread. When it missed, the cost fell within 1.96 ``cost_error`` of its value on
    400 000 rows in 0.915 of them: it read low by a quarter of ``cost_error`` on average and spread
    1.13 times it, and ``noise`` matched the difference's realised spread to 1.1 of its standard
    errors. The power grows with the miss: at 5% it caught 13%, 30%, 78% and 97% of the logs as the
    channel's slope in the state went 0.005, 0.01, 0.02 and 0.03, where the logs above had 1
    (section ``power``).

    On two zones of :mod:`chc.zones`' market, fitted under ``rk4`` unweighted and weighted by the
    first zone's supply (section ``zones``), it rejected 0.5%, 3.25% and 7.5% of 400 logs at 1, 5
    and 10% where the class held. An incentive that recruits less the more drivers are idle moves
    the channel with the state: at 5% the test caught 5%, 5%, 9% and 32% of the logs as that
    saturation went 0.1, 0.2, 0.4 and 0.8, while the regret between the two fits' plans went from
    0.0047 to 0.0145. Trips bent as far as the harmonic law, which the class misses too, left that
    regret and the test's rate where they were.

    How far it reaches: as far as ``W`` does, second order in the difference. On the logs above,
    where the two channels differed by two thirds of the channel's size, ``d' W d / 2`` read half of
    the regret between the two fits' plans, three quarters of it at three tenths of that difference
    and nine tenths at a tenth (section ``reach``). At a large difference the cost is a floor, not
    an estimate. On the zone plant, where the difference is small, ``cost + noise`` matched the
    regret between the fits' plans within 1% on average at every setting.

    What it cannot see: a direction of the channel the log's actions never moved
    (:attr:`chc.dynamics_id.CausalDynamicsFit.unmoved`) has no data in either fit's moment: both
    hold the channel at zero along it, or read it off the log's rates as least squares does, which
    no effect enters, and a miss along it is not theirs to see. ``unseen`` counts the directions the
    plan's regret weighs. A log whose actions the covariates' features determine, as an undithered
    deployed plan's own may be, leaves every direction unmoved. On a log whose second action was
    always twice the first, the gate read ``p = 0.39`` and two unseen directions while the plan lost
    0.0014 against the truth; held to the log's ratio, the plan had none unseen and lost ``5e-6``
    (section ``unseen``). Nor can it see the error both fits share: on the zone plant, the plan lost
    0.38 to 0.51 against the truth at every setting, including where the class held, 32 to 106 times
    the regret between the fits' plans.

    Raises:
        ValueError: on a fit made without ``influence=True``; two fits whose classes, integrators,
            rows or clusters differ, or that carry drivers, whose gain's price needs the plan's
            forecast; a plan that was not made on ``reference``'s model; and whatever
            :meth:`chc.plan.CausalPlan.decision_weight` refuses.
    """
    for name, fit in (("reference", reference), ("alternative", alternative)):
        if fit.influence is None or fit.unmoved is None:
            raise ValueError(
                f"the {name} fit carries no influence: fit both by fit_causal_residual with "
                "influence=True, since the difference's covariance sums their influences row by row"
            )
        if fit.drivers:
            raise ValueError(
                f"the {name} fit carries drivers {list(fit.drivers)}: their gain's price needs the "
                "plan's forecast, which this does not read"
            )
    ours, theirs = reference.residual, alternative.residual
    if (ours.degree, ours.channel_degree, reference.integrator) != (
        theirs.degree,
        theirs.channel_degree,
        alternative.integrator,
    ):
        raise ValueError(
            "the fits are of different classes: degree, channel_degree and integrator "
            f"{(ours.degree, ours.channel_degree, reference.integrator)} against "
            f"{(theirs.degree, theirs.channel_degree, alternative.integrator)}"
        )
    influence, other = np.asarray(reference.influence), np.asarray(alternative.influence)
    unmoved = np.asarray(reference.unmoved, dtype=np.float64)
    other_unmoved = np.asarray(alternative.unmoved, dtype=np.float64)
    if influence.shape != other.shape or unmoved.shape != other_unmoved.shape:
        raise ValueError(
            f"the fits' influences have shapes {influence.shape} and {other.shape}, and they leave "
            f"{unmoved.shape[1]} and {other_unmoved.shape[1]} directions unmoved: they were not "
            "made on one log's rows"
        )
    clusters, other_clusters = reference.clusters, alternative.clusters
    if not (
        clusters is other_clusters
        or (
            clusters is not None
            and other_clusters is not None
            and np.array_equal(clusters, other_clusters)
        )
    ):
        raise ValueError(
            "the fits sum their influences over different clusters: fit both with the same "
            "clusters, or both with none"
        )
    d = _parameters(theirs) - _parameters(ours)
    weight = _parameter_weight(plan, ours, tolerance)
    covariance = max(
        _clustered_squares(other - influence, clusters),
        key=lambda square: float(np.trace(weight @ square)),
    )
    # W vanishes along a direction the plan does not move to second order in rounding, so the
    # square root of the precision separates it from one the plan weighs at all
    weighed = np.linalg.eigvalsh(unmoved.T @ weight @ unmoved) if unmoved.shape[1] else np.zeros(0)
    floor = math.sqrt(np.finfo(np.float64).eps) * float(np.linalg.norm(weight, 2))
    ws = weight @ covariance
    trace, trace_squared = float(np.trace(ws)), float(np.trace(ws @ ws))
    leverage = max(float(d @ ws @ weight @ d) - trace_squared, 0.0)
    quadratic = float(d @ weight @ d)
    return MisspecificationCost(
        cost=(quadratic - trace) / 2.0,
        cost_error=math.sqrt(4.0 * leverage + 2.0 * trace_squared) / 2.0,
        noise=trace / 2.0,
        p_value=_chi_square_mixture_survival(quadratic, _mixture_weights(weight, covariance)),
        unseen=int(np.sum(weighed > floor)),
    )


def _parameter_weight(
    plan: CausalPlan, residual: ControlAffineResidual, tolerance: float | None
) -> NDArray[np.float64]:
    """``W`` over :func:`_parameters`' order: the regret's curvature at ``plan``, made on
    ``residual``'s model, with the model moved by a residual of its class inside its field."""
    problem = _weighable(plan._problem)
    model = problem.model
    if not (
        isinstance(model, HybridDynamics)
        and isinstance(model.residual, ControlAffineResidual)
        and _parameters(model.residual).shape == _parameters(residual).shape
        and np.array_equal(_parameters(model.residual), _parameters(residual))
    ):
        raise ValueError(
            "the plan was not made on the reference fit's model: its model must be a "
            "HybridDynamics whose residual is the reference fit's"
        )
    shape = plan.actions.shape

    def cost(u: Array, change: Array) -> Array:
        return total_cost(
            HybridDynamics(known=model, residual=_residual(change, residual)),
            problem.x0,
            u.reshape(shape),
            problem.dt,
            problem.cost,
        )

    size = _parameters(residual).size
    return _regret_curvature(problem, plan.actions, cost, size, tolerance)[0]


def _parameters(residual: ControlAffineResidual) -> NDArray[np.float64]:
    """The channel, raveled, then the drift's transpose, raveled: the order of
    :attr:`chc.dynamics_id.CausalDynamicsFit.influence`."""
    return np.concatenate(
        [np.asarray(residual.channel).ravel(), np.asarray(residual.drift).T.ravel()]
    ).astype(np.float64)


def _residual(change: Array, like: ControlAffineResidual) -> ControlAffineResidual:
    """A residual of ``like``'s class whose parameters are ``change``, laid out as
    :func:`_parameters` lays them."""
    size = like.channel.size
    return ControlAffineResidual(
        drift=change[size:].reshape(like.drift.shape[::-1]).T,
        channel=change[:size].reshape(like.channel.shape),
        degree=like.degree,
        channel_degree=like.channel_degree,
    )


def _mixture_weights(
    weight: NDArray[np.float64], covariance: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The eigenvalues of ``W S``, read off the symmetric ``S^1/2 W S^1/2``: ``dh' W dh`` is their
    chi-square mixture when ``dh ~ N(0, S)``."""
    values, vectors = np.linalg.eigh(covariance)
    root = (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T
    return np.clip(np.linalg.eigvalsh(root @ weight @ root), 0.0, None)


def _chi_square_mixture_survival(q: float, weights: NDArray[np.float64]) -> float:
    """``P(sum_k weights_k chi2_1 > q)``, by Imhof's inversion.

    The integrand ``sin(theta(u)) / (u rho(u))`` decays like ``u^(-1 - k/2)`` and oscillates, and
    with one or two terms a plain rule on ``[0, inf)`` misses the tail badly (0.0036 against 0.00053
    for one term at ``q = 30`` times its weight). Past ``u = 20 / omega``, ``omega = q / 2``, the
    phase is ``sum arctan(w u) / 2 - omega u``: a slowly moving amplitude times a pure oscillation,
    which QUADPACK's Fourier rule integrates to the end.
    """
    largest = float(np.max(weights, initial=0.0))
    if largest <= 0.0:
        return 1.0 if q <= 0.0 else 0.0
    weights = weights[weights > 1e-12 * largest]
    if q <= 0.0:
        return 1.0
    omega = 0.5 * q

    def amplitude(u: float) -> float:
        return 1.0 / (u * float(np.prod((1.0 + (weights * u) ** 2) ** 0.25)))

    def phase(u: float) -> float:
        return 0.5 * float(np.sum(np.arctan(weights * u)))

    cut = 20.0 / omega
    head, _ = integrate.quad(
        lambda u: math.sin(phase(u) - omega * u) * amplitude(u),
        0.0,
        cut,
        limit=2000,
        epsabs=1e-13,
        epsrel=1e-12,
    )
    cosine, _ = integrate.quad(
        lambda u: math.sin(phase(u)) * amplitude(u),
        cut,
        np.inf,
        weight="cos",
        wvar=omega,
        limlst=200,
    )
    sine, _ = integrate.quad(
        lambda u: math.cos(phase(u)) * amplitude(u),
        cut,
        np.inf,
        weight="sin",
        wvar=omega,
        limlst=200,
    )
    return min(max(0.5 + (head + cosine - sine) / math.pi, 0.0), 1.0)
