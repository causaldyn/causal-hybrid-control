"""Causal identification of the residual's *control channel*, not just of a scalar effect.

``chc.train`` fits a residual by prediction error. Under a confounded logging policy that is the
wrong object: if the historical action was chosen from a covariate that also drives the state,
then ``d(dx/dt)/du`` fitted on ``(x, u, x_next)`` is the **observational** response, and a
controller that optimises against it moves the state the wrong way. That is the same failure the
scalar headline benchmark shows -- and until now nothing protected the residual from it.

This module closes that gap with Robinson partialling-out lifted from a scalar effect to a
state-dependent matrix. With nuisances ``g(x,z) = E[y | x,z]`` and ``m(x,z) = E[u | x,z]``, where
``y = (x_next - x)/dt - f_known(t, x, u)`` is the part the residual must explain::

    y - g(x,z)  =  B_θ(x) (u - m(x,z))  +  eps,     E[eps | x, z, u - m] = 0

so the estimator regresses the **residualised state rate on the residualised action**, with
K-fold cross-fitting of both nuisances. Neyman orthogonality is what buys the guarantee: the
channel error is *second* order in nuisance error, which is exactly the order transfer
``p -> 2p`` of :func:`chc.research.regret.composition_transfer_certificate` and the per-channel
debiasing by cross-fit Robinson DML of :func:`chc.research.regret.multichannel_control_certificate`.
This module is those results' missing consumer, not a new one.

HONEST SCOPE, four limits worth stating before the code:

* Only the **channel** ``B_θ`` is interventional. ``a_θ`` is fitted on the remainder and therefore
  absorbs whatever the omitted confounder contributes to the drift in-sample, so it is an
  *observational-conditional* drift. Planning is unbiased in the direction the optimiser moves;
  the predicted trajectory *level* still shifts if the confounder's distribution does. On real
  data this bites harder than it sounds: a confounder that trends *with* the state gets charged to
  positive feedback in ``x``, so the fitted drift can come back **unstable** even where the channel
  is clean -- and an MPC uses the drift too. Check the drift's spectrum, and give measured exogenous
  drivers a place in the drift as regressors instead of leaving them only in ``adjust_for``. The
  adjustment set joins the drift regression under either integrator, so ``a_θ`` is conditional on
  it, at its mean on the log; under ``integrator="rk4"`` a push left out would reach the channel
  as well, since RK4 carries the drift into it.
* The residual must be control-affine (:class:`chc.residual.ControlAffineResidual`). A general
  ``r_θ(x, u)`` has no partialling-out moment and gets no guarantee here.
* With no adjustment set and no instrument nothing in the log identifies the channel, unless the
  caller knows the actions are unconfounded given the state (``unconfounded=True``), and an
  instrument identifies it only along the directions it moves the action. The estimator reports
  ``identified=False`` in both cases rather than returning a confident wrong answer -- that case
  belongs to :mod:`chc.research.sensitivity`, which prices the radius instead of pretending it away.
* The premise is Markov: given ``x``, ``z`` and ``u``, a transition's noise does not depend on the
  past. Where the noise persists within a unit and a lever persists too, the state is a common
  effect of the past lever and the past noise, so ``E[eps | x, z, u - m]`` is not 0 and the channel
  is biased by an amount ``channel_error`` does not cover. :func:`persistence_check` reads the
  noise's persistence off the moment's residual; adjusting for the state, the levers and ``z`` a
  period earlier removes the bias where the noise is AR(1) (ADR 0064).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.stats import f as fisher
from scipy.stats import t as student_t

from chc import _units
from chc.causal import (
    _ACTION_AND_OUTCOME,
    _TRANSITION,
    _polynomial_features,
    _refuse_reread,
    _ridge_predict,
    _stream_key,
)
from chc.dynamics import DrivenDynamics, Dynamics, HybridDynamics
from chc.frames import _at, _real_entries, _real_numbers, _unmasked
from chc.integrate import rk4_step
from chc.residual import ControlAffineResidual, control_affine_features

Integrator = Literal["euler", "rk4"]
"""Which one-step map the fit is asked to be consistent with. See :func:`fit_causal_residual`."""

_MAX_NEWTON_STEPS = 20
"""Cap on Newton's method for the ``rk4`` fixed point. From the Euler fit it converges in three to
six steps up to ``|A|dt = 1.2``, so reaching the cap means there is no fixed point to reach."""
_NUISANCE_SHIFT_WARN = 3.0
"""How far, in the errors of the move, a spline refit may move a fit with the polynomial
nuisances alone before :func:`fit_causal_residual` warns that they miss the log's."""

_log = logging.getLogger(__name__)

_NOT_IDENTIFIED = (
    "no adjustment set and no instrument: the control channel is not identified from this log. "
    "Use chc.research.sensitivity to price the identification radius instead of trusting the "
    "estimate."
)


@dataclass(frozen=True)
class ConfoundedControlAffineSystem:
    """Euler-logged ``dx/dt = f_known(x,u) + A x + B_true u + C z`` under ``u = K z + eta``.

    The vector-state analogue of :class:`chc.causal.ConfoundedLinearSystem` and the DGP
    :func:`fit_causal_residual` is scored against. ``z`` is the confounder: it drives the logged
    action through ``K`` *and* the state rate through ``C``, so with ``z`` withheld the response of
    the rate to the action is ``B_true + C sigma_z^2 K^T / var(u)`` rather than ``B_true``, where
    ``var(u) = K K^T sigma_z^2 + G G^T + sigma_eta^2`` collects every source of action variance.

    A scalar channel with ``B=1, C=2, K=-1.5, sigma_z=1, sigma_eta=0.5`` and **no instrument**
    gives ``1 - 2*1.5/2.5 = -0.2``: the sign flip of the headline benchmark, reproduced per
    channel. Turning the instrument on adds ``G G^T`` to that denominator and *dilutes* the bias
    toward zero -- the exogenous shifter makes the log more informative, so the observational fit
    looks less wrong while being no more identified. Worth knowing before reading a number off it.

    The Euler discretisation is deliberately the one :func:`fit_causal_residual` inverts, so a
    failing recovery test means the identification is wrong, not that the integrator disagreed.
    """

    drift: Array  # (n, n) state -> rate
    channel: Array  # (n, m) causal control channel, the estimand
    confounder_to_rate: Array  # (n, d)
    confounder_to_action: Array  # (m, d) the logging policy
    instrument_to_action: Array | None = None  # (m, d) exogenous shifter, None = no instrument
    dt: float = 0.05
    z_scale: float = 1.0
    eta_scale: float = 0.5
    noise_scale: float = 0.01

    def sample(self, n: int, key: Array, known: Dynamics) -> dict[str, Array]:
        """Draw ``n`` transitions: ``x (n,d_x)``, ``z``, ``u``, ``x_next``, ``w`` (instrument)."""
        state_dim, control_dim = self.channel.shape
        conf_dim = self.confounder_to_rate.shape[1]
        k_x, k_z, k_eta, k_w, k_noise = jax.random.split(key, 5)
        x = jax.random.normal(k_x, (n, state_dim))
        z = self.z_scale * jax.random.normal(k_z, (n, conf_dim))
        eta = self.eta_scale * jax.random.normal(k_eta, (n, control_dim))
        w = jax.random.normal(k_w, (n, conf_dim))
        u = z @ self.confounder_to_action.T + eta
        if self.instrument_to_action is not None:
            u = u + w @ self.instrument_to_action.T
        rate = (
            jax.vmap(lambda xi, ui: known(0.0, xi, ui))(x, u)
            + x @ self.drift.T
            + u @ self.channel.T
            + z @ self.confounder_to_rate.T
        )
        noise = self.noise_scale * jax.random.normal(k_noise, (n, state_dim))
        return {"x": x, "z": z, "u": u, "w": w, "x_next": x + self.dt * rate + noise}


@dataclass(frozen=True, eq=False)
class _ChannelGram:
    """The channel's least squares as :func:`channel_region` reads it: ``matrix``, ``(q, q)``,
    is ``D'D + ridge P``, ``D`` the channel's design on the action residuals and ``P`` the ridge's
    scale per coefficient, ``ridge``, ``(q,)``, is ``ridge P``'s diagonal, and ``design``,
    ``(N, q)``, is ``D``, whose scores ``D_i e_i`` the region sums within the fit's ``clusters``.
    None of the three reads the clusters. Equal where every array is, entry for entry."""

    matrix: np.ndarray
    ridge: np.ndarray
    design: np.ndarray

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _ChannelGram)
            and np.array_equal(self.matrix, other.matrix)
            and np.array_equal(self.ridge, other.ridge)
            and np.array_equal(self.design, other.design)
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, eq=False)
class _FieldTerms:
    """What the error of the fitted field at a state ``x`` and an action ``u`` is read from, to
    first order, for each state ``s``: ``f_s(x, u) = phi(x)' r_s + g(x, u)' c_s``, ``phi`` the
    drift's features then the drivers, ``g(x, u)`` the channel's regressors ``u (x) psi(x)``,
    ``r_s`` the drift regression's coefficients and ``c_s`` the channel's. A channel error ``e``
    moves the drift by ``-projection[s] e``, so the field's error is ``phi' d_s + L_s' e`` with
    ``L_s = e_s (x) g(x, u) - projection[s]' phi``: the channel's error times the action's distance
    from the log's projected action, one term (Frisch and Waugh 1933). ``d_s`` is the drift's own
    error, conditional on the channel, ``cross`` its covariance with ``e``, and the intercept holds
    the adjustment set's push at its mean on the log, off the plant's by ``push``'s variance:

        Var f_s = phi' drift[s] phi + L_s' channel L_s + 2 L_s' cross[:, s] phi + push[s].

    ``channel``, ``(r, n q, n q)``, is the channel's covariance, each of the reads its error takes
    the largest of; ``projection``, ``(n, C, n q)``, the drift's ``C`` carried coefficients' move
    with the channel's, under ``rk4`` through the fixed point's drift block (Murphy and Topel
    1985); ``drift``, ``(r', G, C, C)``, one block every state shares or one a state;
    ``cross``, ``(r', n q, n, C)``; ``push``, ``(r', n)``. ``r'`` is 1 where the drift's side is
    pooled over the rows, as without clusters or periods, and ``r`` where it is read as the
    channel's is. ``states`` ``(N, n)`` and ``drivers`` ``(N, d)`` are the log's rows, which the
    field's error over the log is read at. Equal where every array is.

    The field's errors at two states, or at two of a plan's steps, move together: one error of
    the coefficients makes them all. ``drift_joint``, ``(r', n C, n C)``, is the drift's own
    covariance over every state's carried coefficients, a state's beside each other, whose blocks
    on the diagonal are ``drift``'s; ``push_joint``, ``(r', n, n)``, the push's mean's over the
    states, whose diagonal is ``push``. :meth:`joints` puts them together."""

    mean_action: np.ndarray
    projection: np.ndarray
    channel: np.ndarray
    drift: np.ndarray
    cross: np.ndarray
    push: np.ndarray
    states: np.ndarray
    drivers: np.ndarray
    drift_joint: np.ndarray
    push_joint: np.ndarray

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FieldTerms) and all(
            np.array_equal(getattr(self, field.name), getattr(other, field.name))
            for field in dataclasses.fields(self)
        )

    __hash__ = None  # type: ignore[assignment]

    def joints(self) -> list[np.ndarray]:
        """The covariance of the whole error ``(e, d_1 .. d_n, p)``, ``(n q + n C + n)`` square,
        at each of the reads: the channel's ``e``, each state's drift's own ``d_s`` and the push's
        mean ``p``, a state each. The push's mean moves with neither of the others: its error is
        the adjustment set's sampling mean, which the noise does not move."""
        reads = max(self.channel.shape[0], self.drift_joint.shape[0])
        size, count = self.channel.shape[1], self.push.shape[1]
        drift = self.drift_joint.shape[1]
        joints = []
        for read in range(reads):
            side = read if self.drift_joint.shape[0] > 1 else 0
            joint = np.zeros((size + drift + count, size + drift + count))
            joint[:size, :size] = self.channel[read if self.channel.shape[0] > 1 else 0]
            cross = self.cross[side].reshape(size, drift)
            joint[:size, size : size + drift] = cross
            joint[size : size + drift, :size] = cross.T
            joint[size : size + drift, size : size + drift] = self.drift_joint[side]
            joint[size + drift :, size + drift :] = self.push_joint[side]
            joints.append(joint)
        return joints


@dataclass(frozen=True)
class CausalDynamicsFit:
    """A fitted causal residual next to what is and is not known about it.

    ``identified`` is the load-bearing field: ``False`` means nothing the fit was given pins the
    channel down from the log -- no adjustment set and no instrument, an instrument whose moment
    falls short of rank (``instrument_rank``), or a log that moves no direction of the channel
    (``unmoved``) -- and the residual is then the observational fit, kept only so the caller can
    compare.
    """

    residual: ControlAffineResidual
    identified: bool
    method: str  # "orthogonal" | "iv" | "observational"
    folds: int
    # The standard error of the fitted channel's value B(x) at the log's states, root mean square
    # over the states and the channel's entries, None when not identified. It is read off the
    # channel's robust covariance: each row's squared structural residual carried through the fit's
    # own linear map, cross-fitted nuisances included, or under ``clusters`` each cluster's summed
    # score squared (CR1; two-way, the largest of three reads). Up to 0.12 it was that covariance's
    # root-mean diagonal, whose constant term is the value at x = 0, so it moved with the state's
    # zero: one log read 0.0091, and 0.579 with its states moved by 100 and -50. The figures that
    # follow compared that version with the spread of the same coefficients. Robust since 0.8.0;
    # the homoskedastic error before it read 0.80x the channel's spread over 800 logs whose noise
    # moves with the state, and 0.40x when the class missed the truth, where the robust one's root
    # mean square reads 0.96x and 1.04x. On a confounded plant it came to 1.08x the channel's error
    # against the truth on the adjustment path and 1.05x on the ``iv`` path (200 logs of 2000
    # rows). One log's value scatters, by 24% and 40% of itself on those 800 logs of 4000 rows: a
    # scale, not coverage. Under ``rk4`` it is the fixed point's own, the noise carried through the
    # RK4 map's gain on the estimate: 1.70x the Euler fit's at ``theta*dt = 0.7``, where 200 noise
    # draws on one log scattered the channel 1.73x as far. Read from ``G`` clusters' sums, a test
    # read off it sizes better against ``t(G - 1)``'s quantile than the normal's (ADR 0062). Under
    # ``periods``, one group's, the covariance is read from the scores' projections on ``nu``
    # cosines of the periods (EWC), each period's transitions summed first, and a test read off
    # it against ``t(nu)``'s (ADR 0075). Its mean adds entries in the units of different levers,
    # so where the levers' units differ it reads them; ``_lever_errors`` holds it lever by lever,
    # and the plan's tube reads that, each lever at its own largest action (ADR 0087).
    channel_error: float | None
    # The standard error of the drift regression's fitted value at the log's rows, root mean square
    # over the rows and the states, from the drift stage's own homoskedastic OLS covariance; None
    # when the channel is not identified (the drift is then conditional on a meaningless channel).
    # Up to 0.12 it was that covariance's root-mean diagonal, which the figures here were read on.
    # It is a DIFFERENT object from ``channel_error``: conditional on the fitted channel, whose
    # uncertainty it does not propagate, and homoskedastic -- a scale, not coverage. Reported
    # because on a real plant the drift, not the channel, dominated closed-loop cost. Its noise is
    # the drift regression's residual, so it counts what the model class leaves out: on the
    # marketing-mix plant, 3.4x the drift's scatter over noise draws. Under ``rk4`` it goes through
    # the drift's block of the RK4 map's gain, and the map couples the drift to the channel it is
    # conditional on: on a log the model class fits, the drift scattered 1.26x it at
    # ``theta*dt = 0.7`` over 400 seeds, against 1.02x under Euler. Where the fit is given
    # ``clusters`` or ``periods`` it is read as ``channel_error`` is, from the regression's scores,
    # its weight on each row times the row's residual, summed within each cluster or projected on
    # the periods' cosines, the largest read of their reads; up to 0.21.0 it pooled the residuals
    # as if each row were independent there too.
    drift_error: float | None
    action_residual_variance: (
        float  # overlap proxy: 0 => a deterministic policy, nothing to regress
    )
    nuisance_r2_state: float
    nuisance_r2_action: float
    moment_norm: float  # ||mean(design * residual)|| at the solution; should be ~0
    integrator: Integrator = "euler"  # which one-step map the fit was made consistent with
    # RMS one-step defect in rate units, ``||x_next - step(F, x, u, dt)|| / dt``, under THAT
    # integrator. Comparable across the two settings. It floors at the observation noise, never at
    # zero; above that it is what the model class leaves out, since an ``rk4`` fit that cannot
    # reach its fixed point raises instead.
    integrator_defect: float | None = None
    # (N, n): that defect on each transition, per state, in rate units, ``integrator_defect`` its
    # root mean square: the noise and what the model class leaves out of the log, row by row.
    defect: Array | None = None
    drivers: tuple[str, ...] = ()  # the exogenous columns fitted into the drift, in gain order
    # (n, d): each driver's push on each state's rate, fitted jointly with the drift; None without
    # drivers. :class:`chc.dynamics.DrivenDynamics` replays it against a forecast.
    driver_gain: Array | None = None
    weighted: bool = False  # whether a weight on the state weighed the channel moment
    # Kish's (sum w)^2 / sum w^2: how many equally weighted transitions the moment is worth
    effective_sample_size: float | None = None
    # (N, n, p), kept when the fit was asked for it: each transition's share, per state, of the
    # error in the fitted parameters when the whole log is drawn again -- the channel, raveled, then
    # the drift regression's rows (the drift's transpose, then the drivers' gain's), raveled; the
    # gains of the adjustment set's held push are left out, as the field leaves them. A row moves
    # the channel through the moment's residual, and the drift through the channel and through its
    # own regression's residual, so the sum of psi psi' over both is their robust covariance, the
    # drift's carrying the channel's error. Under Euler its channel block is ``channel_error``'s;
    # under ``rk4`` the RK4 map also carries the drift's residual into the channel, which
    # ``channel_error`` reads as the noise: the drift regression holds the adjustment set's push,
    # so its residual is the noise where the push is linear.
    influence: Array | None = None
    # (p, r), in ``influence``'s order: the directions the log's actions never move. Each moves one
    # state's channel where the action residuals, scaled to the raw actions' size, fall below the
    # square root of the working precision, with the drift regression's response to it on this log.
    # The moment has no data there. Where the drift's features take up a direction's push on the
    # log, no rate of the log tells it apart, and the channel is held at zero along it, in the
    # actions' scaled units; where they cannot, the log's rates rule out all but one value, and the
    # channel takes it by least squares beside the drift: what the log did, not an effect, whatever
    # ``channel_error`` says (ADR 0054). The directions are orthogonal with each channel coefficient
    # scaled to its column of the channel's design on the log's raw actions, and a span of them is
    # read there: in raw units, where the units of the state and of the actions set those columns'
    # sizes apart, they lean together. ``r`` is 0 when the log moves every direction, and ``m k``,
    # each state's whole channel, when it moves none: the fit is then not identified.
    unmoved: Array | None = None
    # (N, n, q), kept with ``influence``, ``q`` the channel's size in its raveled order: ``N`` times
    # each transition's weight, per state, in each channel coefficient through the channel's moment
    # with the cross-fitted nuisances held, less the drift regression's own weight on the row, which
    # reads the state alone and so carries no omitted confounder's push. It is the plug-in Riesz
    # representer of every linear functional of the channel, ``N (D'D)^-1 D_i`` under Euler, which
    # :func:`omitted_confounder_bound` reads. Under ``rk4`` it goes through the fixed point, to
    # first order, as ``influence`` does.
    representer: Array | None = None
    # (N, n), where the channel is identified: its moment's residual, ``y_res - D c``, per state,
    # which :func:`persistence_check` reads.
    moment_residual: Array | None = None
    # (N,), each transition's cluster as a code from 0 where the fit was given ``clusters``, or
    # (N, 2) where it was given two, a code a dimension: the channel's covariance then sums each
    # cluster's scores over its transitions and states before squaring them (CR1), two-way the
    # largest read of the two ways and of each alone, and a reader of ``influence`` sums its rows
    # the same way (:func:`_clustered_squares`). None where each transition is its own, its states
    # summed.
    clusters: np.ndarray | None = None
    # Experimental: it may change or be withdrawn in any release. Under ``instrument``, ``(p,)``:
    # how far the instrument moves each of the ``p`` directions a state of the channel the log's
    # actions move, all but ``unmoved``'s. They are the canonical correlations, largest first,
    # between the first stage's push on the actions, fitted without the ridge, and the actions,
    # each less its least-squares projection on the nuisance's features, on the channel's features
    # and with the moment's weights: the unregularised moment's singular values with each side
    # whitened, read the same in any units of the instrument, the actions and the state. A push of
    # at most 64 eps of the raw actions' size reads 0, as does a correlation of at most 64 eps. A 0
    # leaves the moment short of rank, and the fit not identified; so does ``p = 0``, a log that
    # moves no direction, which leaves the instrument none to move. The smallest says how weak the
    # instrument is where it is weakest; it is not a test. None without an instrument.
    instrument_relevance: Array | None = None
    # Experimental, as ``instrument_relevance``: how many of its entries are above 0, the rank of
    # the instrument's moment along the directions the log moves. None without an instrument.
    instrument_rank: int | None = None
    # (N,), each transition's period as steps from the log's first where the fit was given
    # ``periods``, the transitions of one group, one unit's or the units' of one cluster: the
    # channel's covariance then sums the scores of each period's transitions over them and their
    # states, and reads the sums' dependence across the periods by ``nu`` cosines of them (EWC,
    # :attr:`error_cosines`), and a reader of ``influence`` sums its rows the same way
    # (:func:`_clustered_squares`). None where the fit was given none.
    periods: np.ndarray | None = None
    # (p,), set with ``unmoved``, in ``influence``'s order: each parameter's scale on the log, the
    # norm of its column of the channel's design on the log's raw actions, or of the drift
    # regression's, over the norm of its state's column of the log, each 1 for a column of zeros. A
    # parameter times it reads the same in any units of the state and of the actions. ``unmoved``'s
    # directions so scaled are orthogonal on each state's channel, where the fit read them, and a
    # reader of their span reads it there.
    _parameter_size: Array | None = dataclasses.field(default=None, repr=False, compare=False)
    # (r, m), set with ``channel_error``: the same error read over each lever's column of the
    # channel alone, off each of the ``r`` covariances ``channel_error`` is the largest read of, one
    # or, under two clusterings, three. Lever ``j``'s is in the states' units per unit of lever
    # ``j``, where ``channel_error`` averages over levers in units of their own; with one lever it
    # is the covariance's read, bit for bit. The tube reads it at each lever's largest action.
    _lever_errors: np.ndarray | None = dataclasses.field(default=None, repr=False, compare=False)
    # (r, n m, F, F), set with ``_lever_errors``: the covariances of each channel entry's
    # coefficients, entry ``(i, j)`` at ``i m + j``, that it and ``channel_error`` read at the log's
    # rows. The tube reads them at the plan's own states (``chc.decision._field_errors``).
    _channel_blocks: np.ndarray | None = dataclasses.field(default=None, repr=False, compare=False)
    # (G, F, F), set with ``drift_error``: the covariance of the drift regression's coefficients,
    # its features then its drivers, one block every state shares under Euler without clusters or
    # periods, one a state otherwise, that ``drift_error`` reads at the log's rows as a root mean
    # over the blocks, the largest read's where it takes several.
    _drift_blocks: np.ndarray | None = dataclasses.field(default=None, repr=False, compare=False)
    # Set on an identified fit, under either integrator (ADR 0131, ADR 0133): what the tube reads
    # the field's error from at any state and any action of the box, ``chc.decision._field_errors``.
    # The drift regression reads its target at the fitted channel, so the channel's error reaches
    # the field through the drift where the log's actions lean on the state, and the channel's
    # part and that part are one term, priced from the log's projected action; under ``rk4`` the
    # fixed point also moves the channel with the drift's own error, which they share. The
    # intercept holds the adjustment set's push at its mean on the log, its error read as the
    # channel's is; without clusters or periods that read counts the rows as independent.
    _field_terms: _FieldTerms | None = dataclasses.field(default=None, repr=False, compare=False)
    # Set on an identified Euler fit whose channel is least squares on the action residuals (no
    # instrument, no weight, every direction moved): the matrix its moment solved, which
    # :func:`channel_region` reads.
    _channel_gram: _ChannelGram | None = dataclasses.field(default=None, repr=False, compare=False)

    @property
    def error_cosines(self) -> int | None:
        """``nu``, the cosines of the periods the channel's error was read with where the fit was
        given ``periods``, counted over the distinct periods, and the degrees of freedom of the
        ``t`` a test read off it is sized against; None where it was given none."""
        return None if self.periods is None else _period_cosines(self.periods)


def _r_squared(target: Array, prediction: Array) -> float:
    centred = target - jnp.mean(target, axis=0, keepdims=True)
    total = float(jnp.sum(centred**2))
    if total == 0.0:
        return 0.0
    return float(1.0 - jnp.sum((target - prediction) ** 2) / total)


def _cross_fit_residuals(
    target: Array,
    action: Array,
    covariates: Array,
    *,
    degree: int,
    folds: int,
    ridge: float,
    seed: int,
    features: Array | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Vector-valued cross-fitted partialling-out; the matrix form of ``_dml_residuals``.

    ``features``, ``(N, p)``, is the nuisances' basis at every row (:func:`_nuisance_sieve`) where
    given, in place of the polynomial of degree ``degree`` in the standardised covariates.

    Kept separate rather than generalising that function: it has three callers on a scalar
    contract, and :func:`chc.causal._ridge_predict` already accepts a matrix target, so the only
    thing this adds is the fold bookkeeping over two dimensions -- and the standardisation below.

    Each covariate is centred and scaled first (:func:`chc._units.standardised`), so the nuisance
    basis is conditioned in any caller's units. Without this the polynomial basis inherits whatever
    units the caller happened to use, and the ridge penalises each monomial accordingly --
    ``ridge=1e-6`` means something entirely different against a column of order 1 than against its
    square of order 400. Measured on a 20-day log from a real building emulator (``causaldyn-bench``
    Track D-causal), where the zone enters in Celsius at ~21 and the weather columns are already
    standardised: the degree-2 Gram came out at condition number **1.4e11**, past what float32 can
    carry, and the fit returned ``nan``. The same rows in float64 fitted fine, which is what
    identified this as conditioning rather than data. Scaling first drops it to order 10 -- on a
    reproducible stand-in with that geometry (768 rows, zone at 21 +- 0.9 beside three
    standardised columns) the degree-2 Gram goes from ``2.7e10`` to ``2.4e1`` -- and removes the
    precision dependence.

    Full-sample statistics rather than per-fold ones. Centring and scaling the *inputs* is a linear
    reparametrisation, so the span of the fitted basis -- and hence the partialled-out residual --
    is unchanged by it; only the ridge's meaning moves, and pinning that to the data's own scale is
    the point. Per-fold statistics would make the penalty fold-dependent for no gain. A column whose
    spread is rounding reads 0 in every row: divided by its own spread it would be ``nan``, or its
    rounding blown up to unit size, and a floor of 1e-12 in the caller's units took a column logged
    in small units for a constant and left it unscaled under the ridge.
    """
    covariates = _units.standardised(covariates)
    n = target.shape[0]
    chunks = jnp.array_split(jax.random.permutation(_stream_key(seed, "folds"), n), folds)
    target_hat = jnp.zeros_like(target)
    action_hat = jnp.zeros_like(action)
    for k in range(folds):
        test = chunks[k]
        train = (
            jnp.concatenate([chunks[j] for j in range(folds) if j != k]) if folds > 1 else chunks[0]
        )
        if features is None:
            phi_train = _polynomial_features(covariates[train], degree)
            phi_test = _polynomial_features(covariates[test], degree)
        else:
            phi_train, phi_test = features[train], features[test]
        target_hat = target_hat.at[test].set(
            _ridge_predict(phi_train, target[train], phi_test, ridge)
        )
        action_hat = action_hat.at[test].set(
            _ridge_predict(phi_train, action[train], phi_test, ridge)
        )
    return target - target_hat, action - action_hat, target_hat, action_hat


def _natural_spline(column: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """The natural cubic spline's nonlinear columns at ``knots``, ``(N, K - 2)`` for ``K`` knots, in
    the truncated-power form of Hastie, Tibshirani and Friedman (2009, eq. 5.4-5.5): cubic between
    the knots, linear beyond the outer two, so a fold's predictions do not run off in the tails as a
    polynomial's of high degree do."""

    def cubic(k: int) -> np.ndarray:
        return (
            np.maximum(column - knots[k], 0.0) ** 3 - np.maximum(column - knots[-1], 0.0) ** 3
        ) / (knots[-1] - knots[k])

    last = cubic(knots.size - 2)
    return np.stack([cubic(k) - last for k in range(knots.size - 2)], axis=1)


def _nuisance_sieve(covariates: Array, degree: int, knots: int, powers: bool = True) -> Array:
    """The nuisances' basis at every row, ``(N, p)``: the polynomial of degree ``degree`` in the
    standardised covariates, and a natural cubic spline in each, at ``knots`` of its quantiles,
    their columns spanned by an orthonormal basis over the log and scaled to unit mean square.

    The spline's truncated powers are near collinear: on 16000 rows of two covariates with 26
    knots each, the raw columns' condition number was 1.4e6, past what float32 carries in a Gram.
    The basis of their span, from the full log's singular vectors, has one; full-sample statistics,
    as the standardisation's, so every fold reads one basis. A covariate with fewer than three
    distinct knots, a flag or a constant, adds no spline: its polynomial holds what it can.
    Without ``powers`` a splined covariate keeps no power past its first, in any monomial
    (:func:`_chosen_sieve`)."""
    raw = _sieve_columns(covariates, degree, knots, powers)
    return jnp.asarray(_orthonormal(raw), dtype=jnp.result_type(covariates))


def _sieve_columns(
    covariates: Array, degree: int, knots: int | None, powers: bool = True
) -> np.ndarray:
    """The raw columns :func:`_nuisance_sieve` spans, ``(N, q)``, the bias first: the polynomial
    in the standardised covariates, without ``powers`` less a splined covariate's powers past its
    first, and the splines; with ``knots`` None, the polynomial alone."""
    standardised = np.asarray(_units.standardised(covariates), dtype=np.float64)
    if knots is None:
        at = [np.empty(0)] * standardised.shape[1]
    else:
        positions = np.linspace(0.0, 1.0, knots + 2)[1:-1]
        at = [np.unique(np.quantile(column, positions)) for column in standardised.T]
    splined = [knot.size >= 3 for knot in at]
    polynomial = np.asarray(_polynomial_features(jnp.asarray(standardised), degree))
    monomials = [
        combo
        for power in range(degree + 1)
        for combo in combinations_with_replacement(range(standardised.shape[1]), power)
    ]
    single = [
        powers or all(combo.count(index) <= 1 for index in combo if splined[index])
        for combo in monomials
    ]
    columns = [polynomial[:, np.asarray(single)]]
    for column, knot, spline in zip(standardised.T, at, splined, strict=True):
        if spline:
            columns.append(_natural_spline(np.asarray(column), knot))
    return np.concatenate(columns, axis=1)


def _orthonormal(raw: np.ndarray) -> np.ndarray:
    """An orthonormal basis of ``raw``'s columns' span over its rows, from its singular vectors,
    each column scaled to unit mean square; a direction rounding alone spans is left out."""
    left, singular, _ = np.linalg.svd(raw, full_matrices=False)
    if singular.size == 0 or singular[0] == 0.0:
        return np.zeros((raw.shape[0], 0))
    kept = singular > singular[0] * max(raw.shape) * np.finfo(np.float64).eps
    return left[:, kept] * math.sqrt(raw.shape[0])


def _chosen_sieve(
    covariates: Array,
    target: Array,
    action: Array,
    *,
    degree: int,
    knots: int,
    folds: int,
    ridge: float,
    seed: int,
) -> tuple[int | None, bool]:
    """The nuisances' sieve, as the knots and whether a splined covariate keeps its powers past
    the first: the spline with them, the spline without them, or the polynomial alone, knots
    ``None``, whichever's cross-fitted residuals of ``target`` and ``action`` are the smaller,
    each column's mean square over its variance, summed; the first of these on a tie.

    Beside its spline a covariate's power holds its shape where the spline does, between the
    knots, so the power's coefficient is read off the rows past the outer knots alone, where the
    spline goes on as a line and the power does not; a row of the other fold further out reads it
    at the power of its distance. On a linear plant with two ``t(3)`` confounders, which the
    polynomial holds, a confounder at -18 of its scales left the action a cross-fitted residual of
    11 where its noise is 0.5, and the channel scattered 3.0 and 4.4 times as far as the
    polynomial's over 200 logs of 500 and 2000 rows, 2.2 times with lognormal ones at 1000.
    Without the powers a nuisance quadratic in a normal confounder is a line past the outer knots,
    and over 100 logs of 1000 rows the channel's bias was 2.0 of its errors, its interval
    covering 47 %, and 2.4 and 32 % at 4000 rows. Where the confounder is ``t(5)`` or ``t(3)``
    and the nuisance quadratic both spline sieves miss: with the powers the channel scattered 3.6
    and 2.9 times as far as the polynomial's over 40 logs of 1000 rows, without them its bias
    was 4.4 and 21 of its errors. No one sieve serves every log, and the log tells them apart:
    the polynomial was chosen on 78 and 80 of 80 such fits, and the channel then scattered 0.99
    and 1.00 times as far as the polynomial's. The choice reads the outcome, as any learner chosen
    by cross-validation does, and the moment is orthogonal to the nuisances' error, so it enters
    at second order."""
    rows = target.shape[0]
    candidates: tuple[tuple[int | None, bool], ...] = ((knots, True), (knots, False), (None, True))
    error = []
    for knot_count, powers in candidates:
        features = (
            None if knot_count is None else _nuisance_sieve(covariates, degree, knot_count, powers)
        )
        target_res, action_res, _, _ = _cross_fit_residuals(
            target,
            action,
            covariates,
            degree=degree,
            folds=folds,
            ridge=ridge,
            seed=seed,
            features=features,
        )
        error.append(
            sum(
                float(jnp.sum(residual**2) / jnp.maximum(rows * jnp.var(column), 1e-300))
                for residual, column in (
                    *zip(target_res.T, target.T, strict=True),
                    *zip(action_res.T, action.T, strict=True),
                )
            )
        )
    return candidates[int(np.argmin(error))]


def _held_pushes(columns: Array, degree: int, knots: int | None, powers: bool = True) -> Array:
    """``(N, q)``: the pushes ``columns`` make on the rate in the nuisances' own basis, for the
    drift regression -- the sieve's columns in them alone, at ``degree`` 1 at the least, about
    their means on the log, in an orthonormal basis of their span."""
    if columns.shape[1] == 0:
        return columns
    raw = _sieve_columns(columns, max(degree, 1), knots, powers)[:, 1:]
    centred = raw - raw.mean(axis=0, keepdims=True)
    return jnp.asarray(_orthonormal(centred), dtype=columns.dtype)


def _first_stage(
    states: Array, shifter: Array, degree: int, knots: int | None, powers: bool = True
) -> Array:
    """The instrument's first stage, ``(N, q)``, the bias first: the nuisances' own sieve in the
    states, and the monomials of the polynomial of ``degree`` in the standardised states and
    ``shifter`` that read the shifter; with ``knots`` None, that polynomial whole.

    The state's part is in the nuisances' basis, so a shifter that moves nothing pushes the
    actions only along directions the nuisances take out, and the moment has no rank from it. Read
    through the polynomial beside a sieve without a splined state's powers past its first, the
    first stage's push held those powers, which the nuisances leave in the action's residual:
    an instrument of zeros on 200 rows at ``nuisance_degree=3`` read relevances of 0.15, 0.10 and
    0.055, and an identified channel, set by the functional form alone."""
    joined = _units.standardised(jnp.concatenate([states, shifter], axis=1))
    polynomial = _polynomial_features(joined, degree)
    if knots is None:
        return polynomial
    monomials = [
        combo
        for power in range(degree + 1)
        for combo in combinations_with_replacement(range(joined.shape[1]), power)
    ]
    reads = np.asarray([any(index >= states.shape[1] for index in combo) for combo in monomials])
    own = jnp.asarray(_sieve_columns(states, degree, knots, powers), dtype=states.dtype)
    return jnp.concatenate([own, polynomial[:, reads]], axis=1)


def _knot_count(knots: int | Literal["auto"], rows: int) -> int:
    """``knots``, or for ``"auto"`` ``ceil(N^(1/3))``, at least 3: it grows with the log, so the
    spline's error at a smooth nuisance shrinks faster than the moment's ``N^(-1/2)``."""
    if knots == "auto":
        return max(3, math.ceil(rows ** (1.0 / 3.0) - 1e-9))
    if isinstance(knots, bool) or not isinstance(knots, int) or knots < 3:
        raise ValueError(
            f"nuisance_knots must be an integer of at least 3 or 'auto'; got {knots!r}"
        )
    return knots


def _ridge_parts(design: Array, ridge: float) -> tuple[Array, Array, Array, Array]:
    """The columns after the bias, centred twice, their means and second-pass shifts
    (:func:`chc._units.centred`), and the inverse of their Gram under the ridge: a ridge solve of
    ``X'X + ridge diag(s)`` written with the columns centred, where the bias is apart from the rest,
    for a design whose first column is its bias, as :func:`chc.residual.control_affine_features`
    and :func:`chc.causal._polynomial_features` put it. ``s`` is 0 on the bias, left free to take
    up each column's mean, and each other column's variance about its mean (ADR 0057). Scaled by
    its mean square, a column far from zero, such as a temperature in kelvin, had its slope shrunk
    by the square of its offset over its spread.

    The solve is the same, and a column far from zero costs it no digits: on the raw Gram, a column
    1e-11 of its size from constant was the bias's twin to rounding, under a term too small to tell
    them apart, and read nan. Added as a constant, the ridge outweighed a column logged in units a
    millionth of its own, and set its coefficient near zero. A column whose spread is rounding is
    zeroed and its row and column of the inverse are 0: its coefficient is exactly 0, with no error
    of its own, and the bias carries its level. Kept with a ridge scaled by its mean square, beside
    the bias it repeats, it left the bias an error of the order of the inverse of the ridge."""
    columns = _units.centred(design[:, 1:])
    centred = columns.deviations
    inner = jnp.linalg.inv(centred.T @ centred + ridge * jnp.diag(columns.scales))
    kept = ~columns.rounding
    inner = jnp.where(kept[:, None] & kept[None, :], inner, 0.0)
    return centred, columns.centre, columns.shift, inner


def _solve_ridge(design: Array, target: Array, ridge: float) -> Array:
    centred, centre, shift, inner = _ridge_parts(design, ridge)
    slopes = inner @ (centred.T @ target)
    bias = jnp.mean(target, axis=0) - centre @ slopes - shift @ slopes
    return jnp.concatenate([bias[None], slopes])


def _ridge_weight(design: Array, ridge: float) -> Array:
    """``(X'X + ridge diag(s))^-1 X'``: each row's weight on each coefficient of the solve."""
    centred, centre, shift, inner = _ridge_parts(design, ridge)
    slopes = inner @ centred.T
    bias = 1.0 / design.shape[0] - centre @ slopes - shift @ slopes
    return jnp.concatenate([bias[None], slopes])


def _ridge_inverse(design: Array, ridge: float) -> Array:
    """``(X'X + ridge diag(s))^-1``, put together from the centred columns' own."""
    _, centre, shift, inner = _ridge_parts(design, ridge)
    mean = centre + shift
    corner = 1.0 / design.shape[0] + mean @ inner @ mean
    edge = -(inner @ mean)
    return jnp.block([[corner[None, None], edge[None, :]], [edge[:, None], inner]])


def _channel_design(action_residual: Array, states: Array, degree: int) -> Array:
    """Row ``i`` is ``vec(u_res_i (x) phi(x_i))``, so a coefficient block ``C[j, k, l]`` means
    ``B_θ(x)[j, k] = sum_l C[j, k, l] phi_l(x)``."""
    phi = jax.vmap(control_affine_features, in_axes=(0, None))(states, degree)
    n, control_dim = action_residual.shape
    return (action_residual[:, :, None] * phi[:, None, :]).reshape(n, control_dim * phi.shape[1])


def _channel_coefficients(
    state_residual: Array,
    regressor: Array,
    instrument: Array,
    ridge: float,
    penalty: Array,
    weights: Array | None = None,
) -> Array:
    """Solve the just-identified moment ``Z'(y_res - D c) = 0`` for ``c``, ridge-stabilised: the
    ridge times ``penalty`` added to ``Z'D``. The penalty scales each coefficient to its column's
    size, so the ridge reads the same in any units of the actions and the state.

    ``Z is D`` reduces to ordinary least squares; a different ``Z`` is two-stage least squares.
    Writing it as one solve rather than "regress on the projection" matters: the two agree only for
    a scalar action with no state features, because the in-sample identity ``Z'u = Z'Z`` does not
    survive multiplication by ``phi(x)``.
    """
    if weights is not None:
        instrument = instrument * weights[:, None]
    gram = instrument.T @ regressor + ridge * penalty
    return jnp.linalg.solve(gram, instrument.T @ state_residual)


def _small_sample(rows: int, n_coeff: int, clusters: int | None) -> float:
    """The robust covariance's small-sample factor: ``N / (N - k)`` over rows, and CR1's
    ``G / (G - 1) (N - 1) / (N - k)`` over ``G`` clusters, which is the same where each row is its
    own."""
    if clusters is None:
        return rows / max(rows - n_coeff, 1)
    return clusters / (clusters - 1) * (rows - 1) / max(rows - n_coeff, 1)


def _robust_spreads(
    sensitivity: Array,
    score: Array,
    n_coeff: int,
    clusters: np.ndarray | None = None,
    periods: np.ndarray | None = None,
) -> tuple[Array, ...]:
    """``sum_i s_i s_i'`` with ``s_i = J_i e_i``, the scores of transition ``i``'s states summed,
    for a fit linear in its target ``y``, ``J = d coeffs / d y``; under ``clusters``,
    ``sum_g s_g s_g'`` with ``s_g = sum_{i in g} J_i e_i``. Two-way, the three spreads of
    :func:`_clustered_squares`, and an error read off them is the largest of their reads.
    Under ``periods``, one group's, ``sum_j c_j c_j'`` with ``c_j = sum_i w_ji J_i e_i`` over the
    weights ``w`` of :func:`_cosines`, at the rows' factor ``N / (N - k)``: each period's
    transitions share its weight, so their scores are summed before the cosines read across the
    periods (Driscoll and Kraay 1998).

    Each row's own squared residual stands in for its noise, so it holds when the noise differs
    across rows, and ``J`` runs through the cross-fitted nuisances as well as the moment. A weight
    that loads a few rows loads the nuisance fits' error at them too, which a sandwich on the moment
    alone misses: on a log whose noise grew as ``exp(x)``, weighed by ``exp(-x)``, that sandwich
    came to 0.67 of the channel's spread over sixteen redraws of the noise, and this to 1.04. Summed
    within a cluster first, it holds as well when the scores of one cluster's rows and states
    depend on one another in any way.
    """
    return _score_spreads(jnp.einsum("pis,is->ip", sensitivity, score), n_coeff, clusters, periods)


def _score_spreads(
    scores: Array,
    n_coeff: int,
    clusters: np.ndarray | None = None,
    periods: np.ndarray | None = None,
) -> tuple[Array, ...]:
    """:func:`_robust_spreads` of the rows' scores ``(N, p)``, each a transition's states summed:
    the sum of their outer products, or of their sums within each cluster, or across the periods'
    cosines."""
    n = scores.shape[0]
    if periods is not None:
        projected = jnp.asarray(_cosines(periods)) @ scores
        return (_small_sample(n, n_coeff, None) * projected.T @ projected,)
    if clusters is None:
        # a transition's states may share its noise, so its scores are summed over them first
        return (_small_sample(n, n_coeff, None) * scores.T @ scores,)
    factor = _small_sample(n, n_coeff, _cluster_count(clusters))
    if clusters.ndim == 1:
        summed = jax.ops.segment_sum(
            scores, jnp.asarray(clusters), num_segments=int(clusters.max()) + 1
        )
        return (factor * summed.T @ summed,)
    squares = _clustered_squares(np.asarray(scores)[:, None], clusters)
    return tuple(factor * jnp.asarray(square) for square in squares)


def _cluster_count(clusters: np.ndarray) -> int:
    """``G`` in CR1's factor: the clusters' count, or two-way the smaller dimension's, which bounds
    how many independent terms the error is read from (Cameron, Gelbach and Miller 2011)."""
    return int(np.min(clusters.max(axis=0))) + 1


def _cosine_count(rows: int) -> int:
    """``nu``, the cosines ``T`` periods are read with: the most whole ones at or below
    ``0.4 T^(2/3)``, the count Lazarus, Lewis, Stock and Watson (2018) recommend, and one at the
    least. Read in whole numbers, ``1000 nu^3 <= 64 T^2``, since in floating point
    ``0.4 * 1000 ** (2 / 3)`` falls short of 40."""
    count = max(int(0.4 * rows ** (2.0 / 3.0)), 1)
    while 1000 * (count + 1) ** 3 <= 64 * rows**2:
        count += 1
    while count > 1 and 1000 * count**3 > 64 * rows**2:
        count -= 1
    return count


def _period_cosines(periods: np.ndarray) -> int:
    """``nu`` for transitions labelled ``periods``: :func:`_cosine_count` of the distinct periods,
    which on one unit are its transitions."""
    return _cosine_count(int(np.unique(periods).size))


def _cosines(periods: np.ndarray) -> np.ndarray:
    """``(nu, N)``: ``sqrt(2 / nu) cos(pi j (p_i + 1/2) / P)`` for ``j`` from 1 to ``nu``
    (:func:`_period_cosines`), at each transition's period ``p_i`` counted from the log's first,
    over the ``P`` periods it spans, a period it lacks holding no score, as Parzen (1963) reads a
    series with gaps.

    ``sum_j c_j c_j'`` with ``c_j = sum_i w_ji s_i`` is the equal-weighted cosine estimate of the
    covariance of the scores' sum (Lazarus, Lewis, Stock and Watson 2018). Over a span without
    gaps each cosine sums to nothing, so a mean of the scores, which the moment sets to nothing,
    moves none of it. Where the scores are independent each ``c_j c_j'`` reads their covariance;
    where they persist, the cosines' low frequencies read the long-run one. As the log grows at a
    fixed ``nu`` the ``c_j`` read as independent normals, so a ``t`` statistic read off the
    estimate is a ``t(nu)``. The transitions of one period share its weights, so ``c_j`` reads the
    sums of the scores within each period, whatever their dependence across the transitions of
    one period: Driscoll and Kraay's (1998) reading, here with the cosines' kernel.
    """
    count = _period_cosines(periods)
    span = int(periods.max()) + 1
    frequencies = np.arange(1, count + 1)[:, None]
    return math.sqrt(2.0 / count) * np.cos(math.pi * frequencies * (periods[None, :] + 0.5) / span)


def _degrees_of_freedom(rows: int, clusters: np.ndarray | None, periods: np.ndarray | None) -> int:
    """The ``t`` a test read off the channel's error is sized against: one fewer than the
    independent terms its spread sums, the rows or the clusters (ADR 0062), or ``nu`` over one
    group's periods, whose cosines leave out the mean the moment sets to nothing (ADR 0075)."""
    if periods is not None:
        return _period_cosines(periods)
    return (rows if clusters is None else _cluster_count(clusters)) - 1


def _clustered_squares(
    rows: np.ndarray, clusters: np.ndarray | None, periods: np.ndarray | None = None
) -> tuple[np.ndarray, ...]:
    """``rows``' covariance, ``(N, n, ...)`` per transition and state as
    :attr:`CausalDynamicsFit.influence` is: the sum of the outer products of its independent
    terms, each a transition's states summed, or under ``clusters`` a cluster's transitions and
    states summed. Under ``periods``, one group's, each transition's states summed and projected on
    the cosines of :func:`_cosines`, which sums each period's transitions, as the channel's error
    is read.

    Two-way, three covariances, and a reader takes the one it reads largest (MacKinnon, Nielsen
    and Webb 2024): the two dimensions' sums less their intersection's, which both count (Cameron,
    Gelbach and Miller 2011), less any part that reads below nothing (:func:`_less_below_nothing`);
    then each dimension's sums alone, moved from the smaller dimension's CR1 factor, which
    ``rows`` carry, to its own. The two-way sums alone can read less than either dimension does:
    on a log of 3 units over 4 periods they read nothing of a channel affine in the state.
    """
    if periods is not None:
        projected = np.tensordot(_cosines(periods), rows.sum(axis=1), axes=(1, 0))
        return (np.tensordot(projected, projected, axes=(0, 0)),)
    per_row = rows.sum(axis=1)
    if clusters is None:
        return (np.tensordot(per_row, per_row, axes=(0, 0)),)

    def square(codes: np.ndarray) -> np.ndarray:
        summed = np.zeros((int(codes.max()) + 1, *per_row.shape[1:]))
        np.add.at(summed, codes, per_row)
        return np.tensordot(summed, summed, axes=(0, 0))

    if clusters.ndim == 1:
        return (square(clusters),)
    first, second = square(clusters[:, 0]), square(clusters[:, 1])
    cells = square(np.unique(clusters, axis=0, return_inverse=True)[1].reshape(-1))
    spread, every = first + second - cells, first + second + cells
    both = np.maximum(spread, 0.0) if spread.ndim == 0 else _less_below_nothing(spread, every)
    smaller = _cluster_count(clusters)
    alone = []
    for one, codes in ((first, clusters[:, 0]), (second, clusters[:, 1])):
        count = int(codes.max()) + 1
        # the factor whole, so a dimension of the smaller count keeps its sums bit for bit
        alone.append(one * ((count / (count - 1)) / (smaller / (smaller - 1))))
    return (both, *alone)


def _less_below_nothing(spread: np.ndarray, every: np.ndarray) -> np.ndarray:
    """``spread`` less the part it reads below nothing, measured against ``every``, its sums with
    every sign a plus: ``spread - T Q min(L, 0) Q' T'``, ``every = T T'`` and ``Q L Q'`` the
    eigendecomposition of ``T^+ spread T^+'``. ``-every <= spread <= every``, so ``L`` lies in
    ``[-1, 1]``, and a spread with no part below nothing keeps every bit. ``every`` is
    ``spread`` plus twice the cells' sums, and any reference ``a spread + b cells``, the cells'
    sums alone or the units' and the periods' together among them, has the same generalised
    eigenvectors ``v``; each part taken away, ``spread v v' spread / v' spread v``, depends on
    ``v`` alone, so they all take away the same part.

    Both move with the coefficients as the scores do, so a change of their units or of the
    state's zero changes no value read off the result. The eigenvalues of ``spread`` itself, which
    Cameron, Gelbach and Miller (2011) clip, move with them: on a panel of 200 regions over 11
    periods, moving the state's zero moved the channel's error from 0.008164 to 0.009408.
    """
    values, vectors = np.linalg.eigh(every)
    kept = values > values[-1] * values.size * np.finfo(np.float64).eps
    root, whitening = (
        vectors[:, kept] * np.sqrt(values[kept]),
        vectors[:, kept] / np.sqrt(values[kept]),
    )
    shares, turns = np.linalg.eigh(whitening.T @ spread @ whitening)
    lifted = root @ turns
    return spread - (lifted * np.minimum(shares, 0.0)) @ lifted.T


def _error_at_rows(blocks: Array, features: Array) -> float:
    """Root mean square, over the log's rows and the ``G`` blocks, of the standard error of a value
    ``features[i] @ c_g`` whose coefficients ``c_g`` have the covariance ``blocks[g]``:
    ``sqrt(mean_g tr(blocks[g] M))``, ``M`` the features' mean square over the rows.

    A coefficient's own variance moves with the state's zero: the constant term is the value at
    ``x = 0``, however far from it the log was taken. A value at the log's rows does not move.
    """
    moment = features.T @ features / features.shape[0]
    # a covariance two ways read nothing in reads its zero only to rounding, either side of it
    return float(jnp.sqrt(jnp.maximum(jnp.mean(jnp.einsum("gab,ba->g", blocks, moment)), 0.0)))


def _channel_blocks(covariance: Array, channel_shape: tuple[int, ...]) -> Array:
    """``(n m, F, F)``: each channel entry's coefficients' covariance, out of the raveled
    ``(n, m, F)`` channel's."""
    entries, features = channel_shape[0] * channel_shape[1], channel_shape[2]
    every = jnp.arange(entries)
    return covariance.reshape(entries, features, entries, features)[every, :, every, :]


def _field_terms(
    covariances: Sequence[Array],
    channel_map: Array,
    conditional: Array,
    projection: Array,
    pooled: ArrayLike,
    pooled_joint: ArrayLike,
    noise: Array,
    score: Array,
    residual: Array,
    pushed: Array,
    actions: Array,
    states: Array,
    drivers: Array,
    carried: int,
    clusters: np.ndarray | None,
    periods: np.ndarray | None,
) -> _FieldTerms:
    """:class:`_FieldTerms`, the field's first-order error, of a fit whose channel's covariance
    reads are ``covariances``, ``(n q, n q)`` each.

    ``channel_map`` ``(n q, N, n)`` is the channel's weight on each row's rate and
    ``conditional`` ``(D n, N, n)`` the drift regression's, the channel held, each state's ``D``
    coefficients beside the others': every ``n``-th, from the state's. Under Euler the latter is
    the regression's own weight; under ``rk4`` both go through the fixed point, the drift's through
    the drift's block of its Newton matrix. ``projection`` ``(D n, n q)`` is how far a channel
    error moves the drift's coefficients, the regression reading its target at the fitted
    channel: ``Gamma = W G``, ``W`` the regression's weight and ``G`` the channel's regressors on
    the log, the same for each state's own channel under Euler, and under ``rk4`` that less the
    RK4 map's own move of the target with the channel, through the drift's block. The first
    ``carried`` of a state's coefficients are the field's, its features then its drivers; the rest
    hold the adjustment set's push, whose value the field leaves at its mean on the log.

    Without ``clusters`` or ``periods`` the drift's side is pooled, as ``drift_error`` pools it:
    ``pooled`` is that covariance's ``(G, C, C)`` blocks and ``pooled_joint`` the whole of it over
    every state's carried coefficients, ``(n C, n C)``, a state's beside each other, ``noise``
    ``(n,)`` the noise each
    state's residual pools, the drift's covariance with the channel is ``sum_i M_i noise W_i'``,
    nothing under Euler but for cross-fitting (Frisch and Waugh 1933), and the push's mean has the
    variance its rows give independent, ``sum pushed^2 / N^2``. With them, the channel's scores,
    ``M_i`` times the moment's ``score``, the drift's, ``W_i`` times its own ``residual``, and the
    push, ``pushed`` ``(N, n)``, are summed within each cluster or projected on the periods'
    cosines together, each at its own small-sample factor ``N / (N - k)`` for its ``k``
    coefficients a state, ``k = 1`` for a mean, so each part reads as it would alone and their
    cross-covariance between them: the long-run variance of the push's mean (Lazarus, Lewis, Stock
    and Watson 2018), which under an AR(0.8) push is about ``(1 + 0.8) / (1 - 0.8) = 9`` times what
    its rows read independent (``validation/push_mean_variance.mac``)."""
    rows, count = states.shape
    size = channel_map.shape[0]
    features = conditional.shape[0] // count

    def by_state(matrix: np.ndarray) -> np.ndarray:
        """``(n q, D n)`` to ``(n q, n, C)``: each state's carried coefficients."""
        return matrix.reshape(size, features, count).transpose(0, 2, 1)[:, :, :carried]

    if clusters is None and periods is None:
        cross = np.einsum(
            "ais,s,dis->ad",
            np.asarray(channel_map, dtype=np.float64),
            np.asarray(noise, dtype=np.float64),
            np.asarray(conditional, dtype=np.float64),
        )
        crosses = by_state(cross)[None]
        drifts = np.asarray(pooled, dtype=np.float64)[None]
        drift_joints = np.asarray(pooled_joint, dtype=np.float64)[None]
        moved = np.asarray(pushed, dtype=np.float64)
        pushes = (np.sum(moved**2, axis=0) / rows**2)[None]
        push_joints = (moved.T @ moved / rows**2)[None]
    else:
        q = size // count
        stacked = jnp.concatenate(
            [
                jnp.einsum("ais,is->ia", channel_map, score)
                * math.sqrt(_small_sample(rows, q, None)),
                jnp.einsum("dis,is->id", conditional, residual)
                * math.sqrt(_small_sample(rows, features, None)),
                pushed * math.sqrt(_small_sample(rows, 1, None)),
            ],
            axis=1,
        )
        joints = [
            np.asarray(joint, dtype=np.float64)
            for joint in _score_spreads(stacked, 0, clusters, periods)
        ]
        drift = slice(size, size + features * count)
        every = np.arange(count)
        crosses = np.stack([by_state(joint[:size, drift]) for joint in joints])
        drifts = np.stack(
            [
                joint[drift, drift].reshape(features, count, features, count)[:, every, :, every][
                    :, :carried, :carried
                ]
                for joint in joints
            ]
        )
        drift_joints = np.stack(
            [
                joint[drift, drift]
                .reshape(features, count, features, count)[:carried, :, :carried, :]
                .transpose(1, 0, 3, 2)
                .reshape(count * carried, count * carried)
                for joint in joints
            ]
        )
        push_joints = np.stack([joint[drift.stop :, drift.stop :] / rows**2 for joint in joints])
        pushes = np.diagonal(push_joints, axis1=1, axis2=2).copy()
    return _FieldTerms(
        mean_action=np.asarray(jnp.mean(actions, axis=0), dtype=np.float64),
        projection=np.asarray(projection, dtype=np.float64)
        .reshape(features, count, size)
        .transpose(1, 0, 2)[:, :carried],
        channel=np.stack([np.asarray(covariance, dtype=np.float64) for covariance in covariances]),
        drift=drifts,
        cross=crosses,
        push=pushes,
        states=np.asarray(states, dtype=np.float64),
        drivers=np.asarray(drivers, dtype=np.float64),
        drift_joint=drift_joints,
        push_joint=push_joints,
    )


def _drift_reads(terms: _FieldTerms, design: Array) -> tuple[float, np.ndarray]:
    """``drift_error`` and the blocks it is read off where the drift's side is read as the
    channel's is: the largest of :func:`_error_at_rows`' reads of ``terms.drift`` at ``design``,
    the log's rows of the drift's carried columns, and that read's blocks."""
    reads = [_error_at_rows(jnp.asarray(blocks), design) for blocks in terms.drift]
    largest = int(np.argmax(reads))
    return reads[largest], terms.drift[largest]


def _channel_errors(
    covariances: Sequence[Array], channel_shape: tuple[int, ...], features: Array
) -> tuple[float, np.ndarray, np.ndarray]:
    """``channel_error``, the largest of the covariances' reads by :func:`_error_at_rows` over the
    channel's entries, and ``(r, m)``: each covariance's read over each lever's column alone, whose
    entries are every ``m``-th of :func:`_channel_blocks`', from the lever's. With one lever the
    column is the whole channel, and its read is the covariance's, bit for bit. Last, the ``(r, n m,
    F, F)`` blocks both are read off, which :attr:`CausalDynamicsFit._channel_blocks` keeps."""
    levers = channel_shape[1]
    blocks = [_channel_blocks(covariance, channel_shape) for covariance in covariances]
    by_lever = np.array(
        [[_error_at_rows(block[j::levers], features) for j in range(levers)] for block in blocks]
    )
    return (
        max(_error_at_rows(block, features) for block in blocks),
        by_lever,
        np.asarray(jnp.stack(blocks), dtype=np.float64),
    )


def _influence(
    sensitivity: Array,
    score: Array,
    direct: Array,
    drift_score: Array,
    n_coeff: int,
    clusters: np.ndarray | None = None,
) -> Array:
    """``psi[i, s] = sqrt(n / (n - k)) (J[:, i, s] e[i, s] + D[:, i, s] (r[i, s] - e[i, s]))``,
    shape ``(N, n, p)``: each row's and state's share of the error. ``J`` is the whole fit's
    response to the row's rate and ``D`` the part of it that reaches the drift directly, not
    through the channel, so the channel reads the moment's residual ``e`` and the drift's direct
    part its own regression's residual ``r``, which holds what the drift's features leave. Under
    ``clusters`` the factor is CR1's (:func:`_small_sample`), for sums within each cluster."""
    n = score.shape[0]
    count = None if clusters is None else _cluster_count(clusters)
    return jnp.sqrt(_small_sample(n, n_coeff, count)) * (
        jnp.einsum("pis,is->isp", sensitivity, score)
        + jnp.einsum("pis,is->isp", direct, drift_score - score)
    )


def _representer(sensitivity: Array) -> Array:
    """``(q, N, n)`` weights of the channel's coefficients on each row's rate, as ``(N, n, q)``
    times ``N``, the scale whose mean square is the representer's second moment."""
    return sensitivity.shape[1] * jnp.transpose(sensitivity, (1, 2, 0))


def _kept(residual: Array, scale: Array) -> Array:
    """Unit columns ``(k, r)``, in the units ``scale`` divides out: the combinations of
    ``residual``'s columns that, scaled, keep less than the square root of the working precision."""
    residual = residual * scale
    # A log of fewer transitions than coefficients has fewer singular values than directions, and a
    # thin SVD returns no row for the rest, which no transition moves: zero rows bring them back
    # with a singular value of 0.
    short = residual.shape[1] - residual.shape[0]
    if short > 0:
        residual = jnp.concatenate([residual, jnp.zeros((short, residual.shape[1]))])
    _, singular, rows = jnp.linalg.svd(residual, full_matrices=False)
    null = rows[singular <= jnp.sqrt(jnp.finfo(singular.dtype).eps)].T * scale[:, None]
    return null / jnp.linalg.norm(null, axis=0)


def _unmoved_directions(
    actions: Array,
    states: Array,
    covariates: Array,
    nuisance_degree: int,
    channel_degree: int,
    sieve: Array | None = None,
) -> Array:
    """The channel's coefficient directions, unit columns ``(m k, r)``, along which the log's
    actions, less their least-squares projection on the nuisance's features (``sieve`` where
    given, :func:`_nuisance_sieve`), keep less than the square root of the working precision of
    their raw size.

    Projected without the cross-fit's ridge and folds, a policy the covariates determine leaves
    rounding, where the ridge would leave a bias that grows as the log shrinks. Scaled to the raw
    actions, the test reads the same in any units. An action the log never used has a raw column of
    zeros, and its coefficients come back whole."""
    features = (
        _polynomial_features(_units.standardised(covariates), nuisance_degree)
        if sieve is None
        else sieve
    )
    left = actions - features @ jnp.linalg.lstsq(features, actions)[0]
    scale = 1.0 / _units.norms(_channel_design(actions, states, channel_degree))
    return _kept(_channel_design(left, states, channel_degree), scale)


def _instrument_relevance(
    actions: Array,
    states: Array,
    covariates: Array,
    shifter: Array,
    nuisance_degree: int,
    channel_degree: int,
    free: Array | None,
    weights: Array | None,
    sieve: Array | None = None,
    first: Array | None = None,
) -> Array:
    """How far the instrument ``shifter`` moves each direction of the channel: ``(p,)``, along
    ``free``'s ``p`` columns, or every coefficient where it is None. They are the canonical
    correlations, largest first, between the first stage's push on the actions and the actions,
    each less its least-squares projection on the nuisance's features, on the channel's features,
    each row weighed by ``weights``: the singular values of the unregularised moment with each side
    whitened, so the count of those above 0 is its rank.

    The push is the actions' least-squares fit on the first stage's features of the state and the
    instrument, ``first`` (:func:`_first_stage`; the polynomial of the two where None), without
    the ridge. A push of at most ``chc._units.ROUNDING`` eps of the raw
    actions' size along a direction is none, and so is a correlation of at most
    ``chc._units.ROUNDING`` eps: both read 0. Scaled to the raw actions, on features standardised
    as the nuisance's are, and whitened, the reading is the same in any units of the instrument,
    the actions and the state."""
    raw = _channel_design(actions, states, channel_degree)
    along = jnp.diag(1.0 / _units.norms(raw)) if free is None else free
    correlations = jnp.zeros(along.shape[1], dtype=raw.dtype)
    if along.shape[1] == 0:
        return correlations
    rounding = _units.ROUNDING * jnp.finfo(raw.dtype).eps
    features = (
        _polynomial_features(_units.standardised(covariates), nuisance_degree)
        if sieve is None
        else sieve
    )
    if first is None:
        first = _first_stage(states, shifter, nuisance_degree, None)
    root = jnp.ones(raw.shape[0], dtype=raw.dtype) if weights is None else jnp.sqrt(weights)

    def span(columns: Array) -> Array:
        left = columns - features @ jnp.linalg.lstsq(features, columns)[0]
        design = root[:, None] * _channel_design(left, states, channel_degree) @ along
        out, singular, _ = jnp.linalg.svd(design, full_matrices=False)
        return out[:, singular > rounding]

    pushed, moved = span(first @ jnp.linalg.lstsq(first, actions)[0]), span(actions)
    if pushed.shape[1] and moved.shape[1]:
        found = jnp.linalg.svd(pushed.T @ moved, compute_uv=False)
        correlations = correlations.at[: found.size].set(found)
    return jnp.where(correlations > rounding, correlations, 0.0)


def _unmoved_parameters(
    directions: Array, features: Array, actions: Array, design: Array, ridge: float, states: int
) -> Array:
    """Each unmoved direction of the channel, state by state, as a move of every parameter in
    :attr:`CausalDynamicsFit.influence`'s order: the channel's own, and the drift regression's
    response to it on this log."""
    shape = (states, actions.shape[1], features.shape[1])
    moves = []
    for state in range(states):
        for direction in directions.T:
            change = jnp.zeros(shape).at[state].set(direction.reshape(shape[1:]))
            pushed = jax.vmap(lambda c, u, change=change: (change @ c) @ u)(features, actions)
            response = _solve_ridge(design, pushed, ridge)
            moves.append(jnp.concatenate([change.ravel(), -response.ravel()]))
    size = int(np.prod(shape)) + design.shape[1] * states
    return jnp.stack(moves, axis=1) if moves else jnp.zeros((size, 0))


@dataclass(frozen=True)
class _Unmoved:
    """The span of the channel's unmoved directions, split by what the drift regression does with
    their push on the log's raw actions (ADR 0054).

    ``basis`` ``(m k, r)`` spans it and ``free`` ``(m k, m k - r)`` the rest of the channel, the
    columns of both orthonormal once scaled to the raw actions' size, so the split reads the same
    in any units. The channel's moment is solved on ``free`` alone, where it has data. ``taken``
    ``(r, t)`` holds the combinations of ``basis`` whose push the drift's features take up exactly,
    by least squares, so no rate of the log tells them apart, and the fit leaves the channel at
    zero along them. ``pinned`` ``(m k, N)`` reads the rest off a rate, as least squares beside the
    drift's features does: there the log rules out all but one value."""

    raw: Array  # (N, m k): the channel's design on the log's raw actions
    basis: Array
    free: Array
    taken: Array
    pinned: Array

    def hold(self, channel: Array, rate: Array) -> Array:
        """``channel`` ``(n, m, k)``, solved on ``free``, with what the log's ``rate`` ``(N, n)``
        reads along ``basis`` beside the drift added. Linear in both."""
        rows = channel.reshape(channel.shape[0], -1)
        rows = rows + (self.pinned @ (rate - self.raw @ rows.T)).T
        return rows.reshape(channel.shape)


def _split_unmoved(directions: Array, raw: Array, design: Array) -> _Unmoved:
    """:class:`_Unmoved` for the unit columns ``directions`` ``(m k, r)``, with ``raw`` the
    channel's design on the log's raw actions and ``design`` the drift regression's."""
    scale = 1.0 / _units.norms(raw)
    unit = scale[:, None] * jnp.linalg.qr(directions / scale[:, None], mode="complete")[0]
    basis, free = unit[:, : directions.shape[1]], unit[:, directions.shape[1] :]
    push = raw @ basis
    # in each column's own units, so the rank cutoff falls at one share of every column: on the
    # raw columns, relative to the largest singular value, it dropped a column logged in units far
    # from the others'
    left = push - design @ _units.least_squares(design, push)
    # as in _kept: a log of fewer transitions than directions leaves the rest a zero singular value
    short = left.shape[1] - left.shape[0]
    padded = jnp.concatenate([left, jnp.zeros((short, left.shape[1]))]) if short > 0 else left
    out, singular, rows = jnp.linalg.svd(padded, full_matrices=False)
    seen = singular > jnp.sqrt(jnp.finfo(singular.dtype).eps)
    return _Unmoved(
        raw=raw,
        basis=basis,
        free=free,
        taken=rows[~seen].T,
        pinned=basis @ (rows[seen].T / singular[seen]) @ out[: left.shape[0], seen].T,
    )


def _warn_unmoved(fit: CausalDynamicsFit, directions: Array, identified: bool) -> None:
    """Log, at WARNING, the levers the log never moves along some direction of the channel, by
    their columns of ``u``: those whose whole channel it never moves (:func:`_unmoved_actions`),
    and those an unmoved direction ``directions`` ``(m k, r)``, scaled to the raw actions' size,
    mixes."""
    actions, features = fit.residual.channel.shape[1:]
    blocks = np.asarray(directions).reshape(actions, features, -1)
    precision = float(np.sqrt(np.finfo(blocks.dtype).eps))
    mixed = tuple(a for a in range(actions) if float(np.linalg.norm(blocks[a])) > precision)
    whole = _unmoved_actions(fit)
    _log.warning(
        "the log never moves the levers in columns %s of u along %d of the channel's %d "
        "directions, apart from what the states and covariates predict (the whole channel of "
        "%s); the fit holds the channel to the log there, which is not an effect%s",
        list(mixed),
        directions.shape[1],
        directions.shape[0],
        list(whole),
        "" if identified else ", and identifies no direction of it",
        extra={
            "chc_event": "fit_unmoved",
            "levers": list(mixed),
            "whole": list(whole),
            "directions": int(directions.shape[1]),
            "identified": identified,
        },
    )


def _unmoved_actions(fit: CausalDynamicsFit) -> tuple[int, ...]:
    """The actions whose whole channel the log never moved: every coefficient of theirs, on every
    feature of the channel, in the span of :attr:`CausalDynamicsFit.unmoved`'s directions.

    The span is read where the fit read it, each coefficient scaled to its column of the channel's
    design on the log's raw actions, where the directions are orthogonal. In raw coefficient units
    they lean together as the units of the state and of the actions set those columns' sizes apart:
    with the state at 1e-12 of its units, QR there missed the span by 2.5e-5, and a lever the log
    set from the state read as moved; with one lever at 1e9 of its units, a direction both share
    lay 2e-9 of the way along it, under the precision, and the other read as never moved."""
    states, actions, features = fit.residual.channel.shape
    if fit.unmoved is None or fit.unmoved.shape[1] == 0:
        return ()
    assert fit._parameter_size is not None  # the fit sets it with unmoved
    # every state's channel is moved along the same directions, so the first state's block of its
    # first moves holds them all
    width = actions * features
    directions = fit.unmoved[:width, : fit.unmoved.shape[1] // states]
    # the first state's channel: its columns' norms over that state's, one factor for the block
    basis = jnp.linalg.qr(directions * fit._parameter_size[:width, None])[0]
    moved = jnp.eye(width) - basis @ basis.T
    precision = float(jnp.sqrt(jnp.finfo(basis.dtype).eps))
    return tuple(
        action
        for action in range(actions)
        if float(jnp.linalg.norm(moved[:, action * features : (action + 1) * features], 2))
        <= precision
    )


@dataclass(frozen=True)
class _LoggedRelations:
    """What the log's actions kept to: orthonormal bases ``(m, r)``, over the actions each divided
    by ``size``, of the combinations it kept at one level, of those the state alone predicts, and of
    those its covariates predict, each span holding the one before; and each action's least-squares
    rule of the state, the nuisance's polynomial of the state standardised as the log's was."""

    constant: np.ndarray
    state: np.ndarray
    covariates: np.ndarray
    # (m,): each action's norm on the log, 1 for one never used. A basis column over it is a
    # combination's weights on the actions as logged.
    size: np.ndarray
    means: np.ndarray  # (m,): a constant combination's level is its weights times these
    # (n,) each: the state standardised as (x - centre - shift) * factor (chc._units.standardising)
    centre: Array
    shift: Array
    factor: Array
    rule: Array  # (features, m)


def _logged_relations(
    actions: Array, states: Array, covariates: Array, nuisance_degree: int
) -> _LoggedRelations:
    """The combinations of the log's actions that a constant, the state or the covariates predict
    to the precision :func:`_unmoved_directions` reads, projected the same way.

    Each span is orthonormal over the actions scaled to their size, where :func:`_kept` reads it.
    Orthonormalised over the raw actions, a span moved with their units: with one action at 1e12 of
    its units, the span the state predicts lay 3.0e-7 off that action's axis, past the square root
    of the precision, and ``u2 = 2 u1`` read its weight on ``u1`` as -1.99996e-12 where -2e-12."""
    size = _units.norms(actions)
    scale = 1.0 / size

    def kept(features: Array) -> np.ndarray:
        left = actions - features @ jnp.linalg.lstsq(features, actions)[0]
        columns = np.asarray(_kept(left, scale) * size[:, None], dtype=np.float64)
        return np.linalg.qr(columns)[0] if columns.shape[1] else columns

    centre, shift, factor = _units.standardising(states)
    features = _polynomial_features((states - centre - shift) * factor, nuisance_degree)
    return _LoggedRelations(
        constant=kept(jnp.ones((actions.shape[0], 1), dtype=actions.dtype)),
        state=kept(features),
        covariates=kept(_polynomial_features(_units.standardised(covariates), nuisance_degree)),
        size=np.asarray(size, dtype=np.float64),
        means=np.asarray(jnp.mean(actions, axis=0), dtype=np.float64),
        centre=centre,
        shift=shift,
        factor=factor,
        rule=jnp.linalg.lstsq(features, actions)[0],
    )


def _absorbed(
    fit: CausalDynamicsFit, states: Array, actions: Array, drift_design: Array
) -> tuple[Array, Array]:
    """The moves of the fit no transition of the log tells from it (ADR 0054): the unmoved channel
    directions (:attr:`CausalDynamicsFit.unmoved`) whose push on the log's own actions the drift
    regression takes up exactly, ``(m k, r)``, and that regression's response, ``(features +
    drivers, r)``.

    The response is by least squares, not the fit's ridge, so each move leaves every predicted rate
    of the log as it was to rounding. A direction whose push the drift cannot take up moves the
    log's own predicted rates, so within the model class the log rules it out, though the channel's
    moment has no data along it, and the fit reads it off the log's rate (:class:`_Unmoved`, the
    same split). Scaled to the push its coefficients would make on actions of the log's raw size,
    the test reads the same in any units.

    The least squares reads each column in its own units, as the split does. On the raw columns it
    left 1.6e-3 of a push with the state at 1e-12 of its units, and all of it at 1e-13, where the
    rank cutoff dropped the state's columns."""
    states_n, levers, features = fit.residual.channel.shape
    width = levers * features
    if fit.unmoved is None or fit.unmoved.shape[1] == 0:
        return jnp.zeros((width, 0)), jnp.zeros((drift_design.shape[1], 0))
    raw = _channel_design(actions, states, fit.residual.channel_degree)
    split = _split_unmoved(
        fit.unmoved[:width, : fit.unmoved.shape[1] // states_n], raw, drift_design
    )
    taken = split.basis @ split.taken
    push = raw @ taken
    # A push that cancels to the working precision of its terms is none: its regression would read
    # the rounding as a response, which no action at the plan's points could cancel.
    precision = jnp.sqrt(jnp.finfo(push.dtype).eps)
    size = jnp.linalg.norm(jnp.abs(raw) @ jnp.abs(taken), axis=0)
    none = jnp.linalg.norm(push, axis=0) <= precision * size
    return taken, _units.least_squares(drift_design, jnp.where(none, 0.0, push))


def _nuisance_inputs(
    data: dict[str, Array],
    adjust_for: Sequence[str],
    drivers: Sequence[str],
    integrator: Integrator,
) -> tuple[Array, Array, Array, Array]:
    """``(covariates, start, end, read)``: what the nuisances read -- the state, the adjustment
    set, each driver not in it at the transition's start, and every driver at its end -- and the
    drivers at the transition's two ends and as its rate reads them."""
    x = data["x"]
    start = jnp.concatenate([x[:, :0], *[data[name] for name in drivers]], axis=1)
    end = jnp.concatenate([x[:, :0], *[data[f"{name}_next"] for name in drivers]], axis=1)
    # What a transition's rate sees of a driver: its mean over the step, exact for the linear path
    # between the two ends, or its value at the start, which is all the Euler map reads.
    read = start if integrator == "euler" else 0.5 * (start + end)
    unadjusted = [data[name] for name in drivers if name not in adjust_for]
    covariates = jnp.concatenate(
        [x, *[data[name] for name in adjust_for], *unadjusted, end], axis=1
    )
    return covariates, start, end, read


def _state_weights(weights: Callable[[Array], Array], states: Array) -> Array:
    """The caller's weight at each state, checked and scaled to mean 1, so ``ridge`` keeps its
    meaning whatever the weight's units."""
    if not callable(weights):
        raise TypeError(f"weights must be a function of the state, or None; got {weights!r}")
    values = jnp.asarray(weights(states), dtype=states.dtype)
    if values.shape != (states.shape[0],):
        raise ValueError(
            f"the weight must give one value per transition, shape ({states.shape[0]},), from the "
            f"states alone; got shape {values.shape}"
        )
    if not bool(jnp.all(jnp.isfinite(values))) or bool(jnp.any(values < 0.0)):
        raise ValueError("the weight must be finite and non-negative at every state")
    if float(jnp.sum(values)) <= 0.0:
        raise ValueError("the weight is zero at every state")
    return values / jnp.mean(values)


def _ols_blocks(target: Array, design: Array, coeffs: Array, ridge: float) -> Array:
    """``(1, F, F)``: the drift regression's coefficients' covariance under ``sigma^2 (X'X)^-1``,
    the plain homoskedastic OLS covariance, its noise pooled over the rows and the states, one
    block every state shares. :func:`_error_at_rows` reads it at the log's rows as ``drift_error``.

    Used for the drift stage, which is fitted by least squares on the remainder, so there is no
    instrument and no endogenous regressor to sandwich against. Unlike the channel's robust error it
    pools the noise over the rows and the states; a scale rather than a coverage statement.
    """
    n, n_coeff = design.shape
    score = target - design @ coeffs
    sigma2 = jnp.sum(score**2) / (max(n - n_coeff, 1) * target.shape[1])
    return (sigma2 * _ridge_inverse(design, ridge))[None]


def solve_channel_moment(
    state_residual: Array,
    action_residual: Array,
    states: Array,
    *,
    instrument_action: Array | None = None,
    degree: int = 1,
    ridge: float = 1e-6,
    weights: Array | None = None,
) -> Array:
    """Solve ``E[(y_res - B_θ(x) u_res) (x) (z (x) phi(x))] = 0`` for the channel.

    The Robinson moment on its own, separated from the nuisance estimation
    :func:`fit_causal_residual` wraps around it. Public for two reasons: cross-fitted
    ridge-sieve nuisances are a default rather than a commitment, and the point of a debiased
    score is that ``g`` and ``m`` may come from any learner -- gradient boosting, a neural net, a
    model the caller already had -- so the caller needs a way in that does not go through ours.

    It solves the moment it is handed and checks no rank. Along a direction of the channel that
    ``instrument_action`` does not move, the moment holds at every channel, and the ridge picks
    one. Whether an instrument identifies the channel is the caller's to settle before the solve,
    which sees neither the raw shifter nor the covariates the nuisances took out, and a rank that
    decides it reads both.

    Args:
        state_residual: ``y - g(x, z)``, shape ``(N, n)``.
        action_residual: ``u - m(x, z)``, shape ``(N, m)`` -- the regressor.
        states: ``x``, shape ``(N, n)`` -- the channel's own feature argument, which is *not*
            residualised: ``B_θ`` is allowed to depend on the state.
        instrument_action: the action-shaped variable ``z`` that enters the moment, when it differs
            from the regressor: the projection of the action on an exogenous shifter, for the case
            where the confounder is latent. ``None`` means the action residual instruments itself,
            which is the orthogonal (adjusted) case.
        degree: monomial degree of ``B_θ``'s dependence on the state.
        ridge: Tikhonov term on the moment's Gram matrix, on each coefficient scaled by its
            regressor column's mean square, weighted as the rows are, so that it reads the same in
            any units of the actions and the state. Added as a constant, it outweighed actions
            logged in units a millionth of their own, and set their channel near zero.
        weights: one weight per transition, shape ``(N,)``, on its moment; ``None`` weighs all
            alike. A weight that is a function of the state alone keeps the moment orthogonal (see
            :func:`fit_causal_residual`). Experimental: it may change or be withdrawn in any
            release.

    Returns:
        The channel coefficients, shape ``(n, m, n_features)``, consumable directly as
        :attr:`chc.residual.ControlAffineResidual.channel`.
    """
    regressor = _channel_design(action_residual, states, degree)
    moment = (
        regressor
        if instrument_action is None
        else _channel_design(instrument_action, states, degree)
    )
    penalty = jnp.diag(_units.mean_squares(regressor, weights))
    coeffs = _channel_coefficients(state_residual, regressor, moment, ridge, penalty, weights)
    n_features = regressor.shape[1] // action_residual.shape[1]
    return coeffs.T.reshape(state_residual.shape[1], action_residual.shape[1], n_features)


def fit_causal_residual(
    known: Dynamics,
    data: dict[str, Array],
    dt: float,
    *,
    adjust_for: tuple[str, ...] = (),
    instrument: str | None = None,
    unconfounded: bool = False,
    degree: int = 1,
    channel_degree: int | None = None,
    nuisance_degree: int = 2,
    nuisance_knots: int | Literal["auto"] | None = "auto",
    folds: int = 2,
    ridge: float = 1e-6,
    seed: int = 0,
    integrator: Integrator = "euler",
    drivers: tuple[str, ...] = (),
    weights: Callable[[Array], Array] | None = None,
    influence: bool = False,
    clusters: np.ndarray | Array | None = None,
    periods: ArrayLike | None = None,
    times: ArrayLike | None = None,
) -> CausalDynamicsFit:
    """Fit a :class:`ControlAffineResidual` whose channel is the *interventional* control response.

    Args:
        known: the physics kept fixed; its own control dependence is known, not estimated.
        data: columns ``x (N,n)``, ``u (N,m)``, ``x_next (N,n)``, plus any named in ``adjust_for``
            and ``instrument``.
        adjust_for: observed confounders. Empty, with no instrument and ``unconfounded`` false,
            => ``identified=False``. ``identified`` says what the fit was handed, not that the set
            is enough: a column of noise adjusted for reads identified.

            Each covariate that is not a driver also joins the drift regression, held over the
            step at its value at the start, about its mean on the log, under either integrator. A
            push the drift leaves out moves the drift by its lean on the state. On ``x' = x +
            (-0.7 x + 0.8 u + z)``, the Euler map at ``dt = 1``, a confounder ``z`` leaning on the
            state by 0.5 and left out of the drift, as ``"euler"`` left it up to 0.20.1, set the
            decay at -0.195 over 60 logs of 1000 transitions, its 95 % interval covering -0.7 on
            none; drawn apart from the state, it left the drift's error 99 times what it is with
            the push held. Held, the decay reads -0.700 on both, covering on 93 % and 98 %. Under
            ``"rk4"`` the drift also reaches the channel, through RK4's growth factor, where the
            moment cannot see it: on ``x' = -0.7 x + 0.8 u + z`` at ``dt = 1``, the confounder
            leaning on the state by 0.5 and left out of the drift set the channel at 0.618 over 60
            logs, its 95 % interval covering 0.800 on none; drawn apart from the state, it
            scattered the channel 13 times as far as its error said, which covered on 10 %. Held
            in the drift, both read 0.800 and covered on 97 %.
            The push is held in the nuisances' own basis in those covariates alone: their
            polynomial of ``nuisance_degree``, at 1 at the least, and their spline where
            ``nuisance_knots`` gives one. Held linearly, as up to 0.19.0, a push the nuisances
            hold and the drift did not, ``2 z + 1.5 (z^2 - 1)`` on ``x' = -7 x + 8 u`` at
            ``dt = 0.1``, scattered the channel 42 times as far as its error said over 100 logs of
            1000 transitions, which covered on 3 %; held in their basis, 1.02 times over 60,
            covering on 93 %, as the linear push does. A push that moves with the state, ``x z``,
            is not one a step holds, and is not held. The returned field carries no covariate:
            its drift is the one at their mean, and its ``defect`` keeps their push.

            A push outside the polynomial's span, ``3 tanh(2 z)`` on that plant, is in neither
            with the polynomial alone (``nuisance_knots=None``), and ``channel_error`` reads it in
            the moment's residual as if it were noise: there the channel scattered 0.73 as far as
            its error said under ``"euler"`` and 0.75 under ``"rk4"``, and its interval covered on
            98 % of 60 logs. Conservative, not wrong; a push the nuisances miss is a sign they
            miss the policy too (:func:`nuisance_shift`). At the default the spline was read on
            28 of those 60 logs, and there it holds the push: under ``"euler"`` the channel's
            error came out 40 to 62 times smaller on the four of the first twelve logs that read
            it, and over the 60 the interval covered on 97 % under either integrator.
        instrument: name of an exogenous action shifter, for when the confounder is latent. It
            enters as the moment's instrument, not as the regressor, so this is real 2SLS. Expect a
            **variance premium**, not a free lunch: identification here rides on however much of
            the action the shifter explains, and on the reference DGP that is 18%, which costs
            roughly an order of magnitude in channel error against adjusting for a logged
            confounder (0.10 vs 0.002 at ``N=4000``). Still ~10x better than not identifying at all.

            It identifies the channel only where its moment has rank: where the first stage's push
            on the actions moves with the actions along every direction of the channel the actions
            move (all but ``unmoved``'s), each less what the covariates' features predict, read
            without the ridge and in any units (:attr:`CausalDynamicsFit.instrument_relevance`),
            and the actions move one at least. Short of that rank, the moment holds at every
            channel along a direction it misses, and the fit reads as one with no adjustment and
            no instrument: ``identified=False``, ``method="observational"``, the channel of the
            action residuals as their own instrument, kept to compare, and no error. It does so
            beside ``adjust_for`` too: naming an instrument says the covariates leave a
            confounder, so the fit does not fall back on them. Without the check an instrument of
            zeros read identified, the channel set by the moment's ridge and noise: on the
            reference plant ``[0.81, -0.06]`` with an error of 6.0, where the truth is
            ``[1.0, 0.5]``. At ``nuisance_degree=0`` the first stage is a constant, which no
            instrument enters, so no instrument has rank there.

            The rank does not grade an instrument: a column of noise drawn apart from the action
            keeps it, and reads identified with a channel that misses. Its relevance grades it: on
            the reference plant such columns read 0.002 to 0.035, the medians of eight at 500,
            2000 and 8000 rows, of the order of ``1 / sqrt(N)``, where the plant's instrument read
            0.39 to 0.48. No inference here is robust to a weak instrument.
        unconfounded: the caller's claim that the state and the drivers alone leave the actions
            unconfounded -- a randomised lever, or one set by the state alone -- as a graph whose
            adjustment set is empty proves. The fit is then the orthogonal one with the state and
            the drivers as its only covariates, ``identified=True`` with its errors, as with an
            adjustment set. Without it an empty set read ``"observational"``, with no error and no
            moment residual, though its channel is the same number. Not with ``instrument``, which
            says the actions are confounded.
        degree: the feature degree of the drift, and of the channel unless ``channel_degree``
            says otherwise. ``1`` fits an affine drift and, by default, a channel affine in the
            state, which contains the constant channel the orthogonality results cover without
            being restricted to it.
        channel_degree: the feature degree of the channel; ``None`` is ``degree``. ``0`` fits the
            constant channel the orthogonality results cover beside a drift of degree ``degree``:
            ``n m``
            coefficients where the affine channel has ``n m (n + 1)``, and as many fewer tangents
            in the ``rk4`` fixed point. An affine channel is as accurate as a constant one only
            where the log is centred. On eight logs of :class:`chc.zones.ZoneMarketSystem`, against
            one RK4 period's response (``scripts/bench_channel_degree.py``), the constant channel
            was 3.6% (linear matching) and 4.6% (harmonic) off in relative RMS wherever it was
            read; the affine one was as close at the logs' mean state, 16% off at the do-nothing
            point and 159% and 123% at ``x = 0``, where :meth:`chc.decision.Prescription.reach`
            read it up to 0.12.
        nuisance_degree: flexibility of ``g`` and ``m``. Richer nuisances are the whole point of
            cross-fitting -- orthogonality is what makes their error enter only at second order.
            Second order in an error that shrinks: the nuisances are polynomials of this fixed
            degree, and a policy or a drift outside their span leaves an error that does not
            shrink with the log, and a bias in the channel that does not either, which
            ``channel_error`` does not cover. On a lever set by ``-1.5 tanh(2 x)`` and a drift
            with ``sin(2 x)`` in it, with the polynomial alone (``nuisance_knots=None``) at the
            default degree 2, the channel came back 0.61 below its truth of 1 at 1000 rows and
            0.62 at 4000, 26 and 57 of its errors, on none of twelve logs within its 95 %
            interval. Refit at a higher degree, or with more knots, and read how far the channel
            moves in its errors (:func:`nuisance_shift`): a move of more than one or two says the
            nuisances miss. There, at 4000 rows, degree 3 moved the polynomial's by 0.22, 20 of
            its errors, and degree 5 by 0.44. A higher degree is no cure: a polynomial of high
            degree fitted on one fold runs off in the other's tails, and at degrees 5 to 11 the
            channel scattered up to 2.3 times as far as its error said.
        nuisance_knots: a natural cubic spline in each covariate, at this many of its quantiles,
            beside the polynomial of ``nuisance_degree``; ``"auto"``, the default, takes
            ``ceil(N^(1/3))`` knots, which grow with the log; ``None`` adds none, and the
            nuisances are the polynomial alone, the default up to 0.19.0. The spline's error at a
            smooth nuisance shrinks as its knots grow, so the moment's second-order bias shrinks
            faster than its error. The spline with a covariate's powers
            past the first, the spline without them, or the polynomial alone is read, whichever
            predicts the other fold the better (:func:`_chosen_sieve`): beside a spline the powers
            are read off the rows past the outer knots alone, and on heavy tails a row of the
            other fold far out read them at the power of its distance; without them a quadratic
            nuisance is a line there. On the plant above, ``"auto"`` left a bias of 0.0021, 0.0003
            and 0.0002 at 1000, 4000 and 16000 rows, 0.13, 0.04 and 0.08 of the channel's error,
            and its 95 % interval covered on 0.960 and 0.965 of 200 logs and 0.967 of 30. Where
            the polynomial holds the truth it scattered as wide as the polynomial alone, or less:
            1.00 times on a linear plant over 200 logs of 1000 rows, both covering on 0.965; 1.01
            and 1.00 times at 500 and 2000 rows with normal covariates, 0.67 and 1.00 with
            ``t(3)`` ones, 0.43 and 0.76 with lognormal ones, and 1.00 with a nuisance quadratic
            in a normal or ``t(5)`` confounder at 1000 and 4000 rows, each over 100 logs.
            The spline is additive: the covariates' joint shape beyond it is the polynomial's, so a
            confounder that enters through a nonlinear interaction of two covariates is still
            outside the span. Read :func:`nuisance_shift` against a fit with more knots or a
            higher degree. At ``None``, an identified, unweighted fit without an instrument solves
            its channel's moment once more on ``"auto"``'s nuisances, at its own target, and warns
            where that moves the channel by more than three errors of the move, or as far into a
            ``t``'s tail where the error is read off few clusters or periods, ``t(9)``'s 4.1: the
            polynomial then misses the log's nuisances, and ``channel_error`` does not cover the
            bias. The fit returned is the polynomial's. Under ``euler`` that channel is a refit's;
            under ``rk4`` one Newton step through the fit's own Newton matrix carries it to a
            refit's fixed point. The move's error pairs the two fits' scores row by row, so it
            holds whichever fit is the more efficient. It fits nothing more. On the plant above,
            at 2000 rows, the move read 31 to 44 of its errors and the fit warned on 40 of 40
            logs; on 200 logs of the linear plant it read 0.90 at most, and 1.6 where the noise
            grows as ``exp(x)``, and warned on none. Read in the fit's own errors, as up to
            0.19.0, a spline that scatters more than the polynomial moved it past three on 12.5 %
            of linear logs with ``t(3)`` covariates, and a log the known physics solves exactly
            read a ratio of two roundings. Under a weight ``exp(-x)`` that reading passed three
            on 23 %, so a weighted fit is not checked.
        folds: cross-fitting folds. ``1`` fits the nuisances on the same rows it residualises.
            That is *not* biased for the default nuisances and should not be sold as such:
            residualising ``y`` and ``u`` by the same linear projection whose span contains the
            truth is Frisch-Waugh-Lovell, so own-sample partialling-out is exactly unbiased and
            out-of-fold prediction only adds variance -- measurably worse at small ``N``.
            Cross-fitting earns its keep against learners whose fit is adaptive to the sample
            (feature selection, trees, early stopping) or saturated enough to memorise it, where
            own-sample residuals collapse; ``folds>=2`` is the safe default for that reason.
            With one fold the sieve's choice (``nuisance_knots``) reads its own rows, where the
            spline with the powers, the larger basis, always predicts better and is read.
        ridge: the Tikhonov term of the nuisances, the channel's moment, the drift regression and
            the instrument's first stage. The nuisances read standardised covariates. The drift
            regression and the first stage scale each slope's term by its column's variance,
            beside a free bias; the moment, which has no bias, scales each coefficient's by its
            column's mean square, on the log's raw actions. So the ridge reads the same in any
            units of the state, the actions, the drivers and the instrument. Added as a constant,
            it outweighed a column logged in units a millionth of its own: such actions read a
            channel near zero, and such a state a drift slope near zero.
        integrator: the one-step map the fitted field is made consistent with.

            ``"euler"`` (default, and what every release so far did) reads the rate off the log as
            ``(x_next - x)/dt``, which is exactly right if the caller treats the result as a
            *discrete-time* model and steps it the same way, as :func:`chc.decision.prescribe`
            does under ``integrator="euler"``. It is **not** right if the caller then integrates
            the field with RK4, as :func:`chc.plan.causal_plan` and :func:`chc.integrate.rollout`
            do with any field not marked as an Euler map: the two disagree by the RK4
            amplification factor ``1 + z + z^2/2 + z^3/6 + z^4/24`` at ``z = A dt``. On a linear
            plant at ``theta*dt = 0.7`` that is a decay fitted at ``-0.502`` against a true
            ``-0.700`` and, more to the point, a control channel fitted at ``0.574`` against a true
            ``0.800``.

            ``"rk4"`` makes the field consistent with RK4 instead, without inverting RK4 in closed
            form, which does not exist for a nonlinear field. Coefficients read a rate off the log:
            their own fitted rate plus what their RK4 step leaves of ``x_next``. The ``rk4`` fit is
            the one the estimator returns when handed its own reading, found by Newton's method
            from the Euler fit in three to six steps between ``|A|dt = 0.05`` and ``1.2``. Measured
            on the same plant, it recovers ``-0.699`` and ``0.800``. Its standard errors are the
            fixed point's own (:attr:`CausalDynamicsFit.channel_error`).

            Up to 0.6.0 the fixed point was iterated instead -- add the leftover to the target,
            refit -- and each state stopped once its defect's RMS stopped falling. That is not where
            the fixed point is: the RMS flattens under the noise first, and the iteration contracts
            the drift only by ``1 - (1 + z + z^2/2 + z^3/6)`` a pass, 0.51 at ``z = -0.7``. Even run
            until the RMS stopped falling at all, it stopped 0.9 standard errors of the channel and
            1.4 of the decay short at ``theta*dt = 0.7`` and ``N = 1000``, and 0.09 of the decay
            short at 1.25 on a log solved exactly.

            A log that falls by more in one step than RK4 can follow has no ``rk4`` reading at all.
            RK4's growth factor bottoms out at 0.2704, at ``z = -1.596``, and an exact linear mode
            falls below that past ``|A|dt = 1.308``. The fit raises there rather than return a field
            that is neither reading. A channel that reads the state can run out sooner: on the plant
            in :mod:`chc.research.mmm` with its fastest channel decaying at ``theta*dt = 1.2``, one
            seed in eight stalls in that channel's dependence on its own adstock, at a gap a damped
            Newton cannot close either, at both precisions. ``integrator_defect`` floors at the
            noise under both.

            The default stays ``"euler"`` so no shipped fit changes meaning.
            :func:`chc.decision.prescribe` defaults the other way, because it *knows* the field goes
            to the planner.
        drivers: exogenous columns that push the state and that nothing the plan does can move --
            weather, a demand forecast. ``data`` carries each at the start of the transition under
            its own name and at the end under ``f"{name}_next"``, and between the two it moves
            linearly, which is how :class:`chc.dynamics.DrivenDynamics` replays a forecast and what
            the ``rk4`` fit steps. Each enters the drift regression **jointly** with the
            state features, so the drift stops carrying the driver's correlation with the state
            (this module's first scope note): at its mean over the step under ``rk4``, the push a
            linear path puts into a step, and at the start under ``euler``, all the Euler map
            reads. Both ends join the channel's nuisance covariates. A logging policy that reads
            the forecast sets the action on where the driver is going, and with the start alone
            that is confounding: on a random-walk driver logged that way the ``euler`` channel came
            back ``0.656`` against the Euler map's own ``0.780``. Partialling out is safe for any
            driver, since a column nothing downstream of the action moves can be neither a mediator
            nor a collider of its effect. The fitted gain is :attr:`CausalDynamicsFit.driver_gain`;
            its standard error is inside ``drift_error``.
        weights: a weight on each transition's channel moment, a function of its state alone.
            Experimental: it may change or be withdrawn in any release. A callable is called with
            ``x (N, n)`` and nothing else, since a weight that reads the
            action or the next state biases the channel at first order, while any weight on the
            state keeps the moment orthogonal. The weights are scaled to mean 1. The channel's
            standard error is robust, as for every fit, and here it has to be: a weight that loads
            a few rows loads the nuisances' error at them too. ``effective_sample_size`` says how
            few.

            Under a channel class that contains the truth every weight estimates the same channel,
            and only the variance moves. The inverse of the rate's noise variance is the efficient
            weight in the limit, but it loads the states where the noise is small, which can be
            where the nuisances extrapolate: on a log whose rate noise ran ``exp(-x/2)`` against
            the action's ``exp(x/2)``, the channel scattered 0.87 as far as the unweighted fit's at
            ``N = 64000``, its limit, and 1.15 as far at ``N = 4000``.

            Under a class that misses the truth, each weight estimates a different channel: the
            projection of the true one under ``w s^2`` times the log's law of states, where
            ``s^2(x)`` is the variance the adjustment set leaves in the action at ``x``. The weight
            then chooses where the fit is right. For a one-shot decision on a scalar action, made
            at states drawn from ``Q`` with regret curvature ``kappa(x)`` in the channel,
            ``w = kappa (dQ/dP) / s^2`` makes the fit the best of its class for that decision. A
            weight that has to be estimated first -- ``s^2``, or the inverse noise -- carries its
            own error into the channel at first order there, since the estimand moves with the
            weight. No weight on the state alone serves a dynamic plan, which reads the channel's
            slope along its path as well as its level: there the costate weight left 36 to 178
            times the regret of the best fit in the class.
        influence: whether to keep each transition's influence on the fitted parameters,
            :attr:`CausalDynamicsFit.influence`, which
            :func:`chc.misspecification.misspecification_cost` reads to compare two fits of one
            log, and beside it the channel's representer, which :func:`omitted_confounder_bound`
            reads with the moment residual. Off by default: under ``euler`` it costs a
            reverse pass per parameter rather than per channel coefficient, and it is ``N n p``
            numbers the fit then carries, and ``N n q`` more.
        clusters: each transition's cluster, ``N`` labels of any one sortable kind, at least two
            distinct; one cluster's transitions are read across their periods by ``periods``
            instead. The channel's covariance then sums each cluster's scores over its transitions
            and states before squaring them, CR1 with the factor ``G / (G - 1) (N - 1) / (N - k)``
            over ``G`` clusters and ``k`` coefficients a state, so it holds whatever the dependence
            within a cluster; and a reader of ``influence`` sums its rows the same way, as
            ``drift_error`` and the error of the adjustment set's push at its mean on the log, which
            the tube reads, are read (ADR 0133). ``None`` takes every transition as independent of
            the rest, its states summed first since they may share its noise, which a log of units
            followed over time breaks wherever the model leaves out something that persists: the
            transitions of one unit then share it. The
            estimate does not move; only its error does. On panels of 5 to 80 units whose noise
            persists within each unit, a 5 % test of a zero channel rejects 12.25 % to 35 % of the
            time row by row and 4 % to 14 % by unit: CR1 still over-rejects with few clusters
            (ADR 0053). Two labels a transition, shape ``(N, 2)``, cluster it two ways, by its unit
            and by its period say. The error is then the largest of three (MacKinnon, Nielsen and
            Webb 2024): the two dimensions' sums less their intersection's (Cameron, Gelbach and
            Miller 2011), at CR1's factor for the smaller dimension's ``G`` and less any part that
            reads below nothing; and each dimension's sums alone, at its own. A shock every unit
            shares in a period, met by levers every unit moves together, makes one period's scores
            move together across units, which a sum within units leaves out (ADR 0061). A missing
            label, ``None``, nan or NaT, is refused.
        periods: each transition's period, a whole number of steps, on a log whose transitions
            fall in one group, one unit's or the units' of one declared cluster, which leaves no
            second cluster to sum within: one label a transition, two periods at least. The
            channel's covariance then sums the scores of each period's transitions over them and
            their states, and reads the sums' dependence across the periods by ``nu`` cosines of
            them, the equal-weighted cosine estimate with ``nu`` the most whole cosines at or
            below ``0.4 T^(2/3)`` for ``T`` distinct periods (Lazarus, Lewis, Stock and Watson
            2018), at the rows' factor ``N / (N - k)``, and a reader of ``influence`` sums its
            rows the same way, as ``drift_error`` and the error of the push's mean on the log are
            read (ADR 0133). Summed within each period first, it holds whatever the dependence
            between one period's transitions, as between units that share a shock (Driscoll and
            Kraay 1998). A test read off it is sized against ``t(nu)``
            (:attr:`CausalDynamicsFit.error_cosines`). ``None`` takes the transitions as
            independent, which a unit's persistent noise met by a lever that persists breaks: the
            scores, each the lever's residual times the noise, then persist too. On one unit of
            200 transitions whose noise is AR(0.9) and whose lever is AR(0.7), a 95 % interval
            covered the channel on 0.740 of 1000 logs row by row and on 0.938 by cosines, and with
            the polynomial nuisances alone on 0.724 and 0.941 of the same logs (0.703 and 0.925
            on ADR 0075's draws); where the noise is drawn afresh, the cosines' interval is 11 %
            wider. On three units of one
            cluster over 200 periods, which share a shock AR(0.9) and half their lever AR(0.7),
            it covered 0.692 row by row and 0.934 by the period sums (ADR 0075).
        times: each transition's start time, ``N`` finite numbers, where ``known`` reads the time:
            the known physics is called, and each step is taken, from there. ``None`` starts every
            transition at ``t = 0``, which is right for a ``known`` that does not read the time and
            nothing else: a seasonal known push read at ``t = 0`` throughout leaves the season in
            the rate, and a logging policy that follows the season then confounds the channel with
            it: on a log of 3000 Euler steps the channel came back 1.35 where it is 1.0, 37 of
            its errors off.

    Returns:
        A :class:`CausalDynamicsFit`. Read ``identified`` before ``residual``. A log that moves
        no direction of the channel apart from what the states and the covariates predict, a lever
        held constant or set by its own covariate, leaves ``identified=False`` and no error, as
        with no adjustment set; where it leaves any direction unmoved, the fit logs a WARNING
        naming the levers by their columns of ``u`` (:attr:`CausalDynamicsFit.unmoved`).

    Raises:
        ValueError: on an ``integrator`` that is neither ``"euler"`` nor ``"rk4"``, which up to
            0.18.0 was read as ``"euler"``; on a negative ``channel_degree``; on ``nuisance_knots``
            other than ``"auto"``
            or an integer of at least 3; on ``clusters`` that do not label every
            transition once or once in each of two dimensions, or name fewer than two clusters in
            a dimension; on ``periods`` that do not label every transition once with a whole
            number, or name one period alone, or that come with ``clusters``; on an ``rk4`` fixed
            point that does
            not converge; on a covariate named ``u`` or ``x_next``, an instrument or a driver named
            ``x``, ``u`` or ``x_next``, a driver named twice, or an instrument that is also a
            covariate or a driver, at the step's start or its end. Each would read a column the fit
            reads in another role: adjusted for the action, the channel came back as the
            confounded regression's. On ``unconfounded=True`` with an instrument. On ``times``
            that are not one finite number a transition. On ``clusters`` that hold a missing or a
            masked label.
    """
    if integrator not in ("euler", "rk4"):
        raise ValueError(f"integrator={integrator!r} is neither 'euler' nor 'rk4'")
    _refuse_reread("covariate", adjust_for, _ACTION_AND_OUTCOME)
    _refuse_reread("driver", drivers, _TRANSITION)
    twice = sorted({name for name in drivers if drivers.count(name) > 1})
    if twice:
        raise ValueError(
            f"drivers named more than once: {twice}; read twice, a driver's gain is split between "
            "its two copies"
        )
    if instrument is not None:
        if unconfounded:
            raise ValueError(
                "unconfounded=True with an instrument: an instrument is for actions a latent "
                "confounder moves, and unconfounded says nothing does"
            )
        _refuse_reread("instrument", (instrument,), _TRANSITION)
        if instrument in {*adjust_for, *drivers, *(f"{name}_next" for name in drivers)}:
            raise ValueError(
                f"the instrument {instrument!r} is also a covariate or a driver, at the step's "
                "start or its end, and partialled out it lends the action no move of its own"
            )
    channel_degree = degree if channel_degree is None else channel_degree
    if channel_degree < 0:
        raise ValueError(f"channel_degree must be a non-negative integer; got {channel_degree}")
    shifter = () if instrument is None else (instrument,)
    ends = tuple(f"{name}_next" for name in drivers)
    data = _real_entries(data, ("x", "u", "x_next", *adjust_for, *drivers, *ends, *shifter))
    x, u, x_next = data["x"], data["u"], data["x_next"]
    if clusters is not None and periods is not None:
        raise ValueError(
            "clusters and periods are two readings of the error's dependence: clusters read it "
            "within each cluster, periods across the periods of one group; give one"
        )
    codes = None if clusters is None else _cluster_codes(clusters, x.shape[0])
    steps = None if periods is None else _period_codes(periods, x.shape[0])
    starts = jnp.zeros(x.shape[0], dtype=x.dtype) if times is None else _start_times(times, x)
    known_rate = jax.vmap(known)(starts, x, u)

    identified = bool(adjust_for) or instrument is not None or unconfounded
    covariates, driver_start, driver_end, driver_read = _nuisance_inputs(
        data, adjust_for, drivers, integrator
    )
    # the Euler reading of the log's rate less the known physics, whichever the integrator: the
    # target the nuisances' sieve is chosen on, here and in the polynomial fit's check
    euler_target = (x_next - x) / dt - known_rate
    knot_count, powers = (
        (None, True)
        if nuisance_knots is None
        else _chosen_sieve(
            covariates,
            euler_target,
            u,
            degree=nuisance_degree,
            knots=_knot_count(nuisance_knots, x.shape[0]),
            folds=folds,
            ridge=ridge,
            seed=seed,
        )
    )
    sieve = (
        None
        if knot_count is None
        else _nuisance_sieve(covariates, nuisance_degree, knot_count, powers)
    )
    # The covariates that are not drivers join the drift in the nuisances' basis, held over the
    # step, about their mean on the log, and leave the returned field afterwards. A push the drift
    # regression leaves out moves the drift by its lean on the state and widens the drift's error
    # by its spread, under either integrator; under ``rk4`` the drift reaches the channel as well,
    # through RK4's growth factor.
    pushes = _held_pushes(
        jnp.concatenate(
            [x[:, :0], *[data[name] for name in adjust_for if name not in drivers]], axis=1
        ),
        nuisance_degree,
        knot_count,
        powers,
    )
    n_drivers = driver_start.shape[1]
    driver_start, driver_end, driver_read = (
        jnp.concatenate([column, pushes], axis=1)
        for column in (driver_start, driver_end, driver_read)
    )

    method = (
        "orthogonal" if adjust_for or unconfounded else ("iv" if instrument else "observational")
    )
    instrument_action: Array | None = None
    if instrument is not None:
        # 2SLS lifted to the matrix: the part of the action the exogenous shifter explains becomes
        # the moment's instrument -- not the regressor -- so the confounder's contribution drops out
        # of E[z (x) eps] even though z is never observed. u_res stays the regressor.
        w = data[instrument]
        design_w = (
            _polynomial_features(jnp.concatenate([x, w], axis=1), nuisance_degree)
            if knot_count is None
            else _first_stage(x, w, nuisance_degree, knot_count, powers)
        )
        projected = design_w @ _solve_ridge(design_w, u, ridge)
        instrument_action = _units.centred(projected).deviations
        method = "iv"

    # a_θ mops up the rest of y at the fitted channel; see this module's scope note on why that
    # makes the drift observational-conditional while the channel stays interventional. The drivers
    # sit in the same regression, read the way the integrator reads them.
    phi_x = jax.vmap(control_affine_features, in_axes=(0, None))(x, degree)
    phi_c = jax.vmap(control_affine_features, in_axes=(0, None))(x, channel_degree)
    design = jnp.concatenate([phi_x, driver_read], axis=1)

    row_weight = None if weights is None else _state_weights(weights, x)
    raw = _channel_design(u, x, channel_degree)
    # the parameters the returned field carries, in influence's order: the channel, then the drift
    # regression's rows but the pushes' gains, which leave the field with the pushes
    carried = design.shape[1] - pushes.shape[1]
    kept = x.shape[1] * (raw.shape[1] + carried)
    # each parameter's scale on the log, in influence's order: the channel's, state by state, then
    # the drift regression's, feature by feature
    state_size = _units.norms(x)
    parameter_size = jnp.concatenate(
        [
            (_units.norms(raw)[None, :] / state_size[:, None]).ravel(),
            (_units.norms(design[:, :carried])[:, None] / state_size[None, :]).ravel(),
        ]
    )
    # the moment's ridge, scaled to the raw actions' size, as the unmoved directions are
    penalty = jnp.diag(_units.mean_squares(raw, row_weight))
    directions = _unmoved_directions(u, x, covariates, nuisance_degree, channel_degree, sieve)
    unmoved = _unmoved_parameters(directions, phi_c, u, design, ridge, x.shape[1])[:kept]
    # The moment has no data along these directions, and its ridge would set the channel there by
    # the ratio of two roundings, which grows as the square of the actions' units: the fit then
    # missed the log's own rates by ten times their noise in units a thousand times larger.
    split = _split_unmoved(directions, raw, design) if directions.shape[1] else None
    relevance: Array | None = None
    rank: int | None = None
    if instrument is not None:
        relevance = _instrument_relevance(
            u,
            x,
            covariates,
            data[instrument],
            nuisance_degree,
            channel_degree,
            None if split is None else split.free,
            row_weight,
            sieve,
            _first_stage(x, data[instrument], nuisance_degree, knot_count, powers),
        )
        rank = int(jnp.sum(relevance > 0.0))
        if rank == 0 or rank < relevance.shape[0]:
            # The moment has no rank along a direction the instrument does not move: its ridge and
            # its noise set the channel there, and an instrument of zeros read an exact zero, error
            # zero. Where the log moves no direction, the instrument identifies none.
            identified, method, instrument_action = False, "observational", None
    if directions.shape[1] == directions.shape[0]:
        # The moment has no data along any direction: a lever held constant, or set by its own
        # covariate, read identified with the channel at zero and an error of exactly zero.
        identified = False

    def moment(y_res: Array, u_res: Array, unmoved: _Unmoved | None = split) -> Array:
        regressor = _channel_design(u_res, x, channel_degree)
        instrument = (
            regressor
            if instrument_action is None
            else _channel_design(instrument_action, x, channel_degree)
        )
        if unmoved is None:
            coeffs = _channel_coefficients(y_res, regressor, instrument, ridge, penalty, row_weight)
        else:
            # Only where the moment has data: an unmoved direction's regressor is what the
            # nuisance's ridge left of an action the covariates determine, and solved beside the
            # rest it moved them as far as the actions' units made it.
            coeffs = unmoved.free @ _channel_coefficients(
                y_res,
                regressor @ unmoved.free,
                instrument @ unmoved.free,
                ridge,
                unmoved.free.T @ penalty @ unmoved.free,
                row_weight,
            )
        return coeffs.T.reshape(y_res.shape[1], u_res.shape[1], -1)

    def solve(y: Array) -> tuple[Array, Array, tuple[Array, Array, Array, Array, Array]]:
        """The channel and the drift regression's coefficients a target ``y`` fits to, with the
        residualisation behind them. Linear in ``y``, which the ``rk4`` fixed point relies on."""
        y_res, u_res, y_hat, u_hat = _cross_fit_residuals(
            y,
            u,
            covariates,
            degree=nuisance_degree,
            folds=folds,
            ridge=ridge,
            seed=seed,
            features=sieve,
        )
        channel = moment(y_res, u_res)
        if split is not None:
            channel = split.hold(channel, y)
        fitted = jax.vmap(lambda c, ui: (channel @ c) @ ui)(phi_c, u)
        remainder = _solve_ridge(design, y - fitted, ridge)  # (features + drivers, n)
        return channel, remainder, (y_res, u_res, y_hat, u_hat, fitted)

    def _parameters(solved: tuple[Array, Array, object]) -> Array:
        """The channel, then the drift regression's rows: the order the ``rk4`` fixed point uses."""
        return jnp.concatenate([solved[0].ravel(), solved[1].ravel()])

    def drift_direct(channel_size: int) -> Array:
        """``(p, N, n)``: how a row's rate moves the drift regression's rows other than through the
        channel -- the regression's own weight on the row, in the row's state's column."""
        states = x.shape[1]
        weight = _ridge_weight(design, ridge)  # (features + drivers, N)
        rows = jnp.einsum("fi,ts->ftis", weight, jnp.eye(states))
        rows = rows.reshape(weight.shape[0] * states, x.shape[0], states)
        return jnp.concatenate([jnp.zeros((channel_size, x.shape[0], states)), rows])

    def held(u_res: Array) -> Array:
        """``(p, N, n)``: how a row's rate moves the parameters through the channel's moment alone,
        with the cross-fitted nuisances held, less the drift regression's own weight on the row.
        Letting the nuisances move with the row as well, as ``influence`` does, adds their own
        error to the representer, and so an upward bias of the order of the nuisance features over
        ``N`` to its second moment."""

        def respond(y_res: Array) -> Array:
            channel = moment(y_res, u_res)
            if split is not None:  # the row's rate moves y_res and the rate alike
                channel = split.hold(channel, y_res)
            fitted = jax.vmap(lambda c, ui: (channel @ c) @ ui)(phi_c, u)
            return jnp.concatenate([channel.ravel(), -_solve_ridge(design, fitted, ridge).ravel()])

        return jax.jacrev(respond)(jnp.zeros_like(x_next))

    def fit_to(rate: Array) -> CausalDynamicsFit:
        """Everything downstream of the target rate, so the ``rk4`` fixed point can re-run it."""
        y = rate - known_rate
        channel, remainder, (y_res, u_res, y_hat, u_hat, fitted) = solve(y)
        regressor = _channel_design(u_res, x, channel_degree)
        moment = (
            regressor
            if instrument_action is None
            else _channel_design(instrument_action, x, channel_degree)
        )
        coeffs = channel.reshape(x.shape[1], -1).T
        drift = remainder[: phi_x.shape[1]].T
        gain = remainder[phi_x.shape[1] : phi_x.shape[1] + n_drivers].T

        score = y_res - regressor @ coeffs
        weighted_moment = moment if row_weight is None else moment * row_weight[:, None]
        drift_blocks = _ols_blocks(y - fitted, design, remainder, ridge) if identified else None
        return CausalDynamicsFit(
            residual=ControlAffineResidual(
                drift=drift, channel=channel, degree=degree, channel_degree=channel_degree
            ),
            identified=identified,
            method=method,
            folds=folds,
            # the channel's comes from the fit's own sensitivity, which only the caller knows how
            # to take: through the cross-fit under Euler, through the fixed point under ``rk4``
            channel_error=None,
            drift_error=_error_at_rows(drift_blocks[:, :carried, :carried], design[:, :carried])
            if drift_blocks is not None
            else None,
            _drift_blocks=None
            if drift_blocks is None
            else np.asarray(drift_blocks[:, :carried, :carried], np.float64),
            action_residual_variance=float(jnp.mean(u_res**2)),
            nuisance_r2_state=_r_squared(y, y_hat),
            nuisance_r2_action=_r_squared(u, u_hat),
            moment_norm=float(jnp.linalg.norm(weighted_moment.T @ score / moment.shape[0])),
            integrator=integrator,
            drivers=tuple(drivers),
            driver_gain=gain if drivers else None,
            weighted=row_weight is not None,
            effective_sample_size=float(x.shape[0])
            if row_weight is None
            else float(jnp.sum(row_weight) ** 2 / jnp.sum(row_weight**2)),
            unmoved=unmoved,
            _parameter_size=parameter_size,
        )

    def predicted(residual: ControlAffineResidual, gain: Array) -> Array:
        """``step(F, x, u, dt)`` on every transition, under the integrator the fit is made for."""
        model = HybridDynamics(known=known, residual=residual)

        def step(ti: Array, xi: Array, ui: Array, start: Array, end: Array) -> Array:
            # The transition's own stretch of the drivers' path, held the way the planner holds a
            # forecast: one implementation of the hold, so the fit and the plan cannot drift apart.
            field = DrivenDynamics(model, gain, jnp.stack([start, end]), dt, start=ti)
            if integrator == "euler":
                return xi + dt * field(ti, xi, ui)
            return rk4_step(field, ti, xi, ui, dt)

        return jax.vmap(step)(starts, x, u, driver_start, driver_end)

    def defect(fit: CausalDynamicsFit) -> Array:
        """``(x_next - step(F, x, u, dt)) / dt`` -- what the fitted field leaves of the log, the
        covariates' held push with it, which the field does not carry."""
        gain = jnp.zeros((x.shape[1], 0)) if fit.driver_gain is None else fit.driver_gain
        gain = jnp.concatenate([gain, jnp.zeros((x.shape[1], pushes.shape[1]))], axis=1)
        return (x_next - predicted(fit.residual, gain)) / dt

    def rk4_fixed_point(
        start: Array,
    ) -> tuple[CausalDynamicsFit, Array, Callable[[Array], Array] | None, Array]:
        """The fit that is its own ``rk4`` reading of the log, by Newton's method from the fit to
        the target ``start``, the target it reads, and, where it is identified, how a move of its
        parameters ``(p, ...)`` at that target carries to the fixed point.

        Coefficients ``theta`` read a rate off the log: their own fitted rate ``Phi theta`` plus
        what their RK4 step leaves over, ``y(theta) = Phi theta + (x_next - RK4_theta) / dt``.
        Under Euler that is ``(x_next - x) / dt`` whatever ``theta`` is. The fixed point is
        ``theta = G y(theta)``, with ``G`` the fit, which is linear in its target.

        A noise ``e`` in ``x_next`` moves it by ``K^-1 G e / dt``, where ``K = I - G dy/dtheta`` is
        the matrix Newton steps with, the identity under Euler. The drift's error stays conditional
        on the channel, as it is under Euler: its own regression, through the drift's block of K.
        """
        shape = (x.shape[1], u.shape[1], phi_c.shape[1])
        size = math.prod(shape)

        def unpack(theta: Array) -> tuple[ControlAffineResidual, Array, Array]:
            rows = theta[size:].reshape(-1, x.shape[1])  # the drift regression's: drift, then gain
            channel = theta[:size].reshape(shape)
            residual = ControlAffineResidual(
                drift=rows[: phi_x.shape[1]].T,
                channel=channel,
                degree=degree,
                channel_degree=channel_degree,
            )
            return residual, rows[phi_x.shape[1] :].T, rows

        def reading(theta: Array) -> Array:
            residual, gain, rows = unpack(theta)
            own = jax.vmap(lambda c, ui: (residual.channel @ c) @ ui)(phi_c, u) + design @ rows
            return own + (x_next - predicted(residual, gain)) / dt

        def coefficients(y: Array) -> Array:
            channel, remainder, _ = solve(y)
            return jnp.concatenate([channel.ravel(), remainder.ravel()])

        def newton_matrix(theta: Array) -> tuple[Array, Array]:
            sensitivity = jax.jacfwd(reading)(theta)  # (N, n, p)
            gained = jnp.einsum("pis,isq->pq", fit_map, sensitivity)
            return jnp.eye(theta.size) - gained, sensitivity

        fit_map = jax.jacrev(coefficients)(jnp.zeros_like(x_next))  # (p, N, n), exact: linear
        theta = coefficients(start)  # the Euler reading's, the pushes' gains with the rest
        # Each parameter in its own unit: the power of two nearest the size of the fit's weight on
        # the log's rates times theirs, which is in the parameter's units whichever state's rates
        # it reads. K's entries are ratios of the parameters' units, so in the caller's units LU
        # compared entries that are not comparable: with the state in millionths and the actions
        # in millions, in float32, the rounding in a drift row outweighed the channel's own entry,
        # LU pivoted on it, and the channel's error read 30 times its own.
        rates = _units.root_mean_square(reading(theta))
        weight = (fit_map * rates).reshape(theta.size, -1).T
        unit = 1.0 / _units.power_of_two(_units.root_mean_square(weight))

        def unitless(matrix: Array, units: Array) -> Array:
            return matrix * units[None, :] / units[:, None]

        tolerance = float(jnp.sqrt(jnp.finfo(theta.dtype).eps))
        for _ in range(_MAX_NEWTON_STEPS):
            gap = jnp.einsum("pis,is->p", fit_map, reading(theta)) - theta
            step = unit * jnp.linalg.solve(unitless(newton_matrix(theta)[0], unit), gap / unit)
            theta = theta + step
            # relative to the parameters, or to their units, the size of one row's share of each,
            # where the parameters are smaller: a known physics that leaves the residual nothing
            # sets every parameter to 0, which no step is small beside
            if float(jnp.max(jnp.abs(step / unit))) <= tolerance * max(
                1.0, float(jnp.max(jnp.abs(theta / unit)))
            ):
                break
        else:
            raise ValueError(
                f"the rk4 fixed point did not converge in {_MAX_NEWTON_STEPS} Newton steps (last "
                f"step {float(jnp.max(jnp.abs(step))):.3g}): no field stepped by RK4 at dt={dt} "
                "reproduces this log. On a linear mode that happens where the log keeps less "
                "than 0.27 of it from one step to the next, the least any RK4 step keeps, as a "
                "plant solved exactly does once |A|dt passes 1.31. Log at a finer dt; or, where "
                "the panel is discrete-time and its one-step map is the model, fit with "
                "integrator='euler' and treat the result as that map: prescribe(integrator="
                "'euler') plans, prices and certifies by it."
            )
        target = reading(theta)
        fit = fit_to(known_rate + target)
        if not identified:
            return fit, target, None, fit_map

        # Only the noise is random given the log, and the channel reads it where its own sandwich
        # does, after the adjustment set has taken out what it explains. The drift regression holds
        # the adjustment set's push, a season's say, so its own residual is that noise too where
        # the push is linear; left in the drift's residual, K carried it into the channel, which
        # scattered 13 times as far as this error said.
        channel, _, (y_res, u_res, *_) = solve(target)
        regressor = _channel_design(u_res, x, channel_degree)
        score = y_res - regressor @ channel.reshape(x.shape[1], -1).T
        newton, sensitivity = newton_matrix(theta)
        newton = unitless(newton, unit)
        covariances = [
            unit[:, None]
            * jnp.linalg.solve(newton, jnp.linalg.solve(newton, spread / unit[:, None] / unit).T)
            * unit
            for spread in _robust_spreads(fit_map, score, regressor.shape[1], codes, steps)
        ]

        # A row reaches the fixed point through K^-1 G, as the covariance says the noise does.
        def carry(rows: Array) -> Array:
            own = rows.reshape(theta.size, -1) / unit[:, None]
            return (unit[:, None] * jnp.linalg.solve(newton, own)).reshape(rows.shape)

        # The drift keeps the defect, which is its own regression's residual, as under Euler.
        own_defect = (x_next - predicted(*unpack(theta)[:2])) / dt
        noise = jnp.sum(own_defect**2, axis=0) / max(x.shape[0] - design.shape[1], 1)
        gram = _ridge_inverse(design, ridge)
        drift_size = theta.size - size
        drift_gained = jnp.einsum(
            "fi,isq->fsq", _ridge_weight(design, ridge), sensitivity[:, :, size:]
        )
        drift_unit = unit[size:]
        drift_newton = unitless(
            jnp.eye(drift_size) - drift_gained.reshape(drift_size, drift_size), drift_unit
        )
        drift_spread = jnp.kron(gram, jnp.diag(noise)) / drift_unit[:, None] / drift_unit
        drift_covariance = (
            drift_unit[:, None]
            * jnp.linalg.solve(drift_newton, jnp.linalg.solve(drift_newton, drift_spread).T)
            * drift_unit
        )
        # the drift's rows are its features, then its drivers, then the covariates' held pushes,
        # each state's coefficient beside the others': state s's coefficients are every n-th, from
        # s. The field carries the first two.
        rows, states = design.shape[1], x.shape[1]
        every_state = jnp.arange(states)
        drift_blocks = drift_covariance.reshape(rows, states, rows, states)[
            :, every_state, :, every_state
        ][:, :carried, :carried]
        channel_covariances = [covariance[:size, :size] for covariance in covariances]
        channel_error, lever_errors, channel_blocks = _channel_errors(
            channel_covariances, channel.shape, phi_c
        )
        # The field's error beyond the drift's own, through the fixed point. A channel error d_c
        # moves the drift's rows by -K_rr^-1 (Gamma - W Y_c) d_c: the regression reads its target
        # at the fitted channel, and the RK4 map moves that target with the channel too. And the
        # map moves the channel with the drift's own error, which the two then share: the drift's
        # rows depend on the channel through the step, so the cross term Frisch--Waugh sets to
        # nothing under Euler is not nothing here.
        weight = _ridge_weight(design, ridge)
        direct = drift_direct(size)[size:]
        conditional = (
            drift_unit[:, None]
            * jnp.linalg.solve(drift_newton, direct.reshape(drift_size, -1) / drift_unit[:, None])
        ).reshape(direct.shape)
        lean = jnp.einsum("fa,st->fsta", weight @ raw, jnp.eye(states)).reshape(drift_size, size)
        moved = jnp.einsum("fi,isq->fsq", weight, sensitivity[:, :, :size])
        projection = drift_unit[:, None] * jnp.linalg.solve(
            drift_newton, (lean - moved.reshape(drift_size, size)) / drift_unit[:, None]
        )
        terms = _field_terms(
            channel_covariances,
            carry(fit_map)[:size],
            conditional,
            projection,
            drift_blocks,
            drift_covariance.reshape(rows, states, rows, states)[:carried, :, :carried, :]
            .transpose(1, 0, 3, 2)
            .reshape(states * carried, states * carried),
            noise,
            score,
            own_defect,
            pushes @ theta[size:].reshape(rows, states)[carried:],
            u,
            x,
            design[:, phi_x.shape[1] : carried],
            carried,
            codes,
            steps,
        )
        drift_error, drift_read = (
            (_error_at_rows(drift_blocks, design[:, :carried]), drift_blocks)
            if codes is None and steps is None
            else _drift_reads(terms, design[:, :carried])
        )
        return (
            dataclasses.replace(
                fit,
                channel_error=channel_error,
                _lever_errors=lever_errors,
                _channel_blocks=channel_blocks,
                drift_error=drift_error,
                _drift_blocks=np.asarray(drift_read, dtype=np.float64),
                _field_terms=terms,
                influence=_influence(
                    carry(fit_map),
                    score,
                    carry(drift_direct(size)),
                    own_defect,
                    regressor.shape[1],
                    codes,
                )[:, :, :kept]
                if influence
                else None,
                representer=_representer(carry(held(u_res))[:size]) if influence else None,
                moment_residual=score,
            ),
            target,
            carry,
            fit_map,
        )

    # the fit's weight on each row's rate: the whole fixed point's under rk4, the channel's under
    # Euler, set below where the fit is identified
    own_map: Array | None = None
    if integrator == "rk4":
        fit, target, to_fixed_point, own_map = rk4_fixed_point((x_next - x) / dt - known_rate)
    else:
        target, to_fixed_point = (x_next - x) / dt - known_rate, None
        fit = fit_to((x_next - x) / dt)
    if integrator == "euler" and identified:
        y = (x_next - x) / dt - known_rate
        channel, remainder, (y_res, u_res, _, _, fitted) = solve(y)
        regressor = _channel_design(u_res, x, channel_degree)
        score = y_res - regressor @ channel.reshape(x.shape[1], -1).T
        if influence:
            sensitivity = jax.jacrev(lambda target: _parameters(solve(target)))(jnp.zeros_like(y))
        else:
            sensitivity = jax.jacrev(lambda target: solve(target)[0].ravel())(jnp.zeros_like(y))
        spreads = _robust_spreads(
            sensitivity[: channel.size], score, regressor.shape[1], codes, steps
        )
        channel_error, lever_errors, channel_blocks = _channel_errors(spreads, channel.shape, phi_c)
        # The regression reads its target at the fitted channel, so a channel error d_c moves each
        # state's drift by -Gamma d_c, Gamma = W G its weight on the channel's regressors
        # (ADR 0131).
        states = x.shape[1]
        weight = _ridge_weight(design, ridge)
        residual = y - fitted - design @ remainder
        assert fit._drift_blocks is not None  # an identified fit keeps them
        terms = _field_terms(
            spreads,
            sensitivity[: channel.size],
            drift_direct(channel.size)[channel.size :],
            jnp.einsum("fa,st->fsta", weight @ raw, jnp.eye(states)).reshape(-1, channel.size),
            fit._drift_blocks,
            # the states' blocks are one, and independent of each other's: one pooled noise
            np.kron(np.eye(states), fit._drift_blocks[0]),
            # the noise _ols_blocks pools over the rows and the states
            jnp.full(
                (states,),
                jnp.sum(residual**2) / (max(x.shape[0] - design.shape[1], 1) * states),
            ),
            score,
            residual,
            pushes @ remainder[carried:],
            u,
            x,
            design[:, phi_x.shape[1] : carried],
            carried,
            codes,
            steps,
        )
        if codes is not None or steps is not None:
            drift_error, drift_read = _drift_reads(terms, design[:, :carried])
            fit = dataclasses.replace(fit, drift_error=drift_error, _drift_blocks=drift_read)
        own_map = sensitivity[: channel.size]
        least_squares = instrument_action is None and row_weight is None and split is None
        fit = dataclasses.replace(
            fit,
            channel_error=channel_error,
            _lever_errors=lever_errors,
            _channel_blocks=channel_blocks,
            _field_terms=terms,
            _channel_gram=_ChannelGram(
                matrix=np.asarray(regressor.T @ regressor + ridge * penalty, dtype=np.float64),
                ridge=np.asarray(ridge * jnp.diag(penalty), dtype=np.float64),
                design=np.asarray(regressor, dtype=np.float64),
            )
            if least_squares
            else None,
            influence=_influence(
                sensitivity,
                score,
                drift_direct(channel.size),
                y - fitted - design @ remainder,
                regressor.shape[1],
                codes,
            )[:, :, :kept]
            if influence
            else None,
            representer=_representer(held(u_res)[: channel.size]) if influence else None,
            moment_residual=score,
        )
    left = defect(fit)
    fit = dataclasses.replace(
        fit,
        integrator_defect=float(jnp.sqrt(jnp.mean(left**2))),
        defect=left,
        clusters=codes,
        instrument_relevance=relevance,
        instrument_rank=rank,
        periods=steps,
    )
    if directions.shape[1]:
        _warn_unmoved(fit, directions * _units.norms(raw)[:, None], identified)
    error = fit.channel_error
    # under a weight the spline's own second-order error reaches past the channel's on a share of
    # well-specified logs, so the shift no longer separates a miss from noise
    if (
        nuisance_knots is None
        and method == "orthogonal"
        and weights is None
        and error is not None
        and error > 0.0
        and own_map is not None
    ):
        moved, spread = _spline_shift(
            fit,
            own_map,
            target,
            x,
            u,
            covariates,
            phi_c,
            raw,
            design,
            moment,
            to_fixed_point,
            euler_target,
            nuisance_degree=nuisance_degree,
            channel_degree=channel_degree,
            folds=folds,
            ridge=ridge,
            seed=seed,
            clusters=codes,
            periods=steps,
            channel_scale=float(
                _units.root_mean_square(((x_next - x) / dt).ravel())
                / _units.root_mean_square(u.ravel())
            ),
        )
        shift = moved / spread
        # the move's error is read off as many independent terms as the channel's: past a few,
        # three of them is a t's tail, not the normal's, and a t(1)'s passes three on a fifth
        freedom = _degrees_of_freedom(x.shape[0], codes, steps)
        limit = float(student_t.ppf(student_t.cdf(_NUISANCE_SHIFT_WARN, math.inf), freedom))
        if shift > limit:
            warnings.warn(
                f"spline nuisances (nuisance_knots='auto', the default) move the channel by "
                f"{shift:.3g} errors of the move, so the polynomial nuisances "
                "(nuisance_knots=None) miss this log's and its channel_error does not cover the "
                "bias; leave nuisance_knots at its default to read the channel on the spline",
                UserWarning,
                stacklevel=2,
            )
    return fit


def _spline_shift(
    fit: CausalDynamicsFit,
    own_map: Array,
    target: Array,
    x: Array,
    u: Array,
    covariates: Array,
    phi_c: Array,
    raw: Array,
    design: Array,
    moment: Callable[[Array, Array, _Unmoved | None], Array],
    to_fixed_point: Callable[[Array], Array] | None,
    euler_target: Array,
    *,
    nuisance_degree: int,
    channel_degree: int,
    folds: int,
    ridge: float,
    seed: int,
    clusters: np.ndarray | None,
    periods: np.ndarray | None,
    channel_scale: float,
) -> tuple[float, float]:
    """How far spline nuisances with ``"auto"`` knots move an orthogonal fit's channel, and the
    move's own standard error, read off the two fits' scores row by row: each the root mean square
    over the log's rows and the channel's entries.

    The refit's sieve is chosen as a refit's is, on the log's Euler reading ``euler_target``
    (:func:`_chosen_sieve`); where that is the polynomial the move is nothing. The refit's moment
    is solved at the fit's own
    target: under Euler that is the refit's channel;
    under ``rk4`` its first Newton step, through the fit's own Newton matrix. Both fits are linear
    in the target, ``G y``, so the move is ``(G_s - G) y`` and a row's share of it
    ``G_s,i e_s,i - G_i e_i``, each fit's weight on the row times the noise it reads there; their
    squares, summed as the channel's error sums its own (:func:`_score_spreads`), are the move's
    variance where both fits are right, whichever is the more efficient
    (``validation/paired_score_shift.mac``). Read against the fit's own error instead, a refit
    that scatters more than the fit moved by more than three of its errors on 12.5 % of linear
    logs with ``t(3)`` covariates; the difference of the two variances (Hausman 1978) needs the
    fit to be the efficient one, which on heavy tails it was not, and can read below nothing.

    A move is not read below the square root of the working precision of ``channel_scale``, a
    channel that moves the rate by its own size, which is where the ``rk4`` fixed point's Newton
    steps stop: on a log the known physics solves exactly both errors are rounding, and their ratio
    read 13."""
    floor = math.sqrt(float(jnp.finfo(x.dtype).eps)) * channel_scale
    knots, powers = _chosen_sieve(
        covariates,
        euler_target,
        u,
        degree=nuisance_degree,
        knots=_knot_count("auto", x.shape[0]),
        folds=folds,
        ridge=ridge,
        seed=seed,
    )
    if knots is None:
        return 0.0, floor
    spline = _nuisance_sieve(covariates, nuisance_degree, knots, powers)
    found = _unmoved_directions(u, x, covariates, nuisance_degree, channel_degree, spline)
    split = _split_unmoved(found, raw, design) if found.shape[1] else None
    channel = fit.residual.channel

    def solved(y: Array) -> tuple[Array, Array]:
        y_res, u_res, _, _ = _cross_fit_residuals(
            y,
            u,
            covariates,
            degree=nuisance_degree,
            folds=folds,
            ridge=ridge,
            seed=seed,
            features=spline,
        )
        moved = moment(y_res, u_res, split)
        if split is not None:
            moved = split.hold(moved, y)
        regressor = _channel_design(u_res, x, channel_degree)
        return moved, y_res - regressor @ moved.reshape(x.shape[1], -1).T

    def coefficients(y: Array) -> Array:
        moved = solved(y)[0]
        if to_fixed_point is None:
            return moved.ravel()
        fitted = jax.vmap(lambda c, ui: (moved @ c) @ ui)(phi_c, u)
        return jnp.concatenate([moved.ravel(), _solve_ridge(design, y - fitted, ridge).ravel()])

    spline_map = jax.jacrev(coefficients)(jnp.zeros_like(target))
    move = jnp.einsum("pis,is->p", spline_map - own_map, target)
    rows = jnp.einsum("pis,is->ip", spline_map, solved(target)[1]) - jnp.einsum(
        "pis,is->ip", own_map, fit.moment_residual
    )
    if to_fixed_point is not None:
        move = to_fixed_point(move[:, None])[:, 0]
        rows = to_fixed_point(rows.T).T
    move, rows = move[: channel.size], rows[:, : channel.size]
    spreads = _score_spreads(rows, phi_c.shape[1] * u.shape[1], clusters, periods)
    error = _channel_errors(spreads, channel.shape, phi_c)[0]
    at_rows = jnp.einsum("jkl,il->ijk", move.reshape(channel.shape), phi_c)
    return float(jnp.sqrt(jnp.mean(at_rows**2))), max(error, floor)


def _start_times(times: ArrayLike, x: Array) -> Array:
    """Each transition's start time, one finite number a transition, in the states' precision."""
    starts = jnp.asarray(_real_numbers(times, "times"), dtype=x.dtype)
    if starts.shape != (x.shape[0],):
        raise ValueError(
            f"times gives each of the {x.shape[0]} transitions its start, shape ({x.shape[0]},); "
            f"got an array of shape {starts.shape}"
        )
    if not bool(jnp.all(jnp.isfinite(starts))):
        raise ValueError("times holds a value that is not a finite number")
    return starts


def _period_codes(periods: ArrayLike, rows: int) -> np.ndarray:
    """Each transition's period as steps from the log's first: whole numbers, one a transition,
    two periods at least."""
    steps = np.asarray(periods)
    if steps.shape != (rows,):
        raise ValueError(
            f"periods label each of the {rows} transitions once, shape ({rows},); got an array of "
            f"shape {steps.shape}"
        )
    if not np.issubdtype(steps.dtype, np.integer):
        raise ValueError(f"periods are whole numbers of steps; got dtype {steps.dtype}")
    if np.unique(steps).size < 2:
        raise ValueError(
            "periods name one period: the cosines read the scores' dependence across the "
            "periods, and the sum of one period's leaves no spread across them to read"
        )
    return (steps - steps.min()).astype(np.int64)


def _missing_labels(labels: np.ndarray) -> np.ndarray:
    """Whether each label, flattened, is missing: ``None``, a nan, or a NaT."""
    if labels.dtype.kind in "mM":
        return np.isnat(labels).ravel()
    if labels.dtype.kind in "fc":
        return np.isnan(labels).ravel()
    if labels.dtype.kind != "O":
        return np.zeros(labels.size, dtype=bool)
    return np.array(
        [
            label is None
            or (isinstance(label, float | np.floating) and math.isnan(label))
            or (isinstance(label, np.datetime64 | np.timedelta64) and bool(np.isnat(label)))
            for label in labels.ravel()
        ],
        dtype=bool,
    )


def _cluster_codes(clusters: np.ndarray | Array, rows: int) -> np.ndarray:
    """Each transition's cluster as a code from 0, in the labels' sorted order, or two-way a code
    a dimension, ``(N, 2)``."""
    labels = np.asarray(_unmasked("clusters", clusters))
    if labels.shape not in ((rows,), (rows, 2)):
        raise ValueError(
            f"clusters label each of the {rows} transitions once, shape ({rows},), or once in each "
            f"of two dimensions, shape ({rows}, 2); got an array of shape {labels.shape}"
        )
    # np.unique pools every nan into one label, so missing labels read as one cluster of their own
    missing = np.flatnonzero(_missing_labels(labels))
    if missing.size:
        raise ValueError(
            f"clusters hold a missing label at {_at(labels.shape, int(missing[0]))} "
            f"({missing.size} of {labels.size}): a transition of no known cluster cannot be summed "
            "with its own; label it, or drop it"
        )
    codes = []
    for column in labels.reshape(rows, -1).T:
        distinct, code = np.unique(column, return_inverse=True)
        if distinct.size < 2:
            raise ValueError(
                "clusters name one cluster"
                + ("" if labels.ndim == 1 else " in one of their two dimensions")
                + "; a covariance summed within clusters needs two at least, since one cluster's "
                "sum leaves no spread between clusters to read. Give the transitions' periods as "
                "periods instead, to sum the one cluster's scores within each period and read "
                "the sums across the periods"
            )
        codes.append(code.astype(np.int64))
    return codes[0] if labels.ndim == 1 else np.column_stack(codes)


# --- MM7: how strong a confounder the adjustment set left out would have to be ---


@dataclass(frozen=True)
class OmittedConfounderBound:
    """How far a confounder the adjustment set left out could move a linear functional of the
    channel, and how strong it would have to be to move it to ``null``. *Experimental.*

    ``bias_scale`` is identified; ``strength`` is the analyst's, as ``Gamma`` is
    (docs/concepts/gamma.md), in partial R^2s rather than odds.
    """

    estimate: float
    # sum over states of sigma nu: the bias at strength 1, DoubleML's ``max_bias``
    bias_scale: float
    strength: float  # |rho| sqrt(cf_y cf_d / (1 - cf_d))
    lower: float  # estimate - strength * bias_scale
    upper: float
    # with the sampling error of the estimate and of bias_scale, each one-sided at ``level``
    ci_lower: float
    ci_upper: float
    # cf_y = cf_d at which the bound reaches ``null``: the design sensitivity, which the confidence
    # bound's own value, below, reaches as the log grows. 1 when no share moves the bound.
    robustness_value: float
    robustness_value_ci: float


def omitted_confounder_bound(
    fit: CausalDynamicsFit,
    functional: Array,
    *,
    cf_y: float,
    cf_d: float,
    rho: float = 1.0,
    level: float = 0.95,
    null: float = 0.0,
) -> OmittedConfounderBound:
    """Bound a linear functional of the channel against a confounder ``adjust_for`` left out.
    *Experimental.*

    Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2022) bound the bias of a linear
    functional of a partially linear fit by ``|rho| sqrt(cf_y cf_d / (1 - cf_d)) sigma nu``:
    ``sigma^2`` the rate's residual variance after the fit, ``nu^2`` the second moment of the
    functional's Riesz representer, both identified, and two shares that are not. ``cf_y`` is the
    share of that residual variance the latent explains; ``cf_d`` the share of the long
    representer's second moment the fit's short one misses, which for one lever is the latent's
    partial R^2 with it. ``rho`` is the correlation between the two gaps, 1 at the worst. On a
    linear Gaussian plant with one latent the bound is attained, for any functional of any number
    of levers, and with two latents it is strict unless they move the lever and the rate in
    proportion (``validation/omitted_confounder_bound.mac``). Over several states the bound sums
    ``sigma_s nu_s``, taking the shares as common to them; under Euler a functional of one
    state's channel reads that state alone.

    ``functional`` weighs the channel's coefficients, in its shape ``(n, m, n_features)``. One
    lever's effect on one state is a single 1. Moving spend from lever ``a`` to lever ``b`` is
    ``+1`` at ``b`` and ``-1`` at ``a``, and since its value is linear in the channel, its
    ``robustness_value`` is the confounding at which the move stops paying.

    The sampling error follows DoubleML's: each confidence bound is one-sided at ``level``, and
    carries the estimate's influence and ``bias_scale``'s, whose ``nu^2`` part is ``nu^2 -
    alpha^2`` for a representer ``alpha``. Its sign convention for the estimate differs from this
    one in the cross term of the two influences, which vanishes in expectation. A fit given
    ``clusters`` has both influences summed within each cluster, as its channel's error is, and
    one given ``periods`` reads them across its periods by cosines, each period's transitions
    summed first. The bounds take a ``t``'s quantile with ``G - 1`` degrees of freedom, ``G`` the
    clusters, two-way the smaller dimension's count, or the rows where there are none; or ``nu``,
    the cosines', under ``periods`` (ADR 0075). At 5 units, a 5 % test of the channel read off its
    error rejected 9.75 % and 14.75 % of panels against the normal's quantile, and 3.25 % and
    5.75 % against ``t(4)``'s (ADR 0062). DoubleML takes the normal's.

    Args:
        fit: an identified fit of :func:`fit_causal_residual`, by adjustment, unweighted, made
            with ``influence=True``.
        functional: the weight on each channel coefficient.
        cf_y: the share of the residual variance the latent explains, in ``[0, 1)``.
        cf_d: the share of the long representer's second moment the short one misses, in
            ``[0, 1)``.
        rho: the correlation of the two gaps, in ``[-1, 1]``.
        level: each confidence bound's one-sided level, in ``[0.5, 1)``.
        null: the value the robustness values measure the distance to.

    Raises:
        ValueError: on a fit whose channel is not identified by adjustment -- an instrument's
            representer is not the moment's -- or that is weighted, or kept no representer; on a
            functional of another shape; on shares, ``rho`` or ``level`` outside their ranges.
    """
    if fit.method != "orthogonal":
        raise ValueError(
            "the bound is derived for a channel identified by adjustment; this fit's method is "
            f"{fit.method!r}"
        )
    if fit.weighted:
        raise ValueError("the bound is derived for the unweighted moment; this fit is weighted")
    if fit.representer is None or fit.moment_residual is None or fit.influence is None:
        raise ValueError("the fit kept no representer; fit it with influence=True")
    for name, share in (("cf_y", cf_y), ("cf_d", cf_d)):
        if not 0.0 <= share < 1.0:
            raise ValueError(f"{name} is a share of variance and must lie in [0, 1); got {share}")
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho is a correlation and must lie in [-1, 1]; got {rho}")
    if not 0.5 <= level < 1.0:
        raise ValueError(
            f"level is a one-sided confidence level and must lie in [0.5, 1); got {level}"
        )
    channel = np.asarray(fit.residual.channel, dtype=np.float64)
    weights = np.asarray(functional, dtype=np.float64)
    if weights.shape != channel.shape:
        raise ValueError(
            f"the functional weighs a channel of shape {weights.shape}; the fit's is "
            f"{channel.shape}"
        )
    w = weights.ravel()
    estimate = float(w @ channel.ravel())
    alpha = np.asarray(fit.representer, dtype=np.float64) @ w  # (N, n)
    residual = np.asarray(fit.moment_residual, dtype=np.float64)
    psi = np.asarray(fit.influence, dtype=np.float64)[:, :, : w.size] @ w  # the estimate's
    rows = residual.shape[0]
    clusters = fit.clusters
    sigma2, nu2 = np.mean(residual**2, axis=0), np.mean(alpha**2, axis=0)
    product = np.sqrt(sigma2 * nu2)
    scale = float(np.sum(product))
    # d(sigma nu) = (nu^2 d sigma^2 + sigma^2 d nu^2) / (2 sigma nu), in the mean's scaling
    scale_psi = np.divide(
        nu2 * (residual**2 - sigma2) + sigma2 * (nu2 - alpha**2),
        2.0 * product,
        out=np.zeros_like(residual),
        where=product > 0.0,
    )
    quantile = float(student_t.ppf(level, _degrees_of_freedom(rows, clusters, fit.periods)))

    def bounds(strength: float) -> tuple[float, float, float, float]:
        low, high = estimate - strength * scale, estimate + strength * scale
        moved = strength * scale_psi / rows
        spread_low = float(np.sqrt(max(_clustered_squares(psi - moved, clusters, fit.periods))))
        spread_high = float(np.sqrt(max(_clustered_squares(psi + moved, clusters, fit.periods))))
        return low, high, low - quantile * spread_low, high + quantile * spread_high

    def at_share(share: float) -> float:
        """The strength at ``cf_y = cf_d = share``."""
        return abs(rho) * share / float(np.sqrt(1.0 - share))

    strength = abs(rho) * float(np.sqrt(cf_y * cf_d / (1.0 - cf_d)))
    lower, upper, ci_lower, ci_upper = bounds(strength)

    gap = abs(estimate - null)
    facing = 2 if estimate > null else 3  # the confidence bound on the null's side

    def reach(share: float) -> float:
        """How far the confidence bound facing the null still is from it."""
        bound = bounds(at_share(share))[facing]
        return bound - null if estimate > null else null - bound

    if gap == 0.0:
        robustness, robustness_ci = 0.0, 0.0
    elif abs(rho) * scale == 0.0:  # no share moves the bound; the sampling error alone may
        robustness, robustness_ci = 1.0, (0.0 if reach(0.0) <= 0.0 else 1.0)
    else:
        ratio = (gap / (abs(rho) * scale)) ** 2
        robustness = 0.5 * (float(np.sqrt(ratio**2 + 4.0 * ratio)) - ratio)
        robustness_ci = (
            0.0
            if reach(0.0) <= 0.0
            else float(brentq(reach, 0.0, robustness, xtol=1e-12, rtol=4.0 * np.finfo(float).eps))
        )
    return OmittedConfounderBound(
        estimate=estimate,
        bias_scale=scale,
        strength=strength,
        lower=lower,
        upper=upper,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        robustness_value=robustness,
        robustness_value_ci=robustness_ci,
    )


RegionForm = Literal["ridge", "grid"]
""":func:`channel_region`'s radius. ``ridge``: Abbasi-Yadkori, Pal and Szepesvari's (2011), its
log-determinant read against the fit's own ridge. ``grid``: Simpson, Baumgartner, Koehler and
Diehl's (2026, Theorem 2), the least over a fixed grid of prior matrices, the prior's term inside
the root. *Experimental.*"""

# The grid's members besides the prior itself: ``V_bar = 2^j ridge P``, ``j = 1 .. 64``, so its
# ratio to the ridge spans 2 to 2^64, past any log's ``D'D`` at the fit's ridge of 1e-6.
_GRID_POWERS = 64


def _grid_members(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``rho_j`` and ``l_j`` of the grid's ``J = 65`` members, from the eigenvalues ``mu`` of
    ``D'D`` in the ridge's metric: the prior alone, ``V_bar = 0``, whose ``rho`` is 0 and ``l`` is
    ``sum log(1 + mu) = log det(V) - log det(ridge P)``, Corollary 1's; and ``V_bar = 2^j ridge P``,
    whose ``rho_j = rho((ridge P + D'D)^-1 V_bar) = 2^j / (1 + min mu)`` and ``l_j = log det(I +
    (ridge P + V_bar)^-1 D'D) = sum log(1 + mu / (1 + 2^j))``
    (``validation/channel_region_grid.mac``)."""
    powers = 2.0 ** np.arange(1, _GRID_POWERS + 1)
    smallest = float(np.min(spectrum))
    inflation = np.concatenate([[0.0], powers / (1.0 + smallest)])
    log_dets = np.concatenate(
        [
            [float(np.sum(np.log1p(spectrum)))],
            np.sum(np.log1p(spectrum[None, :] / (1.0 + powers[:, None])), axis=1),
        ]
    )
    return inflation, log_dets


@dataclass(frozen=True)
class ChannelRegion:
    """A region that holds the fitted channel's coefficients at a stated level, from one log.

    *Experimental: it may change or be withdrawn in any release.* State ``s``'s coefficients
    ``c_s`` (the channel's row ``s``, raveled lever by lever, each lever's features in turn) lie in
    ``{c : ||c_hat_s - c||_V <= radius[s]}`` for every state at once with probability at least
    ``level``. ``V = D'D + ridge P`` is the matrix the fit's least squares solved, ``D`` the
    channel's design on the action residuals. :func:`channel_region` says where it holds, and
    ``form`` which radius it reads (:data:`RegionForm`).
    """

    level: float
    channel: np.ndarray  # (n, m, k): the centre, the fit's channel
    gram: np.ndarray  # (q, q): V
    radius: np.ndarray  # (n,): each state's radius in V's norm, at the level
    noise: np.ndarray  # (n,): each state's noise scale R, read off the moment's residual, within
    # the fit's clusters where it was given them
    prior: np.ndarray  # (n,): each state's bound on its true coefficients in the ridge's norm
    log_det_ratio: float  # log det V - log det(ridge P)
    channel_degree: int
    form: RegionForm = "ridge"
    # (q,): the eigenvalues of D'D in the ridge's metric, (ridge P)^(-1/2) D'D (ridge P)^(-1/2),
    # which the grid's radius reads; empty under ``ridge``
    spectrum: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))

    def _radii(self, delta: float) -> np.ndarray:
        """Each state's radius at ``1 - delta`` overall, ``delta / n`` a state. Under ``ridge``,
        Abbasi-Yadkori, Pal and Szepesvari's ``R sqrt(2 log(det(V)^(1/2) det(V0)^(-1/2) /
        delta)) + ||c||_V0``. Under ``grid``, the least over :func:`_grid_members` of Simpson,
        Baumgartner, Koehler and Diehl's (2026, Theorem 2) ``sqrt(1 + rho_j) sqrt(||c||_V0^2 + R^2
        (l_j + 2 log(J / delta)))``, a state's ``delta`` shared over the ``J`` members."""
        share = delta / self.noise.size
        if self.form == "ridge":
            return (
                self.noise * math.sqrt(2.0 * math.log(1.0 / share) + self.log_det_ratio)
                + self.prior
            )
        inflation, log_dets = _grid_members(self.spectrum)
        level = log_dets + 2.0 * math.log(inflation.size / share)
        return np.min(
            np.sqrt(1.0 + inflation)[None, :]
            * np.sqrt(self.prior[:, None] ** 2 + self.noise[:, None] ** 2 * level[None, :]),
            axis=1,
        )

    def contains(self, channel: ArrayLike) -> bool:
        """Whether every state's row of ``channel``, shaped as :attr:`channel`, lies in its
        ellipsoid."""
        found = np.asarray(channel, dtype=np.float64)
        if found.shape != self.channel.shape:
            raise ValueError(
                f"a channel of shape {found.shape}; the region's is {self.channel.shape}"
            )
        gap = (np.asarray(self.channel, dtype=np.float64) - found).reshape(found.shape[0], -1)
        return bool(np.all(np.einsum("sp,pq,sq->s", gap, self.gram, gap) <= self.radius**2))

    def _metric(self, state: ArrayLike) -> np.ndarray:
        """``Phi(x)' V^-1 Phi(x)``, ``(m, m)``: an action ``a``'s design row is ``Phi(x) a``."""
        features = np.asarray(
            control_affine_features(
                jnp.asarray(state, dtype=jnp.result_type(float)), self.channel_degree
            ),
            dtype=np.float64,
        )
        levers = self.channel.shape[1]
        lifted = np.kron(np.eye(levers), features[:, None])  # (m k, m)
        return lifted.T @ np.linalg.solve(self.gram, lifted)

    def operator_radius(self, state: ArrayLike) -> float:
        """A bound, at the region's level, on ``||B_hat(x) - B(x)||_2`` at the state ``x``: the
        largest move of the rate that the channel's error makes per unit action norm. It holds at
        every state at once, since each is read off the same ellipsoids."""
        largest = float(np.max(np.linalg.eigvalsh(self._metric(state))))
        return float(np.linalg.norm(self.radius)) * math.sqrt(max(largest, 0.0))

    def rate_error(self, states: ArrayLike, reach: ArrayLike) -> float:
        """A bound, at the region's level, on ``||(B_hat(x) - B(x)) a||`` over the ``(N, n)``
        ``states`` and every action deviation ``a`` with ``|a_j| <= reach[j]``: the rate error
        that the plan's tube reads as a standard error at its studentised quantile
        (``chc.decision``'s ``_model_error`` and ``_field_scale``), read here at a level. The
        largest of a convex form over a box is at a corner, so it reads the ``2^m`` corners."""
        rows = np.atleast_2d(np.asarray(states, dtype=np.float64))
        sizes = np.asarray(reach, dtype=np.float64)
        levers = self.channel.shape[1]
        if sizes.shape != (levers,) or not np.all(sizes >= 0.0):
            raise ValueError(f"reach must be {levers} non-negative numbers, one a lever")
        corners = (
            np.array(
                [
                    [(-1.0) ** ((index >> j) & 1) for j in range(levers)]
                    for index in range(2**levers)
                ]
            )
            * sizes
        )
        largest = max(
            float(np.max(np.einsum("cj,jk,ck->c", corners, self._metric(row), corners)))
            for row in rows
        )
        return float(np.linalg.norm(self.radius)) * math.sqrt(max(largest, 0.0))

    def stability_level(self, rho: float, state: ArrayLike) -> float:
        """The largest level at which :meth:`operator_radius` at ``state`` is within ``rho``: the
        probability the region gives that the channel's error there is at most ``rho``, 0 where
        no level reaches it.

        A certainty-equivalent gain whose stability holds within a radius ``rho`` of the channel
        (``chc.research.regret.ce_explicit_constant_certificate``) then stabilises the plant with at
        least this probability. That certificate's ``rho`` is the true plant's, and its channel is
        the discrete step's, ``dt`` times this one under Euler: pass its radius over ``dt``. Read at
        the estimate, ``rho`` is the estimate's, which the region does not check.

        Solves ``sum_s (R_s A + b_s)^2 = (rho / g)^2`` for ``A``, ``g`` the operator radius per
        unit of the radii, and ``A^2 = 2 log(n / delta) + log_det_ratio`` for ``delta``
        (``validation/tube_levels.mac``).
        """
        if not rho >= 0.0:
            raise ValueError(f"rho={rho!r} must be a non-negative radius")
        unit = math.sqrt(max(float(np.max(np.linalg.eigvalsh(self._metric(state)))), 0.0))
        if unit == 0.0:
            return 1.0
        target = rho / unit
        if self.form == "grid":
            # the radii shrink as delta grows: the largest level whose radii reach rho is a root
            def excess(log_delta: float) -> float:
                return float(np.linalg.norm(self._radii(math.exp(log_delta)))) - target

            if excess(0.0) > 0.0:
                return 0.0
            # below the machine's epsilon ``1 - delta`` is 1, and past it ``J n / delta`` overflows
            floor = math.log(np.finfo(np.float64).eps)
            if excess(floor) <= 0.0:
                return 1.0
            return max(0.0, 1.0 - math.exp(brentq(excess, floor, 0.0, xtol=1e-12)))
        a2 = float(np.sum(self.noise**2))
        a1 = float(np.sum(self.noise * self.prior))
        a0 = float(np.sum(self.prior**2)) - target**2
        if a0 >= 0.0:
            return 0.0
        if a2 == 0.0:
            return 1.0
        root = (-a1 + math.sqrt(a1 * a1 - a2 * a0)) / a2
        delta = self.noise.size * math.exp((self.log_det_ratio - root * root) / 2.0)
        return max(0.0, 1.0 - delta)


def _clustered_scores(
    design: np.ndarray, residual: np.ndarray, clusters: np.ndarray
) -> list[list[np.ndarray]]:
    """Per state, the ``r`` reads of :func:`_clustered_squares`, each ``(q, q)``: the state's
    scores ``D_i e_i``, ``D`` the channel's design ``(N, q)`` and ``e`` the moment's residual
    ``(N, n)``, summed within ``clusters`` and squared at CR1's factor, as the channel's error
    reads them, a state at a time: :func:`channel_region` reads each state's noise off its own."""
    factor = _small_sample(design.shape[0], design.shape[1], _cluster_count(clusters))
    return [
        [
            factor * square
            for square in _clustered_squares(
                (design * residual[:, state, None])[:, None, :], clusters
            )
        ]
        for state in range(residual.shape[1])
    ]


def _whitened(square: np.ndarray, root: np.ndarray) -> np.ndarray:
    """``L^-1 S L^-T`` for ``V = L L'``: ``S`` in ``V``'s norm, whose largest eigenvalue is the
    most ``a' S a`` reaches over ``a' V a = 1``."""
    half = np.linalg.solve(root, square)
    return np.linalg.solve(root, half.T)


def channel_region(
    fit: CausalDynamicsFit,
    level: float = 0.95,
    *,
    bound: float | None = None,
    form: RegionForm = "ridge",
) -> ChannelRegion:
    """The self-normalised confidence ellipsoid of the fitted channel, at ``level``.

    *Experimental: it may change or be withdrawn in any release.* Abbasi-Yadkori, Pal and
    Szepesvari (NeurIPS 2011, Theorems 1 and 2): for ``y_t = <X_t, c> + eta_t`` with ``X_t``
    predictable and ``eta_t`` conditionally ``R``-sub-Gaussian, the ridge estimate ``c_hat``
    satisfies ``||c_hat - c||_V <= R sqrt(2 log(det(V)^(1/2) det(V0)^(-1/2) / delta)) +
    ||c||_V0`` with probability at least ``1 - delta``, ``V = V0 + X'X``. The regressors need only
    be predictable, so a log the plant was controlled in qualifies: a closed-loop policy's actions
    move with the states and the noise before them. Here ``V0 = ridge P``, the fit's own ridge,
    one ellipsoid a state at ``delta / n``.

    It gives the plan's rate error a level (:meth:`ChannelRegion.rate_error`) and the
    certainty-equivalent certificate a probability (:meth:`ChannelRegion.stability_level`),
    beside the standard errors the tube carries at their studentised quantile, which it does not
    move.

    Premises, and where they fail:

    - **No hidden confounder.** The noise must be mean zero given the actions; a confounder the
      fit does not adjust for pushes the rate with the action, and the ellipsoid misses the
      channel, at any ``level``.
    - **No persistent noise.** ``eta`` must be a martingale difference; AR noise is not, unless
      its lags are in the fit (``persistence_check``).
    - **No shared shock the fit was not told of.** Rows are read one at a time, so a shock every
      unit shares in a period, met by levers the units move together, adds up across the
      period's rows beyond each row's ``R``: on 30 units over 20 periods sharing each period's
      shock, the 0.95 region missed the channel on 6.9 % of 1000 logs read row by row. A fit given
      ``clusters`` says which rows share their noise, and ``R`` is then read within them, below.
    - **R is read off the log**: each state's root mean square of the moment's residual, over
      ``N - q`` rows, a plug-in for the theorem's known ``R``, which also counts the cross-fitted
      nuisances' error. The action residuals use nuisances fitted on other rows, so the
      regressors are predictable to the nuisance's first order only. Under the fit's
      ``clusters``, ``R^2`` is the larger of that and the most the state's scores ``D_i e_i``,
      summed within the clusters and squared at CR1's factor, two ways the largest of the three
      reads the fit's error takes, reach in ``V``'s norm: the clusters' sums are then read as the
      theorem's terms.
    - **The prior term** ``||c||_V0`` is ``bound`` where given, else the estimate's own, a
      plug-in; at the default ridge it is about 1e-3 of the coefficients' effect on the rate.
    - **The prior's scale is read off the log too.** ``P`` is the diagonal of the log's mean
      squares of the channel's regressors, so ``V0`` is a plug-in as ``R`` is: the theorem's
      ``V0`` is fixed before the log. The grid's members are multiples of it, on the same terms.

    ``form="grid"``, *experimental*, reads the radius of Simpson, Baumgartner, Koehler and Diehl
    (arXiv 2603.19073, 2026, Theorem 2), proved under the same premises: the noise conditionally
    sub-Gaussian with a known ``R``, the regressors predictable, the true coefficients within
    ``||c||_V0`` of nought. It puts the prior's term inside the root, ``sqrt(||c||_V0^2 + R^2
    (...))`` where Abbasi-Yadkori et al. add it outside, and reads the log-determinant against a
    prior matrix ``V_bar`` near ``D'D`` rather than the fit's ridge, paying ``sqrt(1 + rho((V0 +
    D'D)^-1 V_bar))`` for it: the least over a fixed grid of 65 members, ``V_bar = 0`` and ``2^j
    V0``, each at ``delta / 65`` (:func:`_grid_members`). At the fit's ridge of 1e-6 the
    log-determinant against the ridge is the larger part of the radius, and the grid's is smaller:
    on 2000 closed-loop logs of ``x' = 0.9 x + 0.5 u + e``, ``u = -0.6 x + e'``, 2400 rows, ``V0 =
    1e-6 I``, the median radius of the region of ``(0.9, 0.5)`` read 7.06 by Abbasi-Yadkori et
    al.'s form and 4.76 by the grid's; on 100 logs of 1500 rows of a feedback policy with an
    adjusted confounder, the channel's region at 0.95 read 1.69 and 1.16. Each held the truth on
    every log.

    Under a hidden confounder an instrument's sign-perturbed sums (Szentpeteri and Csaji, 2023)
    stay valid; they are not built here.

    Args:
        fit: an identified Euler fit by adjustment, unweighted, whose actions move every
            direction of the channel.
        level: the joint level over the states, in ``(0, 1)``.
        bound: an upper bound, for every state, on the true coefficients' norm in the ridge's
            metric, ``sqrt(sum_l ridge P_l c_l^2)``; None reads the estimate's.
        form: the radius (:data:`RegionForm`): ``"ridge"``, the default, or ``"grid"``.

    Raises:
        ValueError: on a fit that is not identified, read through an instrument, fitted under
            ``rk4``, weighted, with a direction its actions never move, or with no ridge; a
            ``level`` outside ``(0, 1)``; a negative ``bound``; or a ``form`` not named.
    """
    if form not in ("ridge", "grid"):
        raise ValueError(f"form={form!r} is neither 'ridge' nor 'grid'")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level={level!r} must be in (0, 1)")
    if bound is not None and not 0.0 <= bound < math.inf:
        raise ValueError(f"bound={bound!r} must be a finite non-negative number")
    if not fit.identified:
        raise ValueError("the fit does not identify the channel: there is no region to state")
    if fit.method != "orthogonal":
        raise ValueError(
            f"the ellipsoid is least squares'; a {fit.method!r} fit solves an instrument's moment"
        )
    if fit.integrator != "euler":
        raise ValueError(
            f"the ellipsoid is least squares' on the Euler rates; an {fit.integrator!r} fit is a "
            "fixed point of its own step, whose error the ellipsoid does not carry. Fit with "
            "integrator='euler'"
        )
    if fit.weighted:
        raise ValueError("the ellipsoid is unweighted least squares'; the fit weighed its moment")
    if fit.unmoved is not None and fit.unmoved.shape[1] > 0:
        raise ValueError(
            "the log never moves some direction of the channel: the ellipsoid is unbounded there"
        )
    if fit._channel_gram is None or fit.moment_residual is None:
        raise ValueError("the fit keeps no least-squares Gram of its channel to read")
    ridge = np.asarray(fit._channel_gram.ridge, dtype=np.float64)
    if not np.all(ridge > 0.0):
        raise ValueError("the ridge is zero on some coefficient: V0 is singular, the region void")
    gram = np.asarray(fit._channel_gram.matrix, dtype=np.float64)
    channel = np.asarray(fit.residual.channel, dtype=np.float64)
    residual = np.asarray(fit.moment_residual, dtype=np.float64)
    rows, size = residual.shape[0], gram.shape[0]
    noise = np.sqrt(np.sum(residual**2, axis=0) / max(rows - size, 1))
    if fit.clusters is not None:
        # the fit's clusters share their noise, so a cluster's scores add up beyond the rows' own
        # scale; in the direction they do so most, measured in V's norm, their spread is the R the
        # sum carries. The rows' own R stays the floor: read from few clusters, the clustered
        # spread is noisy, and each row's noise needs its own scale whatever the clusters say
        root = np.linalg.cholesky(gram)
        states = _clustered_scores(fit._channel_gram.design, residual, fit.clusters)
        noise = np.sqrt(
            np.maximum(
                noise**2,
                [
                    max(
                        float(np.max(np.linalg.eigvalsh(_whitened(square, root))))
                        for square in reads
                    )
                    for reads in states
                ],
            )
        )
    coefficients = channel.reshape(channel.shape[0], -1)
    prior = (
        np.sqrt(np.sum(ridge * coefficients**2, axis=1))
        if bound is None
        else np.full(channel.shape[0], float(bound))
    )
    sign, log_det = np.linalg.slogdet(gram)
    if sign <= 0.0:
        raise ValueError("the channel's Gram is not positive definite")
    region = ChannelRegion(
        level=float(level),
        channel=channel,
        gram=gram,
        radius=np.zeros(channel.shape[0]),
        noise=noise,
        prior=prior,
        log_det_ratio=float(log_det - np.sum(np.log(ridge))),
        channel_degree=fit.residual.channel_degree,
        form=form,
        spectrum=np.zeros(0)
        if form == "ridge"
        else np.clip(
            np.linalg.eigvalsh((gram - np.diag(ridge)) / np.sqrt(np.outer(ridge, ridge))),
            0.0,
            None,
        ),
    )
    return dataclasses.replace(region, radius=region._radii(1.0 - level))


@dataclass(frozen=True)
class PersistenceCheck:
    """Whether a fit's transition noise persists within units, read off its channel moment's
    residual by :func:`persistence_check`. *Experimental.*

    The fit's premise is Markov: given the state, the levers and the adjustment set, a transition's
    noise does not depend on the past. Where the noise persists within a unit and a lever persists
    too, the state is a common effect of the past lever and the past noise, so given the state this
    period's lever and noise move together, and the channel is biased by an amount its error does
    not cover (ADR 0064). And a fit given neither ``clusters`` nor ``periods`` reads the error of
    its drift and of the adjustment set's push at its mean on the log, which the tube carries,
    from its rows taken as independent: where the push persists as AR(0.8), that read of the
    mean's variance is 1/8.8 of the mean's own at 200 rows, 1/9 on a long log (ADR 0133).
    Given either, both are read from the
    clusters' sums or the periods' cosines, as the channel's error is.
    """

    correlation: float  # lag-1, pooled over the pairs and the states, each state in its own units
    # two-sided against none, off ``t(G - 1)``, ``G`` the units, or the fit's clusters where it was
    # given them, two ways the smaller count; or on one group, one unit's or a fit's given
    # ``periods``, off ``t(nu)`` across its periods (ADR 0075); nan where the pairs leave no spread
    # to read, as below two pairs, two periods or two clusters
    p_value: float
    pairs: int  # transitions whose unit's transition a period earlier is in the log
    units: int  # ``G``: the units with a pair


def persistence_check(
    fit: CausalDynamicsFit, units: ArrayLike, periods: ArrayLike
) -> PersistenceCheck:
    """The lag-1 autocorrelation of ``fit``'s moment residual within units. *Experimental.*

    ``units`` and ``periods`` label the fit's transitions, ``N`` each: the unit, of any one
    sortable kind, and the period the transition starts in, a whole number of steps. Two
    transitions of one unit whose periods differ by one are a pair. Each state's residual is
    divided by its root mean square, so each state counts alike in any units. The pairs' products,
    summed over the states, are summed within each unit before the test, since the transitions of
    one unit need not be independent; the statistic is their total over its CR1 spread, each
    unit's sum less its pairs' share of the total, against ``t(G - 1)`` (ADR 0062). A fit given
    ``clusters`` says its units' noise moves together within them, several units' or one
    period's, so the units' sums are not independent: the products are summed within the fit's
    clusters instead, each less its pairs' share, two ways the largest of the three reads its
    error takes, against ``t(G - 1)`` for ``G`` the clusters, two ways the smaller count. On panels
    of 4 units a cluster over 10 periods, whose noise and levers share a shock each period within
    a cluster and do not persist, the 5 % test summed within units rejected 12.5 % of 1000 at 5
    clusters and 16.2 % at 30; within the clusters, 4.9 % and 7.0 %. Two ways, the largest of
    the three is conservative where the noise moves with neither label: on panels of 50 and 200
    units over 12 periods, clustered by unit and period, the 5 % test rejected 3.5 % and 2.0 % of
    200 logs, where Cameron, Gelbach and Miller's sum of the two less their intersection rejected
    7.5 % and 6.5 %; under AR(0.2) noise at 50 units it rejected 64.5 %, theirs 72.0 %. The unit
    clusters alone, at 5.5 % and 7.5 % there, rejected 43 % and 63 % of logs whose units shared a
    shock each period that did not persist, and the three 5.5 % and 6.0 %: the largest of the
    three stays, a test that undershoots its size where the units alone would hold it. Pairs of one
    unit alone, or of a fit given ``periods``, whose units share one group's noise, are read
    across their periods instead: the products summed within each period, and the total over the
    spread of their projections on ``nu`` cosines of the pairs' periods, against ``t(nu)``, as
    the fit reads its error (ADR 0075). Pairs that all fall in one period leave no spread to read.

    A rejection says the premise fails, not by how much the channel is off: that depends on how
    far a lever persists as well (:class:`PersistenceCheck`). Adjusting the fit for the state, the
    levers and the adjustment set a period earlier removes the bias where the noise is AR(1): on
    panels of 200 units over 100 periods whose noise and lever are AR(0.7), a channel of 0.8 read
    0.0130 low, 0.0007 low with them, and 0.0250 low with the state's lag alone (ADR 0064).

    A residual that moves with a lever some periods back, a dead time, is
    :func:`lag_check`'s to read: the lag-1 autocorrelation does not name it.

    Raises:
        ValueError: the fit kept no moment residual, as where its channel is not identified; the
            labels are not one a transition; a period is not a whole number; or a unit has two
            transitions starting in one period.
    """
    if fit.moment_residual is None:
        raise ValueError("the fit kept no moment residual: its channel is not identified")
    return _persistence(fit.moment_residual, units, periods, fit.periods is not None, fit.clusters)


def nuisance_shift(fit: CausalDynamicsFit, refit: CausalDynamicsFit, states: ArrayLike) -> float:
    """How far a refit of one log with richer nuisances moves the channel, in the fit's errors:
    the root mean square over ``states``, ``(N, n)``, and the channel's entries of
    ``B_refit(x) - B_fit(x)``, over ``fit.channel_error``. Experimental.

    The moment is orthogonal, so the nuisances' error moves the channel at second order only, but
    a policy or a drift outside their span leaves an error that does not shrink with the log, and
    a bias that ``channel_error`` does not cover. Refit with ``nuisance_knots``, more knots or a
    higher ``nuisance_degree``, and read the shift: within the span, both fits estimate one channel
    and the shift is a fraction of an error; a shift of more than one or two says the fit's
    nuisances miss. On a lever set by ``-1.5 tanh(2 x)`` and a drift with ``sin(2 x)`` in it, at
    4000 rows, the degree-2 polynomial fit (``nuisance_knots=None``) read 53 to 61 errors from its
    refit with ``nuisance_knots="auto"`` on five logs, and that refit 0.02 to 1.9 from its own at
    twice the knots.

    Raises:
        ValueError: if ``fit`` is not identified, so has no error, or the two channels differ in
            shape, so read different features of the state.
    """
    if fit.channel_error is None:
        raise ValueError("the fit is not identified: its channel has no error to read a shift in")
    if fit.residual.channel.shape != refit.residual.channel.shape:
        raise ValueError(
            f"the channels differ in shape, {fit.residual.channel.shape} and "
            f"{refit.residual.channel.shape}: refit with the fit's degree and channel_degree"
        )
    at = jnp.asarray(states).reshape(-1, fit.residual.channel.shape[0])
    shift = jax.vmap(lambda x: refit.residual.control_channel(x) - fit.residual.control_channel(x))(
        at
    )
    return float(jnp.sqrt(jnp.mean(shift**2))) / fit.channel_error


def _persistence(
    residual: ArrayLike,
    units: ArrayLike,
    periods: ArrayLike,
    across_periods: bool,
    clusters: np.ndarray | None = None,
) -> PersistenceCheck:
    """:func:`persistence_check` on a residual of the fit's transitions, ``(N, n)``, read across
    the periods where ``across_periods`` says the fit was given them, and within the fit's
    ``clusters``, its codes a transition, where it was given them."""
    residual = np.asarray(residual, dtype=np.float64)
    labels, steps = np.asarray(units).reshape(-1), np.asarray(periods).reshape(-1)
    if labels.size != residual.shape[0] or steps.size != residual.shape[0]:
        raise ValueError(
            f"units and periods label the fit's {residual.shape[0]} transitions, one each; got "
            f"{labels.size} and {steps.size}"
        )
    if not np.issubdtype(steps.dtype, np.integer):
        raise ValueError(f"periods are whole numbers of steps; got dtype {steps.dtype}")
    codes = np.unique(labels, return_inverse=True)[1].reshape(-1)
    order = np.lexsort((steps, codes))
    earlier, later = order[:-1], order[1:]
    same = codes[earlier] == codes[later]
    if np.any(same & (steps[earlier] == steps[later])):
        raise ValueError("a unit has two transitions starting in one period")
    paired = same & (steps[later] - steps[earlier] == 1)
    earlier, later = earlier[paired], later[paired]
    peak = np.max(np.abs(residual), axis=0)
    live = peak > 0.0
    scaled = residual[:, live] / peak[live]
    scaled = scaled / np.sqrt(np.mean(scaled**2, axis=0))
    products = np.sum(scaled[later] * scaled[earlier], axis=1)
    pairs, count = int(products.size), int(np.unique(codes[later]).size)
    if pairs == 0 or not np.any(live):
        return PersistenceCheck(math.nan, math.nan, pairs, count)
    correlation = float(
        np.sum(products)
        / math.sqrt(float(np.sum(scaled[later] ** 2)) * float(np.sum(scaled[earlier] ** 2)))
    )
    present = np.unique(codes[later])
    sums = np.bincount(codes[later], weights=products)[present]
    # centred, as CR1 centres a regression's scores: uncentred, the statistic stays under
    # sqrt(G - 1), so at six units or fewer it never clears t(G - 1)'s 5 % quantile
    centred = sums - np.bincount(codes[later])[present] * (float(np.sum(products)) / pairs)
    spread = math.sqrt(count / (count - 1) * float(np.sum(centred**2))) if count > 1 else 0.0
    freedom = count - 1
    if clusters is not None:
        # the fit's clusters hold transitions whose noise moves together, several units' or one
        # period's: the units' sums are not independent there, and are read as the fit reads its
        # error, the products summed within its clusters, CR1, two ways the largest of three reads
        codes = np.asarray(clusters).reshape(residual.shape[0], -1)[later]
        codes = np.column_stack(
            [np.unique(column, return_inverse=True)[1].reshape(-1) for column in codes.T]
        )
        groups = _cluster_count(codes)
        spread, freedom = 0.0, groups - 1
        if groups > 1:
            centred_products = (products - float(np.sum(products)) / pairs)[:, None]
            reads = _clustered_squares(
                centred_products[:, :, None], codes[:, 0] if codes.shape[1] == 1 else codes
            )
            spread = math.sqrt(groups / (groups - 1) * max(float(np.squeeze(r)) for r in reads))
    paired_periods = steps[later] - steps[later].min()
    if count == 1 or across_periods:
        # one unit leaves no second unit's sum to read a spread from, and a fit given periods says
        # its units share one group's noise: the products are summed within each period and read
        # across the periods by cosines, as the fit reads its error (ADR 0075)
        spread = 0.0
        if np.unique(paired_periods).size > 1:
            projected = _cosines(paired_periods) @ products
            spread = math.sqrt(float(projected @ projected))
            freedom = _period_cosines(paired_periods)
    p_value = (
        float(2.0 * student_t.sf(abs(float(np.sum(sums))) / spread, freedom))
        if spread > 0.0
        else math.nan
    )
    return PersistenceCheck(correlation, p_value, pairs, count)


@dataclass(frozen=True)
class InterferenceCheck:
    """Whether a fit's transitions move with their neighbours' levers, read off its channel
    moment's residual by :func:`interference_check`. *Experimental.*

    The fit's premise is that a unit's transition depends on its own levers alone. Where a unit's
    levers move its neighbours' states as well, drivers or customers moving between zones, the
    residual carries the neighbours' push, and the channel is the effect of moving one unit with
    its neighbours held, not of a plan that moves them all (ADR 0113).
    """

    # ``(n_states, n_levers)``: each state's residual's slope on the neighbours' mean of each
    # lever, within periods, in the state's rate per unit of the lever; nan for a lever whose
    # neighbours' mean never moves within a period
    coefficient: NDArray[np.float64]
    t: NDArray[np.float64]  # each slope over its CR1 error across the groups
    # the smallest of the slopes' two-sided p-values against ``t(G - 1)``, times the number of
    # slopes, at most 1 (Bonferroni); nan below two groups or with no slope to read
    p_value: float
    transitions: int  # the transitions with a neighbour logged in their period
    groups: int  # ``G``: the groups among them


def interference_check(
    fit: CausalDynamicsFit, exposure: ArrayLike, groups: ArrayLike, periods: ArrayLike
) -> InterferenceCheck:
    """Regress ``fit``'s moment residual on its neighbours' levers, within periods. *Experimental.*

    ``exposure`` is ``(N, m)``: for each of the fit's transitions, the mean of each lever over the
    unit's neighbours in the period the transition starts in, nan where no neighbour is logged
    then; such a transition is left out. ``groups`` and ``periods`` label the transitions, ``N``
    each: the group the errors are summed within, a unit or a declared cluster, and the period.
    The residual and the exposure are taken about their means within each period, so a shock every
    unit shares in a period, which the neighbours' levers may chase, moves neither. Each state's
    residual is regressed on the exposure; a slope's error is the CR1 sandwich over the groups'
    sums of scores, against ``t(G - 1)``, and the check's p-value Bonferroni's over the slopes.

    Under the fit's premise the residual has mean zero given the unit's own states, levers and
    adjustment set, and a neighbour's lever is another unit's: its slope is zero. A rejection says
    the premise fails, by spillover or by a nuisance the fit misses that the neighbours' levers
    track; it does not say by how much the channel is off. Neighbours that are every other unit
    leave an exposure that within a period is the unit's own lever mirrored, which the check cannot
    tell from the unit's own effect.

    Raises:
        ValueError: the fit kept no moment residual, as where its channel is not identified; or
            the exposure or the labels are not one row a transition.
    """
    if fit.moment_residual is None:
        raise ValueError("the fit kept no moment residual: its channel is not identified")
    return _interference(fit.moment_residual, exposure, groups, periods)


def _interference(
    residual: ArrayLike, exposure: ArrayLike, groups: ArrayLike, periods: ArrayLike
) -> InterferenceCheck:
    """:func:`interference_check` on a residual of the fit's transitions, ``(N, n)``."""
    residual = np.asarray(residual, dtype=np.float64)
    rows = residual.shape[0]
    exposed = np.asarray(exposure, dtype=np.float64)
    exposed = exposed[:, None] if exposed.ndim == 1 else exposed
    labels, steps = np.asarray(groups).reshape(-1), np.asarray(periods).reshape(-1)
    if exposed.shape[0] != rows or labels.size != rows or steps.size != rows:
        raise ValueError(
            f"the exposure and the labels are one row for each of the fit's {rows} transitions; "
            f"got {exposed.shape[0]}, {labels.size} and {steps.size}"
        )
    n_states, n_levers = residual.shape[1], exposed.shape[1]
    keep = np.all(np.isfinite(exposed), axis=1)
    codes = np.unique(labels[keep], return_inverse=True)[1].reshape(-1)
    count = int(codes.max()) + 1 if codes.size else 0
    nan = np.full((n_states, n_levers), math.nan)
    if count < 2:
        return InterferenceCheck(nan, nan.copy(), math.nan, int(np.sum(keep)), count)
    period = np.unique(steps[keep], return_inverse=True)[1].reshape(-1)

    def within(values: NDArray[np.float64]) -> NDArray[np.float64]:
        sums = np.zeros((int(period.max()) + 1, values.shape[1]))
        np.add.at(sums, period, values)
        return values - (sums / np.bincount(period)[:, None])[period]

    outcome, design = within(residual[keep]), within(exposed[keep])
    # a column whose spread is rounding of its size moves nothing within the periods
    size = np.max(np.abs(exposed[keep]), axis=0)
    live = np.sqrt(np.mean(design**2, axis=0)) > 64.0 * np.finfo(np.float64).eps * size
    if not np.any(live):
        return InterferenceCheck(nan, nan.copy(), math.nan, int(np.sum(keep)), count)
    design = design[:, live]
    scale = np.sqrt(np.mean(design**2, axis=0))
    standard = design / scale
    gram_inv = np.linalg.inv(standard.T @ standard)
    slope = gram_inv @ standard.T @ outcome  # (live, n_states)
    left = outcome - standard @ slope
    coefficient, t = nan.copy(), nan.copy()
    for state in range(n_states):
        scores = np.zeros((count, standard.shape[1]))
        np.add.at(scores, codes, standard * left[:, state : state + 1])
        cov = count / (count - 1) * gram_inv @ (scores.T @ scores) @ gram_inv
        error = np.sqrt(np.maximum(np.diag(cov), 0.0))
        coefficient[state, live] = slope[:, state] / scale
        with np.errstate(divide="ignore", invalid="ignore"):
            t[state, live] = slope[:, state] / error
    finite = t[np.isfinite(t)]
    p_value = (
        min(1.0, finite.size * float(np.min(2.0 * student_t.sf(np.abs(finite), count - 1))))
        if finite.size
        else math.nan
    )
    return InterferenceCheck(coefficient, t, p_value, int(np.sum(keep)), count)


@dataclass(frozen=True)
class LagCheck:
    """Whether a fit's moment residual moves with its levers some periods back, read by
    :func:`lag_check`. *Experimental.*

    The fit's premise is that a period's lever moves the state over that period alone. Where the
    plant takes the lever some periods late, a dead time, the residual carries the late levers'
    push, and the channel reads the share of it the current lever is correlated with.
    """

    # ``(n_states, lags, n_levers)``: each state's residual's slope on each lever ``lag`` periods
    # back, ``lag`` 1 to ``lags``, the fit's own regressors held, in the state's rate per unit of
    # the lever; nan for a column the log does not move once those are held
    coefficient: NDArray[np.float64]
    t: NDArray[np.float64]  # each slope over its CR1 error across the groups
    # the smallest over the states of each state's joint Wald test of its slopes, against
    # ``F(k, G - 1)`` for its ``k`` slopes, times the number of states, at most 1 (Bonferroni);
    # nan below two groups or with no slope to read
    p_value: float
    lag: int | None  # the lag of the slope largest in ``t``, over the states and levers
    transitions: int  # the transitions whose unit's levers ``lags`` periods back are in the log
    groups: int  # ``G``: the groups among them


def lag_check(
    fit: CausalDynamicsFit,
    data: dict[str, Array],
    units: ArrayLike,
    periods: ArrayLike,
    *,
    lags: int = 3,
    adjust_for: tuple[str, ...] = (),
) -> LagCheck:
    """Regress ``fit``'s moment residual on its levers 1 to ``lags`` periods back. *Experimental.*

    ``data`` holds the columns the fit was given, ``x (N, n)`` and ``u (N, m)``, and those named
    in ``adjust_for``, the fit's adjustment set; ``units`` and ``periods`` label the transitions as
    :func:`persistence_check` reads them. A transition's lever ``k`` periods back is its unit's
    transition's that many periods earlier; a transition without all ``lags`` of them is left out.
    The residual and the late levers are each taken less their least-squares fit on a constant,
    the states, the levers and the adjustment set, the fit's own regressors, and each state's
    residual is regressed on the late levers (Frisch and Waugh 1933). A slope's error is the CR1
    sandwich over the units' sums of scores, or the fit's clusters' where it was given them; each
    state's slopes are tested together by Wald's statistic against ``F(k, G - 1)``, and the check's
    p-value is Bonferroni's over the states. This is the residual test of a model's dynamic part,
    the residual's cross-correlation with past inputs (Ljung 1999, chapter 16).

    Under the fit's premise the residual has mean zero given this period's state, levers and
    adjustment set, and an earlier lever reaches it only through the state: its slope is zero.
    A rejection says the plant reads a lever late. ``lag`` then names the lag that carries it;
    :func:`chc.irf.delay_estimate` reads the delay off the response with an interval, and a state
    that carries the levers' last ``lag`` values holds a discrete dead time exactly
    (:class:`chc.delay.DelayedDynamics` holds a continuous one). It sees what
    :func:`persistence_check` cannot name: on the misspecification suite's two-state plant whose
    lever reaches it two periods late, 200 units over 12 periods, the check rejected on 6 logs of
    6 and named lag 2 on each; on the same plant without the dead time it rejected at 0.05 on 5
    logs of 60, its size.

    It reads across groups, the units or the fit's clusters, and not along one group's periods as
    :func:`persistence_check` does: on one unit, or one cluster, it has no spread to read and
    returns ``p_value`` nan and ``lag`` ``None``, as it does where no late lever moves once the
    fit's regressors are held. With few groups it has little power: ``F(k, G - 1)``'s tail is
    heavy at small ``G``, and at two groups ``F(1, 1)`` rejects at 0.05 only past
    ``tan(0.475 pi)^2 = 161``.

    Raises:
        ValueError: the fit kept no moment residual, as where its channel is not identified; the
            columns or the labels are not one row a transition; ``lags`` is not a positive whole
            number; or a unit has two transitions starting in one period.
    """
    if fit.moment_residual is None:
        raise ValueError("the fit kept no moment residual: its channel is not identified")
    if not (isinstance(lags, (int, np.integer)) and lags >= 1):
        raise ValueError(f"lags={lags!r} is not a positive whole number of periods")
    residual = np.asarray(fit.moment_residual, dtype=np.float64)
    rows = residual.shape[0]
    levers = np.asarray(data["u"], dtype=np.float64).reshape(rows, -1)
    held = np.column_stack(
        [
            np.ones(rows),
            np.asarray(data["x"], dtype=np.float64).reshape(rows, -1),
            levers,
            *(np.asarray(data[name], dtype=np.float64).reshape(rows, -1) for name in adjust_for),
        ]
    )
    labels, steps = np.asarray(units).reshape(-1), np.asarray(periods).reshape(-1)
    if labels.size != rows or steps.size != rows:
        raise ValueError(
            f"units and periods label the fit's {rows} transitions, one each; got {labels.size} "
            f"and {steps.size}"
        )
    codes = np.unique(labels, return_inverse=True)[1].reshape(-1)
    pairs = list(zip(codes.tolist(), steps.tolist(), strict=True))
    where = {pair: row for row, pair in enumerate(pairs)}
    if len(where) != rows:
        raise ValueError("a unit has two transitions starting in one period")
    late = np.array(
        [[where.get((code, step - lag), -1) for lag in range(1, lags + 1)] for code, step in pairs]
    ).reshape(rows, lags)
    keep = np.all(late >= 0, axis=1)
    n_states, n_levers = residual.shape[1], levers.shape[1]
    nan = np.full((n_states, lags, n_levers), math.nan)
    clustered = codes if fit.clusters is None else np.asarray(fit.clusters).reshape(rows, -1)[:, 0]
    groups = np.unique(clustered[keep], return_inverse=True)[1].reshape(-1)
    count = int(groups.max()) + 1 if groups.size else 0
    if count < 2:
        return LagCheck(nan, nan.copy(), math.nan, None, int(np.sum(keep)), count)
    design = levers[late[keep]].reshape(-1, lags * n_levers)
    base = held[keep]

    def partialled(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return values - base @ np.linalg.lstsq(base, values, rcond=None)[0]

    design, outcome = partialled(design), partialled(residual[keep])
    # a column whose spread is rounding of its size moves nothing once the regressors are held
    size = np.max(np.abs(levers), axis=0)
    size = np.tile(np.where(size > 0.0, size, 1.0), lags)
    live = np.sqrt(np.mean(design**2, axis=0)) > 64.0 * np.finfo(np.float64).eps * size
    if not np.any(live):
        return LagCheck(nan, nan.copy(), math.nan, None, int(np.sum(keep)), count)
    design = design[:, live]
    scale = np.sqrt(np.mean(design**2, axis=0))
    standard = design / scale
    gram_inv = np.linalg.inv(standard.T @ standard)
    slope = gram_inv @ standard.T @ outcome
    left = outcome - standard @ slope
    coefficient, t = nan.reshape(n_states, -1).copy(), nan.reshape(n_states, -1).copy()
    p_values = []
    for state in range(n_states):
        scores = np.zeros((count, standard.shape[1]))
        np.add.at(scores, groups, standard * left[:, state : state + 1])
        cov = count / (count - 1) * gram_inv @ (scores.T @ scores) @ gram_inv
        coefficient[state, live] = slope[:, state] / scale
        with np.errstate(divide="ignore", invalid="ignore"):
            t[state, live] = slope[:, state] / np.sqrt(np.maximum(np.diag(cov), 0.0))
        wald = float(slope[:, state] @ np.linalg.pinv(cov) @ slope[:, state])
        p_values.append(float(fisher.sf(wald / standard.shape[1], standard.shape[1], count - 1)))
    t = t.reshape(n_states, lags, n_levers)
    lag = None
    if np.any(np.isfinite(t)):
        largest = np.argmax(np.where(np.isfinite(t), np.abs(t), -np.inf))
        lag = int(np.unravel_index(largest, t.shape)[1]) + 1
    return LagCheck(
        coefficient.reshape(n_states, lags, n_levers),
        t,
        min(1.0, n_states * min(p_values)),
        lag,
        int(np.sum(keep)),
        count,
    )


# --- What a tracked log identifies, and what it does not ---


@dataclass(frozen=True)
class ClosedLoopAttribution:
    """Which coefficients of a control-affine fit the log identifies, on a tracked plant."""

    manifold_slope: float  # m, from regressing the state on the action
    manifold_r2: float  # 1 means exact tracking: the log lies on an affine manifold
    implied_gain: float  # -1/m, the proportional gain of the loop that produced the log
    action_curvature: float  # C, the u^2 coefficient of the response along the manifold
    predicted_interaction: float  # C/m = -gain*C, the identity
    fitted_interaction: float  # b1 from the four-term least squares
    fitted_drift: float  # a from the same fit -- reported so its instability is visible
    design_condition: float  # cond of the standardised design; large means d, a, b0 are not split
    exploration_budget: float  # variance off the manifold, as a share of the state's variance


def closed_loop_gain_attribution(
    states: Array, actions: Array, rates: Array
) -> ClosedLoopAttribution:
    """Attribute a control-affine fit's coefficients to the controller and to the plant.

    The question is *why* the interaction coefficient ``b1`` of
    ``dx/dt = d + a*x + (b0 + b1*x)*u`` comes out large and negative on a setpoint-tracked zone.
    The answer is a property of the log, not of the plant. A proportional loop
    ``u = gain*(setpoint - x) + u0`` puts every sample on an affine manifold ``x = c + m*u`` with
    ``m = -1/gain``, and restricted to that manifold the four-term class collapses to a quadratic in
    the action::

        d + a*x + (b0 + b1*x)*u  =  (d + a*c) + (a*m + b0 + b1*c)*u + (b1*m)*u^2

    Only ``b1`` reaches the ``u^2`` term, so **``b1`` is the identified coefficient and the pole is
    not** -- the inverse of the usual reading, and the reason the reported pole is a units artefact
    while ``lambda(u) = a + b1*u`` holds. Matching against an observed
    response ``A + B*u + C*u^2`` gives

        ``b1 = C/m = -gain*C``,

    so the sign is decided by the curvature of the response in the action and the **magnitude by the
    controller**: a tighter tracker reports a bigger interaction from identical physics.
    ``proofs/closed_loop_attribution.v`` proves the matching identity and exhibits the explicit
    one-parameter family that leaves ``d``, ``a`` and ``b0`` free.

    This also **refutes** the standing guess that ``b1 = -1/gain``. That is the *manifold slope*,
    not the interaction; equating them forces ``C = 1/gain^2``, a constraint on the plant rather
    than an identity (``guess_is_a_constraint_not_an_identity``).

    ``exploration_budget`` is what buys the rest back: variation off the manifold restores the
    design's rank, and with it the separation of ``a`` from ``b0``. Reading a pole off a fit whose
    budget is ~0 is reading the regulariser.
    """
    x = jnp.asarray(_real_numbers(states, "states"), dtype=jnp.float64).ravel()
    u = jnp.asarray(_real_numbers(actions, "actions"), dtype=jnp.float64).ravel()
    y = jnp.asarray(_real_numbers(rates, "rates"), dtype=jnp.float64).ravel()

    affine = jnp.stack([jnp.ones_like(u), u], axis=1)
    manifold = jnp.linalg.lstsq(affine, x, rcond=None)[0]
    slope = float(manifold[1])
    residual = x - affine @ manifold

    quadratic = jnp.stack([jnp.ones_like(u), u, u**2], axis=1)
    response = jnp.linalg.lstsq(quadratic, y, rcond=None)[0]
    curvature = float(response[2])

    design = jnp.stack([jnp.ones_like(x), x, u, x * u], axis=1)
    coefficients = jnp.linalg.lstsq(design, y, rcond=None)[0]
    condition = float(jnp.linalg.cond(design / _units.norms(design)))

    state_variance = float(jnp.var(x))
    return ClosedLoopAttribution(
        manifold_slope=slope,
        manifold_r2=_r_squared(x, affine @ manifold),
        # A loop with zero gain leaves the state unexplained by the action; reporting an infinite
        # gain is the honest answer, not a clamped one.
        implied_gain=float("inf") if slope == 0.0 else -1.0 / slope,
        action_curvature=curvature,
        predicted_interaction=float("nan") if slope == 0.0 else curvature / slope,
        fitted_interaction=float(coefficients[3]),
        fitted_drift=float(coefficients[1]),
        design_condition=condition,
        exploration_budget=(
            0.0 if state_variance == 0.0 else float(jnp.var(residual)) / state_variance
        ),
    )


@dataclass(frozen=True)
class ClosedLoopAttributionCertificate:
    """Two arms: an interaction the plant has, and one the tracking loop manufactures."""

    gains: tuple[float, ...]
    true_interaction: float  # the plant's own b1 in arm A
    recovered: tuple[float, ...]  # fitted b1 under exact tracking -- should equal it at every gain
    drift_error: tuple[float, ...]  # |fitted a - true a| under the same fits; should NOT be small
    curvature: float  # the u^2 term arm B's plant has and the fitted class cannot represent
    spurious: tuple[float, ...]  # fitted b1 in arm B: pure artefact, -gain*curvature
    spurious_predicted: tuple[float, ...]  # -gain*curvature
    refuted_guess: tuple[float, ...]  # -1/gain: the manifold slope, once guessed to be b1
    exploration: tuple[float, ...]  # off-manifold noise levels for the recovery sweep
    drift_error_by_exploration: tuple[float, ...]  # |fitted a - true a| as exploration grows
    condition_by_exploration: tuple[float, ...]
    ok: bool


def closed_loop_attribution_certificate(
    gains: Sequence[float] = (0.5, 1.0, 2.6, 8.0),
    exploration: Sequence[float] = (0.0, 0.05, 0.25, 1.0),
    samples: int = 4000,
    seed: int = 0,
) -> ClosedLoopAttributionCertificate:
    """Separate the interaction a plant HAS from the one a tracking loop MANUFACTURES.

    Arm A: the plant really is ``d + a*x + (b0 + b1*x)*u``. Under exact tracking the design is
    singular, yet ``b1`` comes back exactly while the drift ``a`` does not -- the non-identification
    is real and it is the pole that suffers, not the interaction.

    Arm B: the plant has **no** interaction, but its response carries a ``u^2`` term the fitted
    class cannot represent. The fit answers with ``b1 = -gain*curvature``: a coefficient that is
    entirely an artefact of misspecification amplified by the loop, growing linearly with the
    controller's gain. That is the mechanism behind a large negative ``b1`` on a tracked log, and
    why
    the number cannot be read as authority-falls-with-temperature without checking the budget.

    The exploration sweep is the remedy: variation off the manifold restores the design's rank and
    the drift with it. All three claims are measured here, and the middle one is the one that would
    embarrass the entry if ``b1`` turned out to be as unstable as ``a``.
    """
    rng = np.random.default_rng(seed)
    true_drift, true_offset, true_channel, true_interaction = -0.35, 1.0, 0.40, -0.30
    curvature = 0.1454

    def tracked(gain: float, noise: float) -> tuple[Array, Array]:
        actions = 20.0 + 2.0 * rng.standard_normal(samples)
        states = 17.0 - actions / gain + noise * rng.standard_normal(samples)
        return jnp.asarray(states), jnp.asarray(actions)

    recovered, drift_error, spurious = [], [], []
    for gain in gains:
        states, actions = tracked(gain, 0.0)
        rates = (
            true_offset + true_drift * states + (true_channel + true_interaction * states) * actions
        )
        fit = closed_loop_gain_attribution(states, actions, rates)
        recovered.append(fit.fitted_interaction)
        drift_error.append(abs(fit.fitted_drift - true_drift))

        curved = true_offset + true_drift * states + true_channel * actions + curvature * actions**2
        spurious.append(closed_loop_gain_attribution(states, actions, curved).fitted_interaction)

    drift_by_noise, condition_by_noise = [], []
    for noise in exploration:
        states, actions = tracked(2.6, noise)
        rates = (
            true_offset + true_drift * states + (true_channel + true_interaction * states) * actions
        )
        fit = closed_loop_gain_attribution(states, actions, rates)
        drift_by_noise.append(abs(fit.fitted_drift - true_drift))
        condition_by_noise.append(fit.design_condition)

    predicted = tuple(-g * curvature for g in gains)
    return ClosedLoopAttributionCertificate(
        gains=tuple(float(g) for g in gains),
        true_interaction=true_interaction,
        recovered=tuple(recovered),
        drift_error=tuple(drift_error),
        curvature=curvature,
        spurious=tuple(spurious),
        spurious_predicted=predicted,
        refuted_guess=tuple(-1.0 / g for g in gains),
        exploration=tuple(float(e) for e in exploration),
        drift_error_by_exploration=tuple(drift_by_noise),
        condition_by_exploration=tuple(condition_by_noise),
        ok=(
            all(abs(b - true_interaction) < 1e-6 for b in recovered)
            and max(drift_error) > 1e-2  # the pole is NOT recovered, and that is the point
            and all(abs(s - p) < 1e-6 for s, p in zip(spurious, predicted, strict=True))
            and drift_by_noise[-1] < 1e-6  # exploration buys the drift back
            and condition_by_noise[-1] < condition_by_noise[0]
        ),
    )
