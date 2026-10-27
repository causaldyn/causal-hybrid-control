"""Koopman dynamics + LQR: lift a nonlinear system to a global-linear one, then control it.

The Koopman idea: in a lifted feature space ``phi(x)`` the dynamics are approximately *linear*,
``phi(x') ~ A phi(x) + B u``. Fit ``A, B`` from transitions by least squares (EDMD); the nonlinear
system becomes linear, so control is a fast **discrete LQR** in the lifted space, not a
gradient-descent MPC. The LQR is exact for the linear surrogate, not for the plant: through a
nonlinear dictionary the input of even a control-affine plant can enter the lift with a
state-dependent gain, so a linear lift is misspecified there in general, and LQR or MPC on it
carries no closed-loop guarantee for the plant (Strässer et al. 2026, Annual Reviews in Control
61). This mirrors the CHC hybrid philosophy (a structured global model plus a correction); the
modern, Residual-Koopman-MPC-lineage control backend. The dictionary is polynomial in the state
less its centre, the midpoint of the states the fit read, and ends in a constant; the first
``state_dim`` features are the centred state, so decoding is a slice and the centre added back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np

from chc import _units
from chc.frames import _real_numbers
from chc.regret import dlqr

__all__ = [
    "KoopmanModel",
    "koopman_controller",
    "koopman_lqr_gain",
]


def _lift(x: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial dictionary ``phi(x)``: the state first, then higher-order monomials, then 1."""
    x = np.atleast_2d(np.asarray(x, float))
    n, d = x.shape
    feats = [x[:, j] for j in range(d)]  # raw state components first -> decoding is a slice
    feats.extend(
        np.prod([x[:, j] for j in combo], axis=0)
        for deg in range(2, degree + 1)
        for combo in combinations_with_replacement(range(d), deg)
    )
    feats.append(np.ones(n))
    return np.column_stack(feats)


def _lift_jacobian(x: np.ndarray, degree: int) -> np.ndarray:
    """``d phi / d x`` at one state, one row per feature of :func:`_lift`."""
    x = np.asarray(x, float)
    d = x.size
    rows = [np.eye(d)]
    for deg in range(2, degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            row = np.zeros((1, d))
            for i in set(combo):
                rest = list(combo)
                rest.remove(i)
                row[0, i] = combo.count(i) * np.prod(x[rest])
            rows.append(row)
    rows.append(np.zeros((1, d)))
    return np.vstack(rows)


_FORMS = ("linear", "bilinear")
_FOLDS = 5


def _design(phi: np.ndarray, u: np.ndarray, form: str) -> np.ndarray:
    """``[phi(x), u]``, and for the bilinear form each action's products ``u_j phi(x)`` after,
    less the constant's, which is ``u_j`` itself."""
    if form == "linear":
        return np.column_stack([phi, u])
    return np.column_stack([phi, u, *(u[:, [j]] * phi[:, :-1] for j in range(u.shape[1]))])


def _solve(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    gram = design.T @ design + ridge * np.diag(_units.mean_squares_np(design))
    return np.linalg.solve(gram, design.T @ target)


@dataclass
class KoopmanModel:
    """Koopman model ``phi(x') ~ A phi(x) + B u`` fit by least squares on a polynomial lift.

    ``ridge`` is the Tikhonov term on each coefficient, in units of its column's mean square, so
    the fit reads the same in any units of the state and the action. Added as a constant, it
    outweighed an action logged in millionths of its units and set ``B`` near zero.

    ``form="bilinear"`` fits ``phi(x') ~ A phi(x) + B u + sum_j u_j H_j phi(x)``. A control-affine
    plant moves each nonlinear feature of the lift by a term that is the action times a function of
    the state, so its linear lift is misspecified however rich the dictionary; the bilinear one
    carries that term (Strässer et al., Annual Reviews in Control 61, 2026). The default
    ``form="linear"`` fits as before, bit for bit.

    ``fit`` reports ``closure_residuals``: for each feature of the lift, the root mean square of
    its one-step error on transitions held out of the fit, over the feature's spread about its
    mean at the next state. The transitions are cut into five contiguous blocks, each predicted
    by a model fit on the other four; the model kept is fit on all of them. A lift whose span the
    dynamics map into itself reads 0 on noise-free data (ResDMD's residual, Colbrook and
    Townsend, on the whole dictionary); one whose span they leave reads the share of each feature's
    variation it does not predict. The ratio is read feature by feature, so it reads the same in
    any units of the state and the action. Over the feature's root mean square instead, a level far
    from zero would pass for prediction: a state read 273.15 higher, as in kelvin, takes the
    degree-one lift of ``x' = 2x - x^3 + u/10`` from 0.71, refused, to 0.0024, accepted. A feature
    the log never moves has no spread, and is read over its root mean square.

    The lift is of the state less its centre, the midpoint of the logged states, and it holds a
    constant. Without them a level was read as dynamics: the monomials of ``x + c`` span a
    different space for every ``c``, and none held the constant that an affine map of the state
    needs. At degree 3 the same plant logged at levels 0, 1, 5 and 273.15 was refused, accepted,
    refused and refused, and controllers accepted on a level diverged on the plant. Centred, the
    model and its verdict read the same at every level, and with the constant the span of a
    degree-``d`` lift is every polynomial of degree ``d``, which no shift of the state leaves.
    """

    degree: int = 3
    ridge: float = 1e-6
    state_dim: int = 0
    form: str = "linear"
    closure_residuals: np.ndarray | None = field(default=None, init=False)
    _a: np.ndarray | None = field(default=None, init=False, repr=False)
    _b: np.ndarray | None = field(default=None, init=False, repr=False)
    _h: np.ndarray | None = field(default=None, init=False, repr=False)
    _centre: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.form not in _FORMS:
            raise ValueError(f"form must be one of {_FORMS}, got {self.form!r}")

    def fit(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> KoopmanModel:
        x, u, x_next = (
            np.asarray(_real_numbers(v, name), float)
            for v, name in ((x, "x"), (u, "u"), (x_next, "x_next"))
        )
        object.__setattr__(self, "state_dim", x.shape[1])
        # the midpoint, not the mean: a state the log holds at one value centres to exactly zero
        self._centre = 0.5 * np.max(x, axis=0) + 0.5 * np.min(x, axis=0)
        phi, phi_next, u2 = self._features(x), self._features(x_next), np.atleast_2d(u)
        design = _design(phi, u2, self.form)  # [phi(x), u (, u_j phi(x))] -> phi(x')
        self._set(_solve(design, phi_next, self.ridge), phi.shape[1], u2.shape[1])
        self.closure_residuals = self._held_out(design, phi_next)
        return self

    def _features(self, x: np.ndarray) -> np.ndarray:
        if self._centre is None:
            raise RuntimeError("call fit() before using the model")
        return _lift(np.atleast_2d(np.asarray(x, float)) - self._centre, self.degree)

    def _set(self, coef: np.ndarray, n_feat: int, d_u: int) -> None:
        self._a, self._b = coef[:n_feat].T, coef[n_feat : n_feat + d_u].T
        if self.form == "bilinear":
            # the constant's column of each H_j is u_j's own, already in B: zero
            h = coef[n_feat + d_u :].T.reshape(n_feat, d_u, n_feat - 1).transpose(1, 0, 2)
            self._h = np.concatenate([h, np.zeros((d_u, n_feat, 1))], axis=2)

    def _held_out(self, design: np.ndarray, phi_next: np.ndarray) -> np.ndarray:
        """Each feature's held-out root mean square error over its own, read across the folds."""
        n = design.shape[0]
        if n < 2:
            return np.full(phi_next.shape[1], np.nan)
        error = np.empty_like(phi_next)
        for block in np.array_split(np.arange(n), _FOLDS):
            train = np.ones(n, bool)
            train[block] = False
            coef = _solve(design[train], phi_next[train], self.ridge)
            error[block] = phi_next[block] - design[block] @ coef
        sse = np.sum(error**2, axis=0)
        spread = np.sum((phi_next - np.mean(phi_next, axis=0)) ** 2, axis=0)
        # A feature the log never moves has no spread, and its mean need not round to its value.
        size = np.where(np.ptp(phi_next, axis=0) > 0.0, spread, np.sum(phi_next**2, axis=0))
        ratio = np.divide(sse, size, out=np.where(sse > 0.0, np.inf, 0.0), where=size > 0.0)
        return np.sqrt(ratio)

    def _step(self, phi: np.ndarray, u: np.ndarray) -> np.ndarray:
        if self._a is None or self._b is None:
            raise RuntimeError("call fit() before predict()")
        phi_next = phi @ self._a.T + u @ self._b.T
        if self._h is not None:
            phi_next = phi_next + np.einsum("nj,jkf,nf->nk", u, self._h, phi)
        return phi_next

    def residuals(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        """Each transition's one-step error in the lift, ``phi(x') - A phi(x) - B u (- ...)``.

        Pass transitions the model was not fit on: on its own the error reads the fit, not the
        closure.
        """
        x, u, x_next = (
            np.asarray(_real_numbers(v, name), float)
            for v, name in ((x, "x"), (u, "u"), (x_next, "x_next"))
        )
        return self._features(x_next) - self._step(self._features(x), np.atleast_2d(u))

    def channel(self, x: np.ndarray) -> np.ndarray:
        """The lifted model's derivative in the action at state ``x``: ``B + [H_j phi(x)]_j``."""
        if self._b is None:
            raise RuntimeError("call fit() before channel()")
        if self._h is None:
            return self._b
        return self._b + np.einsum("jkf,f->kj", self._h, self._features(x)[0])

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        phi_next = self._step(self._features(x), np.atleast_2d(u))
        # decode: the centred state is the leading block
        return phi_next[:, : self.state_dim] + self._centre

    def rollout(self, x0: np.ndarray, us: np.ndarray) -> np.ndarray:
        x = np.asarray(x0, float)
        us = np.asarray(us, float)
        states = [x]
        for t in range(us.shape[0]):
            x = self.predict(x[None, :], us[t][None, :])[0]
            states.append(x)
        return np.stack(states)


def koopman_lqr_gain(
    model: KoopmanModel,
    q_state: np.ndarray,
    r: np.ndarray,
    *,
    x_target: np.ndarray | None = None,
    max_residual: float = 0.5,
) -> np.ndarray:
    """Infinite-horizon discrete-LQR gain in the lifted space (state cost via the decode slice).

    The gain is the lifted model's, and it is refused where the lift does not close: where a
    feature's held-out closure residual exceeds ``max_residual``, or was not read. The default of
    0.5 refuses a lift that predicts less than three quarters of some feature's variance on
    transitions it was not fit on. A bilinear model's channel depends on the state: its gain is
    the LQR of the model linearised at ``x_target``, which it then requires. A linear model's gain
    does not depend on ``x_target``.

    The lift's constant moves with no action and costs nothing, so the LQR is solved on the other
    features and the constant's column of the gain is zero: the controller reads the lift's
    difference from the target's, where the constant cancels.

    Raises:
        RuntimeError: before the model is fit.
        ValueError: where the lift does not close, or a bilinear model is given no ``x_target``.
    """
    a, b, residuals = model._a, model._b, model.closure_residuals
    if a is None or b is None or residuals is None:
        raise RuntimeError("fit the model before computing an LQR gain")
    worst = int(np.argmax(np.nan_to_num(residuals, nan=np.inf)))
    if not residuals[worst] <= max_residual:
        raise ValueError(
            f"the lift does not close: feature {worst}'s held-out closure residual is "
            f"{residuals[worst]:.3g}, not within max_residual={max_residual}; enrich the lift, fit "
            "form='bilinear', or pass a larger max_residual"
        )
    if model.form == "bilinear":
        if x_target is None:
            raise ValueError("a bilinear model's channel depends on the state: pass x_target")
        b = model.channel(np.asarray(x_target, float))
    moving = a.shape[0] - 1  # every feature but the constant, which is last
    lift_q = np.zeros((moving, moving))
    lift_q[: model.state_dim, : model.state_dim] = np.asarray(
        q_state, float
    )  # cost only on raw state
    gain = dlqr(a[:moving, :moving], b[:moving], lift_q, np.asarray(r, float))[0]
    return np.column_stack([gain, np.zeros(gain.shape[0])])  # gain G


def koopman_controller(model: KoopmanModel, gain: np.ndarray, x_target: np.ndarray):
    """Feedback law ``u = -G (phi(x) - phi(x_target))`` regulating the system to ``x_target``.

    The law lifts the measured state afresh at every step, so the loop it closes is the model's
    prediction re-lifted, not the lifted model, whose own loop the LQR makes stable whatever the
    lift. Near the target that loop moves a deviation by ``J = [(A - B_t G) dphi/dx]``, the state's
    rows, ``B_t`` the model's channel at the target; the law is refused where ``J``'s spectral
    radius is not below 1, since then the loop leaves the target even in the model.

    Raises:
        ValueError: where the model's closed loop is not stable at ``x_target``.
    """
    x_target = np.asarray(x_target, float)
    phi_target = model._features(x_target)[0]
    a = model._a
    assert a is not None  # _features has checked the model is fit
    centred = x_target - model._centre
    loop = (a - model.channel(x_target) @ gain)[: model.state_dim] @ _lift_jacobian(
        centred, model.degree
    )
    radius = float(np.max(np.abs(np.linalg.eigvals(loop))))
    if not radius < 1.0:
        raise ValueError(
            f"the model's closed loop is not stable at x_target: the controller, re-lifting the "
            f"state at each step, moves a deviation by a map of spectral radius {radius:.3g}"
        )

    def control(x: np.ndarray) -> np.ndarray:
        return -gain @ (model._features(x)[0] - phi_target)

    return control
