"""Koopman dynamics + LQR: lift a nonlinear system to a global-linear one, then control it.

The Koopman idea: in a lifted feature space ``phi(x)`` the dynamics are approximately *linear*,
``phi(x') ~ A phi(x) + B u``. Fit ``A, B`` from transitions by least squares (EDMD); the nonlinear
system becomes linear, so control is a fast, exact **discrete LQR** in the lifted space, not a
gradient-descent MPC. This mirrors the CHC hybrid philosophy (a structured global model plus a
correction); the modern, Residual-Koopman-MPC-lineage control backend. The
dictionary is polynomial; the first ``state_dim`` features are the raw state, decoding is a slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np

from chc import _units
from chc.frames import _real_numbers
from chc.regret import dlqr


def _lift(x: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial dictionary ``phi(x)``: raw state first, then higher-order monomials (no bias)."""
    x = np.atleast_2d(np.asarray(x, float))
    _, d = x.shape
    feats = [x[:, j] for j in range(d)]  # raw state components first -> decoding is a slice
    feats.extend(
        np.prod([x[:, j] for j in combo], axis=0)
        for deg in range(2, degree + 1)
        for combo in combinations_with_replacement(range(d), deg)
    )
    return np.column_stack(feats)


_FORMS = ("linear", "bilinear")
_FOLDS = 5


def _design(phi: np.ndarray, u: np.ndarray, form: str) -> np.ndarray:
    """``[phi(x), u]``, and for the bilinear form each action's products ``u_j phi(x)`` after."""
    if form == "linear":
        return np.column_stack([phi, u])
    return np.column_stack([phi, u, *(u[:, [j]] * phi for j in range(u.shape[1]))])


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
    its one-step error on transitions held out of the fit, over the root mean square of the
    feature at the next state. The transitions are cut into five contiguous blocks, each predicted
    by a model fit on the other four; the model kept is fit on all of them. A lift whose span the
    dynamics map into itself reads 0 on noise-free data (ResDMD's residual, Colbrook and
    Townsend, on the whole dictionary); one whose span they leave reads the share of each feature
    it does not predict. The ratio is read feature by feature, so it reads the same in any units of
    the state and the action.
    """

    degree: int = 3
    ridge: float = 1e-6
    state_dim: int = 0
    form: str = "linear"
    closure_residuals: np.ndarray | None = field(default=None, init=False)
    _a: np.ndarray | None = field(default=None, init=False, repr=False)
    _b: np.ndarray | None = field(default=None, init=False, repr=False)
    _h: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.form not in _FORMS:
            raise ValueError(f"form must be one of {_FORMS}, got {self.form!r}")

    def fit(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> KoopmanModel:
        x, u, x_next = (
            np.asarray(_real_numbers(v, name), float)
            for v, name in ((x, "x"), (u, "u"), (x_next, "x_next"))
        )
        object.__setattr__(self, "state_dim", x.shape[1])
        phi, phi_next, u2 = _lift(x, self.degree), _lift(x_next, self.degree), np.atleast_2d(u)
        design = _design(phi, u2, self.form)  # [phi(x), u (, u_j phi(x))] -> phi(x')
        self._set(_solve(design, phi_next, self.ridge), phi.shape[1], u2.shape[1])
        self.closure_residuals = self._held_out(design, phi_next)
        return self

    def _set(self, coef: np.ndarray, n_feat: int, d_u: int) -> None:
        self._a, self._b = coef[:n_feat].T, coef[n_feat : n_feat + d_u].T
        if self.form == "bilinear":
            self._h = coef[n_feat + d_u :].T.reshape(n_feat, d_u, n_feat).transpose(1, 0, 2)

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
        sse, size = np.sum(error**2, axis=0), np.sum(phi_next**2, axis=0)
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
        phi = _lift(x, self.degree)
        return _lift(x_next, self.degree) - self._step(phi, np.atleast_2d(u))

    def channel(self, x: np.ndarray) -> np.ndarray:
        """The lifted model's derivative in the action at state ``x``: ``B + [H_j phi(x)]_j``."""
        if self._b is None:
            raise RuntimeError("call fit() before channel()")
        if self._h is None:
            return self._b
        return self._b + np.einsum("jkf,f->kj", self._h, _lift(x, self.degree)[0])

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        phi_next = self._step(_lift(x, self.degree), np.atleast_2d(u))
        return phi_next[:, : self.state_dim]  # decode: raw state is the leading block

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
    0.5 refuses a lift that predicts less than three quarters of some feature's mean square on
    transitions it was not fit on. A bilinear model's channel depends on the state: its gain is
    the LQR of the model linearised at ``x_target``, which it then requires. A linear model's gain
    does not depend on ``x_target``.

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
    n_feat = a.shape[0]
    lift_q = np.zeros((n_feat, n_feat))
    lift_q[: model.state_dim, : model.state_dim] = np.asarray(
        q_state, float
    )  # cost only on raw state
    return dlqr(a, b, lift_q, np.asarray(r, float))[0]  # gain G


def koopman_controller(model: KoopmanModel, gain: np.ndarray, x_target: np.ndarray):
    """Feedback law ``u = -G (phi(x) - phi(x_target))`` regulating the system to ``x_target``."""
    phi_target = _lift(x_target, model.degree)[0]

    def control(x: np.ndarray) -> np.ndarray:
        return -gain @ (_lift(x, model.degree)[0] - phi_target)

    return control
