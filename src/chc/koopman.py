"""Koopman dynamics + LQR: lift a nonlinear system to a global-linear one, then control it.

The Koopman idea: in a lifted feature space ``phi(x)`` the dynamics are approximately *linear*,
``phi(x') ~ A phi(x) + B u``. Fit ``A, B`` from transitions by least squares (EDMD); the nonlinear
system becomes linear, so control is a fast, exact **discrete LQR** in the lifted space, not a
gradient-descent MPC. This mirrors the CHC hybrid philosophy (a structured global model plus a
correction); the modern, Residual-Koopman-MPC-lineage control backend (see ``plans/16``). The
dictionary is polynomial; the first ``state_dim`` features are the raw state, decoding is a slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import solve_discrete_are


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


@dataclass
class KoopmanModel:
    """Koopman model ``phi(x') ~ A phi(x) + B u`` fit by least squares on a polynomial lift."""

    degree: int = 3
    ridge: float = 1e-6
    state_dim: int = 0
    _a: np.ndarray | None = field(default=None, init=False, repr=False)
    _b: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> KoopmanModel:
        x, u, x_next = (np.asarray(v, float) for v in (x, u, x_next))
        object.__setattr__(self, "state_dim", x.shape[1])
        phi, phi_next, u2 = _lift(x, self.degree), _lift(x_next, self.degree), np.atleast_2d(u)
        design = np.column_stack([phi, u2])  # [phi(x), u] -> phi(x')
        gram = design.T @ design + self.ridge * np.eye(design.shape[1])
        coef = np.linalg.solve(gram, design.T @ phi_next)  # (n_feat + d_u, n_feat)
        n_feat = phi.shape[1]
        self._a, self._b = coef[:n_feat].T, coef[n_feat:].T  # A (n_feat,n_feat), B (n_feat,d_u)
        return self

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if self._a is None or self._b is None:
            raise RuntimeError("call fit() before predict()")
        phi = _lift(x, self.degree)
        phi_next = phi @ self._a.T + np.atleast_2d(u) @ self._b.T
        return phi_next[:, : self.state_dim]  # decode: raw state is the leading block

    def rollout(self, x0: np.ndarray, us: np.ndarray) -> np.ndarray:
        x = np.asarray(x0, float)
        us = np.asarray(us, float)
        states = [x]
        for t in range(us.shape[0]):
            x = self.predict(x[None, :], us[t][None, :])[0]
            states.append(x)
        return np.stack(states)


def koopman_lqr_gain(model: KoopmanModel, q_state: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Infinite-horizon discrete-LQR gain in the lifted space (state cost via the decode slice)."""
    a, b = model._a, model._b
    if a is None or b is None:
        raise RuntimeError("fit the model before computing an LQR gain")
    n_feat = a.shape[0]
    lift_q = np.zeros((n_feat, n_feat))
    lift_q[: model.state_dim, : model.state_dim] = np.asarray(
        q_state, float
    )  # cost only on raw state
    p = solve_discrete_are(a, b, lift_q, np.asarray(r, float))
    return np.linalg.solve(np.asarray(r, float) + b.T @ p @ b, b.T @ p @ a)  # gain G


def koopman_controller(model: KoopmanModel, gain: np.ndarray, x_target: np.ndarray):
    """Feedback law ``u = -G (phi(x) - phi(x_target))`` regulating the system to ``x_target``."""
    phi_target = _lift(x_target, model.degree)[0]

    def control(x: np.ndarray) -> np.ndarray:
        return -gain @ (_lift(x, model.degree)[0] - phi_target)

    return control
