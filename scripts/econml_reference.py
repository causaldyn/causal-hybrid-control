"""The two checks behind the README's EconML row: what EconML already estimates, run on it.

    matrix   ``econml.dml.LinearDML`` on one-step control-affine data, ``y = B(x) u + g(x, z) + e``,
             with ``u`` confounded by ``z`` and ``B(x) = B0 + B1 x`` a 2x2 matrix. Its
             ``const_marginal_effect`` is that matrix, read at ``x`` in {-1, 0, 1}.
    dynamic  ``econml.panel.dml.DynamicDML`` on a linear plant logged over three periods by a policy
             that reads the state and an observed confounder. Its per-period effects on the last
             period's outcome are the plant's impulse response ``c' A^(m-1-t) B``.

Measured 2026-09-28 with EconML 0.17.0 and pandas 3.0.6, on Python 3.12 and 3.14: the matrix to
0.068 at worst, the impulse response to 0.013, and every period's 95% interval covering the truth.

EconML is not a dependency of chc (``chc.estimators`` says why), so this runs in a throwaway
environment:

    uv run --no-project --with econml==0.17.0 --with 'pandas>=3' python scripts/econml_reference.py
"""

from __future__ import annotations

import numpy as np
from econml.dml import LinearDML
from econml.panel.dml import DynamicDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

SEED = 20260928


def matrix_effect() -> None:
    rng = np.random.default_rng(SEED)
    n = 4000
    x = rng.normal(size=(n, 1))
    z = rng.normal(size=(n, 1))
    u = np.hstack([0.8 * z + 0.3 * x, -0.5 * z]) + rng.normal(size=(n, 2))
    b0 = np.array([[1.0, 0.5], [-0.3, 2.0]])
    b1 = np.array([[0.4, 0.0], [0.0, -0.6]])
    channel = b0[None] + b1[None] * x[:, :, None]
    drift = np.hstack([np.sin(x) + 2.0 * z, x**2 - 1.5 * z])  # confounded through z
    y = np.einsum("nij,nj->ni", channel, u) + drift + 0.1 * rng.normal(size=(n, 2))

    est = LinearDML(
        model_y=RandomForestRegressor(min_samples_leaf=20, random_state=0),
        model_t=RandomForestRegressor(min_samples_leaf=20, random_state=0),
        cv=3,
        random_state=0,
    )
    est.fit(y, u, X=x, W=z)
    at = np.array([[-1.0], [0.0], [1.0]])
    theta = est.const_marginal_effect(at)
    truth = b0[None] + b1[None] * at[:, :, None]
    print("matrix: theta shape", theta.shape)
    print("matrix: max |theta - B(x)| over x in {-1, 0, 1}:", float(np.max(np.abs(theta - truth))))


def dynamic_effect() -> None:
    rng = np.random.default_rng(SEED)
    units, periods = 3000, 3
    a = np.array([[0.9, 0.2], [0.0, 0.7]])
    b = np.array([1.0, 0.5])
    d = np.array([0.8, -0.4])
    c = np.array([1.0, 1.0])
    k = np.array([-0.5, 0.0])

    x = rng.normal(size=(units, 2))
    actions, controls, outcomes = [], [], []
    for _ in range(periods):
        z = rng.normal(size=units)
        u = x @ k + z + 0.7 * rng.normal(size=units)
        x_next = x @ a.T + np.outer(u, b) + np.outer(z, d) + 0.3 * rng.normal(size=(units, 2))
        actions.append(u)
        controls.append(np.column_stack([x, z]))
        outcomes.append(x_next @ c)
        x = x_next
    # long format, unit-major and period-minor: EconML expects each unit's rows contiguous
    t = np.stack(actions, 1).reshape(-1, 1)
    w = np.stack(controls, 1).reshape(units * periods, -1)
    y = np.stack(outcomes, 1).reshape(-1)
    groups = np.repeat(np.arange(units), periods)

    est = DynamicDML(model_y=LinearRegression(), model_t=LinearRegression(), cv=3, random_state=0)
    est.fit(y, t, X=None, W=w, groups=groups)
    theta = np.asarray(est.const_marginal_effect()).ravel()
    truth = np.array([c @ np.linalg.matrix_power(a, periods - 1 - p) @ b for p in range(periods)])
    lo, hi = est.const_marginal_effect_interval(alpha=0.05)
    covered = (np.ravel(lo) <= truth) & (truth <= np.ravel(hi))
    print("dynamic: theta by period", np.round(theta, 4), "impulse response", np.round(truth, 4))
    print("dynamic: max |theta - c' A^(m-1-t) B|:", float(np.max(np.abs(theta - truth))))
    print("dynamic: 95% interval covers the truth by period:", covered.tolist())


if __name__ == "__main__":
    matrix_effect()
    dynamic_effect()
