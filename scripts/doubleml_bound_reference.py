"""DoubleML's omitted-variable bound against chc.dynamics_id.omitted_confounder_bound, on one log.

DoubleML 0.11.4 fits the same partially linear model from chc's own cross-fitted nuisance
predictions, passed as external predictions, so the two read the same residuals and differ only
where they build the bound. Each row prints the largest relative difference in the bounds, the
robustness value, the confidence bounds and their robustness value:

    fit        chc's bound on its own fit
    elements   the same with the estimate's influence replaced by DoubleML's, ``-alpha e`` for the
               representer ``alpha`` and the moment's residual ``e``
    sign       the same with ``+alpha e``, chc's sign, which moves the confidence bounds by the
               cross term of the two influences alone

Measured 2026-10-07 with DoubleML 0.11.4 on Python 3.13, 4000 transitions, ``cf_y = cf_d = 0.1``,
on the cross-fitting folds chc draws since 0.13.0: the bounds agree to 3.8e-10 in every row, the
ridge on chc's moment, scaled by the actions' mean square, 1.88 on this log; unscaled, it left
2e-10. The confidence bounds agree to 1.6e-6 under DoubleML's influence, to 2.1e-5 under chc's
sign and to 7.8e-5 under chc's own influence, which is robust to unequal noise and carries the
nuisances' error and the degrees of freedom. chc reads them against ``t(3999)``'s quantile and
DoubleML against the normal's (ADR 0062); on the normal's they agree to 3.8e-10 under DoubleML's
influence. The robustness values differ by 1.7e-6, and by 5e-7 at the confidence bound under
DoubleML's influence (1.6e-7 on the normal's), because DoubleML finds them by bounded
minimisation of the squared distance at scipy's default tolerance: the lower bound at chc's
robustness value is 5.6e-15 from the null, at DoubleML's 2.9e-5. On the folds before, measured
2026-09-30, those read 3.1e-5, 1e-4, 1.4e-6, 1.8e-7, 1.7e-14 and 2.4e-5.

DoubleML is not a dependency of chc, so this runs in a throwaway environment:

    uv run --no-project --python 3.13 --with doubleml==0.11.4 --with-editable . \\
        python scripts/doubleml_bound_reference.py
"""

from __future__ import annotations

import dataclasses

import doubleml
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from chc.dynamics_id import _cross_fit_residuals, fit_causal_residual, omitted_confounder_bound

jax.config.update("jax_enable_x64", True)
SEED = 20260930
N, DT = 4000, 0.05
SHARES = {"cf_y": 0.1, "cf_d": 0.1}


def _known(t: float, x: jax.Array, u: jax.Array) -> jax.Array:
    return jnp.zeros_like(x)


def _log() -> dict[str, jax.Array]:
    """One state, one lever, an observed confounder and a latent one, Euler-stepped."""
    rng = np.random.default_rng(SEED)
    x, observed, latent = rng.standard_normal(N), rng.standard_normal(N), rng.standard_normal(N)
    u = 0.8 * observed + latent + 0.5 * rng.standard_normal(N)
    after = x + DT * (-0.5 * x + u + 0.7 * observed + latent) + 0.01 * rng.standard_normal(N)
    return {
        "x": jnp.asarray(x)[:, None],
        "u": jnp.asarray(u)[:, None],
        "x_next": jnp.asarray(after)[:, None],
        "observed": jnp.asarray(observed)[:, None],
    }


def _doubleml(rate: np.ndarray, data: dict[str, jax.Array], predictions: dict) -> dict:
    frame = pd.DataFrame(
        {
            "y": rate,
            "d": np.asarray(data["u"])[:, 0],
            "x": np.asarray(data["x"])[:, 0],
            "observed": np.asarray(data["observed"])[:, 0],
        }
    )
    model = doubleml.DoubleMLPLR(
        doubleml.DoubleMLData(frame, "y", "d", ["x", "observed"]),
        LinearRegression(),
        LinearRegression(),
        n_folds=2,
    )
    model.fit(external_predictions={"d": predictions})
    model.sensitivity_analysis(level=0.95, rho=1.0, **SHARES)
    params = model.sensitivity_params
    return {
        "lower": float(params["theta"]["lower"][0]),
        "upper": float(params["theta"]["upper"][0]),
        "robustness_value": float(params["rv"][0]),
        "ci_lower": float(params["ci"]["lower"][0]),
        "ci_upper": float(params["ci"]["upper"][0]),
        "robustness_value_ci": float(params["rva"][0]),
    }


def _gap(ours, theirs: dict, fields: tuple[str, ...]) -> float:
    return max(abs(getattr(ours, name) - theirs[name]) / abs(theirs[name]) for name in fields)


def main() -> None:
    data = _log()
    fit = fit_causal_residual(
        _known, data, DT, adjust_for=("observed",), channel_degree=0, influence=True
    )
    rate = (data["x_next"] - data["x"]) / DT
    _, _, rate_hat, action_hat = _cross_fit_residuals(
        rate,
        data["u"],
        jnp.concatenate([data["x"], data["observed"]], axis=1),
        degree=2,
        folds=2,
        ridge=1e-6,
        seed=0,
    )
    theirs = _doubleml(
        np.asarray(rate)[:, 0],
        data,
        {"ml_l": np.asarray(rate_hat), "ml_m": np.asarray(action_hat)},
    )
    alpha = np.asarray(fit.representer)[:, :, 0]
    residual = np.asarray(fit.moment_residual)
    fields = (("lower", "upper"), ("robustness_value",), ("ci_lower", "ci_upper"))
    fields += (("robustness_value_ci",),)
    print(f"{'':9} {'bounds':>8} {'rv':>8} {'ci':>8} {'rva':>8}")
    for name, sign in (("fit", None), ("elements", -1.0), ("sign", 1.0)):
        replaced = (
            fit
            if sign is None
            else dataclasses.replace(
                fit, influence=jnp.asarray(sign * alpha * residual / N)[:, :, None]
            )
        )
        ours = omitted_confounder_bound(replaced, np.ones((1, 1, 1)), **SHARES)
        print(f"{name:9}" + "".join(f" {_gap(ours, theirs, group):8.2g}" for group in fields))
    for name, share in (("chc", ours.robustness_value), ("DoubleML", theirs["robustness_value"])):
        at = omitted_confounder_bound(fit, np.ones((1, 1, 1)), cf_y=share, cf_d=share)
        print(f"the lower bound at {name}'s robustness value: {at.lower:.2g}")


if __name__ == "__main__":
    main()
