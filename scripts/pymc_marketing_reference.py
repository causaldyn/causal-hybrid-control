"""PyMC-Marketing's transforms against chc.response's, mapped: the check behind its docstring.

    kernels  ``geometric_adstock``, ``delayed_adstock`` and the CDF ``weibull_adstock``, raw and
             normalised, against ``GeometricAdstock``, ``DelayedAdstock`` and ``WeibullAdstock`` on
             one gamma-distributed spend series; and the Weibull kernel's impulse response, which
             says how many weights ``l_max`` buys
    curves   ``logistic_saturation``, ``tanh_saturation``,
             ``inverse_scaled_logistic_saturation``, ``michaelis_menten``, ``hill_function``,
             ``hill_saturation_sigmoid`` and ``root_saturation`` against the family each maps onto,
             on a grid of spend; and ``hill_function`` against the Hill in 50-digit decimals

Measured 2026-09-30 with PyMC-Marketing 1.2.0 on Python 3.12: every kernel, raw and normalised, to
1.6e-16 of its largest value, and every curve to 2.2e-16 but ``hill_function``, to 2.3e-9. PyTensor
makes a Python float it holds exactly a float32 constant, so ``hill_function(x, 2.5, 150.0)`` takes
``kappa ** slope`` in single precision: 2.1e-9 from the Hill in 50 digits at worst, 1.2e-16 given
float64 constants, where chc's Hill is 1.1e-16. The CDF ``weibull_adstock`` with ``l_max = 12``
carries 13 weights: ``WeibullAdstock`` of length 13 matches it to 1.1e-16, of length 12 misses by
3.2e-2.

Each row prints the largest difference over the largest value. PyMC-Marketing is not a dependency
of chc (``chc.response`` says why), so this runs in a throwaway environment:

    uv run --no-project --python 3.12 --with pymc-marketing==1.2.0 --with-editable . \\
        python scripts/pymc_marketing_reference.py
"""

from __future__ import annotations

from decimal import Decimal, getcontext

import jax
import numpy as np
import pytensor.tensor as pt
import pytensor.xtensor as ptx
from pymc_marketing.mmm import transformers as pymc

from chc import response

jax.config.update("jax_enable_x64", True)
SEED = 20260930


def as_x(values: np.ndarray, dim: str):
    return ptx.as_xtensor(pt.as_tensor_variable(values), dims=(dim,))


def row(name: str, ours, theirs) -> None:
    ours, theirs = np.asarray(ours), np.asarray(theirs)
    gap = np.max(np.abs(ours - theirs)) / np.max(np.abs(theirs))
    print(f"| {name} | {gap:.1e} |")


def kernels() -> None:
    spend = np.random.default_rng(SEED).gamma(2.0, 50.0, size=40)
    series = as_x(spend, "time")
    print("| kernel, on a spend series | largest difference / largest value |\n|---|---|")
    for normalize in (False, True):
        tag = "normalised" if normalize else "raw"
        row(
            f"geometric, alpha 0.6, l_max 8, {tag}",
            response.GeometricAdstock(0.6, length=8, normalized=normalize)(spend),
            pymc.geometric_adstock(series, 0.6, 8, normalize=normalize, dim="time").eval(),
        )
        row(
            f"delayed, alpha 0.7, theta 2.5, l_max 10, {tag}",
            response.DelayedAdstock(0.7, 2.5, length=10, normalized=normalize)(spend),
            pymc.delayed_adstock(
                series, 0.7, 2.5, l_max=10, normalize=normalize, dim="time"
            ).eval(),
        )
        row(
            f"Weibull CDF, k 1.5, lam 3, l_max 12 as length 13, {tag}",
            response.WeibullAdstock(1.5, 3.0, length=13, normalized=normalize)(spend),
            pymc.weibull_adstock(
                series, 3.0, 1.5, l_max=12, type="CDF", normalize=normalize, dim="time"
            ).eval(),
        )
    impulse = np.zeros(20)
    impulse[0] = 1.0
    weights = pymc.weibull_adstock(
        as_x(impulse, "time"), 30.0, 0.8, l_max=12, type="CDF", normalize=False, dim="time"
    ).eval()
    print(f"\nWeibull CDF with l_max 12 carries {int(np.count_nonzero(weights))} weights:", end=" ")
    for length in (12, 13):
        ours = response.WeibullAdstock(0.8, 30.0, length=length, normalized=False)(impulse)
        print(
            f"length {length} differs by {np.max(np.abs(np.asarray(ours) - weights)):.1e}", end="; "
        )
    print()


def curves() -> None:
    grid = np.linspace(0.0, 400.0, 81)
    spend = as_x(grid, "spend")
    eps = float(np.log(3.0))  # inverse_scaled_logistic_saturation's default
    amplitude = 2.0 * float(jax.nn.sigmoid(0.03 * 150.0))
    print("\n| PyMC-Marketing | as | largest difference / largest value |\n|---|---|---|")
    for name, family, ours, theirs in [
        (
            "logistic_saturation, lam 0.01",
            "Tanh(K = 2 / lam)",
            response.Tanh(2.0 / 0.01)(grid),
            pymc.logistic_saturation(spend, 0.01),
        ),
        (
            "tanh_saturation, b 2, c 60",
            "b Tanh(K = b c)",
            2.0 * response.Tanh(2.0 * 60.0)(grid),
            pymc.tanh_saturation(spend, 2.0, 60.0),
        ),
        (
            "inverse_scaled_logistic_saturation, lam 100",
            "Tanh(K = 2 lam / eps)",
            response.Tanh(2.0 * 100.0 / eps)(grid),
            pymc.inverse_scaled_logistic_saturation(spend, 100.0),
        ),
        (
            "michaelis_menten, alpha 3, lam 120",
            "alpha MichaelisMenten(K = lam)",
            3.0 * response.MichaelisMenten(120.0)(grid),
            pymc.michaelis_menten(spend, 3.0, 120.0),
        ),
        (
            "hill_function, slope 2.5, kappa 150",
            "Hill(K = kappa, slope)",
            response.Hill(150.0, 2.5)(grid),
            pymc.hill_function(spend, 2.5, 150.0),
        ),
        (
            "hill_saturation_sigmoid, sigma 2, beta 0.03, lam 150",
            "sigma s(beta lam) Logistic(K = lam, s = beta lam)",
            amplitude * response.Logistic(150.0, 0.03 * 150.0)(grid),
            pymc.hill_saturation_sigmoid(spend, 2.0, 0.03, 150.0),
        ),
        (
            "root_saturation, alpha 0.5",
            "Power(K = 1, alpha)",
            response.Power(1.0, 0.5)(grid),
            pymc.root_saturation(spend, 0.5),
        ),
    ]:
        theirs = theirs.eval()
        gap = np.max(np.abs(np.asarray(ours) - theirs)) / np.max(np.abs(theirs))
        print(f"| {name} | {family} | {gap:.1e} |")

    getcontext().prec = 50

    def exact(x: float) -> float:
        z = Decimal(repr(x)) / Decimal(150)
        power = z * z * z.sqrt()
        return float(power / (1 + power))

    truth = np.array([exact(float(x)) for x in grid])
    ours = np.asarray(response.Hill(150.0, 2.5)(grid))
    theirs = pymc.hill_function(spend, 2.5, 150.0).eval()
    # PyTensor makes a Python float it holds exactly a constant of the narrowest float type
    slope, kappa = pt.constant(2.5, dtype="float64"), pt.constant(150.0, dtype="float64")
    wide = pymc.hill_function(spend, slope, kappa).eval()
    print(
        f"\nhill_function against 50 digits: {np.max(np.abs(theirs - truth)):.1e} at worst with "
        f"slope and kappa as Python floats, which PyTensor makes "
        f"{pt.as_tensor_variable(150.0).dtype}; {np.max(np.abs(wide - truth)):.1e} as float64 "
        f"constants; Hill {np.max(np.abs(ours - truth)):.1e}"
    )


if __name__ == "__main__":
    kernels()
    curves()
