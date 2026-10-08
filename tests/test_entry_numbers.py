"""Every entry point that takes a caller's data as arrays reads it as a panel reads a column.

NumPy's cast to float64 read the text ``"0.5"`` as 0.5, a date as its count of days, a duration as
its count of seconds and a complex number as its real part, with a warning at most; a ``Decimal``
of ``1E+400`` as an infinity; and a masked array as what lies under its mask. JAX's cast refused
some of them, naming no argument, and read the others. Each entry point now reads a caller's
arrays by the rule a panel's columns are read by, refuses what is not numbers naming the argument,
and hands numbers on as the caller gave them.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re
from collections.abc import Callable, Mapping
from decimal import Decimal
from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chc import (
    allocation,
    causal,
    decision,
    did,
    discovery,
    dlm,
    dynamics_id,
    estimators,
    evaluation,
    frames,
    gate,
    independence,
    irf,
    koopman,
    lalonde,
    lift,
    marketplace,
    matching,
    metrics,
    network_causal,
    offpolicy,
    pathway,
    residual,
    response,
    scm,
    support,
    surrogate,
    switchback,
    toeplitz,
    train,
    uncertainty,
)
from chc.cost import QuadraticCost
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.graph import CausalGraph
from chc.panel import Panel
from chc.residual import ControlAffineResidual, MLPResidual, SpectralResidual
from chc.response import Channel, GeometricAdstock, Hill, Tanh
from chc.uncertainty import EnsembleResidual, SplitConformal

# --- what is not numbers -------------------------------------------------------------------------


def _text(values: Any) -> np.ndarray:
    return np.asarray(values).astype(str).astype(object)


def _fixed_width_text(values: Any) -> np.ndarray:
    return np.asarray(values).astype(str)


def _dates(values: Any) -> np.ndarray:
    shape = np.shape(values)
    return (np.datetime64("2024-01-01") + np.arange(math.prod(shape))).reshape(shape)


def _durations(values: Any) -> np.ndarray:
    return np.round(np.asarray(values, dtype=np.float64)).astype(np.int64).astype("timedelta64[s]")


def _complex(values: Any) -> np.ndarray:
    return np.asarray(values) + 0j


def _past_float64(values: Any) -> np.ndarray:
    cells = np.array(
        [Decimal(repr(float(value))) for value in np.asarray(values, dtype=np.float64).ravel()],
        dtype=object,
    )
    cells[-1] = Decimal("1E+400")
    return cells.reshape(np.shape(values))


def _masked(values: Any) -> np.ma.MaskedArray:
    mask = np.zeros(np.shape(values), dtype=bool)
    mask.reshape(-1)[-1] = True
    return np.ma.masked_array(np.asarray(values), mask=mask)


_NOT_READ = ", so it is not read as numbers$"
_NOT_REAL = ", not real numbers" + _NOT_READ
KINDS: dict[str, tuple[Callable[[Any], Any], str]] = {
    "text": (_text, "type str, not a real number" + _NOT_READ),
    "fixed-width-text": (_fixed_width_text, r"dtype <U\d+, which holds text" + _NOT_REAL),
    "dates": (_dates, r"dtype datetime64\[D\], which holds dates or times" + _NOT_REAL),
    "durations": (_durations, r"dtype timedelta64\[s\], which holds durations" + _NOT_REAL),
    "complex": (_complex, r"dtype complex(64|128), which holds complex numbers" + _NOT_REAL),
    "past-float64": (
        _past_float64,
        r"Decimal\('1E\+400'\) at .+: a finite number past float64's range, where it is an "
        r"infinity" + _NOT_READ,
    ),
    "masked": (_masked, r"masked at .+ \(1 of \d+ cells\): a masked cell is a missing value"),
}


# --- the entry points ----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Entry:
    """An entry point called on the arguments ``build`` makes. ``paths`` maps the name a refusal
    gives each argument that carries a caller's data to where it lies among them; a key that
    starts with a dot is a dataclass field."""

    name: str
    build: Callable[[], dict[str, Any]]
    call: Callable[[dict[str, Any]], object]
    paths: Mapping[str, tuple[Any, ...]]
    error: type[ValueError] = ValueError


def _replaced(container: Any, path: tuple[Any, ...], convert: Callable[[Any], Any]) -> Any:
    """``container`` with the value at ``path`` converted, every container on the way copied."""
    key, rest = path[0], path[1:]
    if isinstance(key, str) and key.startswith("."):
        value = getattr(container, key[1:])
        new = _replaced(value, rest, convert) if rest else convert(value)
        return dataclasses.replace(container, **{key[1:]: new})
    value = container[key]
    new = _replaced(value, rest, convert) if rest else convert(value)
    if isinstance(container, list):
        return [new if index == key else item for index, item in enumerate(container)]
    return {**container, key: new}


def _named(*names: str) -> dict[str, tuple[str]]:
    return {name: (name,) for name in names}


def _entries(label: str, *names: str) -> dict[str, tuple[str, str]]:
    return {f"{label}[{name!r}]": (label, name) for name in names}


def _columns(*names: str) -> dict[str, tuple[str, str]]:
    return {f"column {name!r}": ("data", name) for name in names}


@functools.cache
def _did() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    return {
        "outcomes": rng.standard_normal((12, 6)) + np.arange(6),
        "group": np.array([2, 2, 3, 3, 4, 4, -1, -1, -1, -1, 3, 2]),
    }


@functools.cache
def _scm() -> dict[str, Any]:
    return {"outcomes": np.random.default_rng(1).standard_normal((6, 10)).cumsum(axis=1)}


@functools.cache
def _xyz() -> dict[str, Any]:
    rng = np.random.default_rng(2)
    z = rng.standard_normal((80, 2))
    return {
        "x": z @ [1.0, 0.5] + rng.standard_normal(80),
        "y": z @ [0.3, -0.2] + rng.standard_normal(80),
        "z": z,
    }


@functools.cache
def _signal() -> dict[str, Any]:
    t = np.linspace(0.0, 5.0, 60)
    return {
        "signal": 1.0 - np.exp(-t) * np.cos(3.0 * t),
        "first_col": np.array([2.0, 0.5, 0.1]),
        "first_row": np.array([2.0, 0.3, 0.2]),
        "rhs": np.array([1.0, 2.0, 3.0]),
        "generators": (np.array([0.6, -0.1, 0.05]), np.array([0.02, -0.1, 0.55])),
        "snapshots": np.random.default_rng(3).standard_normal((3, 8)),
    }


@functools.cache
def _switchback() -> dict[str, Any]:
    rng = np.random.default_rng(4)
    lever = (np.cumsum(rng.random((2, 40)) < 0.3, axis=1) % 2).astype(np.float64)
    outcome = np.zeros((2, 41))
    noise = rng.standard_normal((2, 40))
    for t in range(40):
        outcome[:, t + 1] = 0.5 * outcome[:, t] + lever[:, t] + noise[:, t]
    return {"lever": lever, "outcome": outcome}


@functools.cache
def _pilot() -> dict[str, Any]:
    rng = np.random.default_rng(5)
    lever = (np.cumsum(rng.random(500) < 0.5) % 2).astype(np.float64)
    outcome = np.zeros(501)
    noise = rng.standard_normal(500)
    for t in range(500):
        outcome[t + 1] = 0.8 * outcome[t] + 2.0 * lever[t] + noise[t]
    plan = switchback.design_switchback(
        (switchback.Horizon(2),),
        switchback.PersistencePrior(0.8, 0.8, 1.0, 2.0),
        2000,
        trust_state_model=False,
    )
    return {"plan": plan, "lever": lever, "outcome": outcome}


@functools.cache
def _dlm() -> dict[str, Any]:
    rng = np.random.default_rng(6)
    x = rng.gamma(2.0, 1.0, (30, 2))
    model = dlm.DynamicLinearModel(
        (dlm.Regression(2, 0.99),), dlm.Prior(np.zeros(2), 10.0 * np.eye(2), 1.0, 6.0)
    )
    y = 2.0 + x.sum(axis=1) + rng.standard_normal(30)
    return {
        "model": model,
        "y": y,
        "x": x,
        "fit": dlm.forward_filter(model, y, x),
        "ahead": rng.gamma(2.0, 1.0, (3, 2)),
    }


@functools.cache
def _geos() -> dict[str, Any]:
    rng = np.random.default_rng(7)
    x = rng.gamma(2.0, 1.0, (25, 3, 1))
    national = dlm.Prior(np.zeros(1), 100.0 * np.eye(1), 1.0, math.inf)
    regional = dlm.Prior(np.array([3.0, 0.0]), np.diag([100.0, 100.0]), 1.0, math.inf)
    model = dlm.GeoDLM(
        (dlm.Regression(1, 1.0),),
        (dlm.Polynomial(1, 1.0), dlm.Regression(1, 1.0)),
        3,
        dlm.stacked_prior(national, regional, 3),
    )
    return {"model": model, "y": 3.0 + 1.5 * x[:, :, 0] + rng.standard_normal((25, 3)), "x": x}


@functools.cache
def _spend() -> dict[str, Any]:
    rng = np.random.default_rng(8)
    return {
        "spend": rng.gamma(2.0, 50.0, 20),
        "outcome": rng.normal(100.0, 5.0, 8),
        "history": np.array([[100.0, 50.0], [80.0, 60.0], [90.0, 40.0]]),
        "geo_history": rng.gamma(2.0, 30.0, (3, 2, 2)),
    }


@functools.cache
def _channels() -> tuple[Channel, ...]:
    return (
        Channel(GeometricAdstock(0.3, length=4, normalized=True), Tanh(250.0), 1100.0),
        Channel(GeometricAdstock(0.5, length=4, normalized=True), Tanh(120.0), 900.0),
    )


@functools.cache
def _channel() -> Channel:
    return Channel(GeometricAdstock(0.5, length=4, normalized=True), Hill(100.0, 1.5), 500.0)


@functools.cache
def _logged_plant() -> dict[str, Any]:
    plant = evaluation.LinearGaussianPlant(
        np.array([[0.85, 0.15], [0.25, 0.75]]),
        np.array([[-0.6], [-0.2]]),
        np.zeros(2),
        np.diag([0.05, 0.03]),
    )
    logger = evaluation.AffinePolicy(np.array([[0.3, 0.2]]), np.array([0.1]), np.array([[0.25]]))
    rng = np.random.default_rng(9)
    x, u = np.zeros((401, 2)), np.zeros((400, 1))
    for t in range(400):
        u[t] = logger.gain @ x[t] + logger.offset + 0.5 * rng.standard_normal(1)
        x[t + 1] = (
            plant.a @ x[t] + plant.b @ u[t] + rng.multivariate_normal(np.zeros(2), plant.noise)
        )
    return {
        "logs": {"x": x, "u": u},
        "x": x[:-1],
        "u": u,
        "plant": plant,
        "logger": logger,
        "plan": evaluation.AffinePolicy(np.array([[0.4, 0.3]]), np.array([0.05]), np.zeros((1, 1))),
        "cost": QuadraticCost(
            Q=jnp.diag(jnp.array([2.0, 1.0])),
            R=jnp.array([[0.4]]),
            Qf=jnp.diag(jnp.array([2.0, 1.0])),
            x_target=jnp.zeros(2),
        ),
    }


@functools.cache
def _decisions() -> dict[str, Any]:
    rng = np.random.default_rng(10)
    dither = 0.1 * rng.standard_normal((30, 2))
    return {
        "action": 1.0 + dither,
        "propensity": np.full(30, 0.7),
        "saturated": np.zeros(30, dtype=bool),
        "dither": dither,
    }


@functools.cache
def _drift() -> dict[str, Any]:
    rng = np.random.default_rng(11)
    return {
        "log": gate.DecisionLog(**_decisions()),
        "residual": 0.1 * rng.standard_normal((30, 2)),
        "dither_scale": np.array([0.1, 0.1]),
        "radius": np.full((2, 2), 0.5),
        "residual_scale": np.array([0.1, 0.1]),
        "evalues": np.array([1.2, 0.8, 1.1]),
    }


@functools.cache
def _batch() -> dict[str, Any]:
    return {
        "reward": np.random.default_rng(12).random(20),
        "candidate": np.full(20, 0.5),
        "baseline": np.full(20, 0.4),
        "logged": np.full(20, 0.45),
        "drift": np.ones((20, 2)),
    }


@functools.cache
def _facade() -> dict[str, Any]:
    rng = np.random.default_rng(13)
    units, periods = 10, 6
    frame = {
        "unit": np.repeat(np.arange(units), periods),
        "time": np.tile(np.arange(periods), units),
        **{
            name: rng.standard_normal(units * periods)
            for name in ("supply", "wait", "incentive", "demand")
        },
    }
    return {
        "panel": Panel.from_frame(frame, unit="unit", time="time"),
        "value": np.array([1.0, 1.5, 2.0]),
        "forecast": np.array([0.0, 0.5, 1.0, 1.5]),
        "state": np.array([0.1, 0.2]),
        "x0": np.array([0.5]),
    }


@functools.cache
def _outcomes() -> dict[str, Any]:
    rng = np.random.default_rng(14)
    covariates = rng.standard_normal((200, 2))
    return {
        "outcomes": rng.standard_normal(40) + 0.3,
        "covariates": covariates,
        "treated": (covariates[:, 0] + rng.standard_normal(200) > 0).astype(np.float64),
    }


@functools.cache
def _ensemble() -> dict[str, Any]:
    rng = np.random.default_rng(15)
    x, u = rng.standard_normal((40, 2)), rng.standard_normal((40, 1))
    data = {"x": x, "u": u, "x_next": 0.9 * x + 0.1 * u}
    members = tuple(
        MLPResidual(state_dim=2, control_dim=1, out_dim=2, width=8, key=jax.random.key(i))
        for i in range(3)
    )
    model = HybridDynamics(
        LinearDynamics(jnp.zeros((2, 2)), jnp.zeros((2, 1))), EnsembleResidual(members=members)
    )
    conformal = SplitConformal.calibrate(model, data, 0.1, alpha=0.1)
    return {"data": data, "model": model, "conformal": conformal}


@functools.cache
def _transitions() -> dict[str, Any]:
    rng = np.random.default_rng(16)
    x, u = rng.standard_normal((100, 2)), rng.standard_normal((100, 1))
    return {"x": x, "u": u, "x_next": 0.9 * x + 0.1 * u}


@functools.cache
def _matching() -> dict[str, Any]:
    rng = np.random.default_rng(17)
    values = rng.normal(size=(4, 3))
    supply = rng.dirichlet(2.0 * np.ones(4))
    treatment = 0.5 * rng.normal(size=(4, 3))
    return {
        "cost": -np.vstack([values + treatment, values]),
        "supply": np.concatenate([0.5 * supply, 0.5 * supply]),
        "demand": rng.dirichlet(2.0 * np.ones(3)),
        "treated": np.concatenate([np.ones(4, dtype=bool), np.zeros(4, dtype=bool)]),
        "randomised": np.ones(8, dtype=bool),
    }


@functools.cache
def _series() -> dict[str, Any]:
    rng = np.random.default_rng(18)
    u, x = rng.standard_normal(120), np.zeros(120)
    for t in range(1, 120):
        x[t] = 0.6 * x[t - 1] + 0.8 * u[t - 1] + 0.2 * rng.standard_normal()
    return {
        "data": {"u": u, "x": x},
        "series": x,
        "irf": np.array([0.0, 0.8, 0.48, 0.29]),
        "target": np.ones(6),
    }


@functools.cache
def _trajectory() -> dict[str, Any]:
    rng = np.random.default_rng(19)
    controls, series = rng.standard_normal((150, 1)), np.zeros((150, 2))
    for t in range(1, 150):
        series[t, 0] = (
            0.5 * series[t - 1, 0] + 0.9 * controls[t - 1, 0] + 0.1 * rng.standard_normal()
        )
        series[t, 1] = 0.4 * series[t - 1, 1] + 0.7 * series[t - 1, 0] + 0.1 * rng.standard_normal()
    return {"series": series, "controls": controls}


@functools.cache
def _network() -> dict[str, Any]:
    sample = network_causal.ConfoundedNetworkSystem().sample(jax.random.key(0))
    rng = np.random.default_rng(20)
    return {
        "data": {name: np.asarray(column) for name, column in sample.items()},
        "series": rng.standard_normal((4, 5, 30)),
        "treatments": rng.standard_normal((4, 6, 12)),
        "outcomes": rng.standard_normal((4, 6, 12)),
    }


@functools.cache
def _linear() -> dict[str, Any]:
    sample = causal.ConfoundedLinearSystem().sample(400, jax.random.key(0))
    return {"data": {name: np.asarray(column) for name, column in sample.items()}}


@functools.cache
def _lalonde() -> dict[str, Any]:
    rng = np.random.default_rng(21)
    covariates = {name: 3.0 * rng.standard_normal(300) + 10.0 for name in ("age", "educ")}
    treatment = (rng.random(300) < 0.3).astype(np.float64)
    outcome = 1000.0 * treatment + 50.0 * covariates["age"] + 100.0 * rng.standard_normal(300)
    return {"data": lalonde.LalondeData(treatment, outcome, covariates, 1000.0)}


def _known(t: float, x: jax.Array, u: jax.Array) -> jax.Array:
    return jnp.zeros_like(x)


@functools.cache
def _confounded() -> dict[str, Any]:
    system = dynamics_id.ConfoundedControlAffineSystem(
        drift=jnp.array([[-0.5, 0.1], [0.0, -0.3]]),
        channel=jnp.array([[1.0], [0.5]]),
        confounder_to_rate=jnp.array([[2.0], [1.0]]),
        confounder_to_action=jnp.array([[-1.5]]),
    )
    sample = system.sample(400, jax.random.key(0), _known)
    drive = np.random.default_rng(22).standard_normal(401)
    data = {name: np.asarray(column) for name, column in sample.items()}
    return {"data": {**data, "d": drive[:-1], "d_next": drive[1:]}, "dt": system.dt}


def _affine_model() -> HybridDynamics:
    return HybridDynamics(
        known=_known,
        residual=ControlAffineResidual(drift=jnp.zeros((2, 3)), channel=jnp.zeros((2, 1, 3))),
    )


@functools.cache
def _rollouts() -> dict[str, Any]:
    rng = np.random.default_rng(23)
    return {
        "data": {
            "x0": rng.standard_normal((5, 2)),
            "us": rng.standard_normal((5, 4, 1)),
            "xs": rng.standard_normal((5, 5, 2)),
        }
    }


@functools.cache
def _policy_logs() -> dict[str, Any]:
    rng = np.random.default_rng(24)
    xs = rng.standard_normal((300, 2))
    us = xs @ np.array([[0.4], [-0.2]]) + 0.3 * rng.standard_normal((300, 1))
    return {
        "xs": xs,
        "us": us,
        "data": {"x": xs, "u": us, "r": -np.sum(xs**2, axis=1)},
        "behaviour": offpolicy.fit_behavior_policy(jnp.asarray(xs), jnp.asarray(us)),
    }


@functools.cache
def _market() -> dict[str, Any]:
    return {
        "logs": dict(marketplace.SharedStateMarket(seed=0).generate_logs(200, jax.random.key(1)))
    }


@functools.cache
def _fields() -> dict[str, Any]:
    rng = np.random.default_rng(25)
    xs, us = rng.standard_normal((40, 8)), rng.standard_normal((40, 8))
    return {"xs": xs, "us": us, "ys": 0.5 * xs + 0.2 * us}


@functools.cache
def _closed_loop() -> dict[str, Any]:
    actions = 20.0 + 2.0 * np.random.default_rng(26).standard_normal(300)
    return {
        "states": 17.0 - actions / 2.6,
        "actions": actions,
        "rates": 1.0 + 0.4 * actions + 0.1 * actions**2,
    }


ENTRIES = (
    Entry(
        "did.callaway_santanna",
        _did,
        lambda a: did.callaway_santanna(a["outcomes"], a["group"]),
        _named("outcomes", "group"),
    ),
    Entry(
        "did.twoway_fixed_effects_att",
        _did,
        lambda a: did.twoway_fixed_effects_att(a["outcomes"], a["group"]),
        _named("outcomes", "group"),
    ),
    Entry(
        "did.de_chaisemartin",
        _did,
        lambda a: did.de_chaisemartin(a["outcomes"], a["group"]),
        _named("outcomes", "group"),
    ),
    Entry(
        "scm.synthetic_control",
        _scm,
        lambda a: scm.synthetic_control(a["outcomes"], 0, 6),
        _named("outcomes"),
    ),
    Entry(
        "scm.augmented_synthetic_control",
        _scm,
        lambda a: scm.augmented_synthetic_control(a["outcomes"], 0, 6),
        _named("outcomes"),
    ),
    Entry(
        "independence.partial_corr_test",
        _xyz,
        lambda a: independence.partial_corr_test(a["x"], a["y"], a["z"]),
        _named("x", "y", "z"),
    ),
    Entry(
        "independence.gcm_test",
        _xyz,
        lambda a: independence.gcm_test(a["x"], a["y"], a["z"], draws=19),
        _named("x", "y", "z"),
    ),
    Entry(
        "metrics.overshoot",
        _signal,
        lambda a: metrics.overshoot(a["signal"], 1.0),
        _named("signal"),
    ),
    Entry(
        "metrics.settling_time",
        _signal,
        lambda a: metrics.settling_time(a["signal"], 1.0, 0.1),
        _named("signal"),
    ),
    Entry(
        "metrics.rise_time",
        _signal,
        lambda a: metrics.rise_time(a["signal"], 1.0, 0.1),
        _named("signal"),
    ),
    Entry(
        "metrics.steady_state_error",
        _signal,
        lambda a: metrics.steady_state_error(a["signal"], 1.0),
        _named("signal"),
    ),
    Entry(
        "toeplitz.sample_autocorrelation",
        _signal,
        lambda a: toeplitz.sample_autocorrelation(a["signal"], 3),
        {"x": ("signal",)},
    ),
    Entry(
        "toeplitz.solve_toeplitz",
        _signal,
        lambda a: toeplitz.solve_toeplitz(a["first_col"], a["first_row"], a["rhs"]),
        _named("rhs"),
    ),
    Entry(
        "toeplitz.gohberg_semencul_covariance",
        _signal,
        lambda a: toeplitz.gohberg_semencul_covariance(a["snapshots"], 2),
        _named("snapshots"),
    ),
    Entry(
        "toeplitz.gohberg_semencul_apply",
        _signal,
        lambda a: toeplitz.gohberg_semencul_apply(*a["generators"], a["rhs"]),
        {"v": ("rhs",)},
    ),
    Entry(
        "switchback.read_switchback",
        _switchback,
        lambda a: switchback.read_switchback(
            a["lever"], a["outcome"], switchback.Horizon(2), "state_aware"
        ),
        _named("lever", "outcome"),
    ),
    Entry(
        "switchback.randomisation_test",
        _switchback,
        lambda a: switchback.randomisation_test(
            a["lever"],
            a["outcome"],
            switchback.Horizon(2),
            "state_aware",
            switchback.MarkovDesign(0.3),
            draws=99,
        ),
        _named("lever", "outcome"),
    ),
    Entry(
        "switchback.restate_mde",
        _pilot,
        lambda a: switchback.restate_mde(
            a["plan"], switchback.Horizon(2), a["lever"], a["outcome"]
        ),
        _named("lever", "outcome"),
    ),
    Entry(
        "dlm.forward_filter",
        _dlm,
        lambda a: dlm.forward_filter(a["model"], a["y"], a["x"]),
        _named("y", "x"),
    ),
    Entry("dlm.forecast", _dlm, lambda a: dlm.forecast(a["fit"], 3, a["ahead"]), {"x": ("ahead",)}),
    Entry(
        "dlm.forward_filter_geos",
        _geos,
        lambda a: dlm.forward_filter_geos(a["model"], a["y"], a["x"]),
        _named("y", "x"),
    ),
    Entry(
        "dlm.fit_geo_spread",
        _geos,
        lambda a: dlm.fit_geo_spread(a["model"], a["y"], a["x"], [1], 8, 0),
        _named("y", "x"),
    ),
    Entry(
        "lift.GeoArm",
        _spend,
        lambda a: lift.GeoArm(a["spend"], a["outcome"]),
        _named("spend", "outcome"),
    ),
    Entry(
        "allocation.allocate",
        _spend,
        lambda a: allocation.allocate(
            _channels(), 300.0, 2, lower=np.zeros(2), upper=np.full(2, 300.0), history=a["history"]
        ),
        _named("history"),
    ),
    Entry(
        "allocation.allocate_geos",
        _spend,
        lambda a: allocation.allocate_geos(
            (_channels(), _channels()),
            600.0,
            2,
            lower=np.zeros((2, 2)),
            upper=np.full((2, 2), 300.0),
            history=a["geo_history"],
        ),
        {"history": ("geo_history",)},
    ),
    Entry(
        "evaluation.fit_logger",
        _logged_plant,
        lambda a: evaluation.fit_logger(a["x"], a["u"]),
        _named("x", "u"),
    ),
    Entry(
        "evaluation.evaluate_plan",
        _logged_plant,
        lambda a: evaluation.evaluate_plan(
            a["logs"], a["plan"], "mis", plant=a["plant"], cost=a["cost"], logger=a["logger"]
        ),
        _entries("logs", "x", "u"),
    ),
    Entry(
        "gate.DecisionLog",
        _decisions,
        lambda a: gate.DecisionLog(**a),
        _named("action", "propensity", "saturated", "dither"),
    ),
    Entry(
        "gate.ZoneBatch",
        _batch,
        lambda a: gate.ZoneBatch(**a),
        _named("reward", "candidate", "baseline", "logged", "drift"),
    ),
    Entry(
        "gate.channel_drift_evalues",
        _drift,
        lambda a: gate.channel_drift_evalues(
            a["log"],
            a["residual"],
            dither_scale=a["dither_scale"],
            radius=a["radius"],
            residual_scale=a["residual_scale"],
        ),
        _named("residual", "dither_scale", "radius", "residual_scale"),
    ),
    Entry(
        "gate.channel_move",
        _drift,
        lambda a: gate.channel_move(a["log"], a["residual"], dither_scale=a["dither_scale"]),
        _named("residual", "dither_scale"),
    ),
    Entry(
        "gate.DriftAlarm.update",
        _drift,
        lambda a: gate.DriftAlarm(100.0).update(a["evalues"]),
        _named("evalues"),
    ),
    Entry(
        "decision.Target",
        _facade,
        lambda a: decision.Target("supply", a["value"]),
        {"the value of target 'supply'": ("value",)},
        decision.DecisionError,
    ),
    Entry(
        "decision.Driver",
        _facade,
        lambda a: decision.Driver("demand", a["forecast"]),
        {"the forecast of driver 'demand'": ("forecast",)},
        decision.DecisionError,
    ),
    Entry(
        "decision.PrescribedPolicy.actions_at",
        _facade,
        lambda a: decision.PrescribedPolicy(
            ("incentive",), ("supply", "wait"), 0.1, np.zeros((3, 1))
        ).actions_at(0, a["state"]),
        _named("state"),
    ),
    Entry(
        "decision.prescribe",
        _facade,
        lambda a: decision.prescribe(
            a["panel"],
            levers=[decision.Lever("incentive", lo=-2.0, hi=2.0)],
            target=decision.Target("supply", value=1.0),
            horizon=3,
            adjustment=("demand",),
            x0=a["x0"],
        ),
        _named("x0"),
        decision.DecisionError,
    ),
    Entry(
        "uncertainty.msm_worst_case_mean",
        _outcomes,
        lambda a: uncertainty.msm_worst_case_mean(a["outcomes"], 2.0),
        _named("outcomes"),
    ),
    Entry(
        "uncertainty.confounding_robust_radius",
        _outcomes,
        lambda a: uncertainty.confounding_robust_radius(0.1, a["outcomes"], 2.0),
        _named("outcomes"),
    ),
    Entry(
        "uncertainty.negative_control_gamma",
        _outcomes,
        lambda a: uncertainty.negative_control_gamma(a["outcomes"]),
        _named("outcomes"),
    ),
    Entry(
        "uncertainty.benchmark_gamma",
        _outcomes,
        lambda a: uncertainty.benchmark_gamma(a["treated"], a["covariates"], 2.0),
        _named("treated", "covariates"),
    ),
    Entry(
        "uncertainty.fit_ensemble",
        _ensemble,
        lambda a: uncertainty.fit_ensemble(a["model"], a["data"], 0.1, n_members=2, steps=2),
        _entries("data", "x", "u", "x_next"),
    ),
    Entry(
        "uncertainty.SplitConformal.calibrate",
        _ensemble,
        lambda a: SplitConformal.calibrate(a["model"], a["data"], 0.1, alpha=0.1),
        _entries("data", "x", "u", "x_next"),
    ),
    Entry(
        "uncertainty.SplitConformal.coverage",
        _ensemble,
        lambda a: a["conformal"].coverage(a["data"]),
        _entries("data", "x", "u", "x_next"),
    ),
    Entry(
        "koopman.KoopmanModel.fit",
        _transitions,
        lambda a: koopman.KoopmanModel().fit(a["x"], a["u"], a["x_next"]),
        _named("x", "u", "x_next"),
    ),
    Entry(
        "surrogate.GradientBoostedDynamics.fit",
        _transitions,
        lambda a: surrogate.GradientBoostedDynamics().fit(a["x"], a["u"], a["x_next"]),
        _named("x", "u", "x_next"),
    ),
    Entry(
        "matching.shadow_price_effect",
        _matching,
        lambda a: matching.shadow_price_effect(
            a["cost"], a["supply"], a["demand"], a["treated"], eps=0.3, randomised=a["randomised"]
        ),
        _named("cost", "supply", "demand", "treated", "randomised"),
    ),
    Entry(
        "matching.shadow_price_interval",
        _matching,
        lambda a: matching.shadow_price_interval(
            a["cost"], a["supply"], a["demand"], a["treated"], randomised=a["randomised"]
        ),
        _named("cost", "supply", "demand", "treated", "randomised"),
    ),
    Entry("irf.innovations", _series, lambda a: irf.innovations(a["series"], 2), _named("series")),
    Entry(
        "irf.irf_control_sequence",
        _series,
        lambda a: irf.irf_control_sequence(a["irf"], a["target"]),
        _named("target"),
    ),
    Entry(
        "irf.local_projection_irf",
        _series,
        lambda a: irf.local_projection_irf(a["data"], 3),
        _entries("data", "u", "x"),
    ),
    Entry(
        "irf.delay_estimate",
        _series,
        lambda a: irf.delay_estimate(a["data"], 3, n_resamples=4),
        _entries("data", "u", "x"),
    ),
    Entry(
        "irf.structured_irf",
        _series,
        lambda a: irf.structured_irf(a["data"], 3, order=2),
        _entries("data", "u", "x"),
    ),
    Entry(
        "discovery.discover_lagged_parents",
        _trajectory,
        lambda a: discovery.discover_lagged_parents(a["series"], a["controls"]),
        _named("series", "controls"),
    ),
    Entry(
        "discovery.TigramiteDiscovery.discover",
        _trajectory,
        lambda a: discovery.TigramiteDiscovery().discover(a["series"], a["controls"]),
        _named("series", "controls"),
    ),
    Entry(
        "pathway.causal_pathway",
        _trajectory,
        lambda a: pathway.causal_pathway(a["series"], 1, a["controls"], horizon=3),
        _named("series", "controls"),
    ),
    Entry(
        "network_causal.within_ar1",
        _network,
        lambda a: network_causal.within_ar1(a["series"]),
        _named("series"),
    ),
    Entry(
        "network_causal.estimate_propagation",
        _network,
        lambda a: network_causal.estimate_propagation(
            a["treatments"], a["outcomes"], network_causal.cycle_shells(6, 2), 3, n_resamples=4
        ),
        _named("treatments", "outcomes"),
    ),
    Entry(
        "network_causal.estimate_network_effects",
        _network,
        lambda a: network_causal.estimate_network_effects(a["data"], folds=2),
        _entries("data", "x_next", "u", "e", "x", "z", "x_nb", "z_nb"),
    ),
    Entry(
        "network_causal.estimate_network_effects_gnn",
        _network,
        lambda a: network_causal.estimate_network_effects_gnn(
            a["data"], a["data"]["neighbours"], steps=2, hidden=4
        ),
        _entries("data", "x_next", "u", "e", "x", "z"),
    ),
    Entry(
        "causal.estimate_control_effect",
        _linear,
        lambda a: causal.estimate_control_effect(a["data"], adjust_for=("z",)),
        _entries("data", "x", "u", "x_next", "z"),
    ),
    Entry(
        "causal.estimate_effect_iv",
        _linear,
        lambda a: causal.estimate_effect_iv(a["data"]),
        _entries("data", "x", "u", "w", "x_next"),
    ),
    Entry(
        "causal.sensitivity_analysis",
        _linear,
        lambda a: causal.sensitivity_analysis(a["data"], adjust_for=("z",)),
        _entries("data", "x", "u", "x_next", "z"),
    ),
    Entry(
        "causal.estimate_effect_dml",
        _linear,
        lambda a: causal.estimate_effect_dml(a["data"]),
        _entries("data", "x_next", "u", "x", "z"),
    ),
    Entry(
        "causal.refute_effect",
        _linear,
        lambda a: causal.refute_effect(a["data"]),
        _entries("data", "x", "u", "x_next", "z"),
    ),
    Entry(
        "estimators.BackdoorOLS",
        _linear,
        lambda a: estimators.BackdoorOLS().estimate(a["data"]),
        _columns("u", "x_next", "x", "z"),
    ),
    Entry(
        "estimators.IV2SLS",
        _linear,
        lambda a: estimators.IV2SLS().estimate(a["data"]),
        _columns("u", "x_next", "x", "w"),
    ),
    Entry(
        "estimators.DoubleML",
        _linear,
        lambda a: estimators.DoubleML(folds=2).estimate(a["data"]),
        _columns("u", "x_next", "x", "z"),
    ),
    Entry(
        "estimators.RLearner",
        _linear,
        lambda a: estimators.RLearner().estimate(a["data"]),
        _columns("u", "x_next", "x", "z"),
    ),
    Entry(
        "estimators.EconMLDoubleML",
        _linear,
        lambda a: estimators.EconMLDoubleML().estimate(a["data"]),
        _columns("u", "x_next", "x", "z"),
    ),
    Entry(
        "estimators.DoWhyEstimator",
        _linear,
        lambda a: estimators.DoWhyEstimator().estimate(a["data"]),
        _columns("u", "x_next", "x", "z"),
    ),
    Entry(
        "lalonde.lalonde_ate",
        _lalonde,
        lambda a: lalonde.lalonde_ate(a["data"], estimators.BackdoorOLS()),
        {
            "the treatment": ("data", ".treatment"),
            "the outcome": ("data", ".outcome"),
            "the covariate 'age'": ("data", ".covariates", "age"),
        },
    ),
    Entry(
        "dynamics_id.fit_causal_residual",
        _confounded,
        lambda a: dynamics_id.fit_causal_residual(
            _known, a["data"], a["dt"], adjust_for=("z",), drivers=("d",)
        ),
        _entries("data", "x", "u", "x_next", "z", "d", "d_next"),
    ),
    Entry(
        "dynamics_id.fit_causal_residual-instrument",
        _confounded,
        lambda a: dynamics_id.fit_causal_residual(_known, a["data"], a["dt"], instrument="w"),
        _entries("data", "w"),
    ),
    Entry(
        "dynamics_id.closed_loop_gain_attribution",
        _closed_loop,
        lambda a: dynamics_id.closed_loop_gain_attribution(a["states"], a["actions"], a["rates"]),
        _named("states", "actions", "rates"),
    ),
    Entry(
        "train.fit_residual",
        _confounded,
        lambda a: train.fit_residual(
            _affine_model(), {k: a["data"][k] for k in ("x", "u", "x_next")}, a["dt"], steps=2
        ),
        _entries("data", "x", "u", "x_next"),
    ),
    Entry(
        "train.fit_residual_multistep",
        _rollouts,
        lambda a: train.fit_residual_multistep(_affine_model(), a["data"], 0.1, steps=2),
        _entries("data", "x0", "us", "xs"),
    ),
    Entry(
        "offpolicy.fit_behavior_policy",
        _policy_logs,
        lambda a: offpolicy.fit_behavior_policy(a["xs"], a["us"]),
        _named("xs", "us"),
    ),
    Entry(
        "offpolicy.off_policy_value",
        _policy_logs,
        lambda a: offpolicy.off_policy_value(a["data"], a["behaviour"], a["behaviour"]),
        _entries("data", "x", "u", "r"),
    ),
    Entry(
        "support.SupportModel.fit",
        _policy_logs,
        lambda a: support.SupportModel.fit(a["xs"], a["us"]),
        _named("xs", "us"),
    ),
    Entry(
        "marketplace.calibrate_predictive",
        _market,
        lambda a: marketplace.calibrate_predictive(a["logs"]),
        _entries("logs", "u", "y"),
    ),
    Entry(
        "marketplace.calibrate_naive_causal",
        _market,
        lambda a: marketplace.calibrate_naive_causal(a["logs"]),
        _entries("logs", "u", "y", "demand"),
    ),
    Entry(
        "marketplace.calibrate_shared_state",
        _market,
        lambda a: marketplace.calibrate_shared_state(a["logs"]),
        _entries("logs", "u", "y", "demand", "aggregate"),
    ),
    Entry(
        "residual.fit_spectral_residual",
        _fields,
        lambda a: residual.fit_spectral_residual(
            SpectralResidual(8, 8, key=jax.random.key(0)), a["xs"], a["us"], a["ys"]
        ),
        _named("xs", "us", "ys"),
    ),
    Entry(
        "response.contribution",
        _spend,
        lambda a: response.contribution(_channel(), a["spend"]),
        _named("spend"),
    ),
    Entry(
        "response.roi",
        _spend,
        lambda a: response.roi(_channel(), a["spend"], slice(5, 10)),
        _named("spend"),
    ),
    Entry(
        "response.marginal_roi",
        _spend,
        lambda a: response.marginal_roi(_channel(), a["spend"], slice(5, 10)),
        _named("spend"),
    ),
)


@pytest.mark.parametrize(
    ("entry", "label", "kind"),
    [
        pytest.param(entry, label, kind, id=f"{entry.name}-{label}-{kind}")
        for entry in ENTRIES
        for label in entry.paths
        for kind in KINDS
    ],
)
def test_an_entry_point_refuses_what_is_not_numbers_naming_the_argument(
    entry: Entry, label: str, kind: str
) -> None:
    convert, why = KINDS[kind]
    arguments = _replaced(entry.build(), entry.paths[label], convert)
    with pytest.raises(entry.error, match=rf"^{re.escape(label)} is .*{why}"):
        entry.call(arguments)


@pytest.mark.parametrize("field", ["action", "propensity", "dither"])
@pytest.mark.parametrize(
    ("convert", "why"),
    [
        (lambda value: np.asarray(value).astype(str).tolist(), r"dtype <U\d+, which holds text"),
        (lambda value: (np.asarray(value) + 0j).tolist(), "dtype complex128, which holds complex"),
        (
            lambda value: np.full(np.shape(value), Decimal("1E+400"), dtype=object).tolist(),
            "a finite number past float64's range",
        ),
    ],
    ids=["text", "complex", "past-float64"],
)
def test_a_stored_decision_that_is_not_numbers_is_refused_naming_its_field(
    field: str, convert: Callable[[Any], Any], why: str
) -> None:
    """A record of the text ``"0.7"`` read as 0.7, and a complex number as its real part."""
    records = gate.DecisionLog(**_decisions()).to_records()
    records[3] = {**records[3], field: convert(records[3][field])}
    with pytest.raises(ValueError, match=rf"^{field} is .*: {why}"):
        gate.DecisionLog.from_records(records)


def _group(label: object, dtype: Any = np.float64) -> np.ndarray:
    """``_did``'s groups, row 3's replaced by ``label``; never treated is -1, or 5 where the dtype
    is unsigned."""
    group = _did()["group"]
    if np.dtype(dtype).kind == "u":
        group = np.where(group < 0, 5, group)
    group = group.astype(dtype)
    group[3] = label
    return group


_NOT_GROUPS = {
    "a-half": (_group(2.5), "2.5", "no whole number"),
    "a-negative-half": (_group(-0.5), "-0.5", "no whole number"),
    "an-infinity": (_group(np.inf), "inf", "no whole number"),
    "a-fraction": (_group(Fraction(7, 2), object), "Fraction(7, 2)", "no whole number"),
    "a-nan": (_group(np.nan), "nan", "missing"),
    "a-decimal-nan": (_group(Decimal("NaN"), object), "Decimal('NaN')", "missing"),
    "a-float-past-int64": (_group(1e300), "1e+300", "past int64's range"),
    "an-int-past-int64": (_group(2**70, object), str(2**70), "past int64's range"),
    "a-uint64-past-int64": (_group(2**64 - 1, np.uint64), str(2**64 - 1), "past int64's range"),
}
_DID_ESTIMATES = {
    "callaway_santanna": did.callaway_santanna,
    "callaway_santanna_inference": did.callaway_santanna_inference,
    "twoway_fixed_effects_att": did.twoway_fixed_effects_att,
    "de_chaisemartin": did.de_chaisemartin,
}


@pytest.mark.parametrize("estimate", list(_DID_ESTIMATES.values()), ids=list(_DID_ESTIMATES))
@pytest.mark.parametrize(
    ("group", "shown", "what"), list(_NOT_GROUPS.values()), ids=list(_NOT_GROUPS)
)
def test_a_group_that_is_no_whole_number_or_is_missing_is_refused_naming_it(
    estimate: Callable[..., object], group: np.ndarray, shown: str, what: str
) -> None:
    """NumPy's cast to int64 truncated a unit's group: 2.5 read as 2 and -0.5 as 0, a first-period
    adopter; a nan, an infinity or 1e300 as -2**63; a ``Fraction`` of 7/2 as 3; and 2**64 - 1 as
    -1, never treated. An ``int`` of 2**70 failed with ``OverflowError``, naming nothing."""
    with pytest.raises(
        ValueError, match=rf"^group is {re.escape(shown)} at row 3, which is {what}: "
    ):
        estimate(_did()["outcomes"], group)


def _objects(group: np.ndarray) -> np.ndarray:
    held = (Decimal, Fraction, float)
    return np.array([held[i % 3](int(g)) for i, g in enumerate(group)], dtype=object)


@pytest.mark.parametrize(
    "estimate",
    [did.callaway_santanna, did.twoway_fixed_effects_att, did.de_chaisemartin],
    ids=["callaway_santanna", "twoway_fixed_effects_att", "de_chaisemartin"],
)
@pytest.mark.parametrize(
    "held",
    [lambda group: group.astype(np.float64), lambda group: group.astype(np.float32), _objects],
    ids=["float64", "float32", "objects"],
)
def test_a_group_of_whole_numbers_held_as_floats_reads_as_its_integers(
    estimate: Callable[..., object], held: Callable[[np.ndarray], np.ndarray]
) -> None:
    a = _did()
    assert _same(estimate(a["outcomes"], held(a["group"])), estimate(a["outcomes"], a["group"]))


@pytest.mark.parametrize(
    "value",
    [
        np.array([0.5, -1.0, 2.0]),
        np.array([0.1, 1e-45, 3.4e38], dtype=np.float32),
        np.array([1, -2, 3], dtype=np.int8),
        np.array([0, 1, 2**64 - 1], dtype=np.uint64),
        np.array([True, False, True]),
        np.array([1, 0.5, Decimal("0.25"), Fraction(1, 3), True, np.float32(2.5)], dtype=object),
        np.array([np.nan, np.inf, -np.inf]),
        jnp.array([0.5, -1.0], dtype=jnp.float32),
        jnp.array([0.5, -1.0], dtype=jnp.float64),
        jnp.array([True, False]),
        [0.5, -1.0, 2.0],
        2.5,
    ],
    ids=[
        "float64",
        "float32",
        "int8",
        "uint64",
        "bool",
        "objects",
        "non-finite",
        "jax-float32",
        "jax-float64",
        "jax-bool",
        "list",
        "scalar",
    ],
)
def test_numbers_are_handed_on_as_the_caller_gave_them(value: Any) -> None:
    """Nothing is cast on the way in: each entry point casts as it did, so numbers read as they
    did, bit for bit, a float32 array and a JAX array among them."""
    assert frames._real_numbers(value, "value") is value


def test_a_masked_array_that_masks_no_cell_is_handed_on_as_its_data() -> None:
    """A JAX cast refused a masked array outright, mask or none; a NumPy cast read its data."""
    values = np.ma.masked_array([0.5, 1.0, 2.0], mask=False)
    read = frames._real_numbers(values, "value")
    assert type(read) is np.ndarray
    assert np.shares_memory(read, values)


def test_a_traced_array_is_read_by_its_dtype_and_keeps_tracing() -> None:
    """Under ``jax.jit`` the values are unknown, so the dtype alone decides; a traced entry point
    traced on, and read a complex array as complex numbers through the fit."""
    data = {name: jnp.asarray(column) for name, column in _linear()["data"].items()}

    @jax.jit
    def effect(data: dict[str, jax.Array]) -> jax.Array:
        return causal.estimate_control_effect(data, adjust_for=("z",))

    eager = causal.estimate_control_effect(data, adjust_for=("z",))
    np.testing.assert_allclose(effect(data), eager, rtol=1e-12, atol=0.0)
    with pytest.raises(
        ValueError,
        match=r"^data\['u'\] has dtype complex128, which holds complex numbers, not real numbers, "
        r"so it is not read as numbers",
    ):
        effect({**data, "u": data["u"] + 0j})


def _leaves(value: object) -> list[np.ndarray]:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            leaf
            for item in dataclasses.fields(value)
            for leaf in _leaves(getattr(value, item.name))
        ]
    if isinstance(value, Mapping):
        return [leaf for key in value for leaf in _leaves(value[key])]
    if isinstance(value, tuple | list):
        return [leaf for item in value for leaf in _leaves(item)]
    if value is None or callable(value) or isinstance(value, str):
        return []
    return [np.asarray(value)]


def _same(left: object, right: object) -> bool:
    """Whether two results hold the same numbers, bit for bit, in the same dtypes."""
    pairs = list(zip(_leaves(left), _leaves(right), strict=True))
    return all(
        a.dtype == b.dtype
        and a.shape == b.shape
        and (a.tolist() == b.tolist() if a.dtype == object else a.tobytes() == b.tobytes())
        for a, b in pairs
    )


def _switchback_read(dtype: type) -> object:
    a = _switchback()
    return switchback.read_switchback(
        a["lever"].astype(dtype), a["outcome"], switchback.Horizon(2), "state_aware"
    )


def _gamma_benchmark(dtype: type) -> object:
    a = _outcomes()
    return uncertainty.benchmark_gamma(a["treated"].astype(dtype), a["covariates"], 2.0)


def _lalonde_effect(dtype: type) -> object:
    data = _lalonde()["data"]
    narrowed = dataclasses.replace(data, treatment=data.treatment.astype(dtype))
    return lalonde.lalonde_ate(narrowed, estimators.BackdoorOLS())


def _backdoor_effect(dtype: type) -> object:
    data = _linear()["data"]
    return estimators.BackdoorOLS().estimate({**data, "u": (data["u"] > 0).astype(dtype)}).effect


def _control_effect(dtype: type) -> object:
    data = {name: jnp.asarray(column) for name, column in _linear()["data"].items()}
    flag = jnp.asarray(data["u"] > 0, dtype=dtype)
    return causal.estimate_control_effect({**data, "u": flag}, adjust_for=("z",))


@pytest.mark.parametrize(
    "read",
    [_switchback_read, _gamma_benchmark, _lalonde_effect, _backdoor_effect, _control_effect],
    ids=["switchback", "gamma-benchmark", "lalonde", "estimator", "jax"],
)
def test_a_boolean_array_reads_as_its_zeros_and_ones(read: Callable[[type], object]) -> None:
    assert _same(read(bool), read(np.float64))


@pytest.mark.parametrize(
    "estimator",
    [
        estimators.BackdoorOLS(),
        estimators.IV2SLS(),
        estimators.DoubleML(folds=2),
        estimators.RLearner(),
    ],
    ids=["backdoor", "iv", "dml", "r-learner"],
)
def test_an_estimator_reads_only_the_columns_it_is_given(estimator: Any) -> None:
    """Every column was cast as it entered, so a label of text or a date that no argument named
    failed the cast, and the estimate with it."""
    data = _linear()["data"]
    labelled = {
        **data,
        "region": np.array(["north", "south"] * 200),
        "day": np.datetime64("2024-01-01") + np.arange(400),
    }
    assert estimator.estimate(labelled).effect == estimator.estimate(data).effect


def test_lalonde_reads_float32_data_as_its_float64_widening() -> None:
    """The estimate read a float32 caller's data in float32. NumPy sums a matrix's columns row by
    row, so their standardisation was rounded more with every row, and the scaled outcome too."""
    data = _lalonde()["data"]

    def cast(dtype: type) -> lalonde.LalondeData:
        return lalonde.LalondeData(
            data.treatment.astype(np.float32).astype(dtype),
            data.outcome.astype(np.float32).astype(dtype),
            {
                name: column.astype(np.float32).astype(dtype)
                for name, column in data.covariates.items()
            },
            data.experimental_ate,
        )

    estimator = estimators.BackdoorOLS()
    assert lalonde.lalonde_ate(cast(np.float32), estimator) == lalonde.lalonde_ate(
        cast(np.float64), estimator
    )


_POOL = [
    ("demand", "incentive"),
    ("demand", "supply"),
    ("incentive", "supply"),
    ("incentive", "wait"),
    ("supply", "wait"),
    ("holiday", "incentive"),
]


def _pool(holiday: type) -> Panel:
    """Zones of a driver pool under one incentive, logged by a policy that chases demand and pays
    more on a holiday."""
    rng = np.random.default_rng(27)
    names = ("unit", "time", "supply", "wait", "incentive", "demand", "holiday")
    rows: dict[str, list[float]] = {name: [] for name in names}
    for unit in range(60):
        supply, wait = rng.normal(0.0, 0.2, 2)
        for period in range(8):
            demand, flag = rng.normal(), rng.random() < 0.3
            incentive = 0.9 * demand + 0.6 * flag + rng.normal(0.0, 0.5)
            for name, value in zip(
                names, (unit, period, supply, wait, incentive, demand, flag), strict=True
            ):
                rows[name].append(value)
            supply, wait = (
                supply
                + 0.1 * (-0.6 * supply + 0.3 * wait + 0.8 * incentive + 1.5 * demand)
                + rng.normal(0.0, 0.01),
                wait + 0.1 * (0.25 * wait - 0.4 * incentive) + rng.normal(0.0, 0.01),
            )
    frame = {name: np.asarray(values) for name, values in rows.items()}
    frame["holiday"] = frame["holiday"].astype(holiday)
    return Panel.from_frame(frame, unit="unit", time="time", seed=0)


def test_the_logger_check_reads_a_boolean_parent_of_a_lever_as_its_zeros_and_ones() -> None:
    """The check read a column only of a NumPy number dtype, so it took a lever's parent of
    booleans, which the panel holds and reads as 0 and 1, for one it cannot read, and was not
    run."""

    def check(holiday: type) -> object:
        return decision.prescribe(
            _pool(holiday),
            levers=[decision.Lever("incentive", lo=-2.0, hi=2.0, unit_cost=0.05)],
            target=decision.Target("supply", value=1.0),
            constraints=[decision.Constraint("wait", hi=0.5)],
            adjustment=CausalGraph.from_edges(_POOL),
            horizon=2,
            dt=0.1,
            tolerance=0.5,
        ).logger_check

    flags = check(bool)
    assert flags is not None
    assert "holiday" in flags.given
    assert _same(flags, check(np.float64))
