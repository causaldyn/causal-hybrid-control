"""Calibrated pessimism: the ensemble flags out-of-support states, split conformal hits nominal
coverage, and penalising that uncertainty avoids the model exploitation a greedy controller hits.
"""

import json
import math
import os
import subprocess
import sys
from fractions import Fraction
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from jax import Array

from chc import SplitConformal
from chc.benchmark import ModelUncertaintyTask
from chc.dynamics import HybridDynamics, LinearDynamics
from chc.integrate import rk4_step
from chc.residual import MLPResidual, ZeroResidual
from chc.uncertainty import (
    EnsembleResidual,
    NestedCVaRPenalty,
    _conformal_rank,
    _member_next_states,
    _predictive_std,
    _top_tail_mean,
    cvar_upper,
    fit_ensemble,
    nested_risk_certificate,
    sharded_ensemble_certificate,
)

DT = 0.1


class _CubicDrag(eqx.Module):
    a: Array
    b: Array
    drag: float

    def __call__(self, t: float | Array, x: Array, u: Array) -> Array:
        return self.a @ x + self.b @ (u - self.drag * u**3)


def _fit_ensemble_on_support(seed: int = 0) -> tuple[HybridDynamics, _CubicDrag]:
    a = jnp.array([[0.0, 1.0], [-1.0, -0.2]])
    b = jnp.array([[0.0], [1.0]])
    known = HybridDynamics(known=LinearDynamics(a_matrix=a, b_matrix=b), residual=ZeroResidual(2))
    plant = _CubicDrag(a=a, b=b, drag=0.15)
    k_x, k_u = jax.random.split(jax.random.key(seed))
    xs = jax.random.normal(k_x, (1500, 2))
    us = 0.4 * jax.random.normal(k_u, (1500, 1))  # narrow action support
    x_next = jax.vmap(lambda x, u: rk4_step(plant, 0.0, x, u, DT))(xs, us)
    model, _ = fit_ensemble(known, {"x": xs, "u": us, "x_next": x_next}, DT, n_members=4, steps=700)
    return model, plant


def _sample(plant: _CubicDrag, key: Array, n: int, scale: float, shift: float = 0.0) -> dict:
    k_x, k_u = jax.random.split(key)
    xs = jax.random.normal(k_x, (n, 2))
    us = shift + scale * jax.random.normal(k_u, (n, 1))
    x_next = jax.vmap(lambda x, u: rk4_step(plant, 0.0, x, u, DT))(xs, us)
    return {"x": xs, "u": us, "x_next": x_next}


def test_ensemble_disagreement_flags_out_of_support() -> None:
    model, plant = _fit_ensemble_on_support()
    ensemble = model.residual

    def mean_disagreement(data: dict) -> float:
        per_point = jax.vmap(lambda x, u: ensemble.disagreement(0.0, x, u))(data["x"], data["u"])
        return float(jnp.mean(per_point))

    in_region = mean_disagreement(_sample(plant, jax.random.key(5), 400, 0.4))
    out_region = mean_disagreement(_sample(plant, jax.random.key(6), 400, 1.0, shift=4.0))
    assert out_region > 5.0 * in_region  # the ensemble knows where it is extrapolating


def test_split_conformal_hits_nominal_coverage() -> None:
    model, plant = _fit_ensemble_on_support()
    for alpha in (0.1, 0.2):
        calib = _sample(plant, jax.random.key(7), 800, 0.4)
        test = _sample(plant, jax.random.key(8), 800, 0.4)
        conformal = SplitConformal.calibrate(model, calib, DT, alpha=alpha)
        assert abs(conformal.coverage(test) - (1.0 - alpha)) < 0.05  # finite-sample coverage holds


def _untrained_model() -> HybridDynamics:
    return HybridDynamics(LinearDynamics(jnp.zeros((2, 2)), jnp.zeros((2, 1))), _toy_ensemble())


def _noise(n: int, seed: int) -> dict[str, Array]:
    x, u, x_next = jax.random.split(jax.random.key(seed), 3)
    return {
        "x": jax.random.normal(x, (n, 2)),
        "u": jax.random.normal(u, (n, 1)),
        "x_next": jax.random.normal(x_next, (n, 2)),
    }


def test_split_conformal_refuses_fewer_scores_than_its_coverage_needs() -> None:
    """Five scores at alpha 0.05 ask for the sixth smallest, which five do not hold; reading the
    largest instead covered 5/6 of exchangeable data, not 0.95. Nineteen hold the nineteenth."""
    with pytest.raises(ValueError, match="the 6-th smallest score, so it needs at least 19"):
        SplitConformal.calibrate(_untrained_model(), _noise(5, 0), DT, alpha=0.05)
    assert math.isfinite(
        float(SplitConformal.calibrate(_untrained_model(), _noise(19, 0), DT, alpha=0.05).q_hat)
    )


def test_split_conformal_reads_the_kth_smallest_score() -> None:
    """Ten scores at alpha 0.2: ``k = ceil(11 * 0.8) = 9``, the ninth smallest itself, not an
    interpolation between two."""
    model, data = _untrained_model(), _noise(10, 1)
    conformal = SplitConformal.calibrate(model, data, DT, alpha=0.2)
    ensemble = cast(EnsembleResidual, model.residual)

    def score(x: Array, u: Array, x_next: Array) -> Array:
        mean = jnp.mean(_member_next_states(model.known, ensemble, x, u, DT), axis=0)
        sigma = _predictive_std(model.known, ensemble, x, u, DT) + 1e-6
        return jnp.linalg.norm(x_next - mean) / sigma

    scores = np.sort(np.asarray(jax.vmap(score)(data["x"], data["u"], data["x_next"])))
    assert float(conformal.q_hat) == scores[8]


@given(n=st.integers(0, 100_000), alpha=st.floats(1e-6, 1.0 - 1e-6))
def test_the_conformal_rank_is_the_least_whose_coverage_reaches_the_level(n, alpha) -> None:
    """The ``k``-th smallest of ``n`` exchangeable scores covers a new one with probability
    ``k / (n + 1)`` (no ties): the rank is the least ``k`` that reaches ``1 - alpha``, on alpha as
    written."""
    rank = _conformal_rank(n, alpha)
    level = 1 - Fraction(str(alpha))
    assert Fraction(rank, n + 1) >= level > Fraction(rank - 1, n + 1)


@pytest.mark.parametrize(("n", "alpha", "rank"), [(99, 0.45, 55), (9, 0.3, 7), (19, 0.05, 19)])
def test_the_conformal_rank_is_the_formula_s_integer(n, alpha, rank) -> None:
    """``100 * (1 - 0.45)`` is 55.00000000000001 in floating point, and ``10 * (1 - 0.3)`` exceeds
    7 on 0.3's binary value; on the decimal written both are whole."""
    assert _conformal_rank(n, alpha) == rank


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"alpha": 0.0}, ValueError, r"alpha=0.0 is the share left uncovered"),
        ({"alpha": 1.0}, ValueError, r"alpha=1.0 is the share left uncovered"),
        ({"alpha": float("nan")}, ValueError, r"alpha=nan"),
        ({"eps": 0.0}, ValueError, r"eps=0.0 must be positive"),
        ({"eps": -1e-6}, ValueError, r"eps=-1e-06 must be positive"),
        ({"residual": ZeroResidual(2)}, TypeError, "the residual is a ZeroResidual"),
        ({"rows": 0}, ValueError, "0 calibration transitions cannot certify"),
    ],
)
def test_split_conformal_refuses_what_carries_no_guarantee(change, error, match) -> None:
    model = _untrained_model()
    if "residual" in change:
        model = HybridDynamics(model.known, change["residual"])
    with pytest.raises(error, match=match):
        SplitConformal.calibrate(
            model,
            _noise(change.get("rows", 50), 2),
            DT,
            alpha=change.get("alpha", 0.1),
            eps=change.get("eps", 1e-6),
        )


def test_calibrated_pessimism_avoids_model_exploitation() -> None:
    task = ModelUncertaintyTask(n_members=4, fit_steps=700, n_data=1500, inner_steps=250)
    results = {r.controller: r for r in task.run(seed_data=0)}
    assert results["oracle"].regret == 0.0
    assert results["calibrated"].regret < 0.1 * results["greedy"].regret  # penalising U avoids it
    assert results["greedy"].ood_rate > 0.3  # greedy pushes into the high-uncertainty region
    assert results["calibrated"].ood_rate < 0.1  # calibrated stays where the model is trustworthy


def _toy_ensemble(n_members: int = 6) -> EnsembleResidual:
    return EnsembleResidual(
        members=tuple(
            MLPResidual(state_dim=2, control_dim=1, out_dim=2, width=8, key=jax.random.key(i))
            for i in range(n_members)
        )
    )


def test_cvar_upper_matches_the_exact_numpy_tail_mean() -> None:
    values = jnp.asarray([0.3, -1.2, 2.5, 0.9, -0.4, 1.7, 0.1])
    for alpha in (0.2, 0.35, 0.5, 1.0):  # 0.2*7 = 1.4 exercises the fractional weight
        reference = _top_tail_mean(np.asarray(values, dtype=np.float64), alpha)
        assert float(cvar_upper(values, alpha)) == pytest.approx(reference, abs=1e-6)


def test_nesting_the_tail_per_step_never_scores_below_committing_to_one_scenario() -> None:
    xs = jax.random.normal(jax.random.key(10), (12, 2))
    us = jax.random.normal(jax.random.key(11), (12, 1))
    ensemble = _toy_ensemble()
    strict = nested_risk_certificate(xs, us, ensemble, alpha=0.2)
    assert strict.ok
    assert strict.gap > 0.0  # subadditivity is strict here: the worst members differ across steps
    assert strict.nested > strict.risk_neutral  # and both are above the risk-neutral aggregation
    # alpha = 1 is the mean, where the nested and static aggregations must coincide exactly
    neutral = nested_risk_certificate(xs, us, ensemble, alpha=1.0)
    assert neutral.gap == pytest.approx(0.0, abs=1e-5)


def test_the_nested_penalty_is_differentiable_at_the_solver_start() -> None:
    # pessimistic_control starts from us0 = 0; a NaN gradient there would kill the solve
    xs = jnp.zeros((5, 2))
    us = jnp.zeros((5, 1))
    grad = jax.grad(
        lambda u: NestedCVaRPenalty(ensemble=_toy_ensemble(), alpha=0.3).penalty_trajectory(xs, u)
    )(us)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_stacked_ensemble_reproduces_the_serial_recursion() -> None:
    certificate = sharded_ensemble_certificate(parity_steps=60)
    assert certificate.ok
    assert certificate.parity_ulp < 2000.0
    assert certificate.n_members % certificate.mesh_size == 0
    assert certificate.shard_devices == certificate.mesh_size


def test_ensemble_members_are_independently_seeded() -> None:
    # The stacked fit trains one program over a member axis; if the key derivation collapsed, the
    # members would coincide and the disagreement -- the whole point of the ensemble -- would be 0.
    model, _ = _fit_ensemble_on_support()
    ensemble = cast(EnsembleResidual, model.residual)
    outputs = ensemble.member_outputs(0.0, jnp.array([1.5, -1.0]), jnp.array([0.5]))
    assert float(jnp.var(outputs, axis=0).sum()) > 1e-8


def test_sharded_path_runs_across_a_multi_device_mesh() -> None:
    # XLA_FLAGS is read when the backend initialises, so the >=2-device path needs a fresh process.
    source = (
        "import json, jax; jax.config.update('jax_enable_x64', True);"
        "from chc import sharded_ensemble_certificate as c;"
        "r = c(n_members=8, parity_steps=40);"
        "print(json.dumps(r.__dict__))"
    )
    # The flag multiplies host devices, which a GPU default backend would never use.
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
    }
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, env=env, timeout=600
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["n_devices"] == 8
    assert report["mesh_size"] == 8
    assert report["shard_devices"] == 8  # the member axis really is spread over the mesh
    assert report["ok"]
