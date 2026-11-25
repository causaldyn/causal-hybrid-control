"""probe_design: the probe covariance, at a cap on its power, that leaves the re-solved LQR gain the
least expected regret, ``Phi(U) / T`` with ``Phi(U) = tr(Wn Gamma(U)^-1) / 2``.

Its numbers are validation/dynamic_probe_design.mac's. ``_phi`` below reads ``Phi`` the long way,
against the full inverse of ``Gamma`` and a weight from finite differences of the re-solved gain,
so the split ``probe_design`` minimises is checked against it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

from chc.experiment import dither_covariance, probe_design
from chc.regret import dlqr

# A plant whose second lever is weak and cheap: the gain leans on it, so its channel matters most.
A2 = np.array([[0.9, 0.2], [-0.1, 0.7]])
B2 = np.array([[1.0, 0.1], [0.3, 0.2]])
R2 = np.diag([1.0, 0.05])
C2 = np.array([[2.0, 1.0], [1.0, 1.0]])
Q2 = np.array([[1.0, -0.5], [-0.5, 2.0]])


def _weight(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray, noise: np.ndarray):
    """``2 J' (G (x) Sigma*) J`` over ``(A.ravel(), B.ravel())``, ``J`` by central differences."""
    n, m = b.shape
    theta = np.concatenate([a.ravel(), b.ravel()])
    k_star, p_star = dlqr(a, b, q, r)
    columns = []
    for j in range(theta.size):
        step = np.zeros(theta.size)
        step[j] = 1e-6
        ends = [
            dlqr((theta + s)[: n * n].reshape(n, n), (theta + s)[n * n :].reshape(n, m), q, r)[0]
            for s in (step, -step)
        ]
        columns.append(((ends[0] - ends[1]) / 2e-6).ravel())
    jacobian = np.array(columns)
    spread = solve_discrete_lyapunov(a - b @ k_star, noise)
    return 2.0 * jacobian @ np.kron(r + b.T @ p_star @ b, spread) @ jacobian.T, k_star


def _phi(a, b, noise, u, weight, gain) -> float:
    """``tr(W (noise (x) Gamma^-1)) / 2`` over ``[A B]`` by rows, ``Gamma`` inverted whole."""
    n, m = b.shape
    w, k = weight, gain
    sx = solve_discrete_lyapunov(a - b @ k, noise + b @ u @ b.T)
    gamma = np.block([[sx, -sx @ k.T], [-k @ sx, k @ sx @ k.T + u]])
    covariance = np.zeros_like(w)
    order = [
        i * n + j if j < n else n * n + i * m + (j - n) for i in range(n) for j in range(n + m)
    ]
    covariance[np.ix_(order, order)] = np.kron(noise, np.linalg.inv(gamma))
    return 0.5 * float(np.sum(w * covariance))


def test_the_scalar_plant_reads_maxima_s_phi() -> None:
    # validation/dynamic_probe_design.mac STEP 5: one lever, so the probe is the cap itself
    design = probe_design([[1.2]], [[0.5]], [[1.0]], [[1.0]], noise=[[1.0]], power=0.5, periods=40)
    assert design.method == "closed_form"
    assert design.gap == 0.0
    assert design.covariance[0, 0] == 0.5
    assert design.regret * 40 == pytest.approx(17.124638317676437346, rel=1e-10, abs=0.0)
    assert design.ratio == 1.0


def test_one_state_two_levers_reach_maxima_s_optimum() -> None:
    # validation/dynamic_probe_design.mac STEP 7: Newton in 40 digits on the KKT conditions
    r = [[1.0, 0.2], [0.2, 0.5]]
    design = probe_design([[1.1]], [[1.0, 0.4]], [[1.0]], r, noise=[[1.0]], power=0.5, periods=100)
    assert design.method == "numeric"
    np.testing.assert_allclose(
        design.covariance,
        [
            [0.17334471230122213973, -0.069729108633290548548],
            [-0.069729108633290548548, 0.32665528769877786027],
        ],
        rtol=0.0,
        atol=1e-8,
    )
    assert design.regret * 100 == pytest.approx(6.2217573451312867380, rel=1e-10, abs=0.0)
    assert design.white_regret * 100 == pytest.approx(6.9785812210885939849, rel=1e-12, abs=0.0)
    assert design.ratio == pytest.approx(0.89155046678109024898, rel=1e-10, abs=0.0)
    assert 0.0 <= design.gap <= 1e-10 * design.regret


@pytest.mark.parametrize("noise", [np.eye(2), np.array([[2.0, 0.6], [0.6, 0.5]])])
def test_a_weight_on_the_channel_alone_is_dither_covariance(noise: np.ndarray) -> None:
    # B's information in the fit of [A B] is the probe's alone, so the weight of a plan on B only
    # (as CausalPlan.decision_weight reads the one-step channel) leaves dither_covariance's problem
    rng = np.random.default_rng(5)
    root = rng.standard_normal((4, 4))
    weight = root @ root.T + 0.1 * np.eye(4)
    periods = 60
    traced = np.einsum("ijkl,ki->jl", weight.reshape(2, 2, 2, 2), noise)
    static = dither_covariance(np.eye(2), traced / (2 * periods))
    design = probe_design(
        A2,
        B2,
        np.eye(2),
        R2,
        noise=noise,
        power=float(np.trace(static.covariance)),
        periods=periods,
        weight=weight,
    )
    assert design.method == "closed_form"
    np.testing.assert_allclose(design.covariance, static.covariance, rtol=0.0, atol=1e-12)
    assert design.regret == pytest.approx(static.value / 2.0, rel=1e-12, abs=0.0)


def test_a_cap_on_a_weighted_power_is_the_cap_on_scaled_levers() -> None:
    # dither_covariance's two-lever example, value 2 sqrt(3 + sqrt 7), through C^1/2-scaled levers
    values, vectors = np.linalg.eigh(C2)
    c_root = (vectors * np.sqrt(values)) @ vectors.T
    c_inverse_root = np.linalg.inv(c_root)
    periods = 30
    static = dither_covariance(C2, Q2)
    design = probe_design(
        [[0.5]],
        np.array([[1.0, 0.3]]) @ c_inverse_root,
        [[1.0]],
        c_inverse_root @ c_inverse_root,
        noise=[[1.0]],
        power=float(np.trace(C2 @ static.covariance)),
        periods=periods,
        weight=c_root @ (2 * periods * Q2) @ c_root,
    )
    np.testing.assert_allclose(
        c_inverse_root @ design.covariance @ c_inverse_root, static.covariance, rtol=0.0, atol=1e-12
    )
    assert design.regret == pytest.approx(math.sqrt(3.0 + math.sqrt(7.0)), rel=1e-12, abs=0.0)


@pytest.mark.parametrize("seed", range(3))
def test_the_split_reads_what_the_whole_inverse_does(seed: int) -> None:
    rng = np.random.default_rng(40 + seed)
    a = 0.5 * rng.standard_normal((2, 2))
    b = rng.standard_normal((2, 3))
    r = np.diag(rng.uniform(0.1, 2.0, 3))
    noise = np.array([[1.0, 0.3], [0.3, 0.8]])
    design = probe_design(a, b, np.eye(2), r, noise=noise, power=2.0, periods=50)
    assert design.method == "numeric"
    assert np.trace(design.covariance) == pytest.approx(2.0, rel=1e-12, abs=0.0)
    assert np.linalg.eigvalsh(design.covariance)[0] > 0.0
    weight, gain = _weight(a, b, np.eye(2), r, noise)
    phi = _phi(a, b, noise, design.covariance, weight, gain)
    assert design.regret * 50 == pytest.approx(phi, rel=1e-7, abs=0.0)
    white = _phi(a, b, noise, 2.0 / 3.0 * np.eye(3), weight, gain)
    assert design.white_regret * 50 == pytest.approx(white, rel=1e-7, abs=0.0)
    assert design.ratio <= 1.0


@pytest.mark.parametrize("seed", range(3))
def test_no_probe_nearby_does_better(seed: int) -> None:
    rng = np.random.default_rng(70 + seed)
    a = 0.6 * rng.standard_normal((2, 2))
    b = rng.standard_normal((2, 2))
    design = probe_design(
        a, b, np.eye(2), np.diag([1.0, 0.3]), noise=np.eye(2), power=3.0, periods=20
    )
    best = design.regret * 20
    weight, gain = _weight(a, b, np.eye(2), np.diag([1.0, 0.3]), np.eye(2))
    u = design.covariance
    for _ in range(60):
        step = 0.05 * rng.standard_normal((2, 2))
        candidate = u + step + step.T
        values, vectors = np.linalg.eigh(candidate)
        candidate = (vectors * np.clip(values, 1e-9, None)) @ vectors.T
        candidate *= 3.0 / np.trace(candidate)
        phi = _phi(a, b, np.eye(2), candidate, weight, gain)
        assert phi >= best - design.gap * 20 - 1e-7 * best


@pytest.mark.parametrize(
    ("plant", "power", "channel_alone", "probe_term", "designed"),
    [
        ((A2, B2, R2), 0.1, 0.698, 0.690, 0.690),
        ((A2, B2, R2), 1.0, 0.758, 0.753, 0.753),
        ((A2, B2, R2), 10.0, 0.994, 1.004, 0.942),
        ((np.array([[1.05, 0.5], [0.0, 0.3]]), np.array([[1.0, 0.0], [0.5, 1.0]]), np.eye(2)),
         0.1, 1.044, 0.974, 0.974),
    ],
)  # fmt: skip
def test_the_whole_decision_against_the_channel_s_effect_alone(
    plant, power, channel_alone, probe_term, designed
) -> None:
    # Phi over the white probe's for: the static design on the weight's channel block alone, the
    # closed form of Phi's probe term alone, and the design. At a power well above the noise's the
    # state's excitation is worth designing for, and a design on the channel alone can lose to white
    a, b, r = plant
    weight, gain = _weight(a, b, np.eye(2), r, np.eye(2))
    order = [0, 1, 4, 5, 2, 3, 6, 7]  # (A.ravel(), B.ravel()) to [A B] by rows
    traced = np.einsum("ijkl,ki->jl", weight[np.ix_(order, order)].reshape(2, 4, 2, 4), np.eye(2))
    f = np.vstack([gain.T, np.eye(2)])
    white = _phi(a, b, np.eye(2), power / 2 * np.eye(2), weight, gain)
    ratios = []
    for block in (traced[2:, 2:], f.T @ traced @ f):
        values, vectors = np.linalg.eigh(block)
        root = (vectors * np.sqrt(values)) @ vectors.T
        ratios.append(_phi(a, b, np.eye(2), power * root / np.trace(root), weight, gain) / white)
    design = probe_design(a, b, np.eye(2), r, noise=np.eye(2), power=power, periods=100)
    assert ratios == pytest.approx([channel_alone, probe_term], abs=5e-4)
    assert design.ratio == pytest.approx(designed, abs=5e-4)
    assert design.ratio <= min(ratios) + 1e-9


def test_the_cost_is_what_the_probe_adds_to_the_average_cost() -> None:
    gain = dlqr(A2, B2, np.eye(2), R2)[0] * 0.8  # a stabilising gain off the optimum
    design = probe_design(A2, B2, np.eye(2), R2, noise=np.eye(2), power=1.0, periods=50, gain=gain)
    np.testing.assert_array_equal(design.gain, gain)

    def average_cost(u: np.ndarray) -> float:
        sx = solve_discrete_lyapunov(A2 - B2 @ gain, np.eye(2) + B2 @ u @ B2.T)
        return float(np.trace(sx) + np.trace(R2 @ (gain @ sx @ gain.T + u)))

    base = average_cost(np.zeros((2, 2)))
    assert design.cost == pytest.approx(average_cost(design.covariance) - base, rel=1e-10, abs=0.0)
    assert design.white_cost == pytest.approx(
        average_cost(0.5 * np.eye(2)) - base, rel=1e-10, abs=0.0
    )
    weight = _weight(A2, B2, np.eye(2), R2, np.eye(2))[0]
    phi = _phi(A2, B2, np.eye(2), design.covariance, weight, gain)
    assert design.regret * 50 == pytest.approx(phi, rel=1e-7, abs=0.0)


def _realised_regret(
    u: np.ndarray, gain: np.ndarray, periods: int, rng: np.random.Generator, reps: int
) -> np.ndarray:
    """The re-solved gain's exact regret on A2, B2 after ``periods`` of the probe, per fit."""
    n, m = B2.shape
    closed = A2 - B2 @ gain
    x = (
        rng.standard_normal((reps, n))
        @ np.linalg.cholesky(solve_discrete_lyapunov(closed, np.eye(n) + B2 @ u @ B2.T)).T
    )
    probe = np.linalg.cholesky(u)
    moments = np.zeros((reps, n + m, n + m))
    cross = np.zeros((reps, n + m, n))
    for _ in range(periods):
        action = -x @ gain.T + rng.standard_normal((reps, m)) @ probe.T
        z = np.concatenate([x, action], axis=1)
        x = x @ A2.T + action @ B2.T + rng.standard_normal((reps, n))
        moments += z[:, :, None] * z[:, None, :]
        cross += z[:, :, None] * x[:, None, :]
    fits = np.linalg.solve(moments, cross)
    k_star, p_star = dlqr(A2, B2, np.eye(2), R2)
    curvature = R2 + B2.T @ p_star @ B2
    regrets = np.empty(reps)
    for i, fit in enumerate(fits):
        k_hat = dlqr(fit[:n].T, fit[n:].T, np.eye(2), R2)[0]
        loop = A2 - B2 @ k_hat
        assert np.max(np.abs(np.linalg.eigvals(loop))) < 1.0
        dk = k_hat - k_star
        regrets[i] = float(np.trace(solve_discrete_lyapunov(loop.T, dk.T @ curvature @ dk)))
    return regrets


def test_the_designed_probe_beats_white_by_the_ratio_it_predicts() -> None:
    periods, reps = 2000, 400
    design = probe_design(A2, B2, np.eye(2), R2, noise=np.eye(2), power=1.0, periods=periods)
    assert design.ratio == pytest.approx(0.7530, abs=5e-4)
    assert design.covariance[1, 1] == pytest.approx(0.8267, abs=5e-4)
    assert (design.cost, design.white_cost) == pytest.approx((0.5315, 1.3850), abs=5e-4)
    # each arm reads the same noise and the same standard normals for its probe
    designed = _realised_regret(
        design.covariance, design.gain, periods, np.random.default_rng(20261125), reps
    )
    white = _realised_regret(
        0.5 * np.eye(2), design.gain, periods, np.random.default_rng(20261125), reps
    )
    gain = white - designed
    assert gain.mean() > 3.0 * gain.std(ddof=1) / math.sqrt(reps)
    ratio = designed.mean() / white.mean()
    spread = np.cov(designed, white)
    error = (
        math.sqrt((spread[0, 0] + ratio**2 * spread[1, 1] - 2.0 * ratio * spread[0, 1]) / reps)
        / white.mean()
    )
    assert abs(ratio - design.ratio) <= 3.0 * error
    for realised, predicted in ((designed, design.regret), (white, design.white_regret)):
        standard_error = realised.std(ddof=1) / math.sqrt(reps)
        assert abs(realised.mean() - predicted) <= 3.0 * standard_error


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gain": [[0.0, 0.0], [0.0, 0.0]], "a": [[1.2, 0.0], [0.0, 0.5]]}, "not stable"),
        ({"weight": np.diag([1.0, 1.0, -0.5, 1.0])}, "positive semidefinite"),
        ({"weight": np.zeros((8, 8))}, "weight is zero"),
        ({"weight": np.eye(3)}, "weight must have shape"),
        ({"power": 0.0}, "power must be positive"),
        ({"power": -1.0}, "power must be positive"),
        ({"power": math.inf}, "power must be positive"),
        ({"periods": 5}, "periods must exceed"),
        ({"a": [[1.5, 0.0], [0.0, 0.5]], "b": [[0.0, 0.0], [1.0, 0.0]]}, "no stabilising LQR gain"),
        ({"r": np.diag([1.0, 0.0])}, "r must be positive definite"),
        ({"noise": np.diag([1.0, 0.0])}, "noise must be positive definite"),
        ({"q": np.diag([1.0, -1.0])}, "q must be positive semidefinite"),
        ({"gain": np.eye(3)}, "gain must have shape"),
        ({"a": np.eye(3)}, "a must have shape"),
        ({"b": [[np.nan, 0.0], [0.0, 1.0]]}, "non-finite"),
    ],
)
def test_refusals(kwargs: dict, message: str) -> None:
    arguments = {"a": A2, "b": B2, "q": np.eye(2), "r": R2}
    options = {"noise": np.eye(2), "power": 1.0, "periods": 50}
    for name in ("a", "b", "q", "r"):
        arguments[name] = kwargs.pop(name, arguments[name])
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        probe_design(*arguments.values(), **options)
