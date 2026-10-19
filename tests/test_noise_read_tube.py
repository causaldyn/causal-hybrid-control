"""The tube carries the error of the noise it reads off the log (ADR 0105).

The tube's noise is read off the log: its covariance ``S``, its VAR(1) coefficient ``Phi``, and its
mean, which the fitted drift absorbs. Taken for the plant's own, each one log's read moves that
log's tube, and the tube holds less than 0.95 over logs: the radius's error lowers the share more
where it is too narrow than it raises it where too wide, and the plant's noise misses the mean the
drift absorbed at every step. ``chc.decision._noise_read`` reads the spread of ``S`` and ``Phi``
over the log's units by a jackknife and the covariance of the mean robust to a noise that persists
within units; ``chc.uncertainty._read_covariances`` reads the noise's part at its studentised
scale and adds the mean's error as a constant offset (``validation/noise_read_tube.mac``).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

from chc.decision import _autoregression, _noise_read
from chc.uncertainty import _noise_radius, _noise_tube, _read_covariances, _studentised

_DT, _HORIZON, _SLOPE = 0.1, 15, -0.6


def _labels(units: int, periods: int) -> np.ndarray:
    """``_transitions``' group, period and unit of each row."""
    return np.stack(
        [
            np.zeros(units * periods),
            np.tile(np.arange(periods), units),
            np.repeat(np.arange(units), periods),
        ],
        1,
    )


def _log(rng: np.random.Generator, units: int, periods: int, phi: float) -> np.ndarray:
    """``(units, periods)``: a stationary AR(``phi``) noise of unit variance in each unit."""
    w = np.empty((units, periods))
    w[:, 0] = rng.standard_normal(units)
    for t in range(1, periods):
        w[:, t] = phi * w[:, t - 1] + math.sqrt(1.0 - phi * phi) * rng.standard_normal(units)
    return w


def _coverage(phi: float, logs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per step, the share of the plant's paths each log's tube holds, averaged over ``logs``
    logs of 50 units of 12 periods, read plug-in and with the read's error carried. Each log's
    noise is read about its own mean, as the drift's intercept reads it. The plant's gap
    ``e_k = sum_j M^(k-1-j) (w_j - wbar)`` is Gaussian of mean ``-D_k wbar`` and of the AR's own
    variance, so each log's share is exact and only the logs are drawn."""
    units, periods = 50, 12
    z = _DT * _SLOPE
    step = 1.0 + z + z * z / 2.0 + z**3 / 6.0 + z**4 / 24.0
    gains = [[step ** (k - 1 - j) for j in range(k)] for k in range(1, _HORIZON + 1)]
    spread = np.sqrt(
        [
            sum(a * b * phi ** abs(i - j) for i, a in enumerate(g) for j, b in enumerate(g))
            for g in gains
        ]
    )
    offset = np.array([sum(g) for g in gains])
    rate = np.full((_HORIZON, 1, 1), _SLOPE)
    labels = _labels(units, periods)
    rng = np.random.default_rng(seed)
    plug, read = [], []
    for _ in range(logs):
        w = _log(rng, units, periods, phi)
        mean = float(w.mean())
        error = (w - mean).reshape(-1, 1)
        covariance = error.T @ error / error.shape[0]
        carry = _autoregression(error, covariance, labels[:, 2], labels[:, 1])
        noise_read = _noise_read(SimpleNamespace(defect=error / _DT), _DT, labels)
        for shares, given in ((plug, None), (read, noise_read)):
            radius = _noise_tube(rate, covariance, carry, _DT, _HORIZON, read=given)[1:]
            centre = -offset * mean
            shares.append(
                norm.cdf((radius - centre) / spread) - norm.cdf((-radius - centre) / spread)
            )
    return np.mean(plug, axis=0), np.mean(read, axis=0)


def test_the_tube_holds_its_level_over_logs_when_it_carries_its_read() -> None:
    """Over 300 logs of 50 units of 12 periods of an AR(0.8) noise, the tube that takes the read
    for the plant's noise holds 0.947 at the first step and 0.942 at the fifteenth; carrying the
    read's error it holds 0.9498 to 0.9499. The share's standard error over the logs is under
    0.0015, and the floor is three of them under 0.95. On white noise the read moves the tube's
    share from 0.946 to 0.950 at the fifteenth step, the mean's error alone, and to no more than
    0.951 at any step."""
    plug, read = _coverage(0.8, 300, 0)
    assert read.min() >= 0.95 - 3.0 * 0.0015, np.round(read, 4)
    assert plug.min() < 0.945, np.round(plug, 4)
    plug, read = _coverage(0.0, 300, 0)
    assert read.min() >= 0.95 - 3.0 * 0.0007, np.round(read, 4)
    assert read.max() <= 0.951, np.round(read, 4)
    assert plug.min() < 0.947, np.round(plug, 4)


def test_the_read_is_the_jackknife_and_the_clustered_mean() -> None:
    """``S`` and ``Phi`` read again with each group of units left out, and the mean's covariance
    ``G / (G - 1) sum_g s_g s_g' / N^2`` over the units' sums; on 120 units, 50 groups."""
    rng = np.random.default_rng(3)
    units, periods = 120, 6
    error = np.stack([_log(rng, units, periods, 0.5), _log(rng, units, periods, 0.0)], 2)
    error = (error - error.mean(axis=(0, 1))).reshape(-1, 2)
    labels = _labels(units, periods)
    read = _noise_read(SimpleNamespace(defect=error / _DT), _DT, labels)
    assert read is not None
    assert read.covariances.shape == read.carries.shape == (50, 2, 2)
    sums = error.reshape(units, periods, 2).sum(axis=1)
    np.testing.assert_allclose(
        read.level, units / (units - 1) * sums.T @ sums / error.shape[0] ** 2, rtol=1e-12
    )
    kept = np.repeat(np.arange(units) % 50 != 7, periods)
    left = error[kept]
    np.testing.assert_allclose(read.covariances[7], left.T @ left / left.shape[0], rtol=1e-12)
    np.testing.assert_allclose(
        read.carries[7],
        _autoregression(left, left.T @ left / left.shape[0], labels[kept, 2], labels[kept, 1]),
        rtol=1e-12,
        atol=1e-15,
    )
    one = _noise_read(SimpleNamespace(defect=error[:periods] / _DT), _DT, labels[:periods])
    assert one is None


def test_the_read_studentises_the_noise_and_adds_the_mean_s_error_at_each_step() -> None:
    """At the first step the gap is one step's noise and the mean's error, ``s_1^2 S + V``, read at
    its exact radius on the matrix path and on a norm bound alike. On one state the norm bound is
    the step itself, and its radius carries both as the matrix path does, to rounding, wider than
    the plug-in tube's at every step. A read whose replicates agree studentises by 1."""
    rng = np.random.default_rng(5)
    units, periods = 40, 10
    error = _log(rng, units, periods, 0.6).reshape(-1, 1)
    error = error - error.mean()
    labels = _labels(units, periods)
    covariance = error.T @ error / error.shape[0]
    carry = _autoregression(error, covariance, labels[:, 2], labels[:, 1])
    read = _noise_read(SimpleNamespace(defect=error / _DT), _DT, labels)
    assert read is not None
    rate = np.full((_HORIZON, 1, 1), _SLOPE)
    covariances = _read_covariances(rate, covariance, carry, _DT, _HORIZON, read)
    scale = _studentised(
        np.array(
            [
                [c[0, 0] for c in _read_covariances(rate, s, f, _DT, _HORIZON, None)[1:]]
                for s, f in zip(read.covariances, read.carries, strict=True)
            ]
        )
    )
    np.testing.assert_allclose(covariances[1], scale[0] ** 2 * covariance + read.level, rtol=1e-12)
    matrix = _noise_tube(rate, covariance, carry, _DT, _HORIZON, read=read)
    bound = _noise_tube(np.asarray(_SLOPE), covariance, carry, _DT, _HORIZON, read=read)
    assert matrix[1] == pytest.approx(_noise_radius(covariances[1]), rel=1e-12, abs=0.0)
    assert bound[1] == pytest.approx(matrix[1], rel=1e-12, abs=0.0)
    np.testing.assert_allclose(bound, matrix, rtol=1e-9, atol=0.0)
    plug = _noise_tube(rate, covariance, carry, _DT, _HORIZON)
    assert np.all(matrix[1:] > plug[1:])
    assert _studentised(np.ones((8, 3))).tolist() == [1.0, 1.0, 1.0]


def test_a_size_the_replicates_scatter_over_is_read_at_cauchy_s_scale_at_most() -> None:
    """A norm bound over a pendulum's 40 steps read replicates 1e112 to 1e119 apart, and the
    studentised scale of ``nu = 2 / v`` near 0 passed the largest float. ``nu`` is at least 1, so
    the scale is at most Cauchy's, ``tan(0.475 pi) / z(0.975) = 6.48``; a size read as no number
    gives 1."""
    sizes = np.array([[1.0, 1e112, np.inf], [2.0, 1e119, np.inf], [1.5, 1e115, np.inf]])
    scales = _studentised(sizes)
    assert np.all(np.isfinite(scales))
    cauchy = math.tan(0.475 * math.pi) / float(norm.ppf(0.975))
    assert scales[1] == pytest.approx(cauchy, rel=1e-9, abs=0.0)
    assert scales[2] == 1.0
    assert 1.0 < scales[0] < scales[1]
