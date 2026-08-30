"""Symbolic extraction from a Kolmogorov-Arnold layer (chc.symbolic, proofs/symbolic_kan.v)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chc.residual import RBFKANLayer
from chc.symbolic import (
    extract_symbolic,
    extract_symbolic_edge,
    kan_edge,
    symbolic_extraction_certificate,
)


def test_edges_and_bias_reconstruct_the_layer_exactly() -> None:
    # The extraction rests on the layer being EXACTLY additive in its edges. If that decomposition
    # were an approximation, every per-edge formula below would be about the wrong object.
    layer = RBFKANLayer(3, 2, 8, 3.0, key=jax.random.PRNGKey(1))
    z = jnp.array([0.4, -1.7, 2.2])
    direct = layer(z)
    assembled = layer.bias + sum(
        jnp.stack([kan_edge(layer, j, i, z[i]) for j in range(2)]) for i in range(3)
    )
    assert np.allclose(np.asarray(direct), np.asarray(assembled), atol=1e-12)


def test_extraction_recovers_a_planted_closed_form() -> None:
    # A layer whose edge is exactly a known curve must extract back to that curve. Planting the
    # coefficients directly (rather than fitting) isolates the extractor from the fit.
    layer = RBFKANLayer(1, 1, 24, 3.0, key=jax.random.PRNGKey(0))
    zs = np.linspace(-3.0, 3.0, 400)
    grid = np.linspace(-3.0, 3.0, 24)
    inv_h = 23 / 6.0
    basis = np.exp(-(((zs[:, None] - grid[None, :]) * inv_h) ** 2))
    silu = zs / (1.0 + np.exp(-zs))
    target = 2.5 * np.tanh(zs) - 0.75
    coeff, *_ = np.linalg.lstsq(np.column_stack([basis, silu]), target, rcond=None)
    planted = eqx.tree_at(
        lambda m: (m.coeff, m.base_weight),
        layer,
        (jnp.asarray(coeff[:24]).reshape(1, 1, 24), jnp.asarray(coeff[24:]).reshape(1, 1)),
    )
    edge = extract_symbolic_edge(planted, 0, 0)
    assert edge.terms == ("tanh(z)",)
    assert abs(edge.coefficients[0] - 2.5) < 0.02
    assert abs(edge.offset - (-0.75)) < 0.02
    assert edge.r_squared > 0.999


def test_extraction_certificate_recovers_the_truth_and_prices_both_boundaries() -> None:
    # Result 46 (proofs/symbolic_kan.v). RBFKANLayer advertised "an extractable 1D curve" with no
    # extraction API; this checks that the extraction has a right answer AND that the two places it
    # stops working are measured rather than assumed away. Each block can fail on its own.
    curve = symbolic_extraction_certificate()
    # (1) The layer really can represent the separable target -- otherwise nothing below is about
    # extraction, it is about underfitting.
    assert curve.layer_r_squared > 0.999
    # (2) THE RECOVERY. The right terms, from a library of nine, with the right coefficients.
    assert curve.recovered_terms == (("z**2",), ("sin(z)",))
    assert curve.coefficient_error < 1e-3
    # (3) THE GAUGE. A single edge's intercept is a convention; only the total constant has content.
    # The residual lands ~1000x below the edges' own fit error precisely because the split cancels.
    assert curve.gauge_residual < 1e-4
    assert curve.gauge_residual < max(e.max_abs_error for e in curve.edges)
    # (4) IT MUST BE ABLE TO FAIL -- separability. On a genuine interaction the same pipeline does
    # not error, it silently returns the best additive fit, and that fit explains nothing.
    assert curve.interaction_r_squared < 0.05
    # The floor is PROVED (bilinear_floor_on_box: no additive model beats r^2 on [-r,r]^2), so this
    # asserts against a theorem rather than against a previously observed number.
    assert curve.interaction_sup_error >= curve.interaction_floor
    # (5) IT MUST BE ABLE TO FAIL -- extrapolation. Outside the grid the RBFs have decayed and the
    # layer degenerates; the extracted formula carries no grid, so it keeps holding. The direction
    # of this inequality is the claim that extraction buys something beyond interpretability.
    assert curve.symbolic_extrapolation_error < 0.05 * curve.layer_extrapolation_error
    assert curve.ok


def test_extracted_formula_extrapolates_where_the_layer_stops() -> None:
    # The sharp version of block (5): compare the two on ground truth far outside the grid, where
    # the layer has only its silu term left and is therefore asymptotically linear.
    curve = symbolic_extraction_certificate()
    assert curve.layer_extrapolation_error > 1.0  # the layer genuinely breaks down out there
    assert curve.symbolic_extrapolation_error < 1e-2
    assert curve.layer_extrapolation_error / curve.symbolic_extrapolation_error > 1e3


def test_every_edge_of_a_random_layer_extracts_without_error() -> None:
    # Robustness of the fitter itself: an untrained layer's edges are arbitrary smooth curves, and
    # the API must return a well-formed result for each rather than only for planted ones.
    layer = RBFKANLayer(3, 2, 10, 2.0, key=jax.random.PRNGKey(7))
    edges = extract_symbolic(layer, max_terms=3)
    assert len(edges) == 6
    for edge in edges:
        assert 1 <= len(edge.terms) <= 3
        assert len(edge.coefficients) == len(edge.terms)
        assert np.isfinite(edge.r_squared)
        assert edge.max_abs_error >= 0.0
        zs = np.linspace(-2.0, 2.0, 17)
        assert np.all(np.isfinite(edge(zs)))  # the returned object is callable and finite
