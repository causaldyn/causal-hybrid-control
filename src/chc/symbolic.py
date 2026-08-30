"""Symbolic extraction from a Kolmogorov-Arnold layer: turn a learned edge into a formula.

:class:`~chc.residual.RBFKANLayer` advertises each input-output edge as "an extractable 1D curve".
This module is the extraction. An edge of that layer is exactly

    phi_ji(z) = sum_g coeff[j,i,g] * exp(-((z - grid_g) * inv_h)^2) + base_weight[j,i] * silu(z),

a genuine scalar function of one scalar, so it can be sampled and fitted against a small library of
closed forms. Two structural facts decide the API:

*Gauge.* The layer computes ``bias_j + sum_i phi_ji(z_i)``, which is invariant under
``phi_ji -> phi_ji + k_i``, ``bias_j -> bias_j - sum_i k_i``. Individual edges are therefore
identified only up to an additive constant; only the *sum* of the constants is pinned down. Each
extracted edge carries its own ``offset`` and the identity is checked, rather than the constant
being silently attributed to whichever edge the fit happened to put it in.

*Separability.* A single layer can represent only additively separable functions. On a target with a
genuine interaction the extraction does not fail loudly -- it returns the best additive
approximation, which for independent centred inputs is a poor one. The certificate carries that arm
so the number is visible instead of assumed away.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from chc.residual import RBFKANLayer

Vector = NDArray[np.float64]

# The candidate closed forms, in a fixed order so a fit is reproducible. Deliberately small: the
# selection below is EXHAUSTIVE over subsets, which is exact where a greedy or lasso path is not,
# and exhaustive is only affordable while the library stays this size.
LIBRARY: tuple[tuple[str, Callable[[Vector], Vector]], ...] = (
    ("z", lambda z: z),
    ("z**2", lambda z: z**2),
    ("z**3", lambda z: z**3),
    ("sin(z)", np.sin),
    ("cos(z)", np.cos),
    ("exp(z)", np.exp),
    ("tanh(z)", np.tanh),
    ("abs(z)", np.abs),
    ("sqrt(abs(z))", lambda z: np.sqrt(np.abs(z))),
)


@dataclass(frozen=True)
class SymbolicEdge:
    """A closed form fitted to one input-output edge of an :class:`RBFKANLayer`."""

    out_index: int
    in_index: int
    terms: tuple[str, ...]  # the selected library terms, in LIBRARY order
    coefficients: tuple[float, ...]  # their fitted coefficients, aligned with terms
    offset: float  # the intercept -- gauge-dependent, see the module docstring
    r_squared: float  # fit quality ON THE GRID RANGE; says nothing outside it
    max_abs_error: float  # worst pointwise error on the grid range
    expression: str  # human-readable, e.g. "0.500000*z**2 + 0.700000"

    def __call__(self, z: Vector) -> Vector:
        """Evaluate the extracted formula -- unlike the layer, this extrapolates."""
        out = np.full_like(np.asarray(z, dtype=np.float64), self.offset)
        table = dict(LIBRARY)
        for name, coefficient in zip(self.terms, self.coefficients, strict=True):
            out = out + coefficient * table[name](np.asarray(z, dtype=np.float64))
        return out


def kan_edge(layer: RBFKANLayer, out_index: int, in_index: int, z: Array) -> Array:
    """The exact scalar edge map ``phi_ji`` of ``layer``, as a function of one input coordinate.

    This is not an approximation of the layer: summing ``kan_edge`` over ``in_index`` and adding
    ``layer.bias[out_index]`` reproduces ``layer(z)`` exactly.
    """
    grid = jnp.linspace(-layer.grid_range, layer.grid_range, layer.num_grid)
    inv_h = (layer.num_grid - 1) / (2.0 * layer.grid_range)
    basis = jnp.exp(-(((z - grid) * inv_h) ** 2))
    return layer.coeff[out_index, in_index] @ basis + layer.base_weight[
        out_index, in_index
    ] * jax.nn.silu(z)


def _best_subset(
    design: Vector, target: Vector, max_terms: int, r2_target: float
) -> tuple[tuple[int, ...], Vector, float]:
    """Exhaustive best-subset least squares: smallest subset reaching ``r2_target``, else the best.

    Exhaustive rather than greedy because the library is small enough to afford it and a forward
    path can miss the true subset outright -- ``sin`` and ``z - z**3/6`` are close on a short range,
    so a greedy first pick can lock in the wrong term and never recover.
    """
    n_terms = design.shape[1] - 1  # column 0 is the intercept, always kept and never counted
    centred = target - target.mean()
    total = float(centred @ centred)
    best: tuple[tuple[int, ...], Vector, float] | None = None
    for size in range(1, max_terms + 1):
        for subset in combinations(range(n_terms), size):
            columns = (0, *(i + 1 for i in subset))
            solution, *_ = np.linalg.lstsq(design[:, columns], target, rcond=None)
            residual = target - design[:, columns] @ solution
            r2 = 1.0 - float(residual @ residual) / total if total > 0.0 else 1.0
            if best is None or r2 > best[2]:
                best = (subset, solution, r2)
        if best is not None and best[2] >= r2_target:
            break  # parsimony: stop at the smallest size that is already good enough
    assert best is not None
    return best


def extract_symbolic_edge(
    layer: RBFKANLayer,
    out_index: int,
    in_index: int,
    *,
    max_terms: int = 2,
    n_points: int = 401,
    r2_target: float = 0.999,
    library: Sequence[tuple[str, Callable[[Vector], Vector]]] = LIBRARY,
) -> SymbolicEdge:
    """Fit a closed form to one edge, on the grid range where the RBF basis has support.

    The selection rule is stated rather than tuned: take the smallest subset of ``library`` whose
    least-squares fit reaches ``r2_target``, and if none does, the best subset of size
    ``max_terms``. An intercept is always present and is reported separately as ``offset``, because
    it is the gauge-dependent part.
    """
    zs = np.linspace(-layer.grid_range, layer.grid_range, n_points, dtype=np.float64)
    values = np.asarray(
        jax.vmap(lambda z: kan_edge(layer, out_index, in_index, z))(jnp.asarray(zs)),
        dtype=np.float64,
    )
    design = np.column_stack([np.ones_like(zs), *(fn(zs) for _, fn in library)])
    subset, solution, r2 = _best_subset(design, values, max_terms, r2_target)
    names = tuple(library[i][0] for i in subset)
    coefficients = tuple(float(c) for c in solution[1:])
    offset = float(solution[0])
    columns = (0, *(i + 1 for i in subset))
    error = float(np.max(np.abs(values - design[:, columns] @ solution)))
    pieces = [f"{c:+.6f}*{name}" for c, name in zip(coefficients, names, strict=True)]
    expression = " ".join([*pieces, f"{offset:+.6f}"]).lstrip("+").strip()
    return SymbolicEdge(out_index, in_index, names, coefficients, offset, r2, error, expression)


def extract_symbolic(
    layer: RBFKANLayer,
    *,
    max_terms: int = 2,
    n_points: int = 401,
    r2_target: float = 0.999,
) -> tuple[SymbolicEdge, ...]:
    """Extract every edge of ``layer``, row-major in ``(out_index, in_index)``."""
    return tuple(
        extract_symbolic_edge(
            layer, j, i, max_terms=max_terms, n_points=n_points, r2_target=r2_target
        )
        for j in range(layer.out_dim)
        for i in range(layer.in_dim)
    )


def _fit_layer(spec: RBFKANLayer, inputs: Vector, targets: Vector) -> RBFKANLayer:
    """Fit ``spec``'s parameters by ORDINARY LEAST SQUARES, not gradient descent.

    A single RBF-KAN layer is linear in ``coeff``, ``base_weight`` and ``bias`` once the grid is
    fixed, so the fit is a closed-form linear solve: exact, deterministic, seed-free and orders of
    magnitude faster than Adam. That is only true for one layer -- stack two and the composition is
    nonlinear and this shortcut is gone.
    """
    grid = np.linspace(-spec.grid_range, spec.grid_range, spec.num_grid, dtype=np.float64)
    inv_h = (spec.num_grid - 1) / (2.0 * spec.grid_range)
    rbf = np.exp(-(((inputs[:, :, None] - grid[None, None, :]) * inv_h) ** 2))
    silu = inputs / (1.0 + np.exp(-inputs))
    design = np.column_stack([rbf.reshape(inputs.shape[0], -1), silu, np.ones(inputs.shape[0])])
    solution, *_ = np.linalg.lstsq(design, targets, rcond=None)
    n_rbf = spec.in_dim * spec.num_grid
    coeff = solution[:n_rbf].T.reshape(spec.out_dim, spec.in_dim, spec.num_grid)
    base = solution[n_rbf : n_rbf + spec.in_dim].T.reshape(spec.out_dim, spec.in_dim)
    bias = solution[n_rbf + spec.in_dim].reshape(spec.out_dim)
    return eqx.tree_at(
        lambda m: (m.coeff, m.base_weight, m.bias),
        spec,
        (jnp.asarray(coeff), jnp.asarray(base), jnp.asarray(bias)),
    )


@dataclass(frozen=True)
class SymbolicExtractionCurve:
    """Evidence that extraction recovers the generating formula -- and where it stops working."""

    edges: tuple[SymbolicEdge, ...]  # the recovered forms on an additively separable target
    layer_r_squared: float  # how well the fitted layer reproduces the separable target
    recovered_terms: tuple[tuple[str, ...], ...]  # per edge -- must match the true terms
    coefficient_error: float  # max |recovered coefficient - true coefficient|
    gauge_residual: float  # |bias + sum of edge offsets - true constant|: the gauge identity
    interaction_r_squared: float  # SAME pipeline on z1*z2 -- an additive layer cannot fit it
    interaction_sup_error: float  # the layer's worst error on that target, over the box
    interaction_floor: float  # r^2, the PROVED floor no additive model can beat (see the module)
    layer_extrapolation_error: float  # the LAYER outside the grid: the RBF support ends
    symbolic_extrapolation_error: float  # the FORMULA outside the grid: it does not
    ok: bool


def symbolic_extraction_certificate(
    *,
    n_samples: int = 4000,
    num_grid: int = 16,
    grid_range: float = 3.0,
    seed: int = 0,
) -> SymbolicExtractionCurve:
    """Fit a KAN layer to a KNOWN formula, extract it back, and price the two scope boundaries.

    The generating function is ``0.7 + 0.5*z0**2 - 1.2*sin(z1)`` -- additively separable, so a
    single layer can represent it exactly and the extraction has a right answer to be checked
    against. Two arms exist to make the certificate able to fail: an interaction target ``z0*z1``
    that no additive layer can fit, and evaluation outside the grid range, where the layer loses its
    RBF support but the extracted formula does not.
    """
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(-grid_range, grid_range, size=(n_samples, 2))
    truth = 0.7 + 0.5 * inputs[:, 0] ** 2 - 1.2 * np.sin(inputs[:, 1])
    spec = RBFKANLayer(2, 1, num_grid, grid_range, key=jax.random.PRNGKey(seed))
    layer = _fit_layer(spec, inputs, truth[:, None])
    predicted = np.asarray(jax.vmap(layer)(jnp.asarray(inputs)), dtype=np.float64)[:, 0]
    centred = truth - truth.mean()
    residual = truth - predicted
    layer_r2 = 1.0 - float(residual @ residual) / float(centred @ centred)

    edges = extract_symbolic(layer)
    truth_coefficients = {"z**2": 0.5, "sin(z)": -1.2}
    coefficient_error = max(
        abs(edge.coefficients[0] - truth_coefficients.get(edge.terms[0], np.inf))
        if len(edge.terms) == 1
        else np.inf
        for edge in edges
    )
    gauge_residual = abs(float(layer.bias[0]) + sum(e.offset for e in edges) - 0.7)

    # The separability boundary: an interaction is not additive, so the same pipeline must do badly.
    interaction = inputs[:, 0] * inputs[:, 1]
    inter_layer = _fit_layer(spec, inputs, interaction[:, None])
    inter_pred = np.asarray(jax.vmap(inter_layer)(jnp.asarray(inputs)), dtype=np.float64)[:, 0]
    inter_centred = interaction - interaction.mean()
    inter_resid = interaction - inter_pred
    interaction_r2 = 1.0 - float(inter_resid @ inter_resid) / float(inter_centred @ inter_centred)
    # The floor is not an observation, it is a theorem: the mixed second difference annihilates
    # every additive function, has four terms, and equals (x1-x0)(y1-y0) on the bilinear target --
    # so on [-r,r]^2 the sup error of ANY additive approximation is at least (2r)(2r)/4 = r^2.
    corners = np.array(
        [
            [-grid_range, -grid_range],
            [-grid_range, grid_range],
            [grid_range, -grid_range],
            [grid_range, grid_range],
        ]
    )
    inter_corner = np.asarray(jax.vmap(inter_layer)(jnp.asarray(corners)), dtype=np.float64)[:, 0]
    inter_sup = float(np.max(np.abs(inter_corner - corners[:, 0] * corners[:, 1])))
    inter_floor = grid_range * grid_range

    # The extrapolation boundary: outside the grid the RBFs have decayed and only silu is left, so
    # the LAYER degrades. The extracted formula carries no grid and does not.
    far = np.linspace(2.0 * grid_range, 3.0 * grid_range, 64)
    far_inputs = np.column_stack([far, np.zeros_like(far)])
    far_truth = 0.7 + 0.5 * far**2 - 1.2 * np.sin(np.zeros_like(far))
    far_layer = np.asarray(jax.vmap(layer)(jnp.asarray(far_inputs)), dtype=np.float64)[:, 0]
    layer_far_error = float(np.max(np.abs(far_layer - far_truth)))
    far_symbolic = float(layer.bias[0]) + edges[0](far) + edges[1](np.zeros_like(far))
    symbolic_far_error = float(np.max(np.abs(far_symbolic - far_truth)))

    ok = (
        layer_r2 > 0.999
        and coefficient_error < 0.05
        # Not machine epsilon: the identity is exact in exact arithmetic, so what is left is the
        # accumulated least-squares and symbolic-fit error. The point is that it comes out ~1000x
        # SMALLER than the individual edges' own max_abs_error -- the split cancels.
        and gauge_residual < 1e-4
        and interaction_r2 < 0.05
        and inter_sup >= inter_floor  # the proved floor, checked rather than quoted
        and symbolic_far_error < 0.05 * layer_far_error
    )
    return SymbolicExtractionCurve(
        edges,
        layer_r2,
        tuple(e.terms for e in edges),
        float(coefficient_error),
        float(gauge_residual),
        interaction_r2,
        inter_sup,
        inter_floor,
        layer_far_error,
        symbolic_far_error,
        ok,
    )
