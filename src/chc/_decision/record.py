"""The decision record's JSON form: the inputs that move a prescription, the model's
digest, the values JSON carries, and the first difference between two records."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
import types
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import jax
import numpy as np
from numpy.typing import ArrayLike, NDArray

from chc.dynamics import Dynamics
from chc.dynamics_id import Integrator
from chc.graph import CausalGraph
from chc.mpc import PeriodBudget
from chc.panel import Panel, _sorted_fingerprint
from chc.plan import PlannedCostBias

if TYPE_CHECKING:
    from chc.decision import Constraint, DitherCheck, Driver, Lever, Target


def _plan_sha256(record: Mapping[str, Any]) -> str:
    """The sha256 of a decision ``record``'s ``schema_version``, ``schedule`` and ``inputs`` as
    canonical JSON (keys sorted, no spaces): the plan an evaluation read, and what it was made
    from. ``inputs.panel_sha256``, the rows in the order they were handed in, is left out: the same
    rows in another order make the same decision, and ``inputs.panel_sorted_sha256`` names them."""
    named = {key: record[key] for key in ("schema_version", "schedule", "inputs")}
    named["inputs"] = {k: v for k, v in named["inputs"].items() if k != "panel_sha256"}
    text = json.dumps(named, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _show(value: object) -> str:
    if value is None:
        return "not evaluated"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _strict(value: Any) -> Any:
    """``value`` with every float that is not finite read as None, which JSON can carry."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict(item) for item in value]
    return value


def _json_label(label: Any) -> Any:
    """A period's label as JSON carries it: a number or text as it is, a date or a time in ISO
    8601, anything else as its text."""
    if isinstance(label, np.generic):
        label = label.item()
    if isinstance(label, int | float | str):
        return label
    if isinstance(label, datetime.date | datetime.time):
        return label.isoformat()
    return str(label)


def _bias_json(bias: PlannedCostBias | None) -> dict[str, Any] | None:
    if bias is None:
        return None
    return {
        "planned_cost": bias.planned_cost,
        "optimism": bias.optimism,
        "curvature": bias.curvature,
        "first_order": bias.first_order,
        "bias": bias.bias,
        "bias_upper": bias.bias_upper,
        "direction": bias.direction,
        "corrected_cost": bias.corrected_cost,
        "unseen": bias.unseen,
        "free": bias.free,
        "residual": bias.residual,
    }


def _dither_check_json(check: DitherCheck) -> dict[str, Any]:
    contrast = check.contrast
    return {
        "dither": list(check.dither),
        "status": check.status,
        "share": float(check.share),
        "relevance": [float(value) for value in check.relevance],
        "contrast": None
        if contrast is None
        else {
            "cost": contrast.cost,
            "cost_error": contrast.cost_error,
            "noise": contrast.noise,
            "p_value": contrast.p_value,
            "unseen": contrast.unseen,
        },
    }


def _setting(value: Any) -> Any:
    """A number as the record's ``inputs`` holds it: an integer as one, a finite float as one, an
    infinite one as ``"inf"`` or ``"-inf"``, which None, not given, is not."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    number = float(value)
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return number


def _read_setting(value: Any) -> Any:
    """:func:`_setting`'s inverse."""
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    return value


def _floats(values: ArrayLike) -> Any:
    return np.asarray(values, dtype=np.float64).tolist()


def _model_digest(model: Dynamics) -> str:
    """A sha256 over a model's structure, its arrays' dtypes, shapes and bytes, and its functions'
    code.

    A model is any pytree of arrays and code, which JSON does not carry; the digest names it. An
    address in its structure's text, a function's or an object's, is dropped, and a leaf that is not
    a number is read by its type and its text, its addresses dropped too, so that one model reads
    one digest in every process; a function is read by its code as well (:func:`_code_bytes`),
    where its text names it alone, as every lambda's reads ``<lambda>``."""
    leaves, structure = jax.tree_util.tree_flatten(model)
    digest = hashlib.sha256(_ADDRESS.sub("", str(structure)).encode())
    for leaf in leaves:
        if isinstance(leaf, jax.Array | np.ndarray | np.generic | int | float | complex):
            array = np.ascontiguousarray(np.asarray(leaf))
            digest.update(f"{array.dtype}{array.shape}".encode())
            digest.update(array.tobytes())
        else:
            digest.update(type(leaf).__qualname__.encode())
            digest.update(_ADDRESS.sub("", repr(leaf)).encode())
            code = getattr(leaf, "__code__", None)
            if isinstance(code, types.CodeType):
                digest.update(_code_bytes(code))
    return digest.hexdigest()


_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _code_bytes(value: Any) -> bytes:
    """A function's code as :func:`_model_digest` reads it: its bytecode, the names it reads and
    its constants, a nested function's code read alike and a set's members in sorted order, as
    their text's order moves with the process's string hashing. Two lambdas read one text, and
    their code apart; bytecode is the interpreter's, so another Python minor reads another."""
    if isinstance(value, types.CodeType):
        parts = [value.co_code, repr(value.co_names).encode(), *map(_code_bytes, value.co_consts)]
    elif isinstance(value, tuple | frozenset):
        parts = [type(value).__name__.encode(), *map(_code_bytes, value)]
        if isinstance(value, frozenset):
            parts[1:] = sorted(parts[1:])
    else:
        parts = [repr(value).encode()]
    return b"".join(len(part).to_bytes(8, "little") + part for part in parts)


def _inputs_record(
    panel: Panel,
    *,
    levers: Sequence[Lever],
    target: Target,
    horizon: int,
    adjustment: CausalGraph | Sequence[str],
    constraints: Sequence[Constraint],
    hold_constraints: bool,
    budgets: Sequence[PeriodBudget],
    max_levers: int | None,
    known: Dynamics | None,
    dt: float,
    gamma: float,
    radius: str,
    tolerance: float | None,
    x0: NDArray[np.float64] | None,
    folds: int,
    seed: int,
    integrator: Integrator | None,
    drivers: Sequence[Driver],
    neighbours: Mapping[Any, Iterable[Any]] | str | None,
    clusters: str | None,
    nuisance_knots: int | Literal["auto"] | None,
    tube_actions: str,
    dither: str | Sequence[str] | None,
    planned_cost_bias: bool,
    barrier_drift_error: bool,
    inner: str,
) -> dict[str, Any]:
    """Every argument of :func:`prescribe` that moves its result, as plain JSON values."""
    if isinstance(adjustment, CausalGraph):
        graph = adjustment
        written: dict[str, Any] = {
            "kind": "graph",
            "nodes": list(graph.nodes),
            "edges": [
                {
                    "parent": parent,
                    "child": child,
                    "source": source.source,
                    "test": source.test,
                    "p_value": _setting(source.p_value),
                }
                for parent, child in graph.edges
                for source in (graph.edge_provenance(parent, child),)
            ],
            "latent": sorted(graph.latent),
            "lags": graph.lags,
        }
    else:
        written = {"kind": "asserted", "covariates": list(adjustment)}
    value = np.asarray(target.value, dtype=np.float64)
    return {
        "panel_sha256": panel.provenance.data_sha256,
        "panel_sorted_sha256": _sorted_fingerprint(panel),
        "panel_cluster": panel.cluster,
        "levers": [
            {
                "name": lever.name,
                "lo": _setting(lever.lo),
                "hi": _setting(lever.hi),
                "unit_cost": _setting(lever.unit_cost),
                "cap_per_step": _setting(lever.cap_per_step),
            }
            for lever in levers
        ],
        "target": {
            "name": target.name,
            "value": float(value) if value.ndim == 0 else value.tolist(),
            "weight": _setting(target.weight),
        },
        "horizon": int(horizon),
        "adjustment": written,
        "constraints": [
            {"state": c.state, "lo": _setting(c.lo), "hi": _setting(c.hi)} for c in constraints
        ],
        "hold_constraints": bool(hold_constraints),
        "budgets": [
            {
                "weights": _floats(budget.weights),
                "amount": _setting(budget.amount),
                "period": int(budget.period),
                "start": _setting(budget.start),
            }
            for budget in budgets
        ],
        "max_levers": None if max_levers is None else int(max_levers),
        "known": None if known is None else _model_digest(known),
        "dt": _setting(dt),
        "gamma": _setting(gamma),
        "radius": radius,
        "tolerance": _setting(tolerance),
        "x0": None if x0 is None else _floats(x0),
        "folds": int(folds),
        "seed": int(seed),
        "integrator": integrator,
        "drivers": [
            {"name": driver.name, "forecast": _floats(driver.forecast)} for driver in drivers
        ],
        "neighbours": None
        if neighbours is None
        else {"column": neighbours}
        if isinstance(neighbours, str)
        else {
            "units": [
                [_json_label(unit), [_json_label(other) for other in others]]
                for unit, others in neighbours.items()
            ]
        },
        "clusters": clusters,
        "nuisance_knots": nuisance_knots,
        "tube_actions": tube_actions,
        "dither": dither if dither is None or isinstance(dither, str) else list(dither),
        "planned_cost_bias": bool(planned_cost_bias),
        "barrier_drift_error": bool(barrier_drift_error),
        "inner": inner,
    }


_INPUT_KEYS = (
    "panel_sha256",
    "panel_sorted_sha256",
    "panel_cluster",
    "levers",
    "target",
    "horizon",
    "adjustment",
    "constraints",
    "hold_constraints",
    "budgets",
    "max_levers",
    "known",
    "dt",
    "gamma",
    "radius",
    "tolerance",
    "x0",
    "folds",
    "seed",
    "integrator",
    "drivers",
    "neighbours",
    "clusters",
    "nuisance_knots",
    "tube_actions",
    "dither",
    "planned_cost_bias",
    "barrier_drift_error",
    "inner",
)


def _whole(value: Any) -> int:
    """A count or a seed as the record holds it: a JSON integer, not a number of another kind."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{value!r} is not a whole number")
    return value


def _refusal(error: Exception) -> str:
    """An exception's own words: a ``KeyError``'s without the quotes its text adds."""
    return str(error.args[0]) if isinstance(error, KeyError) and error.args else str(error)


def _refused_input(message: str, inputs: Mapping[str, Any]) -> str:
    """The record's ``inputs.<key>`` that :func:`prescribe`'s refusal ``message`` names: the key
    that names the column, where it refuses a column the panel does not have; else the argument
    its messages open with (``gamma=0.5 is ...``, ``lever 'u' has ...``), or ``inputs`` where
    they open with none."""
    for key, names in _columns_named(inputs).items():
        if any(message.startswith(f"column {name!r} is not in the panel") for name in names):
            return f"inputs.{key}"
    match = re.match(r"[a-z_0-9]+", message)
    word = "" if match is None else match.group()
    key = _REFUSED_BY.get(word, word)
    return f"inputs.{key}" if key in _INPUT_KEYS else "inputs"


_REFUSED_BY = {
    "lever": "levers",
    "constraint": "constraints",
    "driver": "drivers",
    "asserted": "adjustment",
}


def _columns_named(inputs: Mapping[str, Any]) -> dict[str, list[Any]]:
    """The columns each of a record's ``inputs`` names, in the order :func:`prescribe` looks them
    up in the panel."""
    neighbours, dither = inputs["neighbours"], inputs["dither"]
    return {
        "target": [inputs["target"]["name"]],
        "constraints": [constraint["state"] for constraint in inputs["constraints"]],
        "levers": [lever["name"] for lever in inputs["levers"]],
        "drivers": [driver["name"] for driver in inputs["drivers"]],
        "clusters": [] if inputs["clusters"] is None else [inputs["clusters"]],
        "neighbours": [neighbours["column"]]
        if isinstance(neighbours, Mapping) and "column" in neighbours
        else [],
        "dither": [] if dither is None else [dither] if isinstance(dither, str) else list(dither),
    }


def _read_neighbours(
    written: Sequence[Sequence[Any]], panel: Panel, clusters: str | None
) -> dict[Any, list[Any]]:
    """Each recorded label back to the panel's own: its units', or under ``clusters`` its
    clusters'; a label that is neither stays as written, for :func:`prescribe` to refuse."""
    labels = np.unique(panel.columns[clusters]).tolist() if clusters is not None else panel.units
    label_of = {_json_label(label): label for label in labels}

    def read(label: Any) -> Any:
        return label_of.get(label, label)

    return {read(unit): [read(other) for other in others] for unit, others in written}


_ENVIRONMENT = ("versions", "backend", "device_kind")


def _without_environment(record: Mapping[str, Any]) -> dict[str, Any]:
    """``record`` less the machine its ``run`` names (:data:`_ENVIRONMENT`), which a replay reads
    off its own: a replay compares the decision, and the library's own version is checked before
    it runs (ADR 0145)."""
    run = record["run"]
    return {**record, "run": {key: value for key, value in run.items() if key not in _ENVIRONMENT}}


_ROW_ORDER = (("inputs", "panel_sha256"), ("provenance", "data_sha256"))


def _without_row_order(record: Mapping[str, Any]) -> dict[str, Any]:
    """``record`` less the digests that read the panel's rows in the order they were handed in
    (:data:`_ROW_ORDER`): a replay checks the rows sorted (``inputs.panel_sorted_sha256``) before
    it runs, and the same rows in another order make the same decision."""
    out = dict(record)
    for section, key in _ROW_ORDER:
        held = out.get(section)
        if isinstance(held, Mapping):
            out[section] = {name: value for name, value in held.items() if name != key}
    return out


def _first_difference(ours: Any, theirs: Any, path: str = "") -> str | None:
    """The first field, as a dotted path, at which two JSON values differ; None where none does."""
    if isinstance(ours, Mapping) and isinstance(theirs, Mapping):
        for key in (*ours, *(key for key in theirs if key not in ours)):
            where = f"{path}.{key}" if path else str(key)
            if key not in ours or key not in theirs:
                return where
            found = _first_difference(ours[key], theirs[key], where)
            if found is not None:
                return found
        return None
    if isinstance(ours, list) and isinstance(theirs, list):
        if len(ours) != len(theirs):
            return path
        for index, (mine, other) in enumerate(zip(ours, theirs, strict=True)):
            found = _first_difference(mine, other, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    same = type(ours) is type(theirs) or (
        isinstance(ours, int | float)
        and isinstance(theirs, int | float)
        and not isinstance(ours, bool)
        and not isinstance(theirs, bool)
    )
    return None if same and ours == theirs else path or "the record"
