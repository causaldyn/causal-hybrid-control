"""LaLonde-Dehejia-Wahba job-training benchmark: a randomized ATE recovered from confounded data.

The National Supported Work (NSW) demonstration *randomized* a job-training treatment; its
experimental effect on 1978 earnings is ~$1794 (Dehejia & Wahba 1999) -- an author-independent
truth, not a self-authored DGP. The LaLonde (1986) challenge swaps the experimental controls for a
non-experimental CPS comparison pool: the naive treated-minus-control gap is then catastrophically
biased (wrong sign, ~ -$8500), and a causal estimator's job is to recover the experimental number by
covariate adjustment. This exercises CHC's causal backbone (:mod:`chc.estimators`) on real external
data -- the external-validity corroboration `plans/19` E asks for.

Data (public domain, ~445 + 15992 rows) is fetched from the Rdatasets mirror on first use and cached
under ``~/.cache/chc``; no dataset ships in the repo and no package dependency is added (urllib +
NumPy only). Gated like the BOPTEST tasks: if the fetch fails (offline), callers skip.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from chc.estimators import CausalEffectEstimator

_BASE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/{name}.csv"
_COVARIATES = ("age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75")


@dataclass(frozen=True)
class LalondeData:
    """The observational LaLonde sample (NSW-treated + CPS controls) plus the randomized truth."""

    treatment: NDArray[np.float64]  # (n,) 1 = NSW-treated, 0 = CPS control
    outcome: NDArray[np.float64]  # (n,) 1978 earnings, re78 (dollars)
    covariates: dict[str, NDArray[np.float64]]  # covariate name -> (n,) column
    experimental_ate: float  # the randomized NSW effect (~$1794): the author-independent truth

    @property
    def naive_ate(self) -> float:
        """Unadjusted treated-minus-control earnings gap -- badly confounded on the CPS pool."""
        treated = self.treatment == 1
        return float(self.outcome[treated].mean() - self.outcome[~treated].mean())


def _fetch(name: str, cache_dir: Path) -> str:
    """Return the CSV text for a causaldata table, downloading to ``cache_dir`` on first use."""
    cached = cache_dir / f"{name}.csv"
    if not cached.exists():
        with urllib.request.urlopen(_BASE_URL.format(name=name), timeout=30) as response:
            cached.write_bytes(response.read())
    return cached.read_text()


def _parse(text: str, columns: tuple[str, ...]) -> dict[str, NDArray[np.float64]]:
    lines = text.splitlines()
    header = [name.strip('"') for name in lines[0].split(",")]
    index = {name: header.index(name) for name in columns}
    rows = [line.split(",") for line in lines[1:] if line]
    return {name: np.array([float(r[index[name]]) for r in rows]) for name in columns}


def load_lalonde(cache_dir: str | Path | None = None) -> LalondeData:
    """Load the LaLonde benchmark: experimental NSW ATE + the confounded observational sample.

    Fetches ``nsw_mixtape`` (randomized) and ``cps_mixtape`` (CPS controls) from Rdatasets, caching
    under ``cache_dir`` (default ``~/.cache/chc``). Raises on a failed fetch -- callers skip
    when offline, like the BOPTEST tasks.
    """
    cache = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "chc"
    cache.mkdir(parents=True, exist_ok=True)
    columns = ("treat", *_COVARIATES, "re78")
    nsw = _parse(_fetch("nsw_mixtape", cache), columns)
    cps = _parse(_fetch("cps_mixtape", cache), columns)

    treated = nsw["treat"] == 1
    experimental_ate = float(nsw["re78"][treated].mean() - nsw["re78"][~treated].mean())
    treatment = np.concatenate([np.ones(int(treated.sum())), np.zeros(cps["re78"].size)])
    outcome = np.concatenate([nsw["re78"][treated], cps["re78"]])
    covariates = {c: np.concatenate([nsw[c][treated], cps[c]]) for c in _COVARIATES}
    return LalondeData(treatment, outcome, covariates, experimental_ate)


def lalonde_ate(data: LalondeData, estimator: CausalEffectEstimator) -> float:
    """Adjusted ATE (dollars) from a CHC estimator on the confounded observational sample.

    Covariates are standardised (raw ``re74``/``re75`` earnings would blow up polynomial nuisances)
    and the outcome scaled to $1000s for conditioning, then the estimate scaled back to dollars.
    """
    names = tuple(data.covariates)
    matrix = np.column_stack([data.covariates[name] for name in names])
    standardized = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-9)
    payload: dict[str, jnp.ndarray] = {
        "treat": jnp.asarray(data.treatment),
        "re78": jnp.asarray(data.outcome / 1000.0),
    }
    for i, name in enumerate(names):
        payload[name] = jnp.asarray(standardized[:, i])
    estimate = estimator.estimate(payload, treatment="treat", outcome="re78", covariates=names)
    return float(estimate.effect) * 1000.0


def lalonde_report(data: LalondeData, estimators: dict[str, CausalEffectEstimator]) -> str:
    """Format the experimental truth, the naive bias, and each estimator's adjusted ATE."""
    truth = data.experimental_ate
    header = f"{'estimator':<16}{'ATE ($)':>12}{'bias vs experiment':>22}"
    rows = [
        f"{'experimental':<16}{truth:>12.0f}{0.0:>22.0f}",
        f"{'naive':<16}{data.naive_ate:>12.0f}{data.naive_ate - truth:>22.0f}",
    ]
    for name, estimator in estimators.items():
        ate = lalonde_ate(data, estimator)
        rows.append(f"{name:<16}{ate:>12.0f}{ate - truth:>22.0f}")
    return "\n".join([header, *rows])
