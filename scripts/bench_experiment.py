"""The measurements behind docs/adr/0011-designing-an-experiment.md. Regret and coverage only, no
wall time.

The market is the one in ``tests/test_experiment.py``: four zones on a ring, each lever pulling 15%
of its effect from each neighbour, levers boxed to ``[0, 1.2]``. Every experiment is realised here
unit by unit -- each unit's lever at ``+-probe`` at random, the channel read by least squares, the
prior updated in precision form -- which is a different path to the posterior from the design's
own Monte Carlo, so agreement between the two is evidence and not an identity.

    calibration  the regret the design predicts, now and after its experiment, against the regret
                 realised on worlds drawn from its prior, on the three scenarios it runs.
    allocations  (a) the lab's comparison on its unboxed market at channels known up to the
                 experiment: decision-Neyman, classical Neyman, equal units and equal spend at one
                 budget, each realised at the true channels; (b) the design's own allocation
                 against the same three rules at its own spend, on worlds drawn from its prior.
    gate         deploy verdicts: how often acting now stays within ``regret_bound``, nominally
                 95%, on worlds whose truth is known -- drawn from the prior, and fixed while the
                 estimate the prior is centred on is drawn around it.
    wide         the prior's width on the two uncertain zones swept from 6% to 50% of the channel:
                 the local model's gap from Monte Carlo, whether it warns, and whether the numbers
                 it reports stay calibrated.

Run: uv run python scripts/bench_experiment.py {calibration,allocations,gate,wide} [--replicates N]
"""

from __future__ import annotations

import argparse
import json
import logging
import math

import numpy as np
from scipy.stats import beta

from chc import ChannelPrior, ZoneDecision, ZoneExperiment, design_experiment

K = 4
_RING = np.zeros((K, K))
for _k in range(K):
    _RING[_k, (_k + 1) % K] = _RING[_k, (_k - 1) % K] = 0.15
COUPLING = np.eye(K) - _RING
Q = np.diag([2.0, 1.0, 1.5, 0.7])
R = np.diag([0.4, 0.2, 0.6, 0.3])
B_TRUE = np.array([0.8, 1.5, 0.5, 1.1])
NOISE_SD = np.array([1.0, 2.0, 0.7, 1.5])
LAB_COST = np.array([1.0, 0.5, 3.0, 1.0])
EXPERIMENT = ZoneExperiment(unit_cost=1e-5 * LAB_COST, noise_sd=NOISE_SD, probe=np.ones(K))
SEED = 20260928

# baseline, prior mean, prior sd, budget, uses: the scenarios whose verdict is "experiment"
SCENARIOS = {
    "S2 wide on zones 0 and 2": (
        [-1.0, -0.4, -1.3, -0.6],
        B_TRUE,
        [0.3, 0.02, 0.25, 0.02],
        1e-3,
        1.0,
    ),
    "S4 pinned zone far": ([-1.0, 1.5, -1.3, -0.6], B_TRUE, [0.1, 0.4, 0.1, 0.02], 1e-1, 30.0),
    "S5 pinned zone near": ([-1.0, -0.05, -1.3, -0.6], B_TRUE, [0.1, 0.4, 0.1, 0.02], 1e-1, 30.0),
}


def _market(baseline, lo: float = 0.0, hi: float = 1.2) -> ZoneDecision:
    return ZoneDecision(
        coupling=COUPLING,
        baseline=np.asarray(baseline, dtype=float),
        target=np.zeros(K),
        state_weight=Q,
        action_weight=R,
        lo=np.full(K, lo),
        hi=np.full(K, hi),
    )


def _estimate(truth: np.ndarray, units: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each zone's least-squares channel from ``units`` units, levers at ``+-probe`` at random."""
    out = np.full(K, np.nan)
    for k in np.flatnonzero(units > 0):
        probe = EXPERIMENT.probe[k] * rng.choice([-1.0, 1.0], size=int(units[k]))
        response = truth[k] * probe + EXPERIMENT.noise_sd[k] * rng.standard_normal(probe.size)
        out[k] = probe @ response / (probe @ probe)
    return out


def _posterior_mean(prior: ChannelPrior, units: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    precision = np.linalg.inv(prior.covariance)
    information = units * EXPERIMENT.information
    score = precision @ prior.mean + np.where(units > 0, information * estimate, 0.0)
    return np.linalg.solve(precision + np.diag(information), score)


def _mean_and_half_width(values) -> tuple[float, float]:
    values = np.asarray(values)
    return float(values.mean()), float(1.96 * values.std(ddof=1) / math.sqrt(values.size))


def _clopper_pearson(hits: int, n: int) -> tuple[float, float]:
    lo = 0.0 if hits == 0 else float(beta.ppf(0.025, hits, n - hits + 1))
    hi = 1.0 if hits == n else float(beta.ppf(0.975, hits + 1, n - hits))
    return lo, hi


def _worlds(prior: ChannelPrior, replicates: int, rng: np.random.Generator) -> np.ndarray:
    return (
        prior.mean + rng.standard_normal((replicates, K)) @ np.linalg.cholesky(prior.covariance).T
    )


def _realised(decision, prior, units, uses, truths, rng) -> dict:
    """Regret of acting on the prior mean, and on the posterior mean after ``units``, on ``truths``;
    ``rng`` draws only the units, so rules realised on the same truths share their worlds."""
    acting = decision.solve(prior.mean)
    now, after = [], []
    for truth in truths:
        now.append(uses * decision.regret(acting, truth))
        if units.sum():
            posterior = _posterior_mean(prior, units, _estimate(truth, units, rng))
            after.append(uses * decision.regret(decision.solve(posterior), truth))
        else:
            after.append(now[-1])
    return {"now": _mean_and_half_width(now), "after": _mean_and_half_width(after)}


def _calibration(replicates: int) -> dict:
    out = {}
    for index, (name, (baseline, mean, sd, budget, uses)) in enumerate(SCENARIOS.items()):
        decision = _market(baseline)
        prior = ChannelPrior(np.asarray(mean), np.diag(np.asarray(sd) ** 2))
        design = design_experiment(
            decision, prior, EXPERIMENT, budget, uses=uses, draws=20_000, seed=index
        )
        units = np.array(design.sample_sizes)
        rng = np.random.default_rng([SEED, index])
        realised = _realised(decision, prior, units, uses, _worlds(prior, replicates, rng), rng)
        out[name] = {
            "verdict": design.verdict,
            "units": design.sample_sizes,
            "model_gap": design.model_gap,
            "predicted_now": design.regret_now,
            "realised_now": realised["now"],
            "ratio_now": realised["now"][0] / design.regret_now,
            "predicted_after": design.regret_after,
            "realised_after": realised["after"],
            "ratio_after": realised["after"][0] / design.regret_after,
        }
    return out


def _whole(n: np.ndarray, cost: np.ndarray, spend: float) -> np.ndarray:
    whole = np.floor(n * spend / float(cost @ n)).astype(np.int64)
    return np.maximum(whole, 0)


def _allocations(replicates: int) -> dict:
    # (a) the lab's market: unboxed, one zone above target, costs 1 / 0.5 / 3 / 1, budget 2000
    lab = _market([-1.0, -0.4, -1.3, 0.2], lo=-math.inf, hi=math.inf)
    step = 1e-4
    weight = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            ei, ej = np.eye(K)[i] * step, np.eye(K)[j] * step
            weight[i, j] = sum(
                sign * lab.regret(lab.solve(B_TRUE + si * ei + sj * ej), B_TRUE)
                for sign, si, sj in ((1, 1, 1), (-1, 1, -1), (-1, -1, 1), (1, -1, -1))
            ) / (4 * step * step)
    w = np.diag(weight)
    budget = 2000.0

    def scaled(raw: np.ndarray) -> np.ndarray:
        return np.maximum(np.round(budget * raw / float(LAB_COST @ raw)), 2)

    rules = {
        "decision-Neyman": scaled(NOISE_SD * np.sqrt(w / LAB_COST)),
        "classical Neyman": scaled(NOISE_SD / np.sqrt(LAB_COST)),
        "equal units": scaled(np.ones(K)),
        "equal spend": scaled(1.0 / LAB_COST),
    }
    optimum = 0.5 * float(NOISE_SD @ np.sqrt(w * LAB_COST)) ** 2 / budget
    rng = np.random.default_rng([SEED, 10])
    shocks = rng.standard_normal((replicates, K))
    lab_out: dict = {"weight_diagonal": w.tolist(), "closed_form_optimum": optimum}
    for name, n in rules.items():
        sd = NOISE_SD / np.sqrt(n * EXPERIMENT.probe**2)
        regrets = [lab.regret(lab.solve(B_TRUE + sd * z), B_TRUE) for z in shocks]
        lab_out[name] = {
            "units": n.astype(int).tolist(),
            "predicted": 0.5 * float(np.sum(w * sd**2)),
            "realised": _mean_and_half_width(regrets),
        }

    # (b) the design's own allocation on the boxed market, against the same rules at its spend
    decision = _market([-1.0, -0.4, -1.3, -0.6])
    prior = ChannelPrior(B_TRUE, np.diag((0.1 * B_TRUE) ** 2))
    uses = 100.0
    design = design_experiment(decision, prior, EXPERIMENT, 2e-2, uses=uses, draws=20_000)
    units = np.array(design.sample_sizes)
    spend = float(EXPERIMENT.unit_cost @ units)
    candidates = {
        "design_experiment": units,
        "classical Neyman": _whole(NOISE_SD / np.sqrt(LAB_COST), EXPERIMENT.unit_cost, spend),
        "equal units": _whole(np.ones(K), EXPERIMENT.unit_cost, spend),
        "equal spend": _whole(1.0 / LAB_COST, EXPERIMENT.unit_cost, spend),
    }
    bayes_out: dict = {"spend": spend, "regret_now": design.regret_now}
    truths = _worlds(prior, replicates, np.random.default_rng([SEED, 11]))
    for name, n in candidates.items():
        realised = _realised(decision, prior, n, uses, truths, np.random.default_rng([SEED, 12]))
        bayes_out[name] = {
            "units": n.tolist(),
            "spend": float(EXPERIMENT.unit_cost @ n),
            "realised_after": realised["after"],
        }
    return {"lab_market": lab_out, "design_at_its_spend": bayes_out}


def _gate(replicates: int) -> dict:
    decision = _market([-1.0, -0.4, -1.3, -0.6])
    out = {}
    for sd in (0.05, 0.15):
        variance = np.diag(np.full(K, sd**2))
        rng = np.random.default_rng([SEED, 20, int(100 * sd)])
        # worlds drawn from the prior: one design, truths from its prior
        design = design_experiment(
            decision, ChannelPrior(B_TRUE, variance), EXPERIMENT, 0.0, draws=20_000, seed=1
        )
        acting = decision.solve(B_TRUE)
        truths = B_TRUE + sd * rng.standard_normal((replicates, K))
        prior_hits = sum(decision.regret(acting, t) <= design.regret_bound for t in truths)
        # a fixed truth: the estimate is drawn around it and the prior is centred on the estimate
        fixed_hits, scored, verdicts = 0, 0, {"deploy": 0, "abstain": 0, "experiment": 0}
        for replicate in range(replicates):
            estimate = B_TRUE + sd * rng.standard_normal(K)
            design = design_experiment(
                decision,
                ChannelPrior(estimate, variance),
                EXPERIMENT,
                0.0,
                draws=2000,
                seed=replicate,
            )
            verdicts[design.verdict] += 1
            if design.verdict != "deploy":
                continue
            scored += 1
            regret = decision.regret(decision.solve(estimate), B_TRUE)
            fixed_hits += regret <= design.regret_bound
        out[f"sd {sd}"] = {
            "drawn_from_prior": {
                "met": prior_hits / replicates,
                "interval": _clopper_pearson(prior_hits, replicates),
            },
            "fixed_truth": {
                "met": fixed_hits / scored if scored else None,
                "interval": _clopper_pearson(fixed_hits, scored) if scored else None,
                "verdicts": verdicts,
            },
        }
    return out


class _Events(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(getattr(record, "chc_event", ""))


def _wide(replicates: int) -> dict:
    baseline, mean, _, budget, uses = SCENARIOS["S2 wide on zones 0 and 2"]
    decision = _market(baseline)
    handler = _Events()
    logging.getLogger("chc.experiment").addHandler(handler)
    out = {}
    for index, fraction in enumerate((0.06, 0.12, 0.18, 0.25, 0.37, 0.5)):
        sd = np.array([fraction * B_TRUE[0], 0.02, fraction * B_TRUE[2], 0.02])
        prior = ChannelPrior(np.asarray(mean), np.diag(sd**2))
        handler.events.clear()
        design = design_experiment(
            decision, prior, EXPERIMENT, budget, uses=uses, draws=20_000, seed=index
        )
        units = np.array(design.sample_sizes)
        rng = np.random.default_rng([SEED, 30, index])
        realised = _realised(decision, prior, units, uses, _worlds(prior, replicates, rng), rng)
        out[f"sd {fraction:.2f} of b"] = {
            "units": design.sample_sizes,
            "verdict": design.verdict,
            "model_gap": design.model_gap,
            "warned": "design_gap" in handler.events,
            "predicted_after": design.regret_after,
            "realised_after": realised["after"],
            "ratio_after": realised["after"][0] / design.regret_after,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["calibration", "allocations", "gate", "wide"])
    parser.add_argument("--replicates", type=int, default=None)
    args = parser.parse_args()
    run = {"calibration": _calibration, "allocations": _allocations, "gate": _gate, "wide": _wide}
    default = {"calibration": 4000, "allocations": 4000, "gate": 500, "wide": 4000}
    print(json.dumps(run[args.case](args.replicates or default[args.case]), indent=2))


if __name__ == "__main__":
    main()
