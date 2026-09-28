"""The error rates behind docs/adr/0010-a-deployment-gate.md. Verdicts and rates only, no wall time.

The lab's closed loop, run through :class:`chc.gate.DeploymentGate`: eight zones of a synthetic
market, each with a candidate ``N(m, s^2)`` against the baseline ``N(0, 1)``, and a reward that is
Bernoulli with mean ``a_t + 0.6 g(u)``, where ``a_t`` is a shock shared by every zone. Each zone is
logged as its mode asks: under the baseline in shadow, under the mixture in the experiment, under
the candidate once deployed. A check is 96 decisions, and the horizon 150 checks.

    null    eight zones whose candidate is not better than the baseline by more than delta:
            four at the boundary, two at the boundary with chi2 32.6, one heavy-tailed, one harmful.
    mixed   four null zones and four better by 0.05 or more, one of them heavy-tailed.

Per world: the false-deploy rate as FDR at the horizon and at the adversarial stopping time (the
first check at which a null zone is deployed), the chance that any null zone is deployed, the
missed-deploy rate as one minus power, and how often each zone was sent to EXPERIMENT, held or
rolled back.

Run: uv run python scripts/bench_gate.py {null,mixed} [--replications 400] > out.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize, signal, stats

from chc import DeploymentGate, GateConfig, ZoneBatch, ZonePlan
from chc.gate import GateMode

BATCH = 96
CHECKS = 150
REWARD_SCALE = 0.6
CONFIG = GateConfig(
    delta=0.02,
    delta_harm=0.02,
    min_effect=0.02,
    horizon=CHECKS * BATCH,
    alpha=0.10,
    alpha_harm=0.05,
    drift_arl=1e4,
    rho=0.5,
)


def _npdf(u: np.ndarray, m: float, s: float) -> np.ndarray:
    return np.exp(-((u - m) ** 2) / (2 * s * s)) / (math.sqrt(2 * math.pi) * s)


@dataclass(frozen=True)
class Truth:
    m: float
    s: float
    reward: str  # "bump": exp(-(u - 1)^2 / 2); "probit": Phi(u)

    def g(self, u: np.ndarray) -> np.ndarray:
        return np.exp(-((u - 1.0) ** 2) / 2) if self.reward == "bump" else stats.norm.cdf(u)

    @staticmethod
    def value(m: float, s: float, reward: str) -> float:
        v = 1 + s * s
        if reward == "bump":
            return math.exp(-((m - 1.0) ** 2) / (2 * v)) / math.sqrt(v)
        return float(stats.norm.cdf(m / math.sqrt(v)))

    @property
    def contrast(self) -> float:
        return REWARD_SCALE * (
            self.value(self.m, self.s, self.reward) - self.value(0.0, 1.0, self.reward)
        )

    @property
    def plan(self) -> ZonePlan:
        if self.s * self.s >= 2.0:
            chi2 = math.inf
        else:
            q = 1.0 / (self.s * self.s)
            chi2 = q / math.sqrt(2 * q - 1) * math.exp(self.m**2 / (2 - self.s**2)) - 1.0
        tv = (
            0.5
            * integrate.quad(
                lambda u: abs(_npdf(u, self.m, self.s) - _npdf(u, 0.0, 1.0)), -40.0, 40.0, limit=400
            )[0]
        )
        return ZonePlan(chi2, tv)

    def draw(
        self, rng: np.random.Generator, mode: GateMode, a: np.ndarray, rho: float
    ) -> ZoneBatch:
        n = a.size
        if mode == "shadow":
            u = rng.normal(0.0, 1.0, n)
        elif mode == "deployed":
            u = rng.normal(self.m, self.s, n)
        else:
            pick = rng.random(n) < rho
            u = np.where(pick, rng.normal(self.m, self.s, n), rng.normal(0.0, 1.0, n))
        pn, pb = _npdf(u, self.m, self.s), _npdf(u, 0.0, 1.0)
        logged = {"shadow": pb, "deployed": pn, "experiment": (1 - rho) * pb + rho * pn}[mode]
        r = (rng.random(n) < a + REWARD_SCALE * self.g(u)).astype(float)
        return ZoneBatch(reward=r, candidate=pn, baseline=pb, logged=logged)


def _boundary(contrast: float, side: str) -> float:
    def gap(m: float) -> float:
        return Truth(m, 1.0, "bump").contrast - contrast

    return optimize.brentq(gap, -1.0, 1.0) if side == "low" else optimize.brentq(gap, 1.0, 3.0)


def _worlds() -> dict[str, tuple[list[Truth], int]]:
    boundary = Truth(_boundary(CONFIG.delta, "low"), 1.0, "bump")
    boundary_wide = Truth(_boundary(CONFIG.delta, "high"), 1.0, "bump")
    wide_null, harmful = Truth(1.0, 1.6, "bump"), Truth(-0.3, 1.0, "bump")
    better = Truth(_boundary(CONFIG.delta + 0.03, "low"), 1.0, "bump")
    wide_better = Truth(0.5, 1.6, "probit")
    return {
        "null": (
            [boundary] * 4 + [boundary_wide] * 2 + [wide_null, harmful],
            9200,
        ),
        "mixed": (
            [boundary, boundary, boundary_wide, wide_null] + [better] * 3 + [wide_better],
            9201,
        ),
    }


def _run(world: str, replications: int) -> dict:
    truths, seed = _worlds()[world]
    rng = np.random.default_rng(seed)
    names = [f"z{i}" for i in range(len(truths))]
    plans = {z: t.plan for z, t in zip(names, truths, strict=True)}
    null = np.array([t.contrast <= CONFIG.delta + 1e-12 for t in truths])
    harm_null = np.array([-t.contrast <= CONFIG.delta_harm + 1e-12 for t in truths])
    first = np.full((replications, len(truths)), -1)
    seen = {v: np.zeros((replications, len(truths)), dtype=bool) for v in ("experiment", "hold")}
    seen["rollback"] = np.zeros((replications, len(truths)), dtype=bool)
    fdp_star = np.zeros(replications)
    for rep in range(replications):
        gate = DeploymentGate(plans, CONFIG)
        state, stopped = np.zeros(1), False
        for c in range(CHECKS):
            eta, state = signal.lfilter(
                [1.0], [1.0, -0.995], 0.1 * rng.normal(size=BATCH), zi=state
            )
            a = 0.2 + 0.2 * np.tanh(eta)
            batch = {
                z: t.draw(rng, gate.mode(z), a, CONFIG.rho)
                for z, t in zip(names, truths, strict=True)
                if gate.mode(z) != "retired"
            }
            verdicts = gate.update(batch)
            for i, z in enumerate(names):
                if verdicts[z] in seen:
                    seen[verdicts[z]][rep, i] = True
                if verdicts[z] == "deploy" and first[rep, i] < 0:
                    first[rep, i] = c
            deployed = first[rep] >= 0
            if not stopped and (deployed & null).any():
                fdp_star[rep] = (deployed & null).sum() / deployed.sum()
                stopped = True
    deployed = first >= 0
    fdp = (deployed & null).sum(axis=1) / np.maximum(deployed.sum(axis=1), 1)
    any_null = int((deployed & null).any(axis=1).sum())
    alt = ~null
    power = float((deployed & alt).sum() / max(alt.sum() * replications, 1))
    return {
        "world": world,
        "replications": replications,
        "contrast": [t.contrast for t in truths],
        "fdr": float(fdp.mean()),
        "fdr_se": float(fdp.std(ddof=1) / math.sqrt(replications)),
        "fdr_at_first_null_deploy": float(fdp_star.mean()),
        "fdr_at_first_null_deploy_se": float(fdp_star.std(ddof=1) / math.sqrt(replications)),
        "any_null_deployed": any_null,
        "any_null_deployed_cp95": [
            float(stats.beta.ppf(0.025, any_null, replications - any_null + 1))
            if any_null
            else 0.0,
            float(stats.beta.ppf(0.975, any_null + 1, replications - any_null))
            if any_null < replications
            else 1.0,
        ],
        "power": power,
        "missed_deploy_rate": 1.0 - power if alt.any() else None,
        "mean_deploy_check": float(first[:, alt][first[:, alt] >= 0].mean())
        if (deployed & alt).any()
        else None,
        "deployed_share": deployed.mean(axis=0).tolist(),
        "experiment_share": seen["experiment"].mean(axis=0).tolist(),
        "hold_share": seen["hold"].mean(axis=0).tolist(),
        "rollback_share": seen["rollback"].mean(axis=0).tolist(),
        "false_holds": int((seen["hold"] & harm_null & ~deployed).sum()),
        "harm_null_zone_runs": int(harm_null.sum() * replications),
        "rollbacks_of_deployed": int((seen["rollback"] & deployed).sum()),
        "deployments": int(deployed.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("world", choices=["null", "mixed"])
    parser.add_argument("--replications", type=int, default=400)
    args = parser.parse_args()
    print(json.dumps(_run(args.world, args.replications), indent=2))


if __name__ == "__main__":
    main()
