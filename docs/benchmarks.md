# Benchmarks

Two leaderboards ship with the library as code: every task, every controller and every reference it
is scored against live in [`chc.benchmark`](api/benchmark.md) and
[`chc.causal_bench`](api/causal_bench.md). The tables on this page were printed by the command above
each one while this site was built, at JAX's default precision, as the same command runs in a
default environment (see [the dtype policy](concepts/dtype-policy.md)).

## Control: regret against an oracle

Each task ships a confounded offline dataset, a true plant with a computable oracle controller, and
an evaluation that scores every controller on the true plant. The point is to measure *where*
causal control beats predictive control, and to be honest where it does not.

| column | meaning |
|---|---|
| `cost` | the controller's task cost on the true plant |
| `regret` | the controller's cost minus the oracle's, the best plan found on the true system; each task's docstring says how far from the best it is known to be |
| `95% CI` | percentile bootstrap over the seeds, for the mean regret |
| `ood` | fraction of actions outside the logged action support |
| `viol` | fraction of steps outside the safe state set |
| `seeds` | how many data seeds the row aggregates |

The delay-oscillation task runs on a single seed on purpose: its closed loop is deterministic given
the gain, and the gain grid quantises away the one seed-dependent piece, the estimated delay. The
LaLonde-DW rows are external data, scored against the randomised experiment rather than an oracle;
the files come from the Rdatasets mirror on first use and are cached under `~/.cache/chc`, and the
script prints `skipped` in their place when it is offline.

```bash
uv run python scripts/run_benchmark.py
```

```text
--8<-- "docs/output/run_benchmark.txt"
```

## Causal methods: bias against a known effect

Every method in the causal frontier — [`chc.did`](api/did.md), [`chc.scm`](api/scm.md),
[`chc.estimators`](api/estimators.md), [`chc.gmethods`](api/gmethods.md) — is run on a
self-contained synthetic data-generating process with a ground-truth effect, next to the naive
built-in it is meant to beat. `truth` is that effect, `est` the method's estimate, and the last two
columns are the absolute bias of the method and of its baseline.

```bash
uv run python scripts/run_causal_bench.py
```

```text
--8<-- "docs/output/run_causal_bench.txt"
```

## Head to head: DCBO's dynamic SCMs

The two leaderboards above are this library's own. The comparison with the nearest academic method,
DCBO on its own three dynamic SCMs, is Track L of
[`causaldyn-bench`](https://github.com/causaldyn/causaldyn-bench), because running DCBO takes its
reference implementation and GPy, neither of which this library depends on:
[`results/track_l.md`](https://github.com/causaldyn/causaldyn-bench/blob/main/results/track_l.md).
[When not to use it](why.md#when-not-to-use-it) says what it means for choosing a tool.

## As a notebook

[Tutorial 5 (the scoreboard)](tutorials/05_benchmark_scoreboard.md) runs the pricing, inventory and
support-shift tasks on one seed each and plots the regret of each controller against the oracle.
