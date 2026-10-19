# chc.research

**Stability tier: [research](../index.md#research).**

The modules that each carry one research result — a solver, a certificate, a benchmark or a case
study's plant — rather than a stage of a decision taken from logs. They are outside the surface
0.20.0 freezes: any of them may change or be withdrawn in any release, and `import chc` imports
none of them. Import each from its own path, `chc.research.<module>`.

Each moved here in 0.19.0. Until 0.20.0 each name still imports from the path it had before, with
a `DeprecationWarning` that names its path here: `chc.mmm.run_marketing_mix leaves chc.mmm in
0.20.0: import it from chc.research.mmm`.

| module | what it carries | path before 0.19.0 |
|---|---|---|
| [`benchmark`](benchmark.md) | confounded, constrained control tasks with oracle regret | `chc.benchmark` |
| [`causal_bench`](causal_bench.md) | a leaderboard of modern estimators against the naive ones they are meant to beat | `chc.causal_bench` |
| [`deep_galerkin`](deep_galerkin.md) | the Deep Galerkin method, met with classical Galerkin on one PDE, and its mean-field games | `chc.deep_galerkin` |
| [`epidemic`](epidemic.md) | SIR epidemic control down to a hospital capacity | `chc.epidemic` |
| [`flagship`](flagship.md) | causal against predictive control under confounding | `chc.flagship` |
| [`galerkin`](galerkin.md) | 1-D Galerkin finite elements with a tridiagonal solve | `chc.galerkin` |
| [`koopman`](koopman.md) | Koopman dynamics with LQR | `chc.koopman` |
| [`lalonde`](lalonde.md) | the LaLonde–Dehejia–Wahba job-training benchmark | `chc.lalonde` |
| [`meanfield`](meanfield.md) | sequential mean-field control over a zone graph | `chc.meanfield` |
| [`mintime`](mintime.md) | minimum-time control by Pontryagin's principle | `chc.mintime` |
| [`mmm`](mmm.md) | the media-budgets case study's plant, not a media-mix model | `chc.mmm` |
| [`sensitivity`](sensitivity.md) | robust control under hidden confounding, one façade | `chc.sensitivity` |
| [`spine`](spine.md) | confounded logs to a certified plan, as one story | `chc.spine` |
| [`surrogate`](surrogate.md) | gradient-boosted surrogate dynamics, the tabular baseline | `chc.surrogate` |
| [`symbolic`](symbolic.md) | symbolic extraction from a Kolmogorov–Arnold layer | `chc.symbolic` |
| [`transport`](transport.md) | continuum mean-field control by density transport | `chc.transport` |
| [`regret`](regret.md) | the regret certificates that each back one research result | `chc.regret` |
| [`games`](games.md) | the certificates of the equilibrium solver and the leader | `chc.games` |
| [`residual`](residual.md) | the evidence for the circulant residual backbone | `chc.residual` |
