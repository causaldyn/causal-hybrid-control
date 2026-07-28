# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to adhere to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) once the API stabilises (pre-1.0 it may
still change).

## [Unreleased]

Work landed on `main` since `v0.1.0`. The theme is **guarantees**: most of it is a machine-checked
result line at the causal↔control seam (Maxima derivation → Rocq proof → numeric certificate), with the
matching runtime primitives shipped alongside. See `discoveries/theorems.md` (local research log) for
the statements, scopes and proof names.

### Added

- **Sensitivity-aware control under hidden confounding** (`chc.sensitivity`, a facade over
  `chc.regret` + `chc.uncertainty`). Bounded-density-ratio (marginal MSM) worst-case effect as a CVaR
  mixture → pessimism-radius inflation (`confounding_robust_inflation`, `msm_worst_case_mean`,
  `confounding_robust_radius`); the confounding regret floor is *second order* in the effect bias
  (`confounding_robust_lq_regret`, plus a matrix Frobenius lift); a **minimax controller** whose gain
  the radius shifts under asymmetric over/under-shoot loss (`confounding_robust_control`, sign
  dichotomy, piecewise improvement gap); the radius inside the replanning tube
  (`confounding_robust_closed_loop_bound`). Grounded on a synthetic observational confounded
  marketplace, then lifted into a genuine receding-horizon **closed loop** on a confounded plant
  (`confounding_robust_tracking_loop`, `confounding_robust_tracking_benchmark`).
- **`ConfoundingRobustPenalty`** (`chc.uncertainty`) — a `PenaltyModel` carrying the sensitivity radius
  into the general pessimistic-control stack (`radius·Σ‖u_t‖`, from the bound `‖Δ_B·u_t‖ ≤
  radius·‖u_t‖`), and **`ConfoundingRobustTask`** (`chc.benchmark`), its leaderboard row: under a
  *hidden* confounder no estimator can help, and the radius still cuts regret ~40% vs
  certainty-equivalence with separated multi-seed CIs.
- **Certified planning** (`chc.uncertainty`, `chc.residual`) — certified-Lipschitz rollout-error tubes
  via discrete Grönwall feeding the pessimism radius, with time-varying tubes, constraint tightening,
  a certified-safe horizon and a closed-loop (replanning) variant; **`ContractiveResidual`** with a
  certified negative log-norm, which replaces the `e^{LT}` growth with a bounded radius; a
  **port-Hamiltonian** residual with a machine-checked damping-injection Lyapunov certificate;
  **`WassersteinPenalty`**, a W1-DRO distribution-shift margin.
- **`chc.pathway`** — one `causal_pathway(target)` API over the temporal causal graph, with
  Rocq-certified walk-sum / geometric-truncation / weakest-link structural laws.
- **Marketplace layer** — `chc.matching` (Kantorovich OT dispatch with dual surge prices) and
  `chc.marketplace` (offline causal control under equilibrium interference, where naive and MOPO-style
  baselines go negative); influence-function standard errors and CIs on the cross-fit DML effect.
- **Regret / guarantee line** — the orthogonal-to-control transfer theorem (order `p` → `2p`, scalar
  and multivariate-LQ), multi-channel network control (debias *every* channel), the adaptive
  information-exploration duality with its `√T` lower bound, the C2 end-to-end theorem with a clustered
  van-Trees lower bound and an exposure-map generalisation, plus a batch of scoped propositions and
  corollaries (doubly-robust control, H∞-as-pessimism, constrained piecewise-quadratic regret,
  confounded turnpike, transportability, ensemble heterogeneity, partial-identification sign threshold).

### Changed

- Ten rounds of external review folded in as **scope and honesty corrections**, not new claims: the
  explicit-Euler contraction factor was wrong (`√(1+2μΔt+L²Δt²)`, sufficient step `Δt < 2c/L²`); the
  confounding effect error needed the control magnitude to be dimensionally right; the `§35`
  improvement gap is piecewise (the undershoot-dominant branch was unproved while the benchmark ran in
  it); `§32` is the bounded-density-ratio *marginal* special case of Tan's MSM, not the full model, and
  its monotonicity argument is feasible-set nesting; the confounded-marketplace benchmark is
  *observational*, not a randomised switchback. Several results were relabelled to their honest status
  (order-transfer *lemma*, local-not-global, scalar-not-universal, `≈`-not-`=`).
- Documentation counts corrected after an audit found a silently drifting entry count in the research
  log; the README test count was stale by two releases.

### Fixed

- `ConfoundingRobustPenalty` used `‖u‖`, whose gradient is NaN at `u = 0` — exactly where the solver
  starts — so `0·NaN` poisoned every step and the control stayed pinned at zero. Now a smoothed
  `√(‖u‖²+ε)`, which zeroes the gradient at the origin and preserves the linear bound.

### Notes

Version and tag are deliberately untouched: `pyproject.toml` still declares `0.1.0`, and the `v0.1.0`
tag has not been pushed. Publishing is a separate, explicit decision — pushing any `v*` tag triggers
`release.yml` and a PyPI upload.

## [0.1.0] — 2026-07-19

First tagged release. `chc` is a small JAX library that fuses **physics-structured hybrid dynamics**,
**causal identification**, and **constrained optimal control**, made safe on offline/confounded data by
an explicit **pessimism/support** layer — evaluating decisions as interventions, not correlations.

### Added

- **Hybrid dynamics & sensitivity** — additive `f_known + r_θ` (`chc.dynamics`); MLP / RBF-KAN / graph /
  zero residual backends (`chc.residual`); RK4 rollout (`chc.integrate`); a hand-written discrete adjoint
  verified against autodiff and finite differences, plus an adaptive `diffrax` continuous adjoint
  (`chc.adjoint`); Lie–Trotter / Strang–Marchuk operator splitting (`chc.splitting`).
- **Causal frontier** — pluggable effect estimators (backdoor OLS, IV/2SLS, cross-fitted Double ML,
  sensitivity + refutation, VanderWeele–Ding **E-values**) behind one Strategy interface
  (`chc.causal`, `chc.estimators`), with lazy EconML/DoWhy adapters; the **R-learner** CATE meta-learner;
  staggered-adoption **DiD** — Callaway–Sant'Anna group-time ATT and the de Chaisemartin–d'Haultfoeuille
  DID_M (`chc.did`); **augmented synthetic control** (`chc.scm`); Robins **g-methods** for time-varying
  treatment (`chc.gmethods`); autocorrelation-robust CI testing + lagged-parent **discovery**
  (`chc.independence`, `chc.discovery`); network/interference-aware DML with a learned GNN nuisance
  (`chc.network_causal`).
- **Offline safety & guarantees** — density-distance support penalty (`chc.support`) plus **calibrated**
  deep-ensemble + split-conformal uncertainty (`chc.uncertainty`); IPS/SNIPS off-policy value + overlap
  gate (`chc.offpolicy`); an **LQ certainty-equivalence regret bound** and a local nonlinear certificate
  via linearisation (`chc.regret`, `chc.lqr.linearized_regret_certificate`).
- **Control** — projected-gradient optimal control and receding-horizon MPC (`chc.control`, `chc.mpc`);
  LQR / Riccati (`chc.lqr`); Koopman-LQR (`chc.koopman`); mean-field control (`chc.meanfield`); optimal
  transport (`chc.transport`); differentiable Stackelberg games (`chc.games`); PMP time-optimal bang-bang
  (`chc.mintime`); classical step-response metrics (`chc.metrics`).
- **Structured operators & dynamic effects** — Toeplitz FFT matvec, Levinson–Durbin, Gohberg–Semencul
  inverse + few-sample covariance (`chc.toeplitz`); impulse-response / local-projection dynamic effects
  (`chc.irf`); Galerkin FEM and a mesh-free Deep Galerkin Poisson solver (`chc.galerkin`,
  `chc.deep_galerkin`).
- **Benchmark & real-data validation** — oracle-regret tasks (pricing, inventory, support-shift,
  model-uncertainty) with a leaderboard and multi-seed bootstrap CIs (`chc.benchmark`); a causal-methods
  leaderboard vs naive baselines (`chc.causal_bench`); the **LaLonde-Dehejia-Wahba** external causal
  benchmark (`chc.lalonde`, recovers the randomised ATE from CPS-confounded data). The control loop is
  validated live on a real **BOPTEST** `bestest_hydronic_heat_pump` emulator (via the `causaldyn-bench`
  sibling), where the identification + forecast-MPC beat the tuned built-in baseline on every KPI.
- **Tooling** — `src`-layout, `uv`-managed, `py.typed`; `ruff` + astral `ty` gates; CI test matrix on
  Python 3.12 / 3.13 / 3.14.

[0.1.0]: https://github.com/ilgrad/causal-hybrid-control/releases/tag/v0.1.0
