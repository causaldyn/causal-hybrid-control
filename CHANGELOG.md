# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to adhere to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) once the API stabilises (pre-1.0 it may
still change).

## [Unreleased]

Work landed on `main` since `v0.1.0`. The theme is **guarantees**: most of it is a machine-checked
result line at the causal↔control seam (Maxima derivation → Rocq proof → numeric certificate), with the
matching runtime primitives shipped alongside. The proof scripts themselves are in `proofs/` and the
symbolic derivations in `validation/`.

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
  into the general pessimistic-control stack (`radius·Σ‖u_t‖`, from the *transition*-error bound
  `‖Δ_B·u_t‖ ≤ radius·‖u_t‖`; an identification-radius regulariser rather than a certified cost bound,
  since the latter needs a cost-to-go Lipschitz multiplier `lam_unc` currently absorbs), and **`ConfoundingRobustTask`** (`chc.benchmark`), its leaderboard row: under a
  *hidden* confounder no estimator can help, and the radius still cuts regret ~40% vs
  certainty-equivalence with separated multi-seed CIs.
- **Certified planning** (`chc.uncertainty`, `chc.residual`) — certified-Lipschitz rollout-error tubes
  via discrete Grönwall feeding the pessimism radius, with time-varying tubes, constraint tightening,
  a certified-safe horizon and a closed-loop (replanning) variant; **`ContractiveResidual`** with a
  certified negative log-norm, which replaces the `e^{LT}` growth with a bounded radius; a
  **port-Hamiltonian** residual with a machine-checked damping-injection Lyapunov certificate;
  **`WassersteinPenalty`**, a W1-DRO distribution-shift margin.
- **`NestedCVaRPenalty`** (`chc.uncertainty`) — a time-consistent aggregation of ensemble
  disagreement in the same `PenaltyModel` slot. `EnsembleUncertainty` *sums* member variance along the
  trajectory, so one very bad step averages away against many quiet ones; this replaces the sum with
  `rho_t = c_t + CVaR_alpha[rho_{t+1}]`, and `static_penalty_trajectory` keeps the other adversary
  (commit to one member for the whole horizon) so the two are comparable. `nested_risk_certificate`
  checks the ordering that must hold — nested ≥ static ≥ risk-neutral, collapsing at `alpha = 1` — and
  the gap is the price of time consistency, which is what a receding-horizon controller needs if it is
  not to chase its own tail across re-solves. Scoped honestly: with the members re-evaluated
  independently the recursion collapses to `Sum_t CVaR_alpha[c_t]`, a risk-averse *aggregation* rule
  rather than a dynamic-programming solve of a nested-risk MDP.
- **`chc.pathway`** — one `causal_pathway(target)` API over the temporal causal graph, with
  Rocq-certified walk-sum / geometric-truncation / weakest-link structural laws.
- **Marketplace layer** — `chc.matching` (Kantorovich OT dispatch with dual surge prices) and
  `chc.marketplace` (offline causal control under equilibrium interference, where naive and MOPO-style
  baselines go negative); influence-function standard errors and CIs on the cross-fit DML effect.
- **Certified strategic layer** (`chc.games`) — `fixed_point`, a differentiable equilibrium solver that
  iterates to a *relative* residual and returns an `EquilibriumSolution` carrying `residual` and
  `converged`, with the backward pass as the implicit-function VJP (`jax.custom_vjp`, adjoint solved as
  its own fixed point) rather than an unrolled loop; `congestion_contraction_modulus` /
  `congestion_damping` / `congestion_contraction_certificate` to certify or refuse a configuration
  before it runs (`spec(J) ⊆ [0, ½]` sharp ⟹ the uniform Jacobian bound is below 1 iff
  `0 < d < 4/(2+κ)` — sharp over the class of congestion maps, sufficient for any one game — with
  `d* = 4/(4+κ)` certifying every `κ`); `equilibrium_transfer_certificate`, which measures the equilibrium's
  *local* conditioning — exactly 1 in the ambient norm uniformly in `κ` (attained only along the mass
  direction that mass conservation never excites) and strictly below 1 on the fixed-mass tangent space
  where displacements actually live — while the naive contraction constant `1/μ` is loose by up to
  100×. So a `C/μ²` regret bound must not take its constant from *this* solver's contraction margin.
- **`chc.barrier`** — the sensitivity radius spent on a *constraint* instead of on the objective:
  the robust control-barrier margin guaranteed against every effect in the identified set
  (`robust_barrier_margin`), its maximiser — exactly zero once the radius swallows the control
  channel, which is optimal and not conservatism (`robust_safe_action`), the closed-form certified
  action set and the least-restrictive filter that clips a nominal action into it
  (`admissible_action_interval`, `robust_safety_filter`), the sharp radius at which certification
  dies (`identification_radius_threshold`) and the sensitivity level it corresponds to
  (`barrier_gamma_star`) — *the largest sensitivity-model level under which the barrier stays
  certified* (a model parameter, not a measured amount of hidden confounding). Both
  thresholds are **case-split** rather than one formula: a self-satisfying drift gives `inf`, a
  deficit beyond what a perfectly identified channel delivers gives an empty set, and `Δ(Γ)`
  saturating at the CVaR gap gives `Γ* = inf`. The measured consequence: safety margin degrades at
  **first** order in the effect bias — until the radius swallows the channel, past which the loss
  saturates at `U·|g|` — while performance regret degrades at second, because the envelope theorem
  protects an interior optimum and not a binding constraint. `safety_filter_benchmark` pays that argument in a closed loop on an
  unstable plant whose reference sits past the limit: the radius spent on the constraint never
  leaves the safe set, the same-sized budget spent the regret way violates on 93% of steps.
  Re-exported through `chc.sensitivity`.
- **`chc.plan.causal_plan`** — the one-call spine: a `CausalPlan` carrying the actions *together with*
  the certified error tube and the `certified_actions` prefix, so a caller cannot take the plan and
  leave the certificate behind. With no safety arguments it is exactly `projected_gradient_control`;
  each safety argument switches on one existing layer, and an uncertainty penalty without a support
  model is rejected rather than silently ignored.
- **`chc.plan.certify_safety`** — the other half of the spine: §40 evaluated along a finished plan, so
  the safety result has a consumer instead of staying an orphan primitive. Returns the certified
  *prefix* (does the action the planner chose still clear the barrier at this `Γ`?) next to the plan's
  `Γ*` (could **any** admissible action have, and up to which level?) — two questions that come apart
  in both directions and are diagnostic together. The plan-level `Γ*` is the minimum over steps and is
  *attained*: `Δ(Γ)` is increasing, so each step certifies on a down-set and the plan certifies on
  their intersection; a step that certifies at no `Γ` empties it and reports `nan` rather than being
  skipped. A separate call on a finished plan rather than a planner argument — it audits, it does not
  silently change the actions. Re-exported through `chc.sensitivity`.
- **`chc.spine`** (`scripts/spine_demo.py`) — the four layers on one decision, because until now every
  layer had its own demo and none of them ran end to end. Two zones of a mobile driver pool, one
  incentive lever whose `[+b, -b]` column *is* driver conservation, and a supply floor in the zone the
  lever drains: an effect fitted from confounded logs (naively and with the backdoor adjustment), a
  constrained plan with its Grönwall tube, a `certify_safety` audit of that plan, and finally the same
  plan executed on the **true** plant so the offline numbers can be checked against what happened. The
  confounded arm plans a cost of 13.6 and pays 38.5; `Γ*` separates the two arms *before either acts*
  (7.46 vs 1.18) with no access to ground truth. Deliberately control-affine — `certify_safety` reads
  the channel off the Jacobian at `u = 0`, which is exact for an affine plant and only a linearisation
  otherwise, so the softmax-equilibrium market of `chc.marketplace` is the wrong plant to certify.
- **`chc.reachability`** — a Hamilton–Jacobi backward reachable tube whose adversary is the §32
  identification radius: `V(x,T) = max_u min_{ΔB} min_s h(ξ(s))` by Lax–Friedrichs on a 2-D grid, with
  the same robust-margin algebra as `chc.barrier` but `p = ∇V` solved for rather than assumed. It
  exists to price the barrier certificate, and `barrier_reachability_gap` is that price:
  `valid_cbf` makes the CBF theorem executable (condition on all of `{h ≥ 0}` ⟹ the tube **is**
  `{h ≥ 0}`, checked to a cell), and `certified_but_unreachable` measures where the *pointwise* §40
  check certifies a state no controller can hold — on a relative-degree-2 barrier the §40 verdict is
  identical at every radius while the true tube shrinks. Verified against two analytic solutions (a
  rigidly sliding level set to `1e-5`; the double-integrator braking parabola to 100% off-boundary
  agreement), with a CFL guard that refuses rather than reporting an optimistic safe set.
- **Regret / guarantee line** — the orthogonal-to-control transfer theorem (order `p` → `2p`, scalar
  and multivariate-LQ), multi-channel network control (debias *every* channel), the adaptive
  information-exploration duality with its `√T` lower bound, the C2 end-to-end theorem with a clustered
  van-Trees lower bound and an exposure-map generalisation, plus a batch of scoped propositions and
  corollaries (doubly-robust control, H∞-as-pessimism, constrained piecewise-quadratic regret,
  confounded turnpike, transportability, ensemble heterogeneity, partial-identification sign threshold).

### Changed

- Eleven rounds of external review folded in as **scope and honesty corrections**, not new claims: the
  explicit-Euler contraction factor was wrong (`√(1+2μΔt+L²Δt²)`, sufficient step `Δt < 2c/L²`); the
  confounding effect error needed the control magnitude to be dimensionally right; the `§35`
  improvement gap is piecewise (the undershoot-dominant branch was unproved while the benchmark ran in
  it); `§32` is the bounded-density-ratio *marginal* special case of Tan's MSM, not the full model, and
  its monotonicity argument is feasible-set nesting; the confounded-marketplace benchmark is
  *observational*, not a randomised switchback. Several results were relabelled to their honest status
  (order-transfer *lemma*, local-not-global, scalar-not-universal, `≈`-not-`=`). Round eleven: the
  equilibrium conditioning of `1` is an *ambient*-norm statement saturated by the mass-conservation
  direction, and the binding number on the fixed-mass tangent space is strictly smaller; the safety
  margin loss is `U·min(d,|g|)` and so first-order only while control authority lasts; the zero-action
  rule in general reads "the identified interval contains zero", the symmetric ball being one case;
  and the `§38` premium is quoted with both normalisations because they differ by an order of
  magnitude.
- Documentation counts corrected after an audit found a silently drifting entry count in the research
  log; the README test count was stale by two releases.

### Fixed

- `ConfoundingRobustPenalty` used `‖u‖`, whose gradient is NaN at `u = 0` — exactly where the solver
  starts — so `0·NaN` poisoned every step and the control stayed pinned at zero. Now a smoothed
  `√(‖u‖²+ε)`, which zeroes the gradient at the origin and preserves the linear bound.
- The congestion equilibrium ran a fixed trip count and returned whatever point it reached. Outside
  the contraction region that is a 2-cycle, not an equilibrium: at `βc = 200` it returned a point with
  residual 3.21 on a mass of 6.0 and most of the mass on the *least* attractive zone, and no caller
  could tell. The solver now reports convergence; the root cause was the hard-coded damping `d = ½`,
  whose ceiling is `βc < 6`, so `congestion_damping` supplies the modulus-optimal `d*` that contracts
  for every `βc`. Both shipped users (`chc.marketplace`, causaldyn-bench Track H) sit at `βc = 5`,
  inside the certified region, and their docstrings now say so executably.
- The convergence tolerance was absolute while the residual carries the units of the state, so a
  float32 solve at the shipped mass could not reach `1e-8` and reported failure at a residual of
  3.8e-07. The residual is now relative (`‖x−T(x)‖ / max(1, ‖x‖)`).

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

[0.1.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.1.0
