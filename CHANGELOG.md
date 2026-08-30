# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to adhere to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) once the API stabilises (pre-1.0 it may
still change).

## [Unreleased]

### Added

- **`conjugate_time_certificate`** (`chc.regret`) — bounds the horizon on which every other constant
  in this module is valid. `confounded_turnpike_certificate` reads as "a long horizon is benign", and
  that is true only for a POSITIVE-DEFINITE stage cost; its hypothesis was never priced. Under an
  indefinite one -- a state that is rewarded rather than penalised, as a growth or market-share
  objective is -- the reverse-time Riccati solution is a uniform rotation of its phase and escapes at
  `t_conj = (pi/2 + phi0)/mu` with `mu = sqrt(-(a^2 + (b^2/r) q))`. Three objects blow up at three
  DIFFERENT orders there: the cost-to-go has a simple pole with residue `-r/b^2` (free of `a`, `q`
  and the terminal weight), the gain sensitivity `L_K` a DOUBLE one, and the regret constant
  `C ~ L_K^2` a fourth-order one -- so inspecting the cost alone understates the obstruction by two
  orders. And `d t_conj/db < 0`: more control authority moves the obstruction EARLIER. The certificate
  carries a positive-definite arm where the same algebra runs with `tanh` and nothing diverges.
  Derived in `validation/conjugate_time.mac`, machine-checked in `proofs/conjugate_time.v`.

- **`ce_explicit_constant_certificate`** (`chc.regret`) — computes the two constants every earlier C1
  statement took as a hypothesis. `proofs/c2_end_to_end.v` universally quantifies over an arbitrary
  `0 <= cc` in `regret <= cc*||dB||^2` and cites Mania-Tu-Recht's LOCAL quadratic bound; nothing
  computed the ball on which a certainty-equivalent gain stabilises the TRUE plant, and nothing
  computed `cc`. The lever is that the Lyapunov increment is an EXACT perfect square in the gain
  error, `Q + K'RK' + (A-BK')'P(A-BK') - P = (K'-K)'R_K(K'-K)` with `R_K = R + B'PB`, for every gain
  and with no smallness assumed -- so the Mania-Tu-Recht citation leaves the regret half entirely and
  stays only as a comparison point. Summing it along the perturbed loop gives
  `rho = theta/(2 beta_B L_K)` and `C = 2 kappa_P ||R_K|| L_K^2 ||x0||^2 / theta` with
  `theta = 1 - sqrt(1 - eta)`, `eta = lmin(Q + K'RK)/lmax(P)`. Inside `rho` the controller provably
  stabilises with the checkable Lyapunov certificate `(A-BKhat)'P(A-BKhat) <= (1-theta/2)^2 P`; the
  sweep runs past `rho` on purpose and shows the controller really does destabilise there. Derived in
  `validation/ce_explicit_constants.mac`, machine-checked in `proofs/ce_explicit_constants.v`.

- **`cluster_fold_leakage_certificate`** (`chc.regret`) — prices what a *violated* cross-fitting
  assumption costs on a clustered design, which none of the cited theorems do: CCDDHNR assume i.i.d.
  rows, Hansen–Lee assume independent cluster scores as a primitive, Chiang–Kato–Ma–Sasaki assume
  folds are already cluster-level. Splitting folds by row rather than by cluster is **not** a bias —
  Frisch–Waugh–Lovell cancels for any fold assignment — it silently substitutes a within-cluster
  estimator whose variance obeys `Psi = c(m,K)*(1 - rho_ICC)` with `c(m,2) = m*(m+14)/(m+2)^2`, so
  every constant derived from the cluster-robust variance belongs to a different estimator and a
  sandwich computed after row folds under-covers. For a cluster-*measurable* exposure — which is what
  a partial-interference spillover is — the channel is annihilated outright and the estimate loses
  exactly that coefficient. Derived in `validation/cluster_fold_leakage.mac`, machine-checked in
  `proofs/cluster_fold_leakage.v`.

- **`minimax_exploration_certificate`** (`chc.regret`) — the sequential exploration lower bound as an
  infimum over **policies**, not over a schedule class, with the constant written out:
  `c_causal = 2·A·|du*/dθ|·σ/√η_exp`. `adaptive_exploration_certificate` bounded an assumed
  `1/√t` family and carried its numerator as an opaque `K`; replacing the schedule template with the
  conditional-variance identity `E[(u_t − u*)²] = Var(u_t | F_{t−1}) + (E[u_t | F_{t−1}] − u*)²`,
  which holds for every policy, removes the template and names `K = A·(du*/dθ)²·σ²`. The bound is
  taken over a `T^{-1/4}` neighbourhood; a `T^{-1/2}` one carries prior information of order `T` and
  is vacuous. Two things follow from making it tight rather than merely valid: a front-loaded design
  **attains** the floor (to 4e-5), and the best `1/√t` taper sits at exactly `√2` above it, so the
  taper is optimal only when a per-round action cap forbids the burst. Derived in
  `validation/minimax_exploration.mac`, machine-checked in `proofs/minimax_exploration.v`.

### Fixed

- **`regret_scaling` and `interference_regret_certificate` silently conditioned on the stabilising
  event** (`chc.regret`); both now return an `infinite_fraction` array, so `RegretCurve` has a fourth
  field. Each `continue`d past draws where the perturbed model is unstabilisable, and dropped draws
  whose gain fails to stabilise the TRUE plant via an `np.isfinite` filter. Regret is `+inf` on that
  event, so `E[R]` does not exist and the reported `exponent` was a quantity conditional on the
  complement -- presented as if it were unconditional. The share is now reported rather than deleted.
  `ce_explicit_constant_certificate` gives the explicit radius inside which it is 0 by construction.

- **The C2 certificates did not implement assumption A8** (`chc.regret`), so the values returned by
  `multichannel_control_certificate`, `end_to_end_c2_certificate` and `clustered_lower_bound_certificate`
  have all moved. A8 asks for `K >= 2` folds of **whole clusters**; all three built folds as
  `np.mod(np.arange(n), 2)` — row parity — while the cluster id sat one line above, unused, leaving
  every cluster in both folds. `validation/clustered_rate_check.R` had the identical construction, so
  the independent-stack cross-check reproduced the same fold rather than catching it. Folds are now
  `np.mod(cid, 2)`. No order and no qualitative conclusion changed (the leak is not a bias); the
  measured constants did: full-orth slope 3.53 -> 3.82, cluster-SE -0.58 -> -0.55, G-sweep
  -0.89 -> -1.08, the `G^{-1}` plateau `c0` 0.18 -> 0.19, and the R cross-check -0.502 -> -0.541.

- **`adaptive_exploration_certificate` ignored its `sigma` argument** (`chc.regret`), so its returned
  `lower_bound`, `schedule` and cumulative-regret arrays are all different now. The van-Trees floor
  numerator is `K = C·σ²` with `C = A·(du*/db)²`; the function computed `C` under the name `coeff`
  and consumed it where `K` belongs, silently pinning `σ² = 1` while its signature advertised
  `sigma=0.5`. The cause was a name rather than a missing multiplication — the two quantities are
  spelled apart in `validation/adaptive_exploration.mac` but the Rocq file calls the numerator `C`,
  and the code followed the Rocq spelling while implementing the Maxima quantity. They are now named
  apart in the code too (`c_curv` vs `k_vt`). Nothing symbolic changed: the derivation and the proofs
  were already correct. Passing `sigma=1.0` reproduces the old output exactly. A test asserts the
  bound is linear in `σ`, since `ruff` does not flag an unused keyword argument.

## [0.3.0] — 2026-08-01

### Added

- **Named data enters as a mapping, a pandas frame or a polars frame** (`chc.frames`). The
  estimators (`chc.estimators`) and the g-methods (`chc.gmethods`) used to spread whatever they were
  handed with `{**data}` / `for name in data`. That reads a pandas frame correctly and a polars one
  **silently wrong**: `dict(frame)` and iteration both yield polars' columns as *values*, so code
  keying on column names received column data and nothing raised. `as_columns` now normalises once
  at the entry point and recognises a frame structurally — `.columns` plus `frame[name]` — so
  neither library becomes a dependency of the wheel. A mapping passes through **unconverted**, which
  keeps the library's own JAX arrays on the device and leaves the entry points usable under
  `jax.jit`; only the frame branch materialises, and it stops at NumPy so the caller keeps deciding
  precision.

- **`ControlAffineResidual.closed_loop_jacobian(x, u)`** (`chc.residual`) — `∂(a_θ + B_θ u)/∂x`, the
  linearisation an MPC horizon actually follows. `drift_jacobian` is the drift at `u = 0` and its
  docstring claimed a non-negative eigenvalue meant "the plant runs away on its own"; that holds only
  at `degree = 0`. With a state-dependent channel the fitted class is closed under an affine change
  of actuator coordinates `u = alpha v + beta` with `a -> a + beta b1`, so the drift spectrum reports
  where the actuator's units put their zero rather than whether the plant decays. Found on BOPTEST: a
  zone whose actuator is a setpoint in `[15, 25] °C` read `+6.42` from the drift and `-1.40` from the
  closed loop at the setpoint it actually held. The claim is corrected; the stability question to ask
  is `sup` over the admissible action set, which for an affine decay is attained at a box endpoint.
  Derivation and cross-checks in `validation/actuator_reparametrisation.{mac,py}`.

- **Causal identification of the residual's control channel** (`chc.dynamics_id`). Until now every
  residual in the library was fitted by prediction error, which under a confounded logging policy
  learns the **observational** control response — measured, the trained channel is `0.02` where the
  truth is `1.0`, and the planner inherits that. `fit_causal_residual` estimates it instead from the
  Robinson moment lifted from a scalar effect to a state-dependent matrix, with cross-fitting:
  channel error `0.002`. Control payoff on the same plant: regret `0.014` against the
  prediction-error fit's `6.41`. When the confounder is never logged, a 2SLS variant identifies the
  channel from an exogenous action shifter instead — consistent, but at a **variance premium** worth
  stating rather than burying: `0.10` channel error and `0.13` regret, because the shifter explains
  only ~18% of the action's variance.
  `solve_channel_moment` exposes the moment alone, so `g` and `m` may come from any learner rather
  than only the built-in ridge-polynomial default. With neither an adjustment set nor an instrument
  the fit reports `identified=False` and points at `chc.sensitivity` instead of returning a confident
  wrong answer.
- **`ControlAffineResidual`** (`chc.residual`) — `r_θ(x, u) = a_θ(x) + B_θ(x) u`, the plant class
  `chc.plan.certify_safety` and `chc.spine` already assume. Restricting identification to it puts the
  identification and safety layers on the same object: `control_channel(x)` is bit-identical to
  `∂r/∂u`, which is what `certify_safety` differentiates out. A general nonlinear residual has no
  partialling-out moment and gets no guarantee from this estimator.
- **`CausalDynamicsTask`** (`chc.benchmark`) — the leaderboard row, with `oracle` / `causal-id` /
  `causal-iv` / `mse-id`. Its honest trap is that the failure is **silent on the safety columns**: the
  attenuated channel makes the biased planner under-actuate (`max|u|` 0.22 against the oracle's 2.49),
  so it never touches the box or leaves the logged action support and reports `viol = ood = 0` while
  conceding most of the achievable improvement. Regret is the only column that sees it.
- **Two ways to read the half of the fit that is *not* identified causally.**
  `ControlAffineResidual.drift_jacobian(x)` returns `∂a_θ/∂x`, the companion to the existing
  `control_channel(x)`, and `CausalDynamicsFit.drift_error` carries the drift stage's own
  homoskedastic OLS scale beside `channel_error`. Both exist because of a failure on a real building
  emulator (`causaldyn-bench` Track D-causal), where three of four arms came back with a **positive**
  thermal pole: a trending outdoor temperature, present in `adjust_for` but absent from the drift's
  own regressors, gets charged to positive feedback in the state, and the fitted plant heats itself.
  An MPC plans on `a_θ(x) + B_θ(x) u` and only the second term is interventional, so a drift with a
  non-negative eigenvalue describes a plant that runs away on its own and no channel accuracy
  rescues the horizon. `drift_error` is deliberately a *different* object from `channel_error` —
  conditional on the fitted channel, whose uncertainty it does not propagate, and homoskedastic: a
  scale, not coverage.

### Changed

- **`CausalPlan` no longer certifies a plan it never checked.** With no error model supplied
  (`model_error` left at its default) the Gronwall tube is identically zero, and the old code reported
  that as `certified_horizon == horizon` — a full-horizon safety pass obtained by proving nothing.
  `uncertainty_tube` and `certified_horizon` are now `None` in that case, a new
  `CausalPlan.certificate_status` property distinguishes `"not_evaluated"` from `"uncertified"` /
  `"partial"` / `"certified"`, and `certified_actions` raises rather than handing back the whole
  sequence via a `None` slice. **Breaking** for any caller that indexed those two fields
  unconditionally. `causal_plan` also rejects a negative `model_error`, which is not an error budget.
- **The three safety modes are named apart** in `chc.plan`'s documentation and in the README, because
  the word "safety" was doing the work of all three: **plan** (`causal_plan` — box constraints in the
  solve), **audit** (`certify_safety` — read-only on a finished plan), **filter**
  (`robust_safety_filter` — the only one that changes an action). Stated plainly: no barrier or tube
  enters the objective, so `causal_plan` can return a plan that fails its own audit. A
  state-constrained (CBF-QP) solve does not exist yet.
- CI gates the formal claims too: a `proofs` job compiles all 37 `proofs/*.v` under Rocq 9.2, pinned
  because the proofs use the post-rename `From Stdlib Require Import`.

### Fixed

- **The nuisance stage is now scale-invariant, and was silently not.** `_cross_fit_residuals` built
  its degree-2 polynomial basis on the caller's raw covariates, so `ridge=1e-6` penalised a column of
  order 1 and its square of order 400 by the same absolute amount, and the Gram inherited whatever
  units the caller happened to log in. Found on a real building emulator (`causaldyn-bench` Track
  D-causal), where the zone enters in Celsius at ~21 beside already-standardised weather: condition
  number **1.4e11**, past what float32 carries, and the fit returned `nan` — while the same rows in
  float64 fitted fine, which is what identified it as conditioning rather than data. Standardising
  the covariates first drops the condition number to order 10 — on a reproducible stand-in with that
  geometry (768 rows, zone at 21 ± 0.9 beside three standardised columns) the degree-2 Gram goes
  from `2.7e10` to `2.4e1` — makes float32 reproduce float64 to 4 decimals, and makes the estimate
  *exactly* invariant to the units and origin of the adjustment set. Two things this exposed are
  worth stating: reverting the fix, an offset of `1e5` returns a **finite** channel wrong by 1.34
  against a baseline error of 0.009 — silently wrong is the worse failure — and `tests/conftest.py`
  enables `jax_enable_x64`, so **no test in this suite can fail from float32 conditioning**, which is
  how the defect survived all of them.
- `fit_causal_residual`'s `folds` documentation no longer claims that own-sample residualisation
  reintroduces bias. Measured across ten regimes, `folds=1` is never worse than cross-fitting and at
  small `N` is 5–10× better: residualising `y` and `u` by the same linear projection whose span
  contains both nuisances is Frisch–Waugh–Lovell, so it is exactly unbiased and out-of-fold prediction
  only adds variance. Cross-fitting is insurance against nuisance learners that are adaptive to the
  sample or saturated (1-NN including the row itself sends the channel to exactly zero), and the
  tests now assert that case rather than the false one.
- `HybridDynamics`'s docstring said causal design "will live" in the residual's feature map. It does
  now, in `chc.dynamics_id`, and the docstring points there.

## [0.2.0] — 2026-07-29

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

The repository moved to the `causaldyn` organisation and is public as of this release; all project
URLs point there. This is the **first version published to PyPI** — `0.1.0` below was prepared and
tagged locally, then overtaken by 71 commits before it was ever pushed, so it never reached an index.

## [0.1.0] — 2026-07-19

Prepared but never published; superseded by `0.2.0`. `chc` is a small JAX library that fuses
**physics-structured hybrid dynamics**, **causal identification**, and **constrained optimal control**,
made safe on offline/confounded data by an explicit **pessimism/support** layer — evaluating decisions
as interventions, not correlations.

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

[0.3.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.3.0
[0.2.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.2.0
[0.1.0]: https://github.com/causaldyn/causal-hybrid-control/releases/tag/v0.1.0
