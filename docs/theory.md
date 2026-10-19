# Theory

The papers that develop the results behind the library are not public yet, and this page will link
them once they are. What is public, and ships in every source distribution, is the formal and
symbolic layer behind them: Rocq proofs of their algebraic steps and Maxima derivations of their
closed forms. A docstring that rests on a proof or a derivation names its file, and the file is in
one of the two directories below.

A proof checks the algebra of a statement over the reals, not the Python that computes it. What a
proof covers is its own statement, which can be narrower than the docstring that cites it: a
scalar case of a vector claim, the first few horizons of a recursion, or a closed form at a
quantity the code lets move. Each docstring that cites one says which. Each citation is tied to a
test of the float code that implements its statement; [the tests that tie
them](#the-tests-that-tie-the-statements-to-the-code) are listed below.

## Formal proofs — [`proofs/`](https://github.com/causaldyn/causal-hybrid-control/tree/main/proofs)

The algebraic cores of the control and guarantee invariants are proved in Rocq, from the
box-projection bounds and idempotence behind `project_box`
([`box_projection.v`](https://github.com/causaldyn/causal-hybrid-control/blob/main/proofs/box_projection.v))
to the interference-aware regret certificate
([`interference_regret.v`](https://github.com/causaldyn/causal-hybrid-control/blob/main/proofs/interference_regret.v)).
Where Stdlib's lack of matrices limited a proof to its scalar or 2×2 shadow, the matrix statement is
proved separately in
[`proofs/mathcomp/`](https://github.com/causaldyn/causal-hybrid-control/tree/main/proofs/mathcomp)
with MathComp, at any dimension: the robust barrier margin over an operator or Frobenius ball, the
multivariate van Trees regret identity, and the explicit constant of the certainty-equivalence
regret bound.

Each file's header says what it proves and what it takes as given, and several prove only the
algebraic core of a statement whose analytic half is derived in `validation/` and checked
numerically. [The certificates' scopes](concepts/certificates.md#scopes) say which file stands
behind which certificate. Two things bound what "proved" means here:

- **Every lemma under `proofs/` is stated over Rocq's classical reals**, which are themselves
  axiomatic. The gate is therefore not "closed under the global context", which no such lemma can
  be; it is that nothing *outside* Stdlib's own classical-reals axioms gets in — no project
  `Axiom`, no admitted step, no `Hypothesis` leaking out of its `Section`. The MathComp lifts are
  stated over abstract fields instead, so for them "closed under the global context" is the gate.
- **A published statistical result a proof relies on is an input, not a theorem.** It enters the
  theorem's type as a named predicate carrying its citation, rather than as an axiom, so
  `Check <theorem>` shows the cited names and `Print <Name>` what was assumed under each. The
  predicate holds what the algebra takes from the citation, which can be much less than the
  result. In `c2_end_to_end.v` the regret is defined as the right-hand side of Mania, Tu and
  Recht's bound, so the cited predicate asks only that the curvature be nonnegative; in
  `clustered_van_trees.v` the van Trees floor and the lower Lipschitz regret map are the premises
  themselves. Such a theorem shows how the cited bounds combine, not that they hold.

```bash
just proofs                # compile every proof (CI: every push, Rocq 9.2, then the audit below)
just assumptions           # Print Assumptions on every lemma; fails on anything beyond Stdlib's axioms
just proofs-mathcomp       # compile the MathComp lifts (MathComp 2.6, see CONTRIBUTING.md)
just assumptions-mathcomp  # fails unless every lift is closed under the global context
```

## Symbolic derivations — [`validation/`](https://github.com/causaldyn/causal-hybrid-control/tree/main/validation)

Correctness is cross-checked in independent tools, symbolic first. The derivations are Maxima,
which is authoritative; the Riccati equation and the matrix exponential are verified in exact and
high-precision arithmetic against PARI/GP and Octave, with SciPy used only as the fast float64
numeric. Where a bound is a finite first-order statement, z3 and cvc5 check its negation is
unsatisfiable.

```bash
just derivations   # run every Maxima derivation; fails if any did not complete
```

```bash
just crosschecks   # the SMT verdicts against the ones each file states; PARI/GP and Octave complete
```

CI runs both on every push. It checks that each derivation *completes*, not that its output matches
a stored copy, and the same of the PARI/GP and Octave files (`*.gp`, `*.m`); each SMT file
(`*.smt2`) states its verdicts, and z3's must match them. The FriCAS, R and Python files beside
them (`*.input`, `*.R`, `*.py`) are run by hand, and CI would not notice one of them breaking.

## The tests that tie the statements to the code

Every docstring in `src/chc` that cites a file under `proofs/` or `validation/`, or a Rocq lemma
by name, is tied to a test of the float code. The tests are in `tests/test_formal_*.py`. The
`FORMAL_LINKS` literal at the top of each file maps each cited object to the files it cites and
to the tests that check the cited statement. `tests/test_formal_links.py` reads the docstrings
and fails when a citation has no entry, when an entry lists other files than its docstring cites,
or when a listed test does not exist. A new citation therefore needs a test.

A test checks the statement with random inputs inside the lemma's hypotheses, and by a route
other than the code's own where one exists: a brute-force minimum, a series, a quadrature, an
exact enumeration, or a Monte Carlo risk within several of its standard errors. A test samples the
hypotheses' region. It is not a proof about the code.

Where the code does not satisfy the statement, the test is the counterexample, marked as an
expected failure (`xfail(strict=True)`) with its numbers. These are the statements that do not
hold as the code computes them (names relative to `chc`):

- `regret._self_consistent_mass`: its fixed-point iteration can oscillate and return a mass that
  does not balance, at a tight cap or after rounds with no authority.
- `regret.capped_exploration_policy`: the block at the cap is the best whole block. The optimum of
  the convex program has one partial last round and is lower.
- `regret.conjugate_time_certificate`: the definite arm returns `nan` once the terminal weight
  leaves the range of its inverse hyperbolic tangent.
- `regret.constrained_ce_regret_certificate`: `pessimism_budget` is not the naive controller's
  regret when the true optimum is past the bound; it is 45 % lower at the defaults.
- `regret.hinf_robust_regret_certificate`: where b² > r, the robust gain lies above the
  certainty-equivalent gain, not below it, and the certainty-equivalent regret can fall below the
  calibrated minimum.
- `regret.information_lower_bound_certificate`: raises at r = b², where its floor is zero.
- `regret.minimax_action` and `regret.minimax_lq_policy`: the kink rule and the effect that
  realises the value assume a nonnegative effect interval.
- `regret.transportability_regret_certificate`: the W-DRO bound covers the quadratic proxy that
  the proof bounds, not the exact regret, which is 39 % above it at the radius.
- `delay.delay_design_loss`: the over-estimate's two-term series misses the exact root by
  4·10⁻³, not 10⁻³, at ε = 0.05.
- `deep_galerkin.dual_weighted_error_estimate`: exact for an affine model only where the model's
  curvature is the Riccati root.

| Proof or derivation | Code it is tested against | Tests |
|---|---|---|
| `action_van_trees.mac` | `regret.information_lower_bound_certificate` | `test_formal_regret_b.py` |
| `adaptive_exploration.mac` | `regret.adaptive_exploration_certificate` | `test_formal_regret_a.py` |
| `adaptive_exploration.v` | `regret.adaptive_exploration_certificate` | `test_formal_regret_a.py` |
| `allocation_decision_weight.mac` | `allocation.decision_weight` | `test_formal_control.py` |
| `bandit_causal.mac` | `regret.bandit_causal_certificate` | `test_formal_regret_a.py` |
| `bandit_causal.v` | `regret.bandit_causal_certificate` | `test_formal_regret_a.py` |
| `barrier_feasibility.mac` | `barrier` | `test_formal_control.py` |
| `barrier_feasibility.v` | `barrier` | `test_formal_control.py` |
| `barrier_feasibility_lift.v` | `sensitivity` | `test_formal_control.py` |
| `c2_end_to_end.mac` | `regret.end_to_end_c2_certificate` | `test_formal_regret_b.py` |
| `c2_end_to_end.v` | `regret.ce_explicit_constant_certificate`, `regret.end_to_end_c2_certificate` | `test_formal_regret_a.py, test_formal_regret_b.py` |
| `capped_exploration.v` | `regret.capped_exploration_policy` | `test_formal_regret_a.py` |
| `capped_exploration_o1.gp` | `regret.capped_exploration_policy` | `test_formal_regret_a.py` |
| `capped_exploration_schedule.mac` | `regret.CappedExplorationPolicy`, `regret._self_consistent_mass`, `regret.capped_exploration_policy` | `test_formal_regret_a.py` |
| `capped_exploration_schedule.v` | `regret.capped_exploration_policy` | `test_formal_regret_a.py` |
| `causal_mpc.mac` | `regret.causal_vs_predictive_certificate` | `test_formal_regret_a.py` |
| `causal_mpc.v` | `regret.causal_vs_predictive_certificate` | `test_formal_regret_a.py` |
| `causal_pathway.mac` | `pathway` | `test_formal_estimation.py` |
| `causal_pathway.v` | `pathway` | `test_formal_estimation.py` |
| `ce_explicit_constants.mac` | `regret.ce_explicit_constant_certificate` | `test_formal_regret_a.py` |
| `ce_explicit_constants.v` | `regret.ce_explicit_constant_certificate`, `regret.regret_scaling` | `test_formal_regret_a.py, test_formal_regret_c.py` |
| `closed_loop_attribution.v` | `dynamics_id.closed_loop_gain_attribution` | `test_formal_estimation.py` |
| `cluster_fold_leakage.mac` | `regret.cluster_fold_leakage_certificate` | `test_formal_regret_a.py` |
| `cluster_fold_leakage.v` | `regret.cluster_fold_leakage_certificate` | `test_formal_regret_a.py` |
| `clustered_van_trees.v` | `regret.clustered_lower_bound_certificate` | `test_formal_regret_a.py` |
| `composition_transfer.mac` | `regret.composition_transfer_certificate` | `test_formal_regret_a.py` |
| `composition_transfer.v` | `regret.composition_transfer_certificate`, `regret.confounding_robust_lq_regret_matrix`, `regret.multivariate_transfer_certificate` | `test_formal_regret_a.py, test_formal_regret_c.py` |
| `confounded_turnpike.mac` | `regret.confounded_turnpike_certificate` | `test_formal_regret_a.py` |
| `confounded_turnpike.v` | `regret.confounded_turnpike_certificate` | `test_formal_regret_a.py` |
| `confounding_lq_regret.mac` | `regret.lq_regret_sensitivity` | `test_formal_regret_b.py` |
| `confounding_lq_regret.v` | `regret.confounding_robust_lq_regret` | `test_formal_regret_a.py` |
| `confounding_robust_control.mac` | `regret.confounding_robust_control` | `test_formal_regret_a.py` |
| `confounding_robust_control.v` | `regret.asymmetric_control_improvement`, `regret.confounding_robust_control` | `test_formal_regret_a.py` |
| `confounding_robust_cvar.mac` | `uncertainty.confounding_robust_inflation` | `test_formal_control.py` |
| `confounding_robust_cvar.v` | `uncertainty.confounding_robust_inflation`, `uncertainty.confounding_robust_radius` | `test_formal_control.py` |
| `congestion_contraction.mac` | `games.congestion_contraction_modulus` | `test_formal_fields.py` |
| `congestion_contraction.v` | `games.congestion_contraction_modulus` | `test_formal_fields.py` |
| `conjugate_time.mac` | `regret.conjugate_time_certificate` | `test_formal_regret_a.py` |
| `conjugate_time.v` | `regret.conjugate_time_certificate` | `test_formal_regret_a.py` |
| `constrained_ce_regret.mac` | `regret.constrained_ce_regret_certificate` | `test_formal_regret_a.py` |
| `constrained_ce_regret.v` | `regret.constrained_ce_regret_certificate` | `test_formal_regret_a.py` |
| `constrained_plan_regret.mac` | `plan.PlanRegretBound`, `plan.plan_regret_bound` | `test_formal_control.py` |
| `constrained_plan_regret.v` | `plan.plan_regret_bound` | `test_formal_control.py` |
| `contractive_euler.mac` | `uncertainty.contractive_rollout_bound` | `test_formal_control.py` |
| `contractive_euler.v` | `residual.ContractiveResidual`, `uncertainty.contractive_rollout_bound` | `test_formal_control.py, test_formal_fields.py` |
| `convection_diffusion.mac` | `galerkin`, `galerkin.convection_diffusion_certificate`, `galerkin.convection_diffusion_exact`, `galerkin.optimal_upwind` | `test_formal_fields.py` |
| `convection_diffusion.v` | `galerkin`, `galerkin.convection_diffusion_certificate` | `test_formal_fields.py` |
| `convex_port_hamiltonian.mac` | `residual.PortHamiltonianResidual`, `residual._ConvexEnergy`, `residual.convex_energy_certificate` | `test_formal_fields.py` |
| `convex_port_hamiltonian.v` | `residual._critical_points`, `residual.convex_energy_certificate` | `test_formal_fields.py` |
| `delay_ball.mac` | `delay.delay_design_loss` | `test_formal_estimation.py` |
| `delay_estimate_bias.mac` | `irf.peak_lag` | `test_formal_estimation.py` |
| `delay_margin.mac` | `delay.delay_margin` | `test_formal_estimation.py` |
| `delayed_exposure_gate.mac` | `regret.delayed_network_certificate` | `test_formal_regret_b.py` |
| `delayed_network_exposure.mac` | `regret.delayed_network_certificate` | `test_formal_regret_b.py` |
| `delayed_network_exposure.v` | `regret.delayed_network_certificate` | `test_formal_regret_b.py` |
| `dither_channel_move.mac` | `gate.ChannelMove.price`, `gate.channel_move` | `test_formal_estimation.py` |
| `dither_channel_move.v` | `gate.ChannelMove.price` | `test_formal_estimation.py` |
| `dither_drift_evalue.mac` | `gate.channel_drift_evalues` | `test_formal_estimation.py` |
| `doubly_robust.mac` | `regret.doubly_robust_control_certificate` | `test_formal_regret_b.py` |
| `doubly_robust.v` | `regret.doubly_robust_control_certificate` | `test_formal_regret_b.py` |
| `dynamic_causal_mpc.mac` | `regret.dynamic_causal_regret_certificate` | `test_formal_regret_b.py` |
| `dynamic_causal_mpc.v` | `regret.dynamic_causal_regret_certificate` | `test_formal_regret_b.py` |
| `ensemble_control.mac` | `regret.ensemble_control_certificate` | `test_formal_regret_b.py` |
| `ensemble_control.v` | `regret.ensemble_control_certificate` | `test_formal_regret_b.py` |
| `envelope_nonconvexity.mac` | `response.Saturation.nonconvexity` | `test_formal_estimation.py` |
| `equilibrium_monotonicity.mac` | `games.equilibrium_monotonicity_certificate` | `test_formal_fields.py` |
| `equilibrium_monotonicity.v` | `games.equilibrium_monotonicity_certificate` | `test_formal_fields.py` |
| `experiment_bayes_action.mac` | `experiment` | `test_formal_estimation.py` |
| `exposure_map_c2.mac` | `regret.exposure_map_certificate` | `test_formal_regret_b.py` |
| `exposure_map_c2.v` | `regret.exposure_map_certificate` | `test_formal_regret_b.py` |
| `finite_population_gap.mac` | `deep_galerkin.MeanFieldSolution.finite_population_rms`, `deep_galerkin.finite_population_gap_certificate` | `test_formal_fields.py` |
| `fold_cut_duality.v` | `regret.fold_exactness_certificate` | `test_formal_regret_c.py` |
| `fold_shifted_bound.mac` | `regret._shifted_kyfan_bound` | `test_formal_regret_c.py` |
| `fold_spectrum_law.mac` | `regret.optimal_fold_partition` | `test_formal_regret_c.py` |
| `fold_spectrum_law.v` | `regret.optimal_fold_partition` | `test_formal_regret_c.py` |
| `gate_block_contrast.mac` | `gate` | `test_formal_estimation.py` |
| `general_q_ratio_moment.mac` | `regret._sandwich_plan`, `regret.exact_matrix_ratio_moment` | `test_formal_regret_b.py` |
| `general_q_ratio_moment.v` | `regret.exact_matrix_ratio_moment` | `test_formal_regret_b.py` |
| `geo_dlm.mac` | `dlm` | `test_formal_estimation.py` |
| `highprob_regret.mac` | `regret.highprob_regret_certificate` | `test_formal_regret_b.py` |
| `highprob_regret.v` | `regret.highprob_regret_certificate` | `test_formal_regret_b.py` |
| `hinf_robust_regret.mac` | `regret.hinf_robust_regret_certificate` | `test_formal_regret_b.py` |
| `hinf_robust_regret.v` | `regret.hinf_robust_regret_certificate` | `test_formal_regret_b.py` |
| `information_lower_bound.mac` | `regret.information_lower_bound_certificate` | `test_formal_regret_b.py` |
| `information_lower_bound.v` | `regret.information_lower_bound_certificate` | `test_formal_regret_b.py` |
| `interference_convexity.mac` | `regret.interference_convexity_certificate` | `test_formal_regret_b.py` |
| `interference_convexity.v` | `regret.interference_convexity_certificate` | `test_formal_regret_b.py` |
| `interference_orthogonal.mac` | `regret.interference_orthogonal_certificate` | `test_formal_regret_b.py` |
| `interference_regret.v` | `regret.interference_orthogonal_certificate`, `regret.interference_regret_certificate` | `test_formal_regret_b.py` |
| `lipschitz_rollout.mac` | `uncertainty.lipschitz_rollout_bound` | `test_formal_control.py` |
| `lipschitz_rollout.v` | `uncertainty.confounding_robust_closed_loop_bound`, `uncertainty.constraint_margin`, `uncertainty.contractive_rollout_bound`, `uncertainty.lipschitz_rollout_bound`, `uncertainty.time_varying_rollout_bound` | `test_formal_control.py` |
| `lq_mean_field.mac` | `deep_galerkin` | `test_formal_fields.py` |
| `lq_mean_field.v` | `deep_galerkin`, `deep_galerkin.LQMeanFieldGame.obstruction_horizon` | `test_formal_fields.py` |
| `matrix_ratio_moment.v` | `regret.exact_matrix_ratio_moment` | `test_formal_regret_b.py` |
| `mean_field_dwr.mac` | `deep_galerkin.dual_weighted_error_estimate` | `test_formal_fields.py` |
| `mean_field_dwr.v` | `deep_galerkin.dual_weighted_error_estimate`, `deep_galerkin.residual_blindness_sweep` | `test_formal_fields.py` |
| `minimax_exploration.mac` | `regret.minimax_exploration_certificate` | `test_formal_regret_b.py` |
| `minimax_exploration.v` | `regret.minimax_exploration_certificate` | `test_formal_regret_b.py` |
| `minimax_lq.mac` | `regret.minimax_action`, `regret.minimax_lq_policy` | `test_formal_regret_b.py` |
| `minimax_lq.v` | `regret.minimax_action` | `test_formal_regret_b.py` |
| `minimax_regret_move.mac` | `regret.minimax_regret_move` | `test_formal_regret_b.py` |
| `multichannel_control.mac` | `regret.multichannel_control_certificate` | `test_formal_regret_c.py` |
| `multichannel_control.v` | `regret.multichannel_control_certificate`, `regret.multivariate_interference_certificate` | `test_formal_regret_c.py` |
| `multivariate_van_trees.mac` | `regret.multivariate_action_floor`, `regret.multivariate_van_trees_certificate` | `test_formal_regret_c.py` |
| `multivariate_van_trees.v` | `regret.multivariate_action_floor`, `regret.multivariate_van_trees_certificate` | `test_formal_regret_c.py` |
| `multivariate_van_trees_lift.v` | `regret.multivariate_action_floor` | `test_formal_regret_c.py` |
| `nonlinear_dwr.mac` | `deep_galerkin.CongestedMeanFieldGame`, `deep_galerkin.adjoint_weighted_error` | `test_formal_fields.py` |
| `nonlinear_regret.mac` | `regret.nonlinear_regret_certificate` | `test_formal_regret_c.py` |
| `nonlinear_regret.v` | `regret.finite_horizon_pl_certificate`, `regret.nonlinear_regret_certificate`, `regret.strong_convexity_regret_bound` | `test_formal_regret_b.py, test_formal_regret_c.py` |
| `omega_jensen_gap.mac` | `regret.exact_ratio_moment` | `test_formal_regret_b.py` |
| `omega_jensen_gap.v` | `regret.exact_ratio_moment` | `test_formal_regret_b.py` |
| `omitted_confounder_bound.mac` | `dynamics_id.omitted_confounder_bound` | `test_formal_estimation.py` |
| `optimal_exploration.mac` | `regret.optimal_exploration_certificate` | `test_formal_regret_c.py` |
| `optimal_exploration.v` | `regret.optimal_exploration_certificate` | `test_formal_regret_c.py` |
| `order_transfer_window.mac` | `regret.composition_transfer_certificate` | `test_formal_regret_a.py` |
| `orthogonal_control.mac` | `regret.orthogonal_control_certificate` | `test_formal_regret_c.py` |
| `orthogonal_control.v` | `regret.interference_orthogonal_certificate`, `regret.orthogonal_control_certificate` | `test_formal_regret_b.py, test_formal_regret_c.py` |
| `panel_estimator_gate.mac` | `regret._psi_ratio_closed_form`, `regret.panel_estimator_certificate` | `test_formal_regret_c.py` |
| `partial_id_control.mac` | `regret.partial_id_control_certificate` | `test_formal_regret_c.py` |
| `partial_id_control.v` | `regret.partial_id_control_certificate` | `test_formal_regret_c.py` |
| `pessimistic_optimality.mac` | `regret.pessimism_variance_certificate` | `test_formal_regret_c.py` |
| `pessimistic_optimality.v` | `regret.pessimism_variance_certificate` | `test_formal_regret_c.py` |
| `plan_decision_weight.mac` | `plan.CausalPlan.decision_weight` | `test_formal_control.py` |
| `planned_cost_bias.mac` | `plan.CausalPlan.planned_cost_bias` | `test_formal_control.py` |
| `port_hamiltonian_lyapunov.v` | `residual.damping_injection_certificate` | `test_formal_fields.py` |
| `predictive_tube.mac` | `plan.causal_plan`, `uncertainty._noise_tube` | `test_formal_control.py` |
| `response_curves.mac` | `response` | `test_formal_estimation.py` |
| `rk4_rollout.mac` | `uncertainty.linear_rollout_bound`, `uncertainty.time_varying_rollout_bound` | `test_formal_control.py` |
| `space_time_folds.mac` | `regret.space_time_fold_certificate` | `test_formal_regret_c.py` |
| `space_time_folds.v` | `regret.space_time_fold_certificate` | `test_formal_regret_c.py` |
| `spectral_circulant.mac` | `residual.SpectralResidual`, `toeplitz.circulant_symbol`, `transport.advection_diffusion_symbol` | `test_formal_fields.py` |
| `spectral_circulant.v` | `residual.SpectralResidual`, `toeplitz.circulant_operator_norm` | `test_formal_fields.py` |
| `switchback_randomisation.mac` | `switchback.randomisation_interval` | `test_formal_estimation.py` |
| `transportability_regret.mac` | `regret.transportability_regret_certificate` | `test_formal_regret_c.py` |
| `transportability_regret.v` | `regret.transportability_regret_certificate` | `test_formal_regret_c.py` |
| `tube_levels.mac` | `dynamics_id.ChannelRegion.stability_level`, `uncertainty.gaussian_mixing` | `test_formal_control.py, test_formal_estimation.py` |
| `van_trees.mac` | `regret.van_trees_certificate` | `test_formal_regret_c.py` |
| `van_trees.v` | `regret.adaptive_exploration_certificate`, `regret.van_trees_certificate` | `test_formal_regret_a.py, test_formal_regret_c.py` |
| `zone_market.mac` | `zones.ZoneMarketSystem.zone_decision` | `test_formal_estimation.py` |

## Result numbers in docstrings

Docstrings refer to results by number — `§40`, `Result 55` — and those numbers index a research
log that is not public. The proof and derivation files they cite are.
