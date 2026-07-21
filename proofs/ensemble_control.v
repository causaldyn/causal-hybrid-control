(* Rocq: the ENSEMBLE (heterogeneity) control regret floor -- the causal-control analogue of ensemble
   control (Li & Khaneja, "Control of inhomogeneous quantum ensembles", PRA 73 030302, 2006; derived in
   validation/ensemble_control.mac). When the causal effect b varies across a population (CATE
   heterogeneity), a single control u must serve every context. Even with PERFECT per-context knowledge,
   one control pays an irreducible regret equal to the curvature-weighted VARIANCE of the per-context
   optimal actions: R(u_ens) = w1*w2*(u1-u2)^2/(w1+w2). A homogeneous population (u1 = u2) has a zero
   floor -- heterogeneity, not estimation error, is what forces it. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* ensemble regret of a single control u over a two-context population (weights w1,w2, optima u1,u2) *)
Definition ens_regret (w1 w2 u1 u2 u : R) : R := w1 * (u - u1) ^ 2 + w2 * (u - u2) ^ 2.
Definition u_ens (w1 w2 u1 u2 : R) : R := (w1 * u1 + w2 * u2) / (w1 + w2).  (* curvature-weighted mean *)

(* The curvature-weighted mean is the ensemble-optimal single control: the regret gap over it is the
   sum-of-squares (w1+w2)*(u - u_ens)^2 >= 0. *)
Theorem ensemble_optimum : forall w1 w2 u1 u2 u,
  0 < w1 + w2 -> ens_regret w1 w2 u1 u2 (u_ens w1 w2 u1 u2) <= ens_regret w1 w2 u1 u2 u.
Proof.
  intros w1 w2 u1 u2 u Hw. unfold ens_regret, u_ens.
  assert (Hne : w1 + w2 <> 0) by lra.
  assert (Hid :
    (w1 * (u - u1) ^ 2 + w2 * (u - u2) ^ 2)
    - (w1 * ((w1 * u1 + w2 * u2) / (w1 + w2) - u1) ^ 2
       + w2 * ((w1 * u1 + w2 * u2) / (w1 + w2) - u2) ^ 2)
    = (w1 + w2) * (u - (w1 * u1 + w2 * u2) / (w1 + w2)) ^ 2) by (field; exact Hne).
  assert (Hsq : 0 <= (w1 + w2) * (u - (w1 * u1 + w2 * u2) / (w1 + w2)) ^ 2)
    by (apply Rmult_le_pos; [lra | apply pow2_ge_0]).
  lra.
Qed.

(* The heterogeneity floor: the ensemble-optimal control's regret is the curvature-weighted variance of
   the per-context optimal actions. *)
Theorem heterogeneity_floor : forall w1 w2 u1 u2,
  w1 + w2 <> 0 ->
  ens_regret w1 w2 u1 u2 (u_ens w1 w2 u1 u2) = w1 * w2 * (u1 - u2) ^ 2 / (w1 + w2).
Proof.
  intros w1 w2 u1 u2 Hne. unfold ens_regret, u_ens. field. exact Hne.
Qed.

(* A homogeneous population (u1 = u2) has a zero floor: one control serves everyone. Heterogeneity, not
   estimation error, is what forces the ensemble floor. *)
Theorem homogeneous_zero_floor : forall w1 w2 u1 u2,
  w1 + w2 <> 0 -> u1 = u2 -> ens_regret w1 w2 u1 u2 (u_ens w1 w2 u1 u2) = 0.
Proof.
  intros w1 w2 u1 u2 Hne Heq. rewrite heterogeneity_floor by exact Hne.
  subst u2. assert (H0 : (u1 - u1) ^ 2 = 0) by ring. rewrite H0. field. exact Hne.
Qed.
