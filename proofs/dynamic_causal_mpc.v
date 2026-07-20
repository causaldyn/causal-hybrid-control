(* Rocq: the DYNAMIC (horizon) confounding theorem -- the multi-step companion of proofs/causal_mpc.v.
   A setpoint-tracking controller x' = a*x + b*u using a confounded effect estimate b_obs = b + beta
   settles at a persistent steady-state offset xref*beta/b_obs (derived in
   validation/dynamic_causal_mpc.mac), paying a per-step floor q*offset^2 at EVERY step. So the
   cumulative regret over a horizon T grows linearly and is UNBOUNDED as T -> inf, whereas the causal
   controller (beta = 0) tracks exactly and its cumulative regret is zero for every horizon. This is
   the qualitative separation the static result cannot see: a systematic bias compounds over time. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* cumulative tracking regret over a horizon of length T, given a per-step steady-state cost floor *)
Definition cumulative (floor t : R) : R := t * floor.

(* under confounding the per-step floor q*offset^2 is strictly positive (offset = xref*beta/b_obs) *)
Lemma confounded_floor_positive : forall q xref beta bobs,
  0 < q -> xref <> 0 -> beta <> 0 -> bobs <> 0 ->
  0 < q * (xref * beta / bobs) ^ 2.
Proof.
  intros q xref beta bobs Hq Hx Hb Hbo.
  assert (Hoff : xref * beta / bobs <> 0).
  { unfold Rdiv. apply Rmult_integral_contrapositive_currified.
    - apply Rmult_integral_contrapositive_currified; assumption.
    - apply Rinv_neq_0_compat; assumption. }
  assert (Hsq : 0 < (xref * beta / bobs) ^ 2).
  { destruct (Rtotal_order (xref * beta / bobs) 0) as [H | [H | H]].
    - nra.
    - exfalso; apply Hoff; exact H.
    - nra. }
  apply Rmult_lt_0_compat; assumption.
Qed.

(* with a positive per-step floor, cumulative regret grows strictly with the horizon *)
Theorem predictive_horizon_regret_grows : forall floor t1 t2,
  0 < floor -> t1 < t2 -> cumulative floor t1 < cumulative floor t2.
Proof.
  intros floor t1 t2 Hf Ht. unfold cumulative.
  apply Rmult_lt_compat_r; assumption.
Qed.

(* ...and is unbounded: no matter the bound M, some horizon exceeds it (Archimedean) *)
Theorem predictive_horizon_regret_unbounded : forall floor M,
  0 < floor -> exists t, M < cumulative floor t.
Proof.
  intros floor M Hf. exists ((Rabs M + 1) / floor). unfold cumulative.
  assert (Heq : (Rabs M + 1) / floor * floor = Rabs M + 1) by (field; lra).
  assert (HM : M <= Rabs M) by apply Rle_abs.
  lra.
Qed.

(* the causal controller (beta = 0 => floor = 0) has zero cumulative regret at every horizon *)
Lemma causal_horizon_regret_zero : forall t, cumulative 0 t = 0.
Proof. intros t. unfold cumulative. ring. Qed.
