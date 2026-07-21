(* Rocq: the DOUBLY-ROBUST version of result 0 (proofs/orthogonal_control.v). For a binary
   intervention the AIPW estimator of the control effect has a bias that is the PRODUCT of the
   outcome-model error dmu and the propensity error de: bias = dmu*de/(e+de) (derived in
   validation/doubly_robust.mac). Hence it is exactly ZERO if EITHER nuisance is correct -- double
   robustness -- and its magnitude never exceeds the outcome-regression (single-robust) bias dmu. The
   control regret is O((dmu*de)^2), so if one nuisance is estimated well the other may be poor. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* the AIPW / doubly-robust effect bias in one covariate stratum (propensity e) *)
Definition dr_bias (dmu de e : R) : R := dmu * de / (e + de).

(* double robustness (1): a correct outcome model (dmu = 0) makes the bias exactly zero,
   regardless of the propensity error *)
Lemma dr_zero_if_outcome_correct : forall de e, dr_bias 0 de e = 0.
Proof. intros de e. unfold dr_bias, Rdiv. rewrite !Rmult_0_l. reflexivity. Qed.

(* double robustness (2): a correct propensity model (de = 0) makes the bias exactly zero,
   regardless of the outcome error *)
Lemma dr_zero_if_propensity_correct : forall dmu e, dr_bias dmu 0 e = 0.
Proof. intros dmu e. unfold dr_bias, Rdiv. rewrite Rmult_0_r, Rmult_0_l. reflexivity. Qed.

(* the DR bias never exceeds the outcome-regression (single-robust) bias dmu in magnitude: cleared of
   division, (dmu*de)^2 <= dmu^2*(e+de)^2 for e, de >= 0 (i.e. |de/(e+de)| <= 1) *)
Lemma dr_squared_bias_le_outcome : forall dmu de e,
  0 <= e -> 0 <= de -> (dmu * de) ^ 2 <= dmu ^ 2 * (e + de) ^ 2.
Proof.
  intros dmu de e He Hde.
  assert (Hm : 0 <= dmu ^ 2) by apply pow2_ge_0.
  assert (He2 : 0 <= e * (e + 2 * de)) by nra.
  nra.
Qed.
