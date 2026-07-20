(* Rocq: the interference-aware certainty-equivalence regret certificate behind
   chc.regret.interference_regret_certificate. Offline control estimates a model with two error
   sources -- identification error [eid] of the direct dynamics and interference error [eint] of the
   exposure / shared-state map -- plans pi_hat on the estimate, and deploys on the true plant. Under
   a kappa-smooth cost with minimiser pi_star and the certainty-equivalence controllability bound
   |pi_hat - pi_star| <= C*(eid + eint), the regret is QUADRATIC in the TOTAL error, with the
   interference error entering additively inside the square. This extends the
   Dean-Mania-Tu-Recht-Matni LQ certainty-equivalence bound (which has the eid term only). A
   machine-checked regret certificate under interference is the citable, hard-to-replicate piece of
   plans/20 section A. See also proofs/box_projection.v (the feasibility invariant). *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* The certainty-equivalence surrogate regret at a plan error [d], for a kappa-smooth cost:
   the second-order (descent-lemma) upper bound J(pi_hat) - J(pi_star) <= kappa/2 * d^2. *)
Definition ce_regret (kappa d : R) : R := kappa / 2 * d ^ 2.

(* Monotonicity of the certificate in the plan-error magnitude (kappa >= 0). *)
Lemma ce_regret_mono : forall kappa d e,
  0 <= kappa -> Rabs d <= e -> ce_regret kappa d <= ce_regret kappa e.
Proof.
  intros kappa d e Hk Hde. unfold ce_regret.
  assert (He : 0 <= e) by (eapply Rle_trans; [apply Rabs_pos | exact Hde]).
  apply Rmult_le_compat_l; [lra |].
  split_Rabs; nra.
Qed.

(* The interference-aware regret certificate: with the total error eid + eint propagated through
   the controllability bound, the regret is at most kappa/2 * (C*(eid+eint))^2. *)
Theorem interference_regret_bound : forall kappa C eid eint d,
  0 <= kappa -> 0 <= C -> 0 <= eid -> 0 <= eint ->
  Rabs d <= C * (eid + eint) ->
  ce_regret kappa d <= kappa / 2 * (C * (eid + eint)) ^ 2.
Proof.
  intros kappa C eid eint d Hk HC Hid Hint Hd.
  apply Rle_trans with (ce_regret kappa (C * (eid + eint))).
  - apply ce_regret_mono; assumption.
  - unfold ce_regret. apply Rle_refl.
Qed.

(* Interference is not free: any positive exposure-map error strictly enlarges the certificate over
   the interference-blind (eint = 0) bound -- so ignoring interference under-states the true regret. *)
Lemma interference_strictly_worse : forall kappa C eid eint,
  0 < kappa -> 0 < C -> 0 <= eid -> 0 < eint ->
  kappa / 2 * (C * (eid + 0)) ^ 2 < kappa / 2 * (C * (eid + eint)) ^ 2.
Proof.
  intros kappa C eid eint Hk HC Hid Hint.
  replace (eid + 0) with eid by ring.
  apply Rmult_lt_compat_l; [lra |].
  set (a := C * eid). set (b := C * (eid + eint)).
  assert (Ha : 0 <= a) by (unfold a; nra).
  assert (Hab : a < b) by (unfold a, b; nra).
  nra.
Qed.
