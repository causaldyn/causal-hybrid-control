(* Rocq: CONSTRAINED certainty-equivalence regret is PIECEWISE-QUADRATIC (Gros & Diehl, "Numerical
   Optimal Control" 2022, Ch 16: active-set / KKT sensitivity of the optimizer; derived in
   validation/constrained_ce_regret.mac). A budget/safety cap u <= umax clips the control to
   u_opt(b) = min(u*(b), umax). Two structural facts break the program's clean O(eps^2) CE regret into
   pieces: (A) clipping is NON-EXPANSIVE, so the constrained CE regret never exceeds the unconstrained
   one; (B) when the constraint is active the control FREEZES at umax -- it stops tracking the effect,
   so the regret's curvature collapses to 0. The optimizer is therefore C^0 but not C^1 across the
   activation threshold, and the regret is only piecewise-quadratic. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* clipping the optimizer at a cap is non-expansive: min(.,c) is 1-Lipschitz, squared. *)
Lemma clip_sq_nonexpansive : forall a b c, (Rmin a c - Rmin b c) ^ 2 <= (a - b) ^ 2.
Proof.
  intros a b c. unfold Rmin.
  destruct (Rle_dec a c) as [Ha | Ha]; destruct (Rle_dec b c) as [Hb | Hb].
  - nra.
  - apply Rnot_le_lt in Hb.
    assert (H : 0 <= (b - c) * (b + c - 2 * a)) by (apply Rmult_le_pos; lra).
    assert (Hid : (a - b) ^ 2 - (a - c) ^ 2 = (b - c) * (b + c - 2 * a)) by ring. lra.
  - apply Rnot_le_lt in Ha.
    assert (H : 0 <= (a - c) * (a + c - 2 * b)) by (apply Rmult_le_pos; lra).
    assert (Hid : (a - b) ^ 2 - (c - b) ^ 2 = (a - c) * (a + c - 2 * b)) by ring. lra.
  - assert (H0 : (c - c) ^ 2 = 0) by ring. rewrite H0. apply pow2_ge_0.
Qed.

Definition ustar (b xt rr : R) : R := b * xt / (b ^ 2 + rr).         (* unconstrained optimum *)
Definition uopt (b xt rr umax : R) : R := Rmin (ustar b xt rr) umax.  (* clipped at the budget umax *)
Definition ce_regret (bh b xt rr umax : R) : R :=
  (b ^ 2 + rr) * (uopt bh xt rr umax - uopt b xt rr umax) ^ 2.
Definition ce_regret_unc (bh b xt rr : R) : R :=
  (b ^ 2 + rr) * (ustar bh xt rr - ustar b xt rr) ^ 2.

(* (A) The budget constraint never increases the certainty-equivalence regret (clipping non-expansive). *)
Theorem constrained_regret_le_unconstrained : forall bh b xt rr umax,
  0 <= b ^ 2 + rr -> ce_regret bh b xt rr umax <= ce_regret_unc bh b xt rr.
Proof.
  intros bh b xt rr umax Hpos. unfold ce_regret, ce_regret_unc, uopt.
  apply Rmult_le_compat_l; [exact Hpos | apply clip_sq_nonexpansive].
Qed.

(* (B) When the constraint is active the control freezes at umax: it stops tracking the effect. *)
Theorem control_frozen_when_active : forall b1 b2 xt rr umax,
  umax <= ustar b1 xt rr -> umax <= ustar b2 xt rr ->
  uopt b1 xt rr umax = uopt b2 xt rr umax.
Proof.
  intros b1 b2 xt rr umax H1 H2. unfold uopt.
  rewrite (Rmin_right _ _ H1), (Rmin_right _ _ H2). reflexivity.
Qed.

(* Hence the regret's curvature COLLAPSES to zero across the active-set boundary: on the active side an
   effect-estimate error costs nothing (the control does not move) -- the source of the piecewise kink. *)
Theorem regret_zero_when_both_active : forall bh b xt rr umax,
  umax <= ustar bh xt rr -> umax <= ustar b xt rr -> ce_regret bh b xt rr umax = 0.
Proof.
  intros bh b xt rr umax Hh Hb. unfold ce_regret.
  rewrite (control_frozen_when_active bh b xt rr umax Hh Hb). ring.
Qed.
