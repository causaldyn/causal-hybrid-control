(* Rocq (§35 minimax confounding-robust CONTROLLER): under an ASYMMETRIC loss (overshoot penalty al,
   undershoot penalty be) the pessimism radius D SHIFTS the control gain and the robust controller
   STRICTLY beats certainty-equivalence (CE) in worst-case loss. Closed forms in
   validation/confounding_robust_control.mac (residual 0). We prove the control-relevant sign facts:
   the gain-shift factor is nonnegative (so the robust gain is <= CE -- more conservative), and the
   worst-case-loss improvement gap is nonnegative, strictly positive under real asymmetry + confounding,
   and exactly zero when the loss is symmetric (recovering §33: pessimism = the CE centre). Algebraic;
   al,be,D,Gamma are the analyst's inputs, not proved. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* The gain shift: u_rob = u_ce / (1 + shift_factor * D). *)
Definition shift_factor (al be bhat : R) : R := (al - be) / ((al + be) * bhat).

Lemma shift_factor_nonneg :
  forall al be bhat : R, 0 < be -> be <= al -> 0 < bhat -> 0 <= shift_factor al be bhat.
Proof.
  intros al be bhat Hbe Hal Hbhat. unfold shift_factor, Rdiv.
  apply Rmult_le_pos; [ lra |].
  left. apply Rinv_0_lt_compat. apply Rmult_lt_0_compat; lra.
Qed.

(* A nonnegative shift makes the robust gain no larger than CE: more confounding -> more conservative. *)
Lemma robust_gain_conservative :
  forall u_ce s D : R, 0 <= u_ce -> 0 <= s -> 0 <= D -> u_ce / (1 + s * D) <= u_ce.
Proof.
  intros u_ce s D Hu Hs HD.
  assert (HsD : 0 <= s * D) by (apply Rmult_le_pos; assumption).
  apply Rmult_le_reg_r with (r := 1 + s * D); [ lra |].
  unfold Rdiv. rewrite Rmult_assoc, Rinv_l by lra. rewrite Rmult_1_r. nra.
Qed.

(* The worst-case-loss improvement of the robust controller over CE (Maxima gap). *)
Definition rob_gap (al be tau bhat D : R) : R :=
  al * tau * D * (al - be) * (bhat + D) / (bhat * ((al + be) * bhat + (al - be) * D)).

(* Denominator is strictly positive under the maintained sign conditions. *)
Lemma rob_gap_denom_pos :
  forall al be bhat D : R,
    0 < be -> be <= al -> 0 < bhat -> 0 <= D ->
    0 < bhat * ((al + be) * bhat + (al - be) * D).
Proof.
  intros al be bhat D Hbe Hal Hbhat HD.
  apply Rmult_lt_0_compat; [ exact Hbhat |].
  apply Rplus_lt_le_0_compat.
  - apply Rmult_lt_0_compat; lra.
  - apply Rmult_le_pos; lra.
Qed.

(* NEVER WORSE: the robust controller's worst-case loss is <= CE's (improvement gap >= 0). *)
Lemma rob_gap_nonneg :
  forall al be tau bhat D : R,
    0 < be -> be <= al -> 0 <= tau -> 0 < bhat -> 0 <= D ->
    0 <= rob_gap al be tau bhat D.
Proof.
  intros al be tau bhat D Hbe Hal Htau Hbhat HD. unfold rob_gap, Rdiv.
  apply Rmult_le_pos.
  - apply Rmult_le_pos.
    + apply Rmult_le_pos.
      * apply Rmult_le_pos; [ apply Rmult_le_pos; [ lra | exact Htau ] | exact HD ].
      * lra.
    + lra.
  - left. apply Rinv_0_lt_compat. apply rob_gap_denom_pos; assumption.
Qed.

(* STRICTLY BETTER under real asymmetry (al > be) and real confounding (D > 0): pessimism helps. *)
Lemma rob_gap_pos :
  forall al be tau bhat D : R,
    0 < be -> be < al -> 0 < tau -> 0 < bhat -> 0 < D ->
    0 < rob_gap al be tau bhat D.
Proof.
  intros al be tau bhat D Hbe Hal Htau Hbhat HD. unfold rob_gap, Rdiv.
  apply Rmult_lt_0_compat.
  - repeat apply Rmult_lt_0_compat; lra.
  - apply Rinv_0_lt_compat. apply rob_gap_denom_pos; lra.
Qed.

(* SYMMETRIC LOSS: al = be gives zero improvement -- pessimism = the CE centre (recovers §33). *)
Lemma rob_gap_symmetric_zero :
  forall al tau bhat D : R, rob_gap al al tau bhat D = 0.
Proof.
  intros al tau bhat D. unfold rob_gap.
  assert (H0 : al * tau * D * (al - al) * (bhat + D) = 0) by ring.
  rewrite H0. unfold Rdiv. rewrite Rmult_0_l. reflexivity.
Qed.

(* UNDERSHOOT-DOMINANT branch (reviewer-8): the improvement gap is PIECEWISE in max(al,be). The first
   branch (rob_gap, al>=be) has W_ce = al*tau*D/bhat; for be >= al -- undershoot costlier, e.g. Result
   37's churn = 4x waste -- W_ce = be*tau*D/bhat and the sharp improvement is this SECOND branch. Both
   are nonneg for bhat > D > 0 (identified effect sign) and strictly positive for D > 0, al <> be. *)
Definition rob_gap_under (al be tau bhat D : R) : R :=
  be * tau * D * (be - al) * (bhat - D) / (bhat * ((al + be) * bhat + (al - be) * D)).

Lemma rob_gap_under_denom_pos :
  forall al be bhat D : R,
    0 < al -> al <= be -> 0 < D -> D < bhat ->
    0 < bhat * ((al + be) * bhat + (al - be) * D).
Proof.
  intros al be bhat D Hal Hle HD Hbd.
  apply Rmult_lt_0_compat; [ lra |].
  assert (Hid : (al + be) * bhat + (al - be) * D = (al + be) * (bhat - D) + 2 * al * D) by ring.
  rewrite Hid. apply Rplus_le_lt_0_compat; [ apply Rmult_le_pos; lra | nra ].
Qed.

Lemma rob_gap_under_nonneg :
  forall al be tau bhat D : R,
    0 < al -> al <= be -> 0 <= tau -> 0 < D -> D < bhat ->
    0 <= rob_gap_under al be tau bhat D.
Proof.
  intros al be tau bhat D Hal Hle Htau HD Hbd. unfold rob_gap_under, Rdiv.
  apply Rmult_le_pos.
  - apply Rmult_le_pos.
    + apply Rmult_le_pos.
      * apply Rmult_le_pos; [ apply Rmult_le_pos; [ lra | exact Htau ] | lra ].
      * lra.
    + lra.
  - left. apply Rinv_0_lt_compat. apply rob_gap_under_denom_pos; assumption.
Qed.

Lemma rob_gap_under_pos :
  forall al be tau bhat D : R,
    0 < al -> al < be -> 0 < tau -> 0 < D -> D < bhat ->
    0 < rob_gap_under al be tau bhat D.
Proof.
  intros al be tau bhat D Hal Hlt Htau HD Hbd. unfold rob_gap_under, Rdiv.
  apply Rmult_lt_0_compat.
  - repeat apply Rmult_lt_0_compat; lra.
  - apply Rinv_0_lt_compat. apply rob_gap_under_denom_pos; lra.
Qed.

Lemma rob_gap_under_symmetric_zero :
  forall al tau bhat D : R, rob_gap_under al al tau bhat D = 0.
Proof.
  intros al tau bhat D. unfold rob_gap_under.
  assert (H0 : al * tau * D * (al - al) * (bhat - D) = 0) by ring.
  rewrite H0. unfold Rdiv. rewrite Rmult_0_l. reflexivity.
Qed.
