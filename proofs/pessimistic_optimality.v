(* Rocq: pessimism as a correction to the OPTIMALITY CONDITION, not an ad-hoc penalty (CHC's
   offline-safety story, control-first). Distrusting the causal effect by an amount rho >= 0 in the
   scalar control stationarity condition shrinks the certainty-equivalent control u_ce = xt*b0/(b0^2+rr)
   to u_pess = xt*b0/(b0^2+rr+rho). The correction u_ce - u_pess has a closed form, and pessimism
   contracts the control toward the trusted set (u_pess between 0 and u_ce). Maxima
   (validation/pessimistic_optimality.mac) further shows the EXPECTED regret under effect-estimation
   uncertainty s^2 is minimised at rho* = s^2 -- the optimal pessimism equals the estimate's variance,
   so no tuning is needed. See proofs/causal_mpc.v (why the effect must be causal) and
   proofs/orthogonal_control.v (its debiasing rate). *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

Definition u_ce (xt b0 rr : R) : R := xt * b0 / (b0 ^ 2 + rr).
Definition u_pess (xt b0 rr rho : R) : R := xt * b0 / (b0 ^ 2 + rr + rho).

(* The pessimism correction in closed form: u_ce - u_pess = xt*b0*rho / (D*(D+rho)), D = b0^2+rr. *)
Lemma pessimism_correction : forall xt b0 rr rho,
  0 < b0 ^ 2 + rr -> 0 <= rho ->
  u_ce xt b0 rr - u_pess xt b0 rr rho
    = xt * b0 * rho / ((b0 ^ 2 + rr) * (b0 ^ 2 + rr + rho)).
Proof.
  intros xt b0 rr rho HD Hr. unfold u_ce, u_pess.
  assert (H1 : b0 ^ 2 + rr <> 0) by lra.
  assert (H2 : b0 ^ 2 + rr + rho <> 0) by lra.
  field; split; assumption.
Qed.

(* Pessimism shrinks the control: for a control aimed at the target (xt*b0 >= 0), u_pess <= u_ce. *)
Lemma pessimism_shrinks : forall xt b0 rr rho,
  0 < b0 ^ 2 + rr -> 0 <= rho -> 0 <= xt * b0 ->
  u_pess xt b0 rr rho <= u_ce xt b0 rr.
Proof.
  intros xt b0 rr rho HD Hr Hxb.
  assert (Hc := pessimism_correction xt b0 rr rho HD Hr).
  assert (Hpos : 0 <= xt * b0 * rho / ((b0 ^ 2 + rr) * (b0 ^ 2 + rr + rho))).
  { unfold Rdiv. apply Rmult_le_pos; [nra | apply Rlt_le, Rinv_0_lt_compat; nra]. }
  lra.
Qed.

(* ...and never flips its sign: u_pess stays in the trusted region [0, u_ce]. *)
Lemma pessimism_keeps_sign : forall xt b0 rr rho,
  0 < b0 ^ 2 + rr -> 0 <= rho -> 0 <= xt * b0 ->
  0 <= u_pess xt b0 rr rho.
Proof.
  intros xt b0 rr rho HD Hr Hxb. unfold u_pess, Rdiv.
  apply Rmult_le_pos; [nra | apply Rlt_le, Rinv_0_lt_compat; nra].
Qed.

(* The OPTIMAL pessimism equals the effect-estimate variance, formalised. Maxima
   (validation/pessimistic_optimality.mac) differentiates the expected regret under effect
   uncertainty s^2 and gives  d E[regret]/d rho = grad_num / (positive denominator)  with
   grad_num = 2*b0^2*xt^2*(rho - s2). We prove the sign structure of grad_num here: it is zero exactly
   at rho = s2, negative below it and positive above it -- so E[regret] strictly decreases then
   increases, and rho* = s2 is its unique minimiser (no tuning: the optimal pessimism IS the variance). *)
Definition grad_num (b0 xt s2 rho : R) : R := 2 * b0 ^ 2 * xt ^ 2 * (rho - s2).

Lemma sq_pos : forall x, x <> 0 -> 0 < x ^ 2.
Proof.
  intros x Hx. destruct (Rtotal_order x 0) as [H | [H | H]].
  - nra.
  - exfalso; apply Hx; exact H.
  - nra.
Qed.

(* the expected-regret gradient vanishes exactly at rho = s2 (the effect-estimate variance) *)
Lemma pessimism_stationary_at_variance : forall b0 xt s2,
  grad_num b0 xt s2 s2 = 0.
Proof. intros b0 xt s2. unfold grad_num. ring. Qed.

(* below the variance the gradient is negative -- expected regret is still decreasing (raise rho) *)
Lemma pessimism_below_variance_decreasing : forall b0 xt s2 rho,
  b0 <> 0 -> xt <> 0 -> rho < s2 -> grad_num b0 xt s2 rho < 0.
Proof.
  intros b0 xt s2 rho Hb Hx Hlt. unfold grad_num.
  assert (0 < 2 * b0 ^ 2 * xt ^ 2) by (assert (0 < b0 ^ 2) by (apply sq_pos; exact Hb);
                                       assert (0 < xt ^ 2) by (apply sq_pos; exact Hx); nra).
  nra.
Qed.

(* above the variance the gradient is positive -- expected regret is increasing (rho is too large) *)
Lemma pessimism_above_variance_increasing : forall b0 xt s2 rho,
  b0 <> 0 -> xt <> 0 -> s2 < rho -> 0 < grad_num b0 xt s2 rho.
Proof.
  intros b0 xt s2 rho Hb Hx Hgt. unfold grad_num.
  assert (0 < 2 * b0 ^ 2 * xt ^ 2) by (assert (0 < b0 ^ 2) by (apply sq_pos; exact Hb);
                                       assert (0 < xt ^ 2) by (apply sq_pos; exact Hx); nra).
  nra.
Qed.
