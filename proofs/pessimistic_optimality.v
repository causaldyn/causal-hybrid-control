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
