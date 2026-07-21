(* Rocq (CONTRIBUTION 2): MULTI-CHANNEL orthogonal control -- the serious form of result 4x0 (derived
   in validation/multichannel_control.mac). On a network the control-relevant total effect of a uniform
   action is B = b_d + b_s (direct + spillover/exposure channels). The control regret is quadratic in
   the total-effect error e_d + e_s, so it is bounded by the SUM of the channel errors (triangle), and
   the regret ORDER is set by the LEAST-orthogonalised channel: orthogonalising only the direct channel
   caps the regret at O(delta^2) if the spillover channel stays first-order biased; orthogonalising BOTH
   reaches O(delta^4). This is the multi-channel generalisation of the transfer theorem
   (composition_transfer.v). The cross-fitting / cluster-robust / nonasymptotic parts are empirical
   (chc.regret certificate); here we prove the deterministic order-bottleneck core. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

Definition mc_regret (cc e_d e_s : R) : R := cc * (e_d + e_s) ^ 2.

(* the regret is bounded by the SUM of the per-channel error radii (triangle inequality). *)
Theorem bottleneck_bound : forall cc e_d e_s r_d r_s,
  0 <= cc -> Rabs e_d <= r_d -> Rabs e_s <= r_s ->
  mc_regret cc e_d e_s <= cc * (r_d + r_s) ^ 2.
Proof.
  intros cc e_d e_s r_d r_s Hc Hd Hs. unfold mc_regret.
  apply Rmult_le_compat_l; [exact Hc |].
  assert (Ht : Rabs (e_d + e_s) <= r_d + r_s) by (eapply Rle_trans; [apply Rabs_triang | lra]).
  assert (Hr : 0 <= r_d + r_s) by (eapply Rle_trans; [apply Rabs_pos | exact Ht]).
  set (x := e_d + e_s) in *. clearbody x. split_Rabs; nra.
Qed.

(* HALF-orthogonal (direct debiased to order 2, spillover only order 1): the regret is O(delta^2) --
   the spillover channel is the bottleneck, so orthogonalising the direct channel bought nothing at the
   regret order. *)
Theorem half_orthogonal_order2 : forall cc e_d e_s d,
  0 <= cc -> 0 <= d -> d <= 1 -> Rabs e_d <= d ^ 2 -> Rabs e_s <= d ->
  mc_regret cc e_d e_s <= 4 * cc * d ^ 2.
Proof.
  intros cc e_d e_s d Hc Hd0 Hd1 Hed Hes.
  eapply Rle_trans; [apply (bottleneck_bound cc e_d e_s (d ^ 2) d); assumption |].
  assert (Hsq : (d ^ 2 + d) ^ 2 <= 4 * d ^ 2).
  { assert (Hq : 0 <= d ^ 2 * ((d + 3) * (1 - d))).
    { apply Rmult_le_pos; [nra | apply Rmult_le_pos; lra]. }
    nra. }
  replace (4 * cc * d ^ 2) with (cc * (4 * d ^ 2)) by ring.
  apply Rmult_le_compat_l; assumption.
Qed.

(* FULL-orthogonal (both channels debiased to order 2): the regret reaches O(delta^4). *)
Theorem full_orthogonal_order4 : forall cc e_d e_s d,
  0 <= cc -> 0 <= d -> Rabs e_d <= d ^ 2 -> Rabs e_s <= d ^ 2 ->
  mc_regret cc e_d e_s <= 4 * cc * d ^ 4.
Proof.
  intros cc e_d e_s d Hc Hd0 Hed Hes.
  eapply Rle_trans; [apply (bottleneck_bound cc e_d e_s (d ^ 2) (d ^ 2)); assumption |].
  replace (4 * cc * d ^ 4) with (cc * (d ^ 2 + d ^ 2) ^ 2) by ring.
  apply Rle_refl.
Qed.
