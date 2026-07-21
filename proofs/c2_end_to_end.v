(* Rocq (CONTRIBUTION 2, END-TO-END): the control-side reduction of the C2 chain
       multichannel causal estimation  ->  bottleneck rate  ->  dynamic control regret.
   On a clustered network (G clusters) the cross-fit two-channel Robinson-DML total-effect estimate has
   error ||B_hat - B|| <= s + e_d + e_s, where s is the cluster-robust SAMPLING term and e_d, e_s are the
   two channels' nuisance-induced REMAINDERS. The certainty-equivalent controller obeys the Mania-Tu-Recht
   local quadratic bound R <= cc*||B_hat - B||^2 (cc = LQ curvature; small-error + stabilising). We prove
   the deterministic composition: R <= 3*cc*(s^2 + e_d^2 + e_s^2), hence
       full-orth (e_d, e_s = O(delta^2))       => R <= 3*cc*(fG + 2*delta^4) = O(1/G + delta^4),
       half-orth (e_s = O(delta), bottleneck)  => R <= 3*cc*(fG + delta^4 + delta^2) = O(1/G + delta^2),
   with s^2 <= fG the 1/G cluster floor. This ADDS the sampling term that multichannel_control.v omits.
   HONEST SCOPE: the statistical RATES (s ~ 1/sqrt(G), e_j ~ delta_j^{p_j}) are the HYPOTHESES here --
   cited (Chernozhukov et al.; Robinson; Hays & Raghavan) and certified empirically in chc.regret; Rocq
   proves the control-side algebra that turns those rates into the regret order. Derived in
   validation/c2_end_to_end.mac. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

Definition c2_regret (cc s e_d e_s : R) : R := cc * (s + e_d + e_s) ^ 2.

(* Three-way triangle: the regret is bounded by the sum of the sampling + two channel radii. *)
Theorem three_channel_bound : forall cc s e_d e_s r_s r_d r_e,
  0 <= cc -> Rabs s <= r_s -> Rabs e_d <= r_d -> Rabs e_s <= r_e ->
  c2_regret cc s e_d e_s <= cc * (r_s + r_d + r_e) ^ 2.
Proof.
  intros cc s e_d e_s r_s r_d r_e Hc Hs Hd He. unfold c2_regret.
  apply Rmult_le_compat_l; [exact Hc |].
  assert (Ht : Rabs (s + e_d + e_s) <= r_s + r_d + r_e).
  { eapply Rle_trans; [apply Rabs_triang |].
    assert (H1 : Rabs (s + e_d) <= r_s + r_d)
      by (eapply Rle_trans; [apply Rabs_triang | lra]).
    lra. }
  assert (Hr : 0 <= r_s + r_d + r_e) by (eapply Rle_trans; [apply Rabs_pos | exact Ht]).
  set (x := s + e_d + e_s) in *. clearbody x. split_Rabs; nra.
Qed.

(* Sum-of-three-squares bound: (a+b+c)^2 <= 3*(a^2+b^2+c^2), from
   3(a^2+b^2+c^2) - (a+b+c)^2 = (a-b)^2 + (b-c)^2 + (c-a)^2 >= 0. *)
Theorem sum3_sq_bound : forall a b c, (a + b + c) ^ 2 <= 3 * (a ^ 2 + b ^ 2 + c ^ 2).
Proof.
  intros a b c.
  assert (H : 0 <= (a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2).
  { repeat apply Rplus_le_le_0_compat; apply pow2_ge_0. }
  nra.
Qed.

(* END-TO-END, FULL-orthogonal: both channels O(delta^2) and the cluster floor s^2 <= fG give
   R <= 3*cc*(fG + 2*delta^4) = O(1/G + delta^4). *)
Theorem end_to_end_full : forall cc s e_d e_s r_s fG d,
  0 <= cc -> 0 <= d -> Rabs s <= r_s -> r_s ^ 2 <= fG ->
  Rabs e_d <= d ^ 2 -> Rabs e_s <= d ^ 2 ->
  c2_regret cc s e_d e_s <= 3 * cc * (fG + 2 * d ^ 4).
Proof.
  intros cc s e_d e_s r_s fG d Hc Hd0 Hs HfG Hed Hes.
  eapply Rle_trans; [apply (three_channel_bound cc s e_d e_s r_s (d ^ 2) (d ^ 2)); assumption |].
  replace (3 * cc * (fG + 2 * d ^ 4)) with (cc * (3 * (fG + 2 * d ^ 4))) by ring.
  apply Rmult_le_compat_l; [exact Hc |].
  eapply Rle_trans; [apply (sum3_sq_bound r_s (d ^ 2) (d ^ 2)) |].
  nra.
Qed.

(* END-TO-END, HALF-orthogonal: the spillover stays O(delta) (plug-in) and is the BOTTLENECK, giving
   R <= 3*cc*(fG + delta^4 + delta^2) = O(1/G + delta^2) -- orthogonalising only the direct channel
   bought nothing at the regret order. *)
Theorem end_to_end_half : forall cc s e_d e_s r_s fG d,
  0 <= cc -> 0 <= d -> Rabs s <= r_s -> r_s ^ 2 <= fG ->
  Rabs e_d <= d ^ 2 -> Rabs e_s <= d ->
  c2_regret cc s e_d e_s <= 3 * cc * (fG + d ^ 4 + d ^ 2).
Proof.
  intros cc s e_d e_s r_s fG d Hc Hd0 Hs HfG Hed Hes.
  eapply Rle_trans; [apply (three_channel_bound cc s e_d e_s r_s (d ^ 2) d); assumption |].
  replace (3 * cc * (fG + d ^ 4 + d ^ 2)) with (cc * (3 * (fG + d ^ 4 + d ^ 2))) by ring.
  apply Rmult_le_compat_l; [exact Hc |].
  eapply Rle_trans; [apply (sum3_sq_bound r_s (d ^ 2) d) |].
  nra.
Qed.

(* Perfect-nuisance sampling term: with PERFECT nuisances (e_d = e_s = 0) the regret is bounded ABOVE by
   the sampling scale cc*fG -- debiasing cannot remove the O(1/G) term. This is only an UPPER bound; the
   matching LOWER bound (that the 1/G rate is IRREDUCIBLE) is proved separately by the clustered van-Trees
   argument in proofs/clustered_van_trees.v (regret_floor_uniform_positive). *)
Theorem perfect_nuisance_sampling_bound : forall cc s r_s fG,
  0 <= cc -> Rabs s <= r_s -> r_s ^ 2 <= fG ->
  c2_regret cc s 0 0 <= cc * fG.
Proof.
  intros cc s r_s fG Hc Hs HfG. unfold c2_regret.
  apply Rmult_le_compat_l; [exact Hc |].
  replace (s + 0 + 0) with s by ring.
  assert (Hrpos : 0 <= r_s) by (eapply Rle_trans; [apply Rabs_pos | exact Hs]).
  assert (Hs2 : s ^ 2 <= r_s ^ 2) by (split_Rabs; nra).
  lra.
Qed.
