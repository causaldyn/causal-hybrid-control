(* Rocq (CONTRIBUTION 2, exposure-map generalisation): the marketplace network model
       x_{t+1} = A x_t + (B_d + B_s W) u_t + eps,   B_eff(W) = B_d + B_s W,
   with a zonal exposure/migration matrix W (Aronow-Samii exposure map; Bramoulle et al. SAR; Munro, Xu &
   Wager and Wager & Xu market equilibrium; Hays & Raghavan shared-state DML). Estimating (B_d, B_s, W)
   gives THREE effect channels; with the cluster-sampling term s that is FOUR error sources. The bilinear
   spillover term expands by the EXACT product rule (bilinear_product_rule), so
       ||B_eff_hat - B_eff|| <= r_d + r_s + r_W + cross,  r_d=||dB_d||, r_s=||W||||dB_s||, r_W=||B_s||||dW||,
   and the control regret R = cc*(s + r_d + r_s + r_W)^2 <= 4*cc*(s^2 + r_d^2 + r_s^2 + r_W^2)
   (sum4_sq_bound), the square absorbing the cross term. Full-orth => R = O(1/G + delta^4)
   (exposure_full_orth). This is the multi-channel bottleneck (multichannel_control.v) lifted to the
   exposure-map plant, with the sampling term of clustered_van_trees.v / c2_end_to_end.v. Derived in
   validation/exposure_map_c2.mac. LOAD-BEARING (Hays-Raghavan): r_W enters SQUARED only if W is
   orthogonalised/cross-fit; else linearly (a strictly worse bottleneck) -- an assumption, not proved. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

Definition exp_regret (cc s r_d r_s r_W : R) : R := cc * (s + r_d + r_s + r_W) ^ 2.

(* 4-term Cauchy-Schwarz: (a+b+c+d)^2 <= 4(a^2+b^2+c^2+d^2), from the six pairwise squares. *)
Theorem sum4_sq_bound : forall a b c d,
  (a + b + c + d) ^ 2 <= 4 * (a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2).
Proof.
  intros a b c d.
  assert (H : 0 <= (a - b) ^ 2 + (a - c) ^ 2 + (a - d) ^ 2
                 + (b - c) ^ 2 + (b - d) ^ 2 + (c - d) ^ 2).
  { repeat apply Rplus_le_le_0_compat; apply pow2_ge_0. }
  nra.
Qed.

(* The exact product rule for the bilinear spillover term B_s*W, with its triangle bound: the
   effective-effect error from estimating (B_s, W) = spillover-coeff channel + exposure-map channel + a
   second-order cross term. (Scalar surrogate for the operator-norm inequality.) *)
Theorem bilinear_product_rule : forall bs w dbs dw,
  Rabs ((bs + dbs) * (w + dw) - bs * w)
    <= Rabs dbs * Rabs w + Rabs bs * Rabs dw + Rabs dbs * Rabs dw.
Proof.
  intros bs w dbs dw.
  replace ((bs + dbs) * (w + dw) - bs * w) with (dbs * w + bs * dw + dbs * dw) by ring.
  rewrite <- (Rabs_mult dbs w), <- (Rabs_mult bs dw), <- (Rabs_mult dbs dw).
  eapply Rle_trans; [apply Rabs_triang |].
  apply Rplus_le_compat; [apply Rabs_triang | apply Rle_refl].
Qed.

(* END-TO-END exposure map, FULL-orthogonal: four error sources -- cluster sampling s (s^2 <= fG) and
   the three orthogonalised channels r_d, r_s, r_W (each O(delta^2)) -- give
   R <= 4*cc*(fG + 3*delta^4) = O(1/G + delta^4). *)
Theorem exposure_full_orth : forall cc s r_d r_s r_W fG d,
  0 <= cc -> 0 <= s -> s ^ 2 <= fG ->
  0 <= r_d -> r_d <= d ^ 2 -> 0 <= r_s -> r_s <= d ^ 2 -> 0 <= r_W -> r_W <= d ^ 2 ->
  exp_regret cc s r_d r_s r_W <= 4 * cc * (fG + 3 * d ^ 4).
Proof.
  intros cc s r_d r_s r_W fG d Hc Hs HfG Hd0 Hd1 Hs0 Hs1 HW0 HW1. unfold exp_regret.
  replace (4 * cc * (fG + 3 * d ^ 4)) with (cc * (4 * (fG + 3 * d ^ 4))) by ring.
  apply Rmult_le_compat_l; [exact Hc |].
  eapply Rle_trans; [apply sum4_sq_bound |].
  assert (Hrd2 : r_d ^ 2 <= d ^ 4) by nra.
  assert (Hrs2 : r_s ^ 2 <= d ^ 4) by nra.
  assert (HrW2 : r_W ^ 2 <= d ^ 4) by nra.
  nra.
Qed.
