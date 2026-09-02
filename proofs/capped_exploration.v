(* Rocq: the algebraic core of the CAPPED EXPLORATION LAW (Result 56).

   validation/capped_exploration.mac derives, for the minimax exploration objective
   F(v) = A sum_t v_t + K sum_t 1/(I0 + c S_{t-1}) with S the prefix sum: the exchange argument
   (an earlier round is a strictly cheaper place for the same exploration), convexity, the block
   cost and its optimal length n* = sqrt(K T/(A c))/vcap, and the fact that the capped optimum
   carries EXACTLY the uncapped leading constant -- so a per-round action cap costs an ADDITIVE
   logarithm, not a constant factor, and does NOT make a taper optimal.

   Honest scope, as in minimax_exploration.v: Stdlib has no schedules or matrices, so what is
   proved here is the scalar spine -- the exchange inequality, the AM-GM floor with its equality
   case at n*, and the identification of the leading constant with the uncapped one. The
   T-round induction and the convex-programming step stay in Maxima and in the certificate. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Psatz.
Open Scope R_scope.

(* Stdlib has Rdiv_lt_0_compat but not its non-strict companion under this name; both are used
   below to keep the square-root side conditions readable. *)
Lemma nonneg_div : forall x y : R, 0 <= x -> 0 < y -> 0 <= x / y.
Proof.
  intros x y Hx Hy.
  unfold Rdiv.
  apply Rmult_le_pos; [exact Hx | left; apply Rinv_0_lt_compat; exact Hy].
Qed.

Lemma pos_triple : forall x y z : R, 0 < x -> 0 < y -> 0 < z -> 0 < x * y * z.
Proof.
  intros x y z Hx Hy Hz.
  apply Rmult_lt_0_compat; [apply Rmult_lt_0_compat |]; assumption.
Qed.

(* (A) THE EXCHANGE ARGUMENT. Moving exploration from a later round to an earlier one, at equal
   total budget, strictly lowers the estimation term: only the prefix sums matter, and an earlier
   round raises more of them. *)
Lemma earlier_is_strictly_cheaper :
  forall i0 c k s s' : R,
  0 < i0 -> 0 < c -> 0 < k -> 0 <= s -> s < s' ->
  k / (i0 + c * s') < k / (i0 + c * s).
Proof.
  intros i0 c k s s' Hi Hc Hk Hs Hlt.
  assert (H1 : 0 < i0 + c * s) by nra.
  assert (H2 : 0 < i0 + c * s') by nra.
  unfold Rdiv.
  apply Rmult_lt_compat_l; [exact Hk |].
  apply Rinv_lt_contravar; nra.
Qed.

(* (B) THE BLOCK COST HAS AN AM-GM FLOOR. Writing m = c vcap n for the information the block
   buys, the leading part of F is (A/c) m + K T/m, whose minimum over m > 0 is 2 sqrt(A K T/c),
   attained exactly at m* = sqrt(K T c/A). *)
Lemma block_cost_floor :
  forall a k c t m : R,
  0 < a -> 0 < k -> 0 < c -> 0 < t -> 0 < m ->
  2 * sqrt (a * k * t / c) <= (a / c) * m + k * t / m.
Proof.
  intros a k c t m Ha Hk Hc Ht Hm.
  assert (Hakt : 0 <= a * k * t / c)
    by (apply nonneg_div; [left; apply pos_triple; assumption | exact Hc]).
  assert (Hsq : sqrt (a * k * t / c) * sqrt (a * k * t / c) = a * k * t / c)
    by (apply sqrt_sqrt; exact Hakt).
  assert (Hgap : 0 <= ((a / c) * m - k * t / m) * ((a / c) * m - k * t / m))
    by (apply Rle_0_sqr).
  assert (Hprod : (a / c) * m * (k * t / m) = a * k * t / c) by (field; lra).
  assert (Hs0 : 0 <= sqrt (a * k * t / c)) by apply sqrt_pos.
  assert (HP : 0 < a / c * m)
    by (apply Rmult_lt_0_compat; [apply Rdiv_lt_0_compat |]; assumption).
  assert (HQ : 0 < k * t / m)
    by (apply Rdiv_lt_0_compat; [apply Rmult_lt_0_compat |]; assumption).
  apply Rmult_le_reg_r with (r := a / c * m + k * t / m + 2 * sqrt (a * k * t / c)); [lra |].
  nra.
Qed.

Lemma block_cost_attains_the_floor :
  forall a k c t : R,
  0 < a -> 0 < k -> 0 < c -> 0 < t ->
  (a / c) * sqrt (k * t * c / a) + k * t / sqrt (k * t * c / a) = 2 * sqrt (a * k * t / c).
Proof.
  intros a k c t Ha Hk Hc Ht.
  assert (Hpos : 0 < k * t * c / a)
    by (apply Rdiv_lt_0_compat; [apply pos_triple; assumption | exact Ha]).
  assert (Hs : 0 < sqrt (k * t * c / a)) by (apply sqrt_lt_R0; exact Hpos).
  assert (Hsq : sqrt (k * t * c / a) * sqrt (k * t * c / a) = k * t * c / a)
    by (apply sqrt_sqrt; lra).
  assert (Hmul : sqrt (k * t * c / a) * sqrt (a * k * t / c) = k * t).
  { rewrite <- sqrt_mult by (apply nonneg_div; [left; apply pos_triple; assumption | assumption]).
    replace (k * t * c / a * (a * k * t / c)) with ((k * t) * (k * t)) by (field; lra).
    rewrite sqrt_square by (apply Rmult_le_pos; lra). reflexivity. }
  apply (Rmult_eq_reg_r (sqrt (k * t * c / a))); [| lra].
  replace ((a / c * sqrt (k * t * c / a) + k * t / sqrt (k * t * c / a))
           * sqrt (k * t * c / a))
    with (a / c * (sqrt (k * t * c / a) * sqrt (k * t * c / a)) + k * t) by (field; lra).
  rewrite Hsq.
  replace (a / c * (k * t * c / a) + k * t) with (2 * (k * t)) by (field; lra).
  (* only the first occurrence: k * t also sits inside sqrt (k * t * c / a) on the right *)
  rewrite <- Hmul at 1.
  ring.
Qed.

(* (C) THE CAP DOES NOT MOVE THE CONSTANT. The model sets K = A g^2 and c = eta/sigma^2, and with
   those the capped leading term 2 sqrt(A K T/c) is EXACTLY the uncapped floor constant
   c_causal sqrt(T) = 2 A |g| sigma sqrt(T)/sqrt(eta). Compared through squares, so no root
   manipulation is needed. *)
Lemma capped_leading_term_is_the_uncapped_floor :
  forall a g sigma eta t : R,
  0 < sigma -> 0 < eta ->
  4 * a * (a * g * g) * t / (eta / (sigma * sigma))
    = (2 * a * g * sigma) * (2 * a * g * sigma) * t / eta.
Proof.
  intros a g sigma eta t Hsigma Heta.
  field; repeat split; lra.
Qed.

(* (D) WHY A TAPER CANNOT BE RESCUED BY A CAP. The exchange inequality (A) never mentions the
   cap: it compares two rounds at equal budget. So under ANY box constraint the optimum is the
   feasible schedule that front-loads as hard as the box allows -- the cap-saturating prefix.
   Stated here as the two-round consequence: if a feasible schedule leaves the earlier round
   below the cap while the later one is positive, it is strictly improvable. *)
Lemma unsaturated_prefix_is_improvable :
  forall i0 c k v1 v2 vcap delta : R,
  0 < i0 -> 0 < c -> 0 < k -> 0 <= v1 -> 0 < v2 -> v1 + delta <= vcap ->
  0 < delta -> delta <= v2 ->
  k / (i0 + c * (v1 + delta)) < k / (i0 + c * v1).
Proof.
  intros i0 c k v1 v2 vcap delta Hi Hc Hk Hv1 Hv2 Hcap Hd Hdv.
  apply earlier_is_strictly_cheaper; nra.
Qed.
