(* Rocq (CONTRIBUTION 3): the ADAPTIVE information-exploration duality -- the rigorous SEQUENTIAL upgrade
   of results 10 (static Cramer-Rao) and 11 (static Av+B/v). At round t the controller injects
   exploration variance v_t; the accumulated Fisher information is m_t (in variance units). The VAN TREES
   (Bayesian Cramer-Rao) inequality bounds the estimation floor by K/(m_t + v_t) even for adaptive
   observations, so the per-round cost is A*v + K/(m+v). Three facts: (A) the floor is ANTITONE in
   accumulated information (more exploration so far => lower floor); (B) the per-round tradeoff has an
   interior optimum with the sum-of-squares gap A*(v-vstar)^2/(m+v) (van-Trees analog of result 11 with
   an information shift m); (C) the optimal schedule TAPERS -- vstar = sqrt(K/A) - m decreases as info
   accumulates, unlike the static single vstar of result 11. Derived in validation/adaptive_exploration.mac. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

Definition round_cost (a k m v : R) : R := a * v + k / (m + v).  (* explore cost + van-Trees floor *)

(* (A) The van-Trees estimation floor is antitone in the accumulated information: exploring more so far
   (larger info) strictly lowers the floor. *)
Theorem van_trees_floor_antitone : forall k i1 i2,
  0 <= k -> 0 < i1 -> i1 <= i2 -> k / i2 <= k / i1.
Proof.
  intros k i1 i2 Hk H1 Hle. unfold Rdiv.
  apply Rmult_le_compat_l; [exact Hk | apply Rinv_le_contravar; lra].
Qed.

(* (B) Per-round tradeoff optimum: with vstar characterised by A*(m+vstar)^2 = K, the cost gap over it
   is the sum-of-squares A*(v-vstar)^2/(m+v) >= 0 -- the AM-GM optimum of result 11 shifted by the
   accumulated information m. *)
Theorem single_round_tradeoff : forall a k m v vstar,
  0 < a -> 0 < m + v -> 0 < m + vstar -> a * (m + vstar) ^ 2 = k ->
  round_cost a k m vstar <= round_cost a k m v.
Proof.
  intros a k m v vstar Ha Hv Hvs Hchar. unfold round_cost. rewrite <- Hchar.
  assert (Hgap :
    (a * v + a * (m + vstar) ^ 2 / (m + v)) - (a * vstar + a * (m + vstar) ^ 2 / (m + vstar))
    = a * (v - vstar) ^ 2 / (m + v)) by (field; split; lra).
  assert (Hpos : 0 <= a * (v - vstar) ^ 2 / (m + v)).
  { apply Rmult_le_pos.
    - apply Rmult_le_pos; [lra | apply pow2_ge_0].
    - left; apply Rinv_0_lt_compat; lra. }
  lra.
Qed.

(* (C) The optimal exploration schedule TAPERS: as accumulated information grows (m1 <= m2), the optimal
   increment shrinks (v2 <= v1), because both keep the total information at the threshold sqrt(K/A).
   This is a decreasing schedule -- not the constant v* of the static result 11. *)
Theorem schedule_tapers : forall k a m1 v1 m2 v2,
  0 < a -> m1 <= m2 -> 0 < m1 + v1 -> 0 < m2 + v2 ->
  a * (m1 + v1) ^ 2 = k -> a * (m2 + v2) ^ 2 = k -> v2 <= v1.
Proof.
  intros k a m1 v1 m2 v2 Ha Hm H1 H2 Hc1 Hc2.
  assert (Hsq : (m1 + v1) ^ 2 = (m2 + v2) ^ 2) by (apply (Rmult_eq_reg_l a); lra).
  assert (Heq : m1 + v1 = m2 + v2) by nra.
  lra.
Qed.

(* Exploration STOPS once the accumulated information reaches the threshold: if K <= A*m^2 the optimal
   increment is non-positive (clamp to 0) -- the controller has learned enough. *)
Theorem exploration_stops_at_threshold : forall k a m vstar,
  0 < a -> 0 <= m -> a * (m + vstar) ^ 2 = k -> k <= a * m ^ 2 -> vstar <= 0.
Proof.
  intros k a m vstar Ha Hm Hchar Hthr.
  assert (Hle : (m + vstar) ^ 2 <= m ^ 2) by nra.
  nra.
Qed.
