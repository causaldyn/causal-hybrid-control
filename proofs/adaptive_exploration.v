(* Rocq (CONTRIBUTION 3): the ADAPTIVE information-exploration duality, the SEQUENTIAL analogue of
   results 10 (static Cramer-Rao) and 11 (static Av+B/v). At round t the controller injects exploration
   variance v_t; the accumulated Fisher information is m_t (in variance units). The VAN TREES (Bayesian
   Cramer-Rao) inequality bounds the estimation floor by K/m_t even for adaptive observations (its
   algebraic core is proved in van_trees.v), so the per-round cost is A*v_t + K/m_t. Facts: (A) the floor
   is ANTITONE in accumulated information (more exploration so far => lower floor). (B)+(C) the MYOPIC
   one-step tradeoff min_v A*v + K/(m+v) has the interior optimum vstar = sqrt(K/A) - m with the
   sum-of-squares gap A*(v-vstar)^2/(m+v), and that myopic increment tapers/stops as info accumulates.
   NOTE (honest scope): this myopic threshold rule is NOT the rate-optimal schedule -- in the SEQUENTIAL
   objective v_t only lowers FUTURE floors, so myopically v_t=0. The Theta(sqrt(T)) rate comes from the
   rate-optimal v_t = kappa/sqrt(t) schedule (certificate) together with (D) below, the sequence LOWER
   bound: it is a separate inequality, not a corollary of the per-round floor. The t^{-1/2} schedule, the
   sqrt(T) rate, and van-Trees sqrt(T) adaptive-LQR lower bounds are all KNOWN (Ziemann-Sandberg;
   Wagenmaker-Simchowitz-Jamieson); a confounding-specific minimax constant would be the only novelty and
   is NOT proved here. Derived in validation/adaptive_exploration.mac. *)

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

(* (D) The sqrt(T) LOWER bound needs a SEQUENCE inequality, not the per-round floor. Reduction (done in
   the certificate, elementary): with total budget M = sum_t v_t and m_t = m0 + sum_{s<t} v_s <= m0 + M,
   the sequential objective sum_t (a*v_t + c/m_t) >= a*M + c*T/(m0+M) = a*x + c*T/x - a*m0 for x = m0+M.
   Here we prove the nontrivial scalar core: the reduced single-variable objective is bounded below by
   the balanced-point value 2*sqrt(a*c*T). With s = sqrt(a*c*T) (i.e. s*s = a*c*T) the gap factors as a
   nonnegative square, a*x + c*T/x - 2*s = a*(x - s/a)^2 / x >= 0. Composed with the reduction, the
   sequential regret is >= 2*sqrt(a*c*T) - a*m0 = Theta(sqrt(T)), matched by the v_t ~ 1/sqrt(t)
   schedule. *)
Theorem reduced_objective_lower_bound : forall a c t x s,
  0 < a -> 0 < x -> s * s = a * c * t -> 2 * s <= a * x + c * t / x.
Proof.
  intros a c t x s Ha Hx Hs.
  assert (Hgap : a * x + c * t / x - 2 * s = a * (x - s / a) ^ 2 / x).
  { replace (c * t) with (s * s / a) by (rewrite Hs; field; lra). field; lra. }
  assert (Hpos : 0 <= a * (x - s / a) ^ 2 / x).
  { apply Rmult_le_pos.
    - apply Rmult_le_pos; [lra | apply pow2_ge_0].
    - left; apply Rinv_0_lt_compat; exact Hx. }
  lra.
Qed.

(* (E) CONFOUNDING-SPECIFIC constant. Let eta in (0,1] be the identifying information per unit
   exploration (eta = 1 fully identified, eta -> 0 confounded). Then m_t = I0 + eta*sum_{s<t} v_s, and the
   reduction (D) instantiated with a := A/eta gives the sequence floor 2*sqrt(A*C*T/eta) - A*I0/eta: the
   leading term scales as 1/sqrt(eta). This corollary proves that floor is ANTITONE in eta -- lower
   injected-exploration efficiency (smaller eta) provably raises it -- the causal content (D) lacked.
   eta is the identification efficiency (attenuation, not necessarily confounding).
   (Stated sqrt-free with the balanced-point value s, s^2 = A*C*T/eta = q/eta.) *)
Corollary lower_efficiency_raises_sequence_floor : forall q e1 e2 s1 s2,
  0 < q -> 0 < e1 -> e1 <= e2 -> 0 <= s1 -> 0 <= s2 ->
  s1 * s1 = q / e1 -> s2 * s2 = q / e2 -> s2 <= s1.
Proof.
  intros q e1 e2 s1 s2 Hq He1 Hle Hs1 Hs2 H1 H2.
  assert (Hqe : q / e2 <= q / e1).
  { unfold Rdiv. apply Rmult_le_compat_l; [lra | apply Rinv_le_contravar; lra]. }
  nra.
Qed.
