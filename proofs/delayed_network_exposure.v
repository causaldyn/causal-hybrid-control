(* Rocq: the algebraic core of the DELAYED NETWORK EXPOSURE law.

   validation/delayed_exposure_gate.mac STEP 7 showed that a delay is FREE under a separable
   covariance Sigma = Sigma_u (x) T with unit-level folds: tr(T) cancels and Result 43 transfers
   verbatim. It left exactly one escape route -- a NON-separable Sigma, i.e. a delay that interacts
   with the network. validation/delayed_network_exposure.mac takes that route and derives the law.

   The model: a shell-d neighbour's influence arrives with propagation lag delta*d, so
   Sigma = sum_{d,e} g_d g_e (S_d S_e) (x) T_{d-e} with T_l[t,s] = phi^|(t-s) - delta*l|. Because
   tr(S_d S_e) = 0 for d <> e (a vertex cannot sit at two distinct distances from i), the DENOMINATOR
   of the sandwich is delay-blind and the whole delay dependence collapses into a polynomial in
   x := phi^delta whose coefficients are fold-shell overlap traces.

   Honest scope, as in cluster_fold_leakage.v and van_trees.v: Rocq proves the ALGEBRA over reals.
   Stdlib has no matrices, so the projector identities (A_u = r^2 P_fold + P_within, its trace
   values, and tr(S_d S_e) = 0 for d <> e) come from the Maxima file and enter here as hypotheses
   or as scalar shadows. What is proved here is everything downstream of them. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Lia.
Open Scope R_scope.

Lemma pow4_gt_one : forall r, 1 < r -> 1 < r ^ 4.
Proof.
  intros r Hr. replace (r ^ 4) with ((r * r) * (r * r)) by ring.
  assert (1 < r * r) by nra. nra.
Qed.

Lemma cubic_pos : forall K, 2 <= K -> 0 < 4 * K ^ 3 - 6 * K ^ 2 + 4 * K - 1.
Proof.
  intros K HK.
  assert (H : 4 * K ^ 3 - 6 * K ^ 2 + 4 * K - 1 = 2 * K ^ 2 * (2 * K - 3) + (4 * K - 1)) by ring.
  assert (0 < 2 * K ^ 2 * (2 * K - 3)) by nra. lra.
Qed.

(* r = K/(K-1) is the only constant the fold count contributes. *)
Definition rfold (K : R) : R := K / (K - 1).

(* (A) The sandwich splits along A_u's eigenspaces. A_u = r^2 P_fold + P_within kills span(1),
   scales the K-1 fold contrasts by r^2 and leaves the m-K within-fold contrasts alone, so
   tr(A_u^2 X) = r^4 * (X's energy in the fold-contrast subspace) + (its energy within folds).
   ROW-LEVEL CROSS-FITTING OVER-WEIGHTS THE FOLD-CONTRAST SUBSPACE BY EXACTLY r^4, and that single
   factor is the entire penalty. *)
Definition sandwich (r ef ew : R) : R := r ^ 4 * ef + ew.

Theorem sandwich_penalty_is_r4 : forall r ef ew,
  sandwich r ef ew - (ef + ew) = (r ^ 4 - 1) * ef.
Proof. intros; unfold sandwich; ring. Qed.

(* The penalty is strictly positive exactly when the covariance puts energy on fold contrasts. *)
Theorem sandwich_gt_when_fold_energy : forall r ef ew,
  1 < r -> 0 < ef -> ef + ew < sandwich r ef ew.
Proof.
  intros r ef ew Hr Hef. unfold sandwich.
  assert (1 < r ^ 4) by (apply pow4_gt_one; assumption). nra.
Qed.

(* r^4 decreases in K: MORE FOLDS is the primary lever, since r = 1 + 1/(K-1). *)
Theorem rfold_decreasing : forall K L, 2 <= K -> K < L -> rfold L < rfold K.
Proof.
  intros K L HK HKL. unfold rfold.
  assert (HK1 : 0 < K - 1) by lra. assert (HL1 : 0 < L - 1) by lra.
  apply Rmult_lt_reg_r with ((K - 1) * (L - 1)); [nra|].
  replace (L / (L - 1) * ((K - 1) * (L - 1))) with (L * (K - 1)) by (field; lra).
  replace (K / (K - 1) * ((K - 1) * (L - 1))) with (K * (L - 1)) by (field; lra).
  nra.
Qed.

Theorem rfold_gt_one : forall K, 2 <= K -> 1 < rfold K.
Proof.
  intros K HK. unfold rfold.
  assert (HK1 : 0 < K - 1) by lra.
  apply Rmult_lt_reg_r with (K - 1); [lra|].
  replace (K / (K - 1) * (K - 1)) with K by (field; lra). lra.
Qed.

(* (B) THE DESIGN LAW. For nearest-neighbour spillover, u_1 is EXACTLY AFFINE in the same-fold edge
   fraction theta, with slope (r^4-1)K and intercept -r^4 (both times 4 g0 g1 |E| / m). Its root
   theta* = r^4 / ((r^4 - 1) K) is a pure function of K -- independent of the graph. *)
Definition u1 (r K theta : R) : R := - r ^ 4 + (r ^ 4 - 1) * K * theta.
Definition theta_star (r K : R) : R := r ^ 4 / ((r ^ 4 - 1) * K).

Theorem u1_root_at_theta_star : forall r K,
  1 < r -> 0 < K -> u1 r K (theta_star r K) = 0.
Proof.
  intros r K Hr HK. unfold u1, theta_star.
  assert (Hr4 : 1 < r ^ 4) by (apply pow4_gt_one; assumption).
  replace ((r ^ 4 - 1) * K * (r ^ 4 / ((r ^ 4 - 1) * K))) with (r ^ 4) by (field; nra).
  ring.
Qed.

Theorem u1_affine_increasing : forall r K t1 t2,
  1 < r -> 0 < K -> t1 < t2 -> u1 r K t1 < u1 r K t2.
Proof.
  intros r K t1 t2 Hr HK Ht. unfold u1.
  assert (Hr4 : 1 < r ^ 4) by (apply pow4_gt_one; assumption).
  assert (Hc : 0 < (r ^ 4 - 1) * K).
  { apply Rmult_lt_0_compat; [lra | exact HK]. }
  apply Rplus_lt_compat_l. apply Rmult_lt_compat_l; [exact Hc | exact Ht].
Qed.

(* A RANDOM balanced partition sits at theta = 1/K, and theta* > 1/K for every K: random folds
   ALWAYS undershoot the delay-proof point, so u_1 < 0 generically. Psi then DECREASES in phi --
   the network delay makes the row-fold sandwich understate further, i.e. intervals under-cover
   MORE, monotonically in the serial correlation. *)
Theorem theta_star_above_random : forall r K,
  1 < r -> 0 < K -> 1 / K < theta_star r K.
Proof.
  intros r K Hr HK. unfold theta_star.
  assert (Hr4 : 1 < r ^ 4) by (apply pow4_gt_one; assumption).
  apply Rmult_lt_reg_r with K; [assumption|].
  replace (1 / K * K) with 1 by (field; lra).
  replace (r ^ 4 / ((r ^ 4 - 1) * K) * K) with (r ^ 4 / (r ^ 4 - 1)) by (field; nra).
  apply Rmult_lt_reg_r with (r ^ 4 - 1); [lra|].
  replace (r ^ 4 / (r ^ 4 - 1) * (r ^ 4 - 1)) with (r ^ 4) by (field; lra).
  lra.
Qed.

Theorem random_folds_undershoot : forall r K,
  1 < r -> 0 < K -> u1 r K (1 / K) < 0.
Proof.
  intros r K Hr HK.
  rewrite <- (u1_root_at_theta_star r K Hr HK).
  apply u1_affine_increasing; auto. apply theta_star_above_random; auto.
Qed.

(* Substituting r = K/(K-1) collapses theta* to a rational function of the fold count alone. *)
Theorem theta_star_rfold_poly : forall K, 2 <= K ->
  theta_star (rfold K) K = K ^ 3 / (4 * K ^ 3 - 6 * K ^ 2 + 4 * K - 1).
Proof.
  intros K HK. unfold theta_star, rfold.
  assert (H1 : 0 < K - 1) by lra.
  assert (Hd : 0 < 4 * K ^ 3 - 6 * K ^ 2 + 4 * K - 1) by (apply cubic_pos; assumption).
  assert (Hr : (K / (K - 1)) ^ 4 - 1 = (4 * K ^ 3 - 6 * K ^ 2 + 4 * K - 1) / (K - 1) ^ 4)
    by (field; lra).
  rewrite Hr.
  replace ((K / (K - 1)) ^ 4) with (K ^ 4 / (K - 1) ^ 4) by (field; lra).
  field. repeat split; nra.
Qed.

(* theta* stays above 1/4 for every fold count -- the delay-proof fraction never collapses, because
   6K^2 - 4K + 1 > 0 has no real root. So no amount of cross-fitting makes a network delay free by
   fold count alone; the fold-to-graph ALIGNMENT still has to be chosen. *)
Theorem theta_star_above_quarter : forall K,
  2 <= K -> 1 / 4 < theta_star (rfold K) K.
Proof.
  intros K HK. rewrite theta_star_rfold_poly by assumption.
  assert (Hd : 0 < 4 * K ^ 3 - 6 * K ^ 2 + 4 * K - 1) by (apply cubic_pos; assumption).
  apply Rmult_lt_reg_r with (4 * (4 * K ^ 3 - 6 * K ^ 2 + 4 * K - 1)); [lra|].
  replace (K ^ 3 / (4 * K ^ 3 - 6 * K ^ 2 + 4 * K - 1) * (4 * (4 * K ^ 3 - 6 * K ^ 2 + 4 * K - 1)))
    with (4 * K ^ 3) by (field; lra).
  nra.
Qed.

(* (C) THE OBSTRUCTION. Psi(x) = k (u0 + u1 x + u2 x^2) on x = phi^delta in [0,1]. When u2 > 0 and
   0 < -u1 < 2 u2 the vertex is strictly inside (0,1) and the parabola opens upward, so Psi dips
   STRICTLY BELOW BOTH ENDPOINTS. Checking phi = 0 and phi = 1 therefore does NOT bound the
   row-fold variance penalty: there is an interior worst-case serial correlation.
   Witness in the Maxima file: C_6, alternating folds, gamma = (1, 1/5, 1/2), dip 1024/6951. *)
Definition psi_poly (k u0 u1c u2 x : R) : R := k * (u0 + u1c * x + u2 * x ^ 2).
Definition vertex (u1c u2 : R) : R := - u1c / (2 * u2).

Theorem vertex_strictly_interior : forall u1c u2,
  0 < u2 -> u1c < 0 -> - u1c < 2 * u2 -> 0 < vertex u1c u2 < 1.
Proof.
  intros u1c u2 H2 H1 H3. unfold vertex. split.
  - apply Rdiv_lt_0_compat; lra.
  - apply Rmult_lt_reg_r with (2 * u2); [lra|]. field_simplify; lra.
Qed.

(* The exact dip: Psi at x_star = Psi(0) - k u1^2/(4 u2), so the shortfall is quadratic in u1. *)
Theorem dip_below_left_endpoint : forall k u0 u1c u2,
  0 < u2 -> psi_poly k u0 u1c u2 (vertex u1c u2)
            = psi_poly k u0 u1c u2 0 - k * (u1c ^ 2 / (4 * u2)).
Proof. intros k u0 u1c u2 H2. unfold psi_poly, vertex. field; lra. Qed.

Theorem interior_min_below_both_endpoints : forall k u0 u1c u2,
  0 < k -> 0 < u2 -> u1c < 0 -> - u1c < 2 * u2 ->
  psi_poly k u0 u1c u2 (vertex u1c u2) < psi_poly k u0 u1c u2 0 /\
  psi_poly k u0 u1c u2 (vertex u1c u2) < psi_poly k u0 u1c u2 1.
Proof.
  intros k u0 u1c u2 Hk H2 H1 H3. split.
  - rewrite dip_below_left_endpoint by assumption.
    assert (Hd : 0 < k * (u1c ^ 2 / (4 * u2))).
    { apply Rmult_lt_0_compat; [exact Hk|]. apply Rdiv_lt_0_compat; nra. }
    lra.
  - unfold psi_poly, vertex.
    replace (k * (u0 + u1c * (- u1c / (2 * u2)) + u2 * (- u1c / (2 * u2)) ^ 2))
      with (k * u0 - k * (u1c ^ 2 / (4 * u2))) by (field; lra).
    assert (Hgap : k * (u0 + u1c * 1 + u2 * 1 ^ 2) - (k * u0 - k * (u1c ^ 2 / (4 * u2)))
                   = k * ((2 * u2 + u1c) ^ 2 / (4 * u2))) by (field; lra).
    assert (Hs : 0 < 2 * u2 + u1c) by lra.
    assert (Hp : 0 < k * ((2 * u2 + u1c) ^ 2 / (4 * u2))).
    { apply Rmult_lt_0_compat; [exact Hk|]. apply Rdiv_lt_0_compat; nra. }
    lra.
Qed.

(* Psi is a ratio of variances, so it is nonnegative on [0,1] whatever the coefficients do. The
   obstruction above is therefore a dip, never a sign change: the vertex value stays >= 0. *)
Theorem vertex_value_nonneg_iff : forall k u0 u1c u2,
  0 < k -> 0 < u2 ->
  (0 <= psi_poly k u0 u1c u2 (vertex u1c u2) <-> u1c ^ 2 <= 4 * u0 * u2).
Proof.
  intros k u0 u1c u2 Hk H2. unfold psi_poly, vertex.
  replace (k * (u0 + u1c * (- u1c / (2 * u2)) + u2 * (- u1c / (2 * u2)) ^ 2))
    with (k * (4 * u0 * u2 - u1c ^ 2) * / (4 * u2)) by (field; lra).
  assert (Hi : 0 < / (4 * u2)) by (apply Rinv_0_lt_compat; lra).
  split; intros H.
  - apply Rnot_lt_le. intros Hc.
    assert (Hpos : 0 < k * (u1c ^ 2 - 4 * u0 * u2) * / (4 * u2))
      by (apply Rmult_lt_0_compat; [apply Rmult_lt_0_compat; lra | exact Hi]).
    assert (Hsum : k * (4 * u0 * u2 - u1c ^ 2) * / (4 * u2)
                 = - (k * (u1c ^ 2 - 4 * u0 * u2) * / (4 * u2))) by ring.
    lra.
  - apply Rmult_le_pos; [apply Rmult_le_pos; lra | lra].
Qed.

(* (D) THE SEPARABLE LIMIT, recovered from the opposite direction. At delta = 0 every band carries
   the same time kernel, x = phi^0 = 1 identically, and the polynomial collapses to its coefficient
   sum -- a constant in phi. This is STEP 7 of the gate file: A DELAY IS FREE WHEN THE COVARIANCE
   IS SEPARABLE, no matter how strong the serial correlation. *)
Theorem separable_limit_kills_phi : forall k u0 u1c u2 phi1 phi2,
  psi_poly k u0 u1c u2 (phi1 ^ 0) = psi_poly k u0 u1c u2 (phi2 ^ 0).
Proof. intros; unfold psi_poly; simpl; ring. Qed.

(* (E) COMPOSITION with Result 43. At delta = 0 on a complete graph the law reduces to the
   exchangeable one, Psi = c(m,K) * (1 - rho_ICC), with the fold-geometry constant
   c(m,K) = m * tr(A_u^2) / tr(A_u)^2 -- the quantity the shipped certificate only MEASURES.
   Its two trace values come from the same projector algebra as (A). *)
Definition c_fold (m r K : R) : R :=
  m * (m - r ^ 4 + (r ^ 4 - 1) * K) / (m - r ^ 2 + (r ^ 2 - 1) * K) ^ 2.

Theorem c_fold_at_K2 : forall m, 0 < m -> c_fold m 2 2 = m * (m + 14) / (m + 2) ^ 2.
Proof. intros m Hm. unfold c_fold. field. nra. Qed.

(* Singleton folds (K = m) do NOT remove the penalty: c(m,m) = m/(m-1) > 1. Fold COUNT alone
   never buys exactness -- it only shrinks r^4 towards 1. *)
Theorem c_fold_singleton_folds : forall m r,
  1 < m -> 0 < r -> c_fold m r m = m / (m - 1).
Proof.
  intros m r Hm Hr. unfold c_fold.
  replace (m - r ^ 4 + (r ^ 4 - 1) * m) with (r ^ 4 * (m - 1)) by ring.
  replace (m - r ^ 2 + (r ^ 2 - 1) * m) with (r ^ 2 * (m - 1)) by ring.
  field. split; nra.
Qed.

(* At K = 2 the penalty is EXACTLY (10m-4)/(m+2)^2: positive at every cluster size, decaying like
   10/m. This is the cluster-SIZE counterpart of the O(1/G) cluster-COUNT leak of Result 43. *)
Theorem c_fold_excess_at_K2 : forall m,
  0 < m -> c_fold m 2 2 - 1 = (10 * m - 4) / (m + 2) ^ 2.
Proof. intros m Hm. unfold c_fold. field. nra. Qed.

Theorem c_fold_excess_positive : forall m, 1 <= m -> 1 < c_fold m 2 2.
Proof.
  intros m Hm.
  assert (H : c_fold m 2 2 - 1 = (10 * m - 4) / (m + 2) ^ 2) by (apply c_fold_excess_at_K2; lra).
  assert (Hp : 0 < (10 * m - 4) / (m + 2) ^ 2) by (apply Rdiv_lt_0_compat; nra).
  lra.
Qed.

(* ------------------------------------------------------------------------------------------- *)
(* THE DESIGN CROSSOVER. Two fold partitions of the same graph do not have a fixed ranking: the
   one that is better at low persistence is worse at high persistence, and they swap at a single
   point. That point is a root of the DIFFERENCE of their coefficient vectors, with no
   normalisation surviving -- which is what the next two lemmas establish. *)

(* tr(A_u) = 0*1 + r^2*(K-1) + 1*(m-K) counts eigenvalue MULTIPLICITIES, so it sees how many folds
   there are and never which unit went into which. Both partitions therefore share the normaliser,
   and delayed_network_certificate measures the gap as exactly 0. *)
Theorem fold_trace_partition_free : forall m r K,
  r ^ 2 * (K - 1) + (m - K) = m - r ^ 2 + (r ^ 2 - 1) * K.
Proof. intros m r K. ring. Qed.

(* Psi = c * (u . x^l) with c = m^2/(tr(Au)^2 v0) shared, so equal Psi is a root of the difference
   and nothing else. Stated on the value of the polynomial, which is all the scalar shadow needs. *)
Theorem crossover_is_difference_root : forall c p1 p2,
  c <> 0 -> (c * p1 = c * p2 <-> p1 - p2 = 0).
Proof.
  intros c p1 p2 Hc. split; intro H.
  - apply Rminus_diag_eq. apply (Rmult_eq_reg_l c); assumption.
  - apply Rminus_diag_uniq in H. rewrite H. reflexivity.
Qed.

(* The C_6 instance with gammas = (1, 7/10, 2/5) and K = 2, from validation/delayed_network_
   exposure.mac: u_parity - u_block = (26, -392/5, 32), whose root in (0,1) is exactly this. *)
Definition x_star : R := (49 - sqrt 1101) / 40.

Lemma sqrt_1101_bounds : 9 < sqrt 1101 < 49.
Proof.
  assert (H0 : 0 <= sqrt 1101) by apply sqrt_pos.
  assert (Hs : sqrt 1101 * sqrt 1101 = 1101) by (apply sqrt_sqrt; lra).
  split; nra.
Qed.

Theorem x_star_is_the_crossover : 32 * x_star ^ 2 - (392 / 5) * x_star + 26 = 0.
Proof.
  unfold x_star.
  assert (Hs : sqrt 1101 * sqrt 1101 = 1101) by (apply sqrt_sqrt; lra).
  field_simplify. nra.
Qed.

Theorem x_star_in_unit_interval : 0 < x_star < 1.
Proof.
  destruct sqrt_1101_bounds as [Hlo Hhi]. unfold x_star. split; lra.
Qed.

(* The operational half. The crossover is a threshold on x = phi^delta, NOT on phi, so the same
   persistence lands on opposite sides of it at different delays: phi^delta shrinks as delta grows,
   pushing a longer-delay design toward the ALIGNED partition at a persistence where the
   shorter-delay one wants the alternating partition. This is why delta has to be estimated and not
   assumed -- chc.network_causal.estimate_propagation is what supplies it. *)
Theorem longer_delay_favours_alignment : forall phi d,
  0 < phi < 1 -> phi ^ d < x_star -> phi ^ S d < x_star.
Proof.
  intros phi d [Hlo Hhi] H. simpl.
  assert (Hp : 0 < phi ^ d) by (apply pow_lt; assumption).
  nra.
Qed.

(* And it never escapes the interval where a crossover can exist at all. *)
Theorem crossover_threshold_below_one : forall d, (1 <= d)%nat -> x_star ^ d < 1.
Proof.
  intros d Hd. destruct x_star_in_unit_interval as [Hlo Hhi].
  apply pow_lt_1_compat; [lra | lia].
Qed.

(* ------------------------------------------------------------------------------------------- *)
(* THE D = 1 CROSSOVER IN CLOSED FORM. The natural conjecture -- that x* is graph-free, as theta*
   is -- is FALSE, and the algebra says why. What survives the difference is a ratio of two
   fold-overlap counts that the graph, not the fold sizes, decides. *)

(* A_u^2 has eigenvalue 0 on span(1), r^4 on the K-1 fold contrasts and 1 on the m-K within-fold
   contrasts, so like tr(A_u) its trace counts multiplicities and never the assignment. That is
   what makes the whole d = 0 block cancel from the difference of two partitions. *)
Theorem sandwich_trace_partition_free : forall m r K,
  r ^ 4 * (K - 1) + (m - K) = m - r ^ 4 + (r ^ 4 - 1) * K.
Proof. intros m r K. ring. Qed.

(* With that block gone, Delta u_0 = g1^2 c dW11 and Delta u_1 = 4 g0 g1 c de_in share the factor
   c = (r^4 - 1) K / m, so r, K and m all cancel from x* = -Delta u_0 / Delta u_1. *)
Theorem crossover_D1_closed_form : forall g0 g1 c dW de,
  g0 <> 0 -> g1 <> 0 -> c <> 0 -> de <> 0 ->
  - (g1 ^ 2 * c * dW) / (4 * g0 * g1 * c * de) = - (g1 / (4 * g0)) * (dW / de).
Proof. intros g0 g1 c dW de Hg0 Hg1 Hc Hde. field. repeat split; assumption. Qed.

(* So the crossover is exactly LINEAR in the spillover decay ratio: the graph and the two fold
   partitions contribute only the integer ratio dW/de, and scaling g1 scales x* by the same factor.
   This is the sense in which x* is NOT graph-free -- same-fold EDGES do not determine same-fold
   2-WALKS, so two graphs can share (theta_1, theta_2) and disagree on x*, which they do:
   validation/delayed_network_exposure.mac STEP 11 and the C_10 / P_6 clash measured in
   tests/test_network_causal.py. *)
Theorem crossover_linear_in_spillover_decay : forall g0 g1 k dW de,
  - (k * g1 / (4 * g0)) * (dW / de) = k * (- (g1 / (4 * g0)) * (dW / de)).
Proof. intros g0 g1 k dW de. unfold Rdiv. ring. Qed.

(* And it inherits the delay threshold of (h): x* is a bound on phi^delta, so a longer delay still
   pushes the design the same way whatever the graph put into dW/de. *)
Theorem crossover_D1_threshold_shifts : forall phi d g0 g1 dW de,
  0 < phi < 1 -> phi ^ d < - (g1 / (4 * g0)) * (dW / de) ->
  phi ^ S d < - (g1 / (4 * g0)) * (dW / de).
Proof.
  intros phi d g0 g1 dW de [Hlo Hhi] H. simpl.
  assert (Hp : 0 < phi ^ d) by (apply pow_lt; assumption). nra.
Qed.
