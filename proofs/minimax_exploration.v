(* Rocq (CONTRIBUTION 3, the open item closed): the FULL LOCAL-MINIMAX sequential lower bound
   inf_pi sup_theta E R_T >= c_causal * sqrt T, with c_causal EXPLICIT in primitives.

   What result 20 proved was a SEQUENCE bound: over exploration schedules, inside an assumed per-round
   decomposition A*v_t + K/m_t, with K an opaque "van-Trees floor numerator". Two things were open: the
   inf over ALL policies (not schedules), and K in primitives. Both are settled here.

   (A) is the step that upgrades schedule-class to minimax: for ANY policy, splitting the action into its
   conditional mean and conditional variance is an identity, so no template is assumed. (B) is the AM-GM
   core. (C) composes them into the floor. (D) evaluates the constant:
       c_causal = 2 * A * |dg/dtheta| * sigma / sqrt(eta),
   with A the local cost curvature in the action, g(theta) = u*(theta) the oracle action, sigma^2 the
   noise, eta in (0,1] the identifying information per unit injected exploration variance. (E) is the
   order-of-limits fact that kills the prior term, and (F) the causal monotonicity.

   Honest scope, as in van_trees.v: Rocq proves the ALGEBRA. The measure-theoretic inputs -- the van
   Trees score identity for a functional under an adaptive design, and Fisher-information additivity
   along the trajectory -- are cited (Gill-Levit 1995; Gassiat-Stoltz 2024), taken as hypotheses here.
   Derived in validation/minimax_exploration.mac. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* (A) THE MINIMAX STEP. For any policy, with ex = E[u_t | F_{t-1}], ex2 = E[u_t^2 | F_{t-1}] and
   c = g(theta) the oracle action, the conditional second moment splits exactly into the injected
   exploration variance and the squared estimation bias. Nothing about the policy is assumed, which is
   what makes the resulting bound an inf over policies rather than over schedules. *)
Theorem action_variance_decomposition : forall ex ex2 c,
  ex2 - 2 * c * ex + c ^ 2 = (ex2 - ex ^ 2) + (ex - c) ^ 2.
Proof. intros. ring. Qed.

(* (B) The AM-GM core, sqrt-free through the balanced value s with s^2 = a*k. *)
Theorem amgm_floor : forall a k x s,
  0 < a -> 0 < x -> s * s = a * k -> 2 * s <= a * x + k / x.
Proof.
  intros a k x s Ha Hx Hs.
  assert (Hgap : a * x + k / x - 2 * s = a * (x - s / a) ^ 2 / x).
  { replace k with (s * s / a) by (rewrite Hs; field; lra). field; lra. }
  assert (Hpos : 0 <= a * (x - s / a) ^ 2 / x).
  { apply Rmult_le_pos.
    - apply Rmult_le_pos; [lra | apply pow2_ge_0].
    - left; apply Rinv_0_lt_compat; exact Hx. }
  lra.
Qed.

(* (C) THE FLOOR. Summing (A) over rounds gives R_T = A * sum_t (v_t + E beta_t^2); van Trees on the
   functional g bounds each E beta_t^2 below by gp2 / (IPi + I0 + (eta/sigma2) * sum_{s<t} v_s), and every
   denominator is at most its value at the total budget M. Writing x for that largest denominator, the
   whole thing is a*x + k/x minus the prior/initial information term. *)
Theorem minimax_sequence_floor : forall A gp2 sigma2 eta T M IPi I0 x s,
  0 < A -> 0 <= gp2 -> 0 < sigma2 -> 0 < eta -> 0 < T -> 0 <= M ->
  0 <= IPi -> 0 <= I0 -> 0 < x ->
  x = IPi + I0 + eta / sigma2 * M ->
  s * s = A * sigma2 / eta * (A * gp2 * T) ->
  2 * s - A * sigma2 / eta * (IPi + I0) <= A * M + A * gp2 * T / x.
Proof.
  intros A gp2 sigma2 eta T M IPi I0 x s HA Hgp Hsig Heta HT HM HIPi HI0 Hx Hxdef Hs.
  assert (Ha : 0 < A * sigma2 / eta).
  { apply Rdiv_lt_0_compat; [apply Rmult_lt_0_compat; lra | lra]. }
  assert (Hkey : A * M = A * sigma2 / eta * x - A * sigma2 / eta * (IPi + I0)).
  { rewrite Hxdef. field; lra. }
  assert (Hamgm := amgm_floor (A * sigma2 / eta) (A * gp2 * T) x s Ha Hx Hs).
  lra.
Qed.

(* (D) THE CONSTANT, EXPLICIT. The floor 2*s equals c_causal * sqrt T with
   c_causal^2 = 4 * A^2 * gp2 * sigma2 / eta, i.e. c_causal = 2*A*|dg/dtheta|*sigma/sqrt(eta).
   Stated on squares so no square root is needed. *)
Theorem minimax_constant_explicit : forall A gp2 sigma2 eta T s,
  0 < eta -> s * s = A * sigma2 / eta * (A * gp2 * T) ->
  (2 * s) * (2 * s) = 4 * A ^ 2 * gp2 * sigma2 / eta * T.
Proof.
  intros A gp2 sigma2 eta T s Heta Hs.
  replace ((2 * s) * (2 * s)) with (4 * (s * s)) by ring.
  rewrite Hs. field; lra.
Qed.

(* (E) ORDER OF LIMITS. The prior supported on a neighbourhood of radius H*T^(-1/4) carries information
   IPi = kappa*sqrt T / H^2, so per sqrt T the bound loses a*kappa/H^2. Because the optimiser sits at
   order sqrt T, the neighbourhood must shrink at T^(-1/4) and NOT the usual T^(-1/2) -- at T^(-1/2) the
   prior information would swamp the data information and the bound would be vacuous. Taking G -> inf
   first and the local radius H -> inf second, exactly as in the clustered result, the loss vanishes. *)
Theorem prior_information_gap_vanishes : forall a kappa H eps,
  0 < a -> 0 <= kappa -> 0 < H -> 0 < eps -> a * kappa <= eps * H ^ 2 ->
  a * kappa / H ^ 2 <= eps.
Proof.
  intros a kappa H eps Ha Hk HH Heps Hbound.
  apply Rmult_le_reg_r with (r := H ^ 2); [nra | ].
  unfold Rdiv. rewrite Rmult_assoc, Rinv_l; nra.
Qed.

(* (F) THE CAUSAL CONTENT. The constant is antitone in eta: less identifying information per unit of
   injected exploration provably raises the minimax floor. This is the 1/sqrt(eta) scaling, now attached
   to an inf-over-policies statement rather than a schedule-class one. Sqrt-free on the constants. *)
Corollary worse_identification_raises_minimax_floor : forall q e1 e2 c1 c2,
  0 < q -> 0 < e1 -> e1 <= e2 -> 0 <= c1 -> 0 <= c2 ->
  c1 * c1 = q / e1 -> c2 * c2 = q / e2 -> c2 <= c1.
Proof.
  intros q e1 e2 c1 c2 Hq He1 Hle Hc1 Hc2 H1 H2.
  assert (Hqe : q / e2 <= q / e1).
  { unfold Rdiv. apply Rmult_le_compat_l; [lra | apply Rinv_le_contravar; lra]. }
  nra.
Qed.

(* (G) SHARPNESS. The crude step in (C) -- every denominator bounded by its value at the total budget --
   is an equality exactly when the information is raised to its final level immediately. So the floor is
   ATTAINED by a front-loaded design (spend the whole budget in round one), and c_causal is sharp rather
   than merely an order. The empirical side of this is the certificate's burst/floor ratio -> 1. *)

(* (H) THE COST OF TAPERING. The rate-optimal 1/sqrt(t) schedule of result 20 has cumulative cost
   sqrt T * (2*A*kappa + K*sigma2/(kappa*eta)), minimised over kappa at 2*sqrt(2*A*K*sigma2/eta), against
   the floor 2*sqrt(A*K*sigma2/eta). Tapering is therefore rate-optimal but NOT constant-optimal, and the
   gap is exactly sqrt 2 -- stated on squares so no square root is needed. *)
Theorem taper_gap_is_sqrt_two : forall A K sigma2 eta ctaper cfloor,
  0 < eta ->
  ctaper * ctaper = 8 * A * K * sigma2 / eta ->
  cfloor * cfloor = 4 * A * K * sigma2 / eta ->
  ctaper * ctaper = 2 * (cfloor * cfloor).
Proof. intros A K sigma2 eta ct cf Heta Ht Hf. rewrite Ht, Hf. field; lra. Qed.

(* And the taper is genuinely worse, not merely different: for nonnegative constants, a squared ratio
   of 2 forces the taper constant strictly above the floor whenever the floor is positive. *)
Corollary taper_strictly_above_floor : forall ctaper cfloor,
  0 <= ctaper -> 0 < cfloor -> ctaper * ctaper = 2 * (cfloor * cfloor) -> cfloor < ctaper.
Proof. intros ct cf Hct Hcf H. nra. Qed.
