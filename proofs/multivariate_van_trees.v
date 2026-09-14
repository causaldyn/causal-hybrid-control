(* Rocq: the algebraic core of MULTIVARIATE VAN TREES ON THE ACTION (Result 67, Appendix A item
   A18).

   Result 57 removed Result 10's unbiasedness and delta-method caveats by noticing that the
   one-step LQ regret is EXACTLY a squared error in the action, and then applying van Trees to the
   estimand psi(b) = u*(b). Its honest scope closed with: "the model is still the scalar one-step
   LQ plant; the multivariate case has the same structure with psi' a Jacobian and the floor a
   trace." validation/multivariate_van_trees.mac does that case, and the structure is not the only
   thing that changes: the scalar floor is a PRODUCT of a curvature and an information, while the
   multivariate floor is a TRACE that interleaves them -- so the price of confounding stops being a
   ratio and becomes an ALIGNMENT between the directions information is lost in and the directions
   the optimal action depends on.

   What is proved here, all at 2x2 over Stdlib's reals:
     (A) the regret identity, for any symmetric invertible curvature -- no linearisation anywhere;
     (B) trace against a PSD weight is nonnegative, hence monotone in the PSD order, which is what
         turns the matrix van Trees inequality into a scalar regret bound;
     (C) the scalar floor as the 1x1 case, recovering Result 10's constant unchanged;
     (D) a direction the action cannot see contributes exactly nothing -- the knife edge stops
         being a point and becomes a rank condition;
     (E) the diagonal plant, where the floor is a SUM of per-channel Result-10 constants, and the
         confounding factor is their convex combination -- equal to the full factor only when the
         confounded direction carries all the weight, and exactly 1 at the other corner.

   Honest scope, as in action_van_trees.v: the van Trees inequality itself is a cited
   measure-theoretic input (Gill-Levit 1995), not formalised here. What is formalised is the
   algebra that turns it into a control-regret floor. Dimensions above 2 stay in Maxima. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Psatz.
Open Scope R_scope.

(* Stdlib has Rdiv_lt_0_compat but not a non-strict companion under that name; capped_exploration.v
   carries the same helper for the same reason. *)
Lemma nonneg_div : forall x y : R, 0 <= x -> 0 < y -> 0 <= x / y.
Proof.
  intros x y Hx Hy.
  unfold Rdiv.
  apply Rmult_le_pos; [exact Hx | left; apply Rinv_0_lt_compat; exact Hy].
Qed.

Lemma square_zero : forall x : R, x * x = 0 -> x = 0.
Proof.
  intros x H. destruct (Rmult_integral _ _ H); assumption.
Qed.

(* ===== (A) THE REGRET IS AN EXACT QUADRATIC FORM IN THE ACTION ===== *)

(* J(u) = u'Mu + 2c'u, with M = B'QB + R the exact curvature and c = B'Qx. The minimiser is
   u* = -M^(-1)c, written out through the 2x2 inverse. *)
Definition cost (m11 m12 m22 c1 c2 u1 u2 : R) : R :=
  m11 * u1 ^ 2 + 2 * m12 * u1 * u2 + m22 * u2 ^ 2 + 2 * (c1 * u1 + c2 * u2).

Definition form (m11 m12 m22 v1 v2 : R) : R :=
  m11 * v1 ^ 2 + 2 * m12 * v1 * v2 + m22 * v2 ^ 2.

Lemma regret_is_an_exact_quadratic_form_in_the_action :
  forall m11 m12 m22 c1 c2 u1 u2 : R,
  m11 * m22 - m12 ^ 2 <> 0 ->
  let d := m11 * m22 - m12 ^ 2 in
  let s1 := (m12 * c2 - m22 * c1) / d in
  let s2 := (m12 * c1 - m11 * c2) / d in
  cost m11 m12 m22 c1 c2 u1 u2 - cost m11 m12 m22 c1 c2 s1 s2
  = form m11 m12 m22 (u1 - s1) (u2 - s2).
Proof.
  intros m11 m12 m22 c1 c2 u1 u2 Hd d s1 s2.
  unfold cost, form, s1, s2, d. field. exact Hd.
Qed.

(* ===== (B) TRACE AGAINST A PSD WEIGHT, AND MONOTONICITY IN THE PSD ORDER ===== *)

Lemma a_psd_form_is_nonnegative :
  forall d11 d12 d22 v1 v2 : R,
  0 <= d11 -> 0 <= d22 -> 0 <= d11 * d22 - d12 ^ 2 ->
  0 <= form d11 d12 d22 v1 v2.
Proof.
  intros d11 d12 d22 v1 v2 H11 H22 Hdet.
  unfold form.
  destruct (Rle_lt_or_eq_dec 0 d11 H11) as [Hpos | Hzero].
  - (* completing the square: d11*(v1 + (d12/d11) v2)^2 + ((d11 d22 - d12^2)/d11) v2^2 *)
    assert (Hsq : 0 <= d11 * (v1 + d12 / d11 * v2) ^ 2)
      by (apply Rmult_le_pos; [lra | apply pow2_ge_0]).
    assert (Htail : 0 <= (d11 * d22 - d12 ^ 2) / d11 * v2 ^ 2)
      by (apply Rmult_le_pos; [apply nonneg_div; lra | apply pow2_ge_0]).
    assert (Hsplit : d11 * v1 ^ 2 + 2 * d12 * v1 * v2 + d22 * v2 ^ 2
                     = d11 * (v1 + d12 / d11 * v2) ^ 2 + (d11 * d22 - d12 ^ 2) / d11 * v2 ^ 2)
      by (field; lra).
    lra.
  - (* a zero diagonal entry forces the off-diagonal to vanish, by the determinant condition *)
    assert (Hd12 : d12 = 0) by nra.
    rewrite Hd12, <- Hzero.
    assert (0 <= d22 * v2 ^ 2) by (apply Rmult_le_pos; [exact H22 | apply pow2_ge_0]).
    lra.
Qed.

(* tr(M D) written through a Cholesky-type factor M = L L', whose columns are (l11, l21) and
   (0, l22): cyclicity turns the trace into a sum of quadratic forms in D. *)
Lemma trace_of_a_factored_weight_splits_into_forms :
  forall l11 l21 l22 d11 d12 d22 : R,
  (l11 * l11 + 0) * d11 + (l11 * l21 + 0) * d12 + (l21 * l11 + l22 * 0) * d12
  + (l21 * l21 + l22 * l22) * d22
  = form d11 d12 d22 l11 l21 + form d11 d12 d22 0 l22.
Proof.
  intros. unfold form. ring.
Qed.

Lemma trace_against_a_psd_weight_is_nonnegative :
  forall l11 l21 l22 d11 d12 d22 : R,
  0 <= d11 -> 0 <= d22 -> 0 <= d11 * d22 - d12 ^ 2 ->
  0 <= form d11 d12 d22 l11 l21 + form d11 d12 d22 0 l22.
Proof.
  intros l11 l21 l22 d11 d12 d22 H11 H22 Hdet.
  assert (H1 : 0 <= form d11 d12 d22 l11 l21) by (apply a_psd_form_is_nonnegative; assumption).
  assert (H2 : 0 <= form d11 d12 d22 0 l22) by (apply a_psd_form_is_nonnegative; assumption).
  lra.
Qed.

(* Gill & Levit (1995) Bernoulli 1:59-79; van der Vaart (1998) Thm 2.5.2, MATRIX form -- the cited
   input this whole file exists to consume, named so that `Check` says where it enters (plans/24
   P1.2). For ANY estimator of theta, biased or not, the Bayes error covariance DOMINATES
   Psi' G^-1 Psi'^T in the PSD order; at 2x2 that is Sylvester's criterion on the difference. The
   algebraic core of van Trees is proved in proofs/van_trees.v; what this name stands for is the
   measure-theoretic wrapper, which is cited and not formalised. A Definition and not an Axiom, for
   the reason spelled out in proofs/c2_end_to_end.v. *)
Definition PsdDominates (s11 s12 s22 f11 f12 f22 : R) : Prop :=
  0 <= s11 - f11 /\ 0 <= s22 - f22 /\ 0 <= (s11 - f11) * (s22 - f22) - (s12 - f12) ^ 2.

(* The consequence that matters: a lower bound on the error covariance in the PSD order is a lower
   bound on the regret. Sigma - Floor PSD gives tr(M Sigma) >= tr(M Floor). *)
Lemma the_psd_order_transfers_to_the_regret :
  forall l11 l21 l22 s11 s12 s22 f11 f12 f22 : R,
  PsdDominates s11 s12 s22 f11 f12 f22 ->
  form f11 f12 f22 l11 l21 + form f11 f12 f22 0 l22
  <= form s11 s12 s22 l11 l21 + form s11 s12 s22 0 l22.
Proof.
  intros l11 l21 l22 s11 s12 s22 f11 f12 f22 [H1 [H2 Hdet]].
  assert (Hgap : 0 <= form (s11 - f11) (s12 - f12) (s22 - f22) l11 l21
                     + form (s11 - f11) (s12 - f12) (s22 - f22) 0 l22)
    by (apply trace_against_a_psd_weight_is_nonnegative; assumption).
  unfold form in *. lra.
Qed.

(* ===== (C) THE SCALAR CASE, AND RESULT 10'S CONSTANT ===== *)

Definition scalar_constant (rr b xt : R) : R := xt ^ 2 * (rr - b ^ 2) ^ 2 / (rr + b ^ 2) ^ 3.

Lemma the_one_by_one_floor_is_result_ten_s_constant :
  forall rr b xt : R,
  rr + b ^ 2 <> 0 ->
  (b ^ 2 + rr) * (- xt * (rr - b ^ 2) / (rr + b ^ 2) ^ 2) ^ 2 = scalar_constant rr b xt.
Proof.
  intros rr b xt H. unfold scalar_constant. field. exact H.
Qed.

(* ===== (D) A DIRECTION THE ACTION CANNOT SEE COSTS NOTHING ===== *)

Lemma a_rank_one_trace_is_a_quadratic_form :
  forall m11 m12 m22 w1 w2 : R,
  m11 * (w1 * w1) + m12 * (w2 * w1) + m12 * (w1 * w2) + m22 * (w2 * w2)
  = form m11 m12 m22 w1 w2.
Proof.
  intros. unfold form. ring.
Qed.

Lemma a_blind_direction_contributes_nothing :
  forall m11 m12 m22 p11 p12 p21 p22 v1 v2 : R,
  p11 * v1 + p12 * v2 = 0 -> p21 * v1 + p22 * v2 = 0 ->
  form m11 m12 m22 (p11 * v1 + p12 * v2) (p21 * v1 + p22 * v2) = 0.
Proof.
  intros m11 m12 m22 p11 p12 p21 p22 v1 v2 H1 H2.
  rewrite H1, H2. unfold form. ring.
Qed.

(* And the converse direction that makes it a RANK condition rather than a knife edge: with a
   definite curvature the contribution vanishes only when the image direction does. *)
Lemma a_definite_curvature_hides_nothing_else :
  forall m11 m12 m22 w1 w2 : R,
  0 < m11 -> 0 < m11 * m22 - m12 ^ 2 ->
  form m11 m12 m22 w1 w2 = 0 -> w1 = 0 /\ w2 = 0.
Proof.
  intros m11 m12 m22 w1 w2 Hm Hdet Hform.
  assert (Hc : 0 < (m11 * m22 - m12 ^ 2) / m11) by (apply Rdiv_lt_0_compat; lra).
  assert (Hsplit : form m11 m12 m22 w1 w2
                   = m11 * (w1 + m12 / m11 * w2) ^ 2 + (m11 * m22 - m12 ^ 2) / m11 * w2 ^ 2)
    by (unfold form; field; lra).
  assert (Htail : 0 <= (m11 * m22 - m12 ^ 2) / m11 * w2 ^ 2)
    by (apply Rmult_le_pos; [lra | apply pow2_ge_0]).
  assert (Hhead : 0 <= m11 * (w1 + m12 / m11 * w2) ^ 2)
    by (apply Rmult_le_pos; [lra | apply pow2_ge_0]).
  assert (Hw2 : w2 = 0).
  { apply square_zero.
    apply (Rmult_eq_reg_l ((m11 * m22 - m12 ^ 2) / m11)); [| lra].
    rewrite Rmult_0_r. nra. }
  split; [| exact Hw2].
  assert (Hrest : m11 * (w1 + m12 / m11 * w2) ^ 2 = 0) by nra.
  rewrite Hw2 in Hrest.
  apply square_zero.
  apply (Rmult_eq_reg_l m11); [| lra].
  rewrite Rmult_0_r. nra.
Qed.

(* ===== (E) THE DIAGONAL PLANT: A SUM OF SCALAR CONSTANTS, AND A CONVEX COMBINATION ===== *)

(* Two decoupled channels: Psi' is diagonal, M is diagonal, and the floor separates. *)
Lemma the_diagonal_floor_is_a_sum_of_scalar_constants :
  forall rr b1 b2 x1 x2 h1 h2 : R,
  rr + b1 ^ 2 <> 0 -> rr + b2 ^ 2 <> 0 ->
  (b1 ^ 2 + rr) * (- x1 * (rr - b1 ^ 2) / (rr + b1 ^ 2) ^ 2) ^ 2 * h1
  + (b2 ^ 2 + rr) * (- x2 * (rr - b2 ^ 2) / (rr + b2 ^ 2) ^ 2) ^ 2 * h2
  = h1 * scalar_constant rr b1 x1 + h2 * scalar_constant rr b2 x2.
Proof.
  intros rr b1 b2 x1 x2 h1 h2 H1 H2. unfold scalar_constant. field. split; assumption.
Qed.

(* The headline. Confounding that destroys information along theta_2 alone multiplies h2 by k; the
   floor ratio is the convex combination of 1 and k with the per-channel weights. *)
Lemma the_confounding_factor_is_a_convex_combination :
  forall w1 w2 k : R,
  0 < w1 -> 0 < w2 -> 1 <= k ->
  1 <= (w1 + k * w2) / (w1 + w2) <= k.
Proof.
  intros w1 w2 k H1 H2 Hk.
  assert (Hsum : 0 < w1 + w2) by lra.
  unfold Rdiv.
  split; apply (Rmult_le_reg_r (w1 + w2)); try exact Hsum;
    rewrite Rmult_assoc, Rinv_l by lra; nra.
Qed.

(* One corner: a confounded direction the action does not depend on is EXACTLY free. In the
   diagonal plant that happens at the scalar knife edge rr = b2^2, where Result 57(d)'s vanishing
   sensitivity zeroes the whole second weight. *)
Lemma the_knife_edge_makes_a_confounded_direction_free :
  forall b2 x2 : R, b2 <> 0 -> scalar_constant (b2 ^ 2) b2 x2 = 0.
Proof.
  intros b2 x2 Hb. unfold scalar_constant.
  assert (Hpos : 0 < b2 ^ 2) by (destruct (Rdichotomy b2 0 Hb); nra).
  replace (b2 ^ 2 - b2 ^ 2) with 0 by ring.
  field. nra.
Qed.

Lemma a_weightless_direction_costs_nothing :
  forall w1 k : R, 0 < w1 -> (w1 + k * 0) / (w1 + 0) = 1.
Proof.
  intros w1 k H. field. lra.
Qed.

(* The other corner is the scalar model: one direction, so the factor is the full k -- which is
   exactly Result 10's V_exp/V_conf, and the reason a scalar plant cannot see the alignment. *)
Lemma a_single_direction_pays_the_full_factor :
  forall w2 k : R, 0 < w2 -> (0 + k * w2) / (0 + w2) = k.
Proof.
  intros w2 k H. field. lra.
Qed.

(* And the floor is antitone in the information, at every direction separately. *)
Lemma more_information_lowers_the_floor :
  forall c1 c2 h1 h2 g1 g2 : R,
  0 <= c1 -> 0 <= c2 -> g1 <= h1 -> g2 <= h2 ->
  c1 * g1 + c2 * g2 <= c1 * h1 + c2 * h2.
Proof.
  intros c1 c2 h1 h2 g1 g2 Hc1 Hc2 H1 H2. nra.
Qed.
