(* THE LINEAR-QUADRATIC MEAN-FIELD GAME: the algebraic core of Result 49.

   Coupling chc.deep_galerkin's backward HJB to chc.transport's forward Fokker-Planck gives a
   forward-backward system.  Under the quadratic ansatz V = P x^2/2 + S x + Z and a Gaussian
   density, validation/lq_mean_field.mac reduces it to a 2x2 linear two-point boundary value
   problem for the pair (m, S) whose transition matrix is trace-free.  Imposing the terminal
   condition S(T) = -q_T c_T m(T) leaves ONE linear equation for S(0), with the denominator

       den(T) = cosh(lam T) - k sinh(lam T) / lam,     k := A + q_T c_T b^2 / r

   where lam^2 = A^2 - q c b^2 / r.  This file proves what that denominator does: it never
   vanishes when k < lam, it vanishes at an EXPLICIT horizon when k > lam > 0, and on the
   oscillatory branch (lam imaginary) it changes sign inside (0, pi/w) for EVERY k, so the
   obstruction is unavoidable there.  Same shape as Result 45's conjugate time -- a determinant
   crossing zero and a constant blowing up like 1/(T* - T) -- but in the mean-field coupling
   rather than the single-agent second variation.

   Stdlib Reals only.  Nothing here is about probability: the reduction to these scalars is done
   in Maxima and measured by chc.deep_galerkin.lq_mean_field_certificate. *)

From Stdlib Require Import Reals Lra.
Open Scope R_scope.

(* ===== the denominator, and the identity that decides everything about it ===== *)

Definition den_real (lam k T : R) : R := cosh (lam * T) - k * sinh (lam * T) / lam.

(* Multiplying by 2 lam exp(lam T) clears both the hyperbolics and the division, turning a
   transcendental question into a quadratic in E = exp(lam T).  Every result below is a corollary. *)
Lemma den_real_pivot :
  forall lam k T,
    lam <> 0 ->
    2 * lam * exp (lam * T) * den_real lam k T
      = (lam - k) * (exp (lam * T) * exp (lam * T)) + (lam + k).
Proof.
  intros lam k T Hlam. unfold den_real, cosh, sinh.
  assert (HE : exp (lam * T) <> 0) by (apply Rgt_not_eq, exp_pos).
  rewrite exp_Ropp. field. split; assumption.
Qed.

Lemma exp_ge_one : forall x, 0 <= x -> 1 <= exp x.
Proof.
  intros x Hx. destruct (Rle_lt_or_eq_dec 0 x Hx) as [Hlt | Heq].
  - rewrite <- exp_0. left. apply exp_increasing, Hlt.
  - rewrite <- Heq, exp_0. lra.
Qed.

(* BELOW THE THRESHOLD: k < lam is enough for the mean-field fixed point to exist at EVERY
   horizon.  Note it does not need k >= 0: a very negative k only helps. *)
Theorem den_real_positive_below :
  forall lam k T, 0 < lam -> k < lam -> 0 <= T -> 0 < den_real lam k T.
Proof.
  intros lam k T Hlam Hk HT.
  assert (Hlam0 : lam <> 0) by lra.
  assert (HE : 1 <= exp (lam * T)).
  { apply exp_ge_one. apply Rmult_le_pos; lra. }
  assert (Hpos : 0 < exp (lam * T)) by apply exp_pos.
  assert (Hsq : 1 <= exp (lam * T) * exp (lam * T)) by nra.
  assert (Hc : 0 < 2 * lam * exp (lam * T)) by nra.
  apply (Rmult_lt_reg_l (2 * lam * exp (lam * T))); [exact Hc |].
  rewrite Rmult_0_r, den_real_pivot by exact Hlam0.
  (* (lam-k) E^2 + (lam+k) >= (lam-k) + (lam+k) = 2 lam > 0 *)
  nra.
Qed.

(* The interpretable sufficient condition.  With the terminal weight pinned to the stationary
   Riccati root, k = a c_T + A (1 - c_T); a <= 0 and c_T in [0,1] give k <= 0 < lam. *)
Corollary den_real_positive_nonpositive_gain :
  forall lam k T, 0 < lam -> k <= 0 -> 0 <= T -> 0 < den_real lam k T.
Proof. intros. apply den_real_positive_below; lra. Qed.

(* ===== ABOVE THE THRESHOLD: the obstruction horizon, in closed form ===== *)

Definition obstruction_horizon (lam k : R) : R := ln ((k + lam) / (k - lam)) / (2 * lam).

Lemma obstruction_ratio_gt_one :
  forall lam k, 0 < lam -> lam < k -> 1 < (k + lam) / (k - lam).
Proof.
  intros lam k Hlam Hk.
  assert (Hd : 0 < k - lam) by lra.
  apply (Rmult_lt_reg_r (k - lam)); [exact Hd |].
  unfold Rdiv. rewrite Rmult_assoc, Rinv_l by lra. lra.
Qed.

Theorem obstruction_horizon_positive :
  forall lam k, 0 < lam -> lam < k -> 0 < obstruction_horizon lam k.
Proof.
  intros lam k Hlam Hk. unfold obstruction_horizon.
  apply Rdiv_lt_0_compat; [| lra].
  rewrite <- ln_1. apply ln_increasing; [lra |].
  apply obstruction_ratio_gt_one; assumption.
Qed.

(* The horizon is EXACT, not asymptotic: den vanishes there identically. *)
Theorem den_real_vanishes_at_obstruction :
  forall lam k, 0 < lam -> lam < k -> den_real lam k (obstruction_horizon lam k) = 0.
Proof.
  intros lam k Hlam Hk.
  set (rho := (k + lam) / (k - lam)).
  assert (Hrho : 1 < rho) by (apply obstruction_ratio_gt_one; assumption).
  assert (Hrho0 : 0 < rho) by lra.
  set (T := obstruction_horizon lam k).
  (* exp(lam T) * exp(lam T) = exp(2 lam T) = exp(ln rho) = rho *)
  assert (Hsq : exp (lam * T) * exp (lam * T) = rho).
  { rewrite <- exp_plus.
    replace (lam * T + lam * T) with (ln rho).
    - apply exp_ln, Hrho0.
    - unfold T, obstruction_horizon. fold rho. field. lra. }
  assert (Hpivot := den_real_pivot lam k T (Rgt_not_eq lam 0 Hlam)).
  rewrite Hsq in Hpivot.
  (* (lam - k) * rho + (lam + k) = 0 because rho = (k+lam)/(k-lam) *)
  assert (Hzero : (lam - k) * rho + (lam + k) = 0).
  { unfold rho. field. lra. }
  rewrite Hzero in Hpivot.
  assert (HE : 0 < exp (lam * T)) by apply exp_pos.
  apply (Rmult_eq_reg_l (2 * lam * exp (lam * T))).
  - rewrite Hpivot. ring.
  - nra.
Qed.

(* Existence of an obstruction is not automatic on this branch -- k > lam is exactly the
   condition, since coth maps (0, inf) onto (1, inf).  The two theorems above and below are
   therefore a dichotomy, not a one-sided bound. *)
Theorem real_branch_dichotomy :
  forall lam k T,
    0 < lam -> 0 <= T ->
    (k < lam -> 0 < den_real lam k T) /\
    (lam < k -> den_real lam k (obstruction_horizon lam k) = 0
                /\ 0 < obstruction_horizon lam k).
Proof.
  intros lam k T Hlam HT. split; intro Hk.
  - apply den_real_positive_below; assumption.
  - split; [apply den_real_vanishes_at_obstruction | apply obstruction_horizon_positive];
      assumption.
Qed.

(* ===== THE OSCILLATORY BRANCH: the obstruction cannot be avoided at all ===== *)

Definition den_osc (w k T : R) : R := cos (w * T) - k * sin (w * T) / w.

Lemma den_osc_at_zero : forall w k, den_osc w k 0 = 1.
Proof.
  intros w k. unfold den_osc.
  rewrite Rmult_0_r, cos_0, sin_0. lra.
Qed.

Lemma den_osc_at_half_period : forall w k, w <> 0 -> den_osc w k (PI / w) = -1.
Proof.
  intros w k Hw. unfold den_osc.
  replace (w * (PI / w)) with PI by (field; exact Hw).
  rewrite cos_PI, sin_PI. lra.
Qed.

(* For EVERY k the denominator is +1 at T = 0 and -1 at T = pi/w.  A continuous function with a
   sign change has a root, so on this branch the mean-field fixed point degenerates at a horizon
   no larger than pi/w -- whatever the terminal weight.  This is what makes c > 1 + r a^2/(q b^2)
   qualitatively different from a merely large coupling. *)
Theorem oscillatory_obstruction_unavoidable :
  forall w k, 0 < w -> den_osc w k (PI / w) < 0 < den_osc w k 0.
Proof.
  intros w k Hw.
  rewrite den_osc_at_zero, den_osc_at_half_period by lra. lra.
Qed.

(* And the crossing is at an explicit horizon: arccot(k/w)/w, written with atan since Stdlib has
   no arccot.  This is the mean-field analogue of Result 45's t_conj. *)
Definition obstruction_horizon_osc (w k : R) : R := (PI / 2 - atan (k / w)) / w.

Theorem den_osc_vanishes_at_obstruction :
  forall w k, 0 < w -> den_osc w k (obstruction_horizon_osc w k) = 0.
Proof.
  intros w k Hw. unfold den_osc, obstruction_horizon_osc.
  replace (w * ((PI / 2 - atan (k / w)) / w)) with (PI / 2 - atan (k / w))
    by (field; lra).
  rewrite cos_shift, sin_shift, sin_atan, cos_atan.
  assert (Hs : 0 < sqrt (1 + (k / w)²)).
  { apply sqrt_lt_R0. assert (0 <= (k / w)²) by apply Rle_0_sqr. lra. }
  field. split; lra.
Qed.

Theorem obstruction_horizon_osc_in_range :
  forall w k, 0 < w -> 0 < obstruction_horizon_osc w k < PI / w.
Proof.
  intros w k Hw.
  assert (Hatan := atan_bound (k / w)).
  unfold obstruction_horizon_osc. split.
  - apply Rdiv_lt_0_compat; lra.
  - assert (Hinv : 0 < / w) by (apply Rinv_0_lt_compat; lra).
    apply Rmult_lt_compat_r; [exact Hinv | lra].
Qed.

(* ===== WHICH BRANCH: an exact threshold in the mean-field coupling c ===== *)

(* With the terminal weight pinned to the stationary Riccati root, A^2 = a^2 + q b^2 / r, so
   lam^2 = A^2 - q c b^2 / r collapses to a^2 + (q b^2 / r)(1 - c).  The branch therefore depends
   on c alone, measured against r a^2 / (q b^2). *)
Definition lam_sq (a b q r c : R) : R := a * a + (q * (b * b) / r) * (1 - c).

Theorem branch_threshold :
  forall a b q r c,
    0 < q -> 0 < r -> b <> 0 ->
    (0 < lam_sq a b q r c <-> c < 1 + r * (a * a) / (q * (b * b))).
Proof.
  intros a b q r c Hq Hr Hb.
  assert (Hbb : 0 < b * b) by (apply Rlt_0_sqr; exact Hb).
  assert (Hqbb : 0 < q * (b * b) / r) by (apply Rdiv_lt_0_compat; nra).
  unfold lam_sq. split; intro H.
  - apply (Rmult_lt_reg_r (q * (b * b) / r)); [exact Hqbb |].
    replace ((1 + r * (a * a) / (q * (b * b))) * (q * (b * b) / r))
      with (q * (b * b) / r + a * a) by (field; nra).
    nra.
  - apply (Rmult_lt_compat_r (q * (b * b) / r)) in H; [| exact Hqbb].
    replace ((1 + r * (a * a) / (q * (b * b))) * (q * (b * b) / r))
      with (q * (b * b) / r + a * a) in H by (field; nra).
    nra.
Qed.

(* c <= 1 is always the safe branch: the anti-monotone regime needs a running cost that penalises
   deviation from an AMPLIFIED mean. *)
Corollary monotone_coupling_is_real_branch :
  forall a b q r c,
    0 < q -> 0 < r -> b <> 0 -> a <> 0 -> c <= 1 -> 0 < lam_sq a b q r c.
Proof.
  intros a b q r c Hq Hr Hb Ha Hc.
  assert (Hbb : 0 < b * b) by (apply Rlt_0_sqr; exact Hb).
  assert (Haa : 0 < a * a) by (apply Rlt_0_sqr; exact Ha).
  assert (Hqbb : 0 < q * (b * b) / r) by (apply Rdiv_lt_0_compat; nra).
  unfold lam_sq. nra.
Qed.

(* ===== the stationary closed loop, and the sign of the gain k ===== *)

(* A = -sqrt(a^2 + q b^2 / r) is strictly negative WHATEVER the sign of a: the individual problem
   is always stabilised, so anything that goes wrong is the mean-field coupling, not the plant. *)
Theorem closed_loop_rate_negative :
  forall a b q r,
    0 < q -> 0 < r -> b <> 0 -> - sqrt (a * a + q * (b * b) / r) < 0.
Proof.
  intros a b q r Hq Hr Hb.
  assert (Hbb : 0 < b * b) by (apply Rlt_0_sqr; exact Hb).
  assert (Hpos : 0 < a * a + q * (b * b) / r).
  { assert (0 <= a * a) by apply Rle_0_sqr.
    assert (0 < q * (b * b) / r) by (apply Rdiv_lt_0_compat; nra). lra. }
  assert (0 < sqrt (a * a + q * (b * b) / r)) by (apply sqrt_lt_R0; exact Hpos).
  lra.
Qed.

Theorem riccati_identity :
  forall a b q r,
    0 < q -> 0 < r -> b <> 0 ->
    sqrt (a * a + q * (b * b) / r) * sqrt (a * a + q * (b * b) / r)
      = a * a + q * (b * b) / r.
Proof.
  intros a b q r Hq Hr Hb.
  assert (Hbb : 0 < b * b) by (apply Rlt_0_sqr; exact Hb).
  apply sqrt_sqrt.
  assert (0 <= a * a) by apply Rle_0_sqr.
  assert (0 < q * (b * b) / r) by (apply Rdiv_lt_0_compat; nra). lra.
Qed.

Theorem obstruction_gain_nonpositive :
  forall a cT bigA,
    bigA < 0 -> a <= 0 -> 0 <= cT <= 1 -> a * cT + bigA * (1 - cT) <= 0.
Proof.
  intros a cT bigA HA Ha [Hc0 Hc1]. nra.
Qed.

(* ===== the fixed point is AFFINE: unique iff the denominator survives, and it has a pole ===== *)

Theorem fixed_point_unique :
  forall d n m0,
    d <> 0 -> exists! s, d * s = - (n * m0).
Proof.
  intros d n m0 Hd. exists (- (n * m0) / d). split.
  - field. exact Hd.
  - intros s' Hs'. apply (Rmult_eq_reg_l d); [| exact Hd].
    rewrite Hs'. field. exact Hd.
Qed.

(* The pole.  |S(0)| >= |n m0| / eps whenever the denominator is within eps of zero: approaching
   the obstruction horizon, the value function's linear coefficient -- and with it the optimal
   control -- diverges like 1/(T* - T).  A residual-minimising solver sees none of this. *)
Theorem fixed_point_blows_up :
  forall d n m0 s eps,
    0 < eps -> Rabs d <= eps -> d * s = - (n * m0) ->
    Rabs (n * m0) / eps <= Rabs s.
Proof.
  intros d n m0 s eps Heps Hd Hs.
  assert (Habs : Rabs (n * m0) = Rabs d * Rabs s).
  { rewrite <- Rabs_mult, Hs, Rabs_Ropp. reflexivity. }
  assert (Hs0 : 0 <= Rabs s) by apply Rabs_pos.
  apply (Rmult_le_reg_r eps); [exact Heps |].
  unfold Rdiv. rewrite Rmult_assoc, Rinv_l by lra.
  rewrite Habs. nra.
Qed.
