(* Rocq: the algebraic core of the CONDITIONED RESIDUAL (Result 55).

   validation/mean_field_dwr.mac derives, for the LQ mean-field game of Result 49: the numerical
   error at t = 0 is EXACTLY the total defect divided by the fixed-point determinant den(T); the
   dual weight z(s) = Phi(T-s)^T v / den(T) is the exact adjoint solution and equals v/den at
   s = T; the reduced HJB residual is homogeneous of degree 1 in (m, S), so an approximator whose
   reduced state stays bounded keeps a bounded residual while its error diverges (F); and den has
   a SIMPLE zero at the obstruction, so Result 49's fitted pole exponent -0.998 is exactly -1.

   Honest scope, as in lq_mean_field.v: Stdlib has no matrices, so the transition matrix and the
   adjoint ODE stay in Maxima and what is proved here is the scalar spine -- the exact quotient,
   its two-sided consequence (the determinant is the ONLY factor that can blow up), the
   homogeneity and what it does and does not imply about a bounded approximator, and the
   simplicity of the zero. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Psatz.
Open Scope R_scope.

(* (A) THE EXACT ERROR IDENTITY. The mean-field consistency condition is affine in S(0):
   num*m0 + den*S0 = 0 for the exact solution, and num*m0 + den*S0hat = d for a numerical one
   whose total defect is d. Subtracting gives the error as a quotient -- no linearisation, and
   no hypothesis that the defect is small. *)
Lemma error_is_defect_over_den :
  forall num den m0 s0 s0hat d : R,
  den <> 0 ->
  num * m0 + den * s0 = 0 ->
  num * m0 + den * s0hat = d ->
  s0hat - s0 = d / den.
Proof.
  intros num den m0 s0 s0hat d Hden Hexact Hnum.
  apply (Rmult_eq_reg_l den); [| exact Hden].
  replace (den * (d / den)) with d by (field; exact Hden).
  lra.
Qed.

(* (B) THE DETERMINANT IS THE ONLY POLE. The transition matrix is an entire function of the
   horizon with no zero and no pole, so the total defect d stays bounded on every compact set of
   horizons and d/den can blow up only where den vanishes. Given two-sided bounds on |d|, the
   error lies between the matching multiples of 1/|den|. Only the UPPER bound comes for free -- a
   residual bounds |d| from above. A LOWER bound is a property of the approximator, not of the
   equation: the adjoint projection of a nonzero defect can cancel. So a residual conditioned by
   1/|den| bounds the error from above and need not track it. *)
Lemma conditioning_is_two_sided :
  forall den d lo hi : R,
  den <> 0 -> 0 < lo -> lo <= Rabs d <= hi ->
  lo / Rabs den <= Rabs (d / den) <= hi / Rabs den.
Proof.
  intros den d lo hi Hden Hlo [Hl Hh].
  assert (Hpos : 0 < Rabs den) by (apply Rabs_pos_lt; exact Hden).
  unfold Rdiv.
  rewrite Rabs_mult, Rabs_inv by exact Hden.
  split; apply Rmult_le_compat_r; try (left; apply Rinv_0_lt_compat; exact Hpos); assumption.
Qed.

(* (C) WHY THE RAW RESIDUAL CANNOT FOLLOW. The reduced HJB residual S' + A S - q c m is
   homogeneous of degree 1 in the pair (m, S): it scales with the approximator's amplitude, not
   with the solution's. The exact amplitude diverges at the obstruction; an approximator whose
   amplitude stays bounded keeps a residual bounded with it -- (F) states and proves exactly
   that. Neither lemma decides the SIGN of the correlation between residual and error; that the
   residual falls rather than merely stalls is measured, not proved. *)
Lemma reduced_residual_homogeneous :
  forall a q c m s sdot kappa : R,
  (kappa * sdot) + a * (kappa * s) - q * c * (kappa * m)
    = kappa * (sdot + a * s - q * c * m).
Proof. intros; ring. Qed.

(* (D) THE ZERO IS SIMPLE. On the oscillatory branch den(T) = cos(wT) - k sin(wT)/w, so
   w * den'(T) = -(w^2 cos... ) collapses at a root, where w cos(w Tstar) = k sin(w Tstar), to
   -sin(w Tstar) (w^2 + k^2). With sin(w Tstar) <> 0 -- cot is finite at the root -- the derivative is
   nonzero: den has a simple zero. *)
Lemma den_zero_is_simple :
  forall w k sn cs : R,
  0 < w -> w * cs = k * sn ->
  w * (- (w * sn + k * cs)) = - (sn * (w * w + k * k)).
Proof.
  intros w k sn cs Hw Hroot.
  replace (w * (- (w * sn + k * cs))) with (- (w * w * sn) - k * (w * cs)) by ring.
  rewrite Hroot.
  ring.
Qed.

Lemma den_derivative_nonzero :
  forall w k sn cs : R,
  0 < w -> sn <> 0 -> w * cs = k * sn -> - (w * sn + k * cs) <> 0.
Proof.
  intros w k sn cs Hw Hsn Hroot Hzero.
  assert (Hprod : w * (- (w * sn + k * cs)) = - (sn * (w * w + k * k)))
    by (apply den_zero_is_simple; assumption).
  rewrite Hzero, Rmult_0_r in Hprod.
  assert (Hsq : 0 < w * w + k * k) by nra.
  assert (Hne : sn * (w * w + k * k) <> 0)
    by (apply Rmult_integral_contrapositive_currified; [exact Hsn | lra]).
  lra.
Qed.

(* (E) THE POLE EXPONENT IS EXACTLY -1. A simple zero means den(T) = d1 (T - Tstar) to leading
   order, so the error times the gap is CONSTANT -- Result 49 fitted -0.998 on a five-point
   log-log regression of the same quantity. *)
Lemma simple_pole_has_exponent_minus_one :
  forall d d1 gap : R,
  d1 <> 0 -> gap <> 0 ->
  Rabs (d / (d1 * gap)) * Rabs gap = Rabs d / Rabs d1.
Proof.
  intros d d1 gap Hd1 Hgap.
  unfold Rdiv.
  rewrite Rabs_mult, Rabs_inv by (apply Rmult_integral_contrapositive_currified; assumption).
  rewrite Rabs_mult.
  field.
  split; [apply Rabs_no_R0; assumption | apply Rabs_no_R0; assumption].
Qed.

(* (F) THE MECHANISM, WITH ITS HYPOTHESIS STATED. (C) alone decides nothing: homogeneity says the
   residual scales with the approximator's amplitude, not that the amplitude stays small. What
   makes the residual blind is a BOUNDED approximator facing an unbounded solution. If the
   approximator's reduced state and slope rate stay within B, its reduced residual is at most
   (1 + |a| + |q c|) B at every horizon; and wherever |den| <= eps its error at t = 0 is at least
   |num m0|/eps - B. The first bound does not depend on eps, the second grows without bound as
   eps -> 0. That the residual actually FALLS near the obstruction is not implied by any of this,
   and is a measurement. *)
Lemma bounded_residual :
  forall a q c m s sdot B : R,
  Rabs m <= B -> Rabs s <= B -> Rabs sdot <= B ->
  Rabs (sdot + a * s - q * c * m) <= (1 + Rabs a + Rabs (q * c)) * B.
Proof.
  intros a q c m s sdot B Hm Hs Hsdot.
  unfold Rminus.
  eapply Rle_trans; [apply Rabs_triang |].
  eapply Rle_trans; [apply Rplus_le_compat_r, Rabs_triang |].
  rewrite Rabs_Ropp, (Rabs_mult a s), (Rabs_mult (q * c) m).
  assert (Has : Rabs a * Rabs s <= Rabs a * B)
    by (apply Rmult_le_compat_l; [apply Rabs_pos | exact Hs]).
  assert (Hqm : Rabs (q * c) * Rabs m <= Rabs (q * c) * B)
    by (apply Rmult_le_compat_l; [apply Rabs_pos | exact Hm]).
  lra.
Qed.

Lemma error_exceeds_amplitude_gap :
  forall num den m0 s0 s0hat B eps : R,
  den <> 0 -> 0 < eps -> Rabs den <= eps ->
  num * m0 + den * s0 = 0 ->
  Rabs s0hat <= B ->
  Rabs (num * m0) / eps - B <= Rabs (s0hat - s0).
Proof.
  intros num den m0 s0 s0hat B eps Hden Heps Hle Hexact Hb.
  assert (Hpos : 0 < Rabs den) by (apply Rabs_pos_lt; exact Hden).
  assert (Hs0 : s0 = - (num * m0) / den).
  { apply (Rmult_eq_reg_l den); [| exact Hden].
    replace (den * (- (num * m0) / den)) with (- (num * m0)) by (field; exact Hden).
    lra. }
  assert (Habs : Rabs s0 = Rabs (num * m0) * / Rabs den).
  { rewrite Hs0. unfold Rdiv. rewrite Rabs_mult, Rabs_Ropp, Rabs_inv. reflexivity. }
  assert (Hinv : / eps <= / Rabs den) by (apply Rinv_le_contravar; assumption).
  assert (Hmono : Rabs (num * m0) * / eps <= Rabs (num * m0) * / Rabs den)
    by (apply Rmult_le_compat_l; [apply Rabs_pos | exact Hinv]).
  pose proof (Rabs_triang_inv s0 s0hat) as Htri.
  rewrite (Rabs_minus_sym s0 s0hat) in Htri.
  unfold Rdiv.
  lra.
Qed.

Theorem bounded_approximator_is_blind :
  forall a q c num m0 B E : R,
  num * m0 <> 0 -> 0 <= B ->
  exists eps : R, 0 < eps /\
  forall den s0 s0hat m s sdot : R,
    den <> 0 -> Rabs den <= eps -> num * m0 + den * s0 = 0 ->
    Rabs s0hat <= B -> Rabs m <= B -> Rabs s <= B -> Rabs sdot <= B ->
    Rabs (sdot + a * s - q * c * m) <= (1 + Rabs a + Rabs (q * c)) * B /\
    E <= Rabs (s0hat - s0).
Proof.
  intros a q c num m0 B E Hnm HB.
  assert (Hn : 0 < Rabs (num * m0)) by (apply Rabs_pos_lt; exact Hnm).
  assert (HD : 0 < Rabs E + B + 1) by (pose proof (Rabs_pos E); lra).
  assert (Heps : 0 < Rabs (num * m0) / (Rabs E + B + 1)) by (apply Rdiv_lt_0_compat; assumption).
  exists (Rabs (num * m0) / (Rabs E + B + 1)).
  split; [exact Heps |].
  intros den s0 s0hat m s sdot Hden Hle Hexact Hb Hm Hs Hsdot.
  split; [apply bounded_residual; assumption |].
  pose proof (error_exceeds_amplitude_gap num den m0 s0 s0hat B _ Hden Heps Hle Hexact Hb) as Hgap.
  replace (Rabs (num * m0) / (Rabs (num * m0) / (Rabs E + B + 1))) with (Rabs E + B + 1) in Hgap
    by (field; split; lra).
  pose proof (Rle_abs E) as HE.
  lra.
Qed.
