(* Rocq: the algebraic core of the CONDITIONED RESIDUAL (Result 55).

   validation/mean_field_dwr.mac derives, for the LQ mean-field game of Result 49: the numerical
   error at t = 0 is EXACTLY the total defect divided by the fixed-point determinant den(T); the
   dual weight z(s) = Phi(T-s)^T v / den(T) is the exact adjoint solution and equals v/den at
   s = T; the reduced HJB residual is homogeneous of degree 1 in (m, S), which is why a bounded
   approximator's residual FALLS while its error diverges; and den has a SIMPLE zero at the
   obstruction, so Result 49's fitted pole exponent -0.998 is exactly -1.

   Honest scope, as in lq_mean_field.v: Stdlib has no matrices, so the transition matrix and the
   adjoint ODE stay in Maxima and what is proved here is the scalar spine -- the exact quotient,
   its two-sided consequence (the determinant is the ONLY factor that can blow up), the
   homogeneity, and the simplicity of the zero. *)

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

(* (B) THE DETERMINANT IS THE ONLY POLE. Bounding the defect from both sides -- which the
   transition matrix allows, being an entire function of the horizon with no zero and no pole --
   sandwiches the error between two multiples of 1/|den|. So conditioning the residual by
   1/|den| is not a heuristic rescaling: it is the exact order of the error. *)
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

(* (C) WHY THE RAW RESIDUAL FALLS. The reduced HJB residual S' + A S - q c m is homogeneous of
   degree 1 in the pair (m, S). The exact amplitude diverges at the obstruction while a trained
   network of finite capacity stays bounded, so its residual stays bounded with it: the sign of
   the correlation between residual and error is decided by this homogeneity, not by the
   architecture. *)
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
