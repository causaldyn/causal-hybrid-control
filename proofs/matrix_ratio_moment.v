(* Rocq: the algebraic core of the EXACT MATRIX RATIO MOMENT (Result 54).

   validation/matrix_ratio_moment.mac derives, for the two-channel cross-fit estimator's exact
   sampling variance V = E[(X'AX)^{-1} X'A Sigma A X (X'AX)^{-1}]: the 2x2 inverse is
   adj(M)/det(M), so V = E[adj(M) N adj(M)/det(M)^2]; every entry of the sandwich is a signed sum
   of products of THREE quadratic forms in vec(X); det(M)^{-2} is an Ingham-Siegel integral over
   the positive-definite cone (the matrix form of the scalar resolvent Result 51 (l) is built
   on); and the tilted-Gaussian three-form moment is fixed by Isserlis' 15 pairings. Stdlib has
   no matrices: what is proved here is the 2x2 component algebra the evaluator expands --
   stated entrywise over the reals, closed by ring. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* (A) The 2x2 adjugate cancels M: componentwise adj(M) M = det(M) I. *)
Lemma adjugate_cancels_11 :
  forall m11 m12 m22 : R, m22*m11 + (- m12)*m12 = m11*m22 - m12*m12.
Proof. intros; ring. Qed.

Lemma adjugate_cancels_12 :
  forall m11 m12 m22 : R, m22*m12 + (- m12)*m22 = 0.
Proof. intros; ring. Qed.

Lemma adjugate_cancels_22 :
  forall m11 m12 m22 : R, (- m12)*m12 + m11*m22 = m11*m22 - m12*m12.
Proof. intros; ring. Qed.

(* (B) The sandwich entries the evaluator expands: (adj(M) N adj(M))_ab written as the signed
   three-form combinations fed to the tilted Gaussian moment. *)
Lemma sandwich_entry_11 :
  forall m12 m22 n11 n12 n22 : R,
  (m22*n11 - m12*n12)*m22 + (m22*n12 - m12*n22)*(- m12)
  = m22*m22*n11 - 2*m12*m22*n12 + m12*m12*n22.
Proof. intros; ring. Qed.

Lemma sandwich_entry_12 :
  forall m11 m12 m22 n11 n12 n22 : R,
  (m22*n11 - m12*n12)*(- m12) + (m22*n12 - m12*n22)*m11
  = - m12*m22*n11 + m11*m22*n12 + m12*m12*n12 - m11*m12*n22.
Proof. intros; ring. Qed.

Lemma sandwich_entry_22 :
  forall m11 m12 n11 n12 n22 : R,
  ((- m12)*n11 + m11*n12)*(- m12) + ((- m12)*n12 + m11*n22)*m11
  = m12*m12*n11 - 2*m11*m12*n12 + m11*m11*n22.
Proof. intros; ring. Qed.

(* (C) The scalar shadow of the Ingham-Siegel tail: the q = 1 case is Result 51 (l)'s kernel,
   and the positive-cone constraint for q = 2 in Cholesky coordinates
   T = [[a^2, ac], [ac, c^2 + b^2]] holds by construction: det(T) = (a b)^2 >= 0 and the
   leading entry a^2 >= 0 -- the parametrisation cannot leave the cone, which is what makes
   the evaluator's change of variables legitimate. *)
Lemma cholesky_stays_in_the_cone :
  forall a b c : R,
  0 <= a*a /\ (a*a) * (c*c + b*b) - (a*c)*(a*c) = (a*b)*(a*b).
Proof.
  intros a b c; split.
  - nra.
  - ring.
Qed.
