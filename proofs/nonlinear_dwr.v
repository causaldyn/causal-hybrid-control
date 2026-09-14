(* A GENERIC DUAL-WEIGHTED ESTIMATOR: the algebraic core of replacing a known adjoint matrix by a
   linearisation.

   Result 55 gave an EXACT error quotient for the LQ mean-field game and recorded that the
   exactness was a property of the AFFINE reduced problem.  validation/nonlinear_dwr.mac builds the
   smallest game that breaks the affinity without breaking the reduction -- a congestion-shifted
   target (q/2)(x - c m - gam m^3)^2, whose reduced field is

       F(m, S) = (A m - (b^2/r) S,  q (c m + gam m^3) - A S)

   -- and derives the estimator for a general field.  This file proves the four algebraic facts
   that the derivation rests on, none of which mention a particular F:

   (1) the pairing identity d/dt (z . delta) = z . g holds for EVERY Jacobian, provided the adjoint
       is driven by the TRANSPOSE;
   (2) with the untransposed Jacobian it fails by exactly (j21 - j12)(z2 d1 - z1 d2), so it is
       correct precisely on a self-adjoint field -- which the congested one is not;
   (3) the error formula is an exact quotient with z(0)[free] as its only denominator, and the
       error it returns is unique when that denominator survives;
   (4) an affine field has NO remainder at any perturbation size, while the cubic term leaves
       3 gam m h^2 + gam h^3 -- so the estimate is exact in the first case and second-order in the
       second, which is what chc.deep_galerkin.nonlinear_dwr_certificate measures.

   Stdlib Reals only.  Derivatives appear only as the scalars d/dt(.) that the caller supplies:
   the identity is an algebraic cancellation, not an analytic one, which is the reason it needs no
   structure in F. *)

From Stdlib Require Import Reals Lra.
Open Scope R_scope.

(* ===== (1) the pairing identity, for an arbitrary Jacobian ===== *)

(* `dk` are the components of the linearised primal's derivative, `zk` those of the adjoint's.
   Nothing below assumes where they came from. *)
Definition pairing_rate (z1 z2 d1 d2 z1' z2' d1' d2' : R) : R :=
  z1' * d1 + z1 * d1' + z2' * d2 + z2 * d2'.

Theorem transposed_adjoint_pairs_to_the_defect :
  forall j11 j12 j21 j22 z1 z2 d1 d2 g1 g2,
    pairing_rate z1 z2 d1 d2
      (- (j11 * z1 + j21 * z2)) (- (j12 * z1 + j22 * z2))
      (j11 * d1 + j12 * d2 + g1) (j21 * d1 + j22 * d2 + g2)
    = z1 * g1 + z2 * g2.
Proof. intros. unfold pairing_rate. ring. Qed.

(* ===== (2) and the untransposed one does not, by an amount that names its own condition ===== *)

Theorem untransposed_adjoint_leaves_a_skew_term :
  forall j11 j12 j21 j22 z1 z2 d1 d2 g1 g2,
    pairing_rate z1 z2 d1 d2
      (- (j11 * z1 + j12 * z2)) (- (j21 * z1 + j22 * z2))
      (j11 * d1 + j12 * d2 + g1) (j21 * d1 + j22 * d2 + g2)
    = z1 * g1 + z2 * g2 + (j21 - j12) * (z2 * d1 - z1 * d2).
Proof. intros. unfold pairing_rate. ring. Qed.

Corollary untransposed_is_correct_exactly_on_a_symmetric_jacobian :
  forall j11 j12 j21 j22 z1 z2 d1 d2 g1 g2,
    j12 = j21 ->
    pairing_rate z1 z2 d1 d2
      (- (j11 * z1 + j12 * z2)) (- (j21 * z1 + j22 * z2))
      (j11 * d1 + j12 * d2 + g1) (j21 * d1 + j22 * d2 + g2)
    = z1 * g1 + z2 * g2.
Proof.
  intros j11 j12 j21 j22 z1 z2 d1 d2 g1 g2 Hsym.
  rewrite untransposed_adjoint_leaves_a_skew_term, Hsym. ring.
Qed.

(* The congested reduced field is never self-adjoint on a well-posed problem: its off-diagonal
   entries are -b^2/r < 0 and q(c + 3 gam m^2) > 0, which cannot be equal. *)
Theorem congested_jacobian_is_not_symmetric :
  forall b r q c gam m,
    0 < r -> b <> 0 -> 0 < q -> 0 < c -> 0 <= gam ->
    - (b * b / r) <> q * (c + 3 * gam * (m * m)).
Proof.
  intros b r q c gam m Hr Hb Hq Hc Hg.
  assert (Hbb : 0 < b * b) by (apply Rlt_0_sqr; exact Hb).
  assert (Hleft : - (b * b / r) < 0).
  { assert (0 < b * b / r) by (apply Rdiv_lt_0_compat; assumption). lra. }
  assert (Hmm : 0 <= m * m) by apply Rle_0_sqr.
  assert (Hterm : 0 <= 3 * gam * (m * m)) by nra.
  assert (Hsum : 0 < c + 3 * gam * (m * m)) by lra.
  assert (Hright : 0 < q * (c + 3 * gam * (m * m))) by nra.
  lra.
Qed.

(* ===== (3) the error formula, and what its denominator is for ===== *)

(* Integrating (1) across [0, T] leaves `terminal - interior = z(0) . delta(0)`, and the initial
   condition pins every component of delta(0) but one.  So the free component is determined. *)
Theorem error_is_an_exact_quotient :
  forall denominator terminal interior error,
    denominator <> 0 ->
    denominator * error = terminal - interior ->
    error = (terminal - interior) / denominator.
Proof.
  intros denominator terminal interior error Hden Heq.
  rewrite <- Heq. field. exact Hden.
Qed.

Theorem error_is_unique_while_the_denominator_survives :
  forall denominator terminal interior e1 e2,
    denominator <> 0 ->
    denominator * e1 = terminal - interior ->
    denominator * e2 = terminal - interior ->
    e1 = e2.
Proof.
  intros denominator terminal interior e1 e2 Hden H1 H2.
  apply (Rmult_eq_reg_l denominator); [| exact Hden].
  rewrite H1, H2. reflexivity.
Qed.

(* At a vanishing denominator the equation is `0 = terminal - interior`: either no error solves it
   or every error does.  That is the obstruction of Result 49 reappearing in the estimator, and it
   is why `adjoint_weighted_error` reports infinity there rather than a number. *)
Theorem a_vanishing_denominator_determines_nothing :
  forall terminal interior,
    terminal - interior = 0 -> forall e, 0 * e = terminal - interior.
Proof. intros terminal interior H e. rewrite Rmult_0_l. symmetry. exact H. Qed.

(* ===== (4) what the linearisation costs: nothing on an affine field, h^2 on the cubic ===== *)

Definition affine_field (m11 m12 m21 m22 y1 y2 : R) : R * R :=
  (m11 * y1 + m12 * y2, m21 * y1 + m22 * y2).

Theorem an_affine_field_has_no_remainder :
  forall m11 m12 m21 m22 y1 y2 h1 h2,
    fst (affine_field m11 m12 m21 m22 (y1 + h1) (y2 + h2))
      - fst (affine_field m11 m12 m21 m22 y1 y2)
      - (m11 * h1 + m12 * h2) = 0
    /\ snd (affine_field m11 m12 m21 m22 (y1 + h1) (y2 + h2))
      - snd (affine_field m11 m12 m21 m22 y1 y2)
      - (m21 * h1 + m22 * h2) = 0.
Proof. intros. unfold affine_field. simpl. split; ring. Qed.

(* The congestion is the whole nonlinearity, and its remainder is exactly quadratic-plus-cubic. *)
Definition response (c gam m : R) : R := c * m + gam * (m * m * m).

Theorem cubic_remainder_is_second_order :
  forall c gam m h,
    response c gam (m + h) - response c gam m - (c + 3 * gam * (m * m)) * h
      = 3 * gam * m * (h * h) + gam * (h * h * h).
Proof. intros. unfold response. ring. Qed.

(* At gam = 0 the remainder is identically zero at EVERY h -- Result 55's exactness, recovered as
   a special case rather than assumed as a hypothesis. *)
Corollary no_congestion_means_no_remainder :
  forall c m h, response c 0 (m + h) - response c 0 m - (c + 3 * 0 * (m * m)) * h = 0.
Proof. intros. rewrite cubic_remainder_is_second_order. ring. Qed.

(* And at m = 0 the quadratic term drops out too, so a probe that never leaves the origin measures
   an exactness the field has not got.  This is why the certificate's game starts at m0 = 1. *)
Corollary the_origin_hides_the_quadratic_term :
  forall c gam h, response c gam (0 + h) - response c gam 0 - (c + 3 * gam * (0 * 0)) * h
                    = gam * (h * h * h).
Proof. intros. rewrite cubic_remainder_is_second_order. ring. Qed.

(* Second order in the defect is FIRST order in the relative error, which is the honest way to
   quote it: the estimate loses accuracy in proportion to how wrong the solution already is. *)
Theorem second_order_absolute_is_first_order_relative :
  forall const eta truth,
    0 < eta -> truth <> 0 ->
    Rabs (const * (eta * eta)) / Rabs (truth * eta) = Rabs const / Rabs truth * eta.
Proof.
  intros const eta truth Heta Htruth.
  assert (Habs : Rabs (const * (eta * eta)) = Rabs const * (eta * eta)).
  { rewrite Rabs_mult, (Rabs_right (eta * eta)); [reflexivity | nra]. }
  assert (Hden : Rabs (truth * eta) = Rabs truth * eta).
  { rewrite Rabs_mult, (Rabs_right eta); [reflexivity | lra]. }
  assert (Hpos : 0 < Rabs truth) by (apply Rabs_pos_lt; exact Htruth).
  rewrite Habs, Hden. field. lra.
Qed.
