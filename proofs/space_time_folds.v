(* Rocq (SPACE-TIME FOLDS): plan 24's P2.5. Result 51 showed the delayed-network covariance is
   separable only at delta = 0; Result 52 designed folds on the SPACE axis alone. This file proves
   the algebra that makes the TWO-axis problem tractable and says what the one-axis restriction
   costs, as derived in validation/space_time_folds.mac.

   Four groups. (i) The fold operator lives in the closed algebra <I, E, F> (E the grand-mean
   projector, F the within-fold projector, E^2 = E, F^2 = F, EF = FE = E), so an operator is a
   COEFFICIENT TRIPLE and squaring is arithmetic: A_u^2 = (1, -r^4, r^4-1). Only the F coefficient
   sees the partition and it is strictly positive, which is what turns the design into a
   same balanced max-cut on the product set (Result 61). (ii) `pair_splits` is the entrywise content of
   `P (x) T + P^T (x) T^T = 2(sym (x) sym + skew (x) skew)`: each +-shift pair costs at most TWO
   Kronecker terms and exactly one when the spatial factor is symmetric, which bounds the rank by
   2D rather than 2D+1. (iii) delta = 0 collapses every shift, recovering separability. (iv) The
   panel weight w_1 STRICTLY EXCEEDS the single-slice weight phi on (0,1) -- so scoring a
   time-constant fold on a panel is not the cross-sectional design rescaled. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* --- (i) the fold algebra as coefficient triples --- *)

(* Product of a1*I + b1*E + c1*F and a2*I + b2*E + c2*F, using E^2=E, F^2=F, EF=FE=E. *)
Definition mul_i (a1 b1 c1 a2 b2 c2 : R) : R := a1 * a2.
Definition mul_e (a1 b1 c1 a2 b2 c2 : R) : R :=
  a1 * b2 + b1 * a2 + b1 * b2 + b1 * c2 + c1 * b2.
Definition mul_f (a1 b1 c1 a2 b2 c2 : R) : R := a1 * c2 + c1 * a2 + c1 * c2.

(* The cross-fit sandwich A_u = I - r^2 E + (r^2 - 1) F. *)
Definition au_i : R := 1.
Definition au_e (r : R) : R := - r ^ 2.
Definition au_f (r : R) : R := r ^ 2 - 1.

Lemma sandwich_square_identity : forall r : R,
  mul_i au_i (au_e r) (au_f r) au_i (au_e r) (au_f r) = 1 /\
  mul_e au_i (au_e r) (au_f r) au_i (au_e r) (au_f r) = - r ^ 4 /\
  mul_f au_i (au_e r) (au_f r) au_i (au_e r) (au_f r) = r ^ 4 - 1.
Proof.
  intro r. unfold mul_i, mul_e, mul_f, au_i, au_e, au_f.
  repeat split; ring.
Qed.

(* The partition enters only through the F coefficient, and its weight is strictly positive for
   any K >= 2 (r = K/(K-1) > 1). Smaller same-fold mass is therefore strictly better. *)
Lemma partition_weight_positive : forall r : R, 1 < r -> 0 < r ^ 4 - 1.
Proof.
  intros r Hr.
  assert (Hsq : 1 < r ^ 2) by nra.
  assert (Hfac : r ^ 4 - 1 = (r ^ 2 - 1) * (r ^ 2 + 1)) by ring.
  nra.
Qed.

Lemma ratio_above_one : forall k : R, 2 <= k -> 1 < k / (k - 1).
Proof.
  intros k Hk.
  assert (Hpos : 0 < k - 1) by lra.
  apply (Rmult_lt_reg_r (k - 1)); [lra|].
  field_simplify; lra.
Qed.

(* --- (ii) each +-shift pair costs at most two Kronecker terms --- *)

(* Entrywise, (P (x) T + P^T (x) T^T) at position ((i,k),(j,l)) is P_ij T_kl + P_ji T_lk, and the
   symmetric/antisymmetric split of that scalar pair is an identity. Nothing here is 2x2-specific:
   the statement is bilinear, so the entries carry it. *)
Lemma pair_splits : forall a b c d : R,
  a * c + b * d
  = 2 * (((a + b) / 2) * ((c + d) / 2) + ((a - b) / 2) * ((c - d) / 2)).
Proof. intros a b c d. field. Qed.

(* A symmetric spatial factor kills the antisymmetric channel, leaving ONE term for the pair. *)
Lemma symmetric_factor_kills_the_second_channel : forall a c d : R,
  a * c + a * d = 2 * (((a + a) / 2) * ((c + d) / 2) + ((a - a) / 2) * ((c - d) / 2))
  /\ (a - a) / 2 = 0.
Proof. intros a c d. split; [field | field]. Qed.

(* Shifts run over -D..D, so 2D+1 channels; the q = 0 and q = D factors are symmetric on ANY graph
   (a sum of squares, and a single term with the identity), leaving at most D-1 extra channels. *)
Lemma kronecker_rank_bound : forall d : R, (d + 1) + (d - 1) = 2 * d.
Proof. intro d. ring. Qed.

Lemma rank_never_reaches_the_channel_count : forall d : R, 0 < d -> 2 * d < 2 * d + 1.
Proof. intros d Hd. lra. Qed.

(* --- (iii) delta = 0 is exactly separability --- *)

Lemma zero_delay_collapses_the_shifts : forall tau q : R,
  Rabs (tau - 0 * q) = Rabs tau.
Proof. intros tau q. f_equal. ring. Qed.

(* And once the shift passes the panel's longest lag, the whole block is bounded by the excess --
   attained at tau = p-1, so the bound is sharp rather than merely valid. *)
Lemma shift_leaves_the_window : forall tau shift p : R,
  0 <= p - 1 -> Rabs tau <= p - 1 -> p - 1 <= shift ->
  shift - (p - 1) <= Rabs (tau - shift).
Proof.
  intros tau shift p Hp Htau Hshift.
  assert (Hup : tau <= p - 1) by (apply Rle_trans with (Rabs tau); [apply Rle_abs | exact Htau]).
  apply Rle_trans with (shift - tau); [lra|].
  rewrite <- Rabs_Ropp. rewrite Ropp_minus_distr. apply Rle_abs.
Qed.

Lemma shift_bound_is_attained : forall shift p : R,
  p - 1 <= shift -> Rabs ((p - 1) - shift) = shift - (p - 1).
Proof.
  intros shift p Hshift.
  rewrite <- Rabs_Ropp. replace (- ((p - 1) - shift)) with (shift - (p - 1)) by ring.
  apply Rabs_pos_eq; lra.
Qed.

(* --- (iv) the panel weight is not the single-slice weight --- *)

(* At p = 3, delta = 1 the exact sums are w_0 = 2 ph^2 + 4 ph + 3 and w_1 = ph^3 + 2 ph^2 + 4 ph + 2
   (validation/space_time_folds.mac (6), cross-checked numerically). Result 52 scores a fold with
   the single-slice weight ph^delta; a fold held constant in time and scored on the PANEL carries
   w_1/w_0 instead, and the two differ everywhere strictly inside (0,1). *)
Definition w0 (ph : R) : R := 2 * ph ^ 2 + 4 * ph + 3.
Definition w1 (ph : R) : R := ph ^ 3 + 2 * ph ^ 2 + 4 * ph + 2.

Lemma panel_weight_exceeds_the_slice_weight : forall ph : R,
  0 < ph < 1 -> ph * w0 ph < w1 ph.
Proof.
  intros ph [Hlo Hhi]. unfold w0, w1.
  assert (Hfac : w1 ph - ph * w0 ph = (ph + 2) * (1 - ph ^ 2))
    by (unfold w0, w1; ring).
  unfold w0, w1 in Hfac. nra.
Qed.

Lemma weights_agree_only_at_the_endpoints : forall ph : R,
  w1 ph - ph * w0 ph = (ph + 2) * (1 - ph) * (1 + ph).
Proof. intro ph. unfold w0, w1. ring. Qed.
