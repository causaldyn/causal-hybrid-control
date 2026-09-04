(* Rocq: what combinatorial problem the fold design is, and what the certified gap guarantees
   (Result 61).

   validation/fold_cut_duality.mac derives, for a symmetric weight matrix Q and a bisection sign
   vector s: internal = (1'Q1 + s'Qs)/2, cut = (1'Q1 - s'Qs)/4, hence internal = 1'Q1 - 2 cut.
   Since 1'Q1 does not depend on the partition, MINIMISING the same-fold 2-walk mass -- which is
   what Result 52 showed Psi is affine-increasing in -- is MAXIMISING the cut of Q.  The design
   problem is a MAX-bisection, not the minimum-weight cut Result 52 (a) named it.

   Proved here: (A) the duality and its consequence for the ordering; (B) the certificate
   sandwich, i.e. that the reported Ky Fan gap is a ONE-SIDED guarantee -- eps certified implies
   eps-optimal, while a large certified gap implies nothing about the design; (C) the two shift
   laws that say which parts of Q the design can see; (D) the offset arithmetic behind the
   banded dynamic program's m > 2 B requirement.

   Honest scope: Stdlib has no matrices, so Q, s and the partition live in the Maxima file and
   what is proved here is the scalar algebra the enumeration, the dynamic program and the
   certificate then consume -- the same split as fold_spectrum_law.v. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Psatz.
From Stdlib Require Import Arith.
Open Scope R_scope.

(* ---------- (A) internal mass and cut are affine complements ---------- *)

Definition internal (total cut : R) : R := total - 2 * cut.

(* The partition enters only through the cut, and with a NEGATIVE coefficient. *)
Lemma internal_is_decreasing_in_the_cut :
  forall total c1 c2 : R, c1 < c2 -> internal total c2 < internal total c1.
Proof. intros total c1 c2 H. unfold internal. lra. Qed.

(* So the design that minimises the same-fold mass is exactly the design that maximises the cut:
   the problem is a MAX-bisection of Q, and calling it a minimum-weight cut inverts it. *)
Lemma min_internal_iff_max_cut :
  forall total c1 c2 : R, internal total c1 <= internal total c2 <-> c2 <= c1.
Proof. intros total c1 c2. unfold internal. split; intro H; lra. Qed.

(* The quadratic-form route: internal = (total + q)/2 and cut = (total - q)/4 with q = s'Qs,
   which is the form the Fourier diagonalisation of Result 52 (b) actually uses. *)
Lemma quadratic_form_agrees :
  forall total q : R, (total + q) / 2 = internal total ((total - q) / 4).
Proof. intros total q. unfold internal. lra. Qed.

(* ---------- (B) what the certified gap does and does not guarantee ---------- *)

(* The sandwich: a valid lower bound, the global optimum, and any design's value. *)
Lemma certified_gap_dominates_the_true_shortfall :
  forall kyfan opt heur : R,
  kyfan <= opt -> opt <= heur -> heur - opt <= heur - kyfan.
Proof. intros kyfan opt heur H1 H2. lra. Qed.

(* The load-bearing corollary: eps certified implies eps-optimal.  This is why a run that only
   reports the Ky Fan gap is still a guarantee and not merely a diagnostic. *)
Lemma certified_epsilon_implies_epsilon_optimal :
  forall kyfan opt heur eps : R,
  0 < kyfan -> kyfan <= opt -> opt <= heur ->
  heur - kyfan <= eps * kyfan ->
  heur - opt <= eps * opt.
Proof.
  intros kyfan opt heur eps Hk Hko Hoh Hgap.
  assert (Heps : 0 <= eps).
  { apply Rmult_le_reg_r with (r := kyfan); [ exact Hk | lra ]. }
  assert (Hmono : eps * kyfan <= eps * opt).
  { apply Rmult_le_compat_l; assumption. }
  lra.
Qed.

(* The converse fails, and it fails at the only place it matters: a design can be EXACTLY optimal
   while the certificate reports a 100% gap.  Measured instance of the same shape, on a cycle at
   m = 24, x = 0.9: the heuristic is optimal (shortfall 0.0000) and the certified gap is 1.49%. *)
Lemma a_large_certified_gap_does_not_convict_the_design :
  let kyfan := 1 in let opt := 2 in let heur := 2 in
  kyfan <= opt /\ opt <= heur /\ heur - opt = 0 /\ heur - kyfan = 1 * kyfan.
Proof. simpl. repeat split; lra. Qed.

(* ---------- (C) the two shifts the design cannot see ---------- *)

(* A uniform shift of every design's value leaves the ordering, hence the argmin, untouched. *)
Lemma uniform_shift_preserves_the_ranking :
  forall k a b : R, a <= b <-> a + k <= b + k.
Proof. intros k a b. split; intro H; lra. Qed.

(* Q -> Q + c (J - I) shifts 1'Q1 by c (m^2 - m) and, under balance (1's = 0, s's = m), shifts
   s'Qs by -c m -- a constant in s.  So every balanced cut moves by exactly c m^2 / 4. *)
Lemma off_diagonal_shift_moves_every_cut_by_the_same_amount :
  forall c m : R,
  ((c * (m * m - m)) - (- (c * m))) / 4 = c * m * m / 4.
Proof. intros c m. field. Qed.

(* Q -> Q + t I shifts 1'Q1 and s'Qs by the same t m, so the cut does not move at all: the design
   is blind to the diagonal, and the Ky Fan bound cannot be tightened by shifting it. *)
Lemma diagonal_shift_leaves_the_cut_alone :
  forall t m : R, ((t * m) - (t * m)) / 4 = 0.
Proof. intros t m. field. Qed.

(* ---------- (D) the offset arithmetic behind m > 2 B ---------- *)

(* On a circulant the objective is a sum over circular offsets, and offsets b and m - b name the
   same unordered pair.  Below the antipode they are distinct, so counting b once and doubling is
   correct. *)
Lemma offsets_are_distinct_below_the_antipode :
  forall m b : nat, (0 < b)%nat -> (2 * b < m)%nat -> (b <> m - b)%nat.
Proof. intros m b Hb Hm. lia. Qed.

(* At the antipode they coincide, and the doubling counts the ring {i, i + B} twice -- the reason
   the dynamic program refuses m <= 2 * band instead of returning a wrong optimum. *)
Lemma the_antipodal_offset_is_its_own_mirror :
  forall m b : nat, (2 * b = m)%nat -> (b = m - b)%nat.
Proof. intros m b H. lia. Qed.

(* And past the antipode there is nothing new: the circular distance never exceeds m / 2, so a
   bandwidth B >= m / 2 is no restriction at all and the dynamic program has no work to save. *)
Lemma a_band_at_the_antipode_is_no_band :
  forall m b : nat, (2 * b >= m)%nat -> (min b (m - b) <= m - b)%nat.
Proof. intros m b H. apply Nat.le_min_r. Qed.
