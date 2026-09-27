(* Rocq + MathComp: SYMBOLIC EXTRACTION FROM A KOLMOGOROV-ARNOLD LAYER in n inputs (Result 46).

   proofs/symbolic_kan.v states the gauge and the separability floor for a two-input layer. This
   file proves them for a layer b + sum_i f_i(x_i) with ANY number of inputs n, over any real
   field (realFieldType):

   - gauge_invariance, edge_intercept_not_identified, gauge_total_identified, centred_gauge_unique,
     centring_is_free, extraction_exact_up_to_gauge: an edge's intercept is a convention, the total
     constant is identified, and the centred gauge (every edge zero at 0) is unique -- for n edges.
   - additive_mixed_zero: the mixed second difference in ANY pair of coordinates (i, j), at any base
     point and on any rectangle, annihilates every additive function of n inputs.
   - mixed_le_four_error / additive_approximation_floor: on ANY domain containing the four corners,
     sup |F - A| >= |mixed F| / 4 for every additive A at once.
   - bilinear_floor_on_box: on the box [-r, r]^n no additive model approximates y_i y_j (i != j)
     better than r^2. The hypothesis is the error ON THE BOX only, where the two-input file assumes
     it on all of R^2, so this is also the stronger statement at n = 2.

   Honest scope. The extrapolation boundary of the RBF layer is transcendental and stays in Maxima,
   as in the two-input file. *)

Set Warnings "-notation-overridden,-ambiguous-paths".
From mathcomp Require Import boot order algebra.
Set Implicit Arguments. Unset Strict Implicit. Unset Printing Implicit Defensive.
Import GRing.Theory Num.Theory Order.Theory.
Local Open Scope ring_scope.

Section Layer.
Variable R : realFieldType.
Variable n : nat.

(* ---------- the two objects, in n inputs ---------- *)

(* A single Kolmogorov-Arnold layer in n inputs: a bias plus one scalar edge per input. *)
Definition additive (b : R) (f : 'I_n -> R -> R) (x : 'I_n -> R) : R := b + \sum_i f i (x i).

(* Set coordinate i of x to t. *)
Definition upd (x : 'I_n -> R) (i : 'I_n) (t : R) : 'I_n -> R :=
  fun k => if k == i then t else x k.

(* The mixed second difference in coordinates i and j over the rectangle [s0,s1] x [t0,t1], every
   other coordinate held at x. Four evaluations: no derivative, no continuity, no measure. *)
Definition mixed (F : ('I_n -> R) -> R) (x : 'I_n -> R) (i j : 'I_n) (s0 s1 t0 t1 : R) : R :=
  F (upd (upd x i s1) j t1) - F (upd (upd x i s1) j t0)
  - F (upd (upd x i s0) j t1) + F (upd (upd x i s0) j t0).

(* The pairwise bilinear target y_i * y_j. *)
Definition bilinear (i j : 'I_n) (y : 'I_n -> R) : R := y i * y j.

(* ---------- gauge: what an extracted edge does and does not pin down ---------- *)

Definition shift (f : 'I_n -> R -> R) (k : 'I_n -> R) : 'I_n -> R -> R := fun i t => f i t + k i.

Theorem gauge_invariance b f k x :
  additive (b - \sum_i k i) (shift f k) x = additive b f x.
Proof. by rewrite /additive /shift big_split /=; lra. Qed.

Lemma sum_indicator (j : 'I_n) (c : R) : \sum_i (if i == j then c else 0) = c.
Proof.
rewrite (bigD1 j) //= eqxx big1 ?addr0 // => i hij.
by rewrite (negPf hij).
Qed.

(* One edge alone: for EVERY constant c there is a different representation with the same values. *)
Theorem edge_intercept_not_identified b f (j : 'I_n) c x :
  additive (b - c) (shift f (fun i => if i == j then c else 0)) x = additive b f x.
Proof. by rewrite -{1}(sum_indicator j c) gauge_invariance. Qed.

Definition zero_point : 'I_n -> R := fun _ => 0.

(* What IS identified: the total constant. *)
Theorem gauge_total_identified b f b' f' (k : 'I_n -> R) :
  (forall i t, f' i t = f i t + k i) ->
  (forall x, additive b' f' x = additive b f x) ->
  b' + \sum_i k i = b.
Proof.
move=> hf heq; have := heq zero_point; rewrite /additive.
under eq_bigr do rewrite hf.
by rewrite big_split /= => h; lra.
Qed.

Lemma upd_zero_sum (g : 'I_n -> R -> R) (i : 'I_n) t :
  (forall k, g k 0 = 0) -> \sum_k g k (upd zero_point i t k) = g i t.
Proof.
move=> h0; rewrite (bigD1 i) //= /upd eqxx big1 ?addr0 // => k hki.
by rewrite (negPf hki) h0.
Qed.

(* Fixing the gauge -- every edge zero at the reference point -- makes the decomposition unique. *)
Theorem centred_gauge_unique b f b' f' :
  (forall i, f i 0 = 0) -> (forall i, f' i 0 = 0) ->
  (forall x, additive b f x = additive b' f' x) ->
  b = b' /\ (forall i t, f i t = f' i t).
Proof.
move=> h0 h0' heq.
have hb : b = b'.
  have := heq zero_point; rewrite /additive /zero_point.
  by rewrite big1 // big1 // !addr0.
split=> // i t; have := heq (upd zero_point i t); rewrite /additive.
by rewrite !upd_zero_sum // => h; lra.
Qed.

(* Any additive model can be put in that gauge without changing a single value. *)
Theorem centring_is_free b f x :
  additive b f x = additive (b + \sum_i f i 0) (fun i t => f i t - f i 0) x.
Proof. by rewrite /additive sumrB; lra. Qed.

(* ---------- representability: the mixed second difference ---------- *)

(* The operator annihilates the whole additive class in n inputs, for every pair of coordinates
   (equal or not), every rectangle and every base point. *)
Theorem additive_mixed_zero b f x (i j : 'I_n) s0 s1 t0 t1 :
  mixed (additive b f) x i j s0 s1 t0 t1 = 0.
Proof.
rewrite /mixed /additive.
set y11 := upd (upd x i s1) j t1; set y10 := upd (upd x i s1) j t0.
set y01 := upd (upd x i s0) j t1; set y00 := upd (upd x i s0) j t0.
have -> : b + \sum_k f k (y11 k) - (b + \sum_k f k (y10 k)) - (b + \sum_k f k (y01 k))
          + (b + \sum_k f k (y00 k))
          = \sum_k (f k (y11 k) - f k (y10 k) - f k (y01 k) + f k (y00 k)).
  by rewrite big_split /= !sumrB; lra.
apply: big1 => k _; rewrite /y11 /y10 /y01 /y00 /upd.
by case: (k == j); case: (k == i); ring.
Qed.

(* On the pairwise bilinear target it is the rectangle's area, whenever i and j differ. *)
Theorem bilinear_mixed x (i j : 'I_n) s0 s1 t0 t1 :
  i != j -> mixed (bilinear i j) x i j s0 s1 t0 t1 = (s1 - s0) * (t1 - t0).
Proof.
by move=> hij; rewrite /mixed /bilinear /upd !eqxx (negPf hij); ring.
Qed.

(* Hence no additive function of n inputs equals y_i * y_j. *)
Theorem bilinear_not_additive b f (i j : 'I_n) :
  i != j -> ~ (forall y, bilinear i j y = additive b f y).
Proof.
move=> hij heq.
have := bilinear_mixed zero_point 0 1 0 1 hij.
have -> : mixed (bilinear i j) zero_point i j 0 1 0 1 = mixed (additive b f) zero_point i j 0 1 0 1.
  by rewrite /mixed !heq.
by rewrite additive_mixed_zero !subr0 mulr1 => /eqP; rewrite eq_sym oner_eq0.
Qed.

(* ---------- the quantitative floor, on any domain containing the four corners ---------- *)

Lemma abs_four_terms (a b c d e : R) :
  `|a| <= e -> `|b| <= e -> `|c| <= e -> `|d| <= e -> `|a - b - c + d| <= 4 * e.
Proof.
rewrite !ler_norml => /andP[ha1 ha2] /andP[hb1 hb2] /andP[hc1 hc2] /andP[hd1 hd2].
by apply/andP; split; lra.
Qed.

Theorem mixed_le_four_error (inside : ('I_n -> R) -> Prop) F b f e x (i j : 'I_n) s0 s1 t0 t1 :
  (forall y, inside y -> `|F y - additive b f y| <= e) ->
  inside (upd (upd x i s1) j t1) -> inside (upd (upd x i s1) j t0) ->
  inside (upd (upd x i s0) j t1) -> inside (upd (upd x i s0) j t0) ->
  `|mixed F x i j s0 s1 t0 t1| <= 4 * e.
Proof.
move=> h h11 h10 h01 h00.
have hz := additive_mixed_zero b f x i j s0 s1 t0 t1.
have -> : mixed F x i j s0 s1 t0 t1
          = (F (upd (upd x i s1) j t1) - additive b f (upd (upd x i s1) j t1))
            - (F (upd (upd x i s1) j t0) - additive b f (upd (upd x i s1) j t0))
            - (F (upd (upd x i s0) j t1) - additive b f (upd (upd x i s0) j t1))
            + (F (upd (upd x i s0) j t0) - additive b f (upd (upd x i s0) j t0)).
  by move: hz; rewrite /mixed => hz; lra.
by apply: abs_four_terms; apply: h.
Qed.

Theorem additive_approximation_floor (inside : ('I_n -> R) -> Prop) F b f e x (i j : 'I_n)
    s0 s1 t0 t1 :
  (forall y, inside y -> `|F y - additive b f y| <= e) ->
  inside (upd (upd x i s1) j t1) -> inside (upd (upd x i s1) j t0) ->
  inside (upd (upd x i s0) j t1) -> inside (upd (upd x i s0) j t0) ->
  `|mixed F x i j s0 s1 t0 t1| / 4 <= e.
Proof.
move=> h h11 h10 h01 h00; have := mixed_le_four_error h h11 h10 h01 h00.
by rewrite ler_pdivrMr ?ltr0n // mulrC.
Qed.

(* The box [-r, r]^n. *)
Definition box (r : R) (y : 'I_n -> R) : Prop := forall k, `|y k| <= r.

Lemma upd_in_box r y (i : 'I_n) t : box r y -> `|t| <= r -> box r (upd y i t).
Proof. by move=> hy ht k; rewrite /upd; case: (k == i). Qed.

Lemma zero_in_box r : 0 <= r -> box r zero_point.
Proof. by move=> hr k; rewrite /zero_point normr0. Qed.

(* THE FLOOR IN n INPUTS: on [-r, r]^n no additive model of any number of variables approximates
   y_i * y_j (i != j) to better than r^2, and the hypothesis is the error ON THE BOX only. *)
Theorem bilinear_floor_on_box b f e r (i j : 'I_n) :
  i != j -> 0 <= r ->
  (forall y, box r y -> `|bilinear i j y - additive b f y| <= e) ->
  r ^+ 2 <= e.
Proof.
move=> hij hr h.
have hr1 : `|r| <= r by rewrite ger0_norm.
have hr2 : `|- r| <= r by rewrite normrN ger0_norm.
have hz := zero_in_box hr.
have := mixed_le_four_error h
  (upd_in_box j (upd_in_box i hz hr1) hr1) (upd_in_box j (upd_in_box i hz hr1) hr2)
  (upd_in_box j (upd_in_box i hz hr2) hr1) (upd_in_box j (upd_in_box i hz hr2) hr2).
rewrite bilinear_mixed // => hle.
have h0 : 0 <= (r - - r) * (r - - r) by rewrite opprK mulr_ge0 // addr_ge0.
rewrite ger0_norm // in hle.
nra.
Qed.

(* If the truth IS additive and the recovered edges match it up to the gauge, the extracted formula
   reproduces the truth at every point. *)
Theorem extraction_exact_up_to_gauge b f bh fh (k : 'I_n -> R) :
  bh = b - \sum_i k i -> (forall i t, fh i t = f i t + k i) ->
  forall x, additive bh fh x = additive b f x.
Proof.
move=> hb hf x; rewrite hb /additive.
have -> : \sum_i fh i (x i) = \sum_i (f i (x i) + k i) by apply: eq_bigr => i _; rewrite hf.
by rewrite big_split /=; lra.
Qed.

End Layer.
