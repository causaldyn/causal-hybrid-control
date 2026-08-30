(* Rocq: SYMBOLIC EXTRACTION FROM A KOLMOGOROV-ARNOLD LAYER -- what is identified, and the price of
   what cannot be represented.

   chc.residual.RBFKANLayer advertises each input-output edge as "an extractable 1D curve"; chc.symbolic
   is the extraction, and this file is its algebraic core. A single KAN layer computes
   bias + sum_i phi_i(z_i), so two questions decide whether an extracted per-edge formula means
   anything at all, and neither is a matter of fit quality:

   IDENTIFIABILITY. Adding a constant to one edge and subtracting it from the bias leaves the layer
   pointwise unchanged, so an edge's intercept is a convention, not a measurement. Only the TOTAL
   constant is pinned down (gauge_total_identified), and fixing a reference point -- each edge zero
   at 0 -- makes the whole decomposition unique (centred_gauge_unique). chc.symbolic therefore
   reports each edge's own offset AND checks their sum against the bias, which is the part with
   content.

   REPRESENTABILITY. The MIXED SECOND DIFFERENCE F(x1,y1) - F(x1,y0) - F(x0,y1) + F(x0,y0)
   annihilates every additive function, with no smoothness assumed -- only evaluation at four points.
   That turns "an additive model cannot represent an interaction" from a slogan into a QUANTITATIVE
   floor: the operator has four terms, so |mixed F| <= 4 * sup|F - A| for every additive A at once,
   and hence sup|F - A| >= |mixed F| / 4. On the bilinear target x*y over [-r, r]^2 the floor is
   exactly r^2 (bilinear_floor_on_box) -- a bound no additive model, KAN or otherwise, can beat.
   chc.symbolic's certificate measures 9.72 against the proved floor 9.00 at r = 3.

   Derived in validation/symbolic_kan.mac. Honest scope: the extrapolation boundary -- outside the
   grid every Gaussian has decayed and the layer degenerates to its silu term -- is transcendental
   and stays in Maxima; Rocq carries only the algebra, which is the part the certificate's
   assertions rest on. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.

Open Scope R_scope.

(* ---------- the two objects ---------- *)

(* The additive class a single Kolmogorov-Arnold layer can represent, in two inputs. *)
Definition additive (b : R) (f g : R -> R) (x y : R) : R := b + f x + g y.

(* The mixed second difference over the rectangle [x0,x1] x [y0,y1]. Four evaluations, nothing else:
   no derivative, no continuity, no measure. *)
Definition mixed (F : R -> R -> R) (x0 x1 y0 y1 : R) : R :=
  F x1 y1 - F x1 y0 - F x0 y1 + F x0 y0.

Definition bilinear (x y : R) : R := x * y.

(* ---------- gauge: what an extracted edge does and does not pin down ---------- *)

Definition shift (f : R -> R) (k : R) : R -> R := fun x => f x + k.

(* Moving a constant from an edge into the bias is invisible to the layer. This is why an edge's
   intercept alone is never evidence about the data. *)
Theorem gauge_invariance : forall b f g k0 k1 x y,
  additive (b - k0 - k1) (shift f k0) (shift g k1) x y = additive b f g x y.
Proof. intros. unfold additive, shift. ring. Qed.

(* The one-edge case, stated separately because it is the direct witness of non-identifiability:
   for EVERY k there is a different representation with the same values. *)
Theorem edge_intercept_not_identified : forall b f g k x y,
  additive (b - k) (shift f k) g x y = additive b f g x y.
Proof. intros. unfold additive, shift. ring. Qed.

(* What IS identified: the total constant. Two representations that agree pointwise, whose edges
   differ by constants, must have bias offsets that compensate exactly. *)
Theorem gauge_total_identified : forall b f g b' f' g' k0 k1,
  (forall x, f' x = f x + k0) ->
  (forall y, g' y = g y + k1) ->
  (forall x y, additive b' f' g' x y = additive b f g x y) ->
  b' + k0 + k1 = b.
Proof.
  intros b f g b' f' g' k0 k1 Hf Hg Heq.
  specialize (Heq 0 0). unfold additive in Heq.
  rewrite Hf, Hg in Heq. lra.
Qed.

(* Fixing the gauge -- each edge zero at the reference point -- makes the decomposition unique.
   This is the convention that turns an extracted per-edge formula into a statement about the
   function rather than about the fit. *)
Theorem centred_gauge_unique : forall b f g b' f' g',
  f 0 = 0 -> g 0 = 0 -> f' 0 = 0 -> g' 0 = 0 ->
  (forall x y, additive b f g x y = additive b' f' g' x y) ->
  b = b' /\ (forall x, f x = f' x) /\ (forall y, g y = g' y).
Proof.
  intros b f g b' f' g' Hf0 Hg0 Hf0' Hg0' Heq.
  assert (Hb : b = b').
  { specialize (Heq 0 0). unfold additive in Heq.
    rewrite Hf0, Hg0, Hf0', Hg0' in Heq. lra. }
  split; [exact Hb | split].
  - intros x. specialize (Heq x 0). unfold additive in Heq.
    rewrite Hg0, Hg0' in Heq. lra.
  - intros y. specialize (Heq 0 y). unfold additive in Heq.
    rewrite Hf0, Hf0' in Heq. lra.
Qed.

(* Any additive model can be put in that gauge without changing a single value. *)
Theorem centring_is_free : forall b f g x y,
  additive b f g x y
  = additive (b + f 0 + g 0) (fun t => f t - f 0) (fun t => g t - g 0) x y.
Proof. intros. unfold additive. ring. Qed.

(* ---------- representability: the mixed second difference ---------- *)

(* The operator annihilates the whole additive class, for every f and g and every rectangle. No
   smoothness is used -- this is an identity between four evaluations. *)
Theorem additive_mixed_zero : forall b f g x0 x1 y0 y1,
  mixed (additive b f g) x0 x1 y0 y1 = 0.
Proof. intros. unfold mixed, additive. ring. Qed.

(* On the bilinear target it is nonzero on every nondegenerate rectangle. *)
Theorem bilinear_mixed : forall x0 x1 y0 y1,
  mixed bilinear x0 x1 y0 y1 = (x1 - x0) * (y1 - y0).
Proof. intros. unfold mixed, bilinear. ring. Qed.

(* Hence no additive function equals x*y, and the witness is explicit. *)
Theorem bilinear_not_additive : forall b f g r,
  0 < r -> ~ (forall x y, bilinear x y = additive b f g x y).
Proof.
  intros b f g r Hr Heq.
  assert (Hmix : mixed bilinear (-r) r (-r) r = mixed (additive b f g) (-r) r (-r) r).
  { unfold mixed. rewrite !Heq. reflexivity. }
  rewrite bilinear_mixed, additive_mixed_zero in Hmix.
  nra.
Qed.

(* ---------- the quantitative floor ---------- *)

(* Stdlib has Rle_abs (x <= |x|) but no packaged inverse of Rabs_le, so it is derived here. *)
Lemma abs_bound_inv : forall a e, Rabs a <= e -> -e <= a <= e.
Proof.
  intros a e H. split.
  - assert (Hn : - a <= Rabs a).
    { rewrite <- Rabs_Ropp. apply Rle_abs. }
    lra.
  - assert (Hp : a <= Rabs a) by apply Rle_abs. lra.
Qed.

Lemma abs_four_terms : forall a b c d e,
  Rabs a <= e -> Rabs b <= e -> Rabs c <= e -> Rabs d <= e ->
  Rabs (a - b - c + d) <= 4 * e.
Proof.
  intros a b c d e Ha Hb Hc Hd.
  destruct (abs_bound_inv _ _ Ha). destruct (abs_bound_inv _ _ Hb).
  destruct (abs_bound_inv _ _ Hc). destruct (abs_bound_inv _ _ Hd).
  apply Rabs_le. split; lra.
Qed.

(* The operator has four terms, so it cannot see more than four times the approximation error.
   This is what converts an exact non-representability statement into a bound with a number in it. *)
Theorem mixed_le_four_error : forall F b f g e x0 x1 y0 y1,
  (forall x y, Rabs (F x y - additive b f g x y) <= e) ->
  Rabs (mixed F x0 x1 y0 y1) <= 4 * e.
Proof.
  intros F b f g e x0 x1 y0 y1 H.
  replace (mixed F x0 x1 y0 y1)
    with ((F x1 y1 - additive b f g x1 y1)
          - (F x1 y0 - additive b f g x1 y0)
          - (F x0 y1 - additive b f g x0 y1)
          + (F x0 y0 - additive b f g x0 y0))
    by (unfold mixed, additive; ring).
  apply abs_four_terms; apply H.
Qed.

(* The floor, for an arbitrary target and rectangle: the best additive approximation is at least a
   quarter of the mixed difference away, and this holds for EVERY additive A at once. *)
Theorem additive_approximation_floor : forall F b f g e x0 x1 y0 y1,
  (forall x y, Rabs (F x y - additive b f g x y) <= e) ->
  Rabs (mixed F x0 x1 y0 y1) / 4 <= e.
Proof.
  intros F b f g e x0 x1 y0 y1 H.
  assert (H4 := mixed_le_four_error F b f g e x0 x1 y0 y1 H). lra.
Qed.

(* Specialised to the certificate's arm: on [-r, r]^2 no additive model approximates x*y to better
   than r^2. At r = 3 that is 9, against a target that itself only ranges over [-9, 9]. *)
Theorem bilinear_floor_on_box : forall b f g e r,
  0 <= r ->
  (forall x y, Rabs (bilinear x y - additive b f g x y) <= e) ->
  r * r <= e.
Proof.
  intros b f g e r Hr H.
  assert (H4 := mixed_le_four_error bilinear b f g e (-r) r (-r) r H).
  rewrite bilinear_mixed in H4.
  rewrite Rabs_right in H4 by nra.
  lra.
Qed.

(* The floor is not vacuous: it grows with the box, so widening the domain makes an additive
   surrogate worse without bound rather than merely imperfect. *)
Theorem floor_grows_with_box : forall r1 r2,
  0 <= r1 -> r1 <= r2 -> r1 * r1 <= r2 * r2.
Proof. intros. nra. Qed.

Theorem floor_unbounded : forall M, 0 <= M -> exists r, 0 <= r /\ M < r * r.
Proof.
  intros M HM. exists (M + 1). split; [lra | nra].
Qed.

(* ---------- what extraction buys: exactness on the class it can represent ---------- *)

(* If the truth IS additive and the recovered edges match it up to the gauge, the extracted formula
   reproduces the truth everywhere -- including outside any grid, which is the point: the formula
   carries no basis with it, so the RBF support boundary is not its boundary. *)
Theorem extraction_exact_up_to_gauge : forall b f g bh fh gh k0 k1,
  bh = b - k0 - k1 ->
  (forall x, fh x = f x + k0) ->
  (forall y, gh y = g y + k1) ->
  forall x y, additive bh fh gh x y = additive b f g x y.
Proof.
  intros b f g bh fh gh k0 k1 Hb Hf Hg x y.
  unfold additive. rewrite Hb, Hf, Hg. ring.
Qed.
