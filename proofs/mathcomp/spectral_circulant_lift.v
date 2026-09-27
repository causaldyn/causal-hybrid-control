(* Rocq + MathComp: CIRCULANT RESIDUALS over any number of modes and factors (Result 48).

   proofs/spectral_circulant.v carries two modes, enough to exhibit "max of products < product of
   maxes". This file proves the N-mode, M-factor statements over any real field (realFieldType),
   with each mode's symbol p + i q acting on its coordinates (a, b) as the real block
   [[p, q], [-q, p]], exactly as in the two-mode file:

   - composition_is_one_symbol / gain_of_composition / composition_gain_is_exact: M circulant
     factors act on a mode as ONE symbol, the product of theirs, whose gain is the product of their
     gains exactly -- Brahmagupta-Fibonacci, iterated.
   - n_mode_bounded / n_mode_norm_attained: over N modes the energy gain is at most the largest
     per-mode gain, and an input on a maximising mode attains it. composition_norm_bounded /
     composition_norm_attained: the same for the composed operator, whose exact norm is
     max_k prod_m g_mk.
   - product_bound_is_valid, product_bound_is_tight_at_a_common_maximiser,
     product_bound_is_strictly_loose, strictly_loose_iff_no_common_maximiser: max_k prod_m g_mk <=
     prod_m max_k g_mk, with equality when one mode maximises every factor and strict inequality
     EXACTLY when none does (all maxima positive); conservatism_ratio_exceeds_one is the ratio form.

   Honest scope. The DFT diagonalisation of a circulant is complex and stays in Maxima (STEP 1), as
   does the advection-diffusion dispersion relation (STEP 5). *)

Set Warnings "-notation-overridden,-ambiguous-paths".
From mathcomp Require Import boot order algebra.
Set Implicit Arguments. Unset Strict Implicit. Unset Printing Implicit Defensive.
Import GRing.Theory Num.Theory Order.Theory.
Local Open Scope ring_scope.

(* ---------- one mode: a symbol p + i q acting on (a, b) ---------- *)
Section OneMode.
Variable R : realFieldType.

Definition act (z v : R * R) : R * R := (z.1 * v.1 + z.2 * v.2, - z.2 * v.1 + z.1 * v.2).
Definition sq_norm (v : R * R) : R := v.1 * v.1 + v.2 * v.2.
Definition gain (z : R * R) : R := z.1 * z.1 + z.2 * z.2.
Definition cmul (z w : R * R) : R * R := (z.1 * w.1 - z.2 * w.2, z.1 * w.2 + z.2 * w.1).
Definition cone : R * R := (1, 0).

Lemma mode_gain_is_exact z v : sq_norm (act z v) = gain z * sq_norm v.
Proof. by rewrite /sq_norm /act /gain /=; ring. Qed.

Lemma sq_norm_ge0 v : 0 <= sq_norm v.
Proof. by rewrite /sq_norm; nra. Qed.

Lemma gain_ge0 z : 0 <= gain z.
Proof. by rewrite /gain; nra. Qed.

Lemma act_compose z w v : act z (act w v) = act (cmul z w) v.
Proof. by rewrite /act /cmul /=; f_equal; ring. Qed.

Lemma act_one v : act cone v = v.
Proof. by case: v => a b; rewrite /act /cone /=; f_equal; ring. Qed.

Lemma gain_mul z w : gain (cmul z w) = gain z * gain w.
Proof. by rewrite /gain /cmul /=; ring. Qed.

Lemma gain_one : gain cone = 1.
Proof. by rewrite /gain /cone /=; ring. Qed.

End OneMode.
Arguments act {R}. Arguments sq_norm {R}. Arguments gain {R}. Arguments cmul {R}.
Arguments cone {R}.

(* ---------- M factors on one mode: composition is ONE symbol, and gains multiply exactly ---------- *)
Section Composition.
Variable R : realFieldType.
Variable I : Type.
Variable z : I -> R * R.

Lemma composition_is_one_symbol (r : seq I) v :
  foldr (fun m acc => act (z m) acc) v r = act (\big[cmul/cone]_(m <- r) z m) v.
Proof.
elim: r => [|m r IH] /=; first by rewrite big_nil act_one.
by rewrite big_cons IH act_compose.
Qed.

Lemma gain_of_composition (r : seq I) :
  gain (\big[cmul/cone]_(m <- r) z m) = \prod_(m <- r) gain (z m).
Proof.
elim: r => [|m r IH]; first by rewrite !big_nil gain_one.
by rewrite !big_cons gain_mul IH.
Qed.

Theorem composition_gain_is_exact (r : seq I) v :
  sq_norm (foldr (fun m acc => act (z m) acc) v r) = (\prod_(m <- r) gain (z m)) * sq_norm v.
Proof. by rewrite composition_is_one_symbol mode_gain_is_exact gain_of_composition. Qed.

End Composition.

(* ---------- N modes: the operator norm is the max gain, and it is ATTAINED ---------- *)
Section Modes.
Variable R : realFieldType.
Variable N : nat.

Definition energy (v : 'I_N -> R * R) : R := \sum_k sq_norm (v k).
Definition top (g : 'I_N -> R) : R := \big[Num.max/0]_k g k.
Definition spike (k0 : 'I_N) (v0 : R * R) : 'I_N -> R * R :=
  fun k => if k == k0 then v0 else (0, 0).

Lemma top_ge0 g : 0 <= top g.
Proof. exact: bigmax_ge_id. Qed.

Lemma le_top g k : g k <= top g.
Proof. exact: le_bigmax. Qed.

Theorem n_mode_bounded (z : 'I_N -> R * R) v :
  energy (fun k => act (z k) (v k)) <= top (fun k => gain (z k)) * energy v.
Proof.
rewrite /energy mulr_sumr; apply: ler_sum => k _; rewrite mode_gain_is_exact.
by apply: ler_wpM2r; [exact: sq_norm_ge0 | exact: (le_top (fun k => gain (z k)))].
Qed.

Lemma energy_spike k0 v0 : energy (spike k0 v0) = sq_norm v0.
Proof.
rewrite /energy (bigD1 k0) //= /spike eqxx big1 ?addr0 // => k hk.
by rewrite (negPf hk) /sq_norm /=; ring.
Qed.

Lemma energy_act_spike (z : 'I_N -> R * R) k0 v0 :
  energy (fun k => act (z k) (spike k0 v0 k)) = gain (z k0) * sq_norm v0.
Proof.
rewrite /energy (bigD1 k0) //= /spike eqxx mode_gain_is_exact big1 ?addr0 // => k hk.
by rewrite (negPf hk) /act /sq_norm /=; ring.
Qed.

(* THE PART A SCHUR BOUND CANNOT HAVE, for any number of modes: all the input on a maximising mode
   turns the bound into an equality. *)
Theorem n_mode_norm_attained (z : 'I_N -> R * R) (j : 'I_N) :
  exists k0, forall v0,
    energy (fun k => act (z k) (spike k0 v0 k)) = top (fun k => gain (z k)) * energy (spike k0 v0).
Proof.
have [k0 _ hk0] := eq_bigmax j xpredT (fun k => gain (z k)) isT (fun i _ => gain_ge0 (z i)).
by exists k0 => v0; rewrite energy_act_spike energy_spike /top hk0.
Qed.

(* M factors on N modes: the composed operator's exact norm is the max over modes of the PRODUCT of
   the per-mode gains, bounded and attained. *)
Theorem composition_norm_bounded M (z : 'I_M -> 'I_N -> R * R) v :
  energy (fun k => foldr (fun m acc => act (z m k) acc) (v k) (index_enum 'I_M))
  <= top (fun k => \prod_m gain (z m k)) * energy v.
Proof.
have -> : energy (fun k => foldr (fun m acc => act (z m k) acc) (v k) (index_enum 'I_M))
          = energy (fun k => act (\big[cmul/cone]_m z m k) (v k)).
  by apply: eq_bigr => k _; rewrite (composition_is_one_symbol (fun m => z m k)).
have -> : top (fun k => \prod_m gain (z m k)) = top (fun k => gain (\big[cmul/cone]_m z m k)).
  by apply: eq_bigr => k _; rewrite (gain_of_composition (fun m => z m k)).
exact: n_mode_bounded.
Qed.

Theorem composition_norm_attained M (z : 'I_M -> 'I_N -> R * R) (j : 'I_N) :
  exists k0, forall v0,
    energy (fun k => foldr (fun m acc => act (z m k) acc) (spike k0 v0 k) (index_enum 'I_M))
    = top (fun k => \prod_m gain (z m k)) * energy (spike k0 v0).
Proof.
have [k0 hk0] := n_mode_norm_attained (fun k => \big[cmul/cone]_m z m k) j.
exists k0 => v0; rewrite -[RHS]/(top _ * _).
have -> : top (fun k => \prod_m gain (z m k)) = top (fun k => gain (\big[cmul/cone]_m z m k)).
  by apply: eq_bigr => k _; rewrite (gain_of_composition (fun m => z m k)).
rewrite -hk0; apply: eq_bigr => k _.
by rewrite (composition_is_one_symbol (fun m => z m k)).
Qed.

End Modes.

(* ---------- why the tube is tight: max of products against product of maxes ---------- *)
Section ProductBound.
Variable R : realFieldType.
Variables M N : nat.
Variable g : 'I_M -> 'I_N -> R.
Hypothesis hg : forall m k, 0 <= g m k.

(* The exact norm of the composition, and what bounding each factor separately gives. *)
Definition exact_norm : R := top (fun k => \prod_m g m k).
Definition factorwise_bound : R := \prod_m top (g m).

Theorem product_bound_is_valid : exact_norm <= factorwise_bound.
Proof.
apply: bigmax_le; first by apply: prodr_ge0 => m _; exact: top_ge0.
move=> k _; apply: ler_prod => m _.
by rewrite hg; exact: le_top.
Qed.

Theorem product_bound_is_tight_at_a_common_maximiser (k0 : 'I_N) :
  (forall m k, g m k <= g m k0) -> exact_norm = factorwise_bound.
Proof.
move=> hk0; apply/le_anti/andP; split; first exact: product_bound_is_valid.
have htop m : top (g m) = g m k0.
  apply/le_anti/andP; split; last exact: le_top.
  by apply: bigmax_le => [|k _]; [exact: hg | exact: hk0].
have hprod : factorwise_bound = \prod_m g m k0 by apply: eq_bigr => m _; exact: htop.
by rewrite hprod; exact: (le_top (fun k => \prod_m g m k) k0).
Qed.

(* ...and STRICTLY loose when no single mode maximises every factor. *)
Theorem product_bound_is_strictly_loose :
  (forall m, 0 < top (g m)) -> (forall k, exists m, g m k < top (g m)) ->
  exact_norm < factorwise_bound.
Proof.
move=> hpos hloose; apply: bigmax_lt; first by apply: prodr_gt0 => m _; exact: hpos.
move=> k _; have [m0 hm0] := hloose k.
rewrite /factorwise_bound (bigD1 m0) //= [X in _ < X](bigD1 m0) //=.
have hrest : 0 < \prod_(m | m != m0) top (g m) by apply: prodr_gt0 => m _; exact: hpos.
have h1 : g m0 k * \prod_(m | m != m0) g m k <= g m0 k * \prod_(m | m != m0) top (g m).
  apply: ler_wpM2l; first exact: hg.
  by apply: ler_prod => m _; rewrite hg; exact: le_top.
have h2 : g m0 k * \prod_(m | m != m0) top (g m) < top (g m0) * \prod_(m | m != m0) top (g m).
  by rewrite ltr_pM2r.
exact: le_lt_trans h1 h2.
Qed.

(* Both directions: the factorwise bound is strictly loose EXACTLY when the factors peak on
   different modes. *)
Theorem strictly_loose_iff_no_common_maximiser :
  (forall m, 0 < top (g m)) ->
  (exact_norm < factorwise_bound <-> forall k, exists m, g m k < top (g m)).
Proof.
move=> hpos; split; last exact: product_bound_is_strictly_loose.
move=> hlt k; case: (boolP [exists m, g m k < top (g m)]) => [/existsP // | /existsPn hno].
have hmax m k' : g m k' <= g m k.
  by apply: le_trans (le_top (g m) k') _; rewrite leNgt; exact: hno.
by move: hlt; rewrite (product_bound_is_tight_at_a_common_maximiser hmax) ltxx.
Qed.

Corollary conservatism_ratio_exceeds_one :
  (forall m, 0 < top (g m)) -> (forall k, exists m, g m k < top (g m)) -> 0 < exact_norm ->
  1 < factorwise_bound / exact_norm.
Proof.
move=> hpos hl h0; rewrite ltr_pdivlMr // mul1r.
exact: product_bound_is_strictly_loose.
Qed.

End ProductBound.
