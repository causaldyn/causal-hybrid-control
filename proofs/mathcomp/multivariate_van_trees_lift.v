(* Rocq + MathComp: VAN TREES ON THE ACTION at any dimension -- the matrix statements of Results 57
   and 67.

   proofs/multivariate_van_trees.v proves them at 2x2 with explicit reals. This file proves them for
   an action of any dimension p and an information of any dimension q, over an abstract field
   (fieldType for the identities, realFieldType for the order):

   - regret_is_an_exact_quadratic_form_in_the_action: for J(u) = (x + Bu)'Q(x + Bu) + u'Ru with Q, R
     symmetric and M = B'QB + R invertible, J(u) - J(uo) = (u - uo)' M (u - uo), uo = -M^-1 B'Qx.
     Nothing is truncated: the regret is a quadratic form in the action error at every dimension.
   - the_expected_regret_is_a_trace: over any finite mixture of action errors, E[regret] = tr(M S).
   - the_floor_decomposes_over_eigendirections: for G = V diag(lam) V' with V'V = 1,
     tr(M Psi G^-1 Psi') = sum_i (Psi v_i)' M (Psi v_i) / lam_i.
   - the_matrix_regret_floor: if the action-error covariance dominates Psi G^-1 Psi' in the PSD
     order, the expected regret is at least tr(M Psi G^-1 Psi'), for every factored curvature
     M = L L' -- and the_lq_curvature_is_a_gram_matrix says every LQ curvature with Gram Q and R
     is one.
   - the_confounding_factor_is_a_convex_combination, a_single_cut_is_exact and its two corners:
     cutting the information by factors in [1, K] raises the floor by a factor in [1, K]; the
     scalar model is the corner where one direction carries all the weight.

   Cited, not proved: the PSD domination itself -- the multivariate van Trees inequality (Gill and
   Levit 1995, Bernoulli 1:59-79; van der Vaart 1998, Thm 2.5.2) -- enters as the named Prop
   PsdDominates, never as an Axiom; it is measure theory, and everything downstream of it is here.
   Eigendecompositions enter as hypotheses (V'V = 1, lam_i != 0), because MathComp's spectral
   theorem is stated over algebraically closed fields only. *)

Set Warnings "-notation-overridden,-ambiguous-paths".
From mathcomp Require Import boot order algebra.
Set Implicit Arguments. Unset Strict Implicit. Unset Printing Implicit Defensive.
Import GRing.Theory Num.Theory Order.Theory.
Local Open Scope ring_scope.

(* ===== Forms on column vectors, over any field ===== *)
Section Forms.
Variable R : fieldType.

Definition ip {k} (v w : 'cV[R]_k) : R := (v^T *m w) 0 0.
Definition qf {k} (M : 'M[R]_k) (v : 'cV[R]_k) : R := ip v (M *m v).

Lemma ipE k (v w : 'cV[R]_k) : ip v w = \sum_i v i 0 * w i 0.
Proof. by rewrite /ip !mxE; apply: eq_bigr => i _; rewrite mxE. Qed.

Lemma ipC k (v w : 'cV[R]_k) : ip v w = ip w v.
Proof. by rewrite !ipE; apply: eq_bigr => i _; rewrite mulrC. Qed.

Lemma ipDl k (u v w : 'cV[R]_k) : ip (u + v) w = ip u w + ip v w.
Proof. by rewrite !ipE -big_split; apply: eq_bigr => i _; rewrite mxE mulrDl. Qed.

Lemma ipDr k (u v w : 'cV[R]_k) : ip u (v + w) = ip u v + ip u w.
Proof. by rewrite ipC ipDl !(ipC u). Qed.

Lemma ipNl k (v w : 'cV[R]_k) : ip (- v) w = - ip v w.
Proof. by rewrite !ipE -sumrN; apply: eq_bigr => i _; rewrite mxE mulNr. Qed.

Lemma ipNr k (v w : 'cV[R]_k) : ip v (- w) = - ip v w.
Proof. by rewrite ipC ipNl ipC. Qed.

Lemma ipBl k (u v w : 'cV[R]_k) : ip (u - v) w = ip u w - ip v w.
Proof. by rewrite ipDl ipNl. Qed.

Lemma ipBr k (u v w : 'cV[R]_k) : ip u (v - w) = ip u v - ip u w.
Proof. by rewrite ipDr ipNr. Qed.

(* Moving a matrix across the form transposes it. *)
Lemma ip_mull k l (A : 'M[R]_(k, l)) (v : 'cV[R]_l) (w : 'cV[R]_k) :
  ip (A *m v) w = ip v (A^T *m w).
Proof. by rewrite /ip trmx_mul mulmxA. Qed.

Lemma ip_mulr_sym k (M : 'M[R]_k) (v w : 'cV[R]_k) :
  M^T = M -> ip v (M *m w) = ip w (M *m v).
Proof. by move=> hM; rewrite ipC ip_mull hM. Qed.

Lemma col_mul a b c (A : 'M[R]_(a, b)) (B : 'M[R]_(b, c)) j :
  col j (A *m B) = A *m col j B.
Proof. by rewrite !colE mulmxA. Qed.

Lemma entry_of_product a b c (A : 'M[R]_(a, b)) (B : 'M[R]_(b, c)) i j :
  (A *m B) i j = (row i A *m col j B) 0 0.
Proof. by rewrite !mxE; apply: eq_bigr => l _; rewrite !mxE. Qed.

(* The diagonal entries of a congruence are quadratic forms in the columns. *)
Lemma entry_of_congruence k l (L : 'M[R]_(k, l)) (D : 'M[R]_k) j :
  (L^T *m D *m L) j j = qf D (col j L).
Proof.
by rewrite entry_of_product row_mul -tr_col /qf /ip mulmxA.
Qed.

Lemma trace11 (A : 'M[R]_1) : \tr A = A 0 0.
Proof. by rewrite /mxtrace big_ord1. Qed.

End Forms.

(* ===== (A) THE REGRET IS AN EXACT QUADRATIC FORM IN THE ACTION, AT ANY DIMENSION ===== *)
Section RegretIdentity.
Variable R : fieldType.
Variables n p : nat.
Variables (Q : 'M[R]_n) (Rw : 'M[R]_p) (B : 'M[R]_(n, p)) (x : 'cV[R]_n).

(* J(u, B) = (x + B u)' Q (x + B u) + u' R u, its curvature M = B'QB + R, its linear term B'Qx, and
   the minimiser u* = -M^(-1) B'Qx. *)
Definition cost (u : 'cV[R]_p) : R := qf Q (x + B *m u) + qf Rw u.
Definition curvature : 'M[R]_p := B^T *m Q *m B + Rw.
Definition lin : 'cV[R]_p := B^T *m Q *m x.
Definition ustar : 'cV[R]_p := - (invmx curvature *m lin).

Hypotheses (hQ : Q^T = Q) (hR : Rw^T = Rw).

Lemma curvature_sym : curvature^T = curvature.
Proof. by rewrite /curvature linearD /= !trmx_mul trmxK hQ hR mulmxA. Qed.

Lemma cost_expand (u : 'cV[R]_p) :
  cost u = qf Q x + (ip u lin + ip u lin) + qf curvature u.
Proof.
have e1 : ip (B *m u) (Q *m x) = ip u lin by rewrite ip_mull /lin !mulmxA.
have e2 : ip x (Q *m (B *m u)) = ip u lin by rewrite ip_mulr_sym // e1.
have e3 : ip (B *m u) (Q *m (B *m u)) = ip u (B^T *m Q *m B *m u).
  by rewrite ip_mull !mulmxA.
rewrite /cost /qf /curvature mulmxDr !ipDl !ipDr e1 e2 e3 mulmxDl ipDr.
by rewrite !addrA.
Qed.

Hypothesis hM : curvature \in unitmx.

Lemma curvature_ustar : curvature *m ustar = - lin.
Proof. by rewrite /ustar mulmxN mulmxA mulmxV // mul1mx. Qed.

Theorem regret_is_an_exact_quadratic_form_in_the_action (u : 'cV[R]_p) :
  cost u - cost ustar = qf curvature (u - ustar).
Proof.
have hs : curvature^T = curvature := curvature_sym.
have e1 : ip u (curvature *m ustar) = - ip u lin by rewrite curvature_ustar ipNr.
have e2 : ip ustar (curvature *m u) = - ip u lin by rewrite ip_mulr_sym // e1.
have e3 : ip ustar (curvature *m ustar) = - ip ustar lin by rewrite curvature_ustar ipNr.
rewrite !cost_expand /qf mulmxBr !ipBl !ipBr e1 e2 e3.
by rewrite (ipC ustar lin) (ipC u lin); ring.
Qed.

(* The minimiser is where the gradient vanishes: M u* + B'Qx = 0. *)
Lemma stationarity : curvature *m ustar + lin = 0.
Proof. by rewrite curvature_ustar addNr. Qed.

End RegretIdentity.

(* ===== (B) A QUADRATIC FORM IS A TRACE, AND THE TRACE DECOMPOSES OVER EIGENDIRECTIONS ===== *)
Section Traces.
Variable R : fieldType.

Lemma a_quadratic_form_is_a_trace k (M : 'M[R]_k) (d : 'cV[R]_k) :
  qf M d = \tr (M *m (d *m d^T)).
Proof. by rewrite mulmxA mxtrace_mulC trace11 /qf /ip. Qed.

(* Expected regret over any finite mixture of action errors is the trace against their second
   moment: E[regret] = tr(M Sigma), with no distributional assumption beyond finite support. *)
Lemma the_expected_regret_is_a_trace k K (M : 'M[R]_k) (w : 'I_K -> R) (d : 'I_K -> 'cV[R]_k) :
  \sum_i w i * qf M (d i) = \tr (M *m \sum_i w i *: (d i *m (d i)^T)).
Proof.
rewrite mulmx_sumr raddf_sum /=; apply: eq_bigr => i _.
by rewrite -scalemxAr mxtraceZ a_quadratic_form_is_a_trace.
Qed.

(* tr(L L' D) = sum_j col_j(L)' D col_j(L), by cyclicity. *)
Lemma trace_of_a_factored_weight k l (L : 'M[R]_(k, l)) (D : 'M[R]_k) :
  \tr (L *m L^T *m D) = \sum_j qf D (col j L).
Proof.
rewrite -mulmxA mxtrace_mulC /mxtrace; apply: eq_bigr => j _.
by rewrite entry_of_congruence.
Qed.

Lemma trace_against_a_diagonal k l (D : 'rV[R]_k) (A : 'M[R]_(k, l)) (C : 'M[R]_(l, k)) :
  \tr (diag_mx D *m (A *m C)) = \sum_i D 0 i * (A *m C) i i.
Proof. by rewrite mul_diag_mx /mxtrace; apply: eq_bigr => i _; rewrite mxE. Qed.

Variables p q : nat.

(* An orthonormal eigenbasis V of the information G = V diag(lam) V' inverts it directly. *)
Definition eig (V : 'M[R]_q) (lam : 'rV[R]_q) : 'M[R]_q := V *m diag_mx lam *m V^T.
Definition inv_row (lam : 'rV[R]_q) : 'rV[R]_q := \row_i (lam 0 i)^-1.

Lemma inverse_of_an_eigendecomposition (V : 'M[R]_q) (lam : 'rV[R]_q) :
  V^T *m V = 1%:M -> (forall i, lam 0 i != 0) ->
  invmx (eig V lam) = eig V (inv_row lam).
Proof.
move=> hV hlam.
have hVV : V *m V^T = 1%:M by exact: mulmx1C.
have hdiag : diag_mx lam *m diag_mx (inv_row lam) = 1%:M.
  rewrite mulmx_diag; apply/matrixP => i j; rewrite !mxE.
  by case: (i =P j) => [->|_]; rewrite ?mulfV ?mulr1n ?mulr0n.
have hone : eig V lam *m eig V (inv_row lam) = 1%:M.
  rewrite /eig -!mulmxA [V^T *m (V *m _)]mulmxA hV mul1mx.
  by rewrite [diag_mx lam *m _]mulmxA hdiag mul1mx hVV.
have [hu _] := mulmx1_unit hone.
by rewrite -[RHS]mul1mx -(mulVmx hu) -mulmxA hone mulmx1.
Qed.

(* THE FLOOR OVER THE EIGENDIRECTIONS OF G: tr(M Psi' G^-1 Psi'') = sum_i (Psi' v_i)' M (Psi' v_i) / lam_i.
   The scalar floor is a product of a curvature and an information; this one interleaves them. *)
Theorem the_floor_decomposes_over_eigendirections (M : 'M[R]_p) (Psi : 'M[R]_(p, q))
    (V : 'M[R]_q) (lam : 'rV[R]_q) :
  V^T *m V = 1%:M -> (forall i, lam 0 i != 0) ->
  \tr (M *m (Psi *m invmx (eig V lam) *m Psi^T))
  = \sum_i qf M (Psi *m col i V) / lam 0 i.
Proof.
move=> hV hlam; rewrite inverse_of_an_eigendecomposition // /eig.
have -> : M *m (Psi *m (V *m diag_mx (inv_row lam) *m V^T) *m Psi^T)
          = (M *m (Psi *m V)) *m (diag_mx (inv_row lam) *m (Psi *m V)^T).
  by rewrite trmx_mul !mulmxA.
rewrite mxtrace_mulC -mulmxA trace_against_a_diagonal.
apply: eq_bigr => i _.
rewrite mulmxA -[(Psi *m V)^T *m M *m (Psi *m V)]/((Psi *m V)^T *m M *m (Psi *m V)).
by rewrite entry_of_congruence col_mul !mxE mulrC.
Qed.

(* With a diagonal G^-1 = diag(h) the floor is a sum over the columns of Psi' (Maxima STEP 5b). *)
Lemma the_diagonal_floor_is_a_sum_over_columns (M : 'M[R]_p) (Psi : 'M[R]_(p, q)) (h : 'rV[R]_q) :
  \tr (M *m (Psi *m diag_mx h *m Psi^T)) = \sum_i h 0 i * qf M (col i Psi).
Proof.
have -> : M *m (Psi *m diag_mx h *m Psi^T) = (M *m Psi) *m (diag_mx h *m Psi^T).
  by rewrite !mulmxA.
rewrite mxtrace_mulC -mulmxA trace_against_a_diagonal.
by apply: eq_bigr => i _; rewrite mulmxA entry_of_congruence.
Qed.

End Traces.

(* ===== (C) THE PSD ORDER TRANSFERS TO THE REGRET; CONFOUNDING IS PRICED BY ALIGNMENT ===== *)
Section Order.
Variable R : realFieldType.

Definition psd {k} (M : 'M[R]_k) : Prop := forall v : 'cV[R]_k, 0 <= qf M v.
Definition pd {k} (M : 'M[R]_k) : Prop := forall v : 'cV[R]_k, v != 0 -> 0 < qf M v.

Lemma qfD k (M N : 'M[R]_k) (v : 'cV[R]_k) : qf (M + N) v = qf M v + qf N v.
Proof. by rewrite /qf mulmxDl ipDr. Qed.

Lemma qfB k (M N : 'M[R]_k) (v : 'cV[R]_k) : qf (M - N) v = qf M v - qf N v.
Proof. by rewrite /qf mulmxBl ipBr. Qed.

Lemma qf_congruence k l (A : 'M[R]_(k, l)) (M : 'M[R]_k) (v : 'cV[R]_l) :
  qf (A^T *m M *m A) v = qf M (A *m v).
Proof. by rewrite /qf ip_mull -!mulmxA. Qed.

Lemma trace_against_a_psd_weight_is_nonnegative k l (L : 'M[R]_(k, l)) (D : 'M[R]_k) :
  psd D -> 0 <= \tr (L *m L^T *m D).
Proof. by move=> hD; rewrite trace_of_a_factored_weight; apply: sumr_ge0 => j _; exact: hD. Qed.

(* Gill & Levit (1995) Bernoulli 1:59-79; van der Vaart (1998) Thm 2.5.2, MATRIX form -- the cited
   measure-theoretic input, named so that `Check` says where it enters (plans/24 P1.2). For any
   estimator the Bayes action-error covariance dominates Psi' G^-1 Psi'' in the PSD order. A
   Definition and not an Axiom, for the reason spelled out in proofs/c2_end_to_end.v. *)
Definition PsdDominates {k} (Sigma Floor : 'M[R]_k) : Prop := psd (Sigma - Floor).

Theorem the_psd_order_transfers_to_the_regret k l (L : 'M[R]_(k, l)) (Sigma Floor : 'M[R]_k) :
  PsdDominates Sigma Floor -> \tr (L *m L^T *m Floor) <= \tr (L *m L^T *m Sigma).
Proof.
move=> hdom; rewrite -subr_ge0 -raddfB /= -mulmxBr.
exact: trace_against_a_psd_weight_is_nonnegative.
Qed.

(* The factored weight is not an extra assumption for an LQ curvature: with Q = Cq'Cq and R = Cr'Cr
   given as Gram matrices, M = B'QB + R = L L' with L = [ (Cq B)'  Cr' ]. *)
Lemma the_lq_curvature_is_a_gram_matrix n p a b (Cq : 'M[R]_(a, n)) (Cr : 'M[R]_(b, p))
    (B : 'M[R]_(n, p)) :
  curvature (Cq^T *m Cq) (Cr^T *m Cr) B
  = row_mx (Cq *m B)^T Cr^T *m (row_mx (Cq *m B)^T Cr^T)^T.
Proof.
by rewrite /curvature tr_row_mx mul_row_col !trmxK trmx_mul !mulmxA.
Qed.

(* THE MATRIX FLOOR, in the vocabulary of its cited input: E[regret] = tr(M Sigma) >= tr(M Floor). *)
Theorem the_matrix_regret_floor k l (L : 'M[R]_(k, l)) (Sigma Floor : 'M[R]_k) (er : R) :
  PsdDominates Sigma Floor -> \tr (L *m L^T *m Sigma) <= er ->
  \tr (L *m L^T *m Floor) <= er.
Proof. by move=> hdom; apply: le_trans (the_psd_order_transfers_to_the_regret L hdom). Qed.

Lemma the_curvature_is_definite n p (Q : 'M[R]_n) (Rw : 'M[R]_p) (B : 'M[R]_(n, p)) :
  psd Q -> pd Rw -> pd (curvature Q Rw B).
Proof.
move=> hQ hR v hv; rewrite /curvature qfD qf_congruence.
by rewrite ltr_wpDl ?hQ ?hR.
Qed.

Lemma a_blind_direction_contributes_nothing k l (M : 'M[R]_k) (Psi : 'M[R]_(k, l)) (v : 'cV[R]_l) :
  Psi *m v = 0 -> qf M (Psi *m v) = 0.
Proof. by move=> ->; rewrite /qf /ip mulmx0 mulmx0 mxE. Qed.

(* The converse that makes the knife edge a RANK condition: a definite curvature hides nothing else. *)
Lemma a_definite_curvature_hides_nothing_else k l (M : 'M[R]_k) (Psi : 'M[R]_(k, l)) (v : 'cV[R]_l) :
  pd M -> qf M (Psi *m v) = 0 -> Psi *m v = 0.
Proof.
move=> hM h0; apply/eqP; apply: contraT => hne.
by have := hM _ hne; rewrite h0 ltxx.
Qed.

(* --- Alignment: the price of confounding is a convex combination over the eigendirections. --- *)
Variable q : nat.

(* Weights a_i = (Psi' v_i)' M (Psi' v_i) / lam_i; the floor is their sum. Cutting the information
   along each direction by k_i >= 1 multiplies each weight by k_i and nothing else. *)
Lemma cut_information_scales_each_weight p (M : 'M[R]_p) (Psi : 'M[R]_(p, q))
    (V : 'M[R]_q) (lam kk : 'rV[R]_q) :
  V^T *m V = 1%:M -> (forall i, lam 0 i != 0) -> (forall i, kk 0 i != 0) ->
  \tr (M *m (Psi *m invmx (eig V (\row_i (lam 0 i / kk 0 i))) *m Psi^T))
  = \sum_i kk 0 i * (qf M (Psi *m col i V) / lam 0 i).
Proof.
move=> hV hlam hk.
have hlk : forall i, (\row_i (lam 0 i / kk 0 i)) 0 i != 0.
  by move=> i; rewrite mxE mulf_neq0 ?invr_eq0.
rewrite the_floor_decomposes_over_eigendirections //.
apply: eq_bigr => i _; rewrite mxE.
have := hlam i; have := hk i => hki hli.
by field.
Qed.

Lemma the_confounding_factor_is_a_convex_combination (a kk : 'rV[R]_q) (K : R) :
  (forall i, 0 <= a 0 i) -> (forall i, 1 <= kk 0 i <= K) ->
  \sum_i a 0 i <= \sum_i kk 0 i * a 0 i <= K * \sum_i a 0 i.
Proof.
move=> ha hk; apply/andP; split.
  apply: ler_sum => i _; have /andP[h1 _] := hk i; have := ha i; nra.
rewrite mulr_sumr; apply: ler_sum => i _; have /andP[_ h2] := hk i; have := ha i; nra.
Qed.

(* One direction cut by kappa: the floor rises by EXACTLY (kappa - 1) a_w. *)
Lemma a_single_cut_is_exact (a : 'rV[R]_q) (w : 'I_q) (kappa : R) :
  \sum_i (if i == w then kappa else 1) * a 0 i = \sum_i a 0 i + (kappa - 1) * a 0 w.
Proof.
rewrite (bigD1 w) //= eqxx [in RHS](bigD1 w) //=.
have e : \sum_(i < q | i != w) (if i == w then kappa else 1) * a 0 i
         = \sum_(i < q | i != w) a 0 i.
  by apply: eq_bigr => i /negPf ->; rewrite mul1r.
by rewrite e; ring.
Qed.

(* The corners of the combination: a direction the action cannot see costs exactly nothing, and a
   direction carrying all the weight pays the full factor -- which is the scalar model. *)
Lemma a_weightless_direction_costs_nothing (a : 'rV[R]_q) (w : 'I_q) (kappa : R) :
  a 0 w = 0 -> \sum_i (if i == w then kappa else 1) * a 0 i = \sum_i a 0 i.
Proof. by move=> h; rewrite a_single_cut_is_exact h mulr0 addr0. Qed.

Lemma a_single_direction_pays_the_full_factor (a : 'rV[R]_q) (w : 'I_q) (kappa : R) :
  (forall i, i != w -> a 0 i = 0) ->
  \sum_i (if i == w then kappa else 1) * a 0 i = kappa * \sum_i a 0 i.
Proof.
move=> h; have e : \sum_(i < q | i != w) a 0 i = 0 by apply: big1 => i /h.
by rewrite a_single_cut_is_exact (bigD1 w) //= e addr0; ring.
Qed.

End Order.
