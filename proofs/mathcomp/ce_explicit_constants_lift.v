(* Rocq + MathComp: THE MATRIX STEPS OF RESULT 44, and the positive-definite side of Result 45.

   proofs/ce_explicit_constants.v proves the SCALAR shadow of each step of Result 44, because Stdlib
   has no matrices. This file proves the matrix statements for any state dimension n and action
   dimension m, over an abstract field (identities) or real field (order):

   - completing_the_square: for every gain K' and symmetric P and R with R_K := R + B'PB invertible,
       Q + K''RK' + (A - BK')'P(A - BK') = Ric(P) + (K' - K)' R_K (K' - K),   K := R_K^-1 B'PA.
     gain_gap_identity is its DARE case: the Lyapunov increment is EXACTLY a perfect square in the
     gain error, with no smallness assumption.
   - cost_telescopes: along the closed loop of any gain, the T-step cost is x0'Px0 - xT'PxT plus
     the summed gain gap -- the exact regret-to-go.
   - lyapunov_triangle / perturbed_loop_contraction: in the P metric a nominal contraction s^2 and a
     perturbation e^2 with e <= (1 - s)/2 give the contraction (1 - (1 - s)/2)^2 -- the explicit
     stabilising ball, with no square root taken.
   - summed_gain_gap_explicit_bound / ce_regret_explicit_bound_matrix: the certainty-equivalence
     regret over ANY horizon T is at most 2 (hi/lo) ||R_K|| L_K^2 |x0|^2 / theta * d^2 -- the
     scalar shadow's constant C, now for the matrix loop.
   - no_conjugate_point_with_a_psd_cost (Result 45): for Q >= 0, R > 0 and P0 >= 0 every iterate of
     the discrete Riccati recursion P_{k+1} = Ric(P_k) stays symmetric and PSD, and every
     R + B'P_kB stays positive definite, so the recursion never breaks down.

   Cited, not proved. Eigenvalue and norm constants (lo, hi, ||R_K||, the contraction modulus)
   enter as the quadratic-form inequalities they denote, because MathComp's spectral theorem is
   stated over algebraically closed fields only. The gain perturbation bound ||K^ - K|| <= L_K d --
   uniform validity of L_K on the ball (Konstantinov-Petkov-Christov-Angelova 1993; Sun 1998) --
   enters as the hypothesis on the gain gap, as in the scalar file. Result 45's continuous-time
   conjugate time is the escape time of a Riccati ODE, which is analysis, and stays in Maxima. *)

Set Warnings "-notation-overridden,-ambiguous-paths".
From mathcomp Require Import boot order algebra.
Set Implicit Arguments. Unset Strict Implicit. Unset Printing Implicit Defensive.
Import GRing.Theory Num.Theory Order.Theory.
Local Open Scope ring_scope.

(* ===== Forms on column vectors ===== *)
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

Lemma ipZl k (a : R) (v w : 'cV[R]_k) : ip (a *: v) w = a * ip v w.
Proof. by rewrite !ipE mulr_sumr; apply: eq_bigr => i _; rewrite mxE mulrA. Qed.

Lemma ipZr k (a : R) (v w : 'cV[R]_k) : ip v (a *: w) = a * ip v w.
Proof. by rewrite ipC ipZl ipC. Qed.

Lemma ipNr k (v w : 'cV[R]_k) : ip v (- w) = - ip v w.
Proof. by rewrite -scaleN1r ipZr mulN1r. Qed.

Lemma ip0r k (v : 'cV[R]_k) : ip v 0 = 0.
Proof. by rewrite ipE big1 // => i _; rewrite mxE mulr0. Qed.

Lemma ip_mull k l (A : 'M[R]_(k, l)) (v : 'cV[R]_l) (w : 'cV[R]_k) :
  ip (A *m v) w = ip v (A^T *m w).
Proof. by rewrite /ip trmx_mul mulmxA. Qed.

Lemma qfD k (M N : 'M[R]_k) (v : 'cV[R]_k) : qf (M + N) v = qf M v + qf N v.
Proof. by rewrite /qf mulmxDl ipDr. Qed.

Lemma qfB k (M N : 'M[R]_k) (v : 'cV[R]_k) : qf (M - N) v = qf M v - qf N v.
Proof. by rewrite /qf mulmxBl ipDr ipNr. Qed.

Lemma qf_congruence k l (A : 'M[R]_(k, l)) (M : 'M[R]_k) (v : 'cV[R]_l) :
  qf (A^T *m M *m A) v = qf M (A *m v).
Proof. by rewrite /qf ip_mull -!mulmxA. Qed.

Lemma qf0m k (v : 'cV[R]_k) : qf 0 v = 0.
Proof. by rewrite /qf mul0mx ip0r. Qed.

(* For a symmetric weight, the form of a sum expands with a cross term counted twice. *)
Lemma qf_sum k (P : 'M[R]_k) (u v : 'cV[R]_k) :
  P^T = P -> qf P (u + v) = qf P u + (ip u (P *m v) + ip u (P *m v)) + qf P v.
Proof.
move=> hP; rewrite /qf mulmxDr !ipDl !ipDr.
have -> : ip v (P *m u) = ip u (P *m v) by rewrite ipC ip_mull hP.
by rewrite !addrA.
Qed.

End Forms.

(* ===== (A) THE EXACT GAIN-GAP IDENTITY: the matrix statement, with no smallness ===== *)
Section GainGap.
Variable R : fieldType.
Variables n m : nat.
Variables (A : 'M[R]_n) (B : 'M[R]_(n, m)) (Q : 'M[R]_n) (Rw : 'M[R]_m).

(* R_K := R + B'PB, the optimal gain K := R_K^-1 B'PA for a given P, and the Riccati map
   Ric(P) := Q + A'PA - A'PB R_K^-1 B'PA; the DARE is Ric(P) = P. *)
Definition RK (P : 'M[R]_n) : 'M[R]_m := Rw + B^T *m P *m B.
Definition Sx (P : 'M[R]_n) : 'M[R]_(m, n) := B^T *m P *m A.
Definition Kopt (P : 'M[R]_n) : 'M[R]_(m, n) := invmx (RK P) *m Sx P.
Definition riccati (P : 'M[R]_n) : 'M[R]_n :=
  Q + A^T *m P *m A - (Sx P)^T *m invmx (RK P) *m Sx P.
(* The exact quadratic form in the gain error that the increment differs by. *)
Definition gain_gap (P : 'M[R]_n) (K' : 'M[R]_(m, n)) : 'M[R]_n :=
  (K' - Kopt P)^T *m RK P *m (K' - Kopt P).
(* The one-step Lyapunov increment of a gain K': stage cost plus the propagated P. *)
Definition stage (P : 'M[R]_n) (K' : 'M[R]_(m, n)) : 'M[R]_n :=
  Q + K'^T *m Rw *m K' + (A - B *m K')^T *m P *m (A - B *m K').

Variable P : 'M[R]_n.
Hypotheses (hP : P^T = P) (hR : Rw^T = Rw).

Lemma RK_sym : (RK P)^T = RK P.
Proof. by rewrite /RK linearD /= !trmx_mul trmxK hP hR mulmxA. Qed.

(* Additive rearrangements between matrix atoms, decided entrywise by the ring tactic. *)
Lemma rearrange_stage a b (q r c y z d : 'M[R]_(a, b)) :
  q + r + (c - y - (z - d)) = q + c + (r + d) - y - z.
Proof. by apply/matrixP => i j; rewrite !mxE; ring. Qed.

Lemma rearrange_square a b (C X Y Z W : 'M[R]_(a, b)) :
  C + X - Y - Z = C - W + (X - Z - Y + W).
Proof. by apply/matrixP => i j; rewrite !mxE; ring. Qed.

Lemma expand_square a b (E F : 'M[R]_(a, b)) (G : 'M[R]_a) :
  (E - F)^T *m G *m (E - F) = E^T *m G *m E - E^T *m G *m F - F^T *m G *m E + F^T *m G *m F.
Proof.
rewrite [(E - F)^T]linearB /= !mulmxBl !mulmxBr.
by apply/matrixP => i j; rewrite !mxE; ring.
Qed.

Hypothesis hRK : RK P \in unitmx.

Lemma RK_Kopt : RK P *m Kopt P = Sx P.
Proof. by rewrite /Kopt mulmxA mulmxV // mul1mx. Qed.

Lemma Kopt_RK : (Kopt P)^T *m RK P = (Sx P)^T.
Proof.
rewrite /Kopt trmx_mul trmx_inv RK_sym -mulmxA mulVmx //.
by rewrite mulmx1.
Qed.

(* COMPLETING THE SQUARE, for every gain K' and every symmetric P: the increment is the Riccati map
   plus an exact quadratic form in the gain error. *)
Theorem completing_the_square (K' : 'M[R]_(m, n)) :
  stage P K' = riccati P + gain_gap P K'.
Proof.
rewrite /gain_gap.
have eS : A^T *m P *m B = (Sx P)^T by rewrite /Sx !trmx_mul trmxK hP mulmxA.
have eX : K'^T *m Rw *m K' + K'^T *m (B^T *m P *m B) *m K' = K'^T *m RK P *m K'.
  by rewrite /RK mulmxDr mulmxDl.
have eL : stage P K'
          = Q + A^T *m P *m A + K'^T *m RK P *m K' - (Sx P)^T *m K' - K'^T *m Sx P.
  rewrite /stage [(A - B *m K')^T]linearB /= trmx_mul !mulmxBl !mulmxBr -eX.
  have -> : A^T *m P *m (B *m K') = (Sx P)^T *m K' by rewrite -eS !mulmxA.
  have -> : K'^T *m B^T *m P *m A = K'^T *m Sx P by rewrite /Sx !mulmxA.
  have -> : K'^T *m B^T *m P *m (B *m K') = K'^T *m (B^T *m P *m B) *m K'.
    by rewrite !mulmxA.
  exact: rearrange_stage.
have eR : riccati P = Q + A^T *m P *m A - (Sx P)^T *m Kopt P.
  by rewrite /riccati /Kopt mulmxA.
have eG : (K' - Kopt P)^T *m RK P *m (K' - Kopt P)
          = K'^T *m RK P *m K' - K'^T *m Sx P - (Sx P)^T *m K' + (Sx P)^T *m Kopt P.
  by rewrite expand_square Kopt_RK -[K'^T *m RK P *m Kopt P]mulmxA RK_Kopt.
by rewrite eL eR eG; exact: rearrange_square.
Qed.

(* THE GAIN-GAP IDENTITY: at a DARE solution the increment is EXACTLY a perfect square. *)
Corollary gain_gap_identity (K' : 'M[R]_(m, n)) :
  riccati P = P -> stage P K' - P = gain_gap P K'.
Proof. by move=> hdare; rewrite completing_the_square hdare addrC addKr. Qed.

(* The DARE is the K' = K case of the identity. *)
Corollary gain_gap_zero_at_optimum : riccati P = P -> stage P (Kopt P) = P.
Proof.
by move=> hdare; rewrite completing_the_square hdare /gain_gap subrr trmx0 mul0mx mulmx0 addr0.
Qed.

End GainGap.

(* ===== (B) THE ORDER FACTS: positivity, the Lyapunov step, and the exact regret-to-go ===== *)
Section Positivity.
Variable R : realFieldType.

Definition psd {k} (M : 'M[R]_k) : Prop := forall v : 'cV[R]_k, 0 <= qf M v.
Definition pd {k} (M : 'M[R]_k) : Prop := forall v : 'cV[R]_k, v != 0 -> 0 < qf M v.

Lemma qfN k (M : 'M[R]_k) (v : 'cV[R]_k) : qf M (- v) = qf M v.
Proof. by rewrite /qf mulmxN ipNr -scaleN1r ipZl mulN1r opprK. Qed.

(* A definite matrix is invertible: a kernel vector would have a zero form. *)
Lemma pd_unitmx k (M : 'M[R]_k) : pd M -> M \in unitmx.
Proof.
move=> hM; rewrite -unitmx_tr -row_free_unit -kermx_eq0.
apply/eqP/row_matrixP => i; rewrite row0.
apply/eqP; apply: contraT => hne.
have hv : (row i (kermx M^T))^T != 0 by rewrite trmx_eq0.
have h0 : row i (kermx M^T) *m M^T = 0 by rewrite -row_mul mulmx_ker row0.
have hMv : M *m (row i (kermx M^T))^T = 0.
  by have := congr1 trmx h0; rewrite trmx_mul trmxK trmx0.
by have := hM _ hv; rewrite /qf hMv ip0r ltxx.
Qed.

Variables n m : nat.
Variables (A : 'M[R]_n) (B : 'M[R]_(n, m)) (Q : 'M[R]_n) (Rw : 'M[R]_m) (P : 'M[R]_n).

Lemma rk_ge_r (v : 'cV[R]_m) : psd P -> qf Rw v <= qf (RK B Rw P) v.
Proof. by move=> hP; rewrite /RK qfD qf_congruence lerDl hP. Qed.

Lemma rk_pd : psd P -> pd Rw -> pd (RK B Rw P).
Proof. by move=> hP hR v hv; apply: lt_le_trans (hR v hv) (rk_ge_r v hP). Qed.

Lemma gain_gap_psd (E : 'M[R]_(m, n)) : psd (RK B Rw P) -> psd (E^T *m RK B Rw P *m E).
Proof. by move=> hRK v; rewrite qf_congruence. Qed.

Hypotheses (hPs : P^T = P) (hRs : Rw^T = Rw) (hRK : RK B Rw P \in unitmx).
Hypothesis hdare : riccati A B Q Rw P = P.

(* THE LYAPUNOV STEP, exact for every gain: the P-energy falls by the stage cost and rises by the
   gain gap, with nothing dropped. *)
Lemma lyapunov_step (K' : 'M[R]_(m, n)) (x : 'cV[R]_n) :
  qf P ((A - B *m K') *m x) = qf P x - qf (Q + K'^T *m Rw *m K') x + qf (gain_gap A B Rw P K') x.
Proof.
have h := congr1 (fun M => qf M x) (gain_gap_identity hPs hRs hRK K' hdare).
rewrite /= qfB /stage qfD [qf (_^T *m P *m _) x]qf_congruence in h.
lra.
Qed.

Lemma gain_gap_nonneg (K' : 'M[R]_(m, n)) (x : 'cV[R]_n) :
  psd P -> psd Rw -> qf P x <= qf (stage A B Q Rw P K') x.
Proof.
move=> hP hR; have h := congr1 (fun M => qf M x) (gain_gap_identity hPs hRs hRK K' hdare).
rewrite /= qfB in h; rewrite -subr_ge0 h /gain_gap qf_congruence.
exact: le_trans (hR _) (rk_ge_r _ hP).
Qed.

(* THE EXACT REGRET-TO-GO along the closed loop of any gain: summing the step telescopes the
   P-energy, so the cost of K' over T steps is x0'Px0 - xT'PxT plus the summed gain gap. *)
Theorem cost_telescopes (K' : 'M[R]_(m, n)) (xs : nat -> 'cV[R]_n) :
  (forall t, xs t.+1 = (A - B *m K') *m xs t) ->
  forall T, \sum_(t < T) qf (Q + K'^T *m Rw *m K') (xs t)
            = qf P (xs 0) - qf P (xs T) + \sum_(t < T) qf (gain_gap A B Rw P K') (xs t).
Proof.
move=> hstep; elim=> [|T IH]; first by rewrite !big_ord0 subrr addr0.
rewrite !big_ord_recr /= IH hstep lyapunov_step.
by ring.
Qed.

(* At the optimal gain the gap vanishes and the cost-to-go is exactly x0'Px0 - xT'PxT. *)
Corollary optimal_cost_telescopes (xs : nat -> 'cV[R]_n) :
  (forall t, xs t.+1 = (A - B *m Kopt A B Rw P) *m xs t) ->
  forall T, \sum_(t < T) qf (Q + (Kopt A B Rw P)^T *m Rw *m Kopt A B Rw P) (xs t)
            = qf P (xs 0) - qf P (xs T).
Proof.
move=> hstep T; rewrite (cost_telescopes hstep) big1 ?addr0 // => t _.
by rewrite /gain_gap subrr trmx0 !mul0mx /qf mul0mx ip0r.
Qed.

End Positivity.

(* ===== (C) THE STABILISING BALL: a triangle inequality in the P metric, without square roots ===== *)
Section Contraction.
Variable R : realFieldType.
Variable n : nat.
Variable P : 'M[R]_n.
Hypotheses (hPs : P^T = P) (hP : psd P).

Lemma qf_scale (t : R) (v : 'cV[R]_n) : qf P (t *: v) = t ^+ 2 * qf P v.
Proof. by rewrite /qf -scalemxAr ipZl ipZr mulrA expr2. Qed.

(* Cauchy-Schwarz for a PSD form, by the discriminant. *)
Lemma psd_cauchy_schwarz (u v : 'cV[R]_n) : ip u (P *m v) ^+ 2 <= qf P u * qf P v.
Proof.
have quad t : 0 <= qf P u + (t * ip u (P *m v) + t * ip u (P *m v)) + t ^+ 2 * qf P v.
  by have := hP (u + t *: v); rewrite qf_sum // -scalemxAr ipZr qf_scale.
set a := qf P u in quad *; set b := ip u (P *m v) in quad *; set c := qf P v in quad *.
have hc : 0 <= c := hP v.
have [c0|cpos] := eqVneq c 0.
  have [b0|bne] := eqVneq b 0; first by rewrite b0 c0 expr2 !mul0r mulr0.
  have := quad (- (a + 1) / (b + b)); rewrite c0 mulr0 addr0.
  have hbb : b + b != 0 by rewrite -mulr2n mulrn_eq0 negb_or bne.
  have -> : - (a + 1) / (b + b) * b + - (a + 1) / (b + b) * b = - (a + 1).
    by rewrite -mulrDr; field.
  lra.
have c0 : 0 < c by rewrite lt0r cpos hc.
set t := - b / c.
have tc : t * c = - b by rewrite /t divfK.
have := quad t; nra.
Qed.

(* THE PERTURBED LOOP, in the P metric: moduli add. Squared form throughout, so no square root and
   no eigenvalue enters -- the matrix statement behind the scalar shadow's s + beta_B L_K d. *)
Lemma lyapunov_triangle (M E : 'M[R]_n) (s e : R) :
  0 <= s -> 0 <= e ->
  (forall x, qf P (M *m x) <= s ^+ 2 * qf P x) ->
  (forall x, qf P (E *m x) <= e ^+ 2 * qf P x) ->
  forall x, qf P ((M + E) *m x) <= (s + e) ^+ 2 * qf P x.
Proof.
move=> hs he hM hE x; rewrite mulmxDl qf_sum //.
have hcs := psd_cauchy_schwarz (M *m x) (E *m x).
set b := ip (M *m x) (P *m (E *m x)) in hcs *.
have h1 := hM x; have h2 := hE x; have hp := hP x.
have hpM := hP (M *m x); have hpE := hP (E *m x).
have hsep : 0 <= s * e * qf P x by rewrite !mulr_ge0.
have hb2 : b ^+ 2 <= (s * e * qf P x) ^+ 2.
  apply: le_trans hcs _; rewrite !exprMn.
  have -> : s ^+ 2 * e ^+ 2 * qf P x ^+ 2 = (s ^+ 2 * qf P x) * (e ^+ 2 * qf P x).
    by rewrite expr2; ring.
  by apply: ler_pM.
have hb : b <= s * e * qf P x by nra.
nra.
Qed.

Lemma qfN' (v : 'cV[R]_n) : qf P (- v) = qf P v.
Proof. by rewrite -scaleN1r qf_scale sqrrN expr1n mul1r. Qed.

(* Linear algebra behind beta_B: a gain error of operator size dk moves the state by at most
   bb * dk^2 * |x|^2 in the P metric, and |x|^2 <= x'Px / lo. *)
Lemma perturbation_in_the_P_metric m (B : 'M[R]_(n, m)) (DK : 'M[R]_(m, n)) (bb dk lo e : R) :
  0 < lo -> 0 <= bb ->
  (forall y, qf P (B *m y) <= bb * ip y y) ->
  (forall x, ip (DK *m x) (DK *m x) <= dk ^+ 2 * ip x x) ->
  (forall x, lo * ip x x <= qf P x) ->
  bb * dk ^+ 2 <= lo * e ^+ 2 ->
  forall x, qf P ((B *m DK) *m x) <= e ^+ 2 * qf P x.
Proof.
move=> hlo hbb hB hDK hlow he x.
have hxx : 0 <= ip x x.
  by rewrite ipE; apply: sumr_ge0 => i _; rewrite -expr2 sqr_ge0.
have h1 := hB (DK *m x); have h2 := hDK x; have h3 := hlow x.
rewrite -mulmxA; apply: le_trans h1 _.
have h4 : bb * ip (DK *m x) (DK *m x) <= bb * (dk ^+ 2 * ip x x) by exact: ler_wpM2l.
apply: le_trans h4 _.
have h5 : bb * dk ^+ 2 * ip x x <= lo * e ^+ 2 * ip x x by exact: ler_wpM2r.
have h6 : lo * e ^+ 2 * ip x x <= e ^+ 2 * qf P x.
  by rewrite -mulrA mulrCA; apply: ler_wpM2l; [exact: sqr_ge0 | exact: h3].
by rewrite mulrA; apply: le_trans h5 h6.
Qed.

(* The energy of a contracting loop decays geometrically. *)
Lemma lyapunov_decay (N : 'M[R]_n) (mu : R) (xs : nat -> 'cV[R]_n) :
  0 <= mu -> (forall x, qf P (N *m x) <= mu * qf P x) -> (forall t, xs t.+1 = N *m xs t) ->
  forall t, qf P (xs t) <= mu ^+ t * qf P (xs 0).
Proof.
move=> hmu hN hstep; elim=> [|t IH]; first by rewrite expr0 mul1r.
rewrite hstep exprS -mulrA; apply: le_trans (hN _) _.
exact: ler_wpM2l.
Qed.

End Contraction.

(* ===== (D) The geometric sum, with the relaxation the constant uses ===== *)
Section Geometric.
Variable R : realFieldType.

Lemma geo_partial_identity (mu : R) T : (1 - mu) * \sum_(t < T) mu ^+ t = 1 - mu ^+ T.
Proof.
elim: T => [|T IH]; first by rewrite big_ord0 mulr0 expr0 subrr.
by rewrite big_ord_recr /= mulrDr IH exprS; ring.
Qed.

Lemma geo_partial_le (mu : R) T : 0 <= mu -> mu < 1 -> \sum_(t < T) mu ^+ t <= (1 - mu)^-1.
Proof.
move=> h0 h1; have hq : 0 < 1 - mu by rewrite subr_gt0.
rewrite -(ler_pM2l hq) mulfV ?gt_eqF // geo_partial_identity.
by rewrite lerBlDr lerDl exprn_ge0.
Qed.

Lemma geo_relaxation (th : R) : 0 < th -> th <= 1 -> (1 - (1 - th / 2) ^+ 2)^-1 <= 2 / th.
Proof.
move=> h0 h1.
have h2 : (2 : R) != 0 by rewrite pnatr_eq0.
have h4 : (4%:R : R) != 0 by rewrite pnatr_eq0.
have hth : th != 0 by rewrite gt_eqF.
have hd : 0 < 1 - (1 - th / 2) ^+ 2.
  have -> : 1 - (1 - th / 2) ^+ 2 = th * (4%:R - th) / 4%:R by rewrite expr2; field.
  by rewrite divr_gt0 ?mulr_gt0 // ?subr_gt0; lra.
rewrite -(ler_pM2l hd) mulfV ?gt_eqF //.
have -> : (1 - (1 - th / 2) ^+ 2) * (2 / th) = (4%:R - th) / 2 by rewrite expr2; field.
by rewrite ler_pdivlMr //; lra.
Qed.

End Geometric.

(* ===== (E) THEOREM 44, as a matrix statement ===== *)
Section Theorem44.
Variable R : realFieldType.
Variables n m : nat.
Variables (A : 'M[R]_n) (B : 'M[R]_(n, m)) (Q : 'M[R]_n) (Rw : 'M[R]_m) (P : 'M[R]_n).
Hypotheses (hPs : P^T = P) (hRs : Rw^T = Rw) (hRK : RK B Rw P \in unitmx).
Hypotheses (hdare : riccati A B Q Rw P = P) (hP : psd P).

(* The optimal loop contracts at 1 - eta in the P metric, eta the ratio of the stage cost to P. *)
Lemma nominal_loop_contracts (eta : R) :
  (forall x, eta * qf P x <= qf (Q + (Kopt A B Rw P)^T *m Rw *m Kopt A B Rw P) x) ->
  forall x, qf P ((A - B *m Kopt A B Rw P) *m x) <= (1 - eta) * qf P x.
Proof.
move=> heta x; rewrite (lyapunov_step hPs hRs hRK hdare) /gain_gap subrr.
rewrite trmx0 !mul0mx qf0m addr0.
by have := heta x; lra.
Qed.

(* THE MISSING LINK, matrix form: inside the ball the certainty-equivalent gain still contracts the
   TRUE loop, with margin theta/2 where theta = 1 - s. The checkable Lyapunov certificate
   (A - B Khat)' P (A - B Khat) <= (1 - theta/2)^2 P, as a statement about every x. *)
Theorem perturbed_loop_contraction (Khat : 'M[R]_(m, n)) (s e : R) :
  0 <= s -> s < 1 -> 0 <= e -> e <= (1 - s) / 2 ->
  (forall x, qf P ((A - B *m Kopt A B Rw P) *m x) <= s ^+ 2 * qf P x) ->
  (forall x, qf P ((B *m (Khat - Kopt A B Rw P)) *m x) <= e ^+ 2 * qf P x) ->
  forall x, qf P ((A - B *m Khat) *m x) <= (1 - (1 - s) / 2) ^+ 2 * qf P x.
Proof.
move=> hs0 hs1 he0 he1 hnom hpert x.
have -> : A - B *m Khat = (A - B *m Kopt A B Rw P) + - (B *m (Khat - Kopt A B Rw P)).
  by rewrite mulmxBr; apply/matrixP => i j; rewrite !mxE; ring.
have hE : forall y, qf P (- (B *m (Khat - Kopt A B Rw P)) *m y) <= e ^+ 2 * qf P y.
  by move=> y; rewrite mulNmx qfN'.
apply: le_trans (lyapunov_triangle hPs hP hs0 he0 hnom hE x) _.
apply: ler_wpM2r; first exact: hP.
have hle : s + e <= 1 - (1 - s) / 2 by lra.
have hse : 0 <= s + e by lra.
nra.
Qed.

(* THE EXPLICIT CONSTANT. Every eigenvalue and norm of the scalar shadow enters only through the
   quadratic-form inequality it denotes: lo <= lambda_min(P), hi >= lambda_max(P), g >= the gap's
   top eigenvalue (at most ||R_K|| ||dK||^2), and the contraction (1 - theta/2)^2 of the step above.
   Then the summed gain gap -- EXACTLY the regret-to-go, by cost_telescopes -- is at most
   2 (hi/lo) g |x0|^2 / theta, at every horizon. *)
Theorem summed_gain_gap_explicit_bound (Khat : 'M[R]_(m, n)) (xs : nat -> 'cV[R]_n)
    (lo hi g theta : R) :
  0 < lo -> 0 <= g -> 0 < theta -> theta <= 1 ->
  (forall x, lo * ip x x <= qf P x) ->
  (forall x, qf P x <= hi * ip x x) ->
  (forall x, qf (gain_gap A B Rw P Khat) x <= g * ip x x) ->
  (forall x, qf P ((A - B *m Khat) *m x) <= (1 - theta / 2) ^+ 2 * qf P x) ->
  (forall t, xs t.+1 = (A - B *m Khat) *m xs t) ->
  forall T, \sum_(t < T) qf (gain_gap A B Rw P Khat) (xs t)
            <= 2 * (hi / lo) * g * ip (xs 0) (xs 0) / theta.
Proof.
move=> hlo hg hth0 hth1 hlow hhigh hgap hcon hstep T.
have hlo0 : lo != 0 by rewrite gt_eqF.
have hth : theta != 0 by rewrite gt_eqF.
set mu := (1 - theta / 2) ^+ 2.
have hmu0 : 0 <= mu by exact: sqr_ge0.
have hmu1 : mu < 1.
  rewrite /mu -(expr1n _ 2) ltr_sqr ?nnegrE; lra.
have hdecay := lyapunov_decay hmu0 hcon hstep.
have hx0 : qf P (xs 0) <= hi * ip (xs 0) (xs 0) := hhigh (xs 0).
have hxx0 : 0 <= ip (xs 0) (xs 0).
  by rewrite ipE; apply: sumr_ge0 => i _; rewrite -expr2 sqr_ge0.
(* each term: gap <= g |x_t|^2 <= (g/lo) x_t'Px_t <= (g/lo) mu^t x0'Px0 *)
have hterm t : qf (gain_gap A B Rw P Khat) (xs t) <= g / lo * qf P (xs 0) * mu ^+ t.
  apply: le_trans (hgap (xs t)) _.
  have hxt : ip (xs t) (xs t) <= qf P (xs t) / lo by rewrite ler_pdivlMr // mulrC hlow.
  have hxt' : ip (xs t) (xs t) <= mu ^+ t * qf P (xs 0) / lo.
    apply: le_trans hxt _; rewrite ler_pM2r ?invr_gt0 //; exact: hdecay.
  apply: le_trans (ler_wpM2l hg hxt') _.
  by rewrite le_eqVlt; apply/orP; left; apply/eqP; field.
have hsum : \sum_(t < T) qf (gain_gap A B Rw P Khat) (xs t)
            <= \sum_(t < T) (g / lo * qf P (xs 0) * mu ^+ t).
  by apply: ler_sum => t _; exact: hterm.
apply: le_trans hsum _.
rewrite -mulr_sumr.
have hP0 : 0 <= qf P (xs 0) := hP (xs 0).
have hgl : 0 <= g / lo * qf P (xs 0) by rewrite !mulr_ge0 ?invr_ge0 // ltW.
apply: le_trans (ler_wpM2l hgl (geo_partial_le T hmu0 hmu1)) _.
apply: le_trans (ler_wpM2l hgl (geo_relaxation hth0 hth1)) _.
have hq : g / lo * qf P (xs 0) <= g / lo * (hi * ip (xs 0) (xs 0)).
  by apply: ler_wpM2l; rewrite ?mulr_ge0 ?invr_ge0 // ltW.
have h2th : 0 <= 2 / theta by rewrite divr_ge0 // ltW.
apply: le_trans (ler_wpM2r h2th hq) _.
by rewrite le_eqVlt; apply/orP; left; apply/eqP; field.
Qed.

(* THE THEOREM, composed with the cited Lipschitz hypothesis |dK| <= L_K |dB| (Konstantinov et al.
   1993; Sun 1998), entering as g <= ||R_K|| (L_K d)^2. The finite-horizon regret-to-go of the
   certainty-equivalent gain is at most C d^2 with the scalar shadow's constant
   C = 2 kappa_P ||R_K|| L_K^2 |x0|^2 / theta, at EVERY horizon T. *)
Theorem ce_regret_explicit_bound_matrix (Khat : 'M[R]_(m, n)) (xs : nat -> 'cV[R]_n)
    (lo hi rkn lk d theta : R) :
  0 < lo -> 0 <= rkn -> 0 < theta -> theta <= 1 ->
  (forall x, lo * ip x x <= qf P x) ->
  (forall x, qf P x <= hi * ip x x) ->
  (forall x, qf (gain_gap A B Rw P Khat) x <= rkn * (lk * d) ^+ 2 * ip x x) ->
  (forall x, qf P ((A - B *m Khat) *m x) <= (1 - theta / 2) ^+ 2 * qf P x) ->
  (forall t, xs t.+1 = (A - B *m Khat) *m xs t) ->
  forall T, \sum_(t < T) qf (Q + Khat^T *m Rw *m Khat) (xs t) + qf P (xs T) - qf P (xs 0)
            <= 2 * (hi / lo) * rkn * lk ^+ 2 * ip (xs 0) (xs 0) / theta * d ^+ 2.
Proof.
move=> hlo hrkn hth0 hth1 hlow hhigh hgap hcon hstep T.
have hlo0 : lo != 0 by rewrite gt_eqF.
have hth : theta != 0 by rewrite gt_eqF.
have hg : 0 <= rkn * (lk * d) ^+ 2 by rewrite mulr_ge0 ?sqr_ge0.
rewrite (cost_telescopes hPs hRs hRK hdare hstep T).
have -> : qf P (xs 0) - qf P (xs T) + \sum_(t < T) qf (gain_gap A B Rw P Khat) (xs t)
          + qf P (xs T) - qf P (xs 0) = \sum_(t < T) qf (gain_gap A B Rw P Khat) (xs t).
  by ring.
apply: le_trans (summed_gain_gap_explicit_bound hlo hg hth0 hth1 hlow hhigh hgap hcon hstep T) _.
by rewrite le_eqVlt; apply/orP; left; apply/eqP; rewrite exprMn; field.
Qed.

End Theorem44.

(* ===== (F) NO CONJUGATE POINT UNDER A POSITIVE COST, in discrete time (Result 45's PD side) ===== *)
Section RiccatiRecursion.
Variable R : realFieldType.
Variables n m : nat.
Variables (A : 'M[R]_n) (B : 'M[R]_(n, m)) (Q : 'M[R]_n) (Rw : 'M[R]_m).
Hypotheses (hQs : Q^T = Q) (hRs : Rw^T = Rw) (hQ : psd Q) (hR : pd Rw).

Lemma pd_psd k (M : 'M[R]_k) : pd M -> psd M.
Proof.
move=> hM v; have [->|hv] := eqVneq v 0; last exact: ltW (hM v hv).
by rewrite /qf mulmx0 ip0r.
Qed.

Lemma sym_congruence k l (E : 'M[R]_(k, l)) (M : 'M[R]_k) :
  M^T = M -> (E^T *m M *m E)^T = E^T *m M *m E.
Proof. by move=> h; rewrite trmx_mul trmx_mul trmxK h mulmxA. Qed.

Lemma stage_sym (P : 'M[R]_n) (K' : 'M[R]_(m, n)) :
  P^T = P -> (stage A B Q Rw P K')^T = stage A B Q Rw P K'.
Proof.
move=> hP; rewrite /stage.
have eT (X Y Z : 'M[R]_n) : (X + Y + Z)^T = X^T + Y^T + Z^T by rewrite !linearD.
by rewrite eT hQs (sym_congruence K' hRs) (sym_congruence (A - B *m K') hP).
Qed.

(* The Riccati map IS the stage of the optimal gain: completing the square at K' = K. *)
Lemma riccati_is_the_optimal_stage (P : 'M[R]_n) :
  P^T = P -> RK B Rw P \in unitmx -> riccati A B Q Rw P = stage A B Q Rw P (Kopt A B Rw P).
Proof.
move=> hP hRK; rewrite (completing_the_square A Q hP hRs hRK) /gain_gap subrr trmx0.
by rewrite mul0mx mulmx0 addr0.
Qed.

Lemma riccati_psd (P : 'M[R]_n) :
  P^T = P -> psd P -> RK B Rw P \in unitmx -> psd (riccati A B Q Rw P).
Proof.
move=> hPs hP hRK v; rewrite riccati_is_the_optimal_stage // /stage !qfD !qf_congruence.
have hRp := pd_psd hR.
by apply: addr_ge0; [apply: addr_ge0; [exact: hQ | exact: hRp] | exact: hP].
Qed.

(* For Q >= 0 and R > 0 the recursion P_{k+1} = Ric(P_k) never breaks down: every iterate stays
   symmetric and PSD, and every R + B'P_kB stays positive definite, hence invertible. A conjugate
   point needs an indefinite cost. *)
Theorem no_conjugate_point_with_a_psd_cost (P0 : 'M[R]_n) :
  P0^T = P0 -> psd P0 ->
  forall k, (iter k (riccati A B Q Rw) P0)^T = iter k (riccati A B Q Rw) P0
            /\ psd (iter k (riccati A B Q Rw) P0)
            /\ pd (RK B Rw (iter k (riccati A B Q Rw) P0)).
Proof.
move=> hP0s hP0; elim=> [|k [hs [hp hpd]]] /=.
  by split; [|split; [|exact: (rk_pd B hP0 hR)]].
have hRK := pd_unitmx hpd.
have hs' : (riccati A B Q Rw (iter k (riccati A B Q Rw) P0))^T
           = riccati A B Q Rw (iter k (riccati A B Q Rw) P0).
  by rewrite riccati_is_the_optimal_stage // stage_sym.
have hp' := riccati_psd hs hp hRK.
by split; [|split; [|exact: (rk_pd B hp' hR)]].
Qed.

End RiccatiRecursion.
