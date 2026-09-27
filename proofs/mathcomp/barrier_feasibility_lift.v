(* Rocq + MathComp: THE MULTIVARIATE LIFT OF RESULT 40, and the discrete-time core of forward
   invariance.

   proofs/barrier_feasibility.v proves the scalar algebra of the robust barrier margin
   a + g u - d |u|. This file proves the vector statements chc.barrier implements -- a Euclidean
   action ball, a channel w = B^T grad h, an identification radius on B -- over any real closed
   field (rcfType; the Euclidean norm needs a square root):

   - worst_case_over_operator_ball / worst_case_over_frobenius_ball: over every effect matrix
     B^ + E with ||E||_op <= Delta (or ||E||_F <= Delta) the barrier derivative is at least
     a + w.u - Delta ||grad h|| ||u||, and a rank-one E in the ball attains it. The matrix
     uncertainty set reduces EXACTLY to the channel ball of radius d = Delta ||grad h||.
   - margin_upper_bound_mv / margin_attained_mv: the best guaranteed margin over ||u|| <= U is
     a + max(0, ||w|| - d) U, and it is attained -- Result 40 (a) with |g| replaced by ||w||.
   - certified_iff_below_threshold: certifiable iff d <= ||w|| - (-alpha h - a)/U, the sharp
     threshold of Result 40 (b). zero_action_optimal_mv, certified_without_deficit,
     no_certificate_when_deficit_exceeds_authority_mv and margin_loss_saturates_mv lift the rest.
   - discrete_forward_invariance / robust_filter_keeps_the_true_plant_safe: for an AFFINE barrier
     h(x) = c + g.x under the Euler step x' = x + dt v with 0 <= alpha dt <= 1, the pointwise
     condition at every visited state keeps h(x_k) >= (1 - alpha dt)^k h(x_0) >= 0; and a filter
     that certifies the robust margin on the ESTIMATED channel keeps the TRUE plant safe whenever
     the true channel lies in the ball.

   Honest scope. Continuous-time forward invariance of {h >= 0} is Nagumo's theorem (M. Nagumo,
   Proc. Phys.-Math. Soc. Japan 24 (1942) 551-559; H. Brezis, Comm. Pure Appl. Math. 23 (1970)
   261-263) and stays cited: it needs the flow of an ODE, which is analysis. The discrete statement
   above is its algebraic core for the step the library's benchmark takes. A NONLINEAR barrier along
   an Euler step would need a Taylor-remainder bound on h, which is also analysis, and is not
   claimed. *)

Set Warnings "-notation-overridden,-ambiguous-paths".
From mathcomp Require Import boot order algebra.
Set Implicit Arguments. Unset Strict Implicit. Unset Printing Implicit Defensive.
Import GRing.Theory Num.Theory Order.Theory.
Local Open Scope ring_scope.

Section Vectors.
Variable R : rcfType.

Definition dot {k} (u v : 'rV[R]_k) : R := \sum_i u 0 i * v 0 i.
Definition norm {k} (u : 'rV[R]_k) : R := Num.sqrt (dot u u).

Lemma dotC k (u v : 'rV[R]_k) : dot u v = dot v u.
Proof. by apply: eq_bigr => i _; rewrite mulrC. Qed.

Lemma dotDl k (u v w : 'rV[R]_k) : dot (u + v) w = dot u w + dot v w.
Proof. by rewrite /dot -big_split; apply: eq_bigr => i _; rewrite mxE mulrDl. Qed.

Lemma dotZl k (a : R) (u v : 'rV[R]_k) : dot (a *: u) v = a * dot u v.
Proof. by rewrite /dot mulr_sumr; apply: eq_bigr => i _; rewrite mxE mulrA. Qed.

Lemma dotNl k (u v : 'rV[R]_k) : dot (- u) v = - dot u v.
Proof. by rewrite -scaleN1r dotZl mulN1r. Qed.

Lemma dotDr k (u v w : 'rV[R]_k) : dot u (v + w) = dot u v + dot u w.
Proof. by rewrite dotC dotDl !(dotC u). Qed.

Lemma dotZr k (a : R) (u v : 'rV[R]_k) : dot u (a *: v) = a * dot u v.
Proof. by rewrite dotC dotZl dotC. Qed.

Lemma dotNr k (u v : 'rV[R]_k) : dot u (- v) = - dot u v.
Proof. by rewrite dotC dotNl dotC. Qed.

Lemma dot0l k (v : 'rV[R]_k) : dot 0 v = 0.
Proof. by rewrite /dot big1 // => i _; rewrite mxE mul0r. Qed.

Lemma dot0r k (v : 'rV[R]_k) : dot v 0 = 0.
Proof. by rewrite dotC dot0l. Qed.

Lemma dot_ge0 k (u : 'rV[R]_k) : 0 <= dot u u.
Proof. by apply: sumr_ge0 => i _; rewrite -expr2 sqr_ge0. Qed.

Lemma dot_eq0 k (u : 'rV[R]_k) : dot u u = 0 -> u = 0.
Proof.
move=> h; apply/rowP => i; rewrite mxE.
have ge0 : forall j : 'I_k, true -> 0 <= u 0 j * u 0 j.
  by move=> j _; rewrite -expr2 sqr_ge0.
by have /eqP := psumr_eq0P ge0 h (i := i) isT; rewrite -expr2 sqrf_eq0 => /eqP.
Qed.

Lemma norm_ge0 k (u : 'rV[R]_k) : 0 <= norm u.
Proof. exact: sqrtr_ge0. Qed.

Lemma sqr_norm k (u : 'rV[R]_k) : norm u ^+ 2 = dot u u.
Proof. by rewrite sqr_sqrtr // dot_ge0. Qed.

Lemma norm0 k : norm (0 : 'rV[R]_k) = 0.
Proof. by rewrite /norm dot0l sqrtr0. Qed.

Lemma norm_eq0 k (u : 'rV[R]_k) : norm u = 0 -> u = 0.
Proof. by move=> h; apply: dot_eq0; rewrite -sqr_norm h expr0n. Qed.

Lemma normZ k (a : R) (u : 'rV[R]_k) : norm (a *: u) = `|a| * norm u.
Proof.
rewrite /norm dotZl dotZr mulrA -expr2 sqrtrM ?sqr_ge0 //.
by rewrite sqrtr_sqr.
Qed.

(* Cauchy-Schwarz, by the discriminant: no square root is needed for the squared form. *)
Lemma cauchy_schwarz_sq k (u v : 'rV[R]_k) : dot u v ^+ 2 <= dot u u * dot v v.
Proof.
have [hv|hv] := eqVneq (dot v v) 0.
  by rewrite (dot_eq0 hv) !dot0r mulr0 expr2 mul0r.
have vpos : 0 < dot v v by rewrite lt0r hv dot_ge0.
set t := dot u v / dot v v.
have tc : t * dot v v = dot u v by rewrite /t divfK.
have h := dot_ge0 (u - t *: v).
move: h; rewrite dotDl !dotDr !dotNl !dotNr !dotZl !dotZr (dotC v u).
nra.
Qed.

Lemma cauchy_schwarz k (u v : 'rV[R]_k) : `|dot u v| <= norm u * norm v.
Proof.
rewrite -ler_sqr ?nnegrE ?normr_ge0 ?mulr_ge0 ?norm_ge0 //.
by rewrite real_normK ?num_real // exprMn !sqr_norm cauchy_schwarz_sq.
Qed.

Lemma dot_le_norm k (u v : 'rV[R]_k) : dot u v <= norm u * norm v.
Proof. exact: le_trans (ler_norm _) (cauchy_schwarz u v). Qed.

Lemma dot_ge_norm k (u v : 'rV[R]_k) : - (norm u * norm v) <= dot u v.
Proof. by have := cauchy_schwarz u v; rewrite ler_norml => /andP[]. Qed.

Lemma norm_gt0 k (u : 'rV[R]_k) : u != 0 -> 0 < norm u.
Proof.
move=> hu; rewrite lt0r norm_ge0 andbT; apply/eqP => h.
by move: hu; rewrite (norm_eq0 h) eqxx.
Qed.

(* A row vector times a transposed row vector is the 1x1 matrix of their dot product. *)
Lemma mul_row_tr k (u v : 'rV[R]_k) : u *m v^T = (dot u v)%:M.
Proof.
apply/matrixP => i j; rewrite !ord1 !mxE eqxx mulr1n.
by apply: eq_bigr => l _; rewrite mxE.
Qed.

End Vectors.

Section Barrier.
Variable R : rcfType.
Variable m : nat.

(* The guaranteed barrier derivative at action u: drift a, estimated channel w = B^T grad h, and
   an adversary who may move the channel anywhere in the Euclidean ball of radius d. *)
Definition robust_margin (a : R) (w : 'rV[R]_m) (d : R) (u : 'rV[R]_m) : R :=
  a + dot w u - d * norm u.

(* The closed form of Result 40 (a), with |g| replaced by ||B^T grad h||. *)
Definition best_margin (a : R) (w : 'rV[R]_m) (d U : R) : R :=
  a + Num.max 0 (norm w - d) * U.

(* ---- The adversary: robust_margin IS the worst case over the channel ball. ---- *)

Lemma margin_below_every_channel (a d : R) (w u delta : 'rV[R]_m) :
  norm delta <= d -> robust_margin a w d u <= a + dot (w + delta) u.
Proof.
move=> hd; rewrite /robust_margin dotDl.
have h1 := dot_ge_norm delta u; have h2 := norm_ge0 u.
nra.
Qed.

Lemma margin_attained_by_a_channel (a d : R) (w u : 'rV[R]_m) :
  0 <= d -> exists delta : 'rV[R]_m,
    norm delta <= d /\ a + dot (w + delta) u = robust_margin a w d u.
Proof.
move=> hd; rewrite /robust_margin.
have [->|hu] := eqVneq u 0.
  by exists 0; split; [rewrite norm0 | rewrite !dot0r norm0 mulr0 subr0].
have nu := norm_gt0 hu.
exists (- (d / norm u) *: u); split.
  by rewrite normZ normrN ger0_norm ?divr_ge0 ?norm_ge0 // divfK // gt_eqF.
have hnz : norm u != 0 by rewrite gt_eqF.
by rewrite dotDl dotZl -sqr_norm; field.
Qed.

(* ---- The action: the best guaranteed margin, as a maximum that is attained. ---- *)

Lemma margin_upper_bound_mv (a d U : R) (w u : 'rV[R]_m) :
  0 <= U -> norm u <= U -> robust_margin a w d u <= best_margin a w d U.
Proof.
move=> hU huU; rewrite /robust_margin /best_margin.
have h1 := dot_le_norm w u; have h2 := norm_ge0 u.
have hM0 : 0 <= Num.max 0 (norm w - d) by rewrite le_max lexx.
have hM1 : norm w - d <= Num.max 0 (norm w - d) by rewrite le_max lexx orbT.
nra.
Qed.

Lemma margin_attained_mv (a d U : R) (w : 'rV[R]_m) :
  0 <= d -> 0 <= U ->
  exists u : 'rV[R]_m, norm u <= U /\ robust_margin a w d u = best_margin a w d U.
Proof.
move=> hd hU; rewrite /robust_margin /best_margin.
have [hdw|hwd] := ltP d (norm w).
  have hw : w != 0 by apply/eqP => hw0; move: hdw; rewrite hw0 norm0; have := norm_ge0 w; lra.
  have nw := norm_gt0 hw.
  exists ((U / norm w) *: w); split.
    by rewrite normZ ger0_norm ?divr_ge0 ?norm_ge0 // divfK // gt_eqF.
  have hmax : Num.max 0 (norm w - d) = norm w - d by apply: max_r; lra.
  rewrite hmax normZ ger0_norm ?divr_ge0 ?norm_ge0 // divfK ?gt_eqF //.
  have hnz : norm w != 0 by rewrite gt_eqF.
  by rewrite dotZr -sqr_norm; field.
have hmax : Num.max 0 (norm w - d) = 0 by apply: max_l; lra.
exists 0; split; first by rewrite norm0.
by rewrite hmax dot0r norm0 mul0r mulr0 subr0 addr0.
Qed.

(* ---- The zero-action rule: an unidentified channel direction makes every action useless. ---- *)

Lemma zero_action_optimal_mv (a d : R) (w u : 'rV[R]_m) :
  norm w <= d -> robust_margin a w d u <= robust_margin a w d 0.
Proof.
move=> hwd; rewrite /robust_margin dot0r norm0 mulr0 subr0 addr0.
have h1 := dot_le_norm w u; have h2 := norm_ge0 u.
nra.
Qed.

(* ---- Certification, and the sharp threshold d* = ||B^T grad h|| - D/U. ---- *)

Definition certified (a : R) (w : 'rV[R]_m) (d U alpha_h : R) : Prop :=
  exists u : 'rV[R]_m, norm u <= U /\ - alpha_h <= robust_margin a w d u.

Lemma certified_iff_best_margin (a d U alpha_h : R) (w : 'rV[R]_m) :
  0 <= d -> 0 <= U -> certified a w d U alpha_h <-> - alpha_h <= best_margin a w d U.
Proof.
move=> hd hU; split.
  by case=> u [hu hcert]; exact: le_trans hcert (margin_upper_bound_mv a d w hU hu).
move=> hbest; have [u [hu heq]] := margin_attained_mv a w hd hU.
by exists u; rewrite heq.
Qed.

Theorem certified_iff_below_threshold (a d U alpha_h : R) (w : 'rV[R]_m) :
  0 <= d -> 0 < U -> 0 < - alpha_h - a ->
  certified a w d U alpha_h <-> d <= norm w - (- alpha_h - a) / U.
Proof.
move=> hd hU hD.
apply: (iff_trans (certified_iff_best_margin a alpha_h w hd (ltW hU))).
rewrite /best_margin.
have hq : 0 < (- alpha_h - a) / U by rewrite divr_gt0.
have hqU : (- alpha_h - a) / U * U = - alpha_h - a by rewrite divfK // gt_eqF.
have [hdw|hwd] := leP d (norm w).
  have hmax : Num.max 0 (norm w - d) = norm w - d by apply: max_r; lra.
  rewrite hmax; split=> h; nra.
have hmax : Num.max 0 (norm w - d) = 0 by apply: max_l; lra.
rewrite hmax; split=> h; nra.
Qed.

(* The first regime: a drift that already satisfies the barrier needs no action and no radius. *)
Lemma certified_without_deficit (a d U alpha_h : R) (w : 'rV[R]_m) :
  0 <= U -> - alpha_h - a <= 0 -> certified a w d U alpha_h.
Proof.
move=> hU hD; exists 0; rewrite norm0 hU; split=> //.
rewrite /robust_margin dot0r norm0 mulr0 subr0 addr0; lra.
Qed.

(* The second regime: a deficit beyond full authority on a PERFECT channel is uncertifiable at
   every radius, so "the largest admissible radius" does not exist. *)
Lemma no_certificate_when_deficit_exceeds_authority_mv (a d U alpha_h : R) (w : 'rV[R]_m) :
  0 <= d -> 0 <= U -> U * norm w < - alpha_h - a -> ~ certified a w d U alpha_h.
Proof.
move=> hd hU hbig hc.
have := proj1 (certified_iff_best_margin a alpha_h w hd hU) hc; rewrite /best_margin => h.
have hM : Num.max 0 (norm w - d) <= norm w.
  by rewrite ge_max norm_ge0 /=; lra.
nra.
Qed.

(* The margin lost to the radius, against a perfectly identified channel: U * min(d, ||w||).
   First order in d while authority lasts, then flat -- the multivariate form of the saturation. *)
Lemma margin_loss_saturates_mv (a d U : R) (w : 'rV[R]_m) :
  0 <= d ->
  (d <= norm w -> best_margin a w 0 U - best_margin a w d U = U * d) /\
  (norm w <= d -> best_margin a w 0 U - best_margin a w d U = U * norm w).
Proof.
move=> hd; rewrite /best_margin subr0.
have hmax0 : Num.max 0 (norm w) = norm w by apply: max_r; exact: norm_ge0.
split=> h.
  have hmax : Num.max 0 (norm w - d) = norm w - d by apply: max_r; lra.
  by rewrite hmax0 hmax; ring.
have hmax : Num.max 0 (norm w - d) = 0 by apply: max_l; lra.
by rewrite hmax0 hmax; ring.
Qed.

End Barrier.

Section ChannelBall.
Variable R : rcfType.
Variables n m : nat.

(* The effect matrix is known up to an operator-norm ball: B' = B_hat + E with ||v E|| <= Delta ||v||
   for every row v. With g = (grad h)^T the estimated channel is w = g B_hat and the true one is
   w + g E. *)
Definition op_bounded (E : 'M[R]_(n, m)) (Delta : R) : Prop :=
  forall v : 'rV[R]_n, norm (v *m E) <= Delta * norm v.

Lemma channel_moves_within_radius (E : 'M[R]_(n, m)) (Delta : R) (g : 'rV[R]_n) :
  op_bounded E Delta -> norm (g *m E) <= Delta * norm g.
Proof. by move=> hE; exact: hE. Qed.

(* The rank-one perturbation that moves the channel by exactly delta. *)
Definition rank_one (g : 'rV[R]_n) (delta : 'rV[R]_m) : 'M[R]_(n, m) :=
  (dot g g)^-1 *: (g^T *m delta).

Lemma rank_one_channel (g : 'rV[R]_n) (delta : 'rV[R]_m) :
  g != 0 -> g *m rank_one g delta = delta.
Proof.
move=> hg; have hgg : dot g g != 0 by rewrite -sqr_norm expf_neq0 // gt_eqF // norm_gt0.
rewrite /rank_one -scalemxAr mulmxA mul_row_tr mul_scalar_mx scalerA mulVf //.
by rewrite scale1r.
Qed.

Lemma rank_one_bounded (g : 'rV[R]_n) (delta : 'rV[R]_m) (Delta : R) :
  g != 0 -> norm delta <= Delta * norm g -> op_bounded (rank_one g delta) Delta.
Proof.
move=> hg hd v; have ng := norm_gt0 hg.
rewrite /rank_one -scalemxAr mulmxA mul_row_tr mul_scalar_mx scalerA normZ normrM.
rewrite normfV -sqr_norm ger0_norm ?sqr_ge0 //.
have hcs := cauchy_schwarz v g; have hv := norm_ge0 v; have hdl := norm_ge0 delta.
have hdv := normr_ge0 (dot v g).
rewrite -mulrA mulrC ler_pdivrMr ?exprn_gt0 //.
have h1 : `|dot v g| * norm delta <= norm v * norm g * norm delta by exact: ler_wpM2r.
have h2 : norm v * norm g * norm delta <= norm v * norm g * (Delta * norm g).
  have hvg : 0 <= norm v * norm g by rewrite mulr_ge0 // ltW.
  nra.
by apply: le_trans h1 (le_trans h2 _); rewrite le_eqVlt; apply/orP; left; apply/eqP; ring.
Qed.

(* THE MULTIVARIATE LIFT, from the matrix uncertainty set to the closed form. Over every effect
   matrix in the operator-norm ball of radius Delta the barrier derivative is at least
   robust_margin with d = Delta * ||grad h||, and some matrix in the ball attains it: so the
   worst case over the ball IS robust_margin, and best_margin is its max over the action ball. *)
Theorem worst_case_over_operator_ball (a Delta : R) (g : 'rV[R]_n) (w u : 'rV[R]_m) :
  (forall E, op_bounded E Delta ->
     robust_margin a w (Delta * norm g) u <= a + dot (w + g *m E) u) /\
  (0 <= Delta -> exists E, op_bounded E Delta /\
     a + dot (w + g *m E) u = robust_margin a w (Delta * norm g) u).
Proof.
split=> [E hE | hD]; first exact: margin_below_every_channel (hE g).
have [->|hg] := eqVneq g 0.
  exists 0; split.
    by move=> v; rewrite mulmx0 norm0 mulr_ge0 ?norm_ge0.
  by rewrite mulmx0 addr0 /robust_margin norm0 mulr0 mul0r subr0.
have hd : 0 <= Delta * norm g by rewrite mulr_ge0 ?norm_ge0.
have [delta [hdl heq]] := margin_attained_by_a_channel a w u hd.
exists (rank_one g delta); split; first exact: rank_one_bounded.
by rewrite rank_one_channel.
Qed.

(* The same holds for a Frobenius ball: it sits inside the operator ball, and the rank-one
   witness has Frobenius norm exactly ||delta|| / ||grad h||. *)
Definition frob (E : 'M[R]_(n, m)) : R := Num.sqrt (\sum_j dot (col j E)^T (col j E)^T).

Lemma frob2_ge0 (E : 'M[R]_(n, m)) : 0 <= \sum_j dot (col j E)^T (col j E)^T.
Proof. by apply: sumr_ge0 => j _; exact: dot_ge0. Qed.

Lemma entry_of_product (v : 'rV[R]_n) (E : 'M[R]_(n, m)) (j : 'I_m) :
  (v *m E) 0 j = dot v (col j E)^T.
Proof. by rewrite !mxE; apply: eq_bigr => i _; rewrite !mxE. Qed.

Lemma frob_op_bounded (E : 'M[R]_(n, m)) : op_bounded E (frob E).
Proof.
move=> v; rewrite /norm /frob -sqrtrM ?frob2_ge0 // ler_sqrt ?mulr_ge0 ?frob2_ge0 ?dot_ge0 //.
rewrite mulrC mulr_sumr /dot; apply: ler_sum => j _.
rewrite -expr2 entry_of_product; exact: cauchy_schwarz_sq.
Qed.

Lemma rank_one_column (g : 'rV[R]_n) (delta : 'rV[R]_m) (j : 'I_m) :
  (col j (rank_one g delta))^T = ((dot g g)^-1 * delta 0 j) *: g.
Proof.
by apply/rowP => i; rewrite !mxE big_ord1 !mxE; ring.
Qed.

Lemma rank_one_frob (g : 'rV[R]_n) (delta : 'rV[R]_m) :
  g != 0 -> frob (rank_one g delta) = norm delta / norm g.
Proof.
move=> hg; have ng := norm_gt0 hg.
have hgg : dot g g != 0 by rewrite -sqr_norm expf_neq0 // gt_eqF.
have hsum : \sum_j dot (col j (rank_one g delta))^T (col j (rank_one g delta))^T
            = \sum_j (dot g g)^-1 * (delta 0 j * delta 0 j).
  by apply: eq_bigr => j _; rewrite rank_one_column dotZl dotZr; field.
rewrite /frob hsum -mulr_sumr -/(dot delta delta).
rewrite sqrtrM ?invr_ge0 ?dot_ge0 // sqrtrV ?dot_ge0 //.
by rewrite /norm mulrC.
Qed.

Theorem worst_case_over_frobenius_ball (a Delta : R) (g : 'rV[R]_n) (w u : 'rV[R]_m) :
  (forall E, frob E <= Delta ->
     robust_margin a w (Delta * norm g) u <= a + dot (w + g *m E) u) /\
  (0 <= Delta -> exists E, frob E <= Delta /\
     a + dot (w + g *m E) u = robust_margin a w (Delta * norm g) u).
Proof.
split=> [E hE | hD].
  apply: margin_below_every_channel.
  apply: le_trans (frob_op_bounded E g) _.
  by apply: ler_wpM2r; [exact: norm_ge0 | exact: hE].
have [->|hg] := eqVneq g 0.
  have hz : \sum_j dot (col j (0 : 'M[R]_(n, m)))^T (col j (0 : 'M[R]_(n, m)))^T = 0.
    apply: big1 => j _.
    have -> : (col j (0 : 'M[R]_(n, m)))^T = 0 by apply/rowP => i; rewrite !mxE.
    exact: dot0l.
  exists 0; split; first by rewrite /frob hz sqrtr0.
  by rewrite mulmx0 addr0 /robust_margin norm0 mulr0 mul0r subr0.
have hd : 0 <= Delta * norm g by rewrite mulr_ge0 ?norm_ge0.
have [delta [hdl heq]] := margin_attained_by_a_channel a w u hd.
exists (rank_one g delta); split.
  rewrite rank_one_frob // ler_pdivrMr ?norm_gt0 //.
by rewrite rank_one_channel.
Qed.

End ChannelBall.

Section DiscreteInvariance.
Variable R : rcfType.
Variables n m : nat.

(* An affine barrier h(x) = c + g.x -- a half-space safe set, the shape of Result 40 (d)'s
   benchmark (h = x_limit - x). *)
Definition affine_barrier (c : R) (g x : 'rV[R]_n) : R := c + dot g x.

Lemma euler_step_on_affine_barrier (c dt : R) (g x v : 'rV[R]_n) :
  affine_barrier c g (x + dt *: v) = affine_barrier c g x + dt * dot g v.
Proof. by rewrite /affine_barrier dotDr dotZr addrA. Qed.

Lemma one_step_decay (c dt alpha : R) (g x v : 'rV[R]_n) :
  0 <= dt -> - (alpha * affine_barrier c g x) <= dot g v ->
  (1 - alpha * dt) * affine_barrier c g x <= affine_barrier c g (x + dt *: v).
Proof. by move=> hdt hv; rewrite euler_step_on_affine_barrier; nra. Qed.

(* DISCRETE-TIME FORWARD INVARIANCE, the algebraic core of Nagumo/Brezis for the Euler step:
   the barrier condition at every visited state keeps h above a geometric floor, hence >= 0. *)
Theorem discrete_forward_invariance (c dt alpha : R) (g : 'rV[R]_n) (xs vs : nat -> 'rV[R]_n) :
  0 <= dt -> 0 <= alpha * dt -> alpha * dt <= 1 ->
  (forall k, xs k.+1 = xs k + dt *: vs k) ->
  (forall k, - (alpha * affine_barrier c g (xs k)) <= dot g (vs k)) ->
  0 <= affine_barrier c g (xs 0) ->
  forall k, (1 - alpha * dt) ^+ k * affine_barrier c g (xs 0) <= affine_barrier c g (xs k)
            /\ 0 <= affine_barrier c g (xs k).
Proof.
move=> hdt ha0 ha1 hstep hcbf h0.
have hq : 0 <= 1 - alpha * dt by rewrite subr_ge0.
have floor k : (1 - alpha * dt) ^+ k * affine_barrier c g (xs 0) <= affine_barrier c g (xs k).
  elim: k => [|k IH]; first by rewrite expr0 mul1r.
  rewrite hstep exprS -mulrA.
  apply: le_trans (one_step_decay hdt (hcbf k)).
  exact: ler_wpM2l.
move=> k; split; first exact: floor.
by apply: le_trans (floor k); rewrite mulr_ge0 ?exprn_ge0.
Qed.

(* The closed loop Result 40 (d) measures: a filter that certifies robust_margin on the ESTIMATED
   channel keeps the TRUE plant safe, whenever the true channel lies in the identified ball. *)
Theorem robust_filter_keeps_the_true_plant_safe (c d dt alpha : R) (g : 'rV[R]_n)
    (xs vs : nat -> 'rV[R]_n) (a : nat -> R) (w delta us : nat -> 'rV[R]_m) :
  0 <= dt -> 0 <= alpha * dt -> alpha * dt <= 1 ->
  (forall k, xs k.+1 = xs k + dt *: vs k) ->
  (forall k, dot g (vs k) = a k + dot (w k + delta k) (us k)) ->
  (forall k, norm (delta k) <= d) ->
  (forall k, - (alpha * affine_barrier c g (xs k)) <= robust_margin (a k) (w k) d (us k)) ->
  0 <= affine_barrier c g (xs 0) ->
  forall k, 0 <= affine_barrier c g (xs k).
Proof.
move=> hdt ha0 ha1 hstep hv hdl hfilter h0 k.
have hcbf j : - (alpha * affine_barrier c g (xs j)) <= dot g (vs j).
  by rewrite hv; apply: le_trans (hfilter j) (margin_below_every_channel _ _ _ (hdl j)).
exact: (discrete_forward_invariance hdt ha0 ha1 hstep hcbf h0 k).2.
Qed.

End DiscreteInvariance.
