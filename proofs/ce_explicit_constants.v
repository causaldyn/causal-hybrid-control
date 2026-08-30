(* Rocq (CONTRIBUTION 1): the finite-sample end-to-end half -- the EXACT gain-gap identity, the
   explicit ball on which a certainty-equivalent controller still stabilises the TRUE plant, and the
   explicit constant in front of ||dB||^2.

   What this replaces. Every C1 statement so far took "regret <= cc * ||dB||^2" as a HYPOTHESIS
   (c2_end_to_end.v A5 quantifies over an arbitrary 0 <= cc), citing Mania-Tu-Recht's LOCAL quadratic
   bound. Nothing computed the ball on which it holds, and nothing computed cc. Both are here, and
   the route is different: STEP 1 of validation/ce_explicit_constants.mac shows the Lyapunov
   increment is an EXACT perfect square in the gain error,

       Q + K'^T R K' + (A - B K')^T P (A - B K') - P  =  (K' - K)^T R_K (K' - K),   R_K := R + B^T P B,

   with no smallness assumed -- so "quadratic in the gain error" stops being a local approximation
   and the Mania-Tu-Recht citation leaves the regret half entirely (it stays only as a comparison).

   Honest scope, as in c2_end_to_end.v and van_trees.v: Stdlib has no matrices, so the scalar shadow
   of each matrix step is proved here and the matrix statements are verified numerically in
   validation/ce_explicit_constants.mac (STEP 7-8, DARE residual 2.0e-40 at 60 digits) and in the
   ce_explicit_constant_certificate sweep. The ONE remaining citation on the control side is uniform
   validity of the gain map's Lipschitz constant on the ball (Konstantinov-Petkov-Christov-Angelova
   1993; Sun 1998), taken as a hypothesis (l_k) and certified numerically on the ball, exactly as the
   clustered CLT stays cited in the C2 line.

   Parameterisation. The proofs are stated in s with s^2 = 1 - eta rather than in sqrt(1 - eta), so
   every goal stays polynomial and nra stays in scope. theta := 1 - s. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* ---------- (A) The exact gain-gap identity: scalar shadow of the matrix statement. ---------- *)

(* q eliminated through the scalar DARE p = q + a^2 p r/(r + b^2 p), i.e.
   q = p - a^2 p r/(r + b^2 p); k = a b p/(r + b^2 p); R_K = r + b^2 p. *)
Definition rk (b p r : R) : R := r + b * b * p.
Definition kopt (a b p r : R) : R := a * b * p / rk b p r.
Definition qdare (a b p r : R) : R := p - a * a * p * r / rk b p r.

Theorem gain_gap_identity : forall a b p r kp,
  0 < r -> 0 <= p ->
  qdare a b p r + r * kp * kp + (a - b * kp) * (a - b * kp) * p - p
  = (kp - kopt a b p r) * (kp - kopt a b p r) * rk b p r.
Proof.
  intros a b p r kp Hr Hp.
  assert (Hrk : rk b p r <> 0).
  { unfold rk. nra. }
  unfold qdare, kopt, rk in *. field_simplify; [ | assumption | assumption ].
  reflexivity.
Qed.

(* At the optimal gain the increment is exactly 0 -- the DARE IS the k' = k case of the identity. *)
Corollary gain_gap_zero_at_optimum : forall a b p r,
  0 < r -> 0 <= p ->
  qdare a b p r + r * kopt a b p r * kopt a b p r
  + (a - b * kopt a b p r) * (a - b * kopt a b p r) * p - p = 0.
Proof.
  intros a b p r Hr Hp. rewrite gain_gap_identity by assumption.
  replace (kopt a b p r - kopt a b p r) with 0 by ring. ring.
Qed.

(* The quadratic form is positive-definite: R_K >= r > 0, so the gap is NON-NEGATIVE for every gain
   -- the optimum really is optimal, and this needs no smallness either. *)
Theorem gain_gap_nonneg : forall a b p r kp,
  0 < r -> 0 <= p ->
  0 <= qdare a b p r + r * kp * kp + (a - b * kp) * (a - b * kp) * p - p.
Proof.
  intros a b p r kp Hr Hp. rewrite gain_gap_identity by assumption.
  assert (Hrk : 0 <= rk b p r) by (unfold rk; nra).
  remember (kp - kopt a b p r) as e eqn:He.
  assert (Hsq : 0 <= e * e) by nra.
  apply Rmult_le_pos; assumption.
Qed.

Theorem rk_ge_r : forall b p r, 0 <= p -> r <= rk b p r.
Proof. intros b p r Hp. unfold rk. nra. Qed.

(* ---------- (B) The stabilising ball. ---------- *)

(* theta = 1 - s, where s = sqrt(1 - eta) is the P-metric contraction modulus of the OPTIMAL loop.
   The perturbed loop's modulus is at most s + beta_B * L_K * ||dB|| by the triangle inequality in
   the P metric; rho is exactly the radius at which that budget is half spent. *)
Definition theta (s : R) : R := 1 - s.
Definition rho (s bB lk : R) : R := theta s / (2 * bB * lk).
Definition modulus (s bB lk d : R) : R := s + bB * lk * d.

(* THE MISSING LINK: inside the ball the perturbed closed loop still contracts, with an explicit
   margin theta/2. This is the statement nothing in the C1 line previously had. *)
Theorem perturbed_loop_contraction : forall s bB lk d,
  0 <= s -> s < 1 -> 0 < bB -> 0 < lk -> 0 <= d -> d <= rho s bB lk ->
  modulus s bB lk d <= 1 - theta s / 2.
Proof.
  intros s bB lk d Hs0 Hs1 HbB Hlk Hd0 Hd.
  unfold modulus, rho, theta in *.
  assert (Hprod : bB * lk * d <= (1 - s) / 2).
  { apply Rle_trans with (bB * lk * ((1 - s) / (2 * bB * lk))).
    - apply Rmult_le_compat_l; [ nra | exact Hd ].
    - right. field. nra. }
  lra.
Qed.

Corollary perturbed_loop_stable : forall s bB lk d,
  0 <= s -> s < 1 -> 0 < bB -> 0 < lk -> 0 <= d -> d <= rho s bB lk ->
  modulus s bB lk d < 1.
Proof.
  intros s bB lk d Hs0 Hs1 HbB Hlk Hd0 Hd.
  assert (H := perturbed_loop_contraction s bB lk d Hs0 Hs1 HbB Hlk Hd0 Hd).
  unfold theta in *. lra.
Qed.

Theorem theta_pos : forall s, s < 1 -> 0 < theta s.
Proof. intros s H. unfold theta. lra. Qed.

Theorem radius_pos : forall s bB lk, s < 1 -> 0 < bB -> 0 < lk -> 0 < rho s bB lk.
Proof.
  intros s bB lk Hs HbB Hlk. unfold rho, theta.
  apply Rdiv_lt_0_compat; nra.
Qed.

(* A faster nominal loop (smaller s, i.e. larger theta) tolerates a bigger model error. *)
Theorem radius_monotone_in_margin : forall s1 s2 bB lk,
  0 < bB -> 0 < lk -> s1 <= s2 -> s2 < 1 -> rho s2 bB lk <= rho s1 bB lk.
Proof.
  intros s1 s2 bB lk HbB Hlk Hle Hs2. unfold rho, theta.
  apply Rmult_le_compat_r with (r := / (2 * bB * lk)) in Hle;
    [ | left; apply Rinv_0_lt_compat; nra ].
  unfold Rdiv. lra.
Qed.

(* A stiffer gain map shrinks the certified ball, exactly as 1/L_K. *)
Theorem radius_antitone_in_lipschitz : forall s bB l1 l2,
  s < 1 -> 0 < bB -> 0 < l1 -> l1 <= l2 -> rho s bB l2 <= rho s bB l1.
Proof.
  intros s bB l1 l2 Hs HbB Hl1 Hle. unfold rho, theta.
  apply Rmult_le_compat_l with (r := 1 - s) in Hle; [ | lra ].
  unfold Rdiv.
  apply Rmult_le_compat_l; [ lra | ].
  apply Rinv_le_contravar; nra.
Qed.

(* ---------- (C) The geometric sum, with the relaxation the constant actually uses. ---------- *)

(* Partial sums of (1 - theta/2)^(2t), bounded WITHOUT a limit argument: a self-contained induction,
   so no Coquelicot series and no tech3. *)
Fixpoint geo_sq (x : R) (n : nat) : R :=
  match n with O => 0 | S k => 1 + x * x * geo_sq x k end.

Lemma geo_sq_nonneg : forall x n, 0 <= geo_sq x n.
Proof. intros x n. induction n as [ | k IH ]; simpl; nra. Qed.

Lemma geo_sq_partial_le : forall x n,
  0 <= x -> x < 1 -> geo_sq x n <= / (1 - x * x).
Proof.
  intros x n Hx0 Hx1. induction n as [ | k IH ].
  - simpl. left. apply Rinv_0_lt_compat. nra.
  - simpl.
    assert (Hd : 0 < 1 - x * x) by nra.
    assert (Hstep : 1 + x * x * geo_sq x k <= 1 + x * x * / (1 - x * x)).
    { apply Rplus_le_compat_l. apply Rmult_le_compat_l; nra. }
    apply Rle_trans with (1 + x * x * / (1 - x * x)); [ exact Hstep | ].
    right. field. lra.
Qed.

(* The bound uses 2/theta in place of the exact 4/(theta(4-theta)). Valid because theta <= 1 < 2. *)
Theorem geo_relaxation : forall th,
  0 < th -> th <= 1 -> / (1 - (1 - th / 2) * (1 - th / 2)) <= 2 / th.
Proof.
  intros th H0 H1.
  replace (1 - (1 - th / 2) * (1 - th / 2)) with (th * (4 - th) / 4) by field.
  assert (Hd : 0 < th * (4 - th) / 4) by nra.
  apply Rmult_le_reg_l with (r := th * (4 - th) / 4); [ exact Hd | ].
  rewrite Rinv_r by lra.
  replace (th * (4 - th) / 4 * (2 / th)) with ((4 - th) / 2) by (field; lra).
  lra.
Qed.

(* ---------- (D) The explicit constant, and the composed statement. ---------- *)

Definition c_const (kappaP rkn lk x0sq th : R) : R := 2 * kappaP * rkn * lk * lk * x0sq / th.

Theorem c_const_pos : forall kappaP rkn lk x0sq th,
  0 < kappaP -> 0 < rkn -> 0 < lk -> 0 < x0sq -> 0 < th -> 0 < c_const kappaP rkn lk x0sq th.
Proof.
  intros kappaP rkn lk x0sq th Hk Hr Hl Hx Ht. unfold c_const.
  apply Rdiv_lt_0_compat; [ | exact Ht ].
  (* nra cannot see a degree-5 product; build it factor by factor. *)
  repeat apply Rmult_lt_0_compat; lra.
Qed.

(* A slower closed loop (smaller theta) costs strictly more. *)
Theorem c_const_antitone_in_margin : forall kappaP rkn lk x0sq t1 t2,
  0 < kappaP -> 0 < rkn -> 0 < lk -> 0 < x0sq -> 0 < t1 -> t1 <= t2 ->
  c_const kappaP rkn lk x0sq t2 <= c_const kappaP rkn lk x0sq t1.
Proof.
  intros kappaP rkn lk x0sq t1 t2 Hk Hr Hl Hx Ht1 Hle. unfold c_const.
  assert (Hnum : 0 <= 2 * kappaP * rkn * lk * lk * x0sq).
  { left. repeat apply Rmult_lt_0_compat; lra. }
  unfold Rdiv. apply Rmult_le_compat_l; [ exact Hnum | ].
  apply Rinv_le_contravar; lra.
Qed.

(* THE THEOREM: inside the certified ball the regret is bounded by C * ||dB||^2, with C explicit.
   Stated on the assembled pieces -- gain error at most L_K * d (the cited Lipschitz hypothesis),
   state energy geometric at the contraction modulus, quadratic form bounded by ||R_K||. *)
Theorem ce_regret_explicit_bound : forall kappaP rkn lk x0sq th d n dk,
  0 < kappaP -> 0 < rkn -> 0 < lk -> 0 <= x0sq -> 0 < th -> th <= 1 -> 0 <= d ->
  Rabs dk <= lk * d ->
  kappaP * rkn * x0sq * (dk * dk) * geo_sq (1 - th / 2) n
  <= c_const kappaP rkn lk x0sq th * (d * d).
Proof.
  intros kappaP rkn lk x0sq th d n dk Hk Hr Hl Hx Ht0 Ht1 Hd Hdk.
  assert (Hc : 0 <= kappaP * rkn * x0sq).
  { apply Rmult_le_pos; [ apply Rmult_le_pos; lra | lra ]. }
  assert (Hgnn : 0 <= geo_sq (1 - th / 2) n) by apply geo_sq_nonneg.
  assert (Hgeo : geo_sq (1 - th / 2) n <= 2 / th).
  { apply Rle_trans with (/ (1 - (1 - th / 2) * (1 - th / 2)));
      [ apply geo_sq_partial_le; lra | apply geo_relaxation; assumption ]. }
  assert (Hsq : dk * dk <= lk * d * (lk * d)).
  { replace (dk * dk) with (Rabs dk * Rabs dk).
    - apply Rmult_le_compat; try apply Rabs_pos; exact Hdk.
    - rewrite <- Rabs_mult. apply Rabs_pos_eq. nra. }
  apply Rle_trans with (kappaP * rkn * x0sq * (lk * d * (lk * d)) * geo_sq (1 - th / 2) n).
  { apply Rmult_le_compat_r; [ exact Hgnn | ].
    apply Rmult_le_compat_l; [ exact Hc | exact Hsq ]. }
  apply Rle_trans with (kappaP * rkn * x0sq * (lk * d * (lk * d)) * (2 / th)).
  { apply Rmult_le_compat_l; [ apply Rmult_le_pos; [ exact Hc | nra ] | exact Hgeo ]. }
  right. unfold c_const. field. lra.
Qed.

(* The order statement, freed from Mania-Tu-Recht: the ratio regret/d^2 is bounded by a CONSTANT
   that does not depend on d, so the bound is genuinely quadratic and not a local expansion. *)
Theorem regret_quadratic_explicit : forall kappaP rkn lk x0sq th d n dk,
  0 < kappaP -> 0 < rkn -> 0 < lk -> 0 <= x0sq -> 0 < th -> th <= 1 -> 0 < d ->
  Rabs dk <= lk * d ->
  kappaP * rkn * x0sq * (dk * dk) * geo_sq (1 - th / 2) n / (d * d)
  <= c_const kappaP rkn lk x0sq th.
Proof.
  intros kappaP rkn lk x0sq th d n dk Hk Hr Hl Hx Ht0 Ht1 Hd Hdk.
  assert (H := ce_regret_explicit_bound kappaP rkn lk x0sq th d n dk
                 Hk Hr Hl Hx Ht0 Ht1 (Rlt_le _ _ Hd) Hdk).
  apply Rmult_le_reg_r with (r := d * d); [ nra | ].
  unfold Rdiv. rewrite Rmult_assoc, Rinv_l by nra. lra.
Qed.

(* ---------- (E) The safeguard, which is what makes E[regret] FINITE. ---------- *)

(* The estimate can fall outside the ball; on that event the certainty-equivalent gain may not
   stabilise at all and the regret is +infinity, so an unconditional E[R] does not exist. A
   safeguarded policy -- use K_hat when the checkable Lyapunov test passes, else fall back to a known
   stabilising K_0 -- has regret bounded by max(C d^2, R_0) on EVERY draw, hence integrable. *)
Theorem safeguarded_two_branch : forall cbound r0 d reg,
  0 <= cbound -> 0 <= r0 -> 0 <= d ->
  (reg <= cbound * (d * d) \/ reg <= r0) ->
  reg <= Rmax (cbound * (d * d)) r0.
Proof.
  intros cbound r0 d reg Hc Hr0 Hd [H | H].
  - apply Rle_trans with (cbound * (d * d)); [ exact H | apply Rmax_l ].
  - apply Rle_trans with r0; [ exact H | apply Rmax_r ].
Qed.

Theorem safeguard_bound_finite : forall cbound r0 d,
  0 <= cbound -> 0 <= r0 -> 0 <= d -> 0 <= Rmax (cbound * (d * d)) r0.
Proof.
  intros cbound r0 d Hc Hr0 Hd.
  apply Rle_trans with r0; [ exact Hr0 | apply Rmax_r ].
Qed.

(* ---------- (F) Composition with the statistical half. ---------- *)

(* C2 (Result 26) gives ||dB|| = O_p(a_G). Composed with the theorem above, the CONTROL regret is
   bounded by C * a_G^2 inside the ball -- the end-to-end statement the C1 line said was open, now
   with both the ball and the constant explicit rather than hypothesised. *)
Theorem end_to_end_composition : forall cbound aG d,
  0 <= cbound -> 0 <= d -> d <= aG -> forall reg,
  reg <= cbound * (d * d) -> reg <= cbound * (aG * aG).
Proof.
  intros cbound aG d Hc Hd Hle reg Hreg.
  apply Rle_trans with (cbound * (d * d)); [ exact Hreg | ].
  apply Rmult_le_compat_l; [ exact Hc | nra ].
Qed.
