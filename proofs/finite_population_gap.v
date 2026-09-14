(* THE FINITE-POPULATION GAP: the algebraic core of what N agents cost against the density.

   Result 49 solves a mean-field game as a coupled HJB/Fokker-Planck pair, and the density it
   returns is the N -> infinity limit.  validation/finite_population_gap.mac prices the gap for
   the scalar LQ case: under the MEAN-FIELD optimal control the feedback depends on an agent's
   own state and on deterministic coefficients only, so the closed-loop agents are INDEPENDENT
   Ornstein-Uhlenbeck processes, the empirical mean is a sample mean of N i.i.d. draws, and

       E[(m_N(t) - m(t))^2] = v(t) / N      exactly, at every t and every N,

   with v the closed-loop variance.  The 1/sqrt(N) is an identity, not an asymptotic rate.

   This file proves the algebraic content of that statement, which is where the claims that
   transfer actually live: (1) v(t) is a convex combination of v(0) and the stationary variance,
   hence bounded uniformly in time -- which is what lets the gap be quoted once rather than per
   horizon; (2) the exponent is EXACT, in the form "quadruple the population and the gap halves",
   for any closed-loop rate, so it survives the change of rate that agent coupling causes; and
   (3) an O(1/N) bias loses to an O(1/sqrt N) fluctuation past an explicit population size, which
   is why a finite-N Nash correction never overturns the exponent.

   Stdlib Reals only.  Nothing here is about probability: the reduction to these scalars is done
   in Maxima and measured by chc.deep_galerkin.finite_population_gap_certificate. *)

From Stdlib Require Import Reals Lra.
Open Scope R_scope.

(* ===== the closed-loop variance ===== *)

Definition v_inf (sg rate : R) : R := - (sg * sg) / (2 * rate).

Definition v_ou (v0 sg rate t : R) : R :=
  v0 * exp (2 * rate * t) + (sg * sg) * (exp (2 * rate * t) - 1) / (2 * rate).

Lemma exp_le_one : forall x, x <= 0 -> exp x <= 1.
Proof.
  intros x Hx. destruct (Rle_lt_or_eq_dec x 0 Hx) as [Hlt | Heq].
  - rewrite <- exp_0. left. apply exp_increasing, Hlt.
  - rewrite Heq, exp_0. lra.
Qed.

Lemma decay_bounds : forall rate t, rate < 0 -> 0 <= t -> 0 < exp (2 * rate * t) <= 1.
Proof.
  intros rate t Hr Ht. split; [apply exp_pos |].
  apply exp_le_one. nra.
Qed.

Lemma v_inf_nonneg : forall sg rate, rate < 0 -> 0 <= v_inf sg rate.
Proof.
  intros sg rate Hr. unfold v_inf.
  assert (Hsq : 0 <= sg * sg) by apply Rle_0_sqr.
  assert (Hinv : / (2 * rate) < 0) by (apply Rinv_lt_0_compat; lra).
  replace (- (sg * sg) / (2 * rate)) with ((sg * sg) * (- / (2 * rate))) by (field; lra).
  nra.
Qed.

(* The whole shape of v in one identity: a convex combination of where it starts and where it
   settles, with weight exp(2 rate t).  Monotone, and bounded by the larger endpoint for ever. *)
Lemma v_ou_convex :
  forall v0 sg rate t,
    rate <> 0 ->
    v_ou v0 sg rate t
      = exp (2 * rate * t) * v0 + (1 - exp (2 * rate * t)) * v_inf sg rate.
Proof.
  intros v0 sg rate t Hr. unfold v_ou, v_inf. field. exact Hr.
Qed.

Theorem v_ou_bounds :
  forall v0 sg rate t,
    rate < 0 -> 0 <= t -> 0 <= v0 ->
    0 <= v_ou v0 sg rate t <= Rmax v0 (v_inf sg rate).
Proof.
  intros v0 sg rate t Hr Ht Hv0.
  assert (Hd := decay_bounds rate t Hr Ht).
  assert (Hi := v_inf_nonneg sg rate Hr).
  assert (H0 := Rmax_l v0 (v_inf sg rate)).
  assert (H1 := Rmax_r v0 (v_inf sg rate)).
  rewrite v_ou_convex by lra. nra.
Qed.

(* ===== the exponent, as an exact scaling rather than a fitted slope ===== *)

Definition gap (v n : R) : R := sqrt (v / n).

Theorem gap_scaling :
  forall v n c, 0 <= v -> 0 < n -> 0 < c -> gap v (c * c * n) = gap v n / c.
Proof.
  intros v n c Hv Hn Hc.
  assert (Hinv : 0 < / c) by (apply Rinv_0_lt_compat; exact Hc).
  assert (Hvn : 0 <= v / n).
  { apply Rmult_le_pos; [exact Hv |]. left. apply Rinv_0_lt_compat. exact Hn. }
  assert (Hsq : sqrt (/ c * / c) = / c).
  { replace (/ c * / c) with (Rsqr (/ c)) by (unfold Rsqr; ring).
    apply sqrt_Rsqr. lra. }
  unfold gap.
  replace (v / (c * c * n)) with ((v / n) * (/ c * / c)) by (field; lra).
  rewrite sqrt_mult by nra.
  rewrite Hsq. field. lra.
Qed.

(* "Quadruple the population and the gap halves" IS the exponent -1/2, with no logarithm and no
   regression.  A simulation that returns anything else is measuring its own Monte-Carlo error. *)
Corollary gap_quadruple :
  forall v n, 0 <= v -> 0 < n -> gap v (4 * n) = gap v n / 2.
Proof.
  intros v n Hv Hn.
  replace (4 * n) with (2 * 2 * n) by ring.
  apply gap_scaling; lra.
Qed.

(* The exponent does not know the rate, so coupling the agents -- which moves the closed-loop
   rate from A to A + kappa and therefore moves the CONSTANT -- leaves it untouched. *)
Corollary gap_quadruple_any_rate :
  forall v0 sg rate t n,
    rate < 0 -> 0 <= t -> 0 <= v0 -> 0 < n ->
    gap (v_ou v0 sg rate t) (4 * n) = gap (v_ou v0 sg rate t) n / 2.
Proof.
  intros v0 sg rate t n Hr Ht Hv0 Hn.
  apply gap_quadruple; [| exact Hn].
  apply (v_ou_bounds v0 sg rate t Hr Ht Hv0).
Qed.

(* ===== an O(1/N) bias loses to an O(1/sqrt N) fluctuation, past an explicit N ===== *)

(* The N-player Nash equilibrium weights an agent's own state 1/N inside the empirical mean, so
   its coupling gain is kappa = c/N and the resulting drift away from the mean-field mean is
   O(1/N).  This says when that bias is below any prescribed fraction eps of the fluctuation --
   and the threshold is explicit, so it is a deployment number, not an asymptotic remark. *)
(* The core is a statement about a bare ratio x = bias/(gap * eps): once x * x <= N, the O(1/N)
   term is below eps times the O(1/sqrt N) one.  Stated over a variable rather than over the
   quotient so the arithmetic stays polynomial. *)
Lemma bias_beats_fluctuation_core :
  forall x gap_const eps n,
    0 <= x -> 0 < gap_const -> 0 < eps -> 0 < n -> x * x <= n ->
    x * (gap_const * eps) / n <= eps * (gap_const / sqrt n).
Proof.
  intros x gap_const eps n Hx Hg He Hn Hsize.
  assert (Hs : 0 < sqrt n) by (apply sqrt_lt_R0; exact Hn).
  assert (Hss : sqrt n * sqrt n = n) by (apply sqrt_sqrt; lra).
  assert (Hns : n / sqrt n = sqrt n) by (rewrite <- Hss at 1; field; lra).
  assert (Hxs : x <= sqrt n) by nra.
  apply (Rmult_le_reg_r n); [exact Hn |].
  replace (x * (gap_const * eps) / n * n) with (x * (gap_const * eps)) by (field; lra).
  replace (eps * (gap_const / sqrt n) * n) with (eps * gap_const * (n / sqrt n))
    by (field; lra).
  rewrite Hns.
  (* (sqrt n - x) * (gap_const * eps) >= 0 is a product of two hypotheses, so name the second. *)
  assert (Hge : 0 < gap_const * eps) by nra.
  nra.
Qed.

Theorem bias_is_subdominant :
  forall bias_const gap_const eps n,
    0 <= bias_const -> 0 < gap_const -> 0 < eps -> 0 < n ->
    (bias_const / (gap_const * eps)) * (bias_const / (gap_const * eps)) <= n ->
    bias_const / n <= eps * (gap_const / sqrt n).
Proof.
  intros bias_const gap_const eps n Hb Hg He Hn Hsize.
  replace bias_const with (bias_const / (gap_const * eps) * (gap_const * eps))
    by (field; split; apply Rgt_not_eq; assumption).
  apply bias_beats_fluctuation_core; try assumption.
  apply Rmult_le_pos; [exact Hb |]. left. apply Rinv_0_lt_compat. nra.
Qed.

(* The same statement read the other way: the bias-to-fluctuation ratio is itself O(1/sqrt N). *)
Corollary bias_ratio_decays :
  forall bias_const gap_const n,
    0 <= bias_const -> 0 < gap_const -> 0 < n ->
    (bias_const / n) / (gap_const / sqrt n) = (bias_const / gap_const) / sqrt n.
Proof.
  intros bias_const gap_const n Hb Hg Hn.
  assert (Hs : 0 < sqrt n) by (apply sqrt_lt_R0; exact Hn).
  assert (Hss : sqrt n * sqrt n = n) by (apply sqrt_sqrt; lra).
  rewrite <- Hss at 1. field. lra.
Qed.
