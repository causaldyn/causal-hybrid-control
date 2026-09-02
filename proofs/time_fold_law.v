(* Rocq: the algebraic core of the TIME-FOLD LAW (Result 53).

   validation/time_fold_law.mac derives, for cross-fit folds that cut TIME under an AR(1)
   covariance Sigma[t,s] = phi^|t-s|: the fold partition enters the sandwich only through the
   same-fold phi-weighted pair count (the affine reduction of Result 52, linear here because the
   time functional is linear in Sigma), and the stationary mode score is the AR(1) spectral
   density lambda(c) = (1-phi^2)/(1-2 phi c + phi^2), c = cos theta -- STRICTLY INCREASING in c
   for every phi in (0,1). All spectral mass therefore belongs at c = -1: ALTERNATE time points
   between folds at every phi; there is no threshold and no stripe regime, unlike the spatial law
   (fold_spectrum_law.v), because this kernel is completely monotone in the lag. The lag-convexity
   (second difference phi^r (1-phi)^2 >= 0) is Hubbard's most-homogeneous-ground-state hypothesis
   at density 1/2, which is the alternating configuration.

   Honest scope: Stdlib has no matrices, so what is proved here is the scalar spine -- the
   denominator floor, the cross-multiplied strict ordering of mode scores (division-free form; the
   Maxima file carries the division), and the convexity identity with symbolic nat lag. The
   finite-path argmin/argmax facts are exhaustive enumeration in the research log (p <= 14). *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Psatz.
Open Scope R_scope.

(* (A) The denominator of the mode score is bounded below by (1-phi)^2 > 0 on |c| <= 1:
   1 - 2 phi c + phi^2 = (1-phi)^2 + 2 phi (1-c). *)
Lemma ar1_denominator_floor :
  forall phi c : R, 0 < phi < 1 -> -1 <= c <= 1 -> 0 < 1 - 2*phi*c + phi^2.
Proof.
  intros phi c [H0 H1] [Hc1 Hc2].
  replace (phi^2) with (phi*phi) by ring.
  nra.
Qed.

(* (B) Strict ordering of mode scores, division-free: f(c1) < f(c2) over positive denominators
   is exactly N*D2 < N*D1 with N = 1 - phi^2 > 0 and D_i = 1 - 2 phi c_i + phi^2. Larger c means
   smaller denominator means larger score -- so the minimal score over modes sits at c = -1,
   the alternating design, at EVERY phi in (0,1). *)
Lemma ar1_mode_score_ordering :
  forall phi c1 c2 : R, 0 < phi < 1 -> c1 < c2 ->
  (1 - phi^2) * (1 - 2*phi*c2 + phi^2) < (1 - phi^2) * (1 - 2*phi*c1 + phi^2).
Proof.
  intros phi c1 c2 [H0 H1] Hlt.
  replace (phi^2) with (phi*phi) by ring.
  assert (Hn : 0 < (1 - phi) * (1 + phi)) by nra.
  assert (Hd : 0 < phi * (c2 - c1)) by nra.
  nra.
Qed.

(* (C) No interior vertex, unlike the spatial law: the score is monotone on the whole of
   [-1, 1], so the optimum is at the boundary c = -1 -- the derivative 2 phi (1-phi^2) over a
   square never vanishes for phi in (0,1). Stated as: the cross-multiplied difference between
   c and -1 is positive for every c > -1. *)
Lemma alternating_is_the_argmin :
  forall phi c : R, 0 < phi < 1 -> -1 < c ->
  (1 - phi^2) * (1 - 2*phi*c + phi^2) < (1 - phi^2) * (1 - 2*phi*(-1) + phi^2).
Proof.
  intros phi c [H0 H1] Hc.
  replace (phi^2) with (phi*phi) by ring.
  assert (Hn : 0 < (1 - phi) * (1 + phi)) by nra.
  assert (Hd : 0 < phi * (c + 1)) by nra.
  nra.
Qed.

(* (D) Hubbard's convexity hypothesis holds for the AR(1) kernel, with symbolic nat lag:
   phi^(r+2) - 2 phi^(r+1) + phi^r = phi^r (1-phi)^2 >= 0. *)
Lemma hubbard_second_difference :
  forall (phi : R) (r : nat),
  phi^(r + 2) - 2*phi^(r + 1) + phi^r = phi^r * (1 - phi)^2.
Proof.
  intros phi r.
  rewrite pow_add.
  rewrite pow_add.
  ring.
Qed.

Lemma convex_kernel_nonneg :
  forall (phi : R) (r : nat), 0 <= phi -> 0 <= phi^r * (1 - phi)^2.
Proof.
  intros phi r H.
  assert (Hp : 0 <= phi^r) by (apply pow_le; lra).
  assert (Hs := Rle_0_sqr (1 - phi)).
  unfold Rsqr in Hs.
  replace ((1 - phi)^2) with ((1 - phi) * (1 - phi)) by ring.
  now apply Rmult_le_pos.
Qed.
