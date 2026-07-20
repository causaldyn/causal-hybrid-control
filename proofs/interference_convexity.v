(* Rocq: strong convexity UNDER interference -- combines result #3 (proofs/nonlinear_regret.v, the
   Polyak-Lojasiewicz self-certifying bound grad^2/(2 mu)) with section A (proofs/interference_regret.v).
   Marketplace interference is cannibalisation: the benefit of incentivising saturates, adding
   convexity to the control objective, so the effective strong-convexity rises to mu + kappa_int
   (kappa_int >= 0 the cannibalisation curvature; derived in validation/interference_convexity.mac).
   Since the PL bound decreases in the convexity constant, cannibalising interference makes the
   self-certifying regret certificate TIGHTER -- a curse-and-blessing duality: interference hurts
   identification (section A) but helps the control certificate. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* The self-certifying PL regret bound at gradient-norm-squared g and strong-convexity mu. *)
Definition pl_bound (g mu : R) : R := g / (2 * mu).

(* The bound is antitone in the strong-convexity constant: more convexity => a tighter certificate. *)
Lemma pl_bound_monotone_in_convexity : forall g mu1 mu2,
  0 <= g -> 0 < mu1 -> mu1 <= mu2 -> pl_bound g mu2 <= pl_bound g mu1.
Proof.
  intros g mu1 mu2 Hg Hmu Hle. unfold pl_bound, Rdiv.
  apply Rmult_le_compat_l; [exact Hg |].
  apply Rinv_le_contravar; lra.
Qed.

(* Cannibalising interference (kappa_int >= 0) raises the effective convexity to mu + kappa_int, so
   its self-certifying PL bound is at least as tight as the interference-blind one that uses mu. *)
Theorem interference_tightens_pl_bound : forall g mu kappa_int,
  0 <= g -> 0 < mu -> 0 <= kappa_int ->
  pl_bound g (mu + kappa_int) <= pl_bound g mu.
Proof.
  intros g mu kappa_int Hg Hmu Hk.
  apply pl_bound_monotone_in_convexity; lra.
Qed.
