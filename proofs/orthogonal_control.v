(* Rocq: the DOUBLE debiasing of orthogonal certainty-equivalence control -- a novel result of CHC
   (derived symbolically in validation/orthogonal_control.mac, empirically certified by
   chc.regret.orthogonal_control_certificate). Certainty-equivalence control regret is quadratic in
   the causal-effect error d (the descent bound kappa/2 * d^2, cf. proofs/interference_regret.v). A
   plug-in (single-residualisation) effect estimate is O(eps)-biased in the nuisance error eps, so its
   regret is quadratic in eps; the Neyman-orthogonal Double ML effect is O(eps^2)-biased, so through
   the SAME quadratic regret map its regret is QUARTIC in eps -- two debiasings compounding, one from
   statistics (orthogonality) and one from control (quadraticity). For eps in [0,1] the orthogonal
   controller's regret bound therefore dominates the plug-in's. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* Certainty-equivalence regret at causal-effect error d, kappa-smooth cost. *)
Definition ce_regret (kappa d : R) : R := kappa / 2 * d ^ 2.

(* A plug-in effect error d = k*eps (O(eps)) gives a regret quadratic in the nuisance error eps. *)
Lemma plugin_regret_quadratic : forall kappa k eps,
  ce_regret kappa (k * eps) = kappa / 2 * k ^ 2 * eps ^ 2.
Proof. intros kappa k eps. unfold ce_regret. ring. Qed.

(* A Neyman-orthogonal effect error d = k*eps^2 (O(eps^2)) gives a regret QUARTIC in eps. *)
Lemma orthogonal_regret_quartic : forall kappa k eps,
  ce_regret kappa (k * eps ^ 2) = kappa / 2 * k ^ 2 * eps ^ 4.
Proof. intros kappa k eps. unfold ce_regret. ring. Qed.

(* The double debiasing: for eps in [0,1] the orthogonal (quartic) regret is at most the plug-in
   (quadratic) regret at the same constant -- orthogonal certainty-equivalence control dominates. *)
Theorem orthogonal_dominates_plugin : forall kappa k eps,
  0 <= kappa -> 0 <= eps -> eps <= 1 ->
  ce_regret kappa (k * eps ^ 2) <= ce_regret kappa (k * eps).
Proof.
  intros kappa k eps Hk He0 He1. unfold ce_regret.
  apply Rmult_le_compat_l; [lra |].
  (* goal: (k*eps^2)^2 <= (k*eps)^2, i.e. k^2*eps^4 <= k^2*eps^2 for 0<=eps<=1 *)
  assert (Hprod : 0 <= k ^ 2 * eps ^ 2) by nra.
  assert (H1 : 0 <= 1 - eps ^ 2) by nra.
  nra.
Qed.
