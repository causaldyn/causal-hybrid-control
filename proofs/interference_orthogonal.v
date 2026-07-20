(* Rocq: interference x orthogonality -- why you must debias BOTH channels (combines
   proofs/interference_regret.v and proofs/orthogonal_control.v; derived in
   validation/interference_orthogonal.mac). Under interference the control-relevant effect is
   B = direct + spillover, so regret is quadratic in the total estimation error (eps_direct +
   eps_spillover). Double ML makes a channel's error O(eps^2). Orthogonalising ONLY the direct channel
   leaves the spillover error at O(eps): the regret (eps^2 + eps)^2 stays O(eps^2) -- no better than
   plug-in. Orthogonalising BOTH gives (eps^2 + eps^2)^2 = 4 eps^4, and for eps in [0,1] that dominates
   the half measure. Empirically confirmed by chc.regret.interference_orthogonal_certificate (slopes
   ~2, ~2, ~4). *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* orthogonalising the direct channel only: the spillover error keeps the regret at least quadratic *)
Lemma half_measure_stays_quadratic : forall eps,
  0 <= eps -> eps ^ 2 <= (eps ^ 2 + eps) ^ 2.
Proof.
  intros eps H0.
  assert (Hid : (eps ^ 2 + eps) ^ 2 - eps ^ 2 = eps ^ 3 * (eps + 2)) by ring.
  assert (Hp : 0 <= eps ^ 3 * (eps + 2)) by (apply Rmult_le_pos; nra).
  lra.
Qed.

(* orthogonalising both channels: the regret is quartic in the nuisance error *)
Lemma full_measure_is_quartic : forall eps,
  (eps ^ 2 + eps ^ 2) ^ 2 = 4 * eps ^ 4.
Proof. intros eps. ring. Qed.

(* double debiasing dominates the half measure on eps in [0,1] (equal only at the boundary eps = 1) *)
Theorem full_dominates_half : forall eps,
  0 <= eps -> eps <= 1 -> (eps ^ 2 + eps ^ 2) ^ 2 <= (eps ^ 2 + eps) ^ 2.
Proof.
  intros eps H0 H1.
  assert (Hid : (eps ^ 2 + eps) ^ 2 - (eps ^ 2 + eps ^ 2) ^ 2
                = eps ^ 2 * (3 * eps + 1) * (1 - eps)) by ring.
  assert (Hp : 0 <= eps ^ 2 * (3 * eps + 1) * (1 - eps)).
  { repeat apply Rmult_le_pos; nra. }
  lra.
Qed.
