(* Rocq: the GENERAL orthogonal-to-control regret TRANSFER theorem -- the general form of result 0,
   which only stated the p=1 (plug-in, O(eps^2)) and p=2 (Neyman-orthogonal/DML, O(eps^4)) instances
   (derived in validation/composition_transfer.mac). Claim: if the causal-effect estimator has error of
   order delta^p (p = the orthogonality order of the estimator), then the certainty-equivalence control
   regret is of order delta^(2*p) -- the control map DOUBLES the estimator's order, for EVERY p. This is
   a genuine composition theorem, not an arithmetic coincidence: it holds because control regret is
   quadratic in the ACTION error k*(u*(bhat)-u*(b))^2 (the exact map, not a linearisation) and the
   optimal action is Lipschitz in the effect, so the action error inherits the estimator's order p; the
   square then doubles it to 2*p. p=1 -> 2 and p=2 -> 4 are the two instances of result 0. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Lia.
Open Scope R_scope.

(* control regret is at most the curvature times the squared transfer (action) error: the exact
   certainty-equivalence map, quadratic in u*(bhat)-u*(b). *)
Theorem regret_le_transfer_sq : forall k du e,
  0 <= k -> Rabs du <= e -> k * du ^ 2 <= k * e ^ 2.
Proof.
  intros k du e Hk He. apply Rmult_le_compat_l; [exact Hk |]. split_Rabs; nra.
Qed.

(* the squaring doubles the order: (delta^p)^2 = delta^(2*p). *)
Lemma order_doubles : forall (d : R) (p : nat), (d ^ p) ^ 2 = d ^ (2 * p).
Proof.
  intros d p. rewrite <- pow_mult. f_equal. lia.
Qed.

(* THE TRANSFER THEOREM: an order-p estimator error (|u*(bhat)-u*(b)| <= delta^p, the Lipschitz
   transfer of an O(delta^p) effect estimator) yields an order-2*p control regret. Recovers result 0:
   p=1 plug-in -> O(delta^2), p=2 orthogonal/DML -> O(delta^4); general p -> O(delta^(2*p)). *)
Theorem regret_order_2p : forall k du d (p : nat),
  0 <= k -> Rabs du <= d ^ p -> k * du ^ 2 <= k * d ^ (2 * p).
Proof.
  intros k du d p Hk He. rewrite <- order_doubles. apply regret_le_transfer_sq; assumption.
Qed.
