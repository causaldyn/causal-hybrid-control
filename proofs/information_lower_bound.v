(* Rocq: an INFORMATION-THEORETIC LOWER BOUND on control regret -- the first *lower* bound in the
   program (all others are upper bounds), from statistics/information theory (Cramer-Rao). "You cannot
   control better than your information about the effect allows." An unbiased effect estimator has
   variance v; the certainty-equivalence control regret is C*v; the Cramer-Rao bound v >= 1/(n*I)
   (I the Fisher information per sample) makes the expected regret at least C/(n*I). Two consequences
   (validation/information_lower_bound.mac): (1) the 1/n rate matches the online UPPER bound
   (proofs/bandit_causal.v), so the O(log T) cumulative rate is OPTIMAL; (2) confounding reduces the
   Fisher information, so its lower bound is HIGHER -- a fundamental reason causal control is harder. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

Definition expected_regret (c v : R) : R := c * v.   (* CE regret of an unbiased effect estimator *)

(* Cramer-Rao lower bound on control regret: no unbiased causal controller beats C/(n*I). *)
Theorem cramer_rao_control_lower_bound : forall c v n info,
  0 <= c -> 0 < n -> 0 < info -> 1 / (n * info) <= v ->
  c / (n * info) <= expected_regret c v.
Proof.
  intros c v n info Hc Hn Hi Hcr. unfold expected_regret.
  replace (c / (n * info)) with (c * (1 / (n * info))) by (field; nra).
  apply Rmult_le_compat_l; assumption.
Qed.

(* Confounding reduces the Fisher information about the causal effect (it steals identifying
   variation), so the lower bound is antitone in the information: the confounded floor is HIGHER. *)
Theorem confounding_raises_the_floor : forall c n info_conf info_exp,
  0 <= c -> 0 < n -> 0 < info_conf -> info_conf <= info_exp ->
  c / (n * info_exp) <= c / (n * info_conf).
Proof.
  intros c n ic ie Hc Hn Hic Hle. unfold Rdiv.
  apply Rmult_le_compat_l; [exact Hc |].
  apply Rinv_le_contravar; nra.
Qed.
