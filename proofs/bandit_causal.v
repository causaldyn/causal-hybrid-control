(* Rocq: the BANDIT / adaptive-control version -- online learning of the causal effect while
   controlling (derived in validation/bandit_causal.mac). Per round the control regret is the
   certainty-equivalence coefficient C times the squared estimation error (proofs/causal_mpc.v). A
   de-confounded online estimator is consistent (err^2 ~ sigma^2/t), so its per-round regret strictly
   DECREASES and eventually falls below any fixed floor -- cumulative regret ~ C*sigma^2*log(T)
   (sublinear). A confounded estimator has err^2 -> beta^2 (systematic), a per-round FLOOR that never
   vanishes -- cumulative regret ~ C*beta^2*T (linear). Online causal control is O(log T); online
   confounded control is Theta(T). *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

Definition per_round_deconf (c sigma2 t : R) : R := c * sigma2 / t.   (* de-confounded: ~ 1/t *)
Definition per_round_conf   (c beta2 : R)   : R := c * beta2.          (* confounded: a fixed floor *)

(* the de-confounded per-round regret strictly decreases as data accumulates *)
Lemma deconfounded_per_round_decreasing : forall c sigma2 t1 t2,
  0 < c -> 0 < sigma2 -> 0 < t1 -> t1 < t2 ->
  per_round_deconf c sigma2 t2 < per_round_deconf c sigma2 t1.
Proof.
  intros c sigma2 t1 t2 Hc Hs Ht1 Hlt. unfold per_round_deconf, Rdiv.
  apply Rmult_lt_compat_l; [nra | apply Rinv_lt_contravar; [nra | exact Hlt]].
Qed.

(* ...and eventually falls below the confounded floor: once the data exceeds sigma^2/beta^2, online
   causal control beats confounded control (whose per-round regret never vanishes). The hypothesis is
   the cleared form of t > sigma^2/beta^2. *)
Theorem deconfounded_beats_confounded : forall c sigma2 beta2 t,
  0 < c -> 0 < t -> sigma2 < beta2 * t ->
  per_round_deconf c sigma2 t < per_round_conf c beta2.
Proof.
  intros c sigma2 beta2 t Hc Ht Hkey. unfold per_round_deconf, per_round_conf.
  apply Rmult_lt_reg_r with t; [exact Ht |].
  replace (c * sigma2 / t * t) with (c * sigma2) by (field; lra).
  nra.
Qed.
