(* Rocq (DOES Psi DESCRIBE AN ESTIMATOR?): plan 24's P2.6. Result 51 shipped Psi with a scope note
   saying it is a functional of the PROCESS, evaluated with the fold operator held fixed, and not a
   re-derived estimator. This file proves the algebra that says what the experiment can and cannot
   show, as derived in validation/panel_estimator_gate.mac.

   Two groups. (i) The partition enters Psi exactly once, linearly, with a strictly positive weight
   -- so ranking designs is ranking the same-fold mass, which is Result 52's premise, and nothing
   about the ranking depends on the rest of Psi. (ii) With a block-diagonal covariance over g
   independent clusters, every partition-free factor cancels from the RATIO of two designs, leaving
   a Mobius function of g whose limit is 1 and whose distance from 1 is bounded by C/g. The design
   law is therefore a FINITE-CLUSTER statement: it washes out in the number of independent blocks,
   not in the number of rows. The ordering, by contrast, is g-invariant -- which is exactly the
   claim the measured gate confirms while rejecting the point prediction. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* --- (i) the partition enters once, linearly, with positive weight --- *)

(* Psi = n^2 (trS - r^4 v/n + (r^4-1)(k/n) ss) / (trA^2 trS), with ss the same-fold mass. *)
Definition psi (n r k v trS trA ss : R) : R :=
  n ^ 2 * (trS - r ^ 4 * v / n + (r ^ 4 - 1) * (k / n) * ss) / (trA ^ 2 * trS).

Lemma psi_strictly_increasing_in_same_fold_mass :
  forall n r k v trS trA ss1 ss2 : R,
    0 < n -> 1 < r -> 0 < k -> 0 < trS -> 0 < trA -> ss1 < ss2 ->
    psi n r k v trS trA ss1 < psi n r k v trS trA ss2.
Proof.
  intros n r k v trS trA ss1 ss2 Hn Hr Hk HtrS HtrA Hss.
  assert (Hr2 : 1 < r ^ 2) by nra.
  assert (Hpow : r ^ 4 = r ^ 2 * r ^ 2) by ring.
  assert (Hr4 : 1 < r ^ 4) by nra.
  assert (Hsq : 0 < trA ^ 2) by (apply pow_lt; exact HtrA).
  assert (Hden : 0 < trA ^ 2 * trS) by (apply Rmult_lt_0_compat; assumption).
  assert (Hgap : psi n r k v trS trA ss2 - psi n r k v trS trA ss1
                 = n * (r ^ 4 - 1) * k * (ss2 - ss1) / (trA ^ 2 * trS)).
  { unfold psi. field. split; lra. }
  assert (Hleft : 0 < n * (r ^ 4 - 1)) by nra.
  assert (Hright : 0 < k * (ss2 - ss1)) by nra.
  assert (Hnum : 0 < n * (r ^ 4 - 1) * k * (ss2 - ss1)).
  { replace (n * (r ^ 4 - 1) * k * (ss2 - ss1))
      with (n * (r ^ 4 - 1) * (k * (ss2 - ss1))) by ring.
    apply Rmult_lt_0_compat; assumption. }
  assert (Hpos : 0 < n * (r ^ 4 - 1) * k * (ss2 - ss1) / (trA ^ 2 * trS))
    by (apply Rdiv_lt_0_compat; assumption).
  lra.
Qed.

(* --- (ii) the design ratio is a Mobius function of the cluster count --- *)

(* Every partition-free factor cancels in the ratio, leaving branch(s) = base + w*s with
   base = g*n_c*t_c - r^4*v_c growing linearly in g and w = k*(r^4-1) > 0. *)
Definition branch (base w s : R) : R := base + w * s.
Definition ratio (base w s1 s2 : R) : R := branch base w s1 / branch base w s2.

Lemma ratio_gap :
  forall base w s1 s2 : R,
    branch base w s2 <> 0 ->
    ratio base w s1 s2 - 1 = w * (s1 - s2) / branch base w s2.
Proof.
  intros base w s1 s2 Hne. unfold ratio, branch in *. field. exact Hne.
Qed.

(* The better design (smaller same-fold mass) sits below 1 at EVERY cluster count: the ordering
   cannot flip with g, even though the size of the gain does. *)
Lemma ordering_is_cluster_invariant :
  forall base w s1 s2 : R,
    0 < w -> 0 < branch base w s2 -> s1 < s2 -> ratio base w s1 s2 < 1.
Proof.
  intros base w s1 s2 Hw Hpos Hlt.
  assert (Hne : branch base w s2 <> 0) by lra.
  assert (Hgap : ratio base w s1 s2 - 1 = w * (s1 - s2) / branch base w s2)
    by (apply ratio_gap; exact Hne).
  assert (Hneg : w * (s1 - s2) < 0) by nra.
  assert (Hinv : 0 < / branch base w s2) by (apply Rinv_0_lt_compat; exact Hpos).
  assert (Hquot : w * (s1 - s2) * / branch base w s2 < 0) by nra.
  unfold Rdiv in Hgap. lra.
Qed.

(* And the gap is O(1/g): with base = g*c and c > 0 the distance from 1 is at most C/g. *)
Lemma design_gap_is_order_one_over_clusters :
  forall g c w s1 s2 : R,
    0 < g -> 0 < c -> 0 < w -> s1 < s2 -> 0 <= w * s2 ->
    Rabs (ratio (g * c) w s1 s2 - 1) <= (w * (s2 - s1)) / (g * c).
Proof.
  intros g c w s1 s2 Hg Hc Hw Hs Hws2.
  assert (Hbase : 0 < g * c) by (apply Rmult_lt_0_compat; lra).
  assert (Hpos : 0 < branch (g * c) w s2) by (unfold branch; lra).
  assert (Hne : branch (g * c) w s2 <> 0) by lra.
  assert (Hinv : 0 < / branch (g * c) w s2) by (apply Rinv_0_lt_compat; exact Hpos).
  rewrite (ratio_gap (g * c) w s1 s2 Hne).
  rewrite Rabs_left1.
  - replace (- (w * (s1 - s2) / branch (g * c) w s2))
      with (w * (s2 - s1) / branch (g * c) w s2) by (field; exact Hne).
    unfold Rdiv. apply Rmult_le_compat_l; [nra|].
    apply Rinv_le_contravar; unfold branch in *; lra.
  - unfold Rdiv. assert (Hneg : w * (s1 - s2) < 0) by nra. nra.
Qed.

(* Monotone toward 1: a larger base (more clusters) brings the ratio strictly closer to 1. *)
Lemma ratio_rises_with_the_cluster_count :
  forall base1 base2 w s1 s2 : R,
    0 < w -> s1 < s2 -> 0 < branch base1 w s2 -> base1 < base2 ->
    ratio base1 w s1 s2 < ratio base2 w s1 s2.
Proof.
  intros base1 base2 w s1 s2 Hw Hs Hpos1 Hbase.
  assert (Hpos2 : 0 < branch base2 w s2) by (unfold branch in *; lra).
  assert (Hne1 : branch base1 w s2 <> 0) by lra.
  assert (Hne2 : branch base2 w s2 <> 0) by lra.
  assert (G1 : ratio base1 w s1 s2 - 1 = w * (s1 - s2) / branch base1 w s2)
    by (apply ratio_gap; exact Hne1).
  assert (G2 : ratio base2 w s1 s2 - 1 = w * (s1 - s2) / branch base2 w s2)
    by (apply ratio_gap; exact Hne2).
  assert (Hneg : w * (s1 - s2) < 0) by nra.
  assert (Hprod : 0 < branch base1 w s2 * branch base2 w s2)
    by (apply Rmult_lt_0_compat; assumption).
  assert (Hinv : / branch base2 w s2 < / branch base1 w s2)
    by (apply Rinv_lt_contravar; [exact Hprod | unfold branch in *; lra]).
  assert (Hstep : w * (s1 - s2) * / branch base1 w s2
                  < w * (s1 - s2) * / branch base2 w s2) by nra.
  unfold Rdiv in G1, G2. lra.
Qed.
