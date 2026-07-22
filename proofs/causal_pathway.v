(* Rocq (TEMPORAL CAUSAL PATHWAY laws): the algebra behind chc.pathway's causal_pathway(target) --
   discover the lagged graph, estimate the dynamic effect (IRF) along each path, rank + truncate.
   Three laws, each first derived in validation/causal_pathway.mac:

   (L1) IRF PATH-SUM. For a stable AR(2)/SVAR the impulse response g_h (= d target_{t+h}/d shock_t)
        obeys g_h = a1*g_{h-1} + a2*g_{h-2}, g_0=1, g_1=a1, and equals the SUM OVER LAGGED WALKS of
        length h (compositions of h into parts {1,2}) of the product of edge weights: g2 = a1^2+a2,
        g3 = a1^3+2*a1*a2, g4 = a1^4+3*a1^2*a2+a2^2. "Which paths, of what sign, reach the target."
   (L2) GEOMETRIC HORIZON-TRUNCATION. Under a CONTRACTING operator norm ||A|| = q < 1
        (submultiplicative; from spectral radius rho(A)<1 alone one gets only ||A^h|| <= C_q q^h for
        any q in (rho,1) by Gelfand -- the constant C_q may exceed 1 for non-normal / Jordan A), the
        scalar geometric sum (1-q)*sum_{k=0}^{n} q^k = 1 - q^{n+1} gives partial sum <= 1/(1-q) and a
        per-step tail shrinking by q each horizon -- so truncating at finite H loses geometrically little.
   (L3) WEAKEST-LINK / multiplicative bottleneck. A path's contribution is the PRODUCT of edge
        magnitudes: |a*b*c| = |a|*|b|*|c|, and one weak edge (|a|<=eps) caps the whole path at
        eps*|b|*|c|. The temporal-path analogue of the C2 channel bottleneck. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* ---- L1: AR(2) IRF path-sum (walk decomposition of the impulse response) ---- *)

(* g2 = a1*g1 + a2*g0 = a1*a1 + a2*1 = SUM over the two length-2 walks {[1,1],[2]}. *)
Lemma irf_g2_path_sum : forall a1 a2 : R,
  a1 * a1 + a2 * 1 = a1 ^ 2 + a2.
Proof. intros; ring. Qed.

(* g3 = a1*g2 + a2*g1 = a1^3 + 2*a1*a2 = SUM over the three walks {[1,1,1],[1,2],[2,1]}. *)
Lemma irf_g3_path_sum : forall a1 a2 g2 : R,
  g2 = a1 ^ 2 + a2 ->
  a1 * g2 + a2 * a1 = a1 ^ 3 + 2 * a1 * a2.
Proof. intros a1 a2 g2 Hg2. rewrite Hg2. ring. Qed.

(* g4 = a1*g3 + a2*g2 = a1^4 + 3*a1^2*a2 + a2^2 = SUM over the five length-4 walks. *)
Lemma irf_g4_path_sum : forall a1 a2 g3 g2 : R,
  g3 = a1 ^ 3 + 2 * a1 * a2 -> g2 = a1 ^ 2 + a2 ->
  a1 * g3 + a2 * g2 = a1 ^ 4 + 3 * a1 ^ 2 * a2 + a2 ^ 2.
Proof. intros a1 a2 g3 g2 Hg3 Hg2. rewrite Hg3, Hg2. ring. Qed.

(* ---- L2: geometric horizon-truncation ---- *)

(* Partial geometric sum sum_{k=0}^{n} r^k of the truncated pathway response. *)
Fixpoint geo_sum (r : R) (n : nat) : R :=
  match n with
  | O => 1
  | S k => geo_sum r k + r ^ (S k)
  end.

Lemma geo_sum_closed : forall (r : R) (n : nat),
  (1 - r) * geo_sum r n = 1 - r ^ (S n).
Proof.
  intros r n. induction n as [| n IH].
  - simpl. ring.
  - cbn [geo_sum].
    rewrite Rmult_plus_distr_l, IH.
    replace (r ^ S (S n)) with (r * r ^ S n) by (simpl; ring).
    ring.
Qed.

(* Truncating the pathway at any horizon n keeps the response bounded by 1/(1-r). *)
Lemma geo_sum_bounded : forall (r : R) (n : nat),
  0 <= r < 1 -> geo_sum r n <= 1 / (1 - r).
Proof.
  intros r n [Hr0 Hr1].
  assert (Hpow : 0 <= r ^ (S n)) by (apply pow_le; lra).
  apply Rmult_le_reg_l with (r := 1 - r); [lra|].
  rewrite geo_sum_closed.
  replace ((1 - r) * (1 / (1 - r))) with 1 by (field; lra).
  lra.
Qed.

(* The per-horizon tail term shrinks by the factor r<1 each step: geometric decay. *)
Lemma geometric_tail_decreasing : forall (r : R) (H : nat),
  0 <= r < 1 -> r ^ (S (S H)) <= r ^ (S H).
Proof.
  intros r H [Hr0 Hr1].
  assert (Hpow : 0 <= r ^ (S H)) by (apply pow_le; lra).
  replace (r ^ (S (S H))) with (r * r ^ (S H)) by (simpl; ring).
  nra.
Qed.

(* ---- L3: weakest-link multiplicative bottleneck ---- *)

Lemma weakest_link_product : forall a b c : R,
  Rabs (a * b * c) = Rabs a * Rabs b * Rabs c.
Proof. intros a b c. rewrite !Rabs_mult. reflexivity. Qed.

(* One near-zero edge caps the entire temporal path -- the pathway bottleneck. *)
Lemma weak_edge_caps_path : forall a b c eps : R,
  0 <= eps -> Rabs a <= eps ->
  Rabs (a * b * c) <= eps * (Rabs b * Rabs c).
Proof.
  intros a b c eps Heps Ha.
  rewrite weakest_link_product.
  assert (Hbc : 0 <= Rabs b * Rabs c) by (apply Rmult_le_pos; apply Rabs_pos).
  replace (Rabs a * Rabs b * Rabs c) with (Rabs a * (Rabs b * Rabs c)) by ring.
  apply Rmult_le_compat_r; assumption.
Qed.
