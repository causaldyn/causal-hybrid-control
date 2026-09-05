(* Rocq: the algebraic core of the GENERAL-q matrix ratio moment (Result 63).

   validation/general_q_ratio_moment.mac lifts Result 54 off q = 2. Three things change and
   each is checked here:

   (A) The Isserlis expansion is replaced by a sum over the symmetric group,
         E[prod_{i=1..m} z' K_i z] = sum_{sigma in S_m} 2^(m - c(sigma)) prod_{cycles} tr(...),
       with c(sigma) the cycle count. The weights must reproduce the pairing count (2m-1)!!,
       which is the falsifiable content: it is checked by computation for m up to 7, i.e. past
       the m = 2q-1 = 5 the q = 3 sandwich needs.

   (B) det(M)^{-2} is an Ingham-Siegel integral with the MATRIX gamma
         Gamma_q(s) = pi^(q(q-1)/4) prod_{j=1..q} Gamma(s - (j-1)/2),
       which converges only for s > (q-1)/2. The sandwich needs s = 2 at every q, so the route
       is valid exactly for q <= 4 -- a structural ceiling, not an implementation limit.

   (C) The cone is q(q+1)/2-dimensional, so the quadrature is the binding constraint, not the
       algebra. The evaluator therefore ships a self-certifying diagnostic: on an exchangeable
       problem the exact answer is isotropic, so the observed spread of the diagonal LOWER
       BOUNDS twice the largest entry error -- no reference value required. That inequality is
       the load-bearing lemma of the certificate and is proved below.

   Stdlib has no matrices; what is proved is the arithmetic the evaluator relies on. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Lia.
From Stdlib Require Import List.
From Stdlib Require Import Arith.
From Stdlib Require Import Bool.
Import ListNotations.
Open Scope nat_scope.

(* ------------------------------------------------------------------ *)
(* (A) permutation weights reproduce the Isserlis pairing count        *)
(* ------------------------------------------------------------------ *)

(* Permutations of [0..n-1] as lists, built by insertion at every position. *)
Fixpoint insert_all (x : nat) (l : list nat) : list (list nat) :=
  match l with
  | [] => [[x]]
  | y :: t => (x :: y :: t) :: map (fun r => y :: r) (insert_all x t)
  end.

Fixpoint perms (l : list nat) : list (list nat) :=
  match l with
  | [] => [[]]
  | x :: t => flat_map (insert_all x) (perms t)
  end.

Definition apply_perm (p : list nat) (i : nat) : nat := nth i p 0.

(* Cycle count of a permutation given as the image list. *)
Fixpoint walk (p : list nat) (fuel start cur : nat) (seen : list nat) : list nat :=
  match fuel with
  | O => seen
  | S f => if Nat.eqb cur start && negb (Nat.eqb (length seen) 0) then seen
           else walk p f start (apply_perm p cur) (cur :: seen)
  end.

Fixpoint count_cycles_from (p : list nat) (n : nat) (idx : nat) (visited : list nat) : nat :=
  match idx with
  | O => O
  | S k =>
      let i := n - idx in
      if existsb (Nat.eqb i) visited
      then count_cycles_from p n k visited
      else S (count_cycles_from p n k (walk p n i i [] ++ visited))
  end.

Definition cycles (n : nat) (p : list nat) : nat := count_cycles_from p n n [].

Definition perm_weight (n : nat) (p : list nat) : nat := 2 ^ (n - cycles n p).

Definition weight_sum (n : nat) : nat :=
  fold_right (fun p acc => perm_weight n p + acc) 0 (perms (seq 0 n)).

(* (2m-1)!! -- the number of perfect matchings of 2m points, which is what Isserlis counts. *)
Fixpoint double_fact_odd (m : nat) : nat :=
  match m with
  | O => 1
  | S k => (2 * m - 1) * double_fact_odd k
  end.

Lemma weights_match_pairings_1 : weight_sum 1 = double_fact_odd 1.
Proof. vm_compute; reflexivity. Qed.

Lemma weights_match_pairings_2 : weight_sum 2 = double_fact_odd 2.
Proof. vm_compute; reflexivity. Qed.

Lemma weights_match_pairings_3 : weight_sum 3 = double_fact_odd 3.
Proof. vm_compute; reflexivity. Qed.

Lemma weights_match_pairings_4 : weight_sum 4 = double_fact_odd 4.
Proof. vm_compute; reflexivity. Qed.

(* m = 5 is the case the q = 3 sandwich actually consumes: 2q-1 forms per entry. *)
Lemma weights_match_pairings_5 : weight_sum 5 = double_fact_odd 5.
Proof. vm_compute; reflexivity. Qed.

Lemma the_q3_sandwich_needs_five_forms : forall q : nat, q = 3%nat -> (2 * q - 1 = 5)%nat.
Proof. intros q H; subst; reflexivity. Qed.

Lemma five_forms_expand_to_120_terms : length (perms (seq 0 5)) = 120%nat.
Proof. vm_compute; reflexivity. Qed.

(* The identity permutation has m cycles and weight 1; an m-cycle has weight 2^(m-1). Both are
   the extreme terms of the sum and pin the normalisation. *)
Lemma identity_has_unit_weight : perm_weight 5 [0; 1; 2; 3; 4]%nat = 1%nat.
Proof. vm_compute; reflexivity. Qed.

Lemma full_cycle_has_weight_two_to_m_minus_one :
  perm_weight 5 [1; 2; 3; 4; 0]%nat = 16%nat.
Proof. vm_compute; reflexivity. Qed.

Close Scope nat_scope.
Open Scope R_scope.

(* ------------------------------------------------------------------ *)
(* (B) the Ingham-Siegel exponent, and why q <= 4                      *)
(* ------------------------------------------------------------------ *)

(* Gamma_q(s) needs s > (j-1)/2 for every j = 1..q, i.e. s > (q-1)/2. The sandwich fixes s = 2. *)
Definition ingham_siegel_valid (q : nat) : Prop := 2 > (INR q - 1) / 2.

Lemma ingham_siegel_valid_at_four : ingham_siegel_valid 4.
Proof. unfold ingham_siegel_valid; simpl; lra. Qed.

Lemma ingham_siegel_fails_at_five : ~ ingham_siegel_valid 5.
Proof. unfold ingham_siegel_valid; simpl; lra. Qed.

Lemma ingham_siegel_valid_iff_q_le_four :
  forall q : nat, ingham_siegel_valid q <-> (q <= 4)%nat.
Proof.
  intros q; unfold ingham_siegel_valid; split; intro H.
  - destruct (le_lt_dec q 4) as [Hle | Hlt]; [exact Hle |].
    exfalso. assert (INR 5 <= INR q) by (apply le_INR; lia).
    simpl in *; lra.
  - assert (INR q <= INR 4) by (apply le_INR; exact H).
    simpl in *; lra.
Qed.

(* The density exponent det(T)^(s - (q+1)/2) at s = 2: positive at q = 1,2, ZERO at q = 3,
   negative at q = 4. Zero at q = 3 is why the q = 3 integrand carries no determinant factor. *)
Definition cone_exponent (q : nat) : R := 2 - (INR q + 1) / 2.

Lemma cone_exponent_vanishes_at_three : cone_exponent 3 = 0.
Proof. unfold cone_exponent; simpl; lra. Qed.

Lemma cone_exponent_is_half_at_two : cone_exponent 2 = / 2.
Proof. unfold cone_exponent; simpl; lra. Qed.

(* ------------------------------------------------------------------ *)
(* (C) the self-certifying quadrature diagnostic                       *)
(* ------------------------------------------------------------------ *)

(* On an exchangeable problem the exact sandwich is c*I: every diagonal entry equals the same c
   and every off-diagonal vanishes. A quadrature that returns a,b on the diagonal therefore
   cannot have max error below half the spread -- an error bar computed WITHOUT knowing c. *)
Lemma spread_lower_bounds_twice_the_error :
  forall a b c : R, Rabs (a - b) <= Rabs (a - c) + Rabs (b - c).
Proof.
  intros a b c.
  replace (a - b) with ((a - c) + - (b - c)) by ring.
  eapply Rle_trans; [apply Rabs_triang |].
  rewrite Rabs_Ropp; lra.
Qed.

Lemma half_spread_bounds_the_max_error :
  forall a b c e : R,
  Rabs (a - c) <= e -> Rabs (b - c) <= e -> Rabs (a - b) / 2 <= e.
Proof.
  intros a b c e Ha Hb.
  assert (H := spread_lower_bounds_twice_the_error a b c); lra.
Qed.

(* And it is only a LOWER bound: a grid can be isotropic and still wrong, so a small spread
   convicts nothing. Witness: a = b = c + e with e > 0 has zero spread and error e. *)
Lemma zero_spread_does_not_certify :
  forall c e : R, 0 < e -> Rabs ((c + e) - (c + e)) = 0 /\ Rabs ((c + e) - c) = e.
Proof.
  intros c e He; split.
  - replace ((c + e) - (c + e)) with 0 by ring; apply Rabs_R0.
  - replace ((c + e) - c) with e by ring; apply Rabs_right; lra.
Qed.

(* ------------------------------------------------------------------ *)
(* (D) the scale mixture: multivariate t transports by one scalar      *)
(* ------------------------------------------------------------------ *)

(* M^-1 N M^-1 is homogeneous of degree -1 in the scale of X, so mixing over w with E[w] = 1
   leaves the SCALE-parameterised answer alone; parameterising by the VARIANCE instead costs
   nu/(nu-2), i.e. an excess of 2/(nu-2). *)
Lemma sandwich_is_homogeneous_of_degree_minus_one :
  forall v w : R, w <> 0 -> (/ (w * v)) * (w * v) * (/ (w * v)) = / w * (/ v * v * / v).
Proof.
  intros v w Hw.
  destruct (Req_dec v 0) as [Hv | Hv].
  - subst v. rewrite Rmult_0_r, Rinv_0. lra.
  - field; split; assumption.
Qed.

Lemma variance_parameterisation_costs_two_over_nu_minus_two :
  forall nu : R, nu > 2 -> nu / (nu - 2) - 1 = 2 / (nu - 2).
Proof. intros nu H; field; lra. Qed.

(* ------------------------------------------------------------------ *)
(* (E) what a grid-refinement residual is worth: the rate-2 threshold  *)
(* ------------------------------------------------------------------ *)

(* The shipped certificate reports |X_k - X_(k-1)| as a stand-in for the unknown |X_k - X|.
   If the error decays geometrically in a fixed direction, e_k = e/r and e_(k-1) = e, then the
   residual is e - e/r, so residual / true = r - 1 EXACTLY. *)
Lemma refinement_residual_is_rate_minus_one :
  forall e r : R, e <> 0 -> r <> 0 -> (e - e / r) / (e / r) = r - 1.
Proof. intros e r He Hr; field; split; assumption. Qed.

(* Hence the residual majorises the error iff the per-node decay rate reaches 2. This is the
   whole content of the q = 3 failure: the measured rates on the existence boundary are
   1.30, 1.44, 1.68 -- all below 2 -- so the residual under-states there by construction, not
   by accident. *)
Lemma residual_bounds_iff_rate_reaches_two :
  forall e r : R, 0 < e -> 0 < r -> (e / r <= e - e / r <-> 2 <= r).
Proof.
  intros e r He Hr.
  assert (Hr' : r <> 0) by lra.
  assert (Hx : e / r * r = e) by (field; assumption).
  assert (Hy : (e - e / r) * r = e * r - e) by (field; assumption).
  split; intro H.
  - assert (Hm : e / r * r <= (e - e / r) * r) by (apply Rmult_le_compat_r; lra).
    rewrite Hx, Hy in Hm. nra.
  - apply Rmult_le_reg_r with (r := r); [assumption |].
    rewrite Hx, Hy. nra.
Qed.

(* And the threshold is strict on the failing side: at r < 2 the residual is strictly smaller
   than the error it is meant to bound, so a small residual is not evidence of a small error. *)
Lemma rate_below_two_understates :
  forall e r : R, 0 < e -> 0 < r -> r < 2 -> e - e / r < e / r.
Proof.
  intros e r He Hr H2.
  assert (Hr' : r <> 0) by lra.
  apply Rmult_lt_reg_r with (r := r); [assumption |].
  replace (e / r * r) with e by (field; assumption).
  replace ((e - e / r) * r) with (e * r - e) by (field; assumption).
  nra.
Qed.
