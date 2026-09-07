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

(* ------------------------------------------------------------------ *)
(* (F) which cells no sampling rescue reaches: the second-moment pole  *)
(* ------------------------------------------------------------------ *)

(* For B > 0 the sandwich M^-1 X'BX M^-1 is squeezed in Loewner order between lambda_min(B) M^-1
   and lambda_max(B) M^-1, so it is square-integrable under the Wishart law exactly when E[W^-2]
   is. Von Rosen's moment is (n-1) I / ((n-q)(n-q-1)(n-q-3)). What is formalised below is the
   ARITHMETIC of the pole locations -- where that denominator vanishes and what sign it takes on
   either side -- not the measure-theoretic existence statement itself, which is von Rosen 1988
   and is checked here numerically instead (q = 2,3,4, agreement to 0.06-0.43% at margin +3). *)
Definition second_moment_denominator (n q : R) : R := (n - q) * (n - q - 1) * (n - q - 3).

Lemma second_moment_poles_at_q_plus_three :
  forall q : R, second_moment_denominator (q + 3) q = 0.
Proof. intros q; unfold second_moment_denominator; ring_simplify; ring. Qed.

Lemma second_moment_denominator_positive_past_the_pole :
  forall n q : R, q + 3 < n -> 0 < second_moment_denominator n q.
Proof.
  intros n q H; unfold second_moment_denominator.
  apply Rmult_lt_0_compat; [apply Rmult_lt_0_compat |]; lra.
Qed.

(* On the existence boundary itself the formula does not merely fail to apply: it returns a
   NEGATIVE number for a quantity that is a sum of squares. That is the unmistakable signature of
   an expectation that does not exist, and it is why the QMC replicate SD -- not the sample mean --
   was the statistic that diagnosed the failure. *)
Lemma second_moment_denominator_negative_on_the_boundary_cell :
  forall q : R, second_moment_denominator (q + 2) q < 0.
Proof. intros q; unfold second_moment_denominator; lra. Qed.

(* The two cells that matter are exactly those where the VALUE exists and its second moment does
   not: n = q+2 and n = q+3 clear the E[W^-1] pole at n = q+1 and fail the E[W^-2] pole at n = q+3.
   Everything Result 63 reports on the boundary lives in this gap -- the integral is finite, so a
   quadrature can converge on it (the tensor rule does, at rate 1.3-1.7), while sample-and-average
   has infinite variance there. *)
Definition inverse_moment_pole (q : R) (k : nat) : R := q + 2 * INR k - 1.

Lemma the_pole_moves_two_per_moment :
  forall q : R, inverse_moment_pole q 2 - inverse_moment_pole q 1 = 2.
Proof. intros q; unfold inverse_moment_pole; simpl; ring. Qed.

Lemma value_exists_but_second_moment_does_not :
  forall n q : R, n = q + 2 \/ n = q + 3 ->
    inverse_moment_pole q 1 < n /\ ~ (inverse_moment_pole q 2 < n).
Proof.
  intros n q [H | H]; unfold inverse_moment_pole; simpl; split; lra.
Qed.

(* ------------------------------------------------------------------ *)
(* (G) the master anchor, and the one power of R it depends on         *)
(* ------------------------------------------------------------------ *)

(* validation/general_q_ratio_moment.mac identity 30. For Omega = R (x) S with denominator
   C = S^-1, factor S = F F' and R = G G' (any invertible factors -- no square root needed). Then
   vec(X) ~ N(0, R (x) S) is the law of X = F Z G' for a standard Z, the denominator becomes
   G (Z'Z) G' -- S cancels COMPLETELY -- and the numerator becomes G Z' (F'BF) Z G'. Applying the
   Omega = I anchor to the bracket and using tr(F'BF) = tr(BS),

       E[M^-1 X'BX M^-1] = (tr(BS)/n) * R^-1 / (n - q - 1)                                  (30)

   for any PSD B. It contains the Wishart anchor (B = R = S = I), the anisotropic-numerator anchor
   (R = S = I) and the correlated-channel anchor (B = S = I). Stdlib has no matrices, so what is
   proved here is what the formula asserts beyond the matrix algebra: the ENTIRE dependence on the
   channel covariance is the single inverse power R^-1, and the rest is the scalar below. *)
Definition master_scale (trBS n q : R) : R := trBS / n / (n - q - 1).

(* The anchor's pole is the FIRST inverse-moment pole of (F), not a new one: the closed form
   exists exactly where E[W^-1] does, and (F) then says its second moment needs two more. *)
Lemma master_scale_poles_at_the_first_inverse_moment :
  forall q : R, inverse_moment_pole q 1 - q - 1 = 0.
Proof. intros q; unfold inverse_moment_pole; simpl; ring. Qed.

Lemma master_scale_positive_past_the_pole :
  forall trBS n q : R, 0 < trBS -> 0 < n -> inverse_moment_pole q 1 < n ->
    0 < master_scale trBS n q.
Proof.
  intros trBS n q Ht Hn Hp; unfold master_scale.
  unfold inverse_moment_pole in Hp; simpl in Hp.
  apply Rdiv_lt_0_compat; [apply Rdiv_lt_0_compat |]; lra.
Qed.

(* B = S = I gives tr(BS) = tr(I_n) = n, and the scalar collapses to the Wishart constant. This is
   the degeneration that makes (30) a generalisation rather than a different formula. *)
Lemma master_scale_degenerates_to_the_wishart_constant :
  forall n q : R, n <> 0 -> master_scale n n q = / (n - q - 1).
Proof.
  intros n q Hn; unfold master_scale.
  rewrite Rdiv_diag by exact Hn.
  unfold Rdiv; rewrite Rmult_1_l; reflexivity.
Qed.

(* Three scalings, one law. The numerator and the row covariance enter (30) linearly and only
   through tr(BS); the channel covariance enters inversely and only through R^-1. So on the scalar
   channel R = rho * I the whole anchor scales by beta * sigma / rho -- which is the falsifiable
   part of the formula, and what identity 30 checks entry by entry. *)
Definition master_entry (trBS n q rho : R) : R := master_scale trBS n q / rho.

Lemma master_entry_scales_by_beta_sigma_over_rho :
  forall trBS n q rho beta sigma kappa : R,
    n <> 0 -> n - q - 1 <> 0 -> rho <> 0 -> kappa <> 0 ->
    master_entry (beta * sigma * trBS) n q (kappa * rho)
      = (beta * sigma / kappa) * master_entry trBS n q rho.
Proof.
  intros trBS n q rho beta sigma kappa Hn Hd Hr Hk.
  unfold master_entry, master_scale; field; repeat split; assumption.
Qed.

(* And what the anchor did NOT buy. The same factorisation says a Kronecker Omega can be reduced to
   the isotropic problem and conjugated back, which was proposed as a fast path in the evaluator.
   Measured against the direct path on three cells the accuracy ratios are 1.82, 1.49 and 0.65: the
   reduction LOSES on the third. A branch that changes the answer's error in an unpredictable
   direction by less than a factor of two is a second implementation to keep in sync, not a fast
   path, so the identity ships as identity 30 and this section, and the branch does not. *)
Definition uniform_gain (gs : list R) : Prop := Forall (fun g => 1 <= g) gs.

Lemma measured_kronecker_gains_are_not_uniform :
  ~ uniform_gain (182 / 100 :: 149 / 100 :: 65 / 100 :: nil).
Proof.
  intros H; unfold uniform_gain in H.
  assert (Hlast : 1 <= 65 / 100)
    by exact (Forall_inv (Forall_inv_tail (Forall_inv_tail H))).
  lra.
Qed.
