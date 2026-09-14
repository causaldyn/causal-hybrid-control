(* Rocq: the algebraic core of the SCHEDULED and BUDGETED capped exploration law (Result 66).

   Result 56 answered "how long do you explore under a per-round action cap" with a single
   number, n* = sqrt(K T/(A c))/cap, and recorded as honest scope that a cap which VARIES over
   rounds, or a total budget on top of the cap, changes the feasible set so that n* is no longer
   a formula. validation/capped_exploration_schedule.mac closes that: differentiating the
   objective along the greedy fill gives dF/dn = cap(n) * bracket(S(n)), the cap enters only as a
   positive factor, so the stopping MASS is cap-free while the stopping ROUND is not.

   What is proved here is the scalar spine of that statement:
     (A) a positive factor cannot move a root -- the mass does not know the cap level;
     (B) the bracket is strictly increasing in the mass, so it crosses zero once;
     (C) the explicit root, and the constant-cap corollary that recovers Result 56 with the O(1)
         prior-information term it declined to model;
     (D) below the root more mass still pays, which is why a binding budget is spent in full;
     (E) a round that delivers no mass costs exactly nothing -- the flat tail that makes the
         certificate's argmin a tie, and the reason it takes the FIRST minimiser;
     (F) the idle-prefix reduction: m rounds under a zero cap add a constant and hand the
         remaining horizon to the same problem, so a dead actuator is not an approximation;
     (G) the exchange argument with two different caps, feasibility included.

   Honest scope, as in capped_exploration.v: Stdlib has no schedules, so the prefix-sum induction
   and the O(T) sweep stay in Maxima and in the certificate. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Psatz.
Open Scope R_scope.

(* The marginal cost of one more exploring round, stripped of its cap factor:
   validation/capped_exploration_schedule.mac STEP 2a gives dF/dn = cap(n) * bracket(S(n)). *)
Definition bracket (a k c tail i0 s : R) : R := a - k * c * tail / (i0 + c * s) ^ 2.

(* The cost of stopping after n exploring rounds having delivered mass s, with [inside] the
   estimation sum already accrued inside the block: STEP 2 of the .mac, and the curve that
   chc.regret._scheduled_block_costs materialises. *)
Definition stop_cost (a k c t i0 s inside n : R) : R :=
  a * s + inside + k * (t - n) / (i0 + c * s).

(* ===== (A) THE STOPPING MASS DOES NOT KNOW THE CAP ===== *)

Lemma a_positive_cap_cannot_move_the_root :
  forall cap a k c tail i0 s,
  0 < cap -> (cap * bracket a k c tail i0 s = 0 <-> bracket a k c tail i0 s = 0).
Proof.
  intros cap a k c tail i0 s Hcap.
  split; intro H.
  - apply (Rmult_eq_reg_l cap); [rewrite H; ring | lra].
  - rewrite H; ring.
Qed.

Lemma two_cap_levels_stop_at_the_same_mass :
  forall cap1 cap2 a k c tail i0 s,
  0 < cap1 -> 0 < cap2 ->
  cap1 * bracket a k c tail i0 s = 0 -> cap2 * bracket a k c tail i0 s = 0.
Proof.
  intros cap1 cap2 a k c tail i0 s H1 H2 H.
  apply (a_positive_cap_cannot_move_the_root cap2 a k c tail i0 s H2).
  apply (a_positive_cap_cannot_move_the_root cap1 a k c tail i0 s H1).
  exact H.
Qed.

(* ===== (B) SECOND ORDER: ONE CROSSING ===== *)

Lemma the_bracket_strictly_rises_with_the_mass :
  forall a k c tail i0 s s',
  0 < k -> 0 < c -> 0 < tail -> 0 < i0 -> 0 <= s -> s < s' ->
  bracket a k c tail i0 s < bracket a k c tail i0 s'.
Proof.
  intros a k c tail i0 s s' Hk Hc Ht Hi Hs Hlt.
  assert (H1 : 0 < i0 + c * s) by nra.
  assert (H2 : 0 < i0 + c * s') by nra.
  unfold bracket.
  assert (Hsq1 : 0 < (i0 + c * s) ^ 2) by nra.
  assert (Hsq2 : 0 < (i0 + c * s') ^ 2) by nra.
  assert (Hord : (i0 + c * s) ^ 2 < (i0 + c * s') ^ 2).
  { replace ((i0 + c * s') ^ 2)
      with ((i0 + c * s) ^ 2 + (c * (s' - s)) * (2 * i0 + c * (s + s'))) by ring.
    assert (Hprod : 0 < c * (s' - s) * (2 * i0 + c * (s + s')))
      by (apply Rmult_lt_0_compat; nra).
    lra. }
  assert (Hnum : 0 < k * c * tail) by (repeat apply Rmult_lt_0_compat; assumption).
  assert (Hdrop : k * c * tail / (i0 + c * s') ^ 2 < k * c * tail / (i0 + c * s) ^ 2).
  { unfold Rdiv. apply Rmult_lt_compat_l; [exact Hnum |].
    apply Rinv_lt_contravar; nra. }
  lra.
Qed.

(* ===== (C) THE ROOT, AND RESULT 56 AS ITS CONSTANT-CAP COROLLARY ===== *)

(* The bracket vanishes exactly where the marginal estimation gain equals the marginal control
   price; stating it as a product identity keeps the division out of the algebra. *)
Lemma bracket_vanishes_at_a_balanced_mass :
  forall a k c tail i0 s,
  0 < i0 + c * s -> k * c * tail = a * (i0 + c * s) ^ 2 -> bracket a k c tail i0 s = 0.
Proof.
  intros a k c tail i0 s Hpos Heq.
  unfold bracket. rewrite Heq. field. nra.
Qed.

Lemma the_mass_that_zeroes_the_bracket :
  forall a k c tail i0,
  0 < a -> 0 < k -> 0 < c -> 0 < tail -> 0 < i0 ->
  bracket a k c tail i0 (sqrt (k * tail / (a * c)) - i0 / c) = 0.
Proof.
  intros a k c tail i0 Ha Hk Hc Ht Hi.
  assert (Hq : 0 < k * tail / (a * c)).
  { apply Rdiv_lt_0_compat; apply Rmult_lt_0_compat; assumption. }
  assert (Hr : 0 < sqrt (k * tail / (a * c))) by (apply sqrt_lt_R0; exact Hq).
  assert (Hinfo : i0 + c * (sqrt (k * tail / (a * c)) - i0 / c)
                  = c * sqrt (k * tail / (a * c))) by (field; nra).
  apply bracket_vanishes_at_a_balanced_mass.
  - rewrite Hinfo. nra.
  - rewrite Hinfo.
    replace ((c * sqrt (k * tail / (a * c))) ^ 2)
      with (c * c * (sqrt (k * tail / (a * c)) * sqrt (k * tail / (a * c)))) by ring.
    rewrite sqrt_sqrt by nra.
    field. nra.
Qed.

Lemma a_constant_cap_turns_the_mass_into_a_round_count :
  forall cap a k c tail i0,
  0 < cap -> 0 < c ->
  (sqrt (k * tail / (a * c)) - i0 / c) / cap
  = sqrt (k * tail / (a * c)) / cap - i0 / (c * cap).
Proof.
  intros cap a k c tail i0 Hcap Hc. field; lra.
Qed.

(* ===== (D) A BINDING BUDGET IS SPENT IN FULL ===== *)

Lemma below_the_root_more_mass_still_pays :
  forall a k c tail i0 s sstar,
  0 < k -> 0 < c -> 0 < tail -> 0 < i0 -> 0 <= s -> s < sstar ->
  bracket a k c tail i0 sstar = 0 -> bracket a k c tail i0 s < 0.
Proof.
  intros a k c tail i0 s sstar Hk Hc Ht Hi Hs Hlt Hroot.
  rewrite <- Hroot.
  apply the_bracket_strictly_rises_with_the_mass; assumption.
Qed.

(* ===== (E) AN EMPTY ROUND IS EXACTLY FREE ===== *)

(* Stepping past a round that delivers nothing adds one estimation term and removes one
   exploitation term at the SAME information, so the cost curve is flat -- not nearly flat. This
   is why chc.regret.capped_exploration_policy takes the first minimiser: once the budget is
   spent, every later stopping round ties, and floating-point noise alone decides an argmin. *)
Lemma an_empty_round_is_exactly_free :
  forall a k c t i0 s inside n,
  i0 + c * s <> 0 ->
  stop_cost a k c t i0 s (inside + k / (i0 + c * s)) (n + 1)
  = stop_cost a k c t i0 s inside n.
Proof.
  intros a k c t i0 s inside n Hinfo. unfold stop_cost. field; exact Hinfo.
Qed.

(* ===== (F) THE IDLE-PREFIX REDUCTION ===== *)

Lemma an_idle_prefix_is_a_constant_plus_the_shifted_problem :
  forall a k c t i0 s inside m n,
  i0 + c * s <> 0 -> i0 <> 0 ->
  stop_cost a k c t i0 s (m * (k / i0) + inside) (m + n)
  = m * (k / i0) + stop_cost a k c (t - m) i0 s inside n.
Proof.
  intros a k c t i0 s inside m n Hinfo Hi.
  unfold stop_cost. field; split; assumption.
Qed.

Lemma the_dead_prefix_is_a_single_candidate :
  forall k t i0 n, i0 <> 0 -> n * (k / i0) + k * (t - n) / i0 = k * t / i0.
Proof.
  intros k t i0 n Hi. field; exact Hi.
Qed.

(* ===== (G) THE EXCHANGE ARGUMENT UNDER TWO DIFFERENT CAPS ===== *)

(* Result 56's exchange inequality is about prefix masses and never mentions a cap. What a
   varying cap adds is a FEASIBILITY obligation: the earlier round must be able to absorb the
   mass moved into it. Once it can, the inequality is the old one. *)
Lemma moving_mass_earlier_is_feasible_and_strictly_cheaper :
  forall i0 c k cap_early x y s,
  0 < i0 -> 0 < c -> 0 < k -> 0 <= s -> 0 <= x -> 0 < y -> x + y <= cap_early ->
  x + y <= cap_early /\
  k / (i0 + c * (s + x + y)) < k / (i0 + c * (s + x)).
Proof.
  intros i0 c k cap_early x y s Hi Hc Hk Hs Hx Hy Hfeas.
  split; [exact Hfeas |].
  assert (H1 : 0 < i0 + c * (s + x)) by nra.
  assert (H2 : 0 < i0 + c * (s + x + y)) by nra.
  unfold Rdiv.
  apply Rmult_lt_compat_l; [exact Hk |].
  apply Rinv_lt_contravar; nra.
Qed.

(* ===== (H) THE O(1) TERM RESULT 56 DROPPED (P3.2) ===== *)

(* Under a CONSTANT cap the stopping round is n = s/kap, so the remaining horizon inside the
   bracket is itself a function of the mass and the root stops being a square root. Writing
   w = i0 + c*s, the balance a(i0 + c s)^2 = k c (t - s/kap) is a QUADRATIC in w,

       a*w^2 + (k/kap)*w = k*c*t + k*i0/kap                                                 (He)

   while Result 56's closed form s56 = sqrt(k t/(a c)) - i0/c is the root of the SAME balance with
   the remaining horizon held at the FULL t, i.e. w0 = i0 + c*s56 solves

       a*w0^2 = k*c*t                                                                       (H0)

   validation/capped_exploration_schedule.mac STEP 7 expands (He) at large t and
   validation/capped_exploration_o1.gp confirms the expansion to 60 digits over nine horizons.
   What is proved here is the EXACT relation the expansion comes from, which is stronger than an
   asymptotic statement: it holds at every finite horizon and needs no remainder estimate. *)

(* Subtracting the two balances kills the horizon entirely -- and what is left mentions the cap. *)
Lemma the_two_balances_differ_only_by_the_cap_term :
  forall a k c kap t i0 s we w0,
    kap <> 0 ->
    we = i0 + c * s ->
    a * (we * we) + (k / kap) * we = k * c * t + k * i0 / kap ->
    a * (w0 * w0) = k * c * t ->
    a * (w0 * w0 - we * we) = k * c * s / kap.
Proof.
  intros a k c kap t i0 s we w0 Hkap Hw He H0.
  subst we.
  replace (a * (w0 * w0 - (i0 + c * s) * (i0 + c * s)))
    with (a * (w0 * w0) - a * ((i0 + c * s) * (i0 + c * s))) by ring.
  rewrite H0.
  assert (Hsq : a * ((i0 + c * s) * (i0 + c * s))
                = k * c * t + k * i0 / kap - k / kap * (i0 + c * s)) by lra.
  rewrite Hsq. field. exact Hkap.
Qed.

(* So the cap-free form is an UPPER bound on the mass at every horizon: it over-states, never
   under-states, which is why shipping it as the target made the policy aim past the optimum. *)
Lemma the_cap_free_form_overstates_the_mass :
  forall a k c kap s we w0,
    0 < a -> 0 <= k -> 0 < c -> 0 < kap -> 0 <= s -> 0 <= we -> 0 <= w0 ->
    a * (w0 * w0 - we * we) = k * c * s / kap ->
    we <= w0.
Proof.
  intros a k c kap s we w0 Ha Hk Hc Hkap Hs Hwe Hw0 Hgap.
  assert (Hkc : 0 <= k * c) by nra.
  assert (Hnum : 0 <= k * c * s) by nra.
  assert (Hinv : 0 <= / kap) by (left; apply Rinv_0_lt_compat; lra).
  assert (Hpos : 0 <= k * c * s / kap) by (unfold Rdiv; apply Rmult_le_pos; assumption).
  destruct (Rle_lt_dec we w0) as [Hok | Hbad]; [exact Hok |].
  assert (Hwe0 : 0 < we) by lra.
  assert (Hsq : we * we <= w0 * w0) by nra.
  nra.
Qed.

(* The gap in closed form, division-free: this is the O(1) term BEFORE any expansion. As the
   horizon grows both roots grow like c*s, the second factor tends to 2*a*c*s, and the quotient
   tends to k/(2*a*kap) -- which in mass units is the constant Result 56 dropped. *)
Lemma the_overstatement_is_an_exact_product :
  forall a k c kap s we w0,
    a * (w0 * w0 - we * we) = k * c * s / kap ->
    (w0 - we) * (a * (w0 + we)) = k * c * s / kap.
Proof.
  intros a k c kap s we w0 Hgap. nra.
Qed.

(* And it never exceeds that constant, at ANY horizon: in mass units (w0 - we)/c <= k/(2 a c kap),
   stated here cleared of division. The Maxima expansion says what happens in the limit; this says
   the bound is a ceiling the whole way. *)
Lemma the_overstatement_never_exceeds_the_cap_term :
  forall a k c kap i0 s we w0,
    0 < a -> 0 <= k -> 0 < c -> 0 < kap -> 0 <= s -> 0 < i0 ->
    we = i0 + c * s -> we <= w0 ->
    (w0 - we) * (a * (w0 + we)) * kap = k * c * s ->
    (w0 - we) * (2 * a * c * kap) <= k * c.
Proof.
  intros a k c kap i0 s we w0 Ha Hk Hc Hkap Hs Hi Hw Hle Hgap.
  assert (Hd : 0 <= w0 - we) by lra.
  destruct (Req_dec s 0) as [Hs0 | Hs0].
  - subst s.
    assert (Hwe : 0 < we) by nra.
    assert (Hfac : 0 < a * (w0 + we)) by nra.
    assert (Hz : (w0 - we) * (a * (w0 + we)) = 0).
    { apply (Rmult_eq_reg_r kap); [rewrite Hgap; ring | lra]. }
    assert (Heq : w0 - we = 0) by (destruct (Rmult_integral _ _ Hz) as [H | H]; lra).
    rewrite Heq, Rmult_0_l. nra.
  - assert (Hspos : 0 < s) by lra.
    assert (Hsum : 2 * (c * s) <= w0 + we) by nra.
    assert (Hstep : (w0 - we) * (a * (2 * (c * s))) * kap <= k * c * s).
    { rewrite <- Hgap.
      apply Rmult_le_compat_r; [lra |].
      apply Rmult_le_compat_l; [exact Hd |]. nra. }
    apply (Rmult_le_reg_r s); [exact Hspos |]. nra.
Qed.
