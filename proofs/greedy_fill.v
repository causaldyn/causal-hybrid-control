(* Rocq: the greedy fill is optimal for EVERY prefix-antitone objective, at every horizon.

   Results 56 and 66 prove the capped-exploration law for one estimation term, K/(I0 + c S), and
   leave the T-round step -- "saturate the earliest rounds" beats every other schedule -- to Maxima
   and the certificate, on the grounds that Stdlib has no schedules (capped_exploration.v,
   capped_exploration_schedule.v). It needs none. A schedule is a function nat -> R, a prefix sum
   is a recursion, and the whole T-round argument is two facts about prefix sums:

     (A) every feasible schedule of total M has t-th prefix sum at most min(M, C_t), with C_t the
         caps' own prefix sum;
     (B) the greedy fill of total M -- each round at its cap until M is spent, nothing after --
         attains min(M, C_t) at EVERY t, so it dominates every feasible schedule of the same total,
         prefix by prefix, all at once.
     (C) So for ANY estimation term that is a sum over rounds of functions of the prefix sum, each
         non-increasing, the greedy fill costs no more than any feasible schedule with the same
         total, and strictly less when it delivers strictly more information before a round whose
         term is strictly decreasing. The exploration cost A*M is the same for both.

   Nothing here uses convexity, and nothing uses the form K/(I0 + c S): section (D) shows that
   form is one member of the class. The optimisation over the total M is then one-dimensional,
   which is where Result 66's bracket takes over. A budget constrains only M, so it changes the
   one-dimensional problem and not the schedule. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Psatz.
From Stdlib Require Import Arith.
Open Scope R_scope.

(* psum v t = v 0 + ... + v (t-1): the information delivered BEFORE round t. *)
Fixpoint psum (v : nat -> R) (t : nat) : R :=
  match t with
  | O => 0
  | S t' => psum v t' + v t'
  end.

(* Each round at its cap until the total m is spent: the schedule Results 56 and 66 ship. *)
Definition greedy (cap : nat -> R) (m : R) (t : nat) : R :=
  Rmax 0 (Rmin (cap t) (m - psum cap t)).

(* The estimation cost of rounds 0 .. T-1, each charged a function of the information before it. *)
Fixpoint est_cost (phi : nat -> R -> R) (v : nat -> R) (horizon : nat) : R :=
  match horizon with
  | O => 0
  | S t' => est_cost phi v t' + phi t' (psum v t')
  end.

Definition cost (a : R) (phi : nat -> R -> R) (v : nat -> R) (horizon : nat) : R :=
  a * psum v horizon + est_cost phi v horizon.

Definition feasible (cap v : nat -> R) : Prop := forall t, 0 <= v t <= cap t.

Definition antitone_on_nonneg (phi : nat -> R -> R) : Prop :=
  forall t x y, 0 <= x -> x <= y -> phi t y <= phi t x.

Definition strictly_antitone_on_nonneg (phi : nat -> R -> R) : Prop :=
  forall t x y, 0 <= x -> x < y -> phi t y < phi t x.

(* ===== (A) A FEASIBLE SCHEDULE NEVER OUTRUNS min(M, C_t) ===== *)

Lemma psum_nonneg :
  forall v, (forall t, 0 <= v t) -> forall t, 0 <= psum v t.
Proof.
  intros v Hv t. induction t as [| t IH]; simpl; [lra |].
  specialize (Hv t). lra.
Qed.

Lemma psum_below_the_caps :
  forall cap v, feasible cap v -> forall t, psum v t <= psum cap t.
Proof.
  intros cap v Hf t. induction t as [| t IH]; simpl; [lra |].
  destruct (Hf t). lra.
Qed.

Lemma psum_grows :
  forall v, (forall t, 0 <= v t) -> forall t k, psum v t <= psum v (t + k).
Proof.
  intros v Hv t k. induction k as [| k IH].
  - rewrite Nat.add_0_r. lra.
  - rewrite Nat.add_succ_r. simpl. specialize (Hv (t + k)%nat). lra.
Qed.

Lemma psum_below_the_total :
  forall v, (forall t, 0 <= v t) -> forall t horizon, (t <= horizon)%nat ->
  psum v t <= psum v horizon.
Proof.
  intros v Hv t horizon Hle.
  replace horizon with (t + (horizon - t))%nat by lia.
  apply psum_grows. exact Hv.
Qed.

Lemma a_feasible_prefix_is_below_min_of_total_and_caps :
  forall cap v horizon t, feasible cap v -> (t <= horizon)%nat ->
  psum v t <= Rmin (psum v horizon) (psum cap t).
Proof.
  intros cap v horizon t Hf Hle.
  apply Rmin_glb.
  - apply psum_below_the_total; [intro s; destruct (Hf s); lra | exact Hle].
  - apply psum_below_the_caps. exact Hf.
Qed.

(* ===== (B) THE GREEDY FILL ATTAINS min(M, C_t) AT EVERY ROUND ===== *)

Lemma greedy_is_feasible :
  forall cap m, (forall t, 0 <= cap t) -> feasible cap (greedy cap m).
Proof.
  intros cap m Hcap t. unfold greedy. specialize (Hcap t).
  unfold Rmax, Rmin.
  destruct (Rle_dec (cap t) (m - psum cap t));
    destruct (Rle_dec 0 _); lra.
Qed.

Lemma greedy_prefix_is_min_of_total_and_caps :
  forall cap m, (forall t, 0 <= cap t) -> 0 <= m ->
  forall t, psum (greedy cap m) t = Rmin m (psum cap t).
Proof.
  intros cap m Hcap Hm t.
  assert (HC : forall s, 0 <= psum cap s) by (apply psum_nonneg; exact Hcap).
  induction t as [| t IH].
  - simpl. unfold Rmin. destruct (Rle_dec m 0); lra.
  - simpl. rewrite IH. unfold greedy.
    specialize (Hcap t). specialize (HC t).
    unfold Rmax, Rmin.
    destruct (Rle_dec m (psum cap t));
      destruct (Rle_dec (cap t) (m - psum cap t));
      destruct (Rle_dec m (psum cap t + cap t));
      try destruct (Rle_dec 0 _); lra.
Qed.

Lemma greedy_spends_the_total :
  forall cap m horizon, (forall t, 0 <= cap t) -> 0 <= m -> m <= psum cap horizon ->
  psum (greedy cap m) horizon = m.
Proof.
  intros cap m horizon Hcap Hm Hle.
  rewrite (greedy_prefix_is_min_of_total_and_caps cap m Hcap Hm horizon).
  unfold Rmin. destruct (Rle_dec m (psum cap horizon)); lra.
Qed.

Theorem the_greedy_fill_dominates_every_prefix :
  forall cap v horizon t, (forall s, 0 <= cap s) -> feasible cap v -> (t <= horizon)%nat ->
  psum v t <= psum (greedy cap (psum v horizon)) t.
Proof.
  intros cap v horizon t Hcap Hf Hle.
  assert (Hv : forall s, 0 <= v s) by (intro s; destruct (Hf s); lra).
  rewrite (greedy_prefix_is_min_of_total_and_caps cap (psum v horizon) Hcap
             (psum_nonneg v Hv horizon) t).
  apply a_feasible_prefix_is_below_min_of_total_and_caps; assumption.
Qed.

(* ===== (C) SO THE GREEDY FILL IS OPTIMAL, FOR THE WHOLE CLASS ===== *)

Lemma est_cost_is_monotone_in_the_terms :
  forall phi v w horizon,
  (forall t, (t < horizon)%nat -> phi t (psum w t) <= phi t (psum v t)) ->
  est_cost phi w horizon <= est_cost phi v horizon.
Proof.
  intros phi v w horizon H. induction horizon as [| h IH]; simpl; [lra |].
  assert (Hlast : phi h (psum w h) <= phi h (psum v h)) by (apply H; lia).
  assert (Hrest : est_cost phi w h <= est_cost phi v h) by (apply IH; intros t Ht; apply H; lia).
  lra.
Qed.

Lemma est_cost_is_strictly_monotone_in_one_term :
  forall phi v w horizon t0,
  (forall t, (t < horizon)%nat -> phi t (psum w t) <= phi t (psum v t)) ->
  (t0 < horizon)%nat -> phi t0 (psum w t0) < phi t0 (psum v t0) ->
  est_cost phi w horizon < est_cost phi v horizon.
Proof.
  intros phi v w horizon t0 H Ht0 Hstrict.
  induction horizon as [| h IH]; [lia |].
  simpl.
  destruct (Nat.eq_dec t0 h) as [-> | Hne].
  - assert (Hrest : est_cost phi w h <= est_cost phi v h)
      by (apply est_cost_is_monotone_in_the_terms; intros t Ht; apply H; lia).
    lra.
  - assert (Hlast : phi h (psum w h) <= phi h (psum v h)) by (apply H; lia).
    assert (Hrest : est_cost phi w h < est_cost phi v h)
      by (apply IH; [intros t Ht; apply H; lia | lia]).
    lra.
Qed.

Theorem the_greedy_fill_is_optimal_at_its_total :
  forall a phi cap v horizon,
  antitone_on_nonneg phi -> (forall t, 0 <= cap t) -> feasible cap v ->
  cost a phi (greedy cap (psum v horizon)) horizon <= cost a phi v horizon.
Proof.
  intros a phi cap v horizon Hphi Hcap Hf.
  assert (Hv : forall s, 0 <= v s) by (intro s; destruct (Hf s); lra).
  set (m := psum v horizon).
  assert (Hm : 0 <= m) by (apply psum_nonneg; exact Hv).
  unfold cost.
  rewrite (greedy_spends_the_total cap m horizon Hcap Hm (psum_below_the_caps cap v Hf horizon)).
  assert (Hest : est_cost phi (greedy cap m) horizon <= est_cost phi v horizon).
  { apply est_cost_is_monotone_in_the_terms. intros t Ht.
    apply Hphi.
    - apply psum_nonneg. exact Hv.
    - apply the_greedy_fill_dominates_every_prefix; [exact Hcap | exact Hf | lia]. }
  apply Rplus_le_compat_l. exact Hest.
Qed.

Theorem any_information_shortfall_is_strictly_costly :
  forall a phi cap v horizon t0,
  strictly_antitone_on_nonneg phi -> (forall t, 0 <= cap t) -> feasible cap v ->
  (t0 < horizon)%nat -> psum v t0 < psum (greedy cap (psum v horizon)) t0 ->
  cost a phi (greedy cap (psum v horizon)) horizon < cost a phi v horizon.
Proof.
  intros a phi cap v horizon t0 Hphi Hcap Hf Ht0 Hshort.
  assert (Hv : forall s, 0 <= v s) by (intro s; destruct (Hf s); lra).
  set (m := psum v horizon).
  assert (Hm : 0 <= m) by (apply psum_nonneg; exact Hv).
  unfold cost.
  rewrite (greedy_spends_the_total cap m horizon Hcap Hm (psum_below_the_caps cap v Hf horizon)).
  assert (Hweak : antitone_on_nonneg phi).
  { intros t x y Hx Hxy. destruct (Rle_lt_or_eq_dec x y Hxy) as [Hlt | ->].
    - left. apply Hphi; assumption.
    - lra. }
  assert (Hest : est_cost phi (greedy cap m) horizon < est_cost phi v horizon).
  { apply (est_cost_is_strictly_monotone_in_one_term phi v (greedy cap m) horizon t0).
    - intros t Ht. apply Hweak.
      + apply psum_nonneg. exact Hv.
      + apply the_greedy_fill_dominates_every_prefix; [exact Hcap | exact Hf | lia].
    - exact Ht0.
    - apply Hphi; [apply psum_nonneg; exact Hv | exact Hshort]. }
  apply Rplus_lt_compat_l. exact Hest.
Qed.

(* A budget B constrains only the total: the greedy fill of a feasible schedule's total is itself
   within the budget, so the budgeted problem is the same one-dimensional problem on [0, B]. *)
Corollary the_greedy_fill_keeps_the_budget :
  forall cap v horizon budget, (forall t, 0 <= cap t) -> feasible cap v ->
  psum v horizon <= budget -> psum (greedy cap (psum v horizon)) horizon <= budget.
Proof.
  intros cap v horizon budget Hcap Hf Hb.
  assert (Hv : forall s, 0 <= v s) by (intro s; destruct (Hf s); lra).
  rewrite (greedy_spends_the_total cap (psum v horizon) horizon Hcap (psum_nonneg v Hv horizon)
             (psum_below_the_caps cap v Hf horizon)).
  exact Hb.
Qed.

(* ===== (D) THE CAPPED-EXPLORATION TERM IS ONE MEMBER OF THE CLASS ===== *)

Lemma the_van_trees_term_is_strictly_antitone :
  forall k i0 c, 0 < k -> 0 < i0 -> 0 < c ->
  strictly_antitone_on_nonneg (fun _ s => k / (i0 + c * s)).
Proof.
  intros k i0 c Hk Hi Hc t x y Hx Hxy.
  assert (H1 : 0 < i0 + c * x) by nra.
  assert (H2 : 0 < i0 + c * y) by nra.
  unfold Rdiv.
  apply Rmult_lt_compat_l; [exact Hk |].
  apply Rinv_lt_contravar; [nra | nra].
Qed.

Corollary results_56_and_66_follow_at_every_horizon :
  forall a k i0 c cap v horizon,
  0 < k -> 0 < i0 -> 0 < c -> (forall t, 0 <= cap t) -> feasible cap v ->
  cost a (fun _ s => k / (i0 + c * s)) (greedy cap (psum v horizon)) horizon
  <= cost a (fun _ s => k / (i0 + c * s)) v horizon.
Proof.
  intros a k i0 c cap v horizon Hk Hi Hc Hcap Hf.
  apply the_greedy_fill_is_optimal_at_its_total; [| exact Hcap | exact Hf].
  intros t x y Hx Hxy. destruct (Rle_lt_or_eq_dec x y Hxy) as [Hlt | ->].
  - left. apply (the_van_trees_term_is_strictly_antitone k i0 c Hk Hi Hc t x y Hx Hlt).
  - lra.
Qed.
