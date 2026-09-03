(* Rocq (MINIMAX LQ over the identified effect interval): §33 proved by counterexample that
   certainty equivalence is NOT minimax for the LQ loss, and left the robust controller unbuilt.
   validation/minimax_lq.mac derives it in closed form; this file proves the two things the closed
   form rests on and machine-checks §33's own instance.

   The stage cost is convex in the effect b, so the inner maximum sits at an endpoint and the outer
   problem minimises `Rmax (stage blo u) (stage bhi u)`. Two lemmas carry it: `stage_gap` (each
   branch is a quadratic whose deviation from its own minimum is exactly (p*b^2+r) times a square)
   and
   `branch_cross` (the branches differ by 4*D*p*u*(t - bh*u), so the lower one is the worse exactly
   on 0 <= u <= t/bh). `minimax_lower_branch` combines them: when the regime condition
   p*blo*D <= r holds, the CE action for the PESSIMISTIC endpoint minimises the worst case over the
   whole interval. `ce_not_minimax` is §33's counterexample, checked rather than asserted. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* Worst-case-relevant stage cost: p*(b*u - t)^2 + r*u^2, p a cost-to-go coefficient. *)
Definition stage (p r b t u : R) : R := p * (b * u - t) ^ 2 + r * u ^ 2.

(* The certainty-equivalent action for a KNOWN effect b. *)
Definition ce_action (p r b t : R) : R := p * b * t / (p * b ^ 2 + r).

(* Each branch is a convex quadratic: the deviation from its own minimum is exact, not asymptotic.
   Stated with the stationarity condition as a hypothesis rather than with a division, so the
   algebra is a ring identity and the nonzero-denominator side condition appears once, below. *)
Lemma stage_gap :
  forall p r b t u v : R,
    v * (p * b ^ 2 + r) = p * b * t ->
    stage p r b t u - stage p r b t v = (p * b ^ 2 + r) * (u - v) ^ 2.
Proof.
  intros p r b t u v H. unfold stage.
  assert (Hid : p * (b * u - t) ^ 2 + r * u ^ 2 - (p * (b * v - t) ^ 2 + r * v ^ 2)
                - (p * b ^ 2 + r) * (u - v) ^ 2
                = 2 * (u - v) * (v * (p * b ^ 2 + r) - p * b * t)) by ring.
  rewrite H in Hid. ring_simplify in Hid. lra.
Qed.

Lemma ce_action_stationary :
  forall p r b t : R,
    p * b ^ 2 + r <> 0 -> ce_action p r b t * (p * b ^ 2 + r) = p * b * t.
Proof. intros p r b t H. unfold ce_action. field. exact H. Qed.

Lemma ce_action_minimises :
  forall p r b t u : R,
    0 < p * b ^ 2 + r -> stage p r b t (ce_action p r b t) <= stage p r b t u.
Proof.
  intros p r b t u Hpos.
  assert (H := stage_gap p r b t u (ce_action p r b t)
                 (ce_action_stationary p r b t (Rgt_not_eq _ _ Hpos))).
  assert (Hsq : 0 <= (u - ce_action p r b t) ^ 2) by apply pow2_ge_0.
  nra.
Qed.

(* The two endpoint branches cross exactly at u = 0 and at the equalising action u = t/bh. *)
Lemma branch_cross :
  forall p r bh d t u : R,
    stage p r (bh - d) t u - stage p r (bh + d) t u = 4 * d * p * u * (t - bh * u).
Proof. intros. unfold stage. ring. Qed.

(* Below the equalising action the LOWER endpoint is the worse one -- which is what makes the
   worst case there equal to the lower branch, and hence minimised by that branch's minimum. *)
Lemma lower_is_worse :
  forall p r bh d t u : R,
    0 <= d -> 0 <= p -> 0 <= u -> bh * u <= t ->
    stage p r (bh + d) t u <= stage p r (bh - d) t u.
Proof.
  intros p r bh d t u Hd Hp Hu Ht.
  assert (H := branch_cross p r bh d t u).
  (* four nonnegative factors: build the product pairwise, nra only multiplies two at a time. *)
  assert (H1 : 0 <= d * p) by nra.
  assert (H2 : 0 <= u * (t - bh * u)) by nra.
  assert (H3 : 0 <= (d * p) * (u * (t - bh * u))) by nra.
  lra.
Qed.

Definition worst (p r bh d t u : R) : R :=
  Rmax (stage p r (bh - d) t u) (stage p r (bh + d) t u).

(* THE THEOREM. In the regime p*blo*d <= r the CE action for the pessimistic endpoint bh - d is
   the minimax action over the whole interval. The hypothesis Hkink is exactly that regime,
   written as the statement it produces: the action does not overshoot the equalising point. *)
Theorem minimax_lower_branch :
  forall p r bh d t u : R,
    0 <= d -> 0 <= p -> 0 < p * (bh - d) ^ 2 + r ->
    0 <= ce_action p r (bh - d) t ->
    bh * ce_action p r (bh - d) t <= t ->
    worst p r bh d t (ce_action p r (bh - d) t) <= worst p r bh d t u.
Proof.
  intros p r bh d t u Hd Hp Hpos Hnonneg Hkink.
  unfold worst.
  rewrite Rmax_left by (apply lower_is_worse; assumption).
  eapply Rle_trans; [ apply ce_action_minimises; exact Hpos | apply Rmax_l ].
Qed.

(* §33's counterexample, machine-checked: bh = 1, d = 1/2, t = 1, r = 1, p = 1. The CE action 1/2
   has worst case 13/16; the minimax action 2/5 has worst case 4/5 < 13/16. So certainty
   equivalence is strictly suboptimal in the minimax sense for the LQ loss. *)
Lemma minimax_instance_value : worst 1 1 1 (1/2) 1 (2/5) = 4/5.
Proof.
  unfold worst, stage. rewrite Rmax_left; lra.
Qed.

Lemma ce_instance_value : worst 1 1 1 (1/2) 1 (1/2) = 13/16.
Proof.
  unfold worst, stage. rewrite Rmax_left; lra.
Qed.

Theorem ce_not_minimax : worst 1 1 1 (1/2) 1 (2/5) < worst 1 1 1 (1/2) 1 (1/2).
Proof.
  rewrite minimax_instance_value, ce_instance_value. lra.
Qed.

(* The action is 2/5 because that is the CE action for the pessimistic endpoint 1/2 -- the closed
   form, not a number found by search. *)
Lemma minimax_instance_is_lower_ce : ce_action 1 1 (1 - 1/2) 1 = 2/5.
Proof. unfold ce_action. field. Qed.
