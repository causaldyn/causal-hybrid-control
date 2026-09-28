(* Rocq 9.2: the algebraic core of chc.plan.CausalPlan.decision_weight
   (the symbolic side is validation/plan_decision_weight.mac).
   Compile: timeout 300 rocq compile plan_decision_weight.v

   Honest scope. What is proved here is the algebra of an objective quadratic in the actions, with
   the channel's error entering as a push k on them. A rollout through a plant makes the objective
   quadratic in the actions only when the plant is linear in them, and even then the push is
   linear in the error only to first order: that W is the regret's second derivative for the
   library's rollout is Maxima's STEPs 2-4 and the unit tests', not this file's. Two free actions
   stand for any number: the statements are the scalar and two-dimensional cases of
   k' M^-1 k and of the Cauchy-Schwarz inequality in M^-1's inner product.

   (A) the envelope: a plan that ignored the push k loses exactly k^2 / (2 m) to the one that knew
       it, and k' M^-1 k / 2 with two free actions
   (B) pinning an action can only lower the weight: k' M^-1 k >= k1^2 / m11
   (C) a binding row can only lower it: (v'k)^2 / (v'Mv) <= k' M^-1 k for any direction v
   (D) the one-step closed form: J_ue^2 / J_uu at the plan is
       qf^2 (r - gam^2 qf)^2 (phi x0 - s)^2 / (r + gam^2 qf)^3 *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Psatz.
Open Scope R_scope.

(* field's side conditions: each denominator is a positive quantity in the context *)
Ltac nonzero := repeat split; apply Rgt_not_eq; nra.

(* ---------------------------------------------------------------------------------------------- *)
(* (A) THE ENVELOPE. The objective at one value of the channel's error: m u^2 / 2 + (g + k) u + h,
   with k the error's push on the action and h whatever the error does to the rest. *)

Definition J1 (m g k h u : R) : R := m * u ^ 2 / 2 + (g + k) * u + h.

Lemma scalar_plan_is_the_minimiser :
  forall m g k h u : R, 0 < m -> J1 m g k h (- (g + k) / m) <= J1 m g k h u.
Proof.
  intros m g k h u Hm. unfold J1.
  assert (Hid : m * u ^ 2 / 2 + (g + k) * u + h
                - (m * (- (g + k) / m) ^ 2 / 2 + (g + k) * (- (g + k) / m) + h)
                = m * (u + (g + k) / m) ^ 2 / 2) by (field; nonzero).
  assert (0 <= m * (u + (g + k) / m) ^ 2 / 2).
  { pose proof (pow2_ge_0 (u + (g + k) / m)). unfold Rdiv.
    apply Rmult_le_pos; [nra | left; apply Rinv_0_lt_compat; lra]. }
  lra.
Qed.

Theorem envelope_scalar :
  forall m g k h : R, m <> 0 ->
  J1 m g k h (- g / m) - J1 m g k h (- (g + k) / m) = k ^ 2 / (2 * m).
Proof. intros m g k h Hm. unfold J1. field. exact Hm. Qed.

(* Two free actions, M = [[a, b], [b, d]]: the plan -M^-1 p for a push p. *)
Definition J2 (a b d p1 p2 h u1 u2 : R) : R :=
  (a * u1 ^ 2 + 2 * b * u1 * u2 + d * u2 ^ 2) / 2 + p1 * u1 + p2 * u2 + h.

Definition plan1 (a b d p1 p2 : R) : R := - (d * p1 - b * p2) / (a * d - b ^ 2).
Definition plan2 (a b d p1 p2 : R) : R := - (a * p2 - b * p1) / (a * d - b ^ 2).

Lemma two_plan_is_stationary :
  forall a b d p1 p2 : R, a * d - b ^ 2 <> 0 ->
  a * plan1 a b d p1 p2 + b * plan2 a b d p1 p2 + p1 = 0
  /\ b * plan1 a b d p1 p2 + d * plan2 a b d p1 p2 + p2 = 0.
Proof. intros a b d p1 p2 Hdet. unfold plan1, plan2. split; field; exact Hdet. Qed.

Definition weight2 (a b d k1 k2 : R) : R :=
  (d * k1 ^ 2 - 2 * b * k1 * k2 + a * k2 ^ 2) / (a * d - b ^ 2).

Theorem envelope_two :
  forall a b d g1 g2 k1 k2 h : R, a * d - b ^ 2 <> 0 ->
  J2 a b d (g1 + k1) (g2 + k2) h (plan1 a b d g1 g2) (plan2 a b d g1 g2)
  - J2 a b d (g1 + k1) (g2 + k2) h (plan1 a b d (g1 + k1) (g2 + k2)) (plan2 a b d (g1 + k1) (g2 + k2))
  = weight2 a b d k1 k2 / 2.
Proof. intros a b d g1 g2 k1 k2 h Hdet. unfold J2, plan1, plan2, weight2. field. exact Hdet. Qed.

(* ---------------------------------------------------------------------------------------------- *)
(* (B) PINNING. With the second action held, the first alone answers the push: weight k1^2 / a. *)

Theorem pinning_lowers_the_weight :
  forall a b d k1 k2 : R, 0 < a -> 0 < a * d - b ^ 2 ->
  k1 ^ 2 / a <= weight2 a b d k1 k2.
Proof.
  intros a b d k1 k2 Ha Hdet. unfold weight2.
  assert (Hid : (d * k1 ^ 2 - 2 * b * k1 * k2 + a * k2 ^ 2) / (a * d - b ^ 2) - k1 ^ 2 / a
                = (a * k2 - b * k1) ^ 2 / (a * (a * d - b ^ 2))) by (field; nonzero).
  assert (0 <= (a * k2 - b * k1) ^ 2 / (a * (a * d - b ^ 2))).
  { unfold Rdiv. apply Rmult_le_pos; [apply pow2_ge_0 |].
    left. apply Rinv_0_lt_compat. nra. }
  lra.
Qed.

(* ---------------------------------------------------------------------------------------------- *)
(* (C) A BINDING ROW. The plan may move only along v, so it answers the push with weight
   (v'k)^2 / (v'Mv): no more than with both actions free. *)

Lemma positive_definite_form :
  forall a b d v1 v2 : R, 0 < a -> 0 < a * d - b ^ 2 -> (v1 <> 0 \/ v2 <> 0) ->
  0 < a * v1 ^ 2 + 2 * b * v1 * v2 + d * v2 ^ 2.
Proof.
  intros a b d v1 v2 Ha Hdet Hv.
  assert (Hsq : a * (a * v1 ^ 2 + 2 * b * v1 * v2 + d * v2 ^ 2)
                = (a * v1 + b * v2) ^ 2 + (a * d - b ^ 2) * v2 ^ 2) by ring.
  assert (Hpos : 0 < (a * v1 + b * v2) ^ 2 + (a * d - b ^ 2) * v2 ^ 2).
  { pose proof (pow2_ge_0 (a * v1 + b * v2)) as Hs.
    destruct (Req_dec v2 0) as [Hz | Hv2].
    - subst v2. destruct Hv as [Hv1 | Hv2]; [| exfalso; apply Hv2; reflexivity].
      assert (Hav : a * v1 + b * 0 <> 0).
      { replace (a * v1 + b * 0) with (a * v1) by ring.
        apply Rmult_integral_contrapositive_currified; [lra | exact Hv1]. }
      assert (0 < (a * v1 + b * 0) ^ 2) by (rewrite <- Rsqr_pow2; apply Rsqr_pos_lt; exact Hav).
      replace ((a * d - b ^ 2) * 0 ^ 2) with 0 by ring.
      lra.
    - assert (0 < v2 ^ 2) by (rewrite <- Rsqr_pow2; apply Rsqr_pos_lt; exact Hv2).
      assert (0 < (a * d - b ^ 2) * v2 ^ 2) by (apply Rmult_lt_0_compat; assumption).
      lra. }
  rewrite <- Hsq in Hpos.
  apply (Rmult_lt_reg_l a); [exact Ha | lra].
Qed.

Theorem binding_row_lowers_the_weight :
  forall a b d k1 k2 v1 v2 : R, 0 < a -> 0 < a * d - b ^ 2 -> (v1 <> 0 \/ v2 <> 0) ->
  (v1 * k1 + v2 * k2) ^ 2 / (a * v1 ^ 2 + 2 * b * v1 * v2 + d * v2 ^ 2) <= weight2 a b d k1 k2.
Proof.
  intros a b d k1 k2 v1 v2 Ha Hdet Hv.
  pose proof (positive_definite_form a b d v1 v2 Ha Hdet Hv) as Hq.
  unfold weight2.
  assert (Hid : (d * k1 ^ 2 - 2 * b * k1 * k2 + a * k2 ^ 2) / (a * d - b ^ 2)
                - (v1 * k1 + v2 * k2) ^ 2 / (a * v1 ^ 2 + 2 * b * v1 * v2 + d * v2 ^ 2)
                = (b * k2 * v2 - d * k1 * v2 + a * k2 * v1 - b * k1 * v1) ^ 2
                  / ((a * d - b ^ 2) * (a * v1 ^ 2 + 2 * b * v1 * v2 + d * v2 ^ 2)))
    by (field; nonzero).
  assert (0 <= (b * k2 * v2 - d * k1 * v2 + a * k2 * v1 - b * k1 * v1) ^ 2
               / ((a * d - b ^ 2) * (a * v1 ^ 2 + 2 * b * v1 * v2 + d * v2 ^ 2))).
  { unfold Rdiv. apply Rmult_le_pos; [apply pow2_ge_0 |].
    left. apply Rinv_0_lt_compat. nra. }
  lra.
Qed.

(* ---------------------------------------------------------------------------------------------- *)
(* (D) ONE STEP. x1 = phi x0 + (gam + e) u and J = r u^2 / 2 + qf (x1 - s)^2 / 2 (plus terms free of
   u): J_uu = r + qf gam^2 and J_ue = qf ((phi x0 + gam u - s) + gam u) at e = 0 (Maxima's STEP 2),
   evaluated at the plan u* = -gam qf (phi x0 - s) / (r + gam^2 qf). *)

Theorem one_step_weight :
  forall r qf gam phi x0 s : R, 0 < r + gam ^ 2 * qf ->
  (qf * ((phi * x0 + gam * (- gam * qf * (phi * x0 - s) / (r + gam ^ 2 * qf)) - s)
         + gam * (- gam * qf * (phi * x0 - s) / (r + gam ^ 2 * qf)))) ^ 2
  / (r + qf * gam ^ 2)
  = qf ^ 2 * (r - gam ^ 2 * qf) ^ 2 * (phi * x0 - s) ^ 2 / (r + gam ^ 2 * qf) ^ 3.
Proof. intros r qf gam phi x0 s Hpos. field. nonzero. Qed.

(* The knife edge: at r = gam^2 qf the one-step weight is zero. *)
Corollary one_step_knife_edge :
  forall qf gam phi x0 s : R, 0 < gam ^ 2 * qf ->
  qf ^ 2 * (gam ^ 2 * qf - gam ^ 2 * qf) ^ 2 * (phi * x0 - s) ^ 2
  / (gam ^ 2 * qf + gam ^ 2 * qf) ^ 3 = 0.
Proof. intros qf gam phi x0 s Hpos. field. nonzero. Qed.
