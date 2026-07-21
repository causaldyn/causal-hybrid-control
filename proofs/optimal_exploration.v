(* Rocq: the OPTIMAL EXPLORATION law for causal control -- the actionable dual of the Cramer-Rao lower
   bound (information_lower_bound.v). Injecting exploration variance v buys identifying information but
   costs control: per-round excess = A*v + B/v, with A = b^2+rr (control-cost curvature) and
   B = C*sigma^2/n (the estimation floor). The minimiser is v_star = sqrt(B/A) -- characterised
   sqrt-free by A*v_star^2 = B -- and the excess gap over it is the clean square A*(v - v_star)^2 / v,
   which is nonnegative. So exploration is strictly interior: pure exploitation is never optimal, and
   confounding (a larger B) raises both the optimal exploration and the irreducible cost. Derived in
   validation/optimal_exploration.mac. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

Definition excess (A B v : R) : R := A * v + B / v.  (* per-round explore + estimation excess cost *)

(* Optimality of v_star (characterised by A*v_star^2 = B, i.e. v_star = sqrt(B/A)): the excess gap over
   it equals the sum-of-squares A*(v - v_star)^2 / v >= 0. No sqrt in the statement. *)
Theorem exploration_optimum : forall A B v vstar,
  0 < A -> 0 < v -> 0 < vstar -> A * vstar ^ 2 = B ->
  excess A B vstar <= excess A B v.
Proof.
  intros A B v vstar HA Hv Hvs Hchar. unfold excess.
  assert (Hpos : 0 <= A * (v - vstar) ^ 2 / v).
  { unfold Rdiv. apply Rmult_le_pos.
    - apply Rmult_le_pos; [lra | apply pow2_ge_0].
    - left; apply Rinv_0_lt_compat; exact Hv. }
  assert (Hid : (A * v + B / v) - (A * vstar + B / vstar) = A * (v - vstar) ^ 2 / v).
  { rewrite <- Hchar. field. lra. }
  lra.
Qed.

(* Pure exploitation (v_star = 0) is never optimal: with a positive estimation floor B the
   characterisation A*v_star^2 = B forces v_star <> 0 -- you must explore. *)
Theorem pure_exploitation_suboptimal : forall A B vstar,
  0 < B -> A * vstar ^ 2 = B -> vstar <> 0.
Proof.
  intros A B vstar HB Hchar Heq. subst vstar. nra.
Qed.

(* Confounding raises the estimation floor B; the optimal exploration v_star = sqrt(B/A) is monotone in
   B, so a confounded plant demands MORE exploration. *)
Theorem confounding_raises_exploration : forall A B1 B2 v1 v2,
  0 < A -> 0 < v1 -> 0 < v2 -> B1 <= B2 -> A * v1 ^ 2 = B1 -> A * v2 ^ 2 = B2 ->
  v1 <= v2.
Proof.
  intros A B1 B2 v1 v2 HA Hv1 Hv2 Hle H1 H2. subst B1 B2.
  assert (Hsq : v1 ^ 2 <= v2 ^ 2) by (apply Rmult_le_reg_l with A; [exact HA | exact Hle]).
  nra.
Qed.

(* ... and the irreducible cost floor 2*A*v_star rises with it (ties to Result 10's higher floor). *)
Theorem confounding_raises_floor : forall A v1 v2,
  0 <= A -> v1 <= v2 -> 2 * A * v1 <= 2 * A * v2.
Proof.
  intros A v1 v2 HA Hle. nra.
Qed.
