(* Rocq: control under PARTIAL IDENTIFICATION and the "control E-value". Grounded in Bareinboim
   (Causal AI, Ch 5.6, optimization-based / partial ID) and the sensitivity literature (Rosenbaum;
   VanderWeele-Ding E-value). When the effect is only interval-identified, b in [bhat-delta,
   bhat+delta], the optimal action u*(b) = xt*b/(b^2+rr) has the SIGN of b, so the action DIRECTION is
   identified iff the interval excludes 0 -- i.e. iff delta < |bhat|. The critical half-width
   delta* = |bhat| is the CONTROL E-VALUE: the identification uncertainty at which the optimal decision
   could reverse. A decision-relevant sensitivity measure (derived in validation/partial_id_control.mac). *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* BELOW the control E-value (delta < bhat): every effect consistent with the data keeps the sign, so
   the action direction is robust to the partial-identification uncertainty. *)
Theorem sign_robust_below_control_evalue : forall bhat delta b,
  0 < bhat -> delta < bhat -> Rabs (b - bhat) <= delta -> 0 < b.
Proof. intros bhat delta b Hb Hd Hint. split_Rabs; lra. Qed.

(* AT or ABOVE the control E-value (bhat <= delta): the lower endpoint bhat-delta is consistent with
   the data (within delta of bhat) yet non-positive, so the action direction is no longer identified
   -- the decision can reverse. Hence delta* = bhat is the exact threshold. *)
Theorem decision_reversible_above_control_evalue : forall bhat delta,
  0 <= bhat -> bhat <= delta ->
  Rabs ((bhat - delta) - bhat) <= delta /\ bhat - delta <= 0.
Proof.
  intros bhat delta Hb Hd. split.
  - split_Rabs; lra.
  - lra.
Qed.
