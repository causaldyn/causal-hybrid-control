(* Rocq: control under PARTIAL IDENTIFICATION -- the sign-identification threshold. Grounded in Manski
   partial ID / Bareinboim (Causal AI, Ch 5.6, optimization-based) and the sensitivity literature
   (Rosenbaum; VanderWeele-Ding). When the effect is only interval-identified, b in [bhat-delta,
   bhat+delta], the optimal action u*(b) = xt*b/(b^2+rr) has the SIGN of b, so the action DIRECTION is
   identified iff the interval excludes 0 -- i.e. iff delta < |bhat|. The critical half-width
   delta* = |bhat| is the SIGN-IDENTIFICATION THRESHOLD (a directional identification margin, the
   uncertainty at which the optimal decision could reverse; deliberately NOT called an "E-value", whose
   name is reserved for the sensitivity-analysis quantity). Derived in
   validation/partial_id_control.mac. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* BELOW the threshold (delta < bhat): every effect consistent with the data keeps the sign, so the
   action direction is robust to the partial-identification uncertainty. *)
Theorem sign_robust_below_threshold : forall bhat delta b,
  0 < bhat -> delta < bhat -> Rabs (b - bhat) <= delta -> 0 < b.
Proof. intros bhat delta b Hb Hd Hint. split_Rabs; lra. Qed.

(* AT or ABOVE the threshold (bhat <= delta): the lower endpoint bhat-delta is consistent with the
   data (within delta of bhat) yet non-positive, so the action direction is no longer identified
   -- the decision can reverse. Hence delta* = bhat is the exact threshold. *)
Theorem decision_reversible_above_threshold : forall bhat delta,
  0 <= bhat -> bhat <= delta ->
  Rabs ((bhat - delta) - bhat) <= delta /\ bhat - delta <= 0.
Proof.
  intros bhat delta Hb Hd. split.
  - split_Rabs; lra.
  - lra.
Qed.
