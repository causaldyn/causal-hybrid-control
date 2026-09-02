(* Rocq: the algebraic core of the ACTION-SIDE VAN TREES BOUND (Result 57).

   validation/action_van_trees.mac derives: the one-step LQ regret is EXACTLY a squared error in
   the ACTION, J(u) - J(u_star) = (b^2+rr)(u - u_star)^2, so no delta method is needed; van Trees applied
   to the estimand psi(b) = u_star(b) -- valid for every estimator, biased or not -- times that exact
   curvature gives a floor whose large-sample limit is Result 10's constant C sigma^2/V_id
   unchanged; the finite-n shortfall is O(1/n); and the knife-edge caveat rr = b^2 survives,
   because there psi'(b) = 0 as a property of the problem.

   Honest scope, as in information_lower_bound.v and van_trees.v: the van Trees inequality itself
   is a measure-theoretic input, cited (Gill-Levit 1995; van der Vaart Thm 2.5.2) and taken as a
   hypothesis exactly as in Result 42. What is proved here is the algebra that turns it into a
   control-regret floor: the exact regret identity, the constant, the quotient's monotonicity and
   limit, and the two sign facts. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Psatz.
Open Scope R_scope.

(* (A) THE EXACT REGRET IDENTITY. For J(u) = (xt + b u)^2 + rr u^2 with u_star = -b xt/(b^2+rr),
   the excess cost is exactly (b^2+rr)(u - u_star)^2 -- for EVERY u, with no linearisation. This is
   what removes Result 10's delta-method caveat: regret is already a squared error in the action,
   so the nonlinearity of u*(b) never has to be approximated. *)
Lemma regret_is_exact_squared_action_error :
  forall b rr xt u : R,
  b * b + rr <> 0 ->
  ((xt + b * u) * (xt + b * u) + rr * (u * u))
  - ((xt + b * (- b * xt / (b * b + rr))) * (xt + b * (- b * xt / (b * b + rr)))
     + rr * ((- b * xt / (b * b + rr)) * (- b * xt / (b * b + rr))))
  = (b * b + rr) * ((u - - b * xt / (b * b + rr)) * (u - - b * xt / (b * b + rr))).
Proof.
  intros b rr xt u H.
  field; exact H.
Qed.

(* (B) THE CONSTANT IS THE ONE RESULT 10 ALREADY CARRIES. With psi = u_star the sensitivity is
   psi' = -xt (rr - b^2)/(rr + b^2)^2, and (b^2 + rr) psi'^2 is exactly
   C = xt^2 (rr - b^2)^2/(rr + b^2)^3. So switching from Cramer-Rao on the effect to van Trees on
   the action costs nothing in the constant. *)
Lemma action_curvature_gives_the_effect_constant :
  forall b rr xt : R,
  rr + b * b <> 0 ->
  (b * b + rr) * ((- xt * (rr - b * b) / ((rr + b * b) * (rr + b * b)))
                  * (- xt * (rr - b * b) / ((rr + b * b) * (rr + b * b))))
  = xt * xt * ((rr - b * b) * (rr - b * b)) / ((rr + b * b) * (rr + b * b) * (rr + b * b)).
Proof.
  intros b rr xt H.
  field; exact H.
Qed.

(* (C) THE VAN TREES QUOTIENT. n k/(n d + i) is below its limit k/d for every n, and the shortfall
   is exactly k i/(d (n d + i)) -- positive and O(1/n). So the prior information costs something
   at finite n and nothing asymptotically, which is the passage to a local-minimax statement. *)
Lemma van_trees_quotient_below_limit :
  forall k d i n : R,
  0 < k -> 0 < d -> 0 <= i -> 0 < n ->
  n * k / (n * d + i) <= k / d.
Proof.
  intros k d i n Hk Hd Hi Hn.
  assert (Hden : 0 < n * d + i) by nra.
  apply Rmult_le_reg_r with (r := (n * d + i) * d); [nra |].
  replace (n * k / (n * d + i) * ((n * d + i) * d)) with (n * k * d) by (field; lra).
  replace (k / d * ((n * d + i) * d)) with (k * (n * d + i)) by (field; lra).
  nra.
Qed.

Lemma van_trees_shortfall_is_explicit :
  forall k d i n : R,
  0 < d -> 0 < n * d + i ->
  k / d - n * k / (n * d + i) = k * i / (d * (n * d + i)).
Proof.
  intros k d i n Hd Hden.
  field; lra.
Qed.

(* (D) CONFOUNDING RAISES THE FLOOR -- now for every estimator, not only unbiased ones. The floor
   C sigma^2/V is antitone in the identifying variance V, and confounding replaces V_exp by the
   residual V_conf <= V_exp. *)
Lemma worse_identification_raises_the_action_floor :
  forall c sigma vexp vconf : R,
  0 < c -> 0 < sigma -> 0 < vconf -> vconf < vexp ->
  c * (sigma * sigma) / vexp < c * (sigma * sigma) / vconf.
Proof.
  intros c sigma vexp vconf Hc Hs Hconf Hlt.
  assert (Hexp : 0 < vexp) by lra.
  assert (Hsq : 0 < sigma * sigma) by nra.
  unfold Rdiv.
  apply Rmult_lt_compat_l; [nra |].
  apply Rinv_lt_contravar; nra.
Qed.

(* (E) THE CAVEAT THAT SURVIVES. At the knife edge rr = b^2 the optimal action is stationary in
   the effect, so the sensitivity -- and with it the leading floor -- vanishes. That is a property
   of the problem and no change of inequality removes it. *)
Lemma knife_edge_kills_the_leading_term :
  forall b xt : R,
  b * b <> 0 ->
  xt * xt * (((b * b) - b * b) * ((b * b) - b * b))
  / (((b * b) + b * b) * ((b * b) + b * b) * ((b * b) + b * b)) = 0.
Proof.
  intros b xt H.
  replace (b * b - b * b) with 0 by ring.
  field; nra.
Qed.
