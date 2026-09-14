(* Rocq: the self-certifying regret bound a plan inside a BOX actually gets (L3.2, Result 69).

   Result 6 gives the unconstrained self-certifying bound J(U) - J* <= |grad J|^2/(2 mu), computed
   from the achieved gradient with no optimum needed.  chc.decision.DecisionCertificate ships with
   no regret_bound because that bound does not survive a box of lever bounds: at a lever the
   gradient holds against its own bound, grad J is nonzero while the true regret is zero.  Result
   13 read the same fact from the other side -- once the cap is active the control freezes and the
   regret's curvature collapses.

   validation/constrained_plan_regret.mac derives what replaces it.  Writing q(d) = -g d - (mu/2)d^2
   for the certified gain of a feasible move d, the bound is the maximum of q over the box's local
   offsets [a, b] = [lo - U, hi - U], and what is proved here is its scalar spine:
     (A) strong convexity turns any upper bound on q over the box into a regret bound;
     (B) the unconstrained maximum of q IS Result 6's bound, and clipping subtracts exactly a
         square -- so the box can only tighten it, never loosen it;
     (C) a lever the gradient holds against a bound contributes EXACTLY zero, at either end;
     (D) the bound is nonnegative, and it is zero at a constrained optimum where Result 6's is not;
     (E) q is below the linear (Frank-Wolfe) gain, whose bound over the box does not mention mu at
         all -- so a modulus that collapses leaves the certificate finite;
     (F) a smaller modulus gives a larger bound, so a conservative one is never invalid;
     (G) the box separates, which is what makes the bound attributable per lever.

   Honest scope: Stdlib has no vectors here, so the per-coordinate statements are scalar and the
   summation over levers is (G) plus arithmetic, carried out in the certificate. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Psatz.
Open Scope R_scope.

(* The certified gain of moving the lever by d, at gradient g under modulus mu. *)
Definition gain (mu g d : R) : R := - (g * d) - (mu / 2) * (d * d).

(* ---- (A) strong convexity turns a bound on the gain into a bound on the regret ---- *)

(* mu-strong convexity read at the constrained optimum: J* >= J + g*ds + (mu/2)ds^2 with
   ds = U* - U a FEASIBLE offset.  Any G that dominates the gain over the box dominates the
   regret, and G never needs to know where U* is. *)
Lemma strong_convexity_turns_a_gain_bound_into_a_regret_bound :
  forall mu g ds cost optimum bound,
    optimum >= cost + g * ds + (mu / 2) * (ds * ds) ->
    gain mu g ds <= bound ->
    cost - optimum <= bound.
Proof.
  intros mu g ds cost optimum bound Hsc Hb. unfold gain in Hb. lra.
Qed.

(* ---- (B) the unconstrained maximum, and what clipping costs ---- *)

Lemma the_unconstrained_maximum_is_result_six_s_bound :
  forall mu g, mu <> 0 -> gain mu g (- g / mu) = g * g / (2 * mu).
Proof.
  intros mu g Hmu. unfold gain. field. exact Hmu.
Qed.

Lemma clipping_costs_exactly_a_square :
  forall mu g d,
    mu <> 0 ->
    gain mu g (- g / mu) - gain mu g d = (mu / 2) * ((d + g / mu) * (d + g / mu)).
Proof.
  intros mu g d Hmu. unfold gain. field. exact Hmu.
Qed.

(* So every feasible move is worth at most Result 6's bound: the box never loosens it. *)
Lemma the_box_only_tightens_the_bound :
  forall mu g d, 0 < mu -> gain mu g d <= g * g / (2 * mu).
Proof.
  intros mu g d Hmu.
  assert (Hne : mu <> 0) by lra.
  assert (Hsquare : forall x : R, 0 <= x * x) by (intros; nra).
  assert (Hsq : 0 <= (mu / 2) * ((d + g / mu) * (d + g / mu))).
  { apply Rmult_le_pos; [lra | apply Hsquare]. }
  rewrite <- (the_unconstrained_maximum_is_result_six_s_bound mu g Hne).
  pose proof (clipping_costs_exactly_a_square mu g d Hne) as Hc. lra.
Qed.

(* ---- (C) a lever the gradient holds against a bound is exactly free ---- *)

Lemma a_still_lever_gains_nothing : forall mu g, gain mu g 0 = 0.
Proof.
  intros mu g. unfold gain. ring.
Qed.

(* At the LOWER bound the feasible offsets are d >= 0, and optimality there means g >= 0. *)
Lemma a_lever_pinned_at_its_floor_is_exactly_free :
  forall mu g d, 0 < mu -> 0 <= g -> 0 <= d -> gain mu g d <= 0.
Proof.
  intros mu g d Hmu Hg Hd. unfold gain. nra.
Qed.

(* At the UPPER bound the feasible offsets are d <= 0, and optimality there means g <= 0. *)
Lemma a_lever_pinned_at_its_ceiling_is_exactly_free :
  forall mu g d, 0 < mu -> g <= 0 -> d <= 0 -> gain mu g d <= 0.
Proof.
  intros mu g d Hmu Hg Hd. unfold gain. nra.
Qed.

(* ---- (D) the bound is nonnegative, and vanishes at a constrained optimum ---- *)

(* d = 0 is feasible whenever U is, so any bound that dominates the gain over the box is >= 0.
   A certificate that could report a negative regret bound would be reporting a sign error. *)
Lemma a_gain_bound_over_the_box_is_nonnegative :
  forall mu g a b bound,
    a <= 0 -> 0 <= b ->
    (forall d, a <= d -> d <= b -> gain mu g d <= bound) ->
    0 <= bound.
Proof.
  intros mu g a b bound Ha Hb H.
  pose proof (H 0 Ha Hb) as H0. rewrite a_still_lever_gains_nothing in H0. lra.
Qed.

(* The worked case of Maxima STEP 6: J(u) = (mu/2)(u - us)^2 with the unconstrained optimum us
   ABOVE the lever's ceiling, so the ceiling IS the constrained optimum and the true regret is 0.
   The gain vanishes there ... *)
Lemma the_constrained_gap_vanishes_at_a_clipped_optimum :
  forall mu ceiling us d,
    0 < mu -> ceiling < us -> d <= 0 ->
    gain mu (mu * (ceiling - us)) d <= 0.
Proof.
  intros mu ceiling us d Hmu Hlt Hd.
  apply a_lever_pinned_at_its_ceiling_is_exactly_free; [lra | nra | exact Hd].
Qed.

(* ... while Result 6's bound charges strictly positive regret to the optimum itself. *)
Lemma result_six_s_bound_charges_regret_at_that_optimum :
  forall mu ceiling us,
    0 < mu -> ceiling < us ->
    0 < (mu * (ceiling - us)) * (mu * (ceiling - us)) / (2 * mu).
Proof.
  intros mu ceiling us Hmu Hlt.
  assert (Hz : mu * (ceiling - us) < 0) by nra.
  apply Rdiv_lt_0_compat; nra.
Qed.

(* ---- (E) the modulus may collapse and the bound stays finite ---- *)

Lemma the_gain_is_below_its_linear_part :
  forall mu g d, 0 <= mu -> gain mu g d <= - (g * d).
Proof.
  intros mu g d Hmu. unfold gain. nra.
Qed.

(* A linear function on an interval is largest at an endpoint -- so the Frank-Wolfe gap bounds the
   gain over the box, and it does not mention mu. *)
Lemma the_frank_wolfe_gap_bounds_the_gain :
  forall mu g a b d,
    0 <= mu -> a <= d -> d <= b ->
    gain mu g d <= Rmax (- (g * a)) (- (g * b)).
Proof.
  intros mu g a b d Hmu Hd1 Hd2.
  eapply Rle_trans; [apply the_gain_is_below_its_linear_part; exact Hmu |].
  destruct (Rle_or_lt g 0) as [Hg | Hg].
  - eapply Rle_trans; [| apply Rmax_r]. nra.
  - eapply Rle_trans; [| apply Rmax_l]. nra.
Qed.

(* The certificate that survives Result 13's collapsing modulus: the SAME finite number bounds the
   gain at every positive modulus, however small. *)
Lemma a_vanishing_modulus_leaves_a_finite_bound :
  forall g a b d,
    a <= d -> d <= b ->
    forall mu, 0 <= mu -> gain mu g d <= Rmax (- (g * a)) (- (g * b)).
Proof.
  intros g a b d Hd1 Hd2 mu Hmu.
  apply the_frank_wolfe_gap_bounds_the_gain; assumption.
Qed.

(* ---- (F) a conservative modulus loosens the bound and never invalidates it ---- *)

Lemma a_smaller_modulus_gives_a_larger_gain :
  forall mu1 mu2 g d, 0 < mu1 -> mu1 <= mu2 -> gain mu2 g d <= gain mu1 g d.
Proof.
  intros mu1 mu2 g d H1 H12. unfold gain. nra.
Qed.

Lemma a_bound_at_the_smaller_modulus_still_holds :
  forall mu1 mu2 g a b bound,
    0 < mu1 -> mu1 <= mu2 ->
    (forall d, a <= d -> d <= b -> gain mu1 g d <= bound) ->
    forall d, a <= d -> d <= b -> gain mu2 g d <= bound.
Proof.
  intros mu1 mu2 g a b bound H1 H12 H d Hd1 Hd2.
  eapply Rle_trans; [apply a_smaller_modulus_gives_a_larger_gain; eassumption | auto].
Qed.

(* ---- (G) the box separates, so the bound is attributable lever by lever ---- *)

Lemma the_box_separates_into_per_lever_terms :
  forall mu g1 g2 d1 d2 bound1 bound2,
    gain mu g1 d1 <= bound1 ->
    gain mu g2 d2 <= bound2 ->
    gain mu g1 d1 + gain mu g2 d2 <= bound1 + bound2.
Proof.
  intros. lra.
Qed.

(* And two levers' worth of strong convexity closes the loop: a separable bound over the rectangle
   is a regret bound for the pair, which is the induction step the certificate iterates. *)
Lemma a_separable_bound_is_a_regret_bound :
  forall mu g1 g2 s1 s2 cost optimum bound1 bound2,
    optimum >= cost + (g1 * s1 + g2 * s2)
               + (mu / 2) * (s1 * s1) + (mu / 2) * (s2 * s2) ->
    gain mu g1 s1 <= bound1 ->
    gain mu g2 s2 <= bound2 ->
    cost - optimum <= bound1 + bound2.
Proof.
  intros mu g1 g2 s1 s2 cost optimum bound1 bound2 Hsc H1 H2.
  unfold gain in H1, H2. lra.
Qed.
