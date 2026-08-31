(** The stabilising ball in DELAY space: the algebraic core.

    Designing the decay-optimal gain K^ = g/tauhat and running it against the true delay tau puts
    the loop gain at kappa = K^ tau = g/r, r = tauhat/tau. The stability boundary caps that gain at
    some b > 0 (for the scalar delayed loop, b = PI/2 -- delay_margin.v at pole 0). Everything below
    follows from those two facts and needs nothing else, so the constants stay abstract: what is
    proved is that ANY design whose gain is inversely proportional to the assumed delay has a
    one-sided stabilising set with a RELATIVE radius, not that 2/(PI e) is 0.2342.

    Deliberately out of scope: that the root at g = 1/e is double, and the sqrt-versus-linear split
    of the performance loss that follows from it. Those are statements about a transcendental
    characteristic equation; they live in validation/delay_ball.mac and in
    chc.delay.delay_ball_certificate, which is the same scoping delay_margin.v uses. Stdlib also
    carries no numeric bound on PI, so instantiating g = 1/e and b = PI/2 would need a numeric
    development that would prove nothing extra about the structure. *)

From Stdlib Require Import Reals Lra.
Open Scope R_scope.

Section DelayBall.

Variable g : R.  (** design constant: the assumed-delay-normalised gain *)
Variable b : R.  (** stability boundary on the loop gain *)
Hypothesis g_pos : 0 < g.
Hypothesis b_pos : 0 < b.

Definition loop_gain (r : R) : R := g / r.
Definition ratio_floor : R := g / b.

Lemma ratio_floor_pos : 0 < ratio_floor.
Proof. unfold ratio_floor. apply Rdiv_lt_0_compat; [exact g_pos | exact b_pos]. Qed.

(** THE CRITERION. Stability of the true plant is exactly a lower bound on the delay RATIO. *)
Theorem stabilises_iff_above_floor :
  forall r : R, 0 < r -> (loop_gain r < b <-> ratio_floor < r).
Proof.
  intros r Hr. unfold loop_gain, ratio_floor.
  assert (Hg : 0 < g) by exact g_pos.
  assert (Hb : 0 < b) by exact b_pos.
  assert (E1 : g / r * r = g) by (field; lra).
  assert (E2 : g / b * b = g) by (field; lra).
  split; intros H; nra.
Qed.

(** THE BALL IS ONE-SIDED. Over-estimating the delay never destabilises, at any magnitude: the
    admissible set has a floor and no ceiling. This is what does not survive from the symmetric
    ball in dynamics error, and the reason is visible in the statement -- the gain is antitone in
    the assumed delay while the boundary bounds it from above. *)
Section OneSided.

Hypothesis design_within_boundary : g < b.

Theorem floor_below_one : ratio_floor < 1.
Proof.
  unfold ratio_floor.
  assert (E2 : g / b * b = g) by (field; apply Rgt_not_eq, b_pos).
  assert (Hb : 0 < b) by exact b_pos. nra.
Qed.

Theorem over_estimating_never_destabilises :
  forall r : R, 1 <= r -> loop_gain r < b.
Proof.
  intros r Hr.
  apply stabilises_iff_above_floor; [lra |].
  assert (H := floor_below_one). lra.
Qed.

(** ...and formally there is no upper radius to state: the stable set is unbounded above. *)
Theorem no_upper_radius :
  forall bound : R, exists r : R, bound < r /\ loop_gain r < b.
Proof.
  intros bound. exists (Rmax bound 1 + 1). split.
  - assert (bound <= Rmax bound 1) by apply Rmax_l. lra.
  - apply over_estimating_never_destabilises.
    assert (1 <= Rmax bound 1) by apply Rmax_r. lra.
Qed.

(** The admissible under-estimate is a fraction of the true delay, and a proper one. *)
Definition relative_radius : R := 1 - ratio_floor.

Theorem relative_radius_in_unit_interval : 0 < relative_radius < 1.
Proof.
  unfold relative_radius.
  assert (H1 := floor_below_one). assert (H2 := ratio_floor_pos). lra.
Qed.

End OneSided.

(** The radius is RELATIVE, so it carries no length scale: the shortest safe estimate is a fixed
    fraction of whatever the true delay is, and the fraction does not depend on the plant. *)
Theorem floor_is_scale_free :
  forall tau : R, 0 < tau -> ratio_floor * tau / tau = ratio_floor.
Proof. intros tau Ht. field. apply Rgt_not_eq, Ht. Qed.

End DelayBall.
