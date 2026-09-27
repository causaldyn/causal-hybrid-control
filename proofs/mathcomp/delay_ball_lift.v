(* Rocq + MathComp: THE STABILISING BALL IN DELAY SPACE for a multivariate loop (Result 50).

   proofs/delay_ball.v proves the half-line for a scalar loop x' = -K x(t - tau) designed as
   K^ = g / tauhat. This file proves the n-mode version for a design K^ = G / tauhat whose
   normalised gain G = V diag(g) V^-1 is diagonalisable with a positive real spectrum:

   - delayed_feedback_decouples / design_scales_every_mode (any field): in the coordinates
     y = V^-1 x the delayed feedback is diagonal, and running the design against the true delay
     scales every mode's loop gain by the same factor tau / tauhat.
   - stabilises_iff_above_floor (realFieldType): every mode is inside the per-mode boundary b iff
     the delay ratio r = tauhat / tau exceeds max_i g_i / b -- a half-line whose floor is set by the
     FASTEST mode alone. over_estimating_never_destabilises, no_upper_radius,
     relative_radius_in_unit_interval and floor_is_scale_free carry over unchanged, and
     single_mode_floor_is_optimistic says any one mode's scalar floor under-states the true one.

   Honest scope. The per-mode boundary b (pi/2 for the delayed integrator) comes from a
   transcendental characteristic equation and stays abstract, as in the scalar file; so do the
   double root at g = 1/e and the square-root performance loss, which live in
   validation/delay_ball.mac. A G with complex or defective spectrum is not covered: its modes do
   not reduce to scalar delayed integrators. *)

Set Warnings "-notation-overridden,-ambiguous-paths".
From mathcomp Require Import boot order algebra.
Set Implicit Arguments. Unset Strict Implicit. Unset Printing Implicit Defensive.
Import GRing.Theory Num.Theory Order.Theory.
Local Open Scope ring_scope.

(* ---------- decoupling: a diagonalisable delayed gain acts mode by mode ---------- *)
Section Decoupling.
Variable R : fieldType.
Variable n : nat.
Variable V : 'M[R]_n.
Hypothesis hV : V \in unitmx.

Definition modal_gain (lam : 'rV[R]_n) : 'M[R]_n := V *m diag_mx lam *m invmx V.

(* In the coordinates y = V^-1 x, the delayed feedback K x(t - tau) is diagonal: mode i sees only
   its own delayed past, scaled by lam_i. *)
Theorem delayed_feedback_decouples lam (x : 'cV[R]_n) :
  invmx V *m (modal_gain lam *m x) = diag_mx lam *m (invmx V *m x).
Proof. by rewrite /modal_gain !mulmxA mulVmx // mul1mx. Qed.

(* Running the design K^ = G / tauhat against the true delay tau scales EVERY mode's loop gain by
   the same factor tau / tauhat. *)
Theorem design_scales_every_mode (c : R) lam : c *: modal_gain lam = modal_gain (c *: lam).
Proof. by rewrite /modal_gain linearZ /= -scalemxAr -scalemxAl. Qed.

End Decoupling.

(* ---------- the ball in delay space, over n modes ---------- *)
Section DelayBall.
Variable R : realFieldType.
Variable n : nat.
(* The assumed-delay-normalised gain of each mode (the spectrum of G), and the per-mode stability
   boundary on the loop gain (b = pi/2 for the scalar delayed loop, cited from delay_margin.v). *)
Variable g : 'I_n.+1 -> R.
Variable b : R.
Hypotheses (hg : forall i, 0 < g i) (hb : 0 < b).

Definition gmax : R := \big[Num.max/0]_i g i.
Definition stable (r : R) : Prop := forall i, g i / r < b.
Definition ratio_floor : R := gmax / b.

Lemma le_gmax i : g i <= gmax.
Proof. exact: le_bigmax. Qed.

Lemma gmax_attained : exists i0, gmax = g i0.
Proof.
rewrite /gmax; have [i0 _ ->] := eq_bigmax ord0 xpredT g isT (fun i _ => ltW (hg i)).
by exists i0.
Qed.

Lemma gmax_pos : 0 < gmax.
Proof. exact: lt_le_trans (hg ord0) (le_gmax ord0). Qed.

Lemma ratio_floor_pos : 0 < ratio_floor.
Proof. by rewrite /ratio_floor divr_gt0 // gmax_pos. Qed.

(* THE CRITERION, for n modes. Stability of every mode is exactly a lower bound on the delay RATIO,
   and the bound is set by the FASTEST mode alone. *)
Theorem stabilises_iff_above_floor r : 0 < r -> (stable r <-> ratio_floor < r).
Proof.
move=> hr; rewrite /ratio_floor ltr_pdivrMr //; split.
- by move=> hs; have [i0 ->] := gmax_attained; have := hs i0; rewrite ltr_pdivrMr // mulrC.
- move=> hf i; rewrite ltr_pdivrMr // mulrC.
  exact: le_lt_trans (le_gmax i) hf.
Qed.

Section OneSided.
Hypothesis design_within_boundary : gmax < b.

Theorem floor_below_one : ratio_floor < 1.
Proof. by rewrite /ratio_floor ltr_pdivrMr // mul1r. Qed.

(* Over-estimating the delay never destabilises ANY mode, at any magnitude. *)
Theorem over_estimating_never_destabilises r : 1 <= r -> stable r.
Proof.
move=> hr; apply/(stabilises_iff_above_floor (lt_le_trans ltr01 hr)).
exact: lt_le_trans floor_below_one hr.
Qed.

Theorem no_upper_radius (bound : R) : exists r, bound < r /\ stable r.
Proof.
exists (Num.max bound 1 + 1); split.
- by rewrite ltr_pwDr ?ltr01 // le_max lexx.
- apply: over_estimating_never_destabilises.
  by rewrite ler_wpDr ?ler01 // le_max lexx orbT.
Qed.

Definition relative_radius : R := 1 - ratio_floor.

Theorem relative_radius_in_unit_interval : 0 < relative_radius < 1.
Proof.
have h1 := floor_below_one; have h2 := ratio_floor_pos.
by rewrite /relative_radius; apply/andP; split; lra.
Qed.

End OneSided.

(* The radius is RELATIVE: it carries no length scale. *)
Theorem floor_is_scale_free (tau : R) : 0 < tau -> ratio_floor * tau / tau = ratio_floor.
Proof. by move=> ht; rewrite mulfK // lt0r_neq0. Qed.

(* Adding a mode never raises the admissible set: the floor of a subset of the modes is at most the
   floor of all of them, so the scalar floor of any single mode is a LOWER bound on the true one. *)
Theorem single_mode_floor_is_optimistic i : g i / b <= ratio_floor.
Proof. by rewrite /ratio_floor ler_pM2r ?invr_gt0 // le_gmax. Qed.

End DelayBall.
