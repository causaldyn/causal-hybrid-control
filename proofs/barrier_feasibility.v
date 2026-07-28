(* Rocq (Result 40): safety under a PARTIALLY IDENTIFIED control effect.

   Setting (Maxima: validation/barrier_feasibility.mac). Control-affine plant, safe set {h >= 0},
   barrier condition  grad h . (f + B u) >= -alpha*h.  Write a = grad h . f for the drift term,
   g = grad h . B for the control channel, U for the actuation limit, and d for the identification
   radius the effect carries after a marginal sensitivity model at level Gamma (d = Delta(Gamma) *
   ||grad h||; d = 0 at Gamma = 1). The adversary picks the effect inside the identified set, so
   the guaranteed barrier derivative at action u is  a + g*u - d*|u|.

   The robust-CBF tightening itself is standard (Jankovic 2018; Kolathaya-Ames ISSf 2019). What is
   proved here is what that tightening MEANS when d is an identification radius rather than a
   disturbance bound:

     - margin_upper_bound / margin_attained: the best guaranteed margin is exactly a + (g-d)*U on
       the identified side. Pointwise bound plus attainment, which is the constructive form of
       "the max equals the closed form" without quantifying over a continuum.
     - zero_action_optimal_when_unidentified: once d >= g, EVERY action is dominated by u = 0.
       If the sign of the control channel is not identified, the safest thing is to do nothing --
       the safety analogue of the sign-identification threshold in Result 11.
     - certified_below_threshold / violated_above_threshold: feasibility is SHARP at
       d* = g - D/U, with D = -alpha*h - a the deficit the drift leaves for the controller.
     - authority_is_affine_in_radius: the retained control authority falls at a CONSTANT rate U per
       unit of identification radius -- first order, no curvature to hide behind.
     - margin_loss_saturates: and the qualifier that keeps that honest -- the loss is U*min(d,g),
       so "affine with slope U" holds only while control authority lasts. Above d = g the guarantee
       is already gone and the loss is flat at U*g.
     - no_certificate_when_deficit_exceeds_authority: the third regime of the threshold. If the
       drift deficit exceeds U*g the barrier fails at d = 0, so there is no largest admissible
       radius at all and the bare formula g - D/U returns a negative number for an empty set.
     - safety_loss_dominates_regret: and the same effect error costs performance only Lreg*e^2, so
       for every error below U/Lreg the safety loss is the larger of the two, by a ratio that
       diverges as e -> 0. Objectives are protected by the envelope theorem at an interior optimum;
       a binding constraint has no such protection.
     - msm_radius_at_gamma_star: inverting Delta(Gamma) = (Gamma-1)/(Gamma+1)*gap at the threshold
       gives Gamma* = (gap + ds)/(gap - ds) -- the largest confounding level whose safety
       certificate survives.

   Scope: these are the exact SCALAR facts about the tightened condition. The multivariate lift is
   the same statement with g replaced by ||B^T grad h|| (Cauchy-Schwarz, tight -- checked in
   validation/barrier_feasibility.py, not reproved here), and the step from "the pointwise
   condition holds" to "the trajectory stays in the safe set" is Nagumo/Brezis forward invariance,
   assumed as in the CBF literature, not proved. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* The guaranteed barrier derivative at action u: the adversary moves the effect by d against u. *)
Definition robust_margin (a g d u : R) : R := a + g * u - d * Rabs u.

(* --- The identified side: d < g, the channel keeps a usable sign. ------------------------- *)

(* No admissible action beats a + (g-d)*U. *)
Lemma margin_upper_bound :
  forall a g d U u : R,
    0 <= d -> d <= g -> 0 < U -> -U <= u -> u <= U ->
    robust_margin a g d u <= a + (g - d) * U.
Proof.
  intros a g d U u Hd Hdg HU Hlo Hhi. unfold robust_margin.
  assert (Hg : 0 <= g) by lra.
  destruct (Rle_dec 0 u) as [Hpos | Hneg].
  - rewrite Rabs_right by lra.
    assert (Hstep : (g - d) * u <= (g - d) * U) by (apply Rmult_le_compat_l; lra).
    lra.
  - rewrite Rabs_left by lra.
    assert (Hneg2 : (g + d) * u <= 0).
    { replace 0 with ((g + d) * 0) by ring. apply Rmult_le_compat_l; lra. }
    assert (Hnonneg : 0 <= (g - d) * U) by (apply Rmult_le_pos; lra).
    lra.
Qed.

(* And the bound is attained, at the actuation limit -- so it IS the maximum. *)
Lemma margin_attained :
  forall a g d U : R, 0 < U -> robust_margin a g d U = a + (g - d) * U.
Proof.
  intros a g d U HU. unfold robust_margin. rewrite Rabs_right by lra. ring.
Qed.

(* --- The unidentified side: d >= g, the channel's sign is not pinned down. ---------------- *)

(* Every action is dominated by doing nothing. Using control cannot improve the guarantee. *)
Lemma zero_action_optimal_when_unidentified :
  forall a g d u : R, 0 <= g -> g <= d -> robust_margin a g d u <= robust_margin a g d 0.
Proof.
  intros a g d u Hg Hgd. unfold robust_margin.
  rewrite Rabs_R0, Rmult_0_r, Rmult_0_r.
  destruct (Rle_dec 0 u) as [Hpos | Hneg].
  - rewrite Rabs_right by lra.
    assert (H : (g - d) * u <= 0).
    { replace 0 with (0 * u) by ring. apply Rmult_le_compat_r; lra. }
    lra.
  - rewrite Rabs_left by lra.
    assert (H : (g + d) * u <= 0).
    { replace 0 with ((g + d) * 0) by ring. apply Rmult_le_compat_l; lra. }
    lra.
Qed.

(* --- The threshold is sharp. --------------------------------------------------------------- *)

(* Below d* = g - D/U the best action still certifies the barrier condition ... *)
Lemma certified_below_threshold :
  forall a g d U alpha h : R,
    0 < U -> 0 <= d -> d <= g -> d <= g - (- alpha * h - a) / U ->
    - alpha * h <= a + (g - d) * U.
Proof.
  intros a g d U alpha h HU Hd Hdg Hthresh.
  assert (Hcancel : (- alpha * h - a) / U * U = - alpha * h - a) by (field; lra).
  assert (Hstep : (- alpha * h - a) / U <= g - d) by lra.
  apply Rmult_le_compat_r with (r := U) in Hstep; [| lra].
  rewrite Hcancel in Hstep.
  lra.
Qed.

(* ... and above it, no action does: the best guaranteed margin is strictly short. *)
Lemma violated_above_threshold :
  forall a g d U alpha h : R,
    0 < U -> g - (- alpha * h - a) / U < d ->
    a + (g - d) * U < - alpha * h.
Proof.
  intros a g d U alpha h HU Hthresh.
  assert (Hcancel : (- alpha * h - a) / U * U = - alpha * h - a) by (field; lra).
  assert (Hstep : g - d < (- alpha * h - a) / U) by lra.
  apply Rmult_lt_compat_r with (r := U) in Hstep; [| lra].
  rewrite Hcancel in Hstep.
  lra.
Qed.

(* --- First order in the radius, and the contrast with second-order performance regret. ----- *)

(* The retained authority falls at the constant rate U: an affine, curvature-free loss. *)
Lemma authority_is_affine_in_radius :
  forall g U d1 d2 : R, (g - d1) * U - (g - d2) * U = (d2 - d1) * U.
Proof. intros g U d1 d2. ring. Qed.

(* ... but only while there is authority to lose. Relative to a perfectly identified channel the
   margin lost is U*min(d,g): affine with slope U below the kink, flat at U*g above it. Stating
   "affine in the radius, constant slope U" without this qualifier is false globally, and the
   first-order/second-order comparison with Result 33 lives on the low branch. *)
Lemma margin_loss_saturates :
  forall a g d U : R,
    0 <= d -> 0 < U ->
    (d <= g -> (a + (g - 0) * U) - (a + (g - d) * U) = U * d) /\
    (g <= d -> (a + (g - 0) * U) - (a + 0 * U) = U * g).
Proof. intros a g d U Hd HU. split; intros _; ring. Qed.

(* The third regime of the threshold, which the bare formula g - D/U hides. If the drift deficit
   exceeds what a perfectly identified channel delivers at full authority, the barrier is already
   violated at d = 0 -- so the set of admissible radii is EMPTY, and g - D/U is a negative number
   describing nothing. chc.barrier returns nan here rather than a comparable value. *)
Lemma no_certificate_when_deficit_exceeds_authority :
  forall a g U alpha h : R,
    0 < U -> g * U < - alpha * h - a ->
    a + (g - 0) * U < - alpha * h.
Proof. intros a g U alpha h HU Hbig. lra. Qed.

(* The same effect error e costs safety U*e and performance Lreg*e^2. Below e = U/Lreg -- which
   is where any usable controller lives -- safety is the binding loss, and the ratio U/(Lreg*e)
   grows without bound as the error shrinks. *)
Lemma safety_loss_dominates_regret :
  forall U Lreg e : R, 0 < e -> 0 < Lreg -> e < U / Lreg -> Lreg * e * e < U * e.
Proof.
  intros U Lreg e He HL Hsmall.
  assert (Hcancel : U / Lreg * Lreg = U) by (field; lra).
  apply Rmult_lt_compat_r with (r := Lreg) in Hsmall; [| exact HL].
  rewrite Hcancel in Hsmall.
  apply Rmult_lt_compat_r; [ exact He |]. lra.
Qed.

(* --- Inverting the sensitivity model at the threshold. ------------------------------------- *)

(* Result 32's radius is Delta(Gamma) = (Gamma-1)/(Gamma+1)*gap. At Gamma* = (gap+ds)/(gap-ds) it
   equals exactly the threshold radius ds -- so Gamma* is the largest confounding level the safety
   certificate tolerates, and it is finite whenever the design has slack (0 < ds < gap). *)
Lemma msm_radius_at_gamma_star :
  forall gap ds : R,
    0 < ds -> ds < gap ->
    (((gap + ds) / (gap - ds)) - 1) / (((gap + ds) / (gap - ds)) + 1) * gap = ds.
Proof.
  intros gap ds Hds Hlt.
  assert (Hpos : 0 < gap - ds) by lra.
  assert (Hgap : 0 < gap) by lra.
  replace ((gap + ds) / (gap - ds) - 1) with (2 * ds / (gap - ds)) by (field; lra).
  replace ((gap + ds) / (gap - ds) + 1) with (2 * gap / (gap - ds)) by (field; lra).
  field; lra.
Qed.

(* Gamma* exceeds 1, so the threshold is a genuine sensitivity level and not a vacuous one. *)
Lemma gamma_star_above_one :
  forall gap ds : R, 0 < ds -> ds < gap -> 1 < (gap + ds) / (gap - ds).
Proof.
  intros gap ds Hds Hlt.
  assert (Hpos : 0 < gap - ds) by lra.
  assert (Hcancel : (gap + ds) / (gap - ds) * (gap - ds) = gap + ds) by (field; lra).
  apply Rmult_lt_reg_r with (r := gap - ds); [ exact Hpos |].
  rewrite Hcancel.
  lra.
Qed.
