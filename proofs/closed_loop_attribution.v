(* Rocq (WHY b1 IS LARGE AND NEGATIVE): Result 41 ends with "Not established. Why b1 is large and
   negative on a setpoint-tracked zone ... no derivation here fixes the magnitude." These lemmas fix
   it, and refute the guess that stood in for it.

   A proportional tracking loop puts the log on an affine manifold x = c + m*u with m = -1/Kp. On
   that manifold the control-affine fit d + a*x + (b0 + b1*x)*u collapses to a quadratic in u, and
   `interaction_identified` shows the u^2 coefficient pins b1 while `drift_not_identified` exhibits
   an explicit one-parameter family leaving d, a and b0 free. So on a tracked log the INTERACTION is
   the identified coefficient and the pole is not -- the exact inverse of the usual reading.
   `interaction_from_gain` gives the magnitude: b1 = -Kp*C with C the curvature of the response in
   the action, so a tighter loop reports a bigger interaction from the same physics.
   validation/closed_loop_attribution.mac carries the derivation and the numerical anchor. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* The fitted class, and its restriction to the closed-loop manifold x = c + m*u. *)
Definition control_affine (d a b0 b1 x u : R) : R := d + a * x + (b0 + b1 * x) * u.

Definition on_manifold (d a b0 b1 c m u : R) : R :=
  (d + a * c) + (a * m + b0 + b1 * c) * u + (b1 * m) * u ^ 2.

Lemma restriction_is_quadratic :
  forall d a b0 b1 c m u : R,
    control_affine d a b0 b1 (c + m * u) u = on_manifold d a b0 b1 c m u.
Proof. intros. unfold control_affine, on_manifold. ring. Qed.

(* The manifold itself: a proportional loop inverts to an affine relation with slope -1/Kp. *)
Lemma loop_inverts :
  forall kp r u0 x u : R,
    kp <> 0 -> u = kp * (r - x) + u0 -> x = (r + u0 / kp) + (-1 / kp) * u.
Proof.
  intros kp r u0 x u Hkp Hu. rewrite Hu. field. exact Hkp.
Qed.

(* b1 IS identified: it is the only coefficient carrying the u^2 term. *)
Theorem interaction_identified :
  forall b1 m curv : R, m <> 0 -> b1 * m = curv -> b1 = curv / m.
Proof. intros b1 m curv Hm H. rewrite <- H. field. exact Hm. Qed.

(* d, a and b0 are NOT: an explicit one-parameter family with the same restriction. The witness is
   constructive, so this is a non-identification proof rather than a failure to find one. *)
Theorem drift_not_identified :
  forall d a a' b0 b1 c m u : R,
    on_manifold d a b0 b1 c m u
    = on_manifold (d + (a - a') * c) a' (b0 + (a - a') * m) b1 c m u.
Proof. intros. unfold on_manifold. ring. Qed.

(* The magnitude: with the manifold slope -1/Kp the interaction is -Kp times the curvature. *)
Theorem interaction_from_gain :
  forall kp curv : R, kp <> 0 -> curv / (-1 / kp) = - (kp * curv).
Proof. intros kp curv Hkp. field. exact Hkp. Qed.

(* Sign: a CONVEX response under a positive loop gain gives a negative interaction. *)
Theorem interaction_negative_for_convex :
  forall kp curv : R, 0 < kp -> 0 < curv -> - (kp * curv) < 0.
Proof. intros kp curv Hkp Hc. nra. Qed.

(* Magnitude grows with the loop gain: the same physics reported by a tighter tracker looks worse. *)
Theorem magnitude_grows_with_gain :
  forall kp1 kp2 curv : R,
    0 <= curv -> 0 < kp1 -> kp1 <= kp2 -> kp1 * curv <= kp2 * curv.
Proof. intros kp1 kp2 curv Hc Hk1 Hk. nra. Qed.

(* The refutation. b1 = -1/Kp -- the guess plans/24 carried -- is the MANIFOLD slope, not the
   interaction. Equating them constrains the plant to curv = 1/Kp^2, which is a coincidence and not
   an identity: the two agree only on that surface. *)
Theorem guess_is_a_constraint_not_an_identity :
  forall kp curv : R, 0 < kp -> (- (kp * curv) = -1 / kp <-> curv = 1 / kp ^ 2).
Proof.
  intros kp curv Hkp. assert (Hne : kp <> 0) by lra.
  split; intros H.
  - assert (H2 : kp * curv = 1 / kp) by lra.
    apply (Rmult_eq_reg_l kp); [ | exact Hne ].
    rewrite H2. field. exact Hne.
  - rewrite H. field. exact Hne.
Qed.
