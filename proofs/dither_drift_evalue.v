(* Rocq 9.2: the algebraic core of chc.gate.channel_drift_evalues and chc.gate.DriftAlarm
   (docs/adr/0018-a-channel-drift-monitor.md; the symbolic side is
   validation/dither_drift_evalue.mac). Compile: timeout 300 rocq compile dither_drift_evalue.v

   Honest scope. What is proved here is ALGEBRA. The Gaussian integral
   E exp(a xi + b xi^2) = exp(a^2 / (2 (1 - 2 b))) / sqrt(1 - 2 b), xi ~ N(0, 1), 1 - 2 b > 0,
   enters as the Definition [GaussianQuadMGF], so its use is visible in the type of every theorem
   that needs it. Optional stopping for R_t - t, which turns the averaged Shiryaev-Roberts step
   into an average run length of at least A (Shin, Ramdas and Rinaldo 2024), is cited, not
   formalised; the step itself is proved for a finite-support joint law of the detectors.

   (A) the e-value at a residual c + k xi has mean 1/|1 - theta k|: the nuisance c cancels
   (B) the library's residual at the radius's edge splits as c + k xi, with k free of the dither
   (C) both sides: every column is an e-value while the entry lies inside its radius, and the
       side an entry has crossed has mean above 1
   (D) the detectors' averaged Shiryaev-Roberts statistic grows by at most 1 a decision in
       expectation, and e-CUSUM never exceeds e-SR
   (E) a clipped action, read with its draw (ADR 0021): the mean is at most 1 inside the radius,
       exactly 1 on its edge, and at least 1 past it *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Psatz.
From Stdlib Require Import List.
Import ListNotations.
Open Scope R_scope.

(* ---------------------------------------------------------------------------------------------- *)
(* (A) THE MEAN. *)

Lemma dither_exponent_identity :
  forall th c k xi : R,
  th * (c + k * xi) * xi - th ^ 2 * (c + k * xi) ^ 2 / 2
  = - th ^ 2 * c ^ 2 / 2 + (th * c * (1 - th * k)) * xi + (th * k - th ^ 2 * k ^ 2 / 2) * xi ^ 2.
Proof. intros. field. Qed.

Lemma dither_variance_factor :
  forall th k : R, 1 - 2 * (th * k - th ^ 2 * k ^ 2 / 2) = (1 - th * k) ^ 2.
Proof. intros. field. Qed.

Lemma dither_nuisance_cancels :
  forall th c k : R, 1 - th * k <> 0 ->
  (th * c * (1 - th * k)) ^ 2 / (2 * (1 - th * k) ^ 2) = th ^ 2 * c ^ 2 / 2.
Proof. intros th c k H. field. exact H. Qed.

(* The cited Gaussian integral: a Definition, not an Axiom. *)
Definition GaussianQuadMGF (a b m : R) : Prop :=
  0 < 1 - 2 * b /\ m = exp (a ^ 2 / (2 * (1 - 2 * b))) / sqrt (1 - 2 * b).

Theorem dither_evalue_mean :
  forall th c k m : R, 1 - th * k <> 0 ->
  GaussianQuadMGF (th * c * (1 - th * k)) (th * k - th ^ 2 * k ^ 2 / 2) m ->
  exp (- th ^ 2 * c ^ 2 / 2) * m = 1 / Rabs (1 - th * k).
Proof.
  intros th c k m Hne [_ Hm]. subst m.
  rewrite dither_variance_factor.
  rewrite (dither_nuisance_cancels th c k Hne).
  rewrite <- (Rsqr_pow2 (1 - th * k)), sqrt_Rsqr_abs.
  unfold Rdiv. rewrite <- Rmult_assoc, <- exp_plus.
  replace (- th ^ 2 * c ^ 2 * / 2 + th ^ 2 * c ^ 2 * / 2) with 0 by ring.
  rewrite exp_0. reflexivity.
Qed.

(* Whatever the signs of the bet and of k, a product th k <= 0 makes the mean at most 1. *)
Theorem dither_evalue_valid :
  forall th c k m : R, th * k <= 0 ->
  GaussianQuadMGF (th * c * (1 - th * k)) (th * k - th ^ 2 * k ^ 2 / 2) m ->
  exp (- th ^ 2 * c ^ 2 / 2) * m <= 1.
Proof.
  intros th c k m Hk Hmgf.
  assert (Hne : 1 - th * k <> 0) by (intro; lra).
  rewrite (dither_evalue_mean th c k m Hne Hmgf).
  rewrite Rabs_right by lra.
  apply (Rmult_le_reg_r (1 - th * k)); [lra |].
  replace (1 / (1 - th * k) * (1 - th * k)) with 1 by (field; exact Hne).
  lra.
Qed.

(* On the side an entry has crossed, 0 < th k < 1, the mean exceeds 1: that excess is the power. *)
Theorem dither_evalue_grows_on_alternative :
  forall th c k m : R, 0 < th * k < 1 ->
  GaussianQuadMGF (th * c * (1 - th * k)) (th * k - th ^ 2 * k ^ 2 / 2) m ->
  1 < exp (- th ^ 2 * c ^ 2 / 2) * m.
Proof.
  intros th c k m Hk Hmgf.
  assert (Hne : 1 - th * k <> 0) by (intro; lra).
  rewrite (dither_evalue_mean th c k m Hne Hmgf).
  rewrite Rabs_right by lra.
  apply (Rmult_lt_reg_r (1 - th * k)); [lra |].
  replace (1 / (1 - th * k) * (1 - th * k)) with 1 by (field; exact Hne).
  lra.
Qed.

(* ---------------------------------------------------------------------------------------------- *)
(* (B) THE LIBRARY'S RESIDUAL. u = p + sig xi; the residual is g + d u + rest, with d the entry's
   distance from the model's and rest free of xi; the library moves it to the edge, subtracting
   side eps u, and divides by the residual scale sc. *)

Lemma edge_residual_split :
  forall g d p sig xi rest side eps sc : R, sc <> 0 ->
  (g + d * (p + sig * xi) + rest - side * eps * (p + sig * xi)) / sc
  = (g + (d - side * eps) * p + rest) / sc + ((d - side * eps) * sig / sc) * xi.
Proof. intros. field. exact H. Qed.

(* ---------------------------------------------------------------------------------------------- *)
(* (C) BOTH SIDES. The bet is side th, th >= 0: side 1 against growth, side -1 against shrinkage. *)

Theorem bet_times_k_inside_the_radius :
  forall th d eps sig sc side : R,
  0 <= th -> 0 < sig -> 0 < sc -> Rabs d <= eps -> side = 1 \/ side = -1 ->
  (side * th) * ((d - side * eps) * sig / sc) <= 0.
Proof.
  intros th d eps sig sc side Hth Hsig Hsc Hd Hside.
  assert (Hq : 0 <= th * sig / sc)
    by (apply Rmult_le_pos; [nra | left; apply Rinv_0_lt_compat; lra]).
  assert (Hup : d <= eps) by (pose proof (Rle_abs d); lra).
  assert (Hdown : - eps <= d) by (pose proof (Rle_abs (- d)); rewrite Rabs_Ropp in *; lra).
  destruct Hside as [-> | ->].
  - replace (1 * th * ((d - 1 * eps) * sig / sc)) with ((d - eps) * (th * sig / sc))
      by (field; lra).
    nra.
  - replace (-1 * th * ((d - -1 * eps) * sig / sc)) with (- (d + eps) * (th * sig / sc))
      by (field; lra).
    nra.
Qed.

Theorem every_column_is_an_evalue_inside_the_radius :
  forall th d eps sig sc side c m : R,
  0 <= th -> 0 < sig -> 0 < sc -> Rabs d <= eps -> side = 1 \/ side = -1 ->
  GaussianQuadMGF ((side * th) * c * (1 - (side * th) * ((d - side * eps) * sig / sc)))
                  ((side * th) * ((d - side * eps) * sig / sc)
                   - (side * th) ^ 2 * ((d - side * eps) * sig / sc) ^ 2 / 2) m ->
  exp (- (side * th) ^ 2 * c ^ 2 / 2) * m <= 1.
Proof.
  intros th d eps sig sc side c m Hth Hsig Hsc Hd Hside Hmgf.
  apply (dither_evalue_valid (side * th) c ((d - side * eps) * sig / sc) m); [| exact Hmgf].
  exact (bet_times_k_inside_the_radius th d eps sig sc side Hth Hsig Hsc Hd Hside).
Qed.

(* ---------------------------------------------------------------------------------------------- *)
(* (D) THE ALARM. A finite-support joint law of the detectors' e-values: pairs of a probability and
   the vector of e-values, detector i's being [e i]. *)

Fixpoint expect (l : list (R * (nat -> R))) (f : (nat -> R) -> R) : R :=
  match l with
  | [] => 0
  | (p, e) :: t => p * f e + expect t f
  end.

Fixpoint total (n : nat) (g : nat -> R) : R :=
  match n with
  | O => 0
  | S n' => total n' g + g n'
  end.

Lemma total_zero : forall n : nat, total n (fun _ => 0) = 0.
Proof. induction n as [| n IH]; simpl; [reflexivity | rewrite IH; ring]. Qed.

Lemma total_plus :
  forall (n : nat) (a b : nat -> R), total n (fun i => a i + b i) = total n a + total n b.
Proof. induction n as [| n IH]; intros a b; simpl; [ring | rewrite IH; ring]. Qed.

Lemma total_scale :
  forall (n : nat) (p : R) (a : nat -> R), total n (fun i => p * a i) = p * total n a.
Proof. induction n as [| n IH]; intros p a; simpl; [ring | rewrite IH; ring]. Qed.

Lemma total_le :
  forall (n : nat) (a b : nat -> R),
  (forall i, (i < n)%nat -> a i <= b i) -> total n a <= total n b.
Proof.
  induction n as [| n IH]; intros a b Hab; simpl; [lra |].
  assert (H1 : total n a <= total n b) by (apply IH; intros i Hi; apply Hab; lia).
  assert (H2 : a n <= b n) by (apply Hab; lia).
  lra.
Qed.

Lemma expect_total :
  forall (l : list (R * (nat -> R))) (n : nat) (h : (nat -> R) -> nat -> R),
  expect l (fun e => total n (h e)) = total n (fun i => expect l (fun e => h e i)).
Proof.
  induction l as [| [p e] t IH]; intros n h; simpl.
  - symmetry. apply total_zero.
  - rewrite IH, total_plus, total_scale. reflexivity.
Qed.

Lemma expect_scale :
  forall (l : list (R * (nat -> R))) (a : R) (f : (nat -> R) -> R),
  expect l (fun e => a * f e) = a * expect l f.
Proof. induction l as [| [p e] t IH]; intros a f; simpl; [ring | rewrite IH; ring]. Qed.

(* R_i' = (R_i + 1) e_i with E[e_i | past] <= 1 and R_i >= 0: the sum over detectors, hence their
   average, grows by at most one a decision in expectation, under ANY joint law of the e-values.
   So the average less t is a supermartingale. *)
Theorem averaged_sr_step :
  forall (l : list (R * (nat -> R))) (n : nat) (r : nat -> R),
  (forall i, (i < n)%nat -> expect l (fun e => e i) <= 1) -> (forall i, 0 <= r i) ->
  expect l (fun e => total n (fun i => (r i + 1) * e i)) <= total n (fun i => r i + 1).
Proof.
  intros l n r Hmean Hr.
  rewrite (expect_total l n (fun e i => (r i + 1) * e i)).
  apply total_le. intros i Hi.
  rewrite (expect_scale l (r i + 1) (fun e => e i)).
  specialize (Hmean i Hi). specialize (Hr i). nra.
Qed.

(* CUSUM's M' = max(M, 1) e stays below SR's R' = (R + 1) e once M <= R: at a common threshold SR
   alarms no later than CUSUM on every path. *)
Theorem cusum_below_sr :
  forall M r e : R, 0 <= M <= r -> 0 <= e -> Rmax M 1 * e <= (r + 1) * e.
Proof.
  intros M r e HM He.
  apply Rmult_le_compat_r; [exact He |].
  apply Rmax_lub; lra.
Qed.

(* ---------------------------------------------------------------------------------------------- *)
(* (E) A CLIPPED ACTION, READ WITH ITS DRAW (ADR 0021). A box clips the applied dither to
   h(xi) = clip(xi, -m1, m2) in dither units, so rt = c + k h(xi), while the bet reads the draw xi.
   Pointwise, e phi(xi) = phi(xi - a) with a = th rt, a that moves with xi included. Between the
   clips a = th (c + k xi), so xi - a = s xi - th c with s = 1 - th k; beyond them a is constant.
   The three regions' Gaussian integrals, Phi(u1), (Phi(u2) - Phi(u1)) / s and 1 - Phi(u2), with
   u1 = -s m1 - th c and u2 = s m2 - th c, are STEP 8d of validation/dither_drift_evalue.mac and
   enter as the Definition [ClippedMean]: p1 and p2 stand for Phi(u1) <= Phi(u2). *)

Lemma tilted_density_exponent :
  forall a xi : R, a * xi - a ^ 2 / 2 - xi ^ 2 / 2 = - (xi - a) ^ 2 / 2.
Proof. intros. field. Qed.

Lemma clipped_middle_shift :
  forall th c k xi : R, xi - th * (c + k * xi) = (1 - th * k) * xi - th * c.
Proof. intros. ring. Qed.

Lemma clipped_lower_edge :
  forall th c k m1 : R, - m1 - th * (c + k * - m1) = - (1 - th * k) * m1 - th * c.
Proof. intros. ring. Qed.

Lemma clipped_upper_edge :
  forall th c k m2 : R, m2 - th * (c + k * m2) = (1 - th * k) * m2 - th * c.
Proof. intros. ring. Qed.

(* The cited region integrals: a Definition, not an Axiom. *)
Definition ClippedMean (s p1 p2 m : R) : Prop :=
  0 <= p1 <= p2 /\ p2 <= 1 /\ m = p1 + (p2 - p1) / s + (1 - p2).

Lemma clipped_mean_identity :
  forall s p1 p2 m : R, s <> 0 -> ClippedMean s p1 p2 m -> m = 1 - (p2 - p1) * (1 - 1 / s).
Proof. intros s p1 p2 m Hs [_ [_ Hm]]. subst m. field. exact Hs. Qed.

(* Inside the radius, th k <= 0 (Theorem bet_times_k_inside_the_radius), so s >= 1: the clipped
   decision's mean is at most 1, whatever the clip, c and the nominal action. *)
Theorem clipped_evalue_valid :
  forall th k p1 p2 m : R, th * k <= 0 -> ClippedMean (1 - th * k) p1 p2 m -> m <= 1.
Proof.
  intros th k p1 p2 m Hk Hmean.
  assert (Hs : 1 - th * k <> 0) by lra.
  rewrite (clipped_mean_identity (1 - th * k) p1 p2 m Hs Hmean).
  destruct Hmean as [[Hp1 Hp12] _].
  assert (Hinv : 1 / (1 - th * k) <= 1).
  { apply (Rmult_le_reg_r (1 - th * k)); [lra |].
    replace (1 / (1 - th * k) * (1 - th * k)) with 1 by (field; exact Hs). nra. }
  nra.
Qed.

(* On the side an entry has crossed, 0 < th k < 1, the mean is at least 1: a clip shrinks the
   power, and never turns it against the alarm. *)
Theorem clipped_evalue_keeps_its_side :
  forall th k p1 p2 m : R, 0 < th * k < 1 -> ClippedMean (1 - th * k) p1 p2 m -> 1 <= m.
Proof.
  intros th k p1 p2 m Hk Hmean.
  assert (Hs : 1 - th * k <> 0) by lra.
  rewrite (clipped_mean_identity (1 - th * k) p1 p2 m Hs Hmean).
  destruct Hmean as [[Hp1 Hp12] _].
  assert (Hinv : 1 <= 1 / (1 - th * k)).
  { apply (Rmult_le_reg_r (1 - th * k)); [lra |].
    replace (1 / (1 - th * k) * (1 - th * k)) with 1 by (field; exact Hs). nra. }
  nra.
Qed.

(* No clip, p1 = 0 and p2 = 1: the mean is Theorem dither_evalue_mean's 1/s. *)
Lemma clipped_mean_without_a_clip :
  forall s m : R, s <> 0 -> ClippedMean s 0 1 m -> m = 1 / s.
Proof. intros s m Hs [_ [_ Hm]]. subst m. field. exact Hs. Qed.

(* On the radius's edge, k = 0 and s = 1: the mean is exactly 1, whatever the clip. *)
Lemma clipped_mean_on_the_edge :
  forall p1 p2 m : R, ClippedMean 1 p1 p2 m -> m = 1.
Proof. intros p1 p2 m [_ [_ Hm]]. subst m. field. Qed.
