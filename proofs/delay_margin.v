(* Rocq: the algebraic core of chc.delay's delay margin.

   Plant x'(t) = a x(t) - K x(t - tau); characteristic equation lambda - a + K exp(-lambda tau) = 0.
   Splitting at lambda = i w gives cos(w tau) = a/K and sin(w tau) = w/K, so a crossing exists
   exactly when (a/K, w/K) lies on the unit circle -- which pins w = sqrt(K^2 - a^2).

   SCOPE, stated so the proof is not read as more than it is. What is machine-checked here is that
   algebraic core: the Pythagorean constraint, the positivity of the crossing frequency, and the
   a = 0 specialisation tau_c = pi/(2K) with its monotonicity in the gain. The transcendental half
   -- that arccos(a/K)/sqrt(K^2 - a^2) is the SMALLEST positive crossing, and that it tends to
   1/a as K -> a+ -- is derived in validation/delay_margin.mac and checked numerically in
   chc.delay.delay_margin_certificate. Stdlib Reals only, matching the rest of proofs/. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Rtrigo1.
Open Scope R_scope.

Section DelayMargin.

(* The crossing frequency. Everything below assumes a well-posed loop: the gain exceeds the pole,
   which is exactly the condition for the delay-free loop to be stable in the first place. *)
Variables a K : R.
Hypothesis pole_nonneg : 0 <= a.
Hypothesis gain_dominates : a < K.

Definition w : R := sqrt (K ^ 2 - a ^ 2).

Lemma gain_positive : 0 < K.
Proof. lra. Qed.

Lemma radicand_positive : 0 < K ^ 2 - a ^ 2.
Proof.
  assert (0 < K) by lra.
  assert (a ^ 2 < K ^ 2) by (simpl; nra).
  lra.
Qed.

(* The crossing frequency is real and strictly positive: there IS a frequency to cross at. *)
Lemma crossing_frequency_positive : 0 < w.
Proof.
  unfold w. apply sqrt_lt_R0. apply radicand_positive.
Qed.

(* The load-bearing algebraic identity: a^2 + w^2 = K^2. *)
Lemma pythagorean_crossing : a ^ 2 + w ^ 2 = K ^ 2.
Proof.
  unfold w. rewrite pow2_sqrt.
  - ring.
  - left. apply radicand_positive.
Qed.

(* Equivalently (a/K, w/K) is on the unit circle, which is what makes a simultaneous solution of
   cos(w tau) = a/K and sin(w tau) = w/K possible at all. A pair off the circle admits none. *)
Lemma crossing_pair_on_unit_circle : (a / K) ^ 2 + (w / K) ^ 2 = 1.
Proof.
  assert (HK : 0 < K) by apply gain_positive.
  unfold Rdiv.
  replace ((a * / K) ^ 2 + (w * / K) ^ 2) with ((a ^ 2 + w ^ 2) * (/ K) ^ 2) by ring.
  rewrite pythagorean_crossing. field. lra.
Qed.

(* A crossing frequency can never exceed the gain: the loop cannot oscillate faster than it acts.
   Non-strict on purpose -- at a = 0 the two coincide, which is exactly the K tau = pi/2 case. *)
Lemma crossing_frequency_at_most_gain : w <= K.
Proof.
  assert (HK : 0 < K) by apply gain_positive.
  assert (Hw : 0 < w) by apply crossing_frequency_positive.
  assert (H : w ^ 2 = K ^ 2 - a ^ 2) by (generalize pythagorean_crossing; lra).
  nra.
Qed.

(* Strict as soon as the pole is non-zero: any real pole shrinks the oscillation frequency. *)
Lemma crossing_frequency_below_gain : 0 < a -> w < K.
Proof.
  intro Ha.
  assert (HK : 0 < K) by apply gain_positive.
  assert (Hw : 0 < w) by apply crossing_frequency_positive.
  assert (H : w ^ 2 = K ^ 2 - a ^ 2) by (generalize pythagorean_crossing; lra).
  nra.
Qed.

End DelayMargin.

(* The pure-integrator case, where the margin is the textbook K tau = pi/2. Stated separately
   because it is the one branch with a closed form free of arccos. *)
Definition margin_at_zero_pole (K : R) : R := PI / (2 * K).

Lemma margin_at_zero_pole_gain_product :
  forall K, 0 < K -> K * margin_at_zero_pole K = PI / 2.
Proof.
  intros K HK. unfold margin_at_zero_pole. field. lra.
Qed.

(* More gain always costs delay margin -- the trade-off the certificate exhibits numerically. *)
Lemma margin_at_zero_pole_antitone :
  forall K1 K2, 0 < K1 -> K1 < K2 -> margin_at_zero_pole K2 < margin_at_zero_pole K1.
Proof.
  intros K1 K2 H1 H12. unfold margin_at_zero_pole.
  assert (HPI : 0 < PI) by apply PI_RGT_0.
  assert (H2 : 0 < K2) by lra.
  apply Rmult_lt_reg_r with (r := 2 * K1 * K2).
  - nra.
  - field_simplify; nra.
Qed.

(* And it is always positive, so "the margin" always names a real amount of delay. *)
Lemma margin_at_zero_pole_positive : forall K, 0 < K -> 0 < margin_at_zero_pole K.
Proof.
  intros K HK. unfold margin_at_zero_pole. unfold Rdiv.
  apply Rmult_lt_0_compat; [apply PI_RGT_0 |]. apply Rinv_0_lt_compat. lra.
Qed.
