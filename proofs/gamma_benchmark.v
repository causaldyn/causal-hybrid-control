(* Rocq (BENCHMARKING Gamma): Result 32 ships Gamma as the analyst's unfalsifiable input. This file
   proves the algebra that makes it *calibrated* anyway, as derived in validation/gamma_benchmark.mac.

   Three groups. (i) The sharp MSM bound collapses: the shipped three-constant expression equals a
   single blend `mu + (1 - 1/g)*(cp - mu)`, which is monotone in g, equals mu at g = 1, and
   saturates at the extreme order statistic -- so a sample lying on one side of zero is
   unreconcilable at ANY g, and `inf` is a theorem rather than a numerical give-up. (ii) The two
   interval endpoints read OPPOSITE tails; `symmetric_reflex_differs` exhibits an instance where the
   symmetric-interval reflex gives the wrong endpoint, which is the bug this file exists to fence
   off. (iii) Odds ratios compose multiplicatively, so "k times as strong as the strongest observed
   covariate" is an EXPONENT: Gamma = Gs^k with k = ln Gamma / ln Gs. `linear_scale_coincides_once`
   is the refutation -- the linear ratio Gamma/Gs is not a weaker convention, it is a different
   claim that agrees with the exponent at exactly one point. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Rpower.
Open Scope R_scope.

(* The bound as the shipped code builds it: tau = 1/(g+1), the complementary (1-tau) mean closes
   the mean constraint, and the inflation scales the CVaR gap by (g-1)/(g+1). *)
Definition tau (g : R) : R := 1 / (g + 1).
Definition complementary (g mu cp : R) : R := (mu - tau g * cp) / (1 - tau g).
Definition upper (g mu cp : R) : R := mu + (g - 1) / (g + 1) * (cp - complementary g mu cp).

(* The same bound as one blend. *)
Definition blend (g mu cp : R) : R := mu + (1 - 1 / g) * (cp - mu).

Lemma complementary_simpl :
  forall g mu cp, 0 < g -> complementary g mu cp = (mu * (g + 1) - cp) / g.
Proof.
  intros g mu cp Hg. unfold complementary, tau.
  assert (Hg1 : g + 1 <> 0) by lra.
  assert (Hden : 1 - 1 / (g + 1) = g / (g + 1)) by (field; exact Hg1).
  rewrite Hden. field. split; lra.
Qed.

Lemma msm_collapse : forall g mu cp, 0 < g -> upper g mu cp = blend g mu cp.
Proof.
  intros g mu cp Hg.
  unfold upper, blend. rewrite (complementary_simpl g mu cp Hg).
  field. split; lra.
Qed.

(* The blend weight: zero at g = 1, rising to 1, never past it. Every bound below is this weight
   multiplied by a tail gap, so it is worth isolating once. *)
Lemma weight_unit_interval : forall g, 1 <= g -> 0 <= 1 - 1 / g <= 1.
Proof.
  intros g Hg.
  assert (Hgpos : 0 < g) by lra.
  assert (Hle : / g <= / 1) by (apply Rinv_le_contravar; lra).
  assert (Hpos : 0 < / g) by (apply Rinv_0_lt_compat; lra).
  rewrite Rinv_1 in Hle. unfold Rdiv. lra.
Qed.

Lemma msm_point_identification : forall mu cp, blend 1 mu cp = mu.
Proof. intros mu cp. unfold blend. lra. Qed.

(* Monotone in g whenever the tail CVaR sits above the mean, which it does for every sample. *)
Lemma msm_monotone :
  forall g1 g2 mu cp, 1 <= g1 -> g1 <= g2 -> mu <= cp -> blend g1 mu cp <= blend g2 mu cp.
Proof.
  intros g1 g2 mu cp H1 H12 Hcp. unfold blend.
  assert (Hg1 : 0 < g1) by lra.
  assert (Hinv : / g2 <= / g1) by (apply Rinv_le_contravar; lra).
  assert (Hw : 1 - 1 / g1 <= 1 - 1 / g2) by (unfold Rdiv; lra).
  apply Rplus_le_compat_l.
  apply Rmult_le_compat_r; lra.
Qed.

(* SATURATION as a bound, which is the form the `inf` return needs: the blend never passes the
   extreme value, so a sample whose bottom tail is strictly positive can never be pulled to zero. *)
Lemma blend_below_extreme :
  forall g mu cp, 1 <= g -> mu <= cp -> blend g mu cp <= cp.
Proof.
  intros g mu cp Hg Hcp. unfold blend.
  destruct (weight_unit_interval g Hg) as [Hw0 Hw1].
  assert (Hstep : (1 - 1 / g) * (cp - mu) <= 1 * (cp - mu))
    by (apply Rmult_le_compat_r; lra).
  lra.
Qed.

(* --- the endpoints read opposite tails --- *)

(* Lower endpoint, obtained by reflecting the sample: cn is the mean of the BOTTOM tau-tail. *)
Definition lower (g mu cn : R) : R := mu - (1 - 1 / g) * (mu - cn).

Lemma endpoint_reflection :
  forall g mu cn, - blend g (- mu) (- cn) = lower g mu cn.
Proof. intros g mu cn. unfold blend, lower. lra. Qed.

(* The symmetric-interval reflex reuses the TOP tail for the lower endpoint. It is not a
   conservative approximation; it is a different number, and it can sit on the wrong side of zero. *)
Definition symmetric_reflex (g mu cp : R) : R := mu - (1 - 1 / g) * (cp - mu).

(* Unreconcilable: a strictly positive bottom tail keeps the lower endpoint strictly positive at
   every sensitivity, so the smallest reconciling Gamma does not exist. *)
Lemma unreconcilable_stays_positive :
  forall g mu cn, 1 <= g -> 0 < cn -> cn <= mu -> 0 < lower g mu cn.
Proof.
  intros g mu cn Hg Hcn Hmu. unfold lower.
  destruct (weight_unit_interval g Hg) as [Hw0 Hw1].
  assert (Hstep : (1 - 1 / g) * (mu - cn) <= 1 * (mu - cn))
    by (apply Rmult_le_compat_r; lra).
  lra.
Qed.

Lemma symmetric_reflex_differs :
  symmetric_reflex 2 1 10 <> lower 2 1 (1 / 2).
Proof. unfold symmetric_reflex, lower. intro H. lra. Qed.

(* And the disagreement is not cosmetic: at this instance the reflex reports a NEGATIVE lower
   endpoint -- "the null is reconciled at Gamma = 2" -- while the true endpoint is positive, and by
   `unreconcilable_stays_positive` no Gamma reconciles it at all. The reflex turns `inf` into 2. *)
Lemma symmetric_reflex_wrong_verdict :
  symmetric_reflex 2 1 10 < 0 /\ 0 < lower 2 1 (1 / 2).
Proof. unfold symmetric_reflex, lower. lra. Qed.

Lemma symmetric_reflex_instance_is_unreconcilable :
  forall g, 1 <= g -> 0 < lower g 1 (1 / 2).
Proof. intros g Hg. apply unreconcilable_stays_positive; lra. Qed.

(* --- the negative control in closed form --- *)

Definition calibrated (mu cn : R) : R := (cn - mu) / cn.

Lemma calibrated_solves :
  forall mu cn, cn < 0 -> 0 < mu -> lower (calibrated mu cn) mu cn = 0.
Proof.
  intros mu cn Hcn Hmu. unfold lower, calibrated.
  assert (Hne : cn - mu <> 0) by lra.
  assert (Hcne : cn <> 0) by lra.
  field_simplify; lra.
Qed.

Lemma calibrated_ge_one :
  forall mu cn, cn < 0 -> 0 < mu -> 1 <= calibrated mu cn.
Proof.
  intros mu cn Hcn Hmu. unfold calibrated.
  assert (Hstep : (cn - mu) / cn - 1 = mu / (- cn)) by (field; lra).
  assert (Hpos : 0 <= mu / (- cn)).
  { apply Rle_mult_inv_pos; lra. }
  lra.
Qed.

(* --- odds ratios compose, so the benchmark scale is logarithmic --- *)

Lemma gamma_composes : forall a b : R, exp (a + b) = exp a * exp b.
Proof. intros a b. apply exp_plus. Qed.

Lemma log_scale_is_additive : forall a b : R, ln (exp (a + b)) = ln (exp a) + ln (exp b).
Proof. intros a b. rewrite !ln_exp. reflexivity. Qed.

(* k = ln G / ln Gs is exactly the exponent that reproduces G from the benchmark. *)
Lemma exponent_is_log_ratio :
  forall gs g : R, 1 < gs -> 0 < g -> Rpower gs (ln g / ln gs) = g.
Proof.
  intros gs g Hgs Hg.
  assert (Hln : ln gs <> 0).
  { intro H. assert (ln 1 < ln gs) by (apply ln_increasing; lra). rewrite ln_1 in *. lra. }
  unfold Rpower.
  replace (ln g / ln gs * ln gs) with (ln g) by (field; exact Hln).
  apply exp_ln; exact Hg.
Qed.

(* REFUTATION. If the benchmark scale were linear, "twice as strong" would be 2*Gs; composition
   says it is Gs^2. The readings agree at Gs = 2 and nowhere else above 1. *)
Lemma linear_scale_coincides_once :
  forall gs : R, 1 < gs -> (2 * gs = gs * gs <-> gs = 2).
Proof.
  intros gs Hgs. split.
  - intro H. nra.
  - intro H. rewrite H. lra.
Qed.
