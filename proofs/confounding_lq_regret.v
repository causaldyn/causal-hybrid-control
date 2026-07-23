(* Rocq (CVaR -> LQ REGRET): the MSM confounding radius (proofs/confounding_robust_cvar.v) pushed
   through the certainty-equivalence order-doubling to a CONTROL-REGRET bound. In the scalar LQ toy
   (validation/confounding_lq_regret.mac) the regret of the effect-estimate-optimal controller is
   L_reg*(effect error)^2 to leading order -- QUADRATIC in the error (order-doubling). Under Tan's MSM
   the effect error carries an irreducible bias half-width `infl` = the §32 inflation, so the
   confounding-robust regret is `cr_regret L eps infl = L*(eps + infl)^2`. We prove it is nonnegative,
   recovers the pure statistical regret at infl=0 (Gamma=1), is monotone in the confounding, and -- the
   headline -- has a SECOND-ORDER confounding floor `L*infl^2`: a Gamma that biases the EFFECT by `infl`
   biases the REGRET by only `infl^2`, so downstream control is quadratically more robust to confounding
   than effect estimation. Compose `cr_regret_monotone_infl` with §32 `msm_inflation_monotone` for
   monotone-in-Gamma. Algebraic (the CE quadraticity is the cited Mania-Tu-Recht bound, not re-proved). *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

Definition cr_regret (L eps infl : R) : R := L * Rsqr (eps + infl).

(* Nonnegative: a regret bound is a bound (L = regret sensitivity >= 0, square >= 0). *)
Lemma cr_regret_nonneg : forall L eps infl : R, 0 <= L -> 0 <= cr_regret L eps infl.
Proof.
  intros L eps infl HL. unfold cr_regret.
  apply Rmult_le_pos; [ exact HL | apply Rle_0_sqr ].
Qed.

(* Gamma = 1 (infl = 0, point identification): recovers the pure statistical LQ regret L*eps^2. *)
Lemma cr_regret_at_gamma_one : forall L eps : R, cr_regret L eps 0 = L * Rsqr eps.
Proof. intros L eps. unfold cr_regret. rewrite Rplus_0_r. reflexivity. Qed.

(* Large-sample limit (eps = 0): the regret FLOOR is L*infl^2 -- quadratic in the confounding width. *)
Lemma pure_confounding_quadratic : forall L infl : R, cr_regret L 0 infl = L * Rsqr infl.
Proof. intros L infl. unfold cr_regret. rewrite Rplus_0_l. reflexivity. Qed.

(* Decomposition: confounding adds L*(2*eps*infl + infl^2) over the statistical regret L*eps^2. *)
Lemma cr_regret_excess :
  forall L eps infl : R, cr_regret L eps infl = L * Rsqr eps + L * (2 * eps * infl + Rsqr infl).
Proof. intros L eps infl. unfold cr_regret, Rsqr. ring. Qed.

(* Monotone in the assumed confounding: more MSM sensitivity -> a weakly larger regret bound. *)
Lemma cr_regret_monotone_infl :
  forall L eps infl1 infl2 : R,
    0 <= L -> 0 <= eps -> 0 <= infl1 -> infl1 <= infl2 ->
    cr_regret L eps infl1 <= cr_regret L eps infl2.
Proof.
  intros L eps i1 i2 HL He Hi1 Hi12. unfold cr_regret.
  apply Rmult_le_compat_l; [ exact HL |].
  apply Rsqr_incr_1; lra.
Qed.

(* SECOND-ORDER ROBUSTNESS: for a confounding half-width infl in [0,1] the regret floor L*infl^2 is
   below the linear effect-bias floor L*infl -- the order-doubling that damps statistical error also
   damps confounding bias. This is the quotable "control is more robust to confounding" statement. *)
Lemma floor_below_linear :
  forall L infl : R, 0 <= L -> 0 <= infl -> infl <= 1 -> cr_regret L 0 infl <= L * infl.
Proof.
  intros L infl HL Hi Hi1. rewrite pure_confounding_quadratic. unfold Rsqr.
  apply Rmult_le_compat_l; [ exact HL |]. nra.
Qed.
