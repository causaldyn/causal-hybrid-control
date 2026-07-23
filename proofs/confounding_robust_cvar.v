(* Rocq (CONFOUNDING-ROBUST PESSIMISTIC CONTROL): the marginal-sensitivity-model (MSM) worst-case
   causal effect is a CVaR / superquantile mixture, and feeding its Gamma-scaled tail gap into the
   pessimism radius yields a control bound that is (i) never optimistic, (ii) tight under no confounding,
   (iii) monotone in the sensitivity Gamma. Under Tan's MSM the density-ratio weight w has mean 1 and
   lies in [1/Lam, Lam]; the sharp worst-case E[wY] puts w=Lam on the top-tau tail (tau=1/(Lam+1),
   mean-preserving) and w=1/Lam elsewhere, giving the inflation over the point estimate
       infl(Lam) = (Lam-1)/(Lam+1) * (mhi - mlo),   mhi = CVaR_tau(Y) >= mlo.
   The closed forms are checked in validation/confounding_robust_cvar.mac (residual 0); here we prove
   the three control-relevant properties. All algebraic (no confounding assumption is proved -- Gamma is
   the user's unfalsifiable input; this certifies the MAP Gamma -> radius is sound and conservative). *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

Definition msm_inflation (Lam gap : R) : R := gap * ((Lam - 1) / (Lam + 1)).

(* Rewrite the increasing fraction (L-1)/(L+1) as 1 - 2/(L+1) -- makes monotonicity a 1/x argument. *)
Lemma frac_form : forall L : R, L + 1 <> 0 -> (L - 1) / (L + 1) = 1 - 2 / (L + 1).
Proof. intros L H. field. exact H. Qed.

(* (i) NEVER OPTIMISTIC: with Lam >= 1 and a nonneg tail gap, the inflation is >= 0. *)
Lemma msm_inflation_nonneg :
  forall Lam gap : R, 1 <= Lam -> 0 <= gap -> 0 <= msm_inflation Lam gap.
Proof.
  intros Lam gap HL Hg. unfold msm_inflation, Rdiv.
  apply Rmult_le_pos; [ exact Hg |].
  apply Rmult_le_pos; [ lra |].
  left. apply Rinv_0_lt_compat. lra.
Qed.

(* (ii) TIGHT UNDER NO CONFOUNDING: Lam = 1 collapses the bound to the point estimate (infl = 0). *)
Lemma msm_inflation_tight_at_one : forall gap : R, msm_inflation 1 gap = 0.
Proof.
  intros gap. unfold msm_inflation.
  assert (Hz : (1 - 1) / (1 + 1) = 0) by (field; lra).
  rewrite Hz. ring.
Qed.

(* (iii) MONOTONE IN Gamma: more assumed confounding -> a wider (weakly larger) robust radius. *)
Lemma msm_inflation_monotone :
  forall L1 L2 gap : R,
    1 <= L1 -> L1 <= L2 -> 0 <= gap ->
    msm_inflation L1 gap <= msm_inflation L2 gap.
Proof.
  intros L1 L2 gap H1 H12 Hg. unfold msm_inflation.
  apply Rmult_le_compat_l; [ exact Hg |].
  rewrite (frac_form L1) by lra. rewrite (frac_form L2) by lra.
  unfold Rminus. apply Rplus_le_compat_l. apply Ropp_le_contravar.
  unfold Rdiv. apply Rmult_le_compat_l; [ lra | apply Rinv_le_contravar; lra ].
Qed.

(* Control tie-in: the confounding-robust pessimism radius rho0 + infl is never below the nominal rho0
   (pessimism only grows) and inherits monotonicity in Gamma. *)
Definition robust_radius (rho0 Lam gap : R) : R := rho0 + msm_inflation Lam gap.

Lemma robust_radius_ge_nominal :
  forall rho0 Lam gap : R, 1 <= Lam -> 0 <= gap -> rho0 <= robust_radius rho0 Lam gap.
Proof.
  intros rho0 Lam gap HL Hg. unfold robust_radius.
  pose proof (msm_inflation_nonneg Lam gap HL Hg). lra.
Qed.

Lemma robust_radius_monotone :
  forall rho0 L1 L2 gap : R,
    1 <= L1 -> L1 <= L2 -> 0 <= gap ->
    robust_radius rho0 L1 gap <= robust_radius rho0 L2 gap.
Proof.
  intros rho0 L1 L2 gap H1 H12 Hg. unfold robust_radius.
  apply Rplus_le_compat_l. apply msm_inflation_monotone; assumption.
Qed.
