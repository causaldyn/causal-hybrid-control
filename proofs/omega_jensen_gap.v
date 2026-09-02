(* Rocq: the algebraic core of the OMEGA-JENSEN gap (Result 51 (k)-(m)).

   validation/omega_jensen_gap.mac derives, for theta_hat = u'A eps / u'A u with u ~ N(0, Om):

     E[X/Y^2] / (tr(B Om)/tr(C Om)^2) - 1  =  -4 tr(B Om C Om)/(tr(B Om) tr(C Om))
                                              + 6 tr(C Om C Om)/tr(C Om)^2

   (B = A'Sigma A, C = A'A), an affine kurtosis transport for elliptical u, and -- via the
   resolvent representation -- that at Om = I the plug-in design crossover is EXACT: the two fold
   operators share their spectrum, P_fold + P_within = I - P_mean is partition-free, so the two
   spectral loading differences must vanish together.

   Honest scope, as in delayed_network_exposure.v: Rocq proves the ALGEBRA over reals. Stdlib has
   no matrices or integrals, so trace values, moment identities (Isserlis, verified in the Maxima
   file at n = 2) and the positivity of the resolvent integral enter as hypotheses or scalar
   shadows. What is proved here is everything downstream of them. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* (A) The second-order gap is the two-trace expression: substituting the Gaussian
   quadratic-form moments cxy = 2 tBOCO, vy = 2 tCOCO, mx = tBO, my = tCO into the delta
   expansion (whose dx^2 coefficient is 0 -- X enters linearly) gives the claimed form. *)
Lemma second_order_gap_is_two_traces :
  forall tBO tCO tBOCO tCOCO,
    tBO <> 0 -> tCO <> 0 ->
    (tBO / tCO ^ 2 + (-2 / tCO ^ 3) * (2 * tBOCO) + (3 * tBO / tCO ^ 4) * (2 * tCOCO))
      / (tBO / tCO ^ 2) - 1
    = -4 * tBOCO / (tBO * tCO) + 6 * tCOCO / tCO ^ 2.
Proof. intros; field; split; assumption. Qed.

(* (B) At Sigma = I the two matrices coincide and the gap collapses to a POSITIVE number:
   the plug-in is optimistic there. tr(C Om C Om) > 0 is the scalar shadow of C Om <> 0. *)
Lemma sigma_eq_identity_gap_positive :
  forall tCO tCOCO,
    tCO <> 0 -> 0 < tCOCO ->
    0 < -4 * tCOCO / (tCO * tCO) + 6 * tCOCO / tCO ^ 2.
Proof.
  intros tCO tCOCO Hne Hpos.
  replace (-4 * tCOCO / (tCO * tCO) + 6 * tCOCO / tCO ^ 2)
    with (2 * tCOCO / tCO ^ 2) by (field; assumption).
  assert (0 < tCO ^ 2) by nra.
  apply Rdiv_lt_0_compat; lra.
Qed.

(* (C) Elliptical u with kurtosis parameter kap: every fourth moment is the Gaussian one times
   (1 + kap), so cov_e = (1+kap) cov_g + kap mx my and vy_e = (1+kap) vy_g + kap my^2, and the
   gap transforms AFFINELY -- tails scale the Gaussian gap and add a floor. *)
Lemma elliptical_gap_affine :
  forall mx my cov_g vy_g kap,
    mx <> 0 -> my <> 0 ->
    -2 * ((1 + kap) * cov_g + kap * (mx * my)) / (mx * my)
      + 3 * ((1 + kap) * vy_g + kap * my ^ 2) / my ^ 2
    = (1 + kap) * (-2 * cov_g / (mx * my) + 3 * vy_g / my ^ 2) + kap.
Proof. intros; field; split; assumption. Qed.

(* (D) The isotropic crossover is exact. At Om = I both fold operators carry the same spectrum
   {0, r^2, 1}, so the exact moment differs between arms only through the two spectral loadings
   s_r = tr(Sigma P_fold), s_1 = tr(Sigma P_within). Their SUM is partition-free
   (P_fold + P_within = I - P_mean), so d_1 = -d_r; the plug-in crossover is r^4 d_r + d_1 = 0.
   With r^4 <> 1 the two force d_r = d_1 = 0 -- no loading difference survives, so the exact
   ratio crosses 1 at the same point, whatever the resolvent weights are. *)
Lemma isotropic_loading_differences_vanish :
  forall r4 dr d1,
    r4 <> 1 -> d1 = - dr -> r4 * dr + d1 = 0 -> dr = 0 /\ d1 = 0.
Proof.
  intros r4 dr d1 Hne Hsum Hplug.
  assert (Hdr : (r4 - 1) * dr = 0) by lra.
  destruct (Rmult_integral _ _ Hdr) as [Hz | Hz]; [lra | split; lra].
Qed.

(* (E) Away from the crossover the exact difference is d_r times a STRICTLY POSITIVE resolvent
   weight: the integrand's bracket r^4 (1+2t) - (1+2t r^2) factors as (r^2-1)(r^2+1+2t r^2). *)
Lemma isotropic_bracket_positive :
  forall r t, 1 < r -> 0 <= t -> 0 < r ^ 4 * (1 + 2 * t) - (1 + 2 * t * r ^ 2).
Proof.
  intros r t Hr Ht.
  replace (r ^ 4 * (1 + 2 * t) - (1 + 2 * t * r ^ 2))
    with ((r ^ 2 - 1) * (r ^ 2 + 1 + 2 * t * r ^ 2)) by ring.
  assert (1 < r ^ 2) by nra.
  assert (0 < 2 * t * r ^ 2 \/ 0 = 2 * t * r ^ 2) as [Hp | Hp] by nra; nra.
Qed.

(* (F) Consequence: at Om = I the plug-in and the exact comparison agree in SIGN at every phi,
   not only at the crossover -- the plug-in difference is (r^4 - 1) d_r, the exact one is d_r
   times the positive integral, and r^4 > 1. The rule never picks the wrong partition there. *)
Lemma plug_in_and_exact_agree_in_sign :
  forall r4 dr Iw,
    1 < r4 -> 0 < Iw ->
    (0 < (r4 - 1) * dr <-> 0 < dr * Iw).
Proof. intros; split; intros; nra. Qed.
