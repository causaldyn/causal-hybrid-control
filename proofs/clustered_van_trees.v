(* Rocq (CONTRIBUTION 2, the LOWER bound): the clustered van-Trees lower bound making the G^{-1}
   control-regret sampling rate IRREDUCIBLE. Result 24 (c2_end_to_end.v) proved only R <= c/G at zero
   nuisance error -- an UPPER bound, so "irreducible floor" was an overstatement (external review). Here
   we prove the matching LOWER bound: G*E[R] is bounded BELOW by a positive constant for every G, and
   increases toward kappa0/Ic -- so no estimator makes the sampling regret decay faster than 1/G.

   Ingredients (Maxima source of truth: validation/clustered_van_trees.mac):
   (1) van-Trees (Bayesian Cramer-Rao): for ANY estimator and a well-behaved prior Q (density vanishing
       at the boundary, finite prior information I0), Bayes risk >= 1/(I0 + integral of data info). With
       G independent clusters Fisher information is ADDITIVE, integral = G*Ic (Gassiat-Stoltz 2024
       arXiv:2402.06431 Thm 4 + p.8; Gill-Levit 1995 Bernoulli 1:59-79; Lehmann-Casella 1998 sec 2.6),
       so E[(Bhat-B)^2] = mse >= 1/(I0 + G*Ic). The ALGEBRAIC core (Cauchy-Schwarz on the joint score
       => Cov^2 <= vx*vy => MSE >= 1/I) is proved in van_trees.v; the score-covariance identity and
       information additivity are the CITED measure-theoretic inputs.
   (2) a LOWER-Lipschitz regret map: the optimal action u*(b)=xt*b/(b^2+rr) is a diffeomorphism with
       |du*/db| >= L_min > 0 away from the knife-edge b^2=rr, so R = (b^2+rr)*(u*(Bhat)-u*(B))^2
       >= (b^2+rr)*L_min^2*(Bhat-B)^2 =: kappa0*(Bhat-B)^2, hence E[R] >= kappa0*mse.
   Composing: G*E[R] >= kappa0 * G/(I0+G*Ic) >= kappa0/(I0+Ic) > 0 for all G>=1 (clustered_floor_positive),
   nondecreasing in G (clustered_floor_increasing) toward the limit kappa0/Ic (Maxima: limit = 1/Ic).
   So the 1/G rate is TIGHT.
   HONEST SCOPE of the "no estimator beats it" claim: the NAIVE pointwise version
   (liminf_G G*E_{theta0}[(Bhat-B)^2] >= 1/Ic for ARBITRARY estimators) is FALSE -- Hodges superefficiency
   beats 1/Ic on a Lebesgue-null set. The correct forms are (a) the finite-G Bayes-average bound above
   (exact, any estimator -- what we prove here); (b) the local-asymptotic-minimax bound with a sup over a
   shrinking c/sqrt(G) neighbourhood (any estimator; Hajek 1972; van der Vaart 1998 Thm 8.11;
   Gassiat-Stoltz 2024 Thm 16 handles the quadratic/bowl-shaped regret loss directly); (c) the pointwise
   bound for REGULAR estimators (Hajek convolution theorem). The regret composition needs regret
   smoothness R = kappa*u^2+o(u^2) plus uniform integrability of sqrt(G)*(Bhat-B) (Fatou minorant). We
   prove the finite-G algebraic form (a); (b)/(c) are the cited statistical wrappers. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* (2) the LOWER-Lipschitz regret map: if the action error a is at least the Lipschitz floor e
   (e = L_min*|Bhat-B| <= |u*(Bhat)-u*(B)| = a), then the regret cc*a^2 is at least cc*e^2. *)
Theorem regret_ge_from_lipschitz : forall cc e a,
  0 <= cc -> 0 <= e -> e <= a -> cc * e ^ 2 <= cc * a ^ 2.
Proof.
  intros cc e a Hcc He Hea. apply Rmult_le_compat_l; [exact Hcc | nra].
Qed.

(* (1)-floor, uniform: the clustered van-Trees G*MSE bound G/(I0+G*Ic) is >= 1/(I0+Ic) for every G>=1
   -- a POSITIVE floor that never vanishes. (Cross-multiplied core: (G-1)*I0 >= 0.) *)
Theorem clustered_floor_positive : forall I0 Ic G,
  0 <= I0 -> 0 < Ic -> 1 <= G -> 1 / (I0 + Ic) <= G / (I0 + G * Ic).
Proof.
  intros I0 Ic G HI0 HIc HG.
  assert (Hd1 : 0 < I0 + Ic) by lra.
  assert (Hd2 : 0 < I0 + G * Ic) by nra.
  apply Rmult_le_reg_r with ((I0 + Ic) * (I0 + G * Ic)); [nra |].
  replace (1 / (I0 + Ic) * ((I0 + Ic) * (I0 + G * Ic))) with (I0 + G * Ic) by (field; lra).
  replace (G / (I0 + G * Ic) * ((I0 + Ic) * (I0 + G * Ic))) with (G * (I0 + Ic)) by (field; lra).
  nra.
Qed.

(* (1)-floor, monotone: G*MSE = G/(I0+G*Ic) is nondecreasing in G, so it increases toward its limit
   1/Ic (Maxima) -- more clusters lower the floor rate monotonically. *)
Theorem clustered_floor_increasing : forall I0 Ic G1 G2,
  0 <= I0 -> 0 < Ic -> 0 < G1 -> G1 <= G2 ->
  G1 / (I0 + G1 * Ic) <= G2 / (I0 + G2 * Ic).
Proof.
  intros I0 Ic G1 G2 HI0 HIc HG1 Hle.
  assert (Hd1 : 0 < I0 + G1 * Ic) by nra.
  assert (Hd2 : 0 < I0 + G2 * Ic) by nra.
  apply Rmult_le_reg_r with ((I0 + G1 * Ic) * (I0 + G2 * Ic)); [nra |].
  replace (G1 / (I0 + G1 * Ic) * ((I0 + G1 * Ic) * (I0 + G2 * Ic)))
    with (G1 * (I0 + G2 * Ic)) by (field; lra).
  replace (G2 / (I0 + G2 * Ic) * ((I0 + G1 * Ic) * (I0 + G2 * Ic)))
    with (G2 * (I0 + G1 * Ic)) by (field; lra).
  nra.
Qed.

(* MAIN: the composed control-regret sampling floor is uniformly POSITIVE. With the van-Trees floor
   supplying a constant c with c <= G*mse (take c = 1/(I0+Ic) via clustered_floor_positive), and the
   lower-Lipschitz regret giving kappa0*mse <= E[R], we get kappa0*c <= G*E[R] for every G. Since c>0,
   G*E[R] does NOT vanish: E[R] decays no faster than 1/G. This is the irreducibility the upper bound
   alone could not give. *)
Theorem regret_floor_uniform_positive : forall kappa0 c G er mse,
  0 < kappa0 -> 0 <= mse -> 0 < G -> c <= G * mse -> kappa0 * mse <= er ->
  kappa0 * c <= G * er.
Proof.
  intros kappa0 c G er mse Hk Hmse HG Hc Her.
  assert (H1 : kappa0 * (G * mse) <= G * er).
  { replace (kappa0 * (G * mse)) with (G * (kappa0 * mse)) by ring.
    apply Rmult_le_compat_l; lra. }
  assert (H2 : kappa0 * c <= kappa0 * (G * mse)) by (apply Rmult_le_compat_l; lra).
  lra.
Qed.
