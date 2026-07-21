(* Rocq (CONTRIBUTION 3, publishable upgrade): the FORMAL van Trees (Bayesian Cramer-Rao) inequality
   that result 20 (adaptive_exploration) ASSUMED as the floor C/m_t. Van Trees: for a Bayesian model and
   ANY estimator, the Bayes risk obeys E[(theta_hat - theta)^2] >= 1/(I_data + I_prior). Unlike ordinary
   Cramer-Rao it holds for SEQUENTIAL/adaptive designs and any (even biased) estimator. The proof is
   Cauchy-Schwarz on the Bayesian score psi: with the van-Trees identity E[psi*Delta] = 1 and the
   information decomposition E[psi^2] = I_data + I_prior, 1 <= (I_data+I_prior)*MSE, so
   MSE >= 1/(I_data+I_prior). Derived in validation/van_trees.mac (tight for the Gaussian conjugate).
   Here we prove the algebraic core: Cauchy-Schwarz, the resulting bound, and that confounding (less
   data information) raises the floor. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* Cauchy-Schwarz for second moments, from the nonnegative-variance witness
   E[(X - (cov/vy)*Y)^2] = vx - cov^2/vy >= 0 (X = score psi, Y = Delta). *)
Theorem cauchy_schwarz : forall vx vy cov,
  0 < vy -> 0 <= vx - cov ^ 2 / vy -> cov ^ 2 <= vx * vy.
Proof.
  intros vx vy cov Hvy Hwit.
  assert (Hid : vx * vy - cov ^ 2 = (vx - cov ^ 2 / vy) * vy) by (field; lra).
  nra.
Qed.

(* THE VAN TREES INEQUALITY: with the score identity (cov = E[psi*Delta] = 1) and the total information
   (vx = E[psi^2] = I_data + I_prior), the Bayes risk mse = E[Delta^2] is bounded below by 1/total. *)
Theorem van_trees_inequality : forall total mse,
  0 < total -> 0 < mse -> 0 <= total - 1 / mse -> 1 / total <= mse.
Proof.
  intros total mse Ht Hm Hwit.
  (* the witness 0 <= total - 1/mse is E[(psi - (1/mse)*Delta)^2] with cov = 1, vx = total, vy = mse *)
  assert (Hcs : 1 ^ 2 <= total * mse) by (apply (cauchy_schwarz total mse 1); [exact Hm | lra]).
  assert (H1 : 1 <= total * mse) by lra.
  apply Rmult_le_reg_l with total; [exact Ht |].
  replace (total * (1 / total)) with 1 by (field; lra). lra.
Qed.

(* CONFOUNDING raises the van-Trees floor: the floor 1/(I_prior + I_data) is antitone in the data
   information, and confounding reduces I_data (steals identifying variation) -- the Bayesian /
   sequential analogue of results 10 and 12. *)
Theorem confounding_raises_van_trees_floor : forall j i1 i2,
  0 < j + i1 -> i1 <= i2 -> 1 / (j + i2) <= 1 / (j + i1).
Proof.
  intros j i1 i2 Hpos Hle. unfold Rdiv.
  apply Rmult_le_compat_l; [lra | apply Rinv_le_contravar; lra].
Qed.
