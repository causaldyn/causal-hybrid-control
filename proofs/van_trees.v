(* Rocq (CONTRIBUTION 3, publishable upgrade): the FORMAL van Trees (Bayesian Cramer-Rao) inequality
   that result 20 (adaptive_exploration) ASSUMED as the floor C/m_t. Van Trees: for a Bayesian model and
   ANY estimator, the Bayes risk obeys E[(theta_hat - theta)^2] >= 1/(I_data + I_prior). Unlike ordinary
   Cramer-Rao it holds for SEQUENTIAL/adaptive designs and any (even biased) estimator. The proof is
   Cauchy-Schwarz on the Bayesian score psi: with the van-Trees identity E[psi*Delta] = 1 and the
   information decomposition E[psi^2] = I_data + I_prior, 1 <= (I_data+I_prior)*MSE, so
   MSE >= 1/(I_data+I_prior). Derived in validation/van_trees.mac (tight for the Gaussian conjugate).
   HONEST SCOPE: the two premises -- the score identity E[psi*Delta]=1 and the information decomposition
   -- are exactly where the sequential/adaptive statistical content lives (joint trajectory law,
   conditional likelihoods, a martingale score decomposition, differentiation under the integral). We do
   NOT formalise that measure-theoretic model; we take those two identities as HYPOTHESES (cov=1, vx=total)
   and prove the ALGEBRAIC core from them: Cauchy-Schwarz, the resulting 1/I bound, and that confounding
   (less data information) raises the floor. So this certifies the algebra of van Trees, not its sequential
   validity, which is cited. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* ===== THE CITED MEASURE-THEORETIC INPUTS, AS NAMED PREDICATES (plans/24 P1.2) =====

   The two premises the honest-scope note above names are declared here as Props carrying their
   citations, and the theorems are stated in those names. Nothing is assumed that was not assumed
   before -- each unfolds to the formula the statements already carried -- but the dependency is
   machine-visible: `Check van_trees_floor_from_cited_inputs` prints exactly where the sequential
   statistical content enters, and `Print ScoreIdentity` prints what was taken on trust.

   Definitions and not Axioms, for the reason spelled out in proofs/c2_end_to_end.v: an Axiom
   would make `Print Assumptions` name them and would simultaneously make every theorem here true
   only conditionally on our transcription being right. A hypothesis the caller must discharge is
   the stronger statement. *)

(* Gill & Levit (1995), Bernoulli 1:59-79; van der Vaart (1998) Thm 2.5.2 -- the van Trees SCORE
   IDENTITY E[psi * Delta] = 1 for the Bayesian score psi. This is where the sequential/adaptive
   content lives: the joint trajectory law, the conditional likelihoods, the martingale score
   decomposition, and differentiation under the integral sign. None of it is formalised here. *)
Definition ScoreIdentity (cov : R) : Prop := cov = 1.

(* Gassiat & Stoltz (2024) arXiv:2402.06431 Thm 4 and p.8; Lehmann & Casella (1998) sec 2.6 --
   FISHER INFORMATION IS ADDITIVE over independent blocks, so the score's second moment splits
   into a data term and a prior term. With G independent clusters the data term is G * Ic, which
   is what makes the 1/G floor of proofs/clustered_van_trees.v a statement about CLUSTERS. *)
Definition InformationDecomposition (total i_data i_prior : R) : Prop := total = i_data + i_prior.

(* The Cauchy-Schwarz witness itself: E[(X - (cov/vy)*Y)^2] = vx - cov^2/vy >= 0. Not a citation --
   it is a second moment and therefore nonnegative -- but it is the hypothesis that carries the
   existence of the moments, so it is named alongside them. *)
Definition NonnegativeVarianceWitness (vx vy cov : R) : Prop := 0 <= vx - cov ^ 2 / vy.

(* Cauchy-Schwarz for second moments, from the nonnegative-variance witness
   E[(X - (cov/vy)*Y)^2] = vx - cov^2/vy >= 0 (X = score psi, Y = Delta). *)
Theorem cauchy_schwarz : forall vx vy cov,
  0 < vy -> NonnegativeVarianceWitness vx vy cov -> cov ^ 2 <= vx * vy.
Proof.
  intros vx vy cov Hvy Hwit. unfold NonnegativeVarianceWitness in Hwit.
  assert (Hid : vx * vy - cov ^ 2 = (vx - cov ^ 2 / vy) * vy) by (field; lra).
  nra.
Qed.

(* THE VAN TREES INEQUALITY: with the score identity (cov = E[psi*Delta] = 1) and the total information
   (vx = E[psi^2] = I_data + I_prior), the Bayes risk mse = E[Delta^2] is bounded below by 1/total. *)
Theorem van_trees_inequality : forall total mse,
  0 < total -> 0 < mse -> NonnegativeVarianceWitness total mse 1 -> 1 / total <= mse.
Proof.
  intros total mse Ht Hm Hwit.
  (* the witness is E[(psi - (1/mse)*Delta)^2] >= 0 with cov = 1, vx = total, vy = mse *)
  assert (Hcs : 1 ^ 2 <= total * mse) by (apply (cauchy_schwarz total mse 1); assumption).
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

(* THE SAME FLOOR, STATED IN THE VOCABULARY OF ITS TWO CITED INPUTS. The score identity supplies
   cov = 1 and information additivity supplies total = I_data + I_prior, so the Bayes risk of ANY
   estimator -- biased, adaptive, sequential -- is at least 1/(I_data + I_prior). This proves
   nothing `van_trees_inequality` did not already prove; what it adds is that `Check` on it names
   the two places the measure-theoretic content enters, so a reader can audit the dependency
   without reading the header. *)
Theorem van_trees_floor_from_cited_inputs : forall i_data i_prior total cov mse,
  ScoreIdentity cov -> InformationDecomposition total i_data i_prior ->
  0 < total -> 0 < mse -> NonnegativeVarianceWitness total mse cov ->
  1 / (i_data + i_prior) <= mse.
Proof.
  intros i_data i_prior total cov mse Hcov Hinfo Ht Hm Hwit.
  unfold ScoreIdentity in Hcov. unfold InformationDecomposition in Hinfo.
  subst cov. subst total.
  apply van_trees_inequality; assumption.
Qed.
