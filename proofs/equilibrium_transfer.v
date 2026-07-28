(* Rocq (Result 39): the equilibrium layer of chc.games is order-preserving and PERFECTLY
   CONDITIONED, and the contraction modulus of the damped solver is NOT its conditioning.

   Setting (Maxima: validation/equilibrium_transfer.mac). S is the undamped logit best response and
   T_d = (1-d)*x + d*S(x) the damped iteration; they share fixed points. With J = diag(s) - s s^T
   the softmax Jacobian, spec(J) is contained in [0, 1/2] (proofs/congestion_contraction.v), and

       S' = -kappa*J,    I - S' = I + kappa*J,    T_d' = (1-d)*I - d*kappa*J.

   The equilibrium's sensitivity to a perturbation of the agents' operator is ||(I - S')^{-1}||,
   the solver's rate is 1 - ||T_d'||. These are DIFFERENT objects, and conflating them was the
   error in the roadmap proposal that suggested a C/mu^2 regret constant. Proved here:

     - conditioning_at_most_one / conditioning_attains_one: every eigenvalue of (I - S')^{-1} is
       1/(1 + kappa*lam) <= 1, with equality at lam = 0 (J annihilates the constants -- the
       mass-conservation direction). So ||(I - S')^{-1}||_2 = 1 EXACTLY, for every kappa >= 0.
       Measured: 1.0000 at kappa = 4.00, 5.00, 5.50, 5.80, 5.96.
     - naive_bound_exceeds_truth / naive_bound_is_unbounded: 1/mu > 1 strictly on 4 < kappa < 6,
       and at kappa = 6 - 4*eps it equals 1/eps -- so the naive constant is loose without limit
       while the true conditioning does not move. Measured looseness: 2x, 4x, 8x, 20x, 100x.
     - damped_contraction_threshold: T_d' has spectrum in (-1,1) iff d*(1 + kappa/2) < 2.
     - optimal_damping_contracts: d* = 4/(4+kappa) always satisfies it, so EVERY game has a
       certified damping. The kappa < 6 ceiling belonged to the hard-coded d = 1/2, not the game.

   Scope: these are the SCALAR spectral facts. The step from equilibrium error to leader regret
   needs an interior optimum (implicit-function-theorem regularity) and is measured, not proved --
   chc.games.equilibrium_transfer_certificate reports slope 1.97 there, and ~0.9 once a budget
   constraint is active. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* Conditioning: (I - S') = I + kappa*J has eigenvalues 1 + kappa*lam >= 1, so the inverse
   contracts in every direction -- the equilibrium never amplifies an operator perturbation. *)
Lemma conditioning_at_most_one :
  forall kappa lam : R, 0 <= kappa -> 0 <= lam -> / (1 + kappa * lam) <= 1.
Proof.
  intros kappa lam Hk Hl.
  assert (Hprod : 0 <= kappa * lam) by (apply Rmult_le_pos; assumption).
  assert (Hpos : 0 < 1 + kappa * lam) by lra.
  apply Rmult_le_reg_l with (r := 1 + kappa * lam); [ exact Hpos |].
  rewrite Rinv_r by lra. rewrite Rmult_1_r. lra.
Qed.

(* And the bound is attained: J annihilates the constants (sum s = 1), so lam = 0 is always in the
   spectrum and the operator norm is exactly 1 -- not merely at most 1. *)
Lemma conditioning_attains_one : forall kappa : R, / (1 + kappa * 0) = 1.
Proof. intros kappa. rewrite Rmult_0_r, Rplus_0_r. apply Rinv_1. Qed.

(* The damped iteration's spectrum: eigenvalues (1-d) - d*kappa*lam for lam in [0, 1/2]. It stays
   inside (-1,1) exactly under d*(1 + kappa/2) < 2, which at d = 1/2 reads kappa < 6. *)
Lemma damped_contraction_threshold :
  forall d kappa lam : R,
    0 < d -> d <= 1 -> 0 <= kappa -> 0 <= lam -> lam <= 1 / 2 ->
    d * (1 + kappa / 2) < 2 ->
    -1 < (1 - d) - d * kappa * lam < 1.
Proof.
  intros d kappa lam Hd0 Hd1 Hk Hl0 Hlhalf Hthresh.
  assert (Hdk : 0 <= d * kappa) by (apply Rmult_le_pos; lra).
  assert (Hexpand : d + d * kappa / 2 < 2).
  { replace (d + d * kappa / 2) with (d * (1 + kappa / 2)) by field. exact Hthresh. }
  split.
  - assert (Hprod : d * kappa * lam <= d * kappa * (1 / 2))
      by (apply Rmult_le_compat_l; assumption).
    lra.
  - assert (Hnonneg : 0 <= d * kappa * lam) by (apply Rmult_le_pos; assumption).
    lra.
Qed.

(* Every game has a certified damping: d* = 4/(4+kappa) is always below the threshold 4/(2+kappa),
   so the kappa < 6 ceiling is a property of the hard-coded d = 1/2 and not of the game. *)
Lemma optimal_damping_contracts :
  forall kappa : R, 0 <= kappa -> (4 / (4 + kappa)) * (1 + kappa / 2) < 2.
Proof.
  intros kappa Hk.
  assert (Hpos : 0 < 4 + kappa) by lra.
  apply Rmult_lt_reg_l with (r := 4 + kappa); [ exact Hpos |].
  replace ((4 + kappa) * (4 / (4 + kappa) * (1 + kappa / 2))) with (4 * (1 + kappa / 2))
    by (field; lra).
  lra.
Qed.

(* The naive constant 1/mu strictly exceeds the true conditioning 1 on the active branch, ... *)
Lemma naive_bound_exceeds_truth :
  forall kappa : R, 4 < kappa -> kappa < 6 -> 1 < / (3 / 2 - kappa / 4).
Proof.
  intros kappa Hlo Hhi.
  assert (Hpos : 0 < 3 / 2 - kappa / 4) by lra.
  apply Rmult_lt_reg_l with (r := 3 / 2 - kappa / 4); [ exact Hpos |].
  rewrite Rinv_r by lra. rewrite Rmult_1_r. lra.
Qed.

(* ... and without any bound: at kappa = 6 - 4*eps the naive constant is exactly 1/eps, while the
   conditioning proved above stays at 1. *)
Lemma naive_bound_is_unbounded :
  forall eps : R, 0 < eps -> / (3 / 2 - (6 - 4 * eps) / 4) = / eps.
Proof. intros eps Heps. replace (3 / 2 - (6 - 4 * eps) / 4) with eps by field. reflexivity. Qed.
