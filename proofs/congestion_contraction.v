(* Rocq (Result 39 §B): the contraction certificate for the damped logit congestion map used by
   chc.games. The map is T(x) = x/2 + (m/2)*softmax(beta*(a + u - c*x/m)), and Maxima
   (validation/congestion_contraction.mac) reduces its Jacobian to

       T'(x) = I/2 - (kappa/2) * J,     kappa = beta*c,   J = diag(s) - s s^T,

   with v^T J v = Var_s(v). What is proved here is the ALGEBRAIC core of the certificate -- the
   scalar facts about an eigenvalue lam of J -- not the JAX numerics, which are cross-checked in
   validation/congestion_contraction.py:

     - popoviciu_half: the Popoviciu step. With ||v|| = 1 the spread a - b of v satisfies
       (a-b)^2/4 <= 1/2, so Var_s(v) <= 1/2, i.e. spec(J) is contained in [0, 1/2].
     - eigenvalue_in_unit_disc: 0 <= kappa < 6 and lam in [0,1/2] give -1 < 1/2 - (kappa/2)*lam < 1,
       so every eigenvalue of T' is inside the unit disc -- the SUFFICIENT contraction condition.
     - boundary_is_sharp: at kappa = 6, lam = 1/2 the eigenvalue is exactly -1. The threshold 6
       cannot be raised without a bound on lam better than 1/2.
     - operator_norm_bound / modulus_positive: ||T'||_2 <= max(1/2, kappa/4 - 1/2) (T' is symmetric,
       so its 2-norm is its spectral radius), and the modulus mu = 1 - that bound is positive
       exactly on kappa < 6. Symmetry also means the adjoint iteration of the implicit-VJP backward
       pass uses the SAME operator, so one certificate covers both passes.

   Sufficient, NOT necessary: lam = 1/2 is attained only at a two-point uniform s. Measured, the
   iteration still converges at kappa = 7 and 2-cycles from kappa = 8. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* Popoviciu: the variance of a bounded variable is at most a quarter of its squared spread, and a
   unit vector has squared spread at most 2 -- because (a-b)^2 <= 2(a^2+b^2) by (a+b)^2 >= 0. *)
Lemma popoviciu_half : forall a b : R, a ^ 2 + b ^ 2 <= 1 -> (a - b) ^ 2 / 4 <= 1 / 2.
Proof.
  intros a b Hnorm.
  assert (Hsum : 0 <= (a + b) ^ 2) by apply pow2_ge_0.
  nra.
Qed.

(* Every eigenvalue of T' = I/2 - (kappa/2) J lies strictly inside the unit disc when kappa < 6.
   Upper side: kappa*lam >= 0 pushes the value down from 1/2. Lower side: it bottoms out at
   1/2 - kappa/4, which exceeds -1 exactly while kappa < 6. *)
Lemma eigenvalue_in_unit_disc :
  forall kappa lam : R,
    0 <= kappa -> kappa < 6 -> 0 <= lam -> lam <= 1 / 2 ->
    -1 < 1 / 2 - kappa / 2 * lam < 1.
Proof.
  intros kappa lam Hk0 Hk6 Hl0 Hlhalf.
  split.
  - assert (Hprod : kappa / 2 * lam <= kappa / 4) by nra.
    lra.
  - assert (Hnonneg : 0 <= kappa / 2 * lam) by nra.
    lra.
Qed.

(* The threshold is exactly 6: at kappa = 6 the worst eigenvalue sits on the boundary. *)
Lemma boundary_is_sharp : 1 / 2 - 6 / 2 * (1 / 2) = -1.
Proof. field. Qed.

(* T' is symmetric, so ||T'||_2 is its spectral radius; bound it by the two endpoints of lam. *)
Lemma operator_norm_bound :
  forall kappa lam : R,
    0 <= kappa -> 0 <= lam -> lam <= 1 / 2 ->
    Rabs (1 / 2 - kappa / 2 * lam) <= Rmax (1 / 2) (kappa / 4 - 1 / 2).
Proof.
  intros kappa lam Hk0 Hl0 Hlhalf.
  destruct (Rle_dec 0 (1 / 2 - kappa / 2 * lam)) as [Hpos | Hneg].
  - rewrite Rabs_right by lra.
    apply Rle_trans with (r2 := 1 / 2); [ nra | apply Rmax_l ].
  - rewrite Rabs_left by lra.
    apply Rle_trans with (r2 := kappa / 4 - 1 / 2); [ nra | apply Rmax_r ].
Qed.

(* The certified modulus mu = 1 - ||T'||_2 is positive exactly on the certified region kappa < 6. *)
Lemma modulus_positive :
  forall kappa : R, 0 <= kappa -> kappa < 6 -> 0 < 1 - Rmax (1 / 2) (kappa / 4 - 1 / 2).
Proof.
  intros kappa Hk0 Hk6.
  destruct (Rle_dec (kappa / 4 - 1 / 2) (1 / 2)) as [Hsmall | Hlarge].
  - rewrite Rmax_left by exact Hsmall. lra.
  - rewrite Rmax_right by lra. lra.
Qed.

(* Conversely, at kappa >= 6 the bound refuses to certify: mu <= 0. The certificate is one-sided --
   it never claims contraction it cannot prove, and says nothing about kappa in [6, 8) where the
   iteration is measured to still converge. *)
Lemma modulus_nonpositive :
  forall kappa : R, 6 <= kappa -> 1 - Rmax (1 / 2) (kappa / 4 - 1 / 2) <= 0.
Proof.
  intros kappa Hk6.
  assert (Hbig : 1 <= kappa / 4 - 1 / 2) by lra.
  rewrite Rmax_right by lra. lra.
Qed.
