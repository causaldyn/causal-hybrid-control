(* Rocq: SPECTRAL (CIRCULANT) RESIDUALS -- an operator norm that is ATTAINED, and why that makes the
   rollout tube tight instead of merely valid.

   chc.residual.LipschitzResidual gets its constant from a Schur upper bound,
   sigma_max(W) <= sqrt(||W||_1 * ||W||_inf).  That is an inequality with no witness:
   lipschitz_certificate can confirm the bound is not violated and can never confirm it is tight.
   On a PERIODIC, translation-invariant plant -- chc.transport's advection-diffusion field -- the
   operator is circulant, the DFT diagonalises it exactly, and the picture changes.

   This file carries the algebra of that claim in the reals.  A circulant acts on each real Fourier
   mode -- the plane spanned by cos(theta j) and sin(theta j), coordinates (a, b) -- as the 2x2
   block M(p,q) = [[p, q], [-q, p]], the real form of multiplication by the complex symbol p + i q.
   (The off-diagonal sign is a convention fixed by stacking (a,b) in that order; validation/
   spectral_circulant.mac STEP 2 derives it, and getting it backwards is how a sign error enters a
   kernel.)  Four facts follow, none needing a complex number:

   - mode_gain_is_exact: ||M v||^2 = (p^2 + q^2) ||v||^2 for EVERY v.  Not a bound -- an identity.
     So the gain on a mode is |lambda|, attained by every input rather than by a worst case.
   - two_mode_norm_attained: over several modes the operator norm is the MAX of the per-mode gains
     and it is achieved on the maximising mode (two_mode_bounded gives <=, the second gives =).
   - compose_symbols / gain_multiplies: composing circulants multiplies symbols, and the gains
     multiply EXACTLY -- this is the Brahmagupta-Fibonacci identity
     (p p' - q q')^2 + (p q' + q p')^2 = (p^2 + q^2)(p'^2 + q'^2).  Circulants also commute.
   - product_bound_is_strictly_loose: for a COMPOSITION the exact norm is max_k prod_m g_mk, while
     bounding each factor separately gives prod_m max_k g_mk, and the second is STRICTLY larger
     whenever the two operators' maximising modes differ.  That gap is exactly the conservatism the
     Result 28/30 rollout tube pays per step, and what a circulant residual removes.

   Honest scope: two modes, because that is enough to exhibit "max of products < product of maxes";
   the n-mode statement is the same induction with no new idea.  The DFT diagonalisation itself is
   complex and stays in Maxima (STEP 1), as does the advection-diffusion dispersion relation
   (STEP 5) -- Rocq carries the part the certificate's assertions rest on. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.

Open Scope R_scope.

(* ---------- one mode ---------- *)

(* The real 2x2 action of the symbol p + i q on the mode coordinates (a, b). *)
Definition mode_re (p q a b : R) : R := p * a + q * b.
Definition mode_im (p q a b : R) : R := - q * a + p * b.

Definition sq_norm (a b : R) : R := a * a + b * b.
Definition gain (p q : R) : R := p * p + q * q.   (* = |lambda|^2 *)

(* THE IDENTITY.  Every input is a worst-case input: the gain is exact, not bounded. *)
Theorem mode_gain_is_exact : forall p q a b,
  sq_norm (mode_re p q a b) (mode_im p q a b) = gain p q * sq_norm a b.
Proof. intros. unfold sq_norm, mode_re, mode_im, gain. ring. Qed.

Theorem gain_nonneg : forall p q, 0 <= gain p q.
Proof. intros. unfold gain. nra. Qed.

(* A symbol of unit modulus is an isometry on its mode: pure advection moves a wave without
   touching its amplitude, at every wavenumber.  This is the nu = 0 case of STEP 5e. *)
Theorem unit_symbol_is_an_isometry : forall p q a b,
  gain p q = 1 -> sq_norm (mode_re p q a b) (mode_im p q a b) = sq_norm a b.
Proof. intros p q a b H. rewrite mode_gain_is_exact, H. ring. Qed.

(* And a symbol inside the unit disc strictly contracts every nonzero input on that mode: the
   exp(-nu k^2 dt) factor of diffusion, with no appeal to the exponential. *)
Theorem subunit_symbol_strictly_contracts : forall p q a b,
  gain p q < 1 -> 0 < sq_norm a b ->
  sq_norm (mode_re p q a b) (mode_im p q a b) < sq_norm a b.
Proof. intros p q a b Hg Hv. rewrite mode_gain_is_exact. nra. Qed.

(* ---------- composition ---------- *)

(* Applying (p,q) then (p2,q2) is applying the single symbol (p+iq)(p2+iq2). *)
Theorem compose_symbols_re : forall p q p2 q2 a b,
  mode_re p2 q2 (mode_re p q a b) (mode_im p q a b)
  = mode_re (p * p2 - q * q2) (p * q2 + q * p2) a b.
Proof. intros. unfold mode_re, mode_im. ring. Qed.

Theorem compose_symbols_im : forall p q p2 q2 a b,
  mode_im p2 q2 (mode_re p q a b) (mode_im p q a b)
  = mode_im (p * p2 - q * q2) (p * q2 + q * p2) a b.
Proof. intros. unfold mode_re, mode_im. ring. Qed.

(* Circulants commute -- the symbols are complex numbers, and multiplication is commutative. *)
Theorem modes_commute_re : forall p q p2 q2 a b,
  mode_re p2 q2 (mode_re p q a b) (mode_im p q a b)
  = mode_re p q (mode_re p2 q2 a b) (mode_im p2 q2 a b).
Proof. intros. unfold mode_re, mode_im. ring. Qed.

Theorem modes_commute_im : forall p q p2 q2 a b,
  mode_im p2 q2 (mode_re p q a b) (mode_im p q a b)
  = mode_im p q (mode_re p2 q2 a b) (mode_im p2 q2 a b).
Proof. intros. unfold mode_re, mode_im. ring. Qed.

(* Brahmagupta-Fibonacci: gains multiply exactly under composition.  For an upper bound this would
   read <=, and every composition would lose a little more; here nothing is lost. *)
Theorem gain_multiplies : forall p q p2 q2,
  gain (p * p2 - q * q2) (p * q2 + q * p2) = gain p q * gain p2 q2.
Proof. intros. unfold gain. ring. Qed.

(* ---------- several modes: bounded AND attained ---------- *)

Definition two_mode_out (p0 q0 p1 q1 a0 b0 a1 b1 : R) : R :=
  sq_norm (mode_re p0 q0 a0 b0) (mode_im p0 q0 a0 b0)
  + sq_norm (mode_re p1 q1 a1 b1) (mode_im p1 q1 a1 b1).

Definition two_mode_in (a0 b0 a1 b1 : R) : R := sq_norm a0 b0 + sq_norm a1 b1.

Lemma sq_norm_nonneg : forall a b, 0 <= sq_norm a b.
Proof. intros. unfold sq_norm. nra. Qed.

Theorem two_mode_bounded : forall p0 q0 p1 q1 a0 b0 a1 b1,
  two_mode_out p0 q0 p1 q1 a0 b0 a1 b1
  <= Rmax (gain p0 q0) (gain p1 q1) * two_mode_in a0 b0 a1 b1.
Proof.
  intros. unfold two_mode_out, two_mode_in.
  rewrite !mode_gain_is_exact.
  assert (H0 := Rmax_l (gain p0 q0) (gain p1 q1)).
  assert (H1 := Rmax_r (gain p0 q0) (gain p1 q1)).
  assert (N0 := sq_norm_nonneg a0 b0).
  assert (N1 := sq_norm_nonneg a1 b1).
  nra.
Qed.

(* THE PART A SCHUR BOUND CANNOT HAVE.  Put the whole input on the maximising mode and the bound is
   an equality: the operator norm is not merely an upper bound, it is realised by an explicit
   input.  This is what chc.residual.spectral_residual_certificate measures. *)
Theorem two_mode_norm_attained : forall p0 q0 p1 q1 a0 b0,
  gain p1 q1 <= gain p0 q0 ->
  two_mode_out p0 q0 p1 q1 a0 b0 0 0
  = Rmax (gain p0 q0) (gain p1 q1) * two_mode_in a0 b0 0 0.
Proof.
  intros p0 q0 p1 q1 a0 b0 Hle.
  unfold two_mode_out, two_mode_in. rewrite !mode_gain_is_exact.
  rewrite (Rmax_left _ _ Hle). unfold sq_norm, mode_re, mode_im. ring.
Qed.

(* ---------- why the tube is tight ---------- *)

(* Composing two operators, the exact norm is the max over modes of the PRODUCT of per-mode gains;
   bounding each factor separately gives the product of the maxima.  The second dominates... *)
Theorem product_bound_is_valid : forall g0 g1 h0 h1,
  0 <= g0 -> 0 <= g1 -> 0 <= h0 -> 0 <= h1 ->
  Rmax (g0 * h0) (g1 * h1) <= Rmax g0 g1 * Rmax h0 h1.
Proof.
  intros g0 g1 h0 h1 Hg0 Hg1 Hh0 Hh1.
  assert (A0 := Rmax_l g0 g1). assert (A1 := Rmax_r g0 g1).
  assert (B0 := Rmax_l h0 h1). assert (B1 := Rmax_r h0 h1).
  apply Rmax_case; nra.
Qed.

(* ...and it is STRICTLY loose exactly when the two operators peak on different modes.  Bounding
   each factor separately then pays a fixed multiplicative penalty at every step of a rollout, which
   is the conservatism a circulant residual removes from the Result 28/30 tube. *)
Theorem product_bound_is_strictly_loose : forall g0 g1 h0 h1,
  0 < g0 -> 0 <= g1 -> 0 <= h0 -> 0 < h1 ->
  g1 < g0 -> h0 < h1 ->
  Rmax (g0 * h0) (g1 * h1) < Rmax g0 g1 * Rmax h0 h1.
Proof.
  intros g0 g1 h0 h1 Hg0 Hg1 Hh0 Hh1 Hgs Hhs.
  rewrite (Rmax_left g0 g1) by lra. rewrite (Rmax_right h0 h1) by lra.
  apply Rmax_case; nra.
Qed.

(* The same statement in the form the certificate reports it: a ratio strictly above 1. *)
Theorem conservatism_ratio_exceeds_one : forall g0 g1 h0 h1,
  0 < g0 -> 0 <= g1 -> 0 <= h0 -> 0 < h1 ->
  g1 < g0 -> h0 < h1 -> 0 < Rmax (g0 * h0) (g1 * h1) ->
  1 < (Rmax g0 g1 * Rmax h0 h1) / Rmax (g0 * h0) (g1 * h1).
Proof.
  intros g0 g1 h0 h1 Hg0 Hg1 Hh0 Hh1 Hgs Hhs Hpos.
  assert (Hlt := product_bound_is_strictly_loose g0 g1 h0 h1 Hg0 Hg1 Hh0 Hh1 Hgs Hhs).
  apply Rmult_lt_reg_r with (r := Rmax (g0 * h0) (g1 * h1)); [exact Hpos | ].
  unfold Rdiv. rewrite Rmult_assoc, Rinv_l by lra. lra.
Qed.
