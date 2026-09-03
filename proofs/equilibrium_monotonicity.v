(* Rocq (GLOBAL monotonicity of the congestion equilibrium map): Result 39 (b) bounds
   ||(I - S')^{-1}|| at ONE point, and says so -- "Both numbers are LOCAL ... the global version
   would need strong monotonicity of F(x) = x - S(x) and is NOT proved here". This is that missing
   step, at n = 2 where the algebra is concrete and every claim is checkable.

   F' = I + kappa*J with J = diag(s) - s s^T. J is a covariance matrix, so its quadratic form is a
   variance and is nonnegative EVERYWHERE, not merely at the equilibrium: F is 1-strongly monotone
   globally (`ambient_modulus_is_one`). The constant 1 cannot be improved, because J annihilates the
   constants (`ambient_modulus_attained`) -- a direction mass conservation forbids, which is why the
   tangent-space modulus `tangent_bound` is strictly larger. `inverse_lipschitz_from_monotone` is the
   step from strong monotonicity to a FINITE-perturbation displacement bound, which is what §39 (b)'s
   implicit-function derivative could not supply. validation/equilibrium_monotonicity.mac carries the
   general-n identities; z3 and cvc5 both return unsat on the n = 3 negation. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* The softmax Jacobian's quadratic form at n = 2: the s-weighted variance of v. *)
Definition var2 (s1 s2 v1 v2 : R) : R :=
  s1 * v1 ^ 2 + s2 * v2 ^ 2 - (s1 * v1 + s2 * v2) ^ 2.

(* The pairwise form -- the identity that lets the SMALLEST weight bound the variance below, which
   the moment form cannot do. Needs the weights to sum to one. *)
Lemma var2_pairwise :
  forall s1 s2 v1 v2 : R,
    s1 + s2 = 1 -> var2 s1 s2 v1 v2 = s1 * s2 * (v1 - v2) ^ 2.
Proof.
  intros s1 s2 v1 v2 H. unfold var2.
  replace s2 with (1 - s1) by lra. ring.
Qed.

Lemma var2_nonneg :
  forall s1 s2 v1 v2 : R,
    0 <= s1 -> 0 <= s2 -> s1 + s2 = 1 -> 0 <= var2 s1 s2 v1 v2.
Proof.
  intros s1 s2 v1 v2 H1 H2 Hs. rewrite var2_pairwise by exact Hs.
  assert (Hsq : 0 <= (v1 - v2) ^ 2) by apply pow2_ge_0.
  assert (Hprod : 0 <= s1 * s2) by nra.
  nra.
Qed.

(* F' = I + kappa*J, so the quadratic form of F' is ||v||^2 + kappa*Var. GLOBAL strong monotonicity
   at modulus 1: this holds at every x, not only at the equilibrium, because nothing here is an
   implicit-function derivative -- it is the covariance structure of J. *)
Definition fprime_form (kappa s1 s2 v1 v2 : R) : R :=
  v1 ^ 2 + v2 ^ 2 + kappa * var2 s1 s2 v1 v2.

Theorem ambient_modulus_is_one :
  forall kappa s1 s2 v1 v2 : R,
    0 <= kappa -> 0 <= s1 -> 0 <= s2 -> s1 + s2 = 1 ->
    v1 ^ 2 + v2 ^ 2 <= fprime_form kappa s1 s2 v1 v2.
Proof.
  intros kappa s1 s2 v1 v2 Hk H1 H2 Hs. unfold fprime_form.
  assert (H := var2_nonneg s1 s2 v1 v2 H1 H2 Hs). nra.
Qed.

(* And 1 is exactly right, not merely valid: on the constants J vanishes, so the inequality above is
   an equality for every kappa. The ambient modulus is saturated in the one direction a
   mass-conserving equilibrium never moves along. *)
Theorem ambient_modulus_attained :
  forall kappa s1 s2 v : R,
    s1 + s2 = 1 -> fprime_form kappa s1 s2 v v = v ^ 2 + v ^ 2.
Proof.
  intros kappa s1 s2 v Hs. unfold fprime_form.
  rewrite var2_pairwise by exact Hs. ring.
Qed.

(* On the fixed-mass tangent space (v1 + v2 = 0) the modulus is strictly larger, bounded below by
   the smallest weight: Var >= n*m^2*||v||^2 with n = 2. Attained at s = (1/2,1/2), so the constant
   n*m^2 cannot be improved -- see `tangent_bound_attained`. *)
Theorem tangent_bound :
  forall s1 s2 v1 v2 m : R,
    0 < m -> m <= s1 -> m <= s2 -> s1 + s2 = 1 -> v1 + v2 = 0 ->
    2 * m ^ 2 * (v1 ^ 2 + v2 ^ 2) <= var2 s1 s2 v1 v2.
Proof.
  intros s1 s2 v1 v2 m Hm H1 H2 Hs Hv.
  rewrite var2_pairwise by exact Hs.
  replace v2 with (- v1) by lra.
  assert (Hprod : m ^ 2 <= s1 * s2) by nra.
  assert (Hsq : 0 <= v1 ^ 2) by apply pow2_ge_0.
  nra.
Qed.

Lemma tangent_bound_attained :
  var2 (1/2) (1/2) 1 (-1) = 2 * (1/2) ^ 2 * (1 ^ 2 + (-1) ^ 2).
Proof. unfold var2. lra. Qed.

(* The step §39 (b) could not take. Strong monotonicity plus Cauchy-Schwarz gives
   mu*||x-y||^2 <= <F(x)-F(y), x-y> <= ||F(x)-F(y)||*||x-y||, and dividing by ||x-y|| bounds the
   equilibrium displacement by the operator perturbation for FINITE perturbations, not only
   infinitesimal ones. At mu = 1 the bound is one-for-one. *)
Theorem inverse_lipschitz_from_monotone :
  forall mu d g : R,
    0 < mu -> 0 <= d -> 0 <= g -> mu * d * d <= g * d -> mu * d <= g.
Proof.
  intros mu d g Hmu Hd Hg Hle.
  destruct (Rle_lt_or_eq_dec 0 d Hd) as [Hpos | Heq].
  - apply Rmult_le_reg_r with (r := d); [ exact Hpos | lra ].
  - (* d = 0 carries no information about g, so the norm's own nonnegativity is what closes it --
       the hypothesis is not decoration. *)
    rewrite <- Heq. lra.
Qed.
