(* Rocq: a regret certificate BEYOND the local linearisation (control-first result #3). The LQ /
   certainty-equivalence bounds (proofs/interference_regret.v, causal_mpc.v) and the local nonlinear
   certificate (chc.lqr.linearized_regret_certificate) linearise around the optimum. For a
   mu-strongly-convex control cost the regret admits a GLOBAL, self-certifying upper bound
   ||grad J(u)||^2 / (2 mu) -- computed from the achieved gradient alone, WITHOUT knowing the optimum,
   valid arbitrarily far from it. The proof uses only the strong-convexity inequality (the standard
   definition), not a Taylor expansion, so it holds for genuinely nonlinear costs. This is the
   Polyak-Lojasiewicz / strong-convexity regret bound, specialised to control.

   Cleared of division: proved as 2*mu*(regret) <= grad^2, a pure polynomial certificate. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* Strong convexity at (u, ustar) with ustar - u = -d gives, as a hypothesis on the cost values:
   J(ustar) >= J(u) - J'(u)*d + (mu/2)*d^2.  We derive the self-certifying regret bound. *)
Theorem strong_convexity_regret_bound : forall ju jstar jp mu d,
  0 < mu ->
  jstar >= ju - jp * d + mu / 2 * d ^ 2 ->
  2 * mu * (ju - jstar) <= jp ^ 2.
Proof.
  intros ju jstar jp mu d Hmu Hsc.
  (* strong convexity => regret is at most the concave quadratic jp*d - (mu/2) d^2 in the free d *)
  assert (H1 : ju - jstar <= jp * d - mu / 2 * d ^ 2) by lra.
  (* scale that slack by 2*mu >= 0, and complete the square: the sum is exactly jp^2 - 2*mu*regret *)
  assert (H2 : 0 <= 2 * mu * (jp * d - mu / 2 * d ^ 2 - (ju - jstar))).
  { apply Rmult_le_pos; nra. }
  assert (Hsq : 0 <= (jp - mu * d) ^ 2) by apply pow2_ge_0.
  assert (Hid : jp ^ 2 - 2 * mu * (ju - jstar)
                = 2 * mu * (jp * d - mu / 2 * d ^ 2 - (ju - jstar)) + (jp - mu * d) ^ 2) by field.
  lra.
Qed.

(* Corollary: the bound is TIGHT for a purely quadratic cost (mu*d^2/2 regret, gradient mu*d):
   there 2*mu*regret = 2*mu*(mu/2*d^2) = (mu*d)^2 = grad^2, so the certificate is exact. *)
Lemma quadratic_bound_is_tight : forall mu d,
  2 * mu * (mu / 2 * d ^ 2) = (mu * d) ^ 2.
Proof. intros mu d. field. Qed.

(* Finite-horizon (multivariate) justification of taking mu = lambda_min(Hessian): for a single
   eigenmode with curvature h >= mu, the mu-PL bound upper-bounds that mode's regret. In cleared
   (division-free) form: mu*(h*d^2) <= (h*d)^2 (i.e. the mode regret (h/2)d^2 <= (h*d)^2/(2 mu)). The
   min-curvature mode is tight; every stiffer mode is slack. Summing over eigenmodes gives the full
   finite-horizon regret bound, verified numerically by chc.regret.finite_horizon_pl_certificate. *)
Lemma pl_mode_bound : forall h mu d,
  0 < mu -> mu <= h -> mu * (h * d ^ 2) <= (h * d) ^ 2.
Proof.
  intros h mu d Hmu Hle.
  assert (Hd : 0 <= d ^ 2) by apply pow2_ge_0.
  assert (Hh : 0 <= h * d ^ 2) by nra.
  nra.
Qed.
