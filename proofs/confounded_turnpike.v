(* Rocq: the CONFOUNDED TURNPIKE GAP -- from Weber, "Optimal Control Theory with Applications in
   Economics" (2011), Sec 3.5 (infinite-horizon control: turnpike + current-value Hamiltonian; derived
   in validation/confounded_turnpike.mac). A long-horizon optimal trajectory sits near a static turnpike
   (optimal steady state) for almost all of its length. A CONFOUNDED controller, using the biased effect
   b_obs = b + beta, converges to the WRONG turnpike x_conf = b*xref/(b+beta) and pays the offset every
   step. This gives result #1d a proper turnpike footing, plus a NEW discounted-regret certificate:
   undiscounted cumulative regret grows linearly (unbounded), but the discounted sum stays bounded by
   c/(1-g). *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* The turnpike gap: the confounded steady state x_conf = b*xref/(b+beta) sits an offset below xref. *)
Theorem turnpike_gap_formula : forall b beta xref,
  b + beta <> 0 -> xref - b * xref / (b + beta) = xref * beta / (b + beta).
Proof.
  intros b beta xref Hne. field. exact Hne.
Qed.

(* (B) Undiscounted cumulative regret R(T) = T*c grows without bound: with a positive per-step regret c
   the confounded controller never settles -- a longer horizon strictly costs more. *)
Theorem undiscounted_grows_linearly : forall c t1 t2,
  0 < c -> t1 < t2 -> t1 * c < t2 * c.
Proof.
  intros c t1 t2 Hc Hlt. apply Rmult_lt_compat_r; assumption.
Qed.

(* (C) Discounted cumulative regret R_g(T) = c*(1 - g^T)/(1 - g) stays BOUNDED by c/(1-g) for every
   horizon T -- discounting (the current-value Hamiltonian) makes the confounded regret finite even
   though the undiscounted version is unbounded. *)
Theorem discounted_regret_bounded : forall c g n,
  0 <= c -> 0 <= g -> g < 1 -> c * (1 - g ^ n) / (1 - g) <= c / (1 - g).
Proof.
  intros c g n Hc Hg Hg1.
  assert (Hgn : 0 <= g ^ n) by (apply pow_le; exact Hg).
  unfold Rdiv. apply Rmult_le_compat_r.
  - left; apply Rinv_0_lt_compat; lra.
  - nra.
Qed.
