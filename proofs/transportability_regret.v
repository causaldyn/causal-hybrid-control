(* Rocq: a TRANSPORTABILITY regret bound for causal control -- a controller trained on a source domain,
   deployed on a target (Bareinboim, "Causal AI", Ch 9; derived in
   validation/transportability_regret.mac). Deployment regret C*(b_src - b_tgt)^2 splits into: (A) a
   TRANSPORTABLE part -- if the effect is recoverable on the target (b_src = b_tgt) the regret is ZERO
   for ANY distributional distance; (B) a NON-transportable residual bounded by the Wasserstein-1
   distance d = W1(P,P') when the effect shifts Lipschitz in d, giving C*Lip^2*d^2 (the quantity
   chc.WassersteinPenalty penalises); (C) a W-DRO controller planned for a radius eps covers the
   realized regret whenever d <= eps -- the robust radius is a transportability budget. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

Definition deployment_regret (cc bs bt : R) : R := cc * (bs - bt) ^ 2.

(* (A) Transportability, not closeness, kills the regret: if the effect is recoverable on the target
   (b_src = b_tgt), the deployment regret is zero regardless of the distributional distance. *)
Theorem transportable_zero_regret : forall cc bs bt,
  bs = bt -> deployment_regret cc bs bt = 0.
Proof.
  intros cc bs bt Heq. unfold deployment_regret. subst bs. ring.
Qed.

(* (B) Non-transportable residual: when the effect shifts Lipschitz in the transport distance d
   (|b_src - b_tgt| <= Lip*d), the deployment regret is at most C*Lip^2*d^2 -- quadratic in W1(P,P'). *)
Theorem nontransport_regret_bound : forall cc bs bt lip d,
  0 <= cc -> Rabs (bs - bt) <= lip * d ->
  deployment_regret cc bs bt <= cc * (lip ^ 2 * d ^ 2).
Proof.
  intros cc bs bt lip d Hc Hr. unfold deployment_regret.
  apply Rmult_le_compat_l; [exact Hc |]. split_Rabs; nra.
Qed.

(* (C) A W-DRO controller planned for a W1-ball of radius eps has worst-case bound C*Lip^2*eps^2, which
   COVERS the realized regret C*Lip^2*d^2 whenever the true shift d <= eps: the robust radius is a
   transportability budget. *)
Theorem wdro_radius_covers : forall cc lip d eps,
  0 <= cc * lip ^ 2 -> 0 <= d -> d <= eps ->
  cc * lip ^ 2 * d ^ 2 <= cc * lip ^ 2 * eps ^ 2.
Proof.
  intros cc lip d eps Hk Hd Hle. apply Rmult_le_compat_l; [exact Hk | nra].
Qed.
