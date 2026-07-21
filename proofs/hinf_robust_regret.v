(* Rocq: an H-INFINITY / DIFFERENTIAL-GAME robust regret certificate for causal control (Geering,
   "Optimal Control with Engineering Applications" 2007, Ch 4; derived in validation/hinf_robust_regret.mac).
   Model the confounding-induced uncertainty in the causal effect as an ADVERSARY w perturbing the gain,
   penalised by an energy budget g2 = gamma^2. The robust controller solves a zero-sum game min_u max_w.
   Three facts unify H-inf robustness with the program's pessimism=variance result (#2): (A) the worst-case
   value is g2*e^2/(g2-u^2)+rr*u^2 (a genuine max iff g2>u^2, via a sum-of-squares); (B) robustness
   INFLATES the cost above nominal -- that inflation IS pessimism; (C) the inflation is antitone in the
   budget gamma^2, so gamma^-2 is the pessimism knob (gamma->inf recovers certainty equivalence). *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* zero-sum game cost: nominal tracking error e = b*u - xt, adversary w perturbs the gain b -> b+w. *)
Definition game_cost (e u g2 rr w : R) : R := (e + w * u) ^ 2 + rr * u ^ 2 - g2 * w ^ 2.
Definition robust_value (e u g2 rr : R) : R := g2 * e ^ 2 / (g2 - u ^ 2) + rr * u ^ 2.
Definition nominal_cost (e u rr : R) : R := e ^ 2 + rr * u ^ 2.

(* (A) The robust value is the worst case: for g2 > u^2 the game cost never exceeds it, the gap being the
   sum-of-squares (g2 - u^2)*(w - wstar)^2 (concavity in w => a genuine max). *)
Theorem robust_value_is_worst_case : forall e u g2 rr w,
  u ^ 2 < g2 -> game_cost e u g2 rr w <= robust_value e u g2 rr.
Proof.
  intros e u g2 rr w Hlt. unfold game_cost, robust_value.
  assert (Hne : g2 - u ^ 2 <> 0) by (apply Rgt_not_eq; lra).
  assert (Hgap :
    g2 * e ^ 2 / (g2 - u ^ 2) + rr * u ^ 2 - ((e + w * u) ^ 2 + rr * u ^ 2 - g2 * w ^ 2)
    = (g2 - u ^ 2) * (w - e * u / (g2 - u ^ 2)) ^ 2) by (field; exact Hne).
  assert (Hsq : 0 <= (g2 - u ^ 2) * (w - e * u / (g2 - u ^ 2)) ^ 2)
    by (apply Rmult_le_pos; [lra | apply pow2_ge_0]).
  lra.
Qed.

(* (B) Robustness INFLATES the cost above nominal -- this inflation is exactly pessimism. *)
Theorem robustness_inflates_cost : forall e u g2 rr,
  u ^ 2 < g2 -> nominal_cost e u rr <= robust_value e u g2 rr.
Proof.
  intros e u g2 rr Hlt. unfold nominal_cost, robust_value.
  assert (Hne : g2 - u ^ 2 <> 0) by (apply Rgt_not_eq; lra).
  assert (Hid : g2 * e ^ 2 / (g2 - u ^ 2) + rr * u ^ 2 - (e ^ 2 + rr * u ^ 2)
                = e ^ 2 * u ^ 2 / (g2 - u ^ 2)) by (field; exact Hne).
  assert (Hpos : 0 <= e ^ 2 * u ^ 2 / (g2 - u ^ 2)).
  { apply Rmult_le_pos.
    - apply Rmult_le_pos; apply pow2_ge_0.
    - left; apply Rinv_0_lt_compat; lra. }
  lra.
Qed.

(* (C) The inflation is ANTITONE in the budget gamma^2: more budget => less pessimism, and gamma -> inf
   recovers the nominal (certainty-equivalence) cost. So gamma^-2 is the pessimism knob. *)
Theorem robustness_antitone_in_budget : forall e u g1 g2 rr,
  u ^ 2 < g1 -> g1 <= g2 -> robust_value e u g2 rr <= robust_value e u g1 rr.
Proof.
  intros e u g1 g2 rr H1 Hle. unfold robust_value.
  assert (Hn1 : g1 - u ^ 2 <> 0) by (apply Rgt_not_eq; lra).
  assert (Hn2 : g2 - u ^ 2 <> 0) by (apply Rgt_not_eq; lra).
  assert (Hid1 : g1 * e ^ 2 / (g1 - u ^ 2) = e ^ 2 + e ^ 2 * u ^ 2 / (g1 - u ^ 2))
    by (field; exact Hn1).
  assert (Hid2 : g2 * e ^ 2 / (g2 - u ^ 2) = e ^ 2 + e ^ 2 * u ^ 2 / (g2 - u ^ 2))
    by (field; exact Hn2).
  assert (Hnn : 0 <= e ^ 2 * u ^ 2) by (apply Rmult_le_pos; apply pow2_ge_0).
  assert (Hinvle : / (g2 - u ^ 2) <= / (g1 - u ^ 2)) by (apply Rinv_le_contravar; lra).
  assert (Hinv : e ^ 2 * u ^ 2 * / (g2 - u ^ 2) <= e ^ 2 * u ^ 2 * / (g1 - u ^ 2))
    by (apply Rmult_le_compat_l; assumption).
  unfold Rdiv in *. lra.
Qed.
