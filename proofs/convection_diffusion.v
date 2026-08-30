(* Rocq: THE CELL-PECLET THRESHOLD -- when a Galerkin scheme changes the sign structure of the
   solution, and what a Petrov-Galerkin test space buys back.

   chc.galerkin solves the SYMMETRIC operator -u'' = f, where testing with the trial space
   (Bubnov-Galerkin) is the right choice and the stiffness matrix is symmetric positive definite.
   Adding advection, -eps u'' + s u' = 0, destroys the symmetry: the convection stencil is
   ANTISYMMETRIC (+-s/2), and above a threshold in the mesh the scheme stops being merely inaccurate
   and starts producing node-to-node oscillations that the exact solution does not have.

   Writing the discrete solution as u_i = A + B r^i, everything turns on the sign of one number:

       r_h(Pe) = (1 + Pe) / (1 - Pe),   Pe := s*h/(2*eps)   (the cell Peclet number)

   against the exact per-cell amplification exp(2*Pe), which is positive for every Pe. So:

   - r_h < 0 exactly when Pe > 1 (amp_negative_above_threshold, threshold_is_exactly_one), and a
     negative ratio makes CONSECUTIVE FORWARD DIFFERENCES ALTERNATE IN SIGN
     (alternates_when_amplification_negative). That is the oscillation, derived rather than observed.
   - Below the threshold the same product is nonnegative: no alternation (no_alternation_below_threshold).
   - The Petrov-Galerkin test space w_i = phi_i + alpha*(h/2)*phi_i' adds exactly alpha*s*h/2 of
     diffusion, i.e. replaces Pe by Pe/(1 + alpha*Pe). Full upwind (alpha = 1) lands strictly below 1
     for every Pe (upwind_unconditionally_stable) -- but its amplification is EXACTLY 1 + 2*Pe
     (upwind_amplification), the first two terms of exp(2*Pe), so it is first-order and strictly
     under-shoots the truth (upwind_first_order).
   - Any target amplification is reachable by one explicit alpha (optimal_alpha_hits_target); it is
     always inside the stability range (optimal_alpha_is_stable) and, whenever the target beats full
     upwind's, strictly less diffusive than full upwind (optimal_alpha_below_full_upwind).

   Derived in validation/convection_diffusion.mac; certified in
   chc.galerkin.convection_diffusion_certificate. Honest scope: the OPTIMAL target is tanh(Pe), and
   "coth(Pe) - 1/Pe" is therefore transcendental -- that identity stays in Maxima. Rocq carries the
   algebra, so every theorem here is stated for an abstract target t and holds for tanh(Pe) as one
   instance among many. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.

Open Scope R_scope.

(* ---------- the discrete scheme ---------- *)

(* Node equation of the hat-function scheme, divided through by eps/h and written in Pe:
   -(1+Pe) u_{i-1} + 2 u_i + (Pe-1) u_{i+1} = 0.  Seeking u_i = r^i gives this polynomial. *)
Definition chareq (Pe r : R) : R := (Pe - 1) * r ^ 2 + 2 * r - (Pe + 1).

(* Its nonunit root: the amplification of the discrete solution across one cell. *)
Definition amp (Pe : R) : R := (1 + Pe) / (1 - Pe).

(* The Petrov-Galerkin test space contributes alpha*s*h/2 of extra diffusion, so the scheme behaves
   as if the cell Peclet number were this. *)
Definition eff_peclet (a Pe : R) : R := Pe / (1 + a * Pe).

Theorem unit_mode_is_a_root : forall Pe, chareq Pe 1 = 0.
Proof. intros. unfold chareq. ring. Qed.

Theorem amp_is_a_root : forall Pe, Pe <> 1 -> chareq Pe (amp Pe) = 0.
Proof.
  intros Pe H.
  assert (Hne : 1 - Pe <> 0) by (intro Hc; apply H; lra).
  unfold chareq, amp. field. exact Hne.
Qed.

(* ---------- the threshold ---------- *)

Theorem amp_negative_above_threshold : forall Pe, 1 < Pe -> amp Pe < 0.
Proof.
  intros Pe H. unfold amp, Rdiv.
  assert (Hinv : / (1 - Pe) < 0) by (apply Rinv_lt_0_compat; lra).
  nra.
Qed.

Theorem amp_positive_below_threshold : forall Pe, 0 <= Pe -> Pe < 1 -> 0 < amp Pe.
Proof. intros Pe H0 H1. unfold amp. apply Rdiv_lt_0_compat; lra. Qed.

(* The threshold is sharp: the sign of the discrete amplification flips at Pe = 1 and nowhere else.
   This is what the certificate's bisection measures, and it lands on 1.0000000000. *)
Theorem threshold_is_exactly_one :
  forall Pe, 0 <= Pe -> Pe <> 1 -> (amp Pe < 0 <-> 1 < Pe).
Proof.
  intros Pe H0 Hne. split.
  - intros Hneg. destruct (Rtotal_order Pe 1) as [Hlt | [Heq | Hgt]].
    + assert (0 < amp Pe) by (apply amp_positive_below_threshold; lra). lra.
    + exfalso; apply Hne; exact Heq.
    + exact Hgt.
  - apply amp_negative_above_threshold.
Qed.

(* The exact per-cell amplification is exp(2*Pe), positive for EVERY Pe. So above the threshold the
   discrete scheme does not merely lose accuracy, it gets the sign structure wrong. *)
Theorem exact_amplification_positive : forall Pe, 0 < exp (2 * Pe).
Proof. intros. apply exp_pos. Qed.

Theorem discrete_sign_is_wrong_above_threshold :
  forall Pe, 1 < Pe -> amp Pe < 0 /\ 0 < exp (2 * Pe).
Proof.
  intros Pe H. split; [apply amp_negative_above_threshold; exact H | apply exp_pos].
Qed.

(* ---------- what a negative amplification does to the nodal values ---------- *)

Definition mode (A B r : R) (i : nat) : R := A + B * r ^ i.
Definition fwd_diff (A B r : R) (i : nat) : R := mode A B r (S i) - mode A B r i.

(* Two consecutive forward differences.  Its SIGN is the oscillation: negative means the solution
   turned around between i and i+1, which the monotone exact solution never does. *)
Definition consecutive_product (A B r : R) (i : nat) : R :=
  fwd_diff A B r (S i) * fwd_diff A B r i.

Lemma fwd_diff_closed : forall A B r i, fwd_diff A B r i = B * r ^ i * (r - 1).
Proof. intros. unfold fwd_diff, mode. simpl. ring. Qed.

Lemma consecutive_product_closed :
  forall A B r i,
  consecutive_product A B r i = B * B * ((r - 1) * (r - 1)) * (r * (r ^ i * r ^ i)).
Proof.
  intros. unfold consecutive_product. rewrite !fwd_diff_closed. simpl. ring.
Qed.

Lemma sq_pos : forall x, x <> 0 -> 0 < x * x.
Proof.
  intros x Hx. destruct (Rtotal_order x 0) as [H | [H | H]].
  - nra.
  - exfalso; apply Hx; exact H.
  - nra.
Qed.

Lemma sq_nonneg : forall x, 0 <= x * x.
Proof. intros x. destruct (Rtotal_order x 0) as [H | [H | H]]; nra. Qed.

(* THE OSCILLATION.  A negative amplification forces every pair of consecutive differences to have
   opposite signs -- at every node, for every amplitude.  No smallness, no asymptotics. *)
Theorem alternates_when_amplification_negative :
  forall A B r i, B <> 0 -> r < 0 -> consecutive_product A B r i < 0.
Proof.
  intros A B r i HB Hr.
  rewrite consecutive_product_closed.
  assert (HBB : 0 < B * B) by (apply sq_pos; exact HB).
  assert (Hr1 : 0 < (r - 1) * (r - 1)) by (apply sq_pos; lra).
  assert (Hpi : r ^ i <> 0) by (apply pow_nonzero; lra).
  assert (Hpp : 0 < r ^ i * r ^ i) by (apply sq_pos; exact Hpi).
  assert (Hpos : 0 < B * B * ((r - 1) * (r - 1)) * (r ^ i * r ^ i)).
  { apply Rmult_lt_0_compat; [apply Rmult_lt_0_compat; assumption | exact Hpp]. }
  replace (B * B * ((r - 1) * (r - 1)) * (r * (r ^ i * r ^ i)))
    with (r * (B * B * ((r - 1) * (r - 1)) * (r ^ i * r ^ i))) by ring.
  assert (Hneg := Rmult_lt_gt_compat_neg_l _ _ _ Hr Hpos). lra.
Qed.

(* And below the threshold there is no alternation at all -- the same expression is nonnegative for
   every amplitude, including the degenerate amplitudes the theorem above has to exclude. *)
Theorem no_alternation_below_threshold :
  forall A B r i, 0 <= r -> 0 <= consecutive_product A B r i.
Proof.
  intros A B r i Hr.
  rewrite consecutive_product_closed.
  assert (Hpi : 0 <= r ^ i) by (apply pow_le; exact Hr).
  assert (H1 : 0 <= B * B) by apply sq_nonneg.
  assert (H2 : 0 <= (r - 1) * (r - 1)) by apply sq_nonneg.
  assert (H3 : 0 <= r ^ i * r ^ i) by apply sq_nonneg.
  replace (B * B * ((r - 1) * (r - 1)) * (r * (r ^ i * r ^ i)))
    with (B * B * ((r - 1) * (r - 1)) * (r ^ i * r ^ i) * r) by ring.
  apply Rmult_le_pos; [apply Rmult_le_pos; [apply Rmult_le_pos | ] | ]; assumption.
Qed.

(* Composed with the threshold: above Pe = 1 the Bubnov-Galerkin solution oscillates, below it does
   not.  This pair is what the certificate asserts on both sides of Pe = 1. *)
Theorem oscillation_above_threshold :
  forall Pe A B i, 1 < Pe -> B <> 0 -> consecutive_product A B (amp Pe) i < 0.
Proof.
  intros Pe A B i HPe HB.
  apply alternates_when_amplification_negative; [exact HB | apply amp_negative_above_threshold; exact HPe].
Qed.

Theorem monotone_below_threshold :
  forall Pe A B i, 0 <= Pe -> Pe < 1 -> 0 <= consecutive_product A B (amp Pe) i.
Proof.
  intros Pe A B i H0 H1.
  apply no_alternation_below_threshold.
  left. apply amp_positive_below_threshold; assumption.
Qed.

(* ---------- the Petrov-Galerkin cure ---------- *)

(* Full upwind lands strictly below the threshold for EVERY Pe: unconditionally non-oscillatory. *)
Theorem upwind_unconditionally_stable : forall Pe, 0 < Pe -> eff_peclet 1 Pe < 1.
Proof.
  intros Pe H. unfold eff_peclet.
  apply Rmult_lt_reg_r with (r := 1 + 1 * Pe); [lra | ].
  unfold Rdiv. rewrite Rmult_assoc, Rinv_l by lra. lra.
Qed.

(* But its amplification is EXACTLY 1 + 2*Pe -- the first two terms of exp(2*Pe) and nothing more. *)
Theorem upwind_amplification : forall Pe, 0 < Pe -> amp (eff_peclet 1 Pe) = 1 + 2 * Pe.
Proof.
  intros Pe H. unfold amp, eff_peclet.
  assert (H1 : 1 + 1 * Pe <> 0) by lra.
  field_simplify; [reflexivity | lra].
Qed.

(* So full upwind buys stability by systematically under-shooting the exact amplification.  That gap
   is the classical excess numerical diffusion, and it is why the optimal alpha is worth computing
   rather than defaulting to 1. *)
Theorem upwind_first_order : forall Pe, 0 < Pe -> amp (eff_peclet 1 Pe) < exp (2 * Pe).
Proof.
  intros Pe H. rewrite upwind_amplification by exact H.
  assert (Hne : 2 * Pe <> 0) by lra.
  assert (Hex := exp_ineq1 (2 * Pe) Hne). lra.
Qed.

(* ---------- choosing alpha ---------- *)

(* Non-oscillatory means the effective Peclet is below 1, and that is exactly a lower bound on alpha.
   At Pe <= 1 the bound is <= 0, so plain Bubnov-Galerkin already qualifies: the threshold again. *)
Theorem stability_range :
  forall a Pe, 0 < Pe -> 0 < 1 + a * Pe -> (eff_peclet a Pe < 1 <-> 1 - / Pe < a).
Proof.
  intros a Pe HPe Hden. unfold eff_peclet.
  assert (Hinv : Pe * / Pe = 1) by (apply Rinv_r; lra).
  assert (Hpivot : Pe / (1 + a * Pe) < 1 <-> Pe < 1 + a * Pe).
  { split.
    - intros H. apply Rmult_lt_compat_r with (r := 1 + a * Pe) in H; [ | exact Hden].
      unfold Rdiv in H. rewrite Rmult_assoc, Rinv_l in H by lra. lra.
    - intros H. apply Rmult_lt_reg_r with (r := 1 + a * Pe); [exact Hden | ].
      unfold Rdiv. rewrite Rmult_assoc, Rinv_l by lra. lra. }
  rewrite Hpivot. split.
  - intros H. apply Rmult_lt_reg_r with (r := Pe); [exact HPe | ]. nra.
  - intros H. apply Rmult_lt_compat_r with (r := Pe) in H; [ | exact HPe]. nra.
Qed.

(* Any target amplification is reachable, by one explicit alpha.  The optimal choice is the target
   tanh(Pe), which makes the scheme nodally exact; that identity is transcendental and lives in
   Maxima, so this theorem is stated for an abstract t and applies to it as one instance. *)
Lemma optimal_alpha_denominator :
  forall t Pe, 0 < Pe -> 0 < t -> 1 + (/ t - / Pe) * Pe = Pe / t.
Proof. intros t Pe HPe Ht. field. split; lra. Qed.

Theorem optimal_alpha_hits_target :
  forall t Pe, 0 < Pe -> 0 < t -> eff_peclet (/ t - / Pe) Pe = t.
Proof.
  intros t Pe HPe Ht. unfold eff_peclet.
  rewrite optimal_alpha_denominator by assumption.
  field. split; lra.
Qed.

(* It is always inside the stability range, for any target strictly below the threshold. *)
Theorem optimal_alpha_is_stable :
  forall t Pe, 0 < Pe -> 0 < t -> t < 1 -> 1 - / Pe < / t - / Pe.
Proof.
  intros t Pe HPe Ht Ht1.
  assert (H : 1 < / t) by (rewrite <- Rinv_1; apply Rinv_lt_contravar; lra).
  lra.
Qed.

(* And whenever the target beats full upwind's amplification, the optimal scheme is STRICTLY LESS
   diffusive than full upwind.  The certificate measures alpha = 0.6136 at Pe = 2.5. *)
Theorem optimal_alpha_below_full_upwind :
  forall t Pe, 0 < Pe -> 0 < t -> Pe / (1 + Pe) < t -> / t - / Pe < 1.
Proof.
  intros t Pe HPe Ht Hbeat.
  assert (Hden : 0 < 1 + Pe) by lra.
  assert (Hstep : / t < / (Pe / (1 + Pe))).
  { apply Rinv_lt_contravar; [ | exact Hbeat].
    apply Rmult_lt_0_compat; [apply Rdiv_lt_0_compat; lra | exact Ht]. }
  assert (Hval : / (Pe / (1 + Pe)) = 1 + / Pe).
  { field; lra. }
  lra.
Qed.
