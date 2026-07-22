(* Rocq (CERTIFIED-LIPSCHITZ ROLLOUT-ERROR BOUND): the discrete Gronwall inequality that turns the
   certified Lipschitz constant L of chc.residual.LipschitzResidual into a CERTIFIED trajectory-error /
   pessimism radius. Two Euler rollouts of a hybrid field f = f_known + r, one with a per-step model
   error <= eps, deviate by e_k obeying e_{k+1} <= (1 + L*dt)*e_k + dt*eps, e_0 <= 0. We prove:

     - gronwall_closed_mul : (a-1)*gronwall a b k = b*(a^k - 1)      (exact closed form, no division)
     - gronwall_closed     : gronwall a b k = b*(a^k - 1)/(a-1)      (a <> 1)
     - gronwall_comparison : any sequence obeying the inequality is bounded by gronwall a b k   [MONEY]
     - gronwall_nonneg     : 0 <= a, 0 <= b  =>  0 <= gronwall a b k (a valid nonneg error envelope)
     - gronwall_monotone_error : b1 <= b2  =>  gronwall a b1 k <= gronwall a b2 k
     - rollout_error_bound : e_H <= gronwall (1+L*dt) (dt*eps) H = eps*((1+L*dt)^H - 1)/L

   Derived in validation/lipschitz_rollout.mac (continuous limit eps*(exp(L*T)-1)/L; L->0 limit eps*T).
   HONEST SCOPE: the bound is exp(L*T) -- useful for small L*T (bounded-gain residual, short horizon,
   safety-critical), loose otherwise; a contraction metric (one-sided log-norm mu<0) would remove the
   exponential, which a norm-based Lipschitz constant does not provide. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

Fixpoint gronwall (a b : R) (k : nat) : R :=
  match k with
  | O => 0
  | S j => a * gronwall a b j + b
  end.

(* Exact closed form, division-free (multiply through by a-1). *)
Lemma gronwall_closed_mul : forall (a b : R) (k : nat),
  (a - 1) * gronwall a b k = b * (a ^ k - 1).
Proof.
  intros a b k. induction k as [| k IH].
  - simpl. ring.
  - simpl gronwall.
    replace (a ^ S k) with (a * a ^ k) by (simpl; ring).
    replace ((a - 1) * (a * gronwall a b k + b))
      with (a * ((a - 1) * gronwall a b k) + (a - 1) * b) by ring.
    rewrite IH. ring.
Qed.

Lemma gronwall_closed : forall (a b : R) (k : nat),
  a <> 1 -> gronwall a b k = b * (a ^ k - 1) / (a - 1).
Proof.
  intros a b k Hne.
  assert (Hd : a - 1 <> 0) by (intro H; apply Hne; lra).
  apply Rmult_eq_reg_l with (r := a - 1); [| exact Hd].
  rewrite gronwall_closed_mul. field. exact Hd.
Qed.

(* THE discrete Gronwall bound: any sequence obeying the one-step inequality stays under gronwall. *)
Lemma gronwall_comparison : forall (a b : R) (d : nat -> R) (k : nat),
  0 <= a ->
  d 0%nat <= 0 ->
  (forall j : nat, d (S j) <= a * d j + b) ->
  d k <= gronwall a b k.
Proof.
  intros a b d k Ha H0 Hstep. induction k as [| k IH].
  - simpl. exact H0.
  - simpl gronwall.
    eapply Rle_trans; [apply Hstep |].
    apply Rplus_le_compat_r.
    apply Rmult_le_compat_l; [exact Ha | exact IH].
Qed.

Lemma gronwall_nonneg : forall (a b : R) (k : nat),
  0 <= a -> 0 <= b -> 0 <= gronwall a b k.
Proof.
  intros a b k Ha Hb. induction k as [| k IH].
  - simpl. lra.
  - simpl. apply Rplus_le_le_0_compat; [apply Rmult_le_pos; assumption | exact Hb].
Qed.

Lemma gronwall_monotone_error : forall (a b1 b2 : R) (k : nat),
  0 <= a -> b1 <= b2 -> gronwall a b1 k <= gronwall a b2 k.
Proof.
  intros a b1 b2 k Ha Hb. induction k as [| k IH].
  - simpl. lra.
  - simpl. apply Rplus_le_compat.
    + apply Rmult_le_compat_l; [exact Ha | exact IH].
    + exact Hb.
Qed.

(* Capstone: the rollout deviation with a = 1 + L*dt, b = dt*eps. *)
Lemma rollout_error_bound : forall (L dt eps : R) (e : nat -> R) (H : nat),
  0 <= L -> 0 <= dt ->
  e 0%nat <= 0 ->
  (forall j : nat, e (S j) <= (1 + L * dt) * e j + dt * eps) ->
  e H <= gronwall (1 + L * dt) (dt * eps) H.
Proof.
  intros L dt eps e H HL Hdt H0 Hstep.
  apply gronwall_comparison; try assumption.
  assert (0 <= L * dt) by (apply Rmult_le_pos; assumption). lra.
Qed.
