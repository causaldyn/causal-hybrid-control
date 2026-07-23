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

(* CONTRACTION payoff: for a CONTRACTING step 0 <= a < 1 (a = 1 + mu*dt, one-sided Lipschitz mu < 0),
   the Gronwall bound is UNIFORMLY BOUNDED by b/(1-a) for all horizons k -- no e^{L*T} blow-up. With
   b = dt*eps and 1-a = |mu|*dt this is eps/|mu|: a flat certified pessimism radius. *)
Lemma gronwall_bounded : forall (a b : R) (k : nat),
  0 <= a -> a < 1 -> 0 <= b -> gronwall a b k <= b / (1 - a).
Proof.
  intros a b k Ha Ha1 Hb.
  assert (H1a : 0 < 1 - a) by lra.
  assert (Hak0 : 0 <= a ^ k) by (apply pow_le; lra).
  assert (Hak1 : a ^ k <= 1) by (rewrite <- (pow1 k); apply pow_incr; lra).
  apply Rmult_le_reg_r with (r := 1 - a); [lra |].
  replace (b / (1 - a) * (1 - a)) with b by (field; lra).
  (* gronwall a b k * (1-a) = -(a-1)*gronwall = -(b*(a^k-1)) = b*(1-a^k) <= b *)
  assert (Hmul : gronwall a b k * (1 - a) = b * (1 - a ^ k)).
  { pose proof (gronwall_closed_mul a b k) as Hc. nra. }
  rewrite Hmul. nra.
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

(* Result 34 (confounding-robust closed loop): the §32 MSM confounding inflation, added to the per-step
   error budget, widens the §31 CLOSED-LOOP tube (rate a = 1 + (L_x + L_u*L_pi)*dt) monotonically. More
   assumed confounding -> a weakly larger rollout radius at every horizon, hence a shorter
   certified-safe horizon. Reuses gronwall_monotone_error; infl1 <= infl2 is §32 msm_inflation_monotone
   output, so this composes the confounding radius (§32) with the closed-loop replan tube (§31). *)
Lemma closed_loop_confounding_monotone :
  forall (Lcl dt base infl1 infl2 : R) (k : nat),
    0 <= Lcl -> 0 <= dt -> infl1 <= infl2 ->
    gronwall (1 + Lcl * dt) (dt * (base + infl1)) k
      <= gronwall (1 + Lcl * dt) (dt * (base + infl2)) k.
Proof.
  intros Lcl dt base i1 i2 k HL Hdt Hi. apply gronwall_monotone_error.
  - assert (0 <= Lcl * dt) by (apply Rmult_le_pos; assumption). lra.
  - apply Rmult_le_compat_l; lra.
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

(* ---- TIME-VARYING extension (Result 28 upgrade): per-step L_k and eps_k. ---- *)
(* The bound sequence with per-step coefficients a k = 1 + L_k*dt, b k = dt*eps_k. Shows WHICH step
   and WHICH channel dominates the accumulated uncertainty -- the basis for `certified_until_step`. *)
Fixpoint gronwall_var (a b : nat -> R) (k : nat) : R :=
  match k with
  | O => 0
  | S j => a j * gronwall_var a b j + b j
  end.

Lemma gronwall_var_comparison : forall (a b : nat -> R) (d : nat -> R) (k : nat),
  (forall j : nat, 0 <= a j) ->
  d 0%nat <= 0 ->
  (forall j : nat, d (S j) <= a j * d j + b j) ->
  d k <= gronwall_var a b k.
Proof.
  intros a b d k Ha H0 Hstep. induction k as [| k IH].
  - simpl. exact H0.
  - simpl gronwall_var.
    eapply Rle_trans; [apply Hstep |].
    apply Rplus_le_compat_r.
    apply Rmult_le_compat_l; [apply Ha | exact IH].
Qed.

Lemma gronwall_var_nonneg : forall (a b : nat -> R) (k : nat),
  (forall j : nat, 0 <= a j) -> (forall j : nat, 0 <= b j) -> 0 <= gronwall_var a b k.
Proof.
  intros a b k Ha Hb. induction k as [| k IH].
  - simpl. lra.
  - simpl. apply Rplus_le_le_0_compat; [apply Rmult_le_pos; [apply Ha | exact IH] | apply Hb].
Qed.

(* ---- SAFETY-CONSTRAINT tightening (Result 28 upgrade): the error tube -> a safe plan. ---- *)
(* If g is L_g-Lipschitz and the NOMINAL trajectory satisfies the TIGHTENED constraint
   g(x_hat) + L_g*e <= 0 (e = the certified error radius), then the TRUE trajectory is feasible:
   g(x) <= g(x_hat) + L_g*||x - x_hat|| <= g(x_hat) + L_g*e <= 0. *)
Lemma constraint_tightening : forall gx ghat lg e : R,
  gx <= ghat + lg * e ->  (* Lipschitz upper bound with ||x - x_hat|| <= e *)
  ghat + lg * e <= 0 ->   (* the tightened nominal constraint *)
  gx <= 0.
Proof. intros gx ghat lg e Hlip Htight. lra. Qed.
