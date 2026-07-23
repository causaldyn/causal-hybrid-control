(* Rocq (FIX Result 30, reviewer-7): the explicit-Euler contraction factor. A negative one-sided
   Lipschitz mu = mu_2(J_f) < 0 does NOT give the discrete factor 1 + mu*dt; with the full Lipschitz L,
   |d + dt(f(x)-f(y))|^2 <= (1 + 2*mu*dt + L^2*dt^2)*|d|^2 (validation/contractive_euler.mac). We prove
   the two facts that make q(dt) = sqrt(1 + 2*mu*dt + L^2*dt^2) a valid Gronwall factor a in [0,1):
     - euler_factor_nonneg: qsq >= 0 for ALL dt (the quadratic in dt has discriminant 4(mu^2-L^2) <= 0,
       since the full Lipschitz dominates the one-sided constant, |mu| <= L), so q is real;
     - euler_factor_lt_one: qsq < 1 under the step condition dt < -2*mu/L^2 (mu < 0), so q < 1.
   Then chc.uncertainty.contractive_rollout_bound uses a = q with the existing gronwall_bounded
   (lipschitz_rollout.v: 0 <= a < 1 => radius <= b/(1-a)). The old CFL dt <= 1/|mu| was optimistic. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* qsq >= 0 always (q is real): SOS identity L^2*qsq = (L^2*dt + mu)^2 + (L^2 - mu^2), both terms >= 0
   because the full Lipschitz dominates the one-sided constant (mu^2 <= L^2). *)
Lemma euler_factor_nonneg :
  forall mu L dt : R, 0 < L -> mu ^ 2 <= L ^ 2 -> 0 <= 1 + 2 * mu * dt + L ^ 2 * dt ^ 2.
Proof.
  intros mu L dt HL Hmu.
  assert (Hpos : 0 < L ^ 2) by nra.
  apply Rmult_le_reg_l with (r := L ^ 2); [ exact Hpos |].
  rewrite Rmult_0_r.
  assert (Hid : L ^ 2 * (1 + 2 * mu * dt + L ^ 2 * dt ^ 2) = (L ^ 2 * dt + mu) ^ 2 + (L ^ 2 - mu ^ 2))
    by ring.
  rewrite Hid. apply Rplus_le_le_0_compat; [ apply pow2_ge_0 | lra ].
Qed.

(* qsq < 1 under the step condition dt < -2*mu/L^2 (mu < 0): the sharp Euler CFL for contraction. *)
Lemma euler_factor_lt_one :
  forall mu L dt : R,
    mu < 0 -> 0 < L -> 0 < dt -> dt < - 2 * mu / L ^ 2 ->
    1 + 2 * mu * dt + L ^ 2 * dt ^ 2 < 1.
Proof.
  intros mu L dt Hmu HL Hdt Hstep.
  assert (Hpos : 0 < L ^ 2) by nra.
  assert (Hcl : L ^ 2 * dt < - 2 * mu).
  { replace (- 2 * mu) with (L ^ 2 * (- 2 * mu / L ^ 2)) by (field; lra).
    apply Rmult_lt_compat_l; [ exact Hpos | exact Hstep ]. }
  assert (Hneg : 2 * mu + L ^ 2 * dt < 0) by lra.
  nra.
Qed.
