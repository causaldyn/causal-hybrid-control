(* Rocq: THE CONJUGATE-TIME OBSTRUCTION -- the horizon past which every chc.regret constant ceases
   to exist. Third member of the family of Result 13 (the active-set kink, where the constrained CE
   regret's curvature collapses) and Result 14 (the confounded turnpike gap): places where the smooth
   machinery the regret bounds assume quietly stops applying.

   Result 14 says a long horizon is BENIGN -- the discounted confounded gap stays finite. That holds
   for a POSITIVE-DEFINITE stage cost. For an INDEFINITE one (a state that is rewarded rather than
   penalised: growth, market share, an inverted pendulum's angle) the second-order sufficient
   condition fails at a FINITE horizon. The two are the two sides of the definiteness hypothesis, and
   only one of them had been written down.

   Derived in validation/conjugate_time.mac. Continuous-time scalar LQ, x' = a x + b u,
   J = int_0^T (q x^2 + r u^2) dt + sT x(T)^2, backward Riccati -p' = 2 a p - (b^2/r) p^2 + q.
   Writing c := b^2/r and reversing time, the Riccati flow is a UNIFORM ROTATION of the phase
   arctan(c(p - a/c)/mu) at speed mu := sqrt(-(a^2 + c q)), which exists exactly when a^2 + c q < 0.
   The phase runs from phi0 down to -pi/2 in finite time; that time is t_conj.

   Honest scope, as everywhere on this line: Stdlib has no ODEs and no transcendental Riccati theory,
   so what is machine-checked here is the ALGEBRA -- the existence condition, the closed-form escape
   time and its monotonicities, the pole ORDERS (simple for the value, double for the sensitivity,
   fourth-order for the regret constant), and the finiteness of the certified horizon. The flow
   itself is solved and residual-checked in the .mac file, and measured in
   conjugate_time_certificate. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* ---------- (A) When a conjugate point exists at all. ---------- *)

(* c := b^2/r is the control authority; the discriminant of the reverse-time Riccati right-hand side
   is a^2 + c q. Two real equilibria (disc > 0) trap the solution and NOTHING escapes; none (disc<0)
   and the solution sweeps all of R in finite time. *)
Definition disc (a c q : R) : R := a * a + c * q.

Theorem conjugate_needs_indefinite_cost : forall a c q,
  0 < c -> 0 <= q -> 0 <= disc a c q.
Proof. intros a c q Hc Hq. unfold disc. nra. Qed.

(* And indefiniteness alone is not enough: the cost must be indefinite by MORE than the open-loop
   instability can absorb, q < -a^2/c. That threshold is the whole content of the condition. *)
Theorem conjugate_threshold : forall a c q, disc a c q < 0 <-> c * q < - (a * a).
Proof. intros a c q. unfold disc. lra. Qed.

Corollary conjugate_threshold_div : forall a c q,
  0 < c -> (disc a c q < 0 <-> q < - (a * a) / c).
Proof.
  intros a c q Hc. rewrite conjugate_threshold. split; intros H.
  - apply Rmult_lt_reg_r with (r := c); [ exact Hc | ].
    replace (- (a * a) / c * c) with (- (a * a)) by (field; lra). nra.
  - apply Rmult_lt_compat_r with (r := c) in H; [ | exact Hc ].
    replace (- (a * a) / c * c) with (- (a * a)) in H by (field; lra). nra.
Qed.

(* mu^2 = -(a^2 + c q) is the rotation SPEED of the Riccati phase; it is positive exactly there. *)
Definition mu_sq (a c q : R) : R := - disc a c q.

Theorem mu_sq_pos_iff : forall a c q, 0 < mu_sq a c q <-> disc a c q < 0.
Proof. intros a c q. unfold mu_sq. lra. Qed.

(* ---------- (B) The escape time is FINITE, and shrinks with control authority. ---------- *)

(* The phase travels from phi0 down to -pi/2 at constant speed mu, so t_conj = (pi/2 + phi0)/mu.
   Parameterised by mu and phi0 rather than by sqrt/arctan so every goal stays polynomial; the
   .mac file supplies the closed forms and checks the Riccati residual is 0. *)
Definition t_conj (mu phi0 : R) : R := (PI / 2 + phi0) / mu.

Theorem t_conj_finite : forall mu phi0,
  0 < mu -> - (PI / 2) < phi0 -> 0 < t_conj mu phi0.
Proof.
  intros mu phi0 Hmu Hphi. unfold t_conj.
  apply Rdiv_lt_0_compat; lra.
Qed.

(* THE OBSTRUCTION, stated as what it costs: the set of horizons on which the regret constant exists
   is BOUNDED. No amount of extra data buys a horizon past t_conj. *)
Theorem certified_horizon_is_bounded : forall mu phi0 T,
  0 < mu -> - (PI / 2) < phi0 -> T < t_conj mu phi0 -> T < (PI / 2 + phi0) / mu.
Proof. intros mu phi0 T Hmu Hphi HT. unfold t_conj in HT. exact HT. Qed.

(* A FASTER phase (larger mu) means an EARLIER conjugate point. Since mu^2 = -(a^2 + c q), raising
   the control authority c raises mu -- so a stronger actuator brings the obstruction CLOSER, the
   opposite of the reflex that more authority can only help. *)
Theorem t_conj_antitone_in_speed : forall m1 m2 phi0,
  0 < m1 -> m1 <= m2 -> - (PI / 2) < phi0 -> t_conj m2 phi0 <= t_conj m1 phi0.
Proof.
  intros m1 m2 phi0 Hm1 Hle Hphi. unfold t_conj. unfold Rdiv.
  apply Rmult_le_compat_l; [ lra | apply Rinv_le_contravar; lra ].
Qed.

Theorem speed_increases_with_authority : forall a c1 c2 q,
  q < 0 -> c1 <= c2 -> mu_sq a c1 q <= mu_sq a c2 q.
Proof. intros a c1 c2 q Hq Hle. unfold mu_sq, disc. nra. Qed.

(* ---------- (C) The pole orders: simple, double, fourth. ---------- *)

(* Near the escape the tangent is a cotangent in eps, so the VALUE has a simple pole with residue
   -1/c: it depends on neither a, nor q, nor the terminal weight (mac STEP 4c). Modelled here as the
   principal part, which is what the order statements are about. *)
Definition p_pole (c eps : R) : R := - / (c * eps).

Theorem value_pole_residue : forall c eps,
  0 < c -> 0 < eps -> eps * p_pole c eps = - / c.
Proof.
  intros c eps Hc Heps. unfold p_pole.
  rewrite Rinv_mult. field. lra.
Qed.

(* Monotone DOWNWARD as the horizon approaches t_conj: a shorter remaining margin is a more
   negative cost-to-go coefficient, without bound. *)
Theorem value_pole_antitone : forall c e1 e2,
  0 < c -> 0 < e1 -> e1 <= e2 -> p_pole c e1 <= p_pole c e2.
Proof.
  intros c e1 e2 Hc He1 Hle. unfold p_pole.
  apply Ropp_le_contravar. apply Rinv_le_contravar; nra.
Qed.

Theorem value_pole_unbounded : forall c M,
  0 < c -> 0 < M -> exists eps, 0 < eps /\ p_pole c eps < - M.
Proof.
  intros c M Hc HM. exists (/ (c * (M + 1))). split.
  - apply Rinv_0_lt_compat. nra.
  - unfold p_pole.
    replace (c * / (c * (M + 1))) with (/ (M + 1)) by (field; lra).
    rewrite Rinv_inv. lra.
Qed.

(* Differentiating the closed form at fixed horizon turns the tangent into a sec^2, so the
   SENSITIVITY dk/db has a DOUBLE pole (mac STEP 5). This is the object every chc.regret constant is
   built on -- L_K in Result 44, du_star/db in Results 18/20 -- so this is where the damage is. *)
Definition l_k_pole (kappa eps : R) : R := kappa / (eps * eps).

Theorem sensitivity_pole_is_double : forall kappa eps,
  0 < kappa -> 0 < eps -> eps * eps * l_k_pole kappa eps = kappa.
Proof. intros kappa eps Hk He. unfold l_k_pole. field. lra. Qed.

(* The sensitivity outruns the value: their ratio diverges, so the failure is not a rescaling. *)
Theorem sensitivity_dominates_value : forall c kappa eps,
  0 < c -> 0 < kappa -> 0 < eps -> eps < kappa * c ->
  - p_pole c eps < l_k_pole kappa eps.
Proof.
  intros c kappa eps Hc Hk He Hlt. unfold p_pole, l_k_pole.
  rewrite Ropp_involutive.
  assert (Hce : 0 < c * eps) by nra.
  apply Rmult_lt_reg_r with (r := c * eps * (eps * eps));
    [ repeat apply Rmult_lt_0_compat; lra | ].
  replace (/ (c * eps) * (c * eps * (eps * eps))) with (eps * eps) by (field; lra).
  replace (kappa / (eps * eps) * (c * eps * (eps * eps))) with (kappa * c * eps)
    by (field; lra).
  nra.
Qed.

(* Result 44 has C proportional to L_K^2, so a double pole in the sensitivity is a FOURTH-order pole
   in the regret constant: the object the whole C1 line multiplies. *)
Definition c_regret_pole (kap kappa eps : R) : R := kap * (l_k_pole kappa eps) * (l_k_pole kappa eps).

Theorem regret_constant_pole_is_fourth_order : forall kap kappa eps,
  0 < kap -> 0 < kappa -> 0 < eps ->
  eps * eps * eps * eps * c_regret_pole kap kappa eps = kap * kappa * kappa.
Proof.
  intros kap kappa eps Hkap Hk He. unfold c_regret_pole, l_k_pole. field. lra.
Qed.

Theorem regret_constant_unbounded : forall kap kappa M,
  0 < kap -> 0 < kappa -> 0 < M -> exists eps, 0 < eps /\ M < c_regret_pole kap kappa eps.
Proof.
  intros kap kappa M Hkap Hk HM.
  set (d := kap * kappa * kappa / (M + 1)).
  assert (Hd : 0 < d).
  { unfold d. apply Rdiv_lt_0_compat;
      [ repeat apply Rmult_lt_0_compat; lra | lra ]. }
  set (e := Rmin 1 d).
  assert (He0 : 0 < e) by (unfold e; apply Rmin_glb_lt; lra).
  assert (He1 : e <= 1) by apply Rmin_l.
  assert (Hed : e <= d) by apply Rmin_r.
  assert (Ha : e * e <= e) by nra.
  assert (Hb : e * e * e <= e) by nra.
  assert (H4 : e * e * e * e <= e) by nra.
  assert (Hde : e * (M + 1) <= kap * kappa * kappa).
  { apply Rle_trans with (d * (M + 1)).
    - apply Rmult_le_compat_r; [ lra | exact Hed ].
    - unfold d. right. field. lra. }
  exists e. split; [ exact He0 | ].
  unfold c_regret_pole, l_k_pole.
  replace (kap * (kappa / (e * e)) * (kappa / (e * e)))
    with (kap * kappa * kappa / (e * e * e * e)) by (field; lra).
  apply Rmult_lt_reg_r with (r := e * e * e * e);
    [ repeat apply Rmult_lt_0_compat; lra | ].
  replace (kap * kappa * kappa / (e * e * e * e) * (e * e * e * e))
    with (kap * kappa * kappa) by (field; lra).
  nra.
Qed.
