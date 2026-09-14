(* Rocq: the algebraic core of A COERCIVE ENERGY FOR THE PORT-HAMILTONIAN RESIDUAL (Result 68,
   Appendix A item A20).

   chc.residual.PortHamiltonianResidual encodes x' = (J - R) grad H(x) + g(x) u and claims that
   because H' = -grad H . R grad H <= 0, "the residual can't blow up off-support like a black box
   can". The inequality is an identity and is proved here. The conclusion is not implied by it:
   H' <= 0 confines the state to the sublevel set { H <= H(x0) }, which is a bound on the state only
   when that set is BOUNDED, and the shipped tanh-MLP energy is bounded in x, so its high sublevel
   sets are the whole space.

   What is proved here, over Stdlib's reals:
     (A) the passivity identity, and the skew term that cancels inside it;
     (B) a bounded energy has an unbounded sublevel set -- and no finite invariant radius exists;
     (C) a quadratic floor H(x) >= (eps/2) x^2 turns the same inequality into an explicit ball, and
         the radius sqrt(2c/eps) is attained rather than merely sufficient;
     (D) (J - R) is invertible as soon as the interconnection is non-trivial, so equilibria of the
         unforced flow are exactly the critical points of H;
     (E) the ICNN rule -- a convex non-decreasing activation composed with a nonnegative
         combination of convex functions -- and softplus meeting both halves of it;
     (F) strong convexity making the critical point unique, without a search.

   Honest scope: everything here is scalar or 2x2 with explicit entries, as in the rest of
   proofs/. The n-dimensional Hessian statements stay in validation/convex_port_hamiltonian.mac and
   in the certificate, which measures them on the trained network rather than assuming them. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Psatz.
Open Scope R_scope.

Lemma square_zero : forall x : R, x * x = 0 -> x = 0.
Proof.
  intros x H. destruct (Rmult_integral _ _ H); assumption.
Qed.

(* ===== (A) THE PASSIVITY IDENTITY ===== *)

(* The co-energy vector paired with the flow, at 2x2: J = [[0, a], [-a, 0]] is skew by
   construction and R = [[r1, r12], [r12, r2]] is the symmetric dissipation. *)
Definition energy_rate (a r1 r12 r2 g1 g2 h1 h2 u : R) : R :=
  h1 * ((0 - r1) * h1 + (a - r12) * h2 + g1 * u)
  + h2 * ((- a - r12) * h1 + (0 - r2) * h2 + g2 * u).

Definition dissipated (r1 r12 r2 h1 h2 : R) : R :=
  r1 * h1 ^ 2 + 2 * r12 * h1 * h2 + r2 * h2 ^ 2.

Lemma the_skew_term_cancels :
  forall a h1 h2 : R, h1 * (a * h2) + h2 * (- a * h1) = 0.
Proof.
  intros. ring.
Qed.

Lemma the_energy_rate_is_supply_minus_dissipation :
  forall a r1 r12 r2 g1 g2 h1 h2 u : R,
  energy_rate a r1 r12 r2 g1 g2 h1 h2 u
  = (g1 * h1 + g2 * h2) * u - dissipated r1 r12 r2 h1 h2.
Proof.
  intros. unfold energy_rate, dissipated. ring.
Qed.

Lemma a_psd_dissipation_never_adds_energy :
  forall r1 r12 r2 h1 h2 : R,
  0 <= r1 -> 0 <= r2 -> 0 <= r1 * r2 - r12 ^ 2 ->
  0 <= dissipated r1 r12 r2 h1 h2.
Proof.
  intros r1 r12 r2 h1 h2 H1 H2 Hdet.
  unfold dissipated.
  destruct (Rle_lt_or_eq_dec 0 r1 H1) as [Hpos | Hzero].
  - assert (Hsplit : r1 * h1 ^ 2 + 2 * r12 * h1 * h2 + r2 * h2 ^ 2
                     = r1 * (h1 + r12 / r1 * h2) ^ 2 + (r1 * r2 - r12 ^ 2) / r1 * h2 ^ 2)
      by (field; lra).
    assert (Hhead : 0 <= r1 * (h1 + r12 / r1 * h2) ^ 2)
      by (apply Rmult_le_pos; [lra | apply pow2_ge_0]).
    assert (Htail : 0 <= (r1 * r2 - r12 ^ 2) / r1 * h2 ^ 2).
    { apply Rmult_le_pos; [| apply pow2_ge_0].
      unfold Rdiv. apply Rmult_le_pos; [exact Hdet | left; apply Rinv_0_lt_compat; lra]. }
    lra.
  - assert (Hr12 : r12 = 0) by nra.
    rewrite Hr12, <- Hzero.
    assert (0 <= r2 * h2 ^ 2) by (apply Rmult_le_pos; [exact H2 | apply pow2_ge_0]).
    lra.
Qed.

Lemma the_unforced_energy_is_nonincreasing :
  forall a r1 r12 r2 g1 g2 h1 h2 : R,
  0 <= r1 -> 0 <= r2 -> 0 <= r1 * r2 - r12 ^ 2 ->
  energy_rate a r1 r12 r2 g1 g2 h1 h2 0 <= 0.
Proof.
  intros a r1 r12 r2 g1 g2 h1 h2 H1 H2 Hdet.
  rewrite the_energy_rate_is_supply_minus_dissipation.
  assert (0 <= dissipated r1 r12 r2 h1 h2)
    by (apply a_psd_dissipation_never_adds_energy; assumption).
  lra.
Qed.

(* ===== (B) A BOUNDED ENERGY CERTIFIES NOTHING ABOUT THE STATE ===== *)

Lemma a_bounded_energy_has_an_unbounded_sublevel_set :
  forall (energy : R -> R) (ceiling level x : R),
  (forall y : R, energy y <= ceiling) -> ceiling <= level -> energy x <= level.
Proof.
  intros energy ceiling level x Hbound Hlevel.
  apply Rle_trans with (r2 := ceiling); [apply Hbound | exact Hlevel].
Qed.

(* The same fact stated the way the certificate reports it: for a bounded energy, no radius is
   invariant, so the honest number to print is infinity rather than a missing field. *)
Lemma a_bounded_energy_admits_no_finite_radius :
  forall (energy : R -> R) (ceiling radius : R),
  (forall y : R, energy y <= ceiling) ->
  exists x : R, radius < Rabs x /\ energy x <= ceiling.
Proof.
  intros energy ceiling radius Hbound.
  exists (Rabs radius + 1).
  split; [| apply Hbound].
  assert (Habs : Rabs (Rabs radius + 1) = Rabs radius + 1).
  { apply Rabs_right. assert (0 <= Rabs radius) by apply Rabs_pos. lra. }
  rewrite Habs.
  assert (radius <= Rabs radius) by apply Rle_abs.
  lra.
Qed.

(* ===== (C) A QUADRATIC FLOOR TURNS THE INEQUALITY INTO A BALL ===== *)

Lemma a_quadratic_floor_bounds_the_sublevel_set :
  forall (energy : R -> R) (eps level x : R),
  0 < eps ->
  (forall y : R, eps / 2 * y ^ 2 <= energy y) ->
  energy x <= level ->
  Rabs x <= sqrt (2 * level / eps).
Proof.
  intros energy eps level x Heps Hfloor Hlevel.
  assert (Hx : x ^ 2 <= 2 * level / eps).
  { assert (Hchain : eps / 2 * x ^ 2 <= level)
      by (apply Rle_trans with (r2 := energy x); [apply Hfloor | exact Hlevel]).
    apply (Rmult_le_reg_l (eps / 2)); [lra |].
    replace (eps / 2 * (2 * level / eps)) with level by (field; lra).
    exact Hchain. }
  rewrite <- sqrt_Rsqr_abs.
  apply sqrt_le_1; [apply Rle_0_sqr | | unfold Rsqr; nra].
  assert (0 <= x ^ 2) by apply pow2_ge_0. lra.
Qed.

Lemma the_invariant_radius_is_attained :
  forall eps level : R,
  0 < eps -> 0 <= level -> eps / 2 * sqrt (2 * level / eps) ^ 2 = level.
Proof.
  intros eps level Heps Hlevel.
  assert (Hq : 0 <= 2 * level / eps).
  { unfold Rdiv. apply Rmult_le_pos; [lra | left; apply Rinv_0_lt_compat; exact Heps]. }
  replace (sqrt (2 * level / eps) ^ 2)
    with (sqrt (2 * level / eps) * sqrt (2 * level / eps)) by ring.
  rewrite sqrt_sqrt by exact Hq.
  field. lra.
Qed.

(* ===== (D) EQUILIBRIA OF THE UNFORCED FLOW ARE THE CRITICAL POINTS OF H ===== *)

Lemma skew_minus_dissipation_determinant :
  forall a r1 r12 r2 : R,
  (0 - r1) * (0 - r2) - (a - r12) * (- a - r12) = r1 * r2 - r12 ^ 2 + a ^ 2.
Proof.
  intros. ring.
Qed.

Lemma the_interconnection_alone_makes_it_invertible :
  forall a r1 r12 r2 : R,
  0 <= r1 * r2 - r12 ^ 2 -> a <> 0 -> 0 < r1 * r2 - r12 ^ 2 + a ^ 2.
Proof.
  intros a r1 r12 r2 Hdet Ha.
  assert (0 < a ^ 2) by (destruct (Rdichotomy a 0 Ha); nra).
  lra.
Qed.

Lemma a_nonsingular_structure_forces_a_critical_point :
  forall a r1 r12 r2 h1 h2 : R,
  r1 * r2 - r12 ^ 2 + a ^ 2 <> 0 ->
  (0 - r1) * h1 + (a - r12) * h2 = 0 ->
  (- a - r12) * h1 + (0 - r2) * h2 = 0 ->
  h1 = 0 /\ h2 = 0.
Proof.
  intros a r1 r12 r2 h1 h2 Hdet H1 H2.
  (* the two rows eliminated against each other: a linear combination with variable coefficients,
     which is why it is written out rather than left to nra *)
  assert (Hh1 : (r1 * r2 - r12 ^ 2 + a ^ 2) * h1 = 0).
  { replace ((r1 * r2 - r12 ^ 2 + a ^ 2) * h1)
      with ((0 - r2) * ((0 - r1) * h1 + (a - r12) * h2)
            - (a - r12) * ((- a - r12) * h1 + (0 - r2) * h2)) by ring.
    rewrite H1, H2. ring. }
  assert (Hh2 : (r1 * r2 - r12 ^ 2 + a ^ 2) * h2 = 0).
  { replace ((r1 * r2 - r12 ^ 2 + a ^ 2) * h2)
      with ((0 - r1) * ((- a - r12) * h1 + (0 - r2) * h2)
            - (- a - r12) * ((0 - r1) * h1 + (a - r12) * h2)) by ring.
    rewrite H1, H2. ring. }
  split.
  - destruct (Rmult_integral _ _ Hh1); [contradiction | assumption].
  - destruct (Rmult_integral _ _ Hh2); [contradiction | assumption].
Qed.

(* ===== (E) THE ICNN RULE ===== *)

(* One layer's second derivative: sigma''(phi) phi'^2 + sigma'(phi) phi''. Convexity survives when
   sigma is convex (first term) AND non-decreasing with phi convex (second). *)
Lemma a_convex_increasing_activation_preserves_convexity :
  forall second_sigma first_sigma slope second_phi : R,
  0 <= second_sigma -> 0 <= first_sigma -> 0 <= second_phi ->
  0 <= second_sigma * slope ^ 2 + first_sigma * second_phi.
Proof.
  intros s2 s1 slope p2 Hs2 Hs1 Hp2.
  assert (0 <= s2 * slope ^ 2) by (apply Rmult_le_pos; [exact Hs2 | apply pow2_ge_0]).
  assert (0 <= s1 * p2) by (apply Rmult_le_pos; assumption).
  lra.
Qed.

Lemma a_nonnegative_combination_of_convex_is_convex :
  forall w1 w2 z1 z2 : R,
  0 <= w1 -> 0 <= w2 -> 0 <= z1 -> 0 <= z2 -> 0 <= w1 * z1 + w2 * z2.
Proof.
  intros. nra.
Qed.

Lemma softplus_slope_is_a_probability :
  forall t : R, 0 < exp t / (1 + exp t) < 1.
Proof.
  intros t.
  assert (Hpos : 0 < exp t) by apply exp_pos.
  assert (Hden : 0 < 1 + exp t) by lra.
  split.
  - apply Rdiv_lt_0_compat; assumption.
  - apply (Rmult_lt_reg_r (1 + exp t)); [exact Hden |].
    unfold Rdiv. rewrite Rmult_assoc, Rinv_l by lra. lra.
Qed.

Lemma softplus_is_strictly_convex :
  forall t : R, 0 < exp t / (1 + exp t) ^ 2.
Proof.
  intros t.
  assert (Hpos : 0 < exp t) by apply exp_pos.
  apply Rdiv_lt_0_compat; [exact Hpos | nra].
Qed.

(* ===== (F) STRONG CONVEXITY MAKES THE CRITICAL POINT UNIQUE ===== *)

Lemma the_quadratic_floor_is_the_strong_convexity_constant :
  forall convex_part eps x : R,
  0 <= convex_part -> eps / 2 * x ^ 2 <= convex_part + eps / 2 * x ^ 2.
Proof.
  intros. lra.
Qed.

Lemma strong_convexity_gives_a_unique_critical_point :
  forall (gradient : R -> R) (eps x y : R),
  0 < eps ->
  (gradient x - gradient y) * (x - y) >= eps * (x - y) ^ 2 ->
  gradient x = 0 -> gradient y = 0 -> x = y.
Proof.
  intros gradient eps x y Heps Hmono Hx Hy.
  rewrite Hx, Hy in Hmono.
  assert (Hsq : (x - y) * (x - y) = 0).
  { assert (0 <= (x - y) ^ 2) by apply pow2_ge_0. nra. }
  assert (x - y = 0) by (apply square_zero; exact Hsq).
  lra.
Qed.
