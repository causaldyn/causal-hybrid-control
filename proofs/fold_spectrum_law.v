(* Rocq: the algebraic core of the FOLD-SPECTRUM LAW (Result 52).

   validation/fold_spectrum_law.mac derives, for the cross-fit sandwich of Result 51 on a
   delayed-network covariance: (i) the residualiser's square stays inside the idempotent algebra
   <I, E, F> -- squaring promotes r^2 to r^4 and nothing else -- so the fold partition enters the
   variance functional ONLY through the same-fold overlap term, with weight r^4 - 1 > 0: the
   variance-optimal fold design is a MINIMUM-weight balanced cut of the graph weighted by
   Q(x) = sum g_d g_e x^|d-e| S_d S_e; (ii) on a cycle the K = 2 objective diagonalises over
   Fourier modes with score lambda(c) = g0^2 + 4 g1^2 c^2 + 4 g0 g1 x c + shell-2 terms,
   c = cos(theta); (iii) the two pure spectral designs, parity (all mass at c = -1) and width-2
   stripes (all mass at c = 0), differ by the factored two-threshold law
       lambda_parity - lambda_stripes = 4 (g0 x - g1)(2 g2 x - g1).

   Honest scope, as in omega_jensen_gap.v: Stdlib has no matrices, so the <I, E, F> subalgebra is
   handled through its structure constants (E^2 = E, F^2 = F, EF = FE = E) and the spectral facts
   are scalar identities downstream of the Maxima matrix work. What is proved here is the algebra
   that the enumeration and the certificate then consume. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Psatz.
Open Scope R_scope.

(* (A) In coordinates on the basis (I, E, F), the product of a1 I + b1 E + c1 F and
   a2 I + b2 E + c2 F has E-coordinate a1 b2 + b1 a2 + b1 b2 + b1 c2 + c1 b2 and F-coordinate
   a1 c2 + c1 a2 + c1 c2. For A = I - r^2 E + (r^2 - 1) F the square lands on
   I - r^4 E + (r^4 - 1) F: the whole effect of squaring the residualiser is r^2 -> r^4. *)
Lemma residualiser_square_promotes_r :
  forall r : R,
  2 * (- r^2) + (- r^2) * (- r^2) + 2 * ((- r^2) * (r^2 - 1)) = - r^4 /\
  2 * (r^2 - 1) + (r^2 - 1) * (r^2 - 1) = r^4 - 1.
Proof. intro r; split; ring. Qed.

(* (B) The weight on the same-fold term is r^4 - 1 > 0 for every genuine fold count
   (r = K/(K-1) > 1): smaller same-fold weighted 2-walk count means smaller variance,
   so the design problem is a MINIMUM cut, never a maximum one. *)
Lemma partition_weight_positive : forall r : R, 1 < r -> 0 < r^4 - 1.
Proof.
  intros r H.
  replace (r^4) with ((r*r)*(r*r)) by ring.
  assert (1 < r*r) by nra.
  nra.
Qed.

(* (C) The two-threshold law, division-free: the parity-minus-stripes spectral difference
   factors through its roots x = g1/g0 and x = g1/(2 g2). *)
Lemma stripe_parity_difference_factors :
  forall g0 g1 g2 x : R,
  4*g1^2 - 4*g1*(g0 + 2*g2)*x + 8*g0*g2*x^2 = 4 * (g0*x - g1) * (2*g2*x - g1).
Proof. intros; ring. Qed.

(* (D) Sign dichotomy: parity is optimal exactly BETWEEN the thresholds (the two factors have
   opposite signs), stripes exactly OUTSIDE them (same signs) -- in whichever order the two
   thresholds fall. *)
Lemma parity_window_dichotomy :
  forall g0 g1 g2 x : R,
  ((g1 < g0*x /\ 2*g2*x < g1) \/ (g0*x < g1 /\ g1 < 2*g2*x) ->
     4 * (g0*x - g1) * (2*g2*x - g1) < 0) /\
  ((g1 < g0*x /\ g1 < 2*g2*x) \/ (g0*x < g1 /\ 2*g2*x < g1) ->
     0 < 4 * (g0*x - g1) * (2*g2*x - g1)).
Proof. intros; split; intros [[H1 H2] | [H1 H2]]; nra. Qed.

(* (E) D = 1 limit: with g2 = 0 the law keeps a single threshold at x = g1/g0. *)
Lemma d1_single_threshold :
  forall g0 g1 x : R, 0 < g1 ->
  (g0*x < g1 -> 0 < 4 * (g0*x - g1) * (2*0*x - g1)) /\
  (g1 < g0*x -> 4 * (g0*x - g1) * (2*0*x - g1) < 0).
Proof. intros g0 g1 x H; split; intro; nra. Qed.

(* (F) The D = 1 mode score completes the square: lambda(c) = g0^2 - (g0 x)^2 + (2 g1 c + g0 x)^2,
   so the variance-optimal fold frequency satisfies cos(theta_star) = - g0 x / (2 g1): as the delayed
   persistence x grows, the optimal stripes NARROW, from width 2 (c = 0) at x = 0 down to parity
   (c = -1) past x = g1/g0. *)
Lemma stripe_frequency_vertex :
  forall g0 g1 x c : R,
  g0^2 + 4*g1^2*c^2 + 4*g0*g1*x*c = g0^2 - (g0*x)^2 + (2*g1*c + g0*x)^2.
Proof. intros; ring. Qed.

Lemma vertex_is_minimum :
  forall g0 g1 x c : R, g0^2 - (g0*x)^2 <= g0^2 + 4*g1^2*c^2 + 4*g0*g1*x*c.
Proof.
  intros.
  assert (H := Rle_0_sqr (2*g1*c + g0*x)).
  unfold Rsqr in H.
  nra.
Qed.
