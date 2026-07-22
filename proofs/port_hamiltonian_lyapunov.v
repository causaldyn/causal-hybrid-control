(* Rocq (CLOSED-LOOP STABILITY of the port-Hamiltonian residual under damping injection): the first
   machine-checked CLOSED-LOOP result for chc.residual.PortHamiltonianResidual -- the shipped
   port_hamiltonian_certificate only checks the AUTONOMOUS (u=0) passivity Hdot<=0. Here the control
   u = -kappa * g^T gradH (damping injection) makes V = H a strict Lyapunov function of the residual
   field xdot = (J - R) gradH + g u: the skew term cancels EXACTLY and the control ADDS dissipation.

     - skew_energy_neutral    : gradH^T J gradH = 0 (2x2 antisymmetry, lossless interconnection)
     - damping_decrease       : Vdot = -(r+k*b^2)*p^2*x^2 <= -r*p^2*x^2  (scalar; control adds -k*b^2*..)
     - damping_strict         : off the origin the decrease is strictly negative (asymptotic, not marginal)
     - euler_energy_decrease  : the explicit-Euler step decreases H under the CFL condition dt*c < 2

   Derived in validation/port_hamiltonian_lyapunov.mac (all identities residual 0).
   HONEST SCOPE: (i) this is the RESIDUAL field -- for the FULL hybrid f_known + r the whole-system
   claim needs f_known MATCHING (IDA-PBC PDE) or ENERGY-ORTHOGONALITY gradH^T f_known <= 0, else the
   cross term can dominate. (ii) R>=0 gives marginal (Lyapunov) stability; asymptotic needs R>0 (here
   r>0) or damping + zero-state detectability. (iii) H must be positive-definite / radially unbounded
   -- an unconstrained MLP energy is not, so convergence is to argmin H, not necessarily 0. The Rocq
   scope is the scalar / 2x2-quadratic-H algebra; general-n eigenvalue bounds enter as hypotheses. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* 2x2 skew interconnection is energy-neutral: gradH^T J gradH = 0 exactly (J = [[0,w],[-w,0]]). *)
Lemma skew_energy_neutral : forall w v1 v2 : R,
  v1 * (w * v2) + v2 * (- w * v1) = 0.
Proof. intros; ring. Qed.

(* Closed-loop Lyapunov decrease under damping injection u = -k*b*gradH (scalar port-H, H=p*x^2/2):
   Vdot = -(r + k*b^2)*p^2*x^2 <= -r*p^2*x^2 -- the control adds dissipation k*b^2*p^2*x^2 >= 0. *)
Lemma damping_decrease : forall x p r k b : R,
  0 < p -> 0 <= r -> 0 <= k ->
  - (r + k * (b * b)) * (p * p) * (x * x) <= - (r * (p * p) * (x * x)).
Proof.
  intros x p r k b Hp Hr Hk.
  pose proof (Rle_0_sqr b) as Hb. pose proof (Rle_0_sqr p) as Hpp. pose proof (Rle_0_sqr x) as Hxx.
  unfold Rsqr in *.
  assert (0 <= k * (b * b) * (p * p) * (x * x)) by
    (apply Rmult_le_pos;
      [apply Rmult_le_pos; [apply Rmult_le_pos; [exact Hk | exact Hb] | exact Hpp] | exact Hxx]).
  nra.
Qed.

(* Off the origin (x<>0 => x*x>0) with r>0 the decrease is STRICT: asymptotic, not merely marginal. *)
Lemma damping_strict : forall x p r k b : R,
  0 < p -> 0 < r -> 0 <= k -> 0 < x * x ->
  - (r + k * (b * b)) * (p * p) * (x * x) < 0.
Proof.
  intros x p r k b Hp Hr Hk Hx.
  pose proof (Rle_0_sqr b). unfold Rsqr in *.
  assert (Hpp : 0 < p * p) by (apply Rmult_lt_0_compat; assumption).
  assert (Hrk : 0 < r + k * (b * b)) by nra.
  assert (0 < (r + k * (b * b)) * (p * p) * (x * x))
    by (apply Rmult_lt_0_compat; [apply Rmult_lt_0_compat |]; assumption).
  nra.
Qed.

(* Explicit-Euler discrete Lyapunov decrease under the CFL step s = dt*c < 2, c = (r+k*b^2)*p:
   H(x+) - H(x) = (p*x^2/2) * s * (s - 2) <= 0. *)
Lemma euler_energy_decrease : forall x p s : R,
  0 < p -> 0 < s -> s < 2 ->
  (p * (x * x) / 2) * s * (s - 2) <= 0.
Proof.
  intros x p s Hp Hs Hs2.
  pose proof (Rle_0_sqr x) as Hxx. unfold Rsqr in *.
  assert (0 <= p * (x * x)) by (apply Rmult_le_pos; [lra | exact Hxx]).
  assert (s * (s - 2) <= 0) by nra.
  nra.
Qed.
