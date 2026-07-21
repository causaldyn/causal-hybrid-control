(* Rocq: a HIGH-PROBABILITY finite-sample regret bound -- the sub-Gaussian upgrade of the Cramer-Rao
   lower bound (Result 10, which is in EXPECTATION; derived in validation/highprob_regret.mac). The
   effect estimator concentrates: |b_hat - b| <= r(delta) with probability >= 1 - delta, with
   r(delta)^2 = 2*sigma^2*log(2/delta)/(n*V_id). Composing with certainty-equivalence quadraticity
   regret = C*(b_hat-b)^2 gives, on that event, a regret band = 2*log(2/delta) times the Cramer-Rao
   floor. The concentration itself (measure theory) is verified empirically in the certificate; here we
   prove the DETERMINISTIC core: the event implies the bound, the band is a log-multiple of the floor,
   and confounding (smaller V_id) widens it. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* On the concentration event |eps| <= r, the certainty-equivalence regret C*eps^2 is at most C*r^2. *)
Theorem regret_within_radius : forall cc eps r,
  0 <= cc -> Rabs eps <= r -> cc * eps ^ 2 <= cc * r ^ 2.
Proof.
  intros cc eps r Hc Hr. apply Rmult_le_compat_l; [exact Hc |].
  split_Rabs; nra.
Qed.

(* The high-probability band is exactly 2*log(2/delta) times the Cramer-Rao floor (floor = cc*sig2/nv,
   band = cc*2*sig2*L/nv): the log(1/delta) confidence price over the in-expectation bound. *)
Theorem band_is_log_multiple_of_floor : forall cc sig2 nv l,
  nv <> 0 -> cc * (2 * sig2 * l / nv) = 2 * l * (cc * sig2 / nv).
Proof.
  intros cc sig2 nv l Hnv. field. exact Hnv.
Qed.

(* Confounding shrinks the identifying variance V_id, and the band K/V_id is antitone in V_id, so the
   confounded band is WIDER -- the high-probability analogue of Result 10's higher floor. *)
Theorem confounding_widens_band : forall k v1 v2,
  0 <= k -> 0 < v1 -> v1 <= v2 -> k / v2 <= k / v1.
Proof.
  intros k v1 v2 Hk Hv1 Hle. unfold Rdiv.
  apply Rmult_le_compat_l; [exact Hk | apply Rinv_le_contravar; lra].
Qed.
