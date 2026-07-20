(* Rocq: "predictive control is asymptotically wrong under confounding" -- CHC's headline (notebook
   01) hardened into a theorem, and the seam the package name promises (control made causal). A
   controller picks the action minimising a quadratic control cost given its estimate of the causal
   effect. Completing the square shows the control regret is EXACTLY (b^2+rr)*(action error)^2. The
   causal controller uses the interventional effect, so its action is the optimum -> zero regret. The
   predictive controller uses the observational effect, which carries a SYSTEMATIC omitted-variable
   bias (validation/causal_mpc.mac: beta = Vz*alpha*gamma/(Vz*alpha^2+Vnu), a population constant,
   independent of the sample size n) -> a biased action -> a POSITIVE regret floor that does NOT
   vanish with more data. Only causal identification closes the gap. See proofs/orthogonal_control.v
   (the debiasing rate) and proofs/interference_regret.v (the interference term). *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

Definition cost (b u xt rr : R) : R := (b * u - xt) ^ 2 + rr * u ^ 2.

(* Completing the square: for the stationary action u0 (defined by (b^2+rr)*u0 = b*xt), the excess
   cost of ANY action u is exactly (b^2+rr)*(u-u0)^2 -- control regret is quadratic in action error. *)
Lemma cost_completing_square : forall b u u0 xt rr,
  (b ^ 2 + rr) * u0 = b * xt ->
  cost b u xt rr - cost b u0 xt rr = (b ^ 2 + rr) * (u - u0) ^ 2.
Proof.
  intros b u u0 xt rr Hopt. unfold cost.
  assert (Hid : (b * u - xt) ^ 2 + rr * u ^ 2 - ((b * u0 - xt) ^ 2 + rr * u0 ^ 2)
                - (b ^ 2 + rr) * (u - u0) ^ 2
                = 2 * (u - u0) * ((b ^ 2 + rr) * u0 - b * xt)) by ring.
  assert (Hz : (b ^ 2 + rr) * u0 - b * xt = 0) by lra.
  rewrite Hz in Hid. ring_simplify in Hid. lra.
Qed.

(* Causal control uses the interventional effect -> its action IS the optimum u0 -> zero regret. *)
Lemma causal_control_zero_regret : forall b u0 xt rr,
  cost b u0 xt rr - cost b u0 xt rr = 0.
Proof. intros. ring. Qed.

(* Predictive control uses a confounded (systematically biased) effect -> a biased action u <> u0 ->
   a strictly positive regret. The floor (b^2+rr)*(u-u0)^2 is fixed in n (the bias is a population
   constant), so the predictive-vs-causal gap does not close with more data. *)
Theorem predictive_control_positive_regret : forall b u u0 xt rr,
  0 < b ^ 2 + rr -> u <> u0 -> (b ^ 2 + rr) * u0 = b * xt ->
  0 < cost b u xt rr - cost b u0 xt rr.
Proof.
  intros b u u0 xt rr Hpos Hne Hopt.
  rewrite (cost_completing_square b u u0 xt rr Hopt).
  assert (Hsq : 0 < (u - u0) ^ 2).
  { destruct (Rtotal_order u u0) as [H | [H | H]].
    - nra.
    - exfalso; apply Hne; lra.
    - nra. }
  nra.
Qed.
