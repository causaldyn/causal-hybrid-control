(* Rocq (CONTRIBUTION 2): what row-level cross-fitting COSTS on a clustered design.

   The C2 write-up assumes A8 -- "K >= 2 folds of whole clusters" -- while the certificates split
   folds by ROW PARITY, so every cluster sits in both folds. The cited literature cannot price that:
   CCDDHNR assume i.i.d. rows (folds are then automatically valid), Hansen-Lee assume independent
   cluster scores as a PRIMITIVE, Chiang-Kato-Ma-Sasaki assume folds are already cluster-level. None
   of them prices a VIOLATED fold structure.

   The headline is that there is NO BIAS: (A) below is the algebraic shadow of the exact
   Frisch-Waugh-Lovell cancellation, which holds for ANY fold assignment. The damage is a change of
   ESTIMATOR, visible only in the second moment. (B)-(F) price it: the leakage ratio factors into a
   fold-geometry constant c(m,K) and an outcome-side factor, and the ICC enters ONLY the latter, so
   the NORMALISED law is exactly 1 - rho_ICC -- exact in m and K, not asymptotic. (G) records that
   the Hansen-Lee independence hypothesis fails at order 1/G, and (H) the composition A6-rate uses.

   Honest scope, as in van_trees.v and c2_end_to_end.v: Rocq proves the ALGEBRA over reals. There are
   no matrices, no independence and no conditional expectation in Stdlib; the trace identities
   tr(M'M) = m+2 and tr((M'M)^2) = m+14 at K=2 come from the projector algebra in
   validation/cluster_fold_leakage.mac and are taken as hypotheses here, exactly as the clustered CLT
   (Hansen-Lee 2019 Thm 2) and cross-fit negligibility (CCDDHNR 2018 Lemma 6.1) remain cited.
   Derived in validation/cluster_fold_leakage.mac. *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* (A) NO BIAS, for any fold assignment. Scalar shadow of E[Dtil*(Ytil - beta*Dtil)] = 0: once the
   Frisch-Waugh-Lovell cancellation has removed f(Z) and beta*D, what is left is a cross moment of
   the treatment noise against the outcome noise, which orthogonality kills whatever h (the leak
   coefficient) is -- i.e. whatever the fold assignment. This is why Result 43(a) is a no-bias
   result and not a bias-correction. *)
Theorem moment_unbiased_under_orthogonality : forall v va a aa e ea h,
  v * a = 0 -> v * aa = 0 -> v * e = 0 -> v * ea = 0 ->
  va * a = 0 -> va * aa = 0 -> va * e = 0 -> va * ea = 0 ->
  (v - h * va) * ((a - h * aa) + (e - h * ea)) = 0.
Proof.
  intros v va a aa e ea h Hva Hvaa Hve Hvea Hwa Hwaa Hwe Hwea.
  replace ((v - h * va) * ((a - h * aa) + (e - h * ea)))
    with ((v * a) + (v * e) - h * (v * aa) - h * (v * ea)
          - h * (va * a) - h * (va * e) + h * h * (va * aa) + h * h * (va * ea)) by ring.
  rewrite Hva, Hvaa, Hve, Hvea, Hwa, Hwaa, Hwe, Hwea. ring.
Qed.

(* The cancellation itself, in its scalar shadow: the cross-fold hat reproduces the design it is
   applied to, so the f(Z) part of the outcome is removed exactly -- for ANY fold assignment. *)
Theorem fwl_cancellation : forall zb za g,
  za <> 0 -> (zb / za) * (za * g) - zb * g = 0.
Proof. intros zb za g Hza. field. exact Hza. Qed.

(* (B) THE FOLD-GEOMETRY CONSTANT. c(m,K) := m * tr((M'M)^2) / tr(M'M)^2 is the leakage ratio at
   zero ICC: pure own-cluster-noise inflation, carrying no cluster effect. At K = 2 the projector
   algebra gives tr(M'M) = m + 2 and tr((M'M)^2) = m + 14. *)
Definition c_fold (m t2 t4 : R) : R := m * t4 / (t2 * t2).

Theorem c_fold_k2 : forall m,
  m + 2 <> 0 -> c_fold m (m + 2) (m + 14) = m * (m + 14) / ((m + 2) * (m + 2)).
Proof. intros m H. unfold c_fold. reflexivity. Qed.

(* Row folds always inflate: c(m,2) > 1 for every positive cluster size, because
   (m+2)^2 - m*(m+14) = 4 - 10*m is negative... i.e. m*(m+14) - (m+2)^2 = 10*m - 4 > 0. *)
Theorem c_fold_k2_gt_one : forall m, 2 <= m -> 1 < c_fold m (m + 2) (m + 14).
Proof.
  intros m Hm. unfold c_fold.
  apply (Rmult_lt_reg_r ((m + 2) * (m + 2))); [nra|].
  field_simplify; nra.
Qed.

(* ... and the inflation decays: m*(m+14) - (m+2)^2 = 10*m - 4, so c - 1 = (10*m-4)/(m+2)^2 -> 0.
   Stated as the explicit gap rather than as a limit, which Stdlib Reals cannot phrase cheaply. *)
Theorem c_fold_k2_gap : forall m,
  m + 2 <> 0 -> c_fold m (m + 2) (m + 14) - 1 = (10 * m - 4) / ((m + 2) * (m + 2)).
Proof. intros m H. unfold c_fold. field. exact H. Qed.

(* (C) THE LEAKAGE RATIO. Psi = Var_row / Var_cluster. The treatment residual is hit by the same
   operator as the outcome residual, and its trace enters SQUARED in the denominator -- pricing only
   the outcome side understates Psi, which is the error a first pass makes. *)
Definition psi (m t2 t4 tau2 sig2 : R) : R :=
  (sig2 * t4 / (t2 * t2)) / ((tau2 + sig2) / m).

Theorem psi_at_zero_icc : forall m t2 t4 sig2,
  sig2 <> 0 -> t2 <> 0 -> m <> 0 -> psi m t2 t4 0 sig2 = c_fold m t2 t4.
Proof. intros m t2 t4 sig2 Hs Ht Hm. unfold psi, c_fold. field. repeat split; assumption. Qed.

(* (D) THE LAW. rho_ICC enters ONLY the outcome-side factor; every fold-geometry factor lives in
   c(m,K) and cancels on normalising. So the normalised leakage law is EXACTLY 1 - rho_ICC, exact in
   m and K rather than asymptotic -- which is what makes it testable at any cluster size. *)
Definition icc (tau2 sig2 : R) : R := tau2 / (tau2 + sig2).

Theorem normalised_leak_eq_one_minus_icc : forall m t2 t4 tau2 sig2,
  0 < sig2 -> 0 <= tau2 -> t2 <> 0 -> t4 <> 0 -> 0 < m ->
  psi m t2 t4 tau2 sig2 / c_fold m t2 t4 = 1 - icc tau2 sig2.
Proof.
  intros m t2 t4 tau2 sig2 Hs Ht Ht2 Ht4 Hm.
  unfold psi, c_fold, icc. field. repeat split; try assumption; nra.
Qed.

(* (E) THE FACTORISATION the rest rests on: psi = c_fold * (1 - rho_ICC). Every fold-geometry
   factor sits in c_fold and every distributional factor in (1 - rho_ICC); they never mix, which is
   what makes (D)'s normalised law exact rather than asymptotic. *)
Theorem psi_factorises : forall m t2 t4 tau2 sig2,
  0 < sig2 -> 0 <= tau2 -> 0 < m -> 0 < t2 ->
  psi m t2 t4 tau2 sig2 = c_fold m t2 t4 * (1 - icc tau2 sig2).
Proof.
  intros m t2 t4 tau2 sig2 Hs Ht Hm Ht2.
  unfold psi, c_fold, icc. field. nra.
Qed.

Lemma c_fold_pos : forall m t2 t4, 0 < m -> 0 < t2 -> 0 < t4 -> 0 < c_fold m t2 t4.
Proof.
  intros m t2 t4 Hm Ht2 Ht4. unfold c_fold, Rdiv.
  apply Rmult_lt_0_compat.
  - apply Rmult_lt_0_compat; assumption.
  - apply Rinv_0_lt_compat. apply Rmult_lt_0_compat; assumption.
Qed.

Lemma one_minus_icc_eq : forall tau2 sig2,
  0 < sig2 -> 0 <= tau2 -> 1 - icc tau2 sig2 = sig2 / (tau2 + sig2).
Proof. intros tau2 sig2 Hs Ht. unfold icc. field. lra. Qed.

(* (F) MONOTONICITY: the more intra-cluster correlation, the more the row-fold estimator understates
   the cluster-robust one. This is the direction that matters for inference -- a cluster-robust
   sandwich computed after row folds is too SMALL, so a confidence interval built from it
   under-covers, which is the practical cost of the A8 violation. *)
Theorem leak_ratio_antitone_in_icc : forall m t2 t4 tau1 tau2 sig2,
  0 < sig2 -> 0 <= tau1 -> tau1 <= tau2 -> 0 < m -> 0 < t2 -> 0 < t4 ->
  psi m t2 t4 tau2 sig2 <= psi m t2 t4 tau1 sig2.
Proof.
  intros m t2 t4 tau1 tau2 sig2 Hs H1 H12 Hm Ht2 Ht4.
  rewrite (psi_factorises m t2 t4 tau2 sig2) by lra.
  rewrite (psi_factorises m t2 t4 tau1 sig2) by lra.
  apply Rmult_le_compat_l; [ apply Rlt_le, c_fold_pos; assumption |].
  rewrite !one_minus_icc_eq by lra.
  unfold Rdiv. apply Rmult_le_compat_l; [ lra |].
  apply Rinv_le_contravar; lra.
Qed.

(* Strictness: with any cluster effect at all, the leaked estimator's variance is strictly below the
   clean one's -- the understatement is real, not a measure-zero artefact. *)
Theorem leak_ratio_lt_c_fold : forall m t2 t4 tau2 sig2,
  0 < sig2 -> 0 < tau2 -> 0 < m -> 0 < t2 -> 0 < t4 ->
  psi m t2 t4 tau2 sig2 < c_fold m t2 t4.
Proof.
  intros m t2 t4 tau2 sig2 Hs Ht Hm Ht2 Ht4.
  rewrite psi_factorises by lra.
  assert (Hc : 0 < c_fold m t2 t4) by (apply c_fold_pos; assumption).
  rewrite one_minus_icc_eq by lra.
  assert (Hlt : sig2 / (tau2 + sig2) < 1).
  { apply (Rmult_lt_reg_r (tau2 + sig2)); [ lra |]. field_simplify; lra. }
  nra.
Qed.

(* (G) THE HANSEN-LEE HYPOTHESIS VIOLATION. Under cluster folds distinct cluster scores are exactly
   uncorrelated. Under row folds they are coupled through the shared global nuisance coefficients,
   estimated from all G clusters, so the coupling is O(1/G): the HYPOTHESIS of Hansen-Lee Thm 2 (and
   of CCDDHNR Lemma 6.1) fails, while for a fixed correctly-specified linear span the CONCLUSION
   survives. Both halves of that sentence matter, and neither is in the cited papers. *)
Theorem cross_cluster_cov_vanishes : forall cbar g1 g2,
  0 < g1 -> g1 <= g2 -> 0 <= cbar ->
  cbar * cbar / g2 <= cbar * cbar / g1.
Proof.
  intros cbar g1 g2 H1 H12 Hc. unfold Rdiv.
  apply Rmult_le_compat_l; [ nra |].
  apply Rinv_le_contravar; nra.
Qed.

(* (H) THE COMPOSITION A6-rate CONSUMES. The three-term decomposition: a clustered-CLT term at
   1/sqrt(G) (T1, cited), the Neyman-orthogonality remainder (T2, machine-checked in
   orthogonal_control.v and multichannel_control.v), and the cross-fit empirical process (T3,
   cited). This lemma is the arithmetic that turns the three bounds into the additive rate -- the
   only part of the chain that is Rocq's to prove. *)
Theorem three_term_negligible : forall t1 t2 t3 c1 r t g,
  0 < g -> Rabs t1 <= c1 / sqrt g -> Rabs t2 <= r -> Rabs t3 <= t ->
  Rabs (t1 + t2 + t3) <= c1 / sqrt g + r + t.
Proof.
  intros t1 t2 t3 c1 r t g Hg H1 H2 H3.
  eapply Rle_trans; [ apply Rabs_triang |].
  apply Rplus_le_compat; [| exact H3].
  eapply Rle_trans; [ apply Rabs_triang |].
  apply Rplus_le_compat; assumption.
Qed.
