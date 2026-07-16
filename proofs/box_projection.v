(* Rocq: formal proof of the box-projection invariants behind chc.control.project_box.
   The projection clips each control into [lo, hi]; these lemmas certify it is sound and
   idempotent (projecting an already-feasible point changes nothing). See plans/13 (Rocq role)
   and plans/02 (the pessimistic controller must actually stay inside the feasible box). *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

Definition clip (lo hi x : R) : R := Rmax lo (Rmin hi x).

(* The projection never falls below the lower bound. *)
Lemma clip_lower : forall lo hi x, lo <= clip lo hi x.
Proof.
  intros lo hi x. unfold clip, Rmax.
  destruct (Rle_dec lo (Rmin hi x)); lra.
Qed.

(* Given a well-formed box (lo <= hi), the projection never exceeds the upper bound. *)
Lemma clip_upper : forall lo hi x, lo <= hi -> clip lo hi x <= hi.
Proof.
  intros lo hi x H. unfold clip, Rmax.
  destruct (Rle_dec lo (Rmin hi x)).
  - unfold Rmin in *. destruct (Rle_dec hi x); lra.
  - lra.
Qed.

(* Idempotence: re-projecting a feasible point is the identity. *)
Lemma clip_idem : forall lo hi x, lo <= hi -> clip lo hi (clip lo hi x) = clip lo hi x.
Proof.
  intros lo hi x H.
  assert (Hl := clip_lower lo hi x).
  assert (Hu := clip_upper lo hi x H).
  set (y := clip lo hi x) in *.
  unfold clip.
  replace (Rmin hi y) with y by (unfold Rmin; destruct (Rle_dec hi y); lra).
  unfold Rmax; destruct (Rle_dec lo y); lra.
Qed.
