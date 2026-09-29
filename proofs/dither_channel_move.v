(* Rocq 9.2: the algebraic core of chc.gate.channel_move and ChannelMove.price
   (the symbolic side is validation/dither_channel_move.mac).
   Compile: timeout 300 rocq compile dither_channel_move.v

   Honest scope. The laws here have finite support: the dither's, with mean 0 and second moment 1,
   and the rest of the residual's, independent of it. The Gaussian dither the library draws is not
   finitely supported, and its moments enter only through those two sums, so the statements carry
   over as they are; that step, and the forgetting weights' effective size, are Maxima's
   (STEPs 1-3), not this file's. The pricing is proved for one entry; two are Maxima's STEP 4.

   (A) E[(c + k xi) xi] = k when xi has mean 0 and second moment 1 and is independent of c: the
       product of the residual and the standardised dither reads the channel's move, whatever c is
   (B) for an estimate d + e of a move d, with E e = 0 and E e^2 = s, and a weight w:
       E[w (d + e)^2 - w s] / 2 = w d^2 / 2 (the price of keeping the plan, unbiased), and
       E[w e^2] / 2 = w s / 2 (the price of re-planning on the estimate)
   (C) re-planning pays exactly when w d^2 > w s: an oracle's rule, which knows d; read off the
       same log as the estimate, the choice selects on the estimate's error, and chc does not ship
       it (docs/adr/0019-re-reading-a-moved-channel.md) *)

From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import List.
Import ListNotations.
Open Scope R_scope.

(* A finite law: pairs of a probability and a value. *)
Fixpoint expect (l : list (R * R)) (f : R -> R) : R :=
  match l with
  | [] => 0
  | (p, x) :: t => p * f x + expect t f
  end.

Lemma expect_linear :
  forall (l : list (R * R)) (a b : R) (f g : R -> R),
  expect l (fun x => a * f x + b * g x) = a * expect l f + b * expect l g.
Proof. induction l as [| [p x] t IH]; intros a b f g; simpl; [ring | rewrite IH; ring]. Qed.

Lemma expect_ext :
  forall (l : list (R * R)) (f g : R -> R), (forall x, f x = g x) -> expect l f = expect l g.
Proof. induction l as [| [p x] t IH]; intros f g H; simpl; [ring | rewrite H, (IH f g H); ring]. Qed.

Lemma expect_const :
  forall (l : list (R * R)) (a : R), expect l (fun _ => a) = a * expect l (fun _ => 1).
Proof. induction l as [| [p x] t IH]; intros a; simpl; [ring | rewrite IH; ring]. Qed.

(* ---------------------------------------------------------------------------------------------- *)
(* (A) The product reads the move. c is drawn from lc and the dither, independently, from lx. *)

Theorem product_reads_the_move :
  forall (lc lx : list (R * R)) (k : R),
  expect lc (fun _ => 1) = 1 ->
  expect lx (fun x => x) = 0 -> expect lx (fun x => x * x) = 1 ->
  expect lc (fun c => expect lx (fun x => (c + k * x) * x)) = k.
Proof.
  intros lc lx k Hc Hmean Hsq.
  assert (Hinner : forall c, expect lx (fun x => (c + k * x) * x) = k).
  { intros c.
    rewrite (expect_ext lx (fun x => (c + k * x) * x) (fun x => c * x + k * (x * x)))
      by (intros; ring).
    rewrite (expect_linear lx c k (fun x => x) (fun x => x * x)), Hmean, Hsq. ring. }
  rewrite (expect_ext lc _ (fun _ : R => k) Hinner).
  rewrite expect_const, Hc. ring.
Qed.

(* ---------------------------------------------------------------------------------------------- *)
(* (B) The prices. The estimate's error e is drawn from le. *)

Theorem keep_price_is_unbiased :
  forall (le : list (R * R)) (d s w : R),
  expect le (fun _ => 1) = 1 -> expect le (fun e => e) = 0 -> expect le (fun e => e * e) = s ->
  expect le (fun e => (w * (d + e) ^ 2 - w * s) / 2) = w * d ^ 2 / 2.
Proof.
  intros le d s w Hone Hmean Hsq.
  rewrite (expect_ext le (fun e => (w * (d + e) ^ 2 - w * s) / 2)
             (fun e => (w * d ^ 2 / 2 - w * s / 2) * 1 + (w * d) * e + (w / 2) * (e * e)))
    by (intros; field).
  assert (Hsplit : forall a b c : R,
    expect le (fun e => a * 1 + b * e + c * (e * e))
    = a * expect le (fun _ => 1) + b * expect le (fun e => e) + c * expect le (fun e => e * e)).
  { intros a b c.
    rewrite (expect_ext le (fun e => a * 1 + b * e + c * (e * e))
               (fun e => 1 * (a * 1 + b * e) + c * (e * e))) by (intros; ring).
    rewrite (expect_linear le 1 c (fun e => a * 1 + b * e) (fun e => e * e)).
    rewrite (expect_linear le a b (fun _ => 1) (fun e => e)). ring. }
  rewrite Hsplit, Hone, Hmean, Hsq. field.
Qed.

Theorem replan_price :
  forall (le : list (R * R)) (s w : R),
  expect le (fun e => e * e) = s -> expect le (fun e => w * (e * e) / 2) = w * s / 2.
Proof.
  intros le s w Hsq.
  rewrite (expect_ext le (fun e => w * (e * e) / 2) (fun e => (w / 2) * (e * e) + 0 * (e * e)))
    by (intros; field).
  rewrite (expect_linear le (w / 2) 0 (fun e => e * e) (fun e => e * e)), Hsq. field.
Qed.

(* ---------------------------------------------------------------------------------------------- *)
(* (C) The rule. *)

Theorem replanning_pays_iff :
  forall d s w : R, w * s / 2 < w * d ^ 2 / 2 <-> w * s < w * d ^ 2.
Proof. intros d s w. split; intro H; lra. Qed.
