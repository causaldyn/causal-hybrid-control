; SMT: the T = 3 capped-exploration instance, checked over the WHOLE feasible box -- plans/25 P3.2.
;
; proofs/capped_exploration_schedule.v proves the exchange argument and the stopping condition
; POINTWISE: given two rounds it shows the earlier one is cheaper, and given the root it shows the
; bracket crosses zero once.  Neither says that the greedy prefix fill is optimal over the whole
; continuum of feasible schedules -- that is a quantified statement over a box, which is what an
; SMT solver decides and a proof assistant would need a real optimisation argument for.
;
; The objective is chc.regret's, at T = 3 rounds with a per-round cap:
;
;   J(x) = A (x0 + x1 + x2) + K [ 1/I0 + 1/(I0 + c x0) + 1/(I0 + c (x0 + x1)) ],  0 <= xt <= kap
;
; Instance: A = 1, K = 4, I0 = 1, c = 1, kap = 4/5.  Stationarity gives I0 + c(x0+x1) = sqrt(K/A) = 2
; and dJ/dx0 = -K/(I0+c x0)^2 < 0 after that, so the optimum is x* = (4/5, 1/5, 0) with J* = 83/9 --
; the first round SATURATED, the second interior, the third empty.  Each block below asserts the
; NEGATION of one claim; `unsat` is the proof, `sat` would print a counterexample.
;
; Division is encoded by defining reciprocals with a product equation, so every assertion stays
; polynomial and inside QF_NRA.
;
; Run BOTH -- they use different nonlinear procedures and fail on different instances:
;   timeout 300 z3 -T:120 validation/capped_exploration_t3.smt2
;   timeout 300 cvc5 --tlimit=120000 --incremental validation/capped_exploration_t3.smt2
;
; cvc5 without --incremental aborts on the first `push` with "cannot push when not solving
; incrementally", and it reports block (E)'s model as algebraic numbers rather than rationals:
; 115x^2 - 22x - 56 = (5x - 4)(23x + 14) and 115x^2 - 208x + 37 = (5x - 1)(23x - 37), i.e. the
; same x* = (4/5, 1/5, 0) z3 prints directly, through a non-minimal defining polynomial.

(set-logic QF_NRA)
(set-option :produce-models true)

(declare-fun x0 () Real) (declare-fun x1 () Real) (declare-fun x2 () Real)
(declare-fun y1 () Real) (declare-fun y2 () Real)

; the box
(assert (and (<= 0 x0) (<= x0 (/ 4 5))))
(assert (and (<= 0 x1) (<= x1 (/ 4 5))))
(assert (and (<= 0 x2) (<= x2 (/ 4 5))))
; the two reciprocals the objective needs, as defining products
(assert (= (* y1 (+ 1 x0)) 1))
(assert (= (* y2 (+ 1 x0 x1)) 1))
(assert (> y1 0))
(assert (> y2 0))
; J(x) = (x0+x1+x2) + 4*(1 + y1 + y2)
(define-fun J () Real (+ x0 x1 x2 (* 4 (+ 1 y1 y2))))

; ---- (A) no feasible schedule beats the greedy prefix fill.  J* = 83/9.
(push 1)
(assert (< (* 9 J) 83))
(check-sat)
(pop 1)

; ---- (B) the first round is SATURATED at every optimum: nothing with x0 < kap reaches J*.
(push 1)
(assert (< x0 (/ 4 5)))
(assert (<= (* 9 J) 83))
(check-sat)
(pop 1)

; ---- (C) the last round's mass never pays: it adds A*x2 and buys no information inside T.
(push 1)
(assert (> x2 0))
(assert (<= (+ x0 x1 x2 (* 4 (+ 1 y1 y2))) (+ x0 x1 (* 4 (+ 1 y1 y2)))))
(check-sat)
(pop 1)

; ---- (D) the exchange argument over the whole box, not two rounds: at EQUAL total mass, a
;          schedule with larger prefix sums is never worse.  This is what makes greedy optimal and
;          is the statement proofs/capped_exploration_schedule.v only gives pointwise.
(push 1)
(declare-fun e0 () Real) (declare-fun e1 () Real) (declare-fun e2 () Real)
(declare-fun z1 () Real) (declare-fun z2 () Real)
(assert (and (<= 0 e0) (<= e0 (/ 4 5))))
(assert (and (<= 0 e1) (<= e1 (/ 4 5))))
(assert (and (<= 0 e2) (<= e2 (/ 4 5))))
(assert (= (* z1 (+ 1 e0)) 1))
(assert (= (* z2 (+ 1 e0 e1)) 1))
(assert (> z1 0))
(assert (> z2 0))
(assert (= (+ e0 e1 e2) (+ x0 x1 x2)))          ; same total mass
(assert (>= e0 x0))                             ; ... and pointwise larger prefix sums
(assert (>= (+ e0 e1) (+ x0 x1)))
(assert (> (+ e0 e1 e2 (* 4 (+ 1 z1 z2))) (+ x0 x1 x2 (* 4 (+ 1 y1 y2)))))
(check-sat)
(pop 1)

; ---- (E) THE CONTROL.  Four `unsat`s prove nothing if the base is inconsistent, and they would
;          also be vacuous if 83/9 were merely a valid bound rather than the attained minimum.  So:
;          the same box with `9 J <= 83` must be SAT, and its model must be x* = (4/5, 1/5, 0).
;          `unsat` at `< 83` and `sat` at `<= 83` together say the bound is exactly attained.
(push 1)
(assert (<= (* 9 J) 83))
(check-sat)
(get-value (x0 x1 x2))
(pop 1)

(exit)
