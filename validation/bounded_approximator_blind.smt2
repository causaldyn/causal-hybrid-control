; SMT: the two bounds behind bounded_approximator_is_blind -- plans/24 P4.1.
;
; proofs/mean_field_dwr.v (F) proves them in Rocq; this is the independent second route, the
; negation of each asserted over the reals and expected unsat from BOTH z3 and cvc5:
;
;   timeout 60 z3 -T:30 validation/bounded_approximator_blind.smt2
;   timeout 60 cvc5 --incremental --tlimit=30000 validation/bounded_approximator_blind.smt2
;
; (1) the error bound. With n = num*m0 and the exact slope fixed by n + den*s0 = 0, an
;     approximator with |s0hat| <= B has |s0hat - s0| >= |n|/eps - B wherever 0 < |den| <= eps.
;     Stated multiplied through by eps > 0, so the block stays polynomial.
; (2) the residual bound. With k = q*c, a reduced state and slope rate all within B keep the
;     residual sdot + a*s - k*m within (1 + |a| + |k|) B -- a bound with no eps in it, which is
;     the whole point: (1) grows without limit as eps -> 0 and (2) does not move.
;
; Expected output: unsat, unsat.

(set-logic QF_NRA)
(define-fun absr ((x Real)) Real (ite (>= x 0.0) x (- x)))

(push 1)
(declare-const n Real)
(declare-const den Real)
(declare-const s0 Real)
(declare-const sh Real)
(declare-const B Real)
(declare-const eps Real)
(assert (not (= den 0.0)))
(assert (> eps 0.0))
(assert (<= (absr den) eps))
(assert (= (+ n (* den s0)) 0.0))
(assert (<= (absr sh) B))
(assert (not (<= (- (absr n) (* B eps)) (* eps (absr (- sh s0))))))
(check-sat)
(pop 1)

(push 1)
(declare-const a Real)
(declare-const k Real)
(declare-const m Real)
(declare-const s Real)
(declare-const sd Real)
(declare-const B2 Real)
(assert (<= (absr m) B2))
(assert (<= (absr s) B2))
(assert (<= (absr sd) B2))
(assert (not (<= (absr (- (+ sd (* a s)) (* k m))) (* (+ 1.0 (absr a) (absr k)) B2))))
(check-sat)
(pop 1)
