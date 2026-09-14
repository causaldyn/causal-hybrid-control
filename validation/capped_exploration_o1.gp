\\ PARI/GP: the O(1) term of the capped-exploration stopping mass, at 60 digits -- plans/25 P3.2.
\\
\\ Result 56 gave the stopping mass as S* = sqrt(K T/(A c)) - I0/c.  Result 66 replaced it with the
\\ fixed point of A(I0 + cS)^2 = K c (T - n), n = S/kap, because the leading form runs 25% high once
\\ the caps open late.  validation/capped_exploration_schedule.mac STEP 7 derives WHY: the fixed
\\ point is the positive root of a QUADRATIC, and its expansion at large T is
\\
\\     S* = sqrt(K T/(A c)) - I0/c - K/(2 A c kap) + (K + 4 A I0 kap) sqrt(K/(A c))/(8 A c kap^2 sqrt(T))
\\          + O(T^-3/2)
\\
\\ so Result 56 drops a CONSTANT, not a vanishing remainder, and that constant is the only place the
\\ cap enters the mass below O(1/sqrt(T)).  Float64 cannot separate a 1/sqrt(T) tail from a constant
\\ at T = 10^12 -- the difference is ~1e-6 of a quantity of size ~1e5, and the cancellation in
\\ S56 - Sexact eats most of the digits -- so this is checked in exact rationals at 60 digits.
\\
\\ Parameters are chc.regret.capped_exploration_policy's defaults, exactly:
\\   b = 1, rr = 1/2, xt = 1, sigma = 7/10, eta = 3/5, i0 = 1
\\   A = b^2 + rr = 3/2,  psi' = (b^2 - rr)/(b^2 + rr)^2 = 2/9,  K = A psi'^2 = 2/27,
\\   c = eta/sigma^2 = 60/49
\\
\\ Run: timeout 120 gp -q -f validation/capped_exploration_o1.gp

\p 60

A = 3/2; K = 2/27; c = 60/49; I0 = 1; kap = 1/100;

Sex(T) = ((-K/kap + sqrt(K^2/kap^2 + 4*A*(K*c*T + K*I0/kap)))/(2*A) - I0)/c;
S56(T) = sqrt(K*T/(A*c)) - I0/c;
o1     = K/(2*A*c*kap);
So1(T) = S56(T) - o1;
coef   = (K + 4*A*I0*kap)*sqrt(K/(A*c))/(8*A*c*kap^2);

print("A = ", A, "  K = ", K, "  c = ", c, "  I0 = ", I0, "  kap = ", kap);
print("predicted O(1) term  K/(2 A c kap) = ", o1);
print("predicted 1/sqrt(T) coefficient    = ", coef);
print("");

\\ (1) the root really is a root: residual of the quadratic it was derived from, at 60 digits.
print("(1) quadratic residual at the root  [A w^2 + (K/kap) w - K c T - K I0/kap]");
{for(j = 4, 12, T = 10^j; w = I0 + c*Sex(T);
     print("    T = 10^", j, "   residual = ", A*w^2 + (K/kap)*w - K*c*T - K*I0/kap));}
print("");

\\ (2) Result 56's form minus the exact root converges to the CONSTANT, not to zero.
\\     Five horizons, three decades apart, so the trend is a curve and not a two-point slope (R1).
print("(2) S56(T) - Sexact(T)   ->   K/(2 A c kap) = ", o1);
{for(j = 4, 12, T = 10^j; d = S56(T) - Sex(T);
     print("    T = 10^", j, "   gap = ", d, "   gap - o1 = ", d - o1));}
print("");

\\ (3) with the O(1) term restored the remainder is exactly the predicted 1/sqrt(T) tail.
print("(3) (So1(T) - Sexact(T)) * sqrt(T)   ->   -coef = ", -coef);
{for(j = 4, 12, T = 10^j;
     print("    T = 10^", j, "   scaled = ", (So1(T) - Sex(T))*sqrt(T)));}
print("");

\\ (4) and the term after that is bounded, which is what makes (3) an expansion and not a fit.
print("(4) ((So1(T) + coef/sqrt(T)) - Sexact(T)) * T^(3/2)   -> bounded");
{for(j = 4, 12, T = 10^j;
     print("    T = 10^", j, "   scaled = ", ((So1(T) + coef/sqrt(T)) - Sex(T))*T^(3/2)));}
print("");

\\ (5) what it costs in practice: the relative over-statement at a horizon the certificate uses.
print("(5) relative over-statement (S56 - Sexact)/Sexact at T = 4000, by cap level");
{for(i = 1, 6, kk = 10^(-i/2);
     o = K/(2*A*c*kk);
     s = ((-K/kk + sqrt(K^2/kk^2 + 4*A*(K*c*4000 + K*I0/kk)))/(2*A) - I0)/c;
     print("    kap = ", kk, "   Sexact = ", s, "   over = ", (S56(4000) - s)/s));}
quit
