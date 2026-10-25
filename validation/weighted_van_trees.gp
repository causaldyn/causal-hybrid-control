\\ PARI/GP: the weighted van Trees floors on the one-step LQ regret, at 38 digits.
\\
\\ The regret of an action T at channel b is W(b) (T - psi(b))^2, with W = b^2 + rr and
\\ psi(b) = u*(b) = -xt b/(b^2 + rr) (validation/action_van_trees.mac STEP 1). The prior is
\\ b ~ N(mu, tau^2) and the data carry Fisher information In about b. For every estimator T and
\\ every weight C(b) with C lambda -> 0 at both ends (Gill and Levit, Bernoulli 1995;
\\ validation/weighted_van_trees.mac STEP 1),
\\
\\     E[W (T - psi)^2] >= (E[C psi'])^2 / ( In E[C^2/W] + E[(C' - C (b - mu)/tau^2)^2/W] ).
\\
\\ C = 1 with W >= rr gives rr (E psi')^2/(In + 1/tau^2), the least-curvature floor; C = W psi' is
\\ the closed form (validation/weighted_van_trees.mac STEP 3); the best C over
\\ span{W psi', He_0(z), ..., He_8(z)}, z = (b - mu)/tau, is a' B^-1 a (STEP 4). The Bayes regret is
\\ exact: the posterior is N(m, v), v = 1/(In + 1/tau^2), the Bayes action -xt m/(m^2 + v + rr), and
\\ m ~ N(mu, tau^2 - v), so the regret is xt^2 (E[b^2/(b^2 + rr)] - E[m^2/(m^2 + v + rr)]).
\\
\\ Each mean is a sum of intnum over pieces one standard deviation wide, out to 16 of them: intnum
\\ over one long interval mis-integrates a narrow Gaussian.
\\
\\ Certificates: chc.research.regret.information_lower_bound_certificate (weighted_bayes_floor,
\\ best_weighted_bayes_floor) and chc.research.regret.multivariate_van_trees_certificate
\\ (weighted_bayes_floor). tests/test_formal_regret_b.py and tests/test_formal_regret_c.py read the
\\ numbers below.
\\
\\ Run: timeout 300 gp -q -f validation/weighted_van_trees.gp </dev/null

default(realprecision, 38);

Eg(f, mu, s2) = {
  my(s = sqrt(s2));
  sum(k = -16, 15, intnum(b = mu + k*s, mu + (k + 1)*s, f(b)*exp(-(b - mu)^2/(2*s2))/sqrt(2*Pi*s2)));
}

\\ He_0 .. He_K, probabilists' Hermite polynomials in 'z, orthonormal under N(0, 1) once scaled
he(K) = {
  my(v = vector(K + 1));
  v[1] = 1; if(K >= 1, v[2] = 'z);
  for(n = 2, K, v[n + 1] = 'z*v[n] - (n - 1)*v[n - 1]);
  vector(K + 1, k, v[k]/sqrt((k - 1)!));
}

floors(rr, xt, mu, tau, In, K) = {
  my(t2 = tau^2, Wf, dps, Cw, dCw, A, Pw, w1, best, H, dH, nb, phi, dphi, a, B, v, bayes, chc, one);
  Wf = (b -> b^2 + rr);
  dps = (b -> -xt*(rr - b^2)/(rr + b^2)^2);
  Cw = (b -> -xt*(rr - b^2)/(rr + b^2));
  dCw = (b -> 4*xt*b*rr/(rr + b^2)^2);
  A = Eg(b -> (Wf(b)*dps(b)^2), mu, t2);
  Pw = Eg(b -> ((dCw(b) - Cw(b)*(b - mu)/t2)^2/Wf(b)), mu, t2);
  w1 = A^2/(In*A + Pw);
  one = Eg(dps, mu, t2)^2/Eg(b -> ((In + (b - mu)^2/t2^2)/Wf(b)), mu, t2);
  chc = rr*Eg(dps, mu, t2)^2/(In + 1/t2);
  H = he(K); dH = vector(K + 1, k, deriv(H[k], 'z)/tau);
  nb = K + 2;
  phi = ((i, b) -> if(i == 1, Cw(b), subst(H[i - 1], 'z, (b - mu)/tau)));
  dphi = ((i, b) -> if(i == 1, dCw(b), subst(dH[i - 1], 'z, (b - mu)/tau)));
  a = vector(nb, i, Eg(b -> (phi(i, b)*dps(b)), mu, t2))~;
  B = matrix(nb, nb);
  for(i = 1, nb, for(j = i, nb,
    B[i, j] = Eg(b -> (In*phi(i, b)*phi(j, b)
                       + (dphi(i, b) - phi(i, b)*(b - mu)/t2)*(dphi(j, b) - phi(j, b)*(b - mu)/t2))/Wf(b),
                 mu, t2);
    B[j, i] = B[i, j]));
  best = (a~ * matsolve(B, a));
  v = 1/(In + 1/t2);
  bayes = xt^2*(Eg(b -> (b^2/(b^2 + rr)), mu, t2) - Eg(m -> (m^2/(m^2 + v + rr)), mu, t2 - v));
  [bayes, chc, one, w1, best, A/In];
}

show(label, f) = {
  print(label);
  print("    Bayes regret                          ", f[1]);
  print("    rr (E psi')^2/(In + 1/tau^2)          ", f[2], "   ratio ", f[2]/f[1]);
  print("    weighted, C = 1                       ", f[3], "   ratio ", f[3]/f[1]);
  print("    weighted, C = W psi'                  ", f[4], "   ratio ", f[4]/f[1]);
  print("    weighted, best C (W psi', He_0..He_8) ", f[5], "   ratio ", f[5]/f[1]);
  print("    E[W psi'^2]/In                        ", f[6], "   ratio ", f[6]/f[1]);
  print("    order: rr floor <= C = 1 <= best, W psi' <= best <= Bayes: ",
        if(f[2] <= f[3] && f[3] <= f[5] && f[4] <= f[5] && f[5] <= f[1], "(PASS)", "(FAIL)"));
}

\\ (1) ADR 0130's example: rr = 1/2, xt = 1, b ~ N(0, 1), In = 5
f = floors(1/2, 1, 0, 1, 5, 8);
show("(1) rr = 1/2, xt = 1, b ~ N(0, 1), In = 5", f);
{print("    the least-curvature floor's ratio as In -> oo, rr (E psi')^2/E[W psi'^2] = ",
      (1/2)*Eg(b -> (-(1/2 - b^2)/(1/2 + b^2)^2), 0, 1)^2
      /Eg(b -> ((b^2 + 1/2)*((1/2 - b^2)/(1/2 + b^2)^2)^2), 0, 1));}
print("");

\\ (2) the same prior over In: the weighted floors close on the Bayes regret, the least-curvature
\\ floor does not
{for(j = 1, 6, my(In = [1, 5, 20, 100, 1000, 10000][j], g = floors(1/2, 1, 0, 1, In, 8));
   print("(2) In = ", In, ":  Bayes ", g[1], "   C = W psi' ", g[4], " (", g[4]/g[1], ")   best ",
         g[5], " (", g[5]/g[1], ")   rr floor ratio ", g[2]/g[1]));}
print("");

\\ (3) the certificate's own prior, N(b, 0.25^2) at b = 1 and In = 4000 * 2/0.5^2, a wide prior
\\ N(3, 1), and the knife edge b = sqrt(rr) at the centre
{show("(3) rr = 1/2, xt = 1, b ~ N(1, 1/16), In = 32000 (the certificate's defaults)",
     floors(1/2, 1, 1, 1/4, 32000, 8));}
show("(3') rr = 1/2, xt = 1, b ~ N(3, 1), In = 1", floors(1/2, 1, 3, 1, 1, 8));
{show("(3'') the knife edge: rr = 1/2, xt = 1, b ~ N(sqrt(1/2), 1/16), In = 20",
     floors(1/2, 1, sqrt(1/2), 1/4, 20, 8));}
print("");

\\ (4) the multivariate certificate: two decoupled channels, gains (1, 0.4), targets (1, 0.8),
\\ rr = 1/2, the prior N(gain, 0.25^2) on each and information n/sigma^2 in each. With
\\ C = M Psi' I^-1 the matrix floor is (F1 + F2)^2/(F1 + F2 + P1 + P2), F_i = E[W psi_i'^2]/I and
\\ P_i = E[(C_i' - C_i (b - g_i)/tau^2)^2/W]/I^2, C_i = W psi_i' (validation/weighted_van_trees.mac
\\ STEP 2c); the Bayes regret is the sum of the two channels' own.
mv(g, x, rr, tau, I) = {
  my(t2 = tau^2, F = 0, P = 0, bayes = 0, v = 1/(I + 1/t2));
  for(i = 1, #g,
    my(c = g[i], xt = x[i]);
    F += Eg(b -> ((b^2 + rr)*(xt*(rr - b^2)/(rr + b^2)^2)^2), c, t2)/I;
    P += Eg(b -> ((4*xt*b*rr/(rr + b^2)^2 + xt*(rr - b^2)/(rr + b^2)*(b - c)/t2)^2/(b^2 + rr)), c, t2)/I^2;
    bayes += xt^2*(Eg(b -> (b^2/(b^2 + rr)), c, t2) - Eg(m -> (m^2/(m^2 + v + rr)), c, t2 - v)));
  [F^2/(F + P), bayes, F];
}
{mvshow(label, m) = print(label, ": weighted floor ", m[1], "   Bayes regret ", m[2], "   ratio ",
                          m[1]/m[2], "   ", if(m[1] <= m[2], "(PASS)", "(FAIL)"));}
mvshow("(4) n = 4000, the defaults", mv([1, 2/5], [1, 4/5], 1/2, 1/4, 4000/(1/4)));
mvshow("(4') n = 20", mv([1, 2/5], [1, 4/5], 1/2, 1/4, 20/(1/4)));
mvshow("(4'') n = 4000, rr = 1/10", mv([1, 2/5], [1, 4/5], 1/10, 1/4, 4000/(1/4)));
mvshow("(4''') n = 4000, prior width 0.6", mv([1, 2/5], [1, 4/5], 1/2, 3/5, 4000/(1/4)));
quit;
