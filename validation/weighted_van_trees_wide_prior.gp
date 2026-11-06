\\ PARI/GP: the weighted van Trees floors under a prior wide against sqrt(rr), at 38 digits.
\\
\\ The floors and the Bayes regret of validation/weighted_van_trees.gp, whose formulas this file
\\ repeats, read where the prior's width is 10 to 3000 times sqrt(rr). The integrands have poles at
\\ b = +-i sqrt(rr), so near b = 0 they vary on the scale sqrt(rr), and under such a prior that
\\ feature is a spike of width sqrt(rr)/tau in standard units. Each mean here is a sum of intnum over
\\ pieces matched to BOTH scales: one prior standard deviation wide out to 16 of them, and cut
\\ again at b = +-k sqrt(rr) for k from 0 to 10^4, so no piece holds a feature narrower than itself.
\\
\\ (1) ADR 0130's example, read through these pieces: the digits of validation/weighted_van_trees.gp.
\\ (2) A prior N(5, 30^2) at rr = 1/1000, where the library's quadrature, split at b = 0 alone, read
\\     the prior term as ~0 and returned E[W psi'^2]/In, 1.02 to 203 times the Bayes regret.
\\ (3) A sweep: mu = tau/6, rr = 1/1000, tau/sqrt(rr) from 10 to 3000, In from 1/100 to 10^4, the
\\     floor of C = W psi' and the Bayes regret.
\\
\\ Certificate: chc.research.regret.information_lower_bound_certificate (weighted_bayes_floor,
\\ best_weighted_bayes_floor). tests/test_formal_regret_b.py reads the numbers below.
\\
\\ Run: timeout 300 gp -q -f validation/weighted_van_trees_wide_prior.gp </dev/null

default(realprecision, 38);

pieces(mu, s, pole) = {
  my(v = List(), ks = [0, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000]);
  for(k = -16, 16, listput(v, mu + k*s));
  for(i = 1, #ks, listput(v, ks[i]*pole); listput(v, -ks[i]*pole));
  v = vecsort(Vec(v), , 8);
  select(t -> (t >= mu - 16*s && t <= mu + 16*s), v);
}

\\ E f(b) over b ~ N(mu, s2), the integrand's poles at distance pole from the real line
Eg(f, mu, s2, pole) = {
  my(s = sqrt(s2), p = pieces(mu, s, pole));
  sum(i = 1, #p - 1, intnum(b = p[i], p[i + 1], f(b)*exp(-(b - mu)^2/(2*s2))/sqrt(2*Pi*s2)));
}

he(K) = {
  my(v = vector(K + 1));
  v[1] = 1; if(K >= 1, v[2] = 'z);
  for(n = 2, K, v[n + 1] = 'z*v[n] - (n - 1)*v[n - 1]);
  vector(K + 1, k, v[k]/sqrt((k - 1)!));
}

\\ the Bayes regret: posterior N(m, v), v = 1/(In + 1/tau^2), m ~ N(mu, tau^2 - v)
bayes(rr, xt, mu, tau, In) = {
  my(t2 = tau^2, v = 1/(In + 1/t2));
  xt^2*(Eg(b -> (b^2/(b^2 + rr)), mu, t2, sqrt(rr))
        - Eg(m -> (m^2/(m^2 + v + rr)), mu, t2 - v, sqrt(v + rr)));
}

\\ [A, P, the floor of C = W psi'], A = E[W psi'^2], P = E[(C' - C (b - mu)/tau^2)^2/W]
closed(rr, xt, mu, tau, In) = {
  my(t2 = tau^2, pole = sqrt(rr), A, Pw);
  A = Eg(b -> ((b^2 + rr)*(xt*(rr - b^2)/(rr + b^2)^2)^2), mu, t2, pole);
  Pw = Eg(b -> ((4*xt*b*rr/(rr + b^2)^2 + xt*(rr - b^2)/(rr + b^2)*(b - mu)/t2)^2/(b^2 + rr)),
          mu, t2, pole);
  [A, Pw, A^2/(In*A + Pw)];
}

\\ the best C over span{W psi', He_0(z), ..., He_K(z)}, z = (b - mu)/tau: a' B^-1 a
best(rr, xt, mu, tau, In, K) = {
  my(t2 = tau^2, pole = sqrt(rr), dps, Cw, dCw, H, dH, nb, phi, dphi, a, B);
  dps = (b -> -xt*(rr - b^2)/(rr + b^2)^2);
  Cw = (b -> -xt*(rr - b^2)/(rr + b^2));
  dCw = (b -> 4*xt*b*rr/(rr + b^2)^2);
  H = he(K); dH = vector(K + 1, k, deriv(H[k], 'z)/tau);
  nb = K + 2;
  phi = ((i, b) -> if(i == 1, Cw(b), subst(H[i - 1], 'z, (b - mu)/tau)));
  dphi = ((i, b) -> if(i == 1, dCw(b), subst(dH[i - 1], 'z, (b - mu)/tau)));
  a = vector(nb, i, Eg(b -> (phi(i, b)*dps(b)), mu, t2, pole))~;
  B = matrix(nb, nb);
  for(i = 1, nb, for(j = i, nb,
    B[i, j] = Eg(b -> ((In*phi(i, b)*phi(j, b)
                        + (dphi(i, b) - phi(i, b)*(b - mu)/t2)*(dphi(j, b) - phi(j, b)*(b - mu)/t2))
                       /(b^2 + rr)), mu, t2, pole);
    B[j, i] = B[i, j]));
  a~ * matsolve(B, a);
}

\\ the best floor's 55 means take most of the time, so it is read where a test reads it
show(label, rr, xt, mu, tau, In, withbest) = {
  my(c = closed(rr, xt, mu, tau, In), r = bayes(rr, xt, mu, tau, In), w);
  w = if(withbest, best(rr, xt, mu, tau, In, 8), c[3]);
  print(label);
  print("    Bayes regret                          ", r);
  print("    weighted, C = W psi'                  ", c[3], "   ratio ", c[3]/r);
  if(withbest, print("    weighted, best C (W psi', He_0..He_8) ", w, "   ratio ", w/r));
  print("    E[W psi'^2]/In, the prior term read 0 ", c[1]/In, "   ratio ", c[1]/In/r);
  print("    order: W psi' <= best <= Bayes: ", if(c[3] <= w && w <= r, "(PASS)", "(FAIL)"));
}

\\ (1) expect 0.074950277196394276769, 0.048886656770088468288, 0.059599430154587394402
show("(1) rr = 1/2, xt = 1, b ~ N(0, 1), In = 5 (ADR 0130's example)", 1/2, 1, 0, 1, 5, 1);
print("");

\\ (2) the corner; In = 100 is information_lower_bound_certificate(b=5, rr=1e-3, sigma=1,
\\ sample_sizes=(25, 50), prior_width=30), 50 * 2/1^2
{for(j = 1, 4, my(In = [1/100, 1, 100, 10000][j]);
   show(Str("(2) rr = 1/1000, xt = 1, b ~ N(5, 30^2), In = ", In), 1/1000, 1, 5, 30, In, In == 100));}
print("");

\\ (3) the sweep: tau = r sqrt(rr), mu = tau/6
{for(i = 1, 7, for(j = 1, 4,
   my(r = [10, 30, 100, 300, 1000, 2000, 3000][i], In = [1/100, 1, 100, 10000][j],
      rr = 1/1000, tau = r*sqrt(1/1000), c, g);
   c = closed(rr, 1, tau/6, tau, In);
   g = bayes(rr, 1, tau/6, tau, In);
   print("(3) tau/sqrt(rr) = ", r, ", In = ", In, ":  Bayes ", g, "   C = W psi' ", c[3],
         "   ratio ", c[3]/g, "   ", if(c[3] <= g, "(PASS)", "(FAIL)"))));}
quit;
