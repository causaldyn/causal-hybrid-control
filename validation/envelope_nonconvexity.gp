\\ PARI/GP: the envelope's nonconvexity at 80 digits, against envelope_nonconvexity.mac.
\\
\\ rho = max over 0 < z < t of m z - g(z), m = g(t)/t, t the tangency g(t) = t g'(t). The
\\ derivation's polynomials are its claims: rho is a root of 4r^3 + 12r^2 + 14r - 1 at Hill n = 2, of
\\ 3r^2 + 6r - 1 at n = 3 and of 125r^4 + 375r^3 + 375r^2 - 35r - 32 at n = 5; Kumaraswamy at b = 2
\\ has Hill's rho at n = 2a - 1. Here rho is found the plain way, by the root of g'(z) = m below the
\\ inflection, with no use of the reduction, and each claim is checked to 70 digits.

\p 80
hill(z, n) = z^n / (1 + z^n);
hill1(z, n) = n * z^(n - 1) / (1 + z^n)^2;
hillrho(n) = {
  my(t = (n - 1)^(1/n), m = hill(t, n) / t, bend = ((n - 1)/(n + 1))^(1/n), zs);
  zs = solve(z = 10^-30, bend, hill1(z, n) - m);
  m * zs - hill(zs, n);
}
kuma(z, a) = 1 - (1 - z^a)^2;
kuma1(z, a) = 2 * a * z^(a - 1) * (1 - z^a);
kumarho(a) = {
  my(t = (2*(a - 1)/(2*a - 1))^(1/a), m = kuma(t, a) / t, bend, zs);
  \\ g'' = 0 where (a - 1)(1 - w) = a w, w = z^a
  bend = ((a - 1)/(2*a - 1))^(1/a);
  zs = solve(z = 10^-30, bend, kuma1(z, a) - m);
  m * zs - kuma(zs, a);
}
claims = [[2, 4*r^3 + 12*r^2 + 14*r - 1], [3, 3*r^2 + 6*r - 1], [5, 125*r^4 + 375*r^3 + 375*r^2 - 35*r - 32]];
{
  for (i = 1, #claims,
    my(n = claims[i][1], p = claims[i][2], rho = hillrho(n), miss = abs(subst(p, r, rho)));
    print("hill n = ", n, ": rho = ", rho * 1.0, "  |p(rho)| = ", miss * 1.0);
    if (miss > 10^-70, error("hill n = ", n, ": rho is not a root of its polynomial")));
  my(miss3 = abs(hillrho(3) - (2/sqrt(3) - 1)));
  print("hill n = 3 less 2/sqrt(3) - 1: ", miss3 * 1.0);
  if (miss3 > 10^-70, error("hill n = 3 is not 2/sqrt(3) - 1"));
  foreach ([3/2, 2, 5/2, 3, 4, 7], a,
    my(gap = abs(kumarho(a) - hillrho(2*a - 1)));
    print("kumaraswamy a = ", a, " less hill n = ", 2*a - 1, ": ", gap * 1.0);
    if (gap > 10^-70, error("kumaraswamy a = ", a, " does not match hill n = ", 2*a - 1)));
}
quit
