\\ PARI/GP: the envelope's nonconvexity at 80 digits, against envelope_nonconvexity.mac.
\\
\\ rho = max over 0 < z < t of m z - g(z), m = g(t)/t, t the tangency g(t) = t g'(t). The
\\ derivation's polynomials are its claims: rho is a root of 4r^3 + 12r^2 + 14r - 1 at Hill n = 2, of
\\ 3r^2 + 6r - 1 at n = 3 and of 125r^4 + 375r^3 + 375r^2 - 35r - 32 at n = 5; Kumaraswamy at b = 2
\\ has Hill's rho at n = 2a - 1. Here rho is found the plain way, by the root of g'(z) = m below the
\\ inflection, with no use of the reduction, and each claim is checked to 70 digits.
\\
\\ At the threshold, n = 1 + eps, the derivation's laws are a(1 - a) eps^2 for Hill, with a the
\\ root of a = e^(2a - 2) below 1, and eps^3/(12 (e - 1)) for Gompertz at b = 1 + eps: Hill's is read
\\ off its reduced root at eps = 1e-25, Gompertz's found the plain way at eps = 1e-20, each to
\\ 1e-18. The tangencies of Weibull and Chapman-Richards near shape 1 are the positive root of
\\ e^u - 1 = k u (chc.response._lambert_root): the tests' anchors are printed here, at the shapes'
\\ binary values.

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

{
  my(a = -lambertw(-2*exp(-2))/2, eps = 10^-25, x, law, miss);
  if (abs(a - exp(2*a - 2)) > 10^-75, error("a is not a root of a = e^(2a - 2)"));
  print("a = ", a * 1.0, "  a (1 - a) = ", a * (1 - a) * 1.0);
  \\ Hill's reduced root, A = 1 + x eps: x = ((1 + x eps)/(1 + eps))^(2 (1 + eps)/eps)
  x = solve(y = 1/10, 1/2, y - ((1 + y*eps)/(1 + eps))^(2*(1 + eps)/eps));
  law = x * (1 - x) / (1 + x*eps)^2;
  miss = abs(law - a * (1 - a));
  print("hill at n = 1 + 1e-25: rho/eps^2 less a (1 - a) = ", miss * 1.0);
  if (miss > 10^-18, error("hill's rho/eps^2 does not tend to a (1 - a)"));
}
\p 150
gomp(z, b) = (exp(-b*exp(-z)) - exp(-b)) / (1 - exp(-b));
gomp1(z, b) = b * exp(-z) * exp(-b*exp(-z)) / (1 - exp(-b));
{
  my(eps = 10^-20, b = 1 + eps, t, m, zs, rho, miss);
  t = solve(z = 6*eps/5, 2*eps, gomp(z, b) - z * gomp1(z, b));
  m = gomp(t, b) / t;
  zs = solve(z = eps/10^10, log(b), gomp1(z, b) - m);
  rho = m * zs - gomp(zs, b);
  miss = abs(rho / (eps^3 / (12 * (exp(1) - 1))) - 1);
  print("gompertz at b = 1 + 1e-20: rho / (eps^3/(12 (e - 1))) less 1 = ", miss * 1.0);
  if (miss > 10^-18, error("gompertz's rho/eps^3 does not tend to 1/(12 (e - 1))"));
}
\p 80
lambertroot(k) = my(e = k - 1); solve(u = 10^-200, min(2*e, 2*log(1 + k)), (exp(u) - 1 - u)/u - e);
{
  \\ 1 + 10^-j at its double: the shapes the tests build
  foreach ([[1001, 1000], [10001, 10000], [1000001, 1000000], [1000000001, 1000000000],
            [1000000000001, 1000000000000]], q,
    my(k = round(q[1]/q[2] * 2^52) / 2^52, u = lambertroot(k));
    if (abs(exp(u) - 1 - k*u) > 10^-75 * u, error("not a root at k = ", k));
    print("k = 1 + ", (k - 1) * 1.0, ": chapman-richards ", u * 1.0, "  weibull ", u^(1/k) * 1.0));
  my(k = 10000, u = lambertroot(k));
  if (abs(exp(u) - 1 - k*u) > 10^-75 * k * u, error("not a root at k = ", k));
  print("k = ", k, ": chapman-richards ", u * 1.0, "  weibull ", u^(1/k) * 1.0);
}
quit
