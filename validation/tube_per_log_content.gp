\\ PARI/GP: the spread over logs of the share of the plant's paths a tube calibrated on average
\\ holds (chc.decision._per_log_spread), at 38 digits; validation/tube_per_log_content.mac derives
\\ it and checks it in double precision.
\\
\\ One state, noise sigma, the field off by kappa sigma W across logs; the radius
\\ z sqrt(1 + kappa^2) sigma. A log's share is C(W) = Phi(c - kappa W) - Phi(-c - kappa W),
\\ c = z sqrt(1 + kappa^2), and Var C = 4 (2 T(z, 1) - T(z, a) - T(z, 1/a)), a = sqrt(1 + 2 kappa^2),
\\ T Owen's; `share` is C, PARI's `content` being a built-in. Checked here: the moments by
\\ quadrature over W against Owen's form at kappa = 1/2, 1, 2, 0.9410 and 0.2249; the mean 0.95;
\\ and the spreads 0.0324434, 0.0821814, 0.1394346 at kappa 1/2, 1, 2.
\\
\\ The mean over W is a sum of intnum over unit pieces out to 40: intnum over one long interval
\\ mis-integrates a narrow Gaussian.
\\
\\ Run: timeout 300 gp -q -f validation/tube_per_log_content.gp </dev/null

default(realprecision, 38);
Phi(x) = erfc(-x / sqrt(2)) / 2;
\\ z: the normal's 0.975 quantile, by Newton on Phi
zq = solve(x = 1, 3, Phi(x) - 0.975);
owen(h, a) = intnum(x = 0, a, exp(-h^2 * (1 + x^2) / 2) / (1 + x^2)) / (2 * Pi);
share(w, k) = {my(c = zq * sqrt(1 + k^2)); Phi(c - k * w) - Phi(-c - k * w)};
moment(k, j) = sum(i = -40, 39, intnum(w = i, i + 1, share(w, k)^j * exp(-w^2 / 2) / sqrt(2 * Pi)));
spread(k) = {my(a = sqrt(1 + 2 * k^2)); sqrt(4 * (2 * owen(zq, 1) - owen(zq, a) - owen(zq, 1 / a)))};

check(label, x, tol) = if(abs(x) <= tol, print(label, ": ", x, " (PASS)"), error(label, ": ", x, " (FAIL)"));

check("8 T(z, 1) - (1 - 0.95^2)", 8 * owen(zq, 1) - (1 - 0.95^2), 1e-30);
{
  foreach([1/2, 1, 2, 0.9410345976678, 0.2249010646494], k,
    my(m1 = moment(k, 1), m2 = moment(k, 2), a = sqrt(1 + 2 * k^2));
    check(Str("kappa ", k, ": E C - 0.95"), m1 - 0.95, 1e-30);
    check(Str("kappa ", k, ": E C^2 - (1 - 4 T(z, s) - 4 T(z, 1/s))"),
          m2 - (1 - 4 * owen(zq, 1 / a) - 4 * owen(zq, a)), 1e-30);
    print("kappa ", k, ": per-log spread ", spread(k)));
}
check("spread at kappa 1/2 - 0.0324434", spread(1/2) - 0.0324434, 1e-7);
check("spread at kappa 1 - 0.0821814", spread(1) - 0.0821814, 1e-7);
check("spread at kappa 2 - 0.1394346", spread(2) - 0.1394346, 1e-7);
check("spread at kappa 0.9410346 - 0.077", spread(0.9410345976678) - 0.077, 1e-10);
check("spread at kappa 0.2249011 - 0.0078", spread(0.2249010646494) - 0.0078, 1e-10);
