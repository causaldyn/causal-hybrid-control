% Independent cross-check of the space-time covariance's KRONECKER structure in Octave, built
% from the generative law rather than from chc.network_causal.panel_covariance. Result 59 claims
% (i) Sigma = sum_q P_q (x) T_q with P_{-q} = P_q', T_{-q} = T_q', (ii) the Van Loan-Pitsianis
% rearrangement has rank dmax+1 on a cycle (commuting shells) and 2*dmax on a path, (iii) rank 1
% at lag = 0, and (iv) the rank SATURATES once lag*q passes the panel's longest lag p-1.
1;

function S = shells_of(A, dmax)
  m = rows(A);
  reach = eye(m);
  seen = eye(m);
  S = {eye(m)};
  for d = 1:dmax
    reach = double((reach * (A + eye(m))) > 0);
    Sd = double(reach > 0 & seen == 0);
    seen = double((seen + Sd) > 0);
    S{end+1} = Sd;
  end
end

function Sig = panel_cov(S, g, ph, lag, p)
  m = rows(S{1});
  dmax = numel(g) - 1;
  [T, Ss] = meshgrid(0:p-1, 0:p-1);   % Ss - T is (t - s) with t the row
  gaps = Ss - T;
  Sig = zeros(m * p);
  for d = 0:dmax
    for e = 0:dmax
      block = ph .^ abs(gaps - lag * (d - e));
      Sig = Sig + g(d+1) * g(e+1) * kron(S{d+1} * S{e+1}, block);
    end
  end
end

function r = kron_rank(Sig, m, p)
  R = zeros(m * m, p * p);
  for i = 1:m
    for j = 1:m
      blk = Sig((i-1)*p + (1:p), (j-1)*p + (1:p));
      R((i-1)*m + j, :) = reshape(blk', 1, []);
    end
  end
  s = svd(R);
  r = sum(s > 1e-9 * s(1));
end

m = 8; ph = 0.6;
cyc = zeros(m); pth = zeros(m);
for i = 1:m
  cyc(i, mod(i, m) + 1) = 1; cyc(mod(i, m) + 1, i) = 1;
end
for i = 1:m-1
  pth(i, i+1) = 1; pth(i+1, i) = 1;
end

printf('dmax  p  cycle  path  (law: dmax+1 and 2*dmax)\n');
for dmax = 1:4
  g = 0.9 .^ (0:dmax);
  p = dmax + 2;                                   % long enough to resolve every shift
  rc = kron_rank(panel_cov(shells_of(cyc, dmax), g, ph, 1, p), m, p);
  rp = kron_rank(panel_cov(shells_of(pth, dmax), g, ph, 1, p), m, p);
  printf('%4d %2d %6d %5d   %s\n', dmax, p, rc, rp, ...
         merge(rc == dmax + 1 && rp == max(2 * dmax, 2), 'ok', 'MISMATCH'));
end

g = 0.9 .^ (0:4);
printf('separable at lag = 0: rank %d (must be 1)\n', ...
       kron_rank(panel_cov(shells_of(cyc, 4), g, ph, 0, 6), m, 6));
printf('saturated at p = 4 : cycle %d path %d (must fall below 5 and 8)\n', ...
       kron_rank(panel_cov(shells_of(cyc, 4), g, ph, 1, 4), m, 4), ...
       kron_rank(panel_cov(shells_of(pth, 4), g, ph, 1, 4), m, 4));
