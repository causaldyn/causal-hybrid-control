% Independent cross-check of chc.matching.sinkhorn against the exact Kantorovich
% transportation LP in Octave `glpk`. Entropic OT (Sinkhorn, eps -> 0) converges to this
% primal cost, and its dual potentials to these shadow prices -- Kantorovich-Rubinstein
% strong duality, gap = 0. Same 3x3 instance as tests/test_matching.py (LP optimum = 18).
1;

cost   = [1 2 3; 4 1 2; 3 2 1];
supply = [4; 5; 3];
demand = [6; 3; 3];
[m, n] = size(cost);

c = reshape(cost', [], 1);           % row-major flatten: var (i-1)*n + j
A = zeros(m + n, m * n);
for i = 1:m                          % supply rows: sum_j x_ij = a_i
  A(i, (i - 1) * n + (1:n)) = 1;
end
for j = 1:n                          % demand cols: sum_i x_ij = b_j
  A(m + j, j:n:end) = 1;
end
b = [supply; demand];

ctype = repmat('S', m + n, 1);       % 'S' = equality constraint
vtype = repmat('C', m * n, 1);       % continuous
lb = zeros(m * n, 1);
ub = Inf(m * n, 1);

[xopt, fmin, err, extra] = glpk(c, A, b, lb, ub, ctype, vtype, 1);
plan   = reshape(xopt, n, m)';       % undo the row-major flatten
lambda = extra.lambda;               % constraint duals (shadow prices)
u = lambda(1:m);                     % supply-side potentials  (f)
v = lambda(m + (1:n));               % demand-side potentials  (g) = surge prices
dual_obj = supply' * u + demand' * v;

printf('primal transport cost (glpk LP) = %.6f\n', fmin);
printf('dual  <a,u> + <b,v>             = %.6f\n', dual_obj);
printf('Kantorovich-Rubinstein gap      = %.3e\n', abs(fmin - dual_obj));
printf('surge prices v (demand duals)   = '); printf('%.3f ', v); printf('\n');
printf('optimal plan =\n'); disp(plan);
printf('marginals: |rows - supply| = %.1e, |cols - demand| = %.1e\n', ...
       norm(sum(plan, 2) - supply), norm(sum(plan, 1)' - demand));
% chc.matching.sinkhorn(eps=0.01) returns transport_cost ~ 18.0 (tests assert approx),
% converging to this LP primal; its dual potentials g converge to the surge prices v.
