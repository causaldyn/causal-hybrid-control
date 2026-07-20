% Independent cross-check of the two identities behind chc.uncertainty.WassersteinPenalty:
%   (1) W1 = Kantorovich-Rubinstein dual: the transport LP equals sup over 1-Lipschitz f of
%       <f, mu - nu> (so the penalty's "radius * Lipschitz" is the W1-DRO worst case), and
%   (2) the W1-DRO worst case of a linear loss over a radius-ball = empirical + radius * Lipschitz.
% Both solved in Octave `glpk`, independent of the JAX implementation.
1;

% (1) W1 between two discrete distributions on a line: transport LP == 1-Lipschitz dual.
z  = [0 1 2 3]';
mu = [0.4 0.1 0.1 0.4]';
nu = [0.1 0.4 0.4 0.1]';
N  = numel(z);
Dc = abs(z - z');                         % ground cost |z_i - z_j|

c = reshape(Dc', [], 1);                  % primal: min <Dc, P>, row sums = mu, col sums = nu
A = zeros(2 * N, N * N);
for i = 1:N, A(i, (i - 1) * N + (1:N)) = 1; end
for j = 1:N, A(N + j, j:N:end) = 1; end
[~, w1_primal] = glpk(c, A, [mu; nu], zeros(N * N, 1), Inf(N * N, 1), ...
                      repmat('S', 2 * N, 1), repmat('C', N * N, 1), 1);

cc = -(mu - nu);                          % dual: max f'(mu - nu) s.t. f_i - f_j <= |z_i - z_j|
Alip = zeros(N * N, N); rhs = zeros(N * N, 1); k = 0;
for i = 1:N
  for j = 1:N
    k = k + 1; Alip(k, i) = Alip(k, i) + 1; Alip(k, j) = Alip(k, j) - 1; rhs(k) = Dc(i, j);
  end
end
lb = -Inf(N, 1); ub = Inf(N, 1); lb(1) = 0; ub(1) = 0;   % pin f_1 = 0 (dual is shift-invariant)
[~, neg_dual] = glpk(cc, Alip, rhs, lb, ub, repmat('U', N * N, 1), repmat('C', N, 1), 1);
w1_dual = -neg_dual;
printf('(1) W1 transport LP (primal)   = %.6f\n', w1_primal);
printf('    W1 1-Lipschitz dual        = %.6f\n', w1_dual);
printf('    Kantorovich-Rubinstein gap = %.3e\n', abs(w1_primal - w1_dual));

% (2) W1-DRO worst case of c(z) = w*z over { Q : W1(Q, delta_z0) <= radius }, solved on a grid.
w = 1.5; z0 = 0.7; radius = 0.3;
zg = linspace(z0 - 2, z0 + 2, 401)';
Aq = [abs(zg - z0)'; ones(1, numel(zg))]; % E|z - z0| <= radius ; sum q = 1
[~, neg_wc] = glpk(-(w * zg), Aq, [radius; 1], zeros(numel(zg), 1), Inf(numel(zg), 1), ...
                   ['U'; 'S'], repmat('C', numel(zg), 1), 1);
wc_grid = -neg_wc;
wc_form = w * z0 + radius * abs(w);       % mean + radius * Lipschitz, Lipschitz = |w|
printf('(2) DRO worst case (grid LP)   = %.6f\n', wc_grid);
printf('    mean + radius*Lipschitz    = %.6f  (Lipschitz |w| = %.3f)\n', wc_form, abs(w));
printf('    identity gap               = %.3e\n', abs(wc_grid - wc_form));
