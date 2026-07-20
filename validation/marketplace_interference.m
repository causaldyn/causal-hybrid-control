% Independent cross-check of the two mechanisms behind chc.marketplace, in Octave:
%   (1) INTERFERENCE = driver conservation. The softmax congestion equilibrium conserves total
%       drivers (sum x = mass), so incentivising one zone *takes drivers from* the others
%       (negative cross-response) -- SUTVA is false; a per-zone additive model over-counts.
%   (2) CONFOUNDING is removed by a backdoor adjustment. When the logging policy chases demand,
%       regressing outcome on treatment alone is biased (omitted-variable bias); controlling for
%       demand recovers the structural coefficient.
1;

% ---- (1) equilibrium conservation + cannibalisation --------------------------------------------
n = 6; mass = 6.0; congestion = 2.0; beta = 2.5;
rand('seed', 1);
attract = 1.0 + rand(n, 1);

function x = equilibrium(attract, u, congestion, mass, beta)
  x = (mass / numel(attract)) * ones(numel(attract), 1);
  for it = 1:400
    value = beta * (attract + u - congestion * x / mass);
    sm = exp(value - max(value)); sm = sm / sum(sm);
    x = 0.5 * x + 0.5 * mass * sm;   % damped best-response (matches chc.games)
  end
end

u0 = zeros(n, 1);
x0 = equilibrium(attract, u0, congestion, mass, beta);
du = 0.5;                              % incentivise zone 1
u1 = u0; u1(1) = du;
x1 = equilibrium(attract, u1, congestion, mass, beta);
printf('(1) sum x (u=0)      = %.6f   (mass = %.1f)\n', sum(x0), mass);
printf('    sum x (u_1=%.1f)   = %.6f   (conserved)\n', du, sum(x1));
printf('    own response  dx_1 = %+.4f  (drivers pulled in)\n', x1(1) - x0(1));
printf('    others        dx_j = %+.4f  (cannibalised: SUTVA is false)\n', sum(x1(2:end) - x0(2:end)));

% ---- (2) confounding: omitted-variable bias vs backdoor adjustment -----------------------------
N = 8000; theta_d = 1.0; theta_c = 2.0; gamma = 1.5;
C = randn(N, 1);                       % demand (the confounder)
D = gamma * C + 0.5 * randn(N, 1);     % logging policy chases demand
y = theta_d * D + theta_c * C + 0.3 * randn(N, 1);
b_naive = [ones(N,1) D] \ y;           % omit C -> biased
b_adj   = [ones(N,1) D C] \ y;         % control C (backdoor) -> unbiased
printf('(2) true incentive effect      = %.3f\n', theta_d);
printf('    naive slope (omit demand)  = %.3f  (omitted-variable bias)\n', b_naive(2));
printf('    backdoor slope (adj demand)= %.3f  (de-confounded)\n', b_adj(2));
printf('    OVB predicted = theta_c*gamma*Var(C)/Var(D) = %.3f\n', ...
       theta_c * gamma * var(C) / var(D));
