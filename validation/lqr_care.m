% Independent cross-check of the continuous ARE (chc.lqr.continuous_lqr) in Octave.
% Solves it via the Hamiltonian eigen-decomposition (no control package needed) for the
% damped oscillator omega=1, zeta=0.1:  A=[0 1; -1 -0.2], B=[0;1], Q=I, R=1.
1;

A = [0 1; -1 -0.2];
B = [0; 1];
Q = eye(2);
R = 1;

Ham = [A, -B * (R \ B'); -Q, -A'];
[V, D] = eig(Ham);
lam = diag(D);
stable = real(lam) < 0;          % pick the stable invariant subspace
X = V(:, stable);
P = real(X(3:4, :) / X(1:2, :));
K = R \ (B' * P);

printf('P_octave =\n');
disp(P);
printf('K_octave = ');
disp(K);
printf('CARE residual norm = %.3e\n', norm(A' * P + P * A - P * B * (R \ (B' * P)) + Q));
