function results = run_ieo(R_A, R_B, lambda, x0)
%RUN_IEO Full-sample anchor-state IEO implementation.
% At each stage, the affine policy is identified from the deterministic
% wealth anchors x=0 and x=1 using interior-point-convex quadprog solves.

[n, N_A, T] = size(R_A);
[~, N_B, ~] = size(R_B);
options = optimoptions('quadprog', 'Algorithm', ...
    'interior-point-convex', 'Display', 'off');

A_coef = zeros(n, T);
B_coef = zeros(n, T);
kappa_2 = zeros(T, 1);
kappa_1 = zeros(T, 1);
kappa_0 = zeros(T, 1);

for t = T:-1:1
    if t == T
        next_k2 = 0;
        next_k1 = 0;
        next_k0 = 0;
    else
        next_k2 = kappa_2(t + 1);
        next_k1 = kappa_1(t + 1);
        next_k0 = kappa_0(t + 1);
    end

    R_t = R_A(:, :, t);
    mu_hat = mean(R_t, 2);
    M2_hat = (R_t * R_t') / N_A;
    Sigma_hat = M2_hat - mu_hat * mu_hat';
    Sigma1_hat = Sigma_hat + next_k2 * M2_hat;
    Sigma1_hat = (Sigma1_hat + Sigma1_hat') / 2;
    lambda1_hat = lambda(t) - next_k1;

    H = 2 * Sigma1_hat;
    f = -lambda1_hat * mu_hat;
    [u0, ~, exit0] = quadprog(H, f, [], [], ones(1, n), 0, ...
        [], [], [], options);
    [u1, ~, exit1] = quadprog(H, f, [], [], ones(1, n), 1, ...
        [], [], [], options);
    if exit0 <= 0 || exit1 <= 0 || isempty(u0) || isempty(u1)
        error('run_ieo:AnchorSolveFailed', ...
            'Anchor-state quadprog failed at paper stage %d.', t - 1);
    end

    alpha_hat = u0;
    beta_hat = u1 - u0;
    B_coef(:, t) = alpha_hat;
    A_coef(:, t) = beta_hat;

    kappa_2(t) = beta_hat' * Sigma1_hat * beta_hat;
    kappa_1(t) = 2 * alpha_hat' * Sigma1_hat * beta_hat ...
        - lambda1_hat * (mu_hat' * beta_hat);
    kappa_0(t) = alpha_hat' * Sigma1_hat * alpha_hat ...
        - lambda1_hat * (mu_hat' * alpha_hat) + next_k0;

    assert(abs(sum(A_coef(:, t)) - 1) < 1e-8);
    assert(abs(sum(B_coef(:, t))) < 1e-8);
end

X = zeros(N_B, T + 1);
X(:, 1) = x0;
U = zeros(n, N_B, T);
for t = 1:T
    U(:, :, t) = A_coef(:, t) * X(:, t)' + B_coef(:, t);
    X(:, t + 1) = sum(R_B(:, :, t) .* U(:, :, t), 1)';
end

results.strategy = struct('A_coef', A_coef, 'B_coef', B_coef);
results.value_function = struct('kappa_2', kappa_2, ...
    'kappa_1', kappa_1, 'kappa_0', kappa_0);
results.evaluation = struct('X', X, 'X_mean', mean(X, 1), 'U', U);
end
