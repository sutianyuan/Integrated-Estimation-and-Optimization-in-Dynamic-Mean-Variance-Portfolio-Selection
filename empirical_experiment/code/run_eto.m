function results = run_eto(R_A, R_B, lambda, x0)
%RUN_ETO Recursive plug-in ETO implementation used by the empirical release.
% The covariance convention and numerical ridge are documented in README.md.

[n, ~, T] = size(R_A);
[~, N_B, ~] = size(R_B);
ones_n = ones(n, 1);

mu_hat = zeros(T, n);
Sigma_hat = zeros(n, n, T);
ER_t = zeros(n, n, T);
for t = 1:T
    mu_hat(t, :) = mean(R_A(:, :, t), 2)';
    Sigma_hat(:, :, t) = cov(R_A(:, :, t)') + 1e-8 * eye(n);
    ER_t(:, :, t) = Sigma_hat(:, :, t) ...
        + mu_hat(t, :)' * mu_hat(t, :);
end

lambda_t1 = zeros(T, 1);
a_t = zeros(T, 1);
b_t = zeros(T, 1);
q_t = zeros(T, 1);
kappa_0 = zeros(T, 1);
A_coef = zeros(n, T);
B_coef = zeros(n, T);

base_reg = 1e-8;
min_a_val = 1e-6;
max_reg_try = 10;

for t = T:-1:1
    if t == T
        lambda_t1(t) = lambda(t);
        current_sigma = Sigma_hat(:, :, t);
        next_k0 = 0;
    else
        lambda_t1(t) = lambda(t) + lambda_t1(t + 1) * q_t(t + 1);
        current_sigma = Sigma_hat(:, :, t) ...
            + (1 / a_t(t + 1)) * ER_t(:, :, t);
        next_k0 = kappa_0(t + 1);
    end

    current_reg = base_reg;
    success = false;
    for attempt = 1:max_reg_try %#ok<NASGU>
        sigma_reg = current_sigma + current_reg * eye(n);
        try
            L = chol(sigma_reg, 'lower');
            inv_one = L' \ (L \ ones_n);
            inv_mu = L' \ (L \ mu_hat(t, :)');
            candidate_a = ones_n' * inv_one;
            if candidate_a > min_a_val
                a_t(t) = candidate_a;
                b_t(t) = ones_n' * inv_mu;
                success = true;
                break;
            end
        catch
        end
        current_reg = current_reg * 10;
    end
    if ~success
        error('run_eto:RegularizedSolveFailed', ...
            'ETO linear solve failed at paper stage %d.', t - 1);
    end

    q_t(t) = b_t(t) / a_t(t);
    alpha_t = mu_hat(t, :) * inv_mu - q_t(t) * b_t(t);
    A_coef(:, t) = inv_one / a_t(t);
    B_coef(:, t) = (lambda_t1(t) / 2) ...
        * (inv_mu - q_t(t) * inv_one);
    kappa_0(t) = next_k0 ...
        - ((lambda_t1(t)^2) / 4) * alpha_t;
end

X = zeros(N_B, T + 1);
X(:, 1) = x0;
U = zeros(n, N_B, T);
for t = 1:T
    U(:, :, t) = A_coef(:, t) * X(:, t)' + B_coef(:, t);
    X(:, t + 1) = sum(R_B(:, :, t) .* U(:, :, t), 1)';
end

results.recursion = struct('lambda_t1', lambda_t1, ...
    'a_t', a_t, 'b_t', b_t, 'q_t', q_t);
results.strategy = struct('A_coef', A_coef, 'B_coef', B_coef);
results.wealth = struct('X', X, 'X_mean', mean(X, 1), 'U', U);
results.cost_to_go = struct('S_t', -kappa_0);
end
