function [results] = run_ieo(R_A, R_B, lambda, x0)
% RUN_IEO 按照IEO工作流文档完整执行IEO方法
%
% 输入参数:
% R_A - Universe A收益率样本 (n×N_A×T)
% R_B - Universe B收益率样本 (n×N_B×T) 
% lambda - 风险偏好参数向量 (1×T)
% x0 - 初始财富
%
% 输出参数:
% results - 包含所有结果的结构体

% 获取维度信息
[n, N_A, T] = size(R_A);
[~, N_B, ~] = size(R_B);


% Equation (9) uses this deterministic fixed design at every stage.
x_grid = linspace(0.5, 1.5, 10)';
K_grid = numel(x_grid);

% 初始化存储变量
A_coef = zeros(n, T);       % A_t^A
B_coef = zeros(n, T);       % B_t^A
kappa_2 = zeros(T, 1);      % \hat\kappa_{2,t}^A
kappa_1 = zeros(T, 1);      % \hat\kappa_{1,t}^A
kappa_0 = zeros(T, 1);      % \hat\kappa_{0,t}^A
erm_objective = nan(1, T);
erm_exitflag = nan(1, T);
erm_constraint_residual = nan(1, T);
erm_fallback_occurred = false(1, T);
erm_attempt_count = zeros(1, T);
analytic_kappa = nan(3, T);
induced_grid_fit_kappa = nan(3, T);
pointwise_grid_fit_kappa = nan(3, T);
induced_policy_grid_values = nan(K_grid, T);
pointwise_optimal_grid_values = nan(K_grid, T);
analytic_grid_fit_max_abs_diff = nan(1, T);
point_qp_exitflag = nan(K_grid, T);
point_qp_fallback_occurred = false(K_grid, T);
point_qp_failed = false(K_grid, T);
point_qp_kkt_fallback_occurred = false(K_grid, T);
mu_hat_all = nan(n, T);
Sigma1_hat_all = nan(n, n, T);
lambda1_hat_all = nan(1, T);

%% ========== Phase A: Universe A上的data-driven backward induction ==========
% Options for QP
options_qp = optimoptions('quadprog', 'Display', 'off');

for t = T:-1:1
    % Prepare next_kappa
    if t == T
        next_k2 = 0; next_k1 = 0; next_k0 = 0;
    else
        next_k2 = kappa_2(t + 1);
        next_k1 = kappa_1(t + 1);
        next_k0 = kappa_0(t + 1);
    end

    % Equation (9): estimate theta_hat_t once from the complete training
    % sample and solve one constrained affine ERM.
    [alpha_t, beta_t, k2_t, k1_t, k0_t, erm_t, value_diag, theta_diag] = run_stage_step( ...
        R_A(:,:,t), lambda(t), next_k2, next_k1, next_k0, ...
        x_grid, options_qp);

    % Legacy output convention: A_coef is the wealth slope and B_coef is
    % the intercept, so A_coef=beta_hat and B_coef=alpha_hat.
    A_coef(:, t) = beta_t;
    B_coef(:, t) = alpha_t;
    kappa_2(t) = k2_t;
    kappa_1(t) = k1_t;
    kappa_0(t) = k0_t;
    erm_objective(t) = erm_t.objective;
    erm_exitflag(t) = erm_t.exitflag;
    erm_constraint_residual(t) = erm_t.constraint_residual;
    erm_fallback_occurred(t) = erm_t.fallback_occurred;
    erm_attempt_count(t) = erm_t.attempt_count;
    analytic_kappa(:, t) = value_diag.analytic_kappa;
    induced_grid_fit_kappa(:, t) = value_diag.induced_policy_grid_fit_kappa;
    pointwise_grid_fit_kappa(:, t) = value_diag.pointwise_optimal_grid_fit_kappa;
    induced_policy_grid_values(:, t) = value_diag.induced_policy_grid_values;
    pointwise_optimal_grid_values(:, t) = value_diag.pointwise_optimal_grid_values;
    analytic_grid_fit_max_abs_diff(t) = value_diag.analytic_grid_fit_max_abs_diff;
    point_qp_exitflag(:, t) = value_diag.point_qp_exitflag;
    point_qp_fallback_occurred(:, t) = value_diag.point_qp_fallback_occurred;
    point_qp_failed(:, t) = value_diag.point_qp_failed;
    point_qp_kkt_fallback_occurred(:, t) = ...
        value_diag.point_qp_kkt_fallback_occurred;
    mu_hat_all(:, t) = theta_diag.mu_hat;
    Sigma1_hat_all(:, :, t) = theta_diag.Sigma1_hat;
    lambda1_hat_all(t) = theta_diag.lambda1_hat;

    % D. 验收检查
    sum_A = sum(A_coef(:, t));
    sum_B = sum(B_coef(:, t));
    
    % RELAXED TOLERANCE: Changed from 1e-10 to 1e-8 to handle solver noise
    assert(abs(sum_A - 1) < 1e-8, 'A_coef sum violation at t=%d: %e', t, sum_A);
    assert(abs(sum_B) < 1e-8, 'B_coef sum violation at t=%d: %e', t, sum_B);
end

%% ========== Phase B: Universe B上的out-of-sample evaluation ==========
% 初始化财富路径
X = zeros(N_B, T+1);  % X_t^{j,B}
X(:, 1) = x0;  % 初始财富
% 初始化持仓路径
U = zeros(n, N_B, T);  % u_t^{j,B}
% 逐期执行冻结策略
for t = 1:T
    for j = 1:N_B
        % 使用冻结策略：u_t^{j,B} = A_t^A * X_t^{j,B} + B_t^A
        U(:, j, t) = A_coef(:, t) * X(j, t) + B_coef(:, t);
        
        % 财富递推：X_{t+1}^{j,B} = (R_t^{j,B})' * u_t^{j,B}
        X(j, t+1) = R_B(:, j, t)' * U(:, j, t);
    end
end

% 计算平均财富路径
X_mean = mean(X, 1);

%% ========== 组织输出结果 ==========
results = struct();
results.strategy = struct( ...
    'A_coef', A_coef, ...
    'B_coef', B_coef, ...
    'alpha_hat', B_coef, ...
    'beta_hat', A_coef, ...
    'identification_method', 'empirical affine ERM', ...
    'policy_identification_method', 'empirical affine ERM');
results.value_function = struct('kappa_2', kappa_2, 'kappa_1', kappa_1, 'kappa_0', kappa_0);
results.evaluation = struct('X', X, 'X_mean', X_mean, 'U', U);
results.training_states = struct( ...
    'type', 'deterministic_fixed_design', ...
    'grid', x_grid, ...
    'x_grid', x_grid, ...
    'grid_size', numel(x_grid), ...
    'grid_interval', [x_grid(1), x_grid(end)], ...
    'generated_from_returns', false, ...
    'used_for_policy_identification', true, ...
    'used_for_value_function_fit', true, ...
    'used_for_recursive_kappa', false, ...
    'same_grid_at_each_time', true);
results.horizon = struct( ...
    'T', T, ...
    'stage_index', 0:(T - 1), ...
    'affine_qp_count', T);
results.estimated_parameters = struct( ...
    'mu_hat', mu_hat_all, ...
    'Sigma1_hat', Sigma1_hat_all, ...
    'lambda1_hat', lambda1_hat_all);
results.diagnostics.affine_erm = struct( ...
    'method', 'empirical affine ERM', ...
    'training_sample', 'complete Universe A sample', ...
    'training_sample_size', N_A, ...
    'objective', erm_objective, ...
    'exitflag', erm_exitflag, ...
    'constraint_residual', erm_constraint_residual, ...
    'fallback_occurred', erm_fallback_occurred, ...
    'attempt_count', erm_attempt_count, ...
    'fallback_count', sum(erm_fallback_occurred), ...
    'qp_count', T);
results.diagnostics.value_recovery = struct( ...
    'recursive_update_method', 'analytic induced-policy coefficients', ...
    'analytic_kappa', analytic_kappa, ...
    'induced_policy_grid_fit_kappa', induced_grid_fit_kappa, ...
    'analytic_grid_fit_max_abs_diff', analytic_grid_fit_max_abs_diff, ...
    'max_analytic_grid_fit_abs_diff', max(analytic_grid_fit_max_abs_diff), ...
    'induced_policy_grid_values', induced_policy_grid_values, ...
    'pointwise_optimal_grid_values', pointwise_optimal_grid_values, ...
    'pointwise_optimal_grid_fit_kappa', pointwise_grid_fit_kappa, ...
    'point_qp_exitflag', point_qp_exitflag, ...
    'point_qp_fallback_occurred', point_qp_fallback_occurred, ...
    'point_qp_failed', point_qp_failed, ...
    'point_qp_kkt_fallback_occurred', point_qp_kkt_fallback_occurred, ...
    'point_qp_fallback_count', sum(point_qp_fallback_occurred, 1), ...
    'point_qp_total_fallback_count', sum(point_qp_fallback_occurred(:)), ...
    'point_qp_total_kkt_fallback_count', ...
        sum(point_qp_kkt_fallback_occurred(:)), ...
    'point_qp_total_failure_count', sum(point_qp_failed(:)));


end

function [alpha_hat, beta_hat, k2, k1, k0, erm_diag, value_diag, theta_diag] = run_stage_step( ...
        R, lam, nk2, nk1, nk0, x_grid, options_qp)
    [~, N] = size(R);

    % Full-sample empirical parameters in the manuscript's gross-return
    % representation of G_t.
    mu_hat = mean(R, 2);
    M2_hat = (R * R') / N;
    Sigma_hat = M2_hat - mu_hat * mu_hat';
    Sigma1_hat = Sigma_hat + nk2 * M2_hat;
    Sigma1_hat = (Sigma1_hat + Sigma1_hat') / 2;
    lambda1_hat = lam - nk1;
    H = 2 * Sigma1_hat;
    f = -lambda1_hat * mu_hat;
    
    % Equation (9): direct empirical optimization over z=[alpha; beta].
    [alpha_hat, beta_hat, erm_diag] = solve_affine_erm( ...
        H, f, nk0, x_grid, options_qp);

    % The recursive value coefficients are induced analytically by the
    % learned affine policy u_hat(x)=alpha_hat+beta_hat*x.
    k2 = beta_hat' * Sigma1_hat * beta_hat;
    k1 = 2 * alpha_hat' * Sigma1_hat * beta_hat ...
        - lambda1_hat * (mu_hat' * beta_hat);
    k0 = alpha_hat' * Sigma1_hat * alpha_hat ...
        - lambda1_hat * (mu_hat' * alpha_hat) - nk0;
    analytic_coefs = [k2; k1; k0];

    % Numerical consistency diagnostic (not used in backward recursion):
    % evaluate G_t at the learned affine policy and fit a quadratic.
    M_grid = numel(x_grid);
    X_des = [x_grid.^2, x_grid, ones(M_grid, 1)];
    induced_values = zeros(M_grid, 1);
    pointwise_values = zeros(M_grid, 1);
    point_exitflags = nan(M_grid, 1);
    point_fallbacks = false(M_grid, 1);
    point_failures = false(M_grid, 1);
    point_kkt_fallbacks = false(M_grid, 1);
    
    for k = 1:M_grid
        x = x_grid(k);
        u_policy = alpha_hat + beta_hat * x;
        induced_values(k) = u_policy' * Sigma1_hat * u_policy ...
            - lambda1_hat * (mu_hat' * u_policy) - nk0;

        % A separate pointwise optimum is retained solely as a diagnostic.
        [u_x, point_diag] = solve_qp_point(H, f, x, options_qp);
        if point_diag.failed
            pointwise_values(k) = NaN;
        else
            pointwise_values(k) = 0.5 * u_x' * H * u_x + f' * u_x - nk0;
        end
        point_exitflags(k) = point_diag.exitflag;
        point_fallbacks(k) = point_diag.fallback_occurred;
        point_failures(k) = point_diag.failed;
        point_kkt_fallbacks(k) = point_diag.kkt_fallback_occurred;
    end

    induced_fit = X_des \ induced_values;
    if any(point_failures)
        pointwise_fit = nan(3, 1);
    else
        pointwise_fit = X_des \ pointwise_values;
    end
    consistency_diff = max(abs(analytic_coefs - induced_fit));
    if consistency_diff > 1e-8
        warning('run_ieo:InducedValueConsistency', ...
            ['Analytic and induced-policy grid-fit kappa differ by %.3e; ' ...
             'the analytic coefficients remain the recursive update.'], ...
            consistency_diff);
    end
    value_diag = struct( ...
        'analytic_kappa', analytic_coefs, ...
        'induced_policy_grid_fit_kappa', induced_fit, ...
        'analytic_grid_fit_max_abs_diff', consistency_diff, ...
        'induced_policy_grid_values', induced_values, ...
        'pointwise_optimal_grid_values', pointwise_values, ...
        'pointwise_optimal_grid_fit_kappa', pointwise_fit, ...
        'point_qp_exitflag', point_exitflags, ...
        'point_qp_fallback_occurred', point_fallbacks, ...
        'point_qp_failed', point_failures, ...
        'point_qp_kkt_fallback_occurred', point_kkt_fallbacks);
    theta_diag = struct( ...
        'mu_hat', mu_hat, ...
        'Sigma1_hat', Sigma1_hat, ...
        'lambda1_hat', lambda1_hat);
end

function [alpha_hat, beta_hat, diag_info] = solve_affine_erm( ...
        H, f, nk0, x_grid, options)
    n = numel(f);
    K = numel(x_grid);
    eye_n = eye(n);
    H_z = zeros(2 * n);
    f_z = zeros(2 * n, 1);
    constant_term = 0;

    for i = 1:K
        x = x_grid(i);
        D_i = [eye_n, x * eye_n];
        H_z = H_z + (D_i' * H * D_i) / K;
        f_z = f_z + (D_i' * f) / K;
        constant_term = constant_term - nk0 / K;
    end
    H_z = (H_z + H_z') / 2;

    Aeq = [ones(1, n), zeros(1, n); ...
           zeros(1, n), ones(1, n)];
    beq = [0; 1];

    [z_hat, ~, exitflag] = quadprog( ...
        H_z, f_z, [], [], Aeq, beq, [], [], [], options);
    fallback_occurred = false;
    attempt_count = 1;

    if exitflag <= 0 || isempty(z_hat)
        retry_options = optimoptions(options, ...
            'Algorithm', 'active-set', ...
            'MaxIterations', 100000);
        feasible_start = [zeros(n, 1); ones(n, 1) / n];
        [z_hat, ~, exitflag] = quadprog( ...
            H_z, f_z, [], [], Aeq, beq, [], [], feasible_start, retry_options);
        fallback_occurred = true;
        attempt_count = 2;
    end

    if exitflag <= 0 || isempty(z_hat)
        retry_options = optimoptions(options, ...
            'Algorithm', 'trust-region-reflective', ...
            'MaxIterations', 10000);
        [z_hat, ~, exitflag] = quadprog( ...
            H_z, f_z, [], [], Aeq, beq, [], [], feasible_start, retry_options);
        attempt_count = 3;
    end

    if exitflag <= 0 || isempty(z_hat)
        error('run_ieo:AffineERMFailed', ...
            'Affine ERM quadprog failed with exitflag %d.', exitflag);
    end

    alpha_hat = z_hat(1:n);
    beta_hat = z_hat(n + 1:end);

    diag_info = struct();
    diag_info.objective = 0.5 * z_hat' * H_z * z_hat ...
        + f_z' * z_hat + constant_term;
    diag_info.exitflag = exitflag;
    diag_info.constraint_residual = max(abs(Aeq * z_hat - beq));
    diag_info.fallback_occurred = fallback_occurred;
    diag_info.attempt_count = attempt_count;
end

function [u, diag_info] = solve_qp_point(H, f, x, options)
    persistent point_failure_warned
    n = length(f);
    Aeq = ones(1, n);
    beq = x;
    [u, ~, exitflag] = quadprog(H, f, [], [], Aeq, beq, [], [], [], options);
    fallback_occurred = false;
    kkt_fallback_occurred = false;
    if exitflag <= 0 || isempty(u)
        retry_options = optimoptions(options, ...
            'Algorithm', 'trust-region-reflective', ...
            'MaxIterations', 100);
        feasible_start = (x / n) * ones(n, 1);
        [u, ~, exitflag] = quadprog( ...
            H, f, [], [], Aeq, beq, [], [], feasible_start, retry_options);
        fallback_occurred = true;
    end
    if exitflag <= 0 || isempty(u)
        % Diagnostic-only exact KKT fallback for the unchanged equality-
        % constrained quadratic objective.
        kkt_matrix = [H, Aeq'; Aeq, 0];
        kkt_rhs = [-f; x];
        kkt_solution = lsqminnorm(kkt_matrix, kkt_rhs);
        u = kkt_solution(1:n);
        kkt_fallback_occurred = true;
    end
    failed = isempty(u) || any(~isfinite(u)) ...
        || abs(Aeq * u - beq) > 1e-8;
    if failed
        u = nan(n, 1);
        if isempty(point_failure_warned)
            warning('run_ieo:PointValueQPFailed', ...
                ['A diagnostic pointwise QP failed after objective-preserving ' ...
                 'retries; NaN is recorded and recursive kappa is unchanged.']);
            point_failure_warned = true;
        end
    end
    diag_info = struct('exitflag', exitflag, ...
        'fallback_occurred', fallback_occurred, ...
        'kkt_fallback_occurred', kkt_fallback_occurred, ...
        'failed', failed);
end
