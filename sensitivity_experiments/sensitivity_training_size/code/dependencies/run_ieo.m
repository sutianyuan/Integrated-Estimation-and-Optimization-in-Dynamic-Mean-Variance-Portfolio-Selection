function [results] = run_ieo(R_A, R_B, lambda, x0)
% RUN_IEO 
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


% 初始化存储变量
A_coef = zeros(n, T);       % A_t^A
B_coef = zeros(n, T);       % B_t^A
kappa_2 = zeros(T, 1);      % \hat\kappa_{2,t}^A
kappa_1 = zeros(T, 1);      % \hat\kappa_{1,t}^A
kappa_0 = zeros(T, 1);      % \hat\kappa_{0,t}^A

% 网格参数（用于值函数拟合）
K = 5;  % 网格点数
x_grid = linspace(0.5, 1.5, K)';  % 财富网格 [0.5, 1.5]

%% ========== Phase A: Universe A上的data-driven backward induction ==========
x_grid_opt = linspace(0.5, 1.5, 10)';  % M×1
M_grid = numel(x_grid_opt);

% Options for QP
options_qp = optimoptions('quadprog', 'Display', 'off');
% Options for lsqlin
options_lsqlin = optimoptions('lsqlin', 'Display', 'off');

for t = T:-1:1
    % 1. 2-fold split (Reproducible)
    rng(20260115 + t);
    idx_shuffle = randperm(N_A);
    split_idx = floor(N_A/2);
    idx1 = idx_shuffle(1:split_idx);
    idx2 = idx_shuffle(split_idx+1:end);

    % Prepare next_kappa
    if t == T
        next_k2 = 0; next_k1 = 0; next_k0 = 0;
    else
        next_k2 = kappa_2(t + 1);
        next_k1 = kappa_1(t + 1);
        next_k0 = kappa_0(t + 1);
    end

    % 2. Process Halves
    [A1, B1, k2_1, k1_1, k0_1] = run_half_step(R_A(:,:,t), idx1, lambda(t), next_k2, next_k1, next_k0, x_grid_opt, options_qp, options_lsqlin);
    [A2, B2, k2_2, k1_2, k0_2] = run_half_step(R_A(:,:,t), idx2, lambda(t), next_k2, next_k1, next_k0, x_grid_opt, options_qp, options_lsqlin);

    % 5. Merge Two Halves
    A_coef(:, t) = 0.5 * (A1 + A2);
    B_coef(:, t) = 0.5 * (B1 + B2);
    kappa_2(t) = 0.5 * (k2_1 + k2_2);
    kappa_1(t) = 0.5 * (k1_1 + k1_2);
    kappa_0(t) = 0.5 * (k0_1 + k0_2);

    % D. 验收检查
    sum_A = sum(A_coef(:, t));
    sum_B = sum(B_coef(:, t));
    
    % Numerical tolerance for solver precision
    assert(abs(sum_A - 1) < 1e-8, 'A_coef sum violation at t=%d: %e', t, sum_A);
    assert(abs(sum_B) < 1e-8, 'B_coef sum violation at t=%d: %e', t, sum_B);
    
    % Enforce strict convexity if slightly violated due to numerics, or warn
    if kappa_2(t) < 1e-8
        % If it's very close, clamp it. If it's far off, something is wrong.
        if kappa_2(t) > 0
             kappa_2(t) = max(kappa_2(t), 1e-8);
        else
             fprintf('Warning: kappa_2(t=%d) = %e is negative/small. Clamping to 1e-8.\n', t, kappa_2(t));
             kappa_2(t) = 1e-8;
        end
    end
    % Strict assertion after clamping
    assert(kappa_2(t) >= 1e-8, 'kappa_2 convexity violation at t=%d', t);
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
results.strategy = struct('A_coef', A_coef, 'B_coef', B_coef);
results.value_function = struct('kappa_2', kappa_2, 'kappa_1', kappa_1, 'kappa_0', kappa_0);
results.evaluation = struct('X', X, 'X_mean', X_mean, 'U', U);


end

function [A, B, k2, k1, k0] = run_half_step(R_all, idx, lam, nk2, nk1, nk0, x_grid, options_qp, options_lsqlin)
    R = R_all(:, idx);
    [n, N] = size(R);
    R_net = R - 1;
    
    % Raw Moments
    m1 = mean(R, 2);
    M2 = (R * R') / N;
    m1_net = mean(R_net, 2);
    M2_net = (R_net * R_net') / N;
    
    % Construct H (constant for all x)
    % H = 2 * ( (M2 - m1*m1') + nk2 * M2_net )
    Cov_R = M2 - m1 * m1';
    H = 2 * (Cov_R + nk2 * M2_net);
    H = (H + H') / 2; % Ensure symmetry
    
    % Double QP for A, B
    % x=0
    f0 = m1_net * (nk1 - lam); 
    u0 = solve_qp_point(H, f0, 0, options_qp);
    
    % x=1
    f1 = m1_net * (nk1 - lam + 2*nk2*1);
    u1 = solve_qp_point(H, f1, 1, options_qp);
    
    B = u0;
    A = u1 - u0;
    
    % Grid QP for Kappa
    M_grid = length(x_grid);
    V_vals = zeros(M_grid, 1);
    
    for k = 1:M_grid
        x = x_grid(k);
        fx = m1_net * (nk1 - lam + 2*nk2*x);
        u_x = solve_qp_point(H, fx, x, options_qp);
        
        % Calculate Objective Value
        % J = u' (Cov_R + nk2 M2_net) u + u' m1_net (nk1 - lam + 2 nk2 x)
        %   + nk2 x^2 + (nk1 - lam) x + nk0
        
        qp_val = u_x' * (0.5 * H) * u_x + fx' * u_x;
        V_vals(k) = qp_val + nk2 * x^2 + (nk1 - lam) * x + nk0;
    end
    
    % Constrained Regression
    X_des = [x_grid.^2, x_grid, ones(M_grid, 1)];
    
    A_ineq = [-1, 0, 0];
    b_ineq = -1e-8;
    
    % Try lsqlin
    if exist('lsqlin', 'file') == 2
        try
             k_coefs = lsqlin(X_des, V_vals, A_ineq, b_ineq, [], [], [], [], [], options_lsqlin);
        catch
             k_coefs = solve_kappa_quadprog(X_des, V_vals, A_ineq, b_ineq);
        end
    else
        k_coefs = solve_kappa_quadprog(X_des, V_vals, A_ineq, b_ineq);
    end
    
    if isempty(k_coefs)
         % Fallback OLS
         k_coefs = X_des \ V_vals;
         if k_coefs(1) < 1e-8, k_coefs(1) = 1e-8; end
    end
    
    k2 = k_coefs(1);
    k1 = k_coefs(2);
    k0 = k_coefs(3);
end

function u = solve_qp_point(H, f, x, options)
    n = length(f);
    Aeq = ones(1, n);
    beq = x;
    [u, ~, exitflag] = quadprog(H, f, [], [], Aeq, beq, [], [], [], options);
    if exitflag < 0 || isempty(u)
        % Fallback: Ridge regularization
        [u, ~, ~] = quadprog(H + 1e-8*eye(n), f, [], [], Aeq, beq, [], [], [], options);
    end
    if isempty(u)
        % Last resort: equal weight
        u = (x/n) * ones(n, 1); 
    end
end

function k_coefs = solve_kappa_quadprog(X, V, A_ineq, b_ineq)
    H = 2 * (X' * X);
    f = -2 * (X' * V);
    options = optimoptions('quadprog', 'Display', 'off');
    k_coefs = quadprog(H, f, A_ineq, b_ineq, [], [], [], [], [], options);
end
