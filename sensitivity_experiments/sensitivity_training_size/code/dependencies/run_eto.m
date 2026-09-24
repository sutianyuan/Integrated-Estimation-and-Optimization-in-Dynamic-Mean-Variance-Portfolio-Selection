function [results] = run_eto(R_A, R_B, lambda, x0)
% RUN_ETO 
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


%% ========== 步骤1: 参数估计与冻结（对应文档2.2 Step2）==========


% 初始化参数矩阵
mu_hat = zeros(T, n);           % \hat\mu_t
Sigma_hat = zeros(n, n, T);      % \hat\Sigma_t  
ER_t = zeros(n, n, T);          % \mathbb{E}(R_tR_t^\top)

for t = 1:T
    % 公式4: 均值估计 \hat\mu_t := (1/N_A)*sum_j R_t^{j,A}
    mu_hat(t, :) = mean(R_A(:, :, t), 2)';
    
    % 公式5: 协方差估计 matlab内置函数
    Sigma_hat(:, :, t) = cov(R_A(:, :, t)');  % cov输入为N_A×n，输出n×n
    % 3. 数值修复：确保正定
    Sigma_hat(:, :, t) = Sigma_hat(:, :, t) + 1e-8 * eye(n);
    % 验证正定性
    try
        chol(Sigma_hat(:, :, t), 'lower');
    catch
        % 若仍失败，用特征值修正
        [V, D] = eig(Sigma_hat(:, :, t));
        D(D < 1e-10) = 1e-10;
        Sigma_hat(:, :, t) = V * D * V';
    end

    % 公式6: 二阶矩定义 \mathbb{E}(R_tR_t^\top) := \hat\Sigma_t + \hat\mu_t\hat\mu_t^\top
    ER_t(:, :, t) = Sigma_hat(:, :, t) + mu_hat(t, :)' * mu_hat(t, :);
    
    % 验证正定性
    try
        L = chol(Sigma_hat(:, :, t), 'lower');
        if any(diag(L) <= 0)
            error('估计的协方差矩阵在t=%d期不正定', t);
        end
    catch
        error('估计的协方差矩阵在t=%d期无法进行Cholesky分解', t);
    end
    
    fprintf('t=%d: 估计完成\n', t);
end


%% ========== 步骤2 & 3: 鲁棒递推与策略系数计算 ==========
% --- 初始化变量 ---
lambda_t1 = zeros(T, 1);    % \hat\lambda_t^{,1}
Sigma_t1 = zeros(n, n, T);  % \hat\Sigma_t^{,1}
a_t = zeros(T, 1);          % \hat a_t
b_t = zeros(T, 1);          % \hat b_t  
q_t = zeros(T, 1);          % \hat q_t
alpha_t = zeros(T, 1);      % \hat\alpha_t
A_coef = zeros(n, T);       % A_t
B_coef = zeros(n, T);       % B_t
ones_vec = ones(n, 1);

% 数值稳定参数
base_reg = 1e-8;      % 基础正则化参数 (防止完美的零特征值)
min_a_val = 1e-6;     % a_t 的最小安全阈值
max_reg_try = 10;     % 正则化最大尝试次数

% --- 倒序循环 (从 T 到 1) ---
% 注意：MATLAB索引 t=T 对应文档 t=T-1 (终端期)
for t = T:-1:1
    % 1. 构建当前的 Sigma_t1 矩阵
    if t == T
        % 终端期初始化 (公式7)
        lambda_t1(t) = lambda(t);
        Current_Sigma = Sigma_hat(:, :, t); 
    else
        % 向前递推 (公式8)
        % \hat\lambda_t^{,1} := \lambda_t + \hat\lambda_{t+1}^{,1} \hat q_{t+1}
        lambda_t1(t) = lambda(t) + lambda_t1(t+1) * q_t(t+1);
        
        % \hat\Sigma_t^{,1} := \hat\Sigma_t + (1/\hat a_{t+1}) \mathbb{E}(R_tR_t^\top)
        % 注意：这里依赖 t+1 期的 a_t，必须保证上一轮循环的 a_t 不为0
        Current_Sigma = Sigma_hat(:, :, t) + (1/a_t(t+1)) * ER_t(:, :, t);
    end
    
    % 2. 鲁棒求解线性方程组 (替代 inv)
    % 目标：求解 x_a = Sigma^{-1} * 1 和 x_b = Sigma^{-1} * mu
    % 使用动态正则化循环，确保矩阵正定且 a_t 不会过小
    current_reg = base_reg;
    success = false;
    
    for try_iter = 1:max_reg_try
        % 尝试添加正则化项
        Sigma_reg = Current_Sigma + current_reg * eye(n);
        try
            % 尝试 Cholesky 分解 (最快且能检测正定性)
            L = chol(Sigma_reg, 'lower'); 
            
            % 使用 Cholesky 求解线性方程 (比 inv 快且稳)
            % Sigma * x = y  =>  L * L' * x = y
            % x = L' \ (L \ y)
            x_a = L' \ (L \ ones_vec);
            x_b = L' \ (L \ mu_hat(t, :)');
            
            % 计算标量 a_t, b_t
            val_a = ones_vec' * x_a;
            val_b = ones_vec' * x_b;
            
            % 检查 a_t 是否满足数值安全阈值
            if val_a > min_a_val
                % 成功！保存结果并退出尝试循环
                Sigma_t1(:, :, t) = Sigma_reg; % 保存实际使用的矩阵(含正则化)
                a_t(t) = val_a;
                b_t(t) = val_b;
                success = true;
                break; 
            else
                % a_t 过小，视为不稳定，增加正则化
                current_reg = current_reg * 10;
            end
            
        catch
            % Cholesky 失败 (矩阵不正定)，增加正则化
            current_reg = current_reg * 10;
        end
    end
    
    if ~success
        error('在 t=%d 期无法通过正则化获得正定矩阵或安全的 a_t', t);
    end
    
    % 3. 计算 q_t 和 alpha_t (使用已稳定的 a_t, b_t)
    q_t(t) = b_t(t) / a_t(t);
    
    % 优化 alpha_t 计算：避免直接矩阵二次型相减带来的精度损失
    % 原式: mu' * Sigma^{-1} * mu - q * b
    % 代换: mu' * x_b - (b/a) * b
    % 通分: (a * (mu' * x_b) - b^2) / a
    term_mu_sigma_mu = mu_hat(t, :) * x_b;
    alpha_t(t) = term_mu_sigma_mu - q_t(t) * b_t(t);
    
    % 4. 直接计算策略系数 (公式9)
    % 利用当前已解出的 x_a, x_b，避免后续重复计算逆矩阵    
    % A_t = (\Sigma^{-1} 1) / a_t = x_a / a_t
    A_coef(:, t) = x_a / a_t(t);    
    % B_t = (lambda/2) * [x_b - q_t * x_a]
    B_coef(:, t) = (lambda_t1(t)/2) * (x_b - q_t(t) * x_a);

end

%=== 步骤4: Universe B上策略执行与财富递推 (向量化版) ===

% 初始化
X = zeros(N_B, T+1); 
X(:, 1) = x0;        
U = zeros(n, N_B, T);

% 向量化逐期递推
for t = 1:T
    % --- 公式10: 策略计算 
    % A_coef(:,t): n×1
    % X(:,t)': 1×N_B
    % 乘积为 n×N_B，直接与 B_coef (n×1) 相加 (MATLAB R2016b+ 支持自动广播)
    U(:, :, t) = A_coef(:, t) * X(:, t)' + B_coef(:, t); 
    
    % --- 公式11: 财富递推  ---
    % R_B 与 U 均为 n×N_B
    % .* 是逐元素相乘，sum(..., 1) 是按列求和，得到 1×N_B
    % 最后转置为 N_B×1 存入 X
    X(:, t+1) = sum(R_B(:, :, t) .* U(:, :, t), 1)';
    
end

% 计算平均财富
X_mean = mean(X, 1);

%% ========== 步骤5: Cost-to-Go函数计算（对应文档2.3）==========

% 计算尾项和 S_t = sum_{j=t}^{T-1} ((\hat\lambda_j^{,1})^2 / 4) * \hat\alpha_j
S_t = zeros(T, 1);
sum_tail = 0;
% 倒序递推：利用 S_t = term_t + S_{t+1} 的性质
for t = T:-1:1
    % 只有当 t <= T-1 时才累加项 (对应原代码 j=t:T-1)
    if t < T
        % 注意：MATLAB索引 t 对应文档时间 t
        % 累加当前项
        current_term = ((lambda_t1(t)^2) / 4) * alpha_t(t);
        sum_tail = sum_tail + current_term;
    end
    S_t(t) = sum_tail;
end

% 2. 计算每期 Cost-to-Go (向量化计算)
% V_t 初始化
V_t = zeros(T, N_B); 

for t = 1:T
    % 提取当前所有路径的财富 (N_B x 1)
    Xt_vec = X(:, t); 
    
    % 向量化计算 (结果为 N_B x 1)
    Vt_col = (1/a_t(t)) * Xt_vec.^2 - lambda_t1(t) * q_t(t) * Xt_vec - S_t(t);
    
    % 关键修正：必须转置 (.' 或 ') 才能赋值给行切片 V_t(t, :)
    V_t(t, :) = Vt_col'; 
end



%% ========== 步骤6: 结果验证（对应文档2.3/最终输出）==========
fprintf('\n=== 步骤6: 结果验证 ===\n');

% 验证自融资/预算约束: 1^\top u_t^{j,B} = X_t^{j,B}
constraint_violations = zeros(T, N_B);
for t = 1:T
    for j = 1:N_B
        % 公式13: 1^\top u_t^{j,B}
        lhs = ones_vec' * U(:, j, t);
        rhs = X(j, t);
        constraint_violations(t, j) = abs(lhs - rhs);
    end
end

% 计算最大偏差
max_violation = max(constraint_violations(:));
mean_violation = mean(constraint_violations(:));

fprintf('自融资约束验证:\n');
fprintf('  最大偏差: %.2e\n', max_violation);
fprintf('  平均偏差: %.2e\n', mean_violation);

if max_violation < 1e-10
    fprintf('  ✓ 自融资约束满足\n');
else
    fprintf('  ✗ 自融资约束不满足，可能存在数值误差\n');
end

%% ========== 组织输出结果 ==========
results = struct();
results.parameters = struct('mu_hat', mu_hat, 'Sigma_hat', Sigma_hat, 'ER_t', ER_t);
results.recursion = struct('lambda_t1', lambda_t1, 'Sigma_t1', Sigma_t1, 'a_t', a_t, ...
                          'b_t', b_t, 'q_t', q_t, 'alpha_t', alpha_t);
results.strategy = struct('A_coef', A_coef, 'B_coef', B_coef);
results.wealth = struct('X', X, 'X_mean', X_mean, 'U', U);
results.cost_to_go = struct('V_t', V_t, 'S_t', S_t);
results.validation = struct('constraint_violations', constraint_violations, ...
                           'max_violation', max_violation, 'mean_violation', mean_violation);

fprintf('\nETO方法执行完成!\n');
end