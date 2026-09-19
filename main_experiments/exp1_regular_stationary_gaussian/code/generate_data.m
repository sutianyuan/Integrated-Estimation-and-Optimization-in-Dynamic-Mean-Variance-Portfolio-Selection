function [R_A, R_B] = generate_data(T, n, mu_true, Sigma_true, N_A, N_B, seedA, seedB)
% GENERATE_DATA 按照ETO工作流文档生成Universe A/B收益率样本
% 
% 输入参数:
% T - 投资期限
% n - 资产数量  
% mu_true - 真实均值矩阵 (T×n)
% Sigma_true - 真实协方差张量 (T×n×n)
% N_A - Universe A样本量
% N_B - Universe B样本量
% seedA - Universe A随机种子
% seedB - Universe B随机种子
%
% 输出参数:
% R_A - Universe A收益率样本 (n×N_A×T)
% R_B - Universe B收益率样本 (n×N_B×T)

% 为A/B设置独立的随机数流（避免期数间样本重复）
streamA = RandStream('mt19937ar', 'Seed', seedA);   % 设置Universe A的随机种子（每期固定）
streamB = RandStream('mt19937ar', 'Seed', seedB);   % 设置Universe B的随机种子（全程使用seedB，确保与A完全独立）

% 初始化输出数组
R_A = zeros(n, N_A, T);
R_B = zeros(n, N_B, T);

% 对每期t生成收益率样本
for t = 1:T
    % 提取当期参数
    mu_t = mu_true(t, :);  % 1×n
    Sigma_t = squeeze(Sigma_true(t, :, :));  % n×n
    
    % 验证协方差矩阵正定性（Cholesky分解）
    try
        L = chol(Sigma_t, 'lower');
        if any(diag(L) <= 0)
            error('协方差矩阵在t=%d期不正定', t);
        end
    catch
        error('协方差矩阵在t=%d期无法进行Cholesky分解', t);
    end
    
    % 从N(mu_t, Sigma_t)生成样本
    
   
    % 生成Universe A样本（向量化，提升效率）
    z_A = randn(streamA, n, N_A);  % n×N_A 标准正态矩阵
    R_A(:, :, t) = mu_t' + L * z_A;  % n×N_A
    
    % 生成Universe B样本（向量化）
    z_B = randn(streamB, n, N_B);  % n×N_B 标准正态矩阵
    R_B(:, :, t) = mu_t' + L * z_B;  % n×N_B
    
end

% 重置随机种子状态
rng('default');

end